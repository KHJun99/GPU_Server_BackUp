"""NPZ demo loader for BC pretrain.

``scripts/collect_demos.py`` -> ``khj_rl.data.demos.DemoBuffer.write`` lays
out one episode per file under ``runs/demos/{stage}/{ep_id}.npz`` with::

  obs            : (T, obs_dim)    float32  post-NoiseModel observations
  actions        : (T, action_dim) float32  oracle's normalized action
                                             (Method B post-M2: 4-D
                                             [Δx, Δy, Δz, g]. Pre-M2 demos
                                             were 6-D joint-delta and are
                                             폐기됨 — see CLAUDE.md "Method B
                                             pivot". Loader infers action_dim
                                             from array shape so a clean
                                             collect after M3.3 calibration
                                             gives the new 4-D layout
                                             automatically.)
  rewards        : (T,)            float32
  terminated     : ()              bool     episode terminated via success
  success_flag   : ()              bool     episode-level success
  cfg_hash       : ()              str      hash of CubeLiftEnvCfg
  seed           : ()              int
  stage          : ()              str
  ep_id          : ()              int

CLAUDE.md mandates that only **successful** demos feed BC — failures are
exactly the trajectories we don't want the policy to imitate. The loader
filters by ``success_flag`` at load time; ``stage`` lets us mix in
later-stage harder demos as Phase 1 curriculum widens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DemoStats:
    n_episodes: int
    n_steps: int
    success_only: bool
    obs_dim: int
    action_dim: int


class DemoBuffer:
    """Concatenated (obs, action) tensors across episodes for supervised BC.

    For ACT (action chunking), per-episode spans are tracked so the sampler
    never crosses episode boundaries. Pass spans=[(start, end_exclusive), ...]
    when constructing manually; ``from_dir`` builds them automatically.
    """

    def __init__(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        spans: list[tuple[int, int]] | None = None,
    ) -> None:
        if obs.shape[0] != action.shape[0]:
            raise ValueError(
                f"obs/action length mismatch: {obs.shape[0]} vs {action.shape[0]}"
            )
        self._obs = obs.astype(np.float32, copy=False)
        self._action = action.astype(np.float32, copy=False)
        # Default to one big span when spans aren't supplied (legacy callers).
        # ACT codepaths require real per-episode spans to be safe.
        self._spans: list[tuple[int, int]] = (
            list(spans) if spans is not None else [(0, int(obs.shape[0]))]
        )

    # ---- construction ---------------------------------------------------

    @classmethod
    def from_dir(
        cls,
        demo_dir,
        success_only: bool = True,
    ) -> tuple["DemoBuffer", DemoStats]:
        # Accepts a single Path or a sequence of Paths so BC can merge
        # multiple collection batches (e.g. stage0 1k + stage0_extra 2k)
        # into one buffer without copying the NPZ files around.
        if isinstance(demo_dir, (list, tuple)):
            dirs = [Path(d) for d in demo_dir]
        else:
            dirs = [Path(demo_dir)]
        files: list[Path] = []
        for d in dirs:
            files.extend(sorted(d.glob("*.npz")))
        if not files:
            raise FileNotFoundError(f"no demo NPZ files under {dirs}")

        obs_chunks: list[np.ndarray] = []
        action_chunks: list[np.ndarray] = []
        spans: list[tuple[int, int]] = []
        kept = 0
        cursor = 0
        for f in files:
            with np.load(f, allow_pickle=False) as data:
                if success_only:
                    flag = bool(data["success_flag"])
                    if not flag:
                        continue
                ep_obs = data["obs"].astype(np.float32, copy=False)
                ep_act = data["actions"].astype(np.float32, copy=False)
                obs_chunks.append(ep_obs)
                action_chunks.append(ep_act)
                ep_len = int(ep_obs.shape[0])
                spans.append((cursor, cursor + ep_len))
                cursor += ep_len
                kept += 1
        if not obs_chunks:
            raise RuntimeError(
                f"no eligible demos in {dirs} (success_only={success_only})"
            )
        obs = np.concatenate(obs_chunks, axis=0)
        action = np.concatenate(action_chunks, axis=0)
        return cls(obs=obs, action=action, spans=spans), DemoStats(
            n_episodes=kept,
            n_steps=int(obs.shape[0]),
            success_only=success_only,
            obs_dim=int(obs.shape[1]),
            action_dim=int(action.shape[1]),
        )

    # ---- access ---------------------------------------------------------

    def __len__(self) -> int:
        return self._obs.shape[0]

    @property
    def obs(self) -> np.ndarray:
        return self._obs

    @property
    def action(self) -> np.ndarray:
        return self._action

    @property
    def spans(self) -> list[tuple[int, int]]:
        return list(self._spans)

    def sample_batch(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.integers(low=0, high=len(self), size=batch_size)
        return self._obs[idx], self._action[idx]

    def iter_minibatches(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ):
        """Yield (obs, action) minibatches over a single shuffled epoch."""
        idx = rng.permutation(len(self))
        for start in range(0, len(self), batch_size):
            sl = idx[start : start + batch_size]
            yield self._obs[sl], self._action[sl]

    # ---- chunked (ACT) access ------------------------------------------

    def valid_chunk_starts(self, k: int) -> np.ndarray:
        """Indices t such that obs[t] is valid and action[t:t+k] stays in the
        same episode. Codex review (B): precompute to kill off-by-one bugs.

        Condition per span (start, end_exclusive): t ∈ [start, end - k + 1).
        Episodes with T < k are skipped entirely — no zero/end padding (would
        teach the policy to do nothing past terminal frames).
        """
        if k <= 0:
            raise ValueError(f"chunk_size must be >= 1, got {k}")
        valid: list[np.ndarray] = []
        for start, end in self._spans:
            last_start = end - k + 1
            if last_start <= start:
                continue
            valid.append(np.arange(start, last_start, dtype=np.int64))
        if not valid:
            raise RuntimeError(
                f"no episode has length >= chunk_size={k}; max ep length="
                f"{max((e - s) for s, e in self._spans)}"
            )
        return np.concatenate(valid)

    def iter_chunked_minibatches(
        self,
        batch_size: int,
        k: int,
        rng: np.random.Generator,
    ):
        """Yield (obs[t], action[t:t+k]) minibatches for ACT BC.

        - obs out shape : (B, obs_dim)
        - action out shape: (B, k, action_dim)

        Sampling is over precomputed in-episode valid starts (see
        ``valid_chunk_starts``), shuffled once per epoch.
        """
        starts = self.valid_chunk_starts(k)
        order = rng.permutation(starts.shape[0])
        # Per-step offsets [0..k-1] broadcast against base indices to form a
        # (B, k) index matrix; gather actions in one fancy-index op so episode
        # contiguity is preserved by the precomputed starts.
        offsets = np.arange(k, dtype=np.int64)
        n = starts.shape[0]
        for batch_start in range(0, n, batch_size):
            sel = order[batch_start : batch_start + batch_size]
            base = starts[sel]                            # (B,)
            idx_mat = base[:, None] + offsets[None, :]    # (B, k)
            yield self._obs[base], self._action[idx_mat]


@dataclass
class ReplayStats:
    n_episodes: int
    n_transitions: int
    obs_dim: int
    action_dim: int


class DemoReplay:
    """Read NPZ demos as SAC replay tuples (obs, action, reward, next_obs, done).

    Reconstructs ``next_obs[t] = obs[t+1]`` per episode. The last step of each
    episode gets ``done=True`` and ``next_obs=obs[t]`` (terminal next-state is
    masked by ``(1-done)`` in the Bellman update so its value doesn't matter).
    Loads everything into RAM (~50 MB for 4k ep × 160 step × 31 float).

    Demos already filtered to ``success_only`` by default — matches BC pretrain
    so SAC sees the same trajectories the actor is anchored to.
    """

    def __init__(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
        stats: ReplayStats,
    ) -> None:
        self._obs = obs.astype(np.float32, copy=False)
        self._action = action.astype(np.float32, copy=False)
        self._reward = reward.astype(np.float32, copy=False)
        self._next_obs = next_obs.astype(np.float32, copy=False)
        self._done = done.astype(np.float32, copy=False)
        self._stats = stats

    @classmethod
    def from_dir(
        cls,
        demo_dir,
        success_only: bool = True,
    ) -> "DemoReplay":
        if isinstance(demo_dir, (list, tuple)):
            dirs = [Path(d) for d in demo_dir]
        else:
            dirs = [Path(demo_dir)]
        files: list[Path] = []
        for d in dirs:
            files.extend(sorted(d.glob("*.npz")))
        if not files:
            raise FileNotFoundError(f"no demo NPZ files under {dirs}")

        obs_chunks: list[np.ndarray] = []
        act_chunks: list[np.ndarray] = []
        rew_chunks: list[np.ndarray] = []
        next_chunks: list[np.ndarray] = []
        done_chunks: list[np.ndarray] = []
        kept = 0
        for f in files:
            with np.load(f, allow_pickle=False) as data:
                if success_only and not bool(data["success_flag"]):
                    continue
                ep_obs = data["obs"].astype(np.float32, copy=False)
                ep_act = data["actions"].astype(np.float32, copy=False)
                ep_rew = data["rewards"].astype(np.float32, copy=False)
                T = int(ep_obs.shape[0])
                if T < 2:
                    # Single-step "episode" can't form (s, s') — skip rather
                    # than synthesize fake transitions.
                    continue
                ep_next = np.empty_like(ep_obs)
                ep_next[:-1] = ep_obs[1:]
                ep_next[-1] = ep_obs[-1]  # terminal, masked by done
                ep_done = np.zeros(T, dtype=np.float32)
                ep_done[-1] = 1.0
                obs_chunks.append(ep_obs)
                act_chunks.append(ep_act)
                rew_chunks.append(ep_rew)
                next_chunks.append(ep_next)
                done_chunks.append(ep_done)
                kept += 1
        if not obs_chunks:
            raise RuntimeError(
                f"no eligible demos in {dirs} (success_only={success_only})"
            )
        obs = np.concatenate(obs_chunks, axis=0)
        action = np.concatenate(act_chunks, axis=0)
        reward = np.concatenate(rew_chunks, axis=0)
        next_obs = np.concatenate(next_chunks, axis=0)
        done = np.concatenate(done_chunks, axis=0)
        stats = ReplayStats(
            n_episodes=kept,
            n_transitions=int(obs.shape[0]),
            obs_dim=int(obs.shape[1]),
            action_dim=int(action.shape[1]),
        )
        return cls(obs, action, reward, next_obs, done, stats)

    def __len__(self) -> int:
        return int(self._obs.shape[0])

    @property
    def stats(self) -> ReplayStats:
        return self._stats

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = rng.integers(low=0, high=len(self), size=batch_size)
        return (
            self._obs[idx],
            self._action[idx],
            self._reward[idx],
            self._next_obs[idx],
            self._done[idx],
        )


class OnlineReplayBuffer:
    """Ring buffer for online SAC transitions.

    Pre-allocates float32 numpy arrays; ``push`` overwrites the oldest slot
    once ``size == capacity``. Pure numpy: SAC trainer batches sit on GPU but
    the buffer itself stays on host RAM (1M × (31+6+1+31+1) ≈ 280 MB).
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int) -> None:
        self._capacity = int(capacity)
        self._obs_dim = int(obs_dim)
        self._action_dim = int(action_dim)
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reward = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._done = np.zeros(capacity, dtype=np.float32)
        self._size = 0
        self._cursor = 0

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._cursor
        self._obs[i] = obs
        self._action[i] = action
        self._reward[i] = float(reward)
        self._next_obs[i] = next_obs
        self._done[i] = float(done)
        self._cursor = (i + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._size == 0:
            raise RuntimeError("OnlineReplayBuffer empty; cannot sample")
        idx = rng.integers(low=0, high=self._size, size=batch_size)
        return (
            self._obs[idx],
            self._action[idx],
            self._reward[idx],
            self._next_obs[idx],
            self._done[idx],
        )
