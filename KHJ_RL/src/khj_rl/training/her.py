"""Hindsight Experience Replay buffer for the cube-lift env.

Stores full episodes (not flat transitions) so relabel can re-evaluate the
five-condition success criterion with the original time semantics — most
importantly ``stable_placement``'s 20-step rolling window. Single-step
placement approximation was rejected after Codex review (2026-05-19, REVISE
1) for risking false-success Q targets.

Five-condition design (cfg.env.success):
  #1 lift_history   — goal-independent (cube_z ≥ lift_z_m for lift_hold_steps; latching)
  #2 stable_placement — **goal-dependent** (cube_xy in goal radius for place_hold_steps)
  #3 release_retreat  — goal-independent (gripper open ≥ thr AND |ee-cube| ≥ retreat_dist)
  #4 velocity_stability — goal-independent (|v|<v_max AND |w|<w_max)
  #5 visual_agreement   — goal-independent (top AND wrist proxies)

So HER relabel only re-evaluates condition #2; the other four per-step bools
are recomputed once at push time from the stored GT fields (so the buffer's
"future" goal sampler can produce a complete reward array for any choice).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class HERConfig:
    enabled: bool = True
    strategy: str = "future"        # "future" | "final" | "episode"
    k_future: int = 4               # ratio: 4 HER relabels : 1 original
    # Capacity in episodes. Each episode ~160 step, so 6_000 ep ≈ 960k transitions.
    max_episodes: int = 6_000
    # Phase 1 obs layout: indices 14:17 are cube_xyz_base; 17:20 are
    # target_delta_base. Patch these on relabel.
    cube_xyz_obs_slice: tuple[int, int] = (14, 17)
    target_delta_obs_slice: tuple[int, int] = (17, 20)


@dataclass
class _EpisodeStore:
    obs: np.ndarray                      # (T+1, obs_dim) — includes terminal next_obs
    action: np.ndarray                   # (T, action_dim)
    cube_xyz: np.ndarray                 # (T+1, 3)
    cube_lin_vel: np.ndarray             # (T, 3)
    cube_ang_vel: np.ndarray             # (T, 3)
    ee_xyz: np.ndarray                   # (T, 3)
    gripper_opening: np.ndarray          # (T,)
    visibility_top: np.ndarray           # (T,) bool
    visibility_wrist: np.ndarray         # (T,) bool
    original_goal: np.ndarray            # (3,) — env cfg goal at episode start
    done_terminal: bool                  # True if env terminated (success at end)
    # Cached goal-independent per-step bools (computed once at push).
    lift_latch: np.ndarray = field(default=None)              # (T,) bool — latched
    release_retreat: np.ndarray = field(default=None)         # (T,) bool
    velocity_stability: np.ndarray = field(default=None)      # (T,) bool
    visual_agreement: np.ndarray = field(default=None)        # (T,) bool


def _rolling_all(arr: np.ndarray, k: int) -> np.ndarray:
    """Return ``out`` where ``out[i] = bool(all(arr[max(0,i-k+1):i+1]))``.

    Vectorized via cumulative sum on the boolean array.
    """
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    a = arr.astype(np.int32)
    cs = np.concatenate(([0], np.cumsum(a)))   # (T+1,)
    # Window sum at i = cs[i+1] - cs[max(0,i-k+1)]
    T = a.shape[0]
    out = np.zeros(T, dtype=bool)
    for i in range(T):
        lo = max(0, i - k + 1)
        win = i - lo + 1
        if win < k:
            out[i] = False
        else:
            out[i] = (cs[i + 1] - cs[lo]) == k
    return out


class HERBuffer:
    """Per-episode storage with on-the-fly hindsight goal relabeling.

    Push semantics:
      ``begin_episode()`` clears the staging slot. ``push_transition(...)``
      appends a flat transition + GT raw fields. ``end_episode(success_flag)``
      finalizes: caches the four goal-independent per-step bools and stores
      the episode. Capacity is enforced via FIFO eviction.

    Sample semantics:
      ``sample(batch_size, rng)`` returns (obs, action, reward, next_obs,
      done) tuples. For each row: pick a random (episode, t); with probability
      ``k_future/(k_future+1)`` choose a future timestep t' > t and set
      new_goal = cube_xyz[t']; otherwise keep the original goal. Recompute
      obs/next_obs's target_delta and the full 5-condition reward.
    """

    def __init__(
        self,
        cfg: HERConfig,
        env_cfg_goal_radius_xy_m: float,
        env_cfg_success,                  # SuccessCfg
        gripper_open_threshold: float,
    ) -> None:
        self._cfg = cfg
        self._goal_radius = float(env_cfg_goal_radius_xy_m)
        self._success_cfg = env_cfg_success
        self._gripper_open_threshold = float(gripper_open_threshold)
        self._episodes: list[_EpisodeStore] = []
        # Staging while episode is in flight.
        self._stage_obs: list[np.ndarray] = []        # length T+1 (with final next_obs)
        self._stage_action: list[np.ndarray] = []
        self._stage_cube_xyz: list[np.ndarray] = []
        self._stage_cube_lv: list[np.ndarray] = []
        self._stage_cube_av: list[np.ndarray] = []
        self._stage_ee_xyz: list[np.ndarray] = []
        self._stage_gripper: list[float] = []
        self._stage_vis_top: list[bool] = []
        self._stage_vis_wrist: list[bool] = []
        self._stage_goal: np.ndarray | None = None
        # Stats
        self._n_relabeled_success: int = 0
        self._n_relabeled_total: int = 0

    # ---- staging ----------------------------------------------------

    def begin_episode(self, first_obs: np.ndarray, first_raw: dict[str, Any]) -> None:
        self._stage_obs = [first_obs.astype(np.float32, copy=True)]
        self._stage_action = []
        self._stage_cube_xyz = [first_raw["cube_xyz_m"].astype(np.float32, copy=True)]
        self._stage_cube_lv = []
        self._stage_cube_av = []
        self._stage_ee_xyz = []
        self._stage_gripper = []
        self._stage_vis_top = []
        self._stage_vis_wrist = []
        self._stage_goal = first_raw["goal_xyz_m"].astype(np.float32, copy=True)

    def push_transition(
        self,
        action: np.ndarray,
        next_obs: np.ndarray,
        next_raw: dict[str, Any],
    ) -> None:
        self._stage_action.append(action.astype(np.float32, copy=True))
        self._stage_obs.append(next_obs.astype(np.float32, copy=True))
        self._stage_cube_xyz.append(next_raw["cube_xyz_m"].astype(np.float32, copy=True))
        self._stage_cube_lv.append(next_raw["cube_lin_vel_m_s"].astype(np.float32, copy=True))
        self._stage_cube_av.append(next_raw["cube_ang_vel_rad_s"].astype(np.float32, copy=True))
        self._stage_ee_xyz.append(next_raw["ee_xyz_m"].astype(np.float32, copy=True))
        self._stage_gripper.append(float(next_raw["gripper_opening"]))
        self._stage_vis_top.append(bool(next_raw["visibility_top"]))
        self._stage_vis_wrist.append(bool(next_raw["visibility_wrist"]))

    def end_episode(self, terminal: bool) -> None:
        if not self._stage_action:
            # Empty episode — nothing to store.
            return
        obs = np.stack(self._stage_obs, axis=0)
        action = np.stack(self._stage_action, axis=0)
        cube_xyz = np.stack(self._stage_cube_xyz, axis=0)
        cube_lv = np.stack(self._stage_cube_lv, axis=0)
        cube_av = np.stack(self._stage_cube_av, axis=0)
        ee_xyz = np.stack(self._stage_ee_xyz, axis=0)
        gripper = np.asarray(self._stage_gripper, dtype=np.float32)
        vis_top = np.asarray(self._stage_vis_top, dtype=bool)
        vis_wrist = np.asarray(self._stage_vis_wrist, dtype=bool)

        # Goal-independent per-step bools — computed once.
        # #1 lift_history: cube_z[t] ≥ lift_z_m for lift_hold_steps consecutive
        # frames AT OR BEFORE t, latching. Use next_state cube_xyz (indices 1..T).
        cube_z_next = cube_xyz[1:, 2]
        z_above = cube_z_next >= self._success_cfg.lift_z_m
        rolling = _rolling_all(z_above, self._success_cfg.lift_hold_steps)
        lift_latch = np.maximum.accumulate(rolling.astype(bool))

        # #3 release_retreat
        ee_cube_dist = np.linalg.norm(ee_xyz - cube_xyz[1:], axis=1)
        released = gripper >= self._gripper_open_threshold
        retreated = ee_cube_dist >= self._success_cfg.retreat_distance_m
        release_retreat = released & retreated

        # #4 velocity_stability
        v = np.linalg.norm(cube_lv, axis=1)
        w = np.linalg.norm(cube_av, axis=1)
        velocity_stability = (v <= self._success_cfg.cube_v_max) & (
            w <= self._success_cfg.cube_omega_max
        )

        # #5 visual_agreement
        if self._success_cfg.require_visual_agreement:
            visual_agreement = vis_top & vis_wrist
        else:
            visual_agreement = np.ones_like(vis_top, dtype=bool)

        ep = _EpisodeStore(
            obs=obs,
            action=action,
            cube_xyz=cube_xyz,
            cube_lin_vel=cube_lv,
            cube_ang_vel=cube_av,
            ee_xyz=ee_xyz,
            gripper_opening=gripper,
            visibility_top=vis_top,
            visibility_wrist=vis_wrist,
            original_goal=self._stage_goal.copy(),
            done_terminal=bool(terminal),
            lift_latch=lift_latch,
            release_retreat=release_retreat,
            velocity_stability=velocity_stability,
            visual_agreement=visual_agreement,
        )
        self._episodes.append(ep)
        if len(self._episodes) > self._cfg.max_episodes:
            # FIFO eviction.
            self._episodes.pop(0)
        # Reset staging.
        self._stage_action = []

    # ---- sample -----------------------------------------------------

    def __len__(self) -> int:
        # Number of (ep, t) pairs available.
        return sum(int(e.action.shape[0]) for e in self._episodes)

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    def relabeled_success_stats(self) -> tuple[int, int]:
        """Return (successes, total) since buffer was created."""
        return self._n_relabeled_success, self._n_relabeled_total

    def _placement_array(self, ep: _EpisodeStore, goal_xy: np.ndarray) -> np.ndarray:
        """stable_placement[t] under ``goal_xy`` — 20-step rolling AND."""
        cube_xy_next = ep.cube_xyz[1:, :2]
        in_radius = np.linalg.norm(cube_xy_next - goal_xy[None, :], axis=1) <= self._goal_radius
        return _rolling_all(in_radius.astype(bool), self._success_cfg.place_hold_steps)

    def _per_step_success(self, ep: _EpisodeStore, goal: np.ndarray) -> np.ndarray:
        place = self._placement_array(ep, goal[:2])
        return (
            ep.lift_latch
            & place
            & ep.release_retreat
            & ep.velocity_stability
            & ep.visual_agreement
        )

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        target_delta_slice: tuple[int, int] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self._episodes:
            raise RuntimeError("HERBuffer empty; cannot sample")
        if target_delta_slice is None:
            target_delta_slice = self._cfg.target_delta_obs_slice

        ep_count = len(self._episodes)
        # Sample episode indices weighted by length so longer eps aren't
        # under-sampled. Simpler: uniform over episodes, then random t within.
        ep_idx = rng.integers(0, ep_count, size=batch_size)
        # Per-row t (random index in episode).
        ep_lens = np.array([int(e.action.shape[0]) for e in self._episodes])
        t_max = ep_lens[ep_idx]
        t = (rng.random(batch_size) * t_max).astype(np.int64)

        # Decide HER vs original for each row.
        # P(her) = k_future / (k_future + 1); P(orig) = 1 / (k_future + 1)
        her_mask = rng.random(batch_size) < (
            self._cfg.k_future / (self._cfg.k_future + 1)
        )

        obs_dim = self._episodes[0].obs.shape[1]
        action_dim = self._episodes[0].action.shape[1]
        out_obs = np.zeros((batch_size, obs_dim), dtype=np.float32)
        out_next_obs = np.zeros((batch_size, obs_dim), dtype=np.float32)
        out_action = np.zeros((batch_size, action_dim), dtype=np.float32)
        out_reward = np.zeros(batch_size, dtype=np.float32)
        out_done = np.zeros(batch_size, dtype=np.float32)

        d_lo, d_hi = target_delta_slice
        c_lo, c_hi = self._cfg.cube_xyz_obs_slice

        for i in range(batch_size):
            ep = self._episodes[ep_idx[i]]
            ti = int(t[i])
            T = ep.action.shape[0]
            # Choose goal.
            if her_mask[i] and ti < T - 1:
                # future: random t' in (ti, T-1]
                tprime = int(rng.integers(ti + 1, T))
                new_goal = ep.cube_xyz[tprime + 1]
            else:
                new_goal = ep.original_goal
            # Per-step success under this goal (T,)
            succ = self._per_step_success(ep, new_goal)
            reward_t = float(succ[ti])
            # Terminal: env terminates on success. We mark done if the
            # relabeled reward says success at this step.
            done_t = bool(succ[ti])

            # Patch obs / next_obs target_delta.
            o = ep.obs[ti].copy()
            no = ep.obs[ti + 1].copy()
            o[d_lo:d_hi] = new_goal - ep.cube_xyz[ti]
            no[d_lo:d_hi] = new_goal - ep.cube_xyz[ti + 1]
            # cube_xyz_obs (indices 14:17) is the NOISY obs from the env at
            # rollout time — we keep that as stored (sim2real consistency).
            # Only target_delta changes with the new goal.

            out_obs[i] = o
            out_next_obs[i] = no
            out_action[i] = ep.action[ti]
            out_reward[i] = reward_t
            out_done[i] = float(done_t)
            self._n_relabeled_total += 1
            if reward_t > 0.0:
                self._n_relabeled_success += 1
        return out_obs, out_action, out_reward, out_next_obs, out_done

    # ---- normalizer support -----------------------------------------

    def sample_target_delta_for_normfit(
        self, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Return (n_samples, 3) of hypothetical relabeled target_delta values
        for normalizer fitting (Codex REVISE 2). Mixes future-relabel deltas
        from existing episodes; falls back to empty array if buffer empty.
        """
        if not self._episodes:
            return np.zeros((0, 3), dtype=np.float32)
        out = np.zeros((n_samples, 3), dtype=np.float32)
        ep_count = len(self._episodes)
        ep_lens = np.array([int(e.action.shape[0]) for e in self._episodes])
        for i in range(n_samples):
            ei = int(rng.integers(0, ep_count))
            ep = self._episodes[ei]
            T = int(ep.action.shape[0])
            if T < 2:
                continue
            t = int(rng.integers(0, T - 1))
            tp = int(rng.integers(t + 1, T))
            out[i] = ep.cube_xyz[tp + 1] - ep.cube_xyz[t]
        return out
