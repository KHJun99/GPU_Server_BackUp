"""Behavior-cloning pretrainer for Phase 1.

Reads successful oracle demos, fits an ``ObsNormalizer`` to their obs, and
trains the actor head of :class:`ActorCritic` to regress the oracle's
action via MSE. The critic is left at its random init — PPO does a 10-20
iter critic warm-up before unfreezing the actor (CLAUDE.md ## Phase 1
결정 사항 7 step 3).

Output checkpoint at ``runs/{run_name}/bc.pt`` contains:
  - ``actor_critic``: full model state_dict (actor + critic + logstd)
  - ``normalizer``: ``ObsNormalizer.save()`` JSON sidecar at ``norm.json``
  - ``meta``: run config, demo stats, obs/action dims

PPO loads exactly these two artifacts at start so the obs distribution,
network shape, and initial actor weights all stay consistent.
"""

from __future__ import annotations

import json


from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path


def _json_safe(v):
    """Recursively coerce Path / list-of-Path / dataclass into JSON-safe form."""
    if isinstance(v, Path):
        return str(v)
    # Dataclass instances (e.g. ActionChunkingCfg) → dict, then recurse.
    if is_dataclass(v) and not isinstance(v, type):
        return _json_safe(asdict(v))
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    return v

import numpy as np
import torch
import torch.nn as nn

from khj_rl.training.base import EnvLike, Trainer, TrainerConfig
from khj_rl.training.demo_buffer import DemoBuffer, DemoStats
from khj_rl.training.network import ActorCritic, ChunkedActorCritic
from khj_rl.training.normalizer import ObsNormalizer


@dataclass
class ActionChunkingCfg:
    """ACT-style action chunking for BC (Phase 1, 2026-05-18).

    Phase 1 첫 통과는 enabled=True + chunk_size=12. PPO A/D 둘 다 0%로
    실패해 BC 자체를 covariate-shift 면역화하기 위해 도입.

    Codex review 권고(2026-05-18) 반영:
      - inference_alpha=0.2 (lag 완화: 0.1은 4+ step drift 위험)
      - gripper_use_latest=True (release/retreat timing drift 차단; F 항목)
      - off-by-one 회피용 valid_starts precompute는 DemoBuffer 쪽에서 처리
    """

    enabled: bool = True
    chunk_size: int = 12             # 160 horizon의 7.5%
    inference_alpha: float = 0.2     # temporal ensemble decay (chunk[0]에서 멀수록 가벼움)
    gripper_use_latest: bool = True  # gripper dim은 ensemble 우회, chunk[0]만 사용
    loss_type: str = "l1"            # "l1" or "l2"


@dataclass
class BCConfig:
    # Single dir (Path) or list of dirs (multiple demo batches merged in).
    # DemoBuffer.from_dir() handles both branches; the type stays loose
    # here so the CLI can pass a list straight through.
    demo_dir: "Path | list[Path]"
    obs_dim: int
    action_dim: int
    success_only: bool = True
    hidden_dim: int = 256
    batch_size: int = 256
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 0.0
    actor_logstd_init: float = -1.0
    device: str = "cuda:0"
    log_every: int = 100  # mini-batches
    action_chunking: ActionChunkingCfg = field(default_factory=ActionChunkingCfg)
    extras: dict = field(default_factory=dict)


class BCTrainer(Trainer):
    def __init__(
        self,
        env: EnvLike,
        config: TrainerConfig,
        bc_config: BCConfig,
    ) -> None:
        super().__init__(env, config)
        self.bc_config = bc_config

        self._device = torch.device(bc_config.device)
        self._rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)

        # Load demos + fit normalizer.
        self._demos, self._demo_stats = DemoBuffer.from_dir(
            bc_config.demo_dir, success_only=bc_config.success_only
        )
        self._normalizer = ObsNormalizer.fit(self._demos.obs)
        chunking = bc_config.action_chunking
        if chunking.enabled:
            self._net = ChunkedActorCritic(
                obs_dim=bc_config.obs_dim,
                action_dim=bc_config.action_dim,
                chunk_size=chunking.chunk_size,
                hidden_dim=bc_config.hidden_dim,
                actor_logstd_init=bc_config.actor_logstd_init,
            ).to(self._device)
        else:
            self._net = ActorCritic(
                obs_dim=bc_config.obs_dim,
                action_dim=bc_config.action_dim,
                hidden_dim=bc_config.hidden_dim,
                actor_logstd_init=bc_config.actor_logstd_init,
            ).to(self._device)

        # Only the actor (and its logstd) get gradients during BC. Leaving
        # the critic at random init is fine: PPO will warm it up before
        # touching the actor again.
        actor_params = [
            *self._net.actor.parameters(),
            self._net.actor_logstd,
        ]
        self._optim = torch.optim.Adam(
            actor_params, lr=bc_config.lr, weight_decay=bc_config.weight_decay
        )

    # ---- Trainer API ----------------------------------------------------

    def train(self) -> None:
        cfg = self.bc_config
        chunking = cfg.action_chunking
        if chunking.enabled:
            n_pairs = int(self._demos.valid_chunk_starts(chunking.chunk_size).shape[0])
            n_batches = max(1, n_pairs // cfg.batch_size)
            print(
                f"[BC] demos: {self._demo_stats.n_episodes} ep / "
                f"{self._demo_stats.n_steps} steps; "
                f"ACT chunk_size={chunking.chunk_size} loss={chunking.loss_type} "
                f"valid_starts={n_pairs}; "
                f"epochs={cfg.epochs} batch_size={cfg.batch_size} "
                f"(~{n_batches} batches/epoch)",
                flush=True,
            )
            self._train_chunked()
        else:
            n_train = len(self._demos)
            n_batches = max(1, n_train // cfg.batch_size)
            print(
                f"[BC] demos: {self._demo_stats.n_episodes} ep / "
                f"{self._demo_stats.n_steps} steps; "
                f"epochs={cfg.epochs} batch_size={cfg.batch_size} "
                f"(~{n_batches} batches/epoch)",
                flush=True,
            )
            self._train_step1()

    def _train_step1(self) -> None:
        cfg = self.bc_config
        step = 0
        for epoch in range(cfg.epochs):
            epoch_loss = 0.0
            seen = 0
            for obs_np, act_np in self._demos.iter_minibatches(
                cfg.batch_size, self._rng
            ):
                obs = torch.from_numpy(self._normalizer.normalize_np(obs_np)).to(
                    self._device
                )
                target = torch.from_numpy(act_np).to(self._device)
                pred_mean = self._net.actor_mean(obs)
                loss = nn.functional.mse_loss(pred_mean, target)

                self._optim.zero_grad(set_to_none=True)
                loss.backward()
                self._optim.step()

                epoch_loss += float(loss.detach().item()) * obs.shape[0]
                seen += obs.shape[0]
                step += 1
                if step % cfg.log_every == 0:
                    print(
                        f"[BC] step={step} epoch={epoch} batch_loss={float(loss):.5f}",
                        flush=True,
                    )
            print(
                f"[BC] epoch {epoch + 1}/{cfg.epochs} "
                f"mean_loss={epoch_loss / max(1, seen):.5f}",
                flush=True,
            )

    def _train_chunked(self) -> None:
        cfg = self.bc_config
        chunking = cfg.action_chunking
        k = chunking.chunk_size
        # Codex G: track per-offset loss so we can see whether tail of the chunk
        # is well-fit (k-step plan quality) vs head only.
        loss_fn = (
            nn.functional.l1_loss
            if chunking.loss_type == "l1"
            else nn.functional.mse_loss
        )
        step = 0
        for epoch in range(cfg.epochs):
            epoch_loss = 0.0
            seen = 0
            offset_loss_sum = np.zeros(k, dtype=np.float64)
            offset_seen = 0
            for obs_np, chunk_np in self._demos.iter_chunked_minibatches(
                cfg.batch_size, k, self._rng
            ):
                obs = torch.from_numpy(self._normalizer.normalize_np(obs_np)).to(
                    self._device
                )
                target = torch.from_numpy(chunk_np).to(self._device)  # (B, k, A)
                pred_chunks = self._net.actor_chunks(obs)              # (B, k, A)
                loss = loss_fn(pred_chunks, target)

                self._optim.zero_grad(set_to_none=True)
                loss.backward()
                self._optim.step()

                B = obs.shape[0]
                epoch_loss += float(loss.detach().item()) * B
                seen += B
                # mean over (B, A) per offset j → (k,)
                with torch.no_grad():
                    if chunking.loss_type == "l1":
                        per_off = (pred_chunks - target).abs().mean(dim=(0, 2))
                    else:
                        per_off = (pred_chunks - target).pow(2).mean(dim=(0, 2))
                offset_loss_sum += per_off.detach().cpu().numpy().astype(np.float64) * B
                offset_seen += B
                step += 1
                if step % cfg.log_every == 0:
                    print(
                        f"[BC-ACT] step={step} epoch={epoch} "
                        f"batch_loss={float(loss):.5f}",
                        flush=True,
                    )
            off_str = " ".join(
                f"j{j}={offset_loss_sum[j] / max(1, offset_seen):.4f}"
                for j in range(k)
            )
            print(
                f"[BC-ACT] epoch {epoch + 1}/{cfg.epochs} "
                f"mean_loss={epoch_loss / max(1, seen):.5f} "
                f"per_offset[{off_str}]",
                flush=True,
            )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        chunking = self.bc_config.action_chunking
        # Codex I: stamp chunking provenance into the checkpoint so the eval
        # loader can hard-assert and refuse mismatched configs.
        torch.save(
            {
                "actor_critic": self._net.state_dict(),
                "obs_dim": self.bc_config.obs_dim,
                "action_dim": self.bc_config.action_dim,
                "hidden_dim": self.bc_config.hidden_dim,
                "actor_logstd_init": self.bc_config.actor_logstd_init,
                "demo_stats": vars(self._demo_stats),
                "seed": self.config.seed,
                "chunking_enabled": bool(chunking.enabled),
                "chunk_size": int(chunking.chunk_size) if chunking.enabled else 0,
                "inference_alpha": float(chunking.inference_alpha),
                "gripper_use_latest": bool(chunking.gripper_use_latest),
                "loss_type": str(chunking.loss_type),
            },
            path,
        )
        self._normalizer.save(path.with_name("norm.json"))
        meta_path = path.with_name("bc_meta.json")
        with meta_path.open("w") as f:
            json.dump(
                {
                    "run_name": self.config.run_name,
                    "seed": self.config.seed,
                    "bc_config": {
                        k: _json_safe(v)
                        for k, v in vars(self.bc_config).items()
                    },
                    "demo_stats": vars(self._demo_stats),
                },
                f,
                indent=2,
            )

    def load(self, path: Path) -> None:
        ckpt = torch.load(Path(path), map_location=self._device)
        self._net.load_state_dict(ckpt["actor_critic"])
        self._normalizer = ObsNormalizer.load(Path(path).with_name("norm.json"))

    # ---- accessors (PPO uses these to take over after BC) ---------------

    @property
    def network(self) -> ActorCritic:
        return self._net

    @property
    def normalizer(self) -> ObsNormalizer:
        return self._normalizer

    @property
    def demo_stats(self) -> DemoStats:
        return self._demo_stats
