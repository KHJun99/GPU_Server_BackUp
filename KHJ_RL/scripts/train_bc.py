"""Behavior-cloning pretrain script.

Wraps :class:`khj_rl.training.bc.BCTrainer` in a CLI so the operator can
kick off BC after demo collection. BC itself only needs the demo buffer
and the env's obs / action sizes — it never calls ``env.step`` or
``env.reset`` during training. To avoid pulling Isaac Sim into a job
that is purely a torch dataloader + MSE loop, we hand the trainer a
``ShapesOnlyEnv`` that satisfies the ``EnvLike`` Protocol with the obs
and action spaces derived directly from ``CubeLiftEnvCfg``.

Usage::

    python scripts/train_bc.py \\
        --demo-dir runs/demos/stage0 \\
        --run-name stage0_bc \\
        --epochs 50

Output: ``runs/{run_name}/bc.pt`` + ``norm.json`` + ``bc_meta.json``.
PPO loads exactly these artifacts (see CLAUDE.md ## Phase 1 결정 사항 7
step 5: BC dataset's frozen mean/std becomes PPO's obs normalizer).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


class ShapesOnlyEnv:
    """Lightweight ``EnvLike`` that only exposes obs/action shapes.

    BCTrainer doesn't roll out the env — it just needs to know obs and
    action dimensionality (and the Protocol's reset/step signatures so
    the type checker stops complaining). Calling ``reset`` / ``step``
    on this raises; if BCTrainer ever tries to step the env we want a
    loud error rather than silent garbage.
    """

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        self._obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

    @property
    def observation_space(self) -> Any:
        return self._obs_space

    @property
    def action_space(self) -> Any:
        return self._action_space

    def reset(self) -> Any:
        raise RuntimeError("ShapesOnlyEnv does not support reset().")

    def step(self, action: Any) -> Any:
        raise RuntimeError("ShapesOnlyEnv does not support step().")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-dir", type=Path, nargs="+", default=[Path("runs/demos/stage0")],
        help="One or more directories containing collect_demos NPZ files. "
             "When multiple are passed, all NPZ files across them are merged "
             "into a single BC buffer (e.g. stage0 1k + stage0_extra 2k).",
    )
    parser.add_argument(
        "--run-name", type=str, default="stage0_bc",
        help="Subdir under runs/ for the BC checkpoint + sidecar files.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--actor-logstd-init", type=float, default=-1.0,
        help="Initial actor logstd. PPO transfer is fragile if std starts "
             "too large — CLAUDE.md ## Phase 1 결정 사항 7 step 2 mandates "
             "logstd init around -1.0 (std ≈ 0.37).",
    )
    # ACT (action chunking) — 2026-05-18 Phase 1 첫 통과는 chunking ON,
    # chunk_size=12. 1-step BC로 돌아가려면 --no-chunking.
    parser.add_argument(
        "--no-chunking", dest="chunking", action="store_false",
        help="Disable ACT chunking; fall back to 1-step BC.",
    )
    parser.set_defaults(chunking=True)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument(
        "--chunk-loss", choices=("l1", "l2"), default="l1",
        help="Per-element loss for chunked BC. ACT paper uses L1.",
    )
    parser.add_argument(
        "--inference-alpha", type=float, default=0.2,
        help="Temporal-ensemble decay rate at eval time. Stored in the "
             "checkpoint meta and read by eval_policy.py.",
    )
    parser.add_argument(
        "--gripper-use-latest", action="store_true", default=True,
        help="(default ON) eval skips temporal-ensemble for the gripper dim "
             "and uses chunk[0] directly so release/retreat timing isn't "
             "smeared. Disable with --no-gripper-use-latest.",
    )
    parser.add_argument(
        "--no-gripper-use-latest", dest="gripper_use_latest", action="store_false",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--log-dir", type=Path, default=Path("runs"),
        help="Parent directory for run outputs; checkpoint lands at "
             "{log_dir}/{run_name}/bc.pt.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Imports stay inside main() so importing this module doesn't pull
    # torch / khj_rl until the CLI actually runs.
    from khj_rl.envs.cube_lift.cfg import CubeLiftEnvCfg
    from khj_rl.training.base import TrainerConfig
    from khj_rl.training.bc import ActionChunkingCfg, BCConfig, BCTrainer

    cfg = CubeLiftEnvCfg()
    obs_dim = cfg.obs_total_size
    action_dim = cfg.action_size

    env = ShapesOnlyEnv(obs_dim=obs_dim, action_dim=action_dim)

    out_root = Path(args.log_dir) / args.run_name
    chunking_cfg = ActionChunkingCfg(
        enabled=bool(args.chunking),
        chunk_size=int(args.chunk_size),
        inference_alpha=float(args.inference_alpha),
        gripper_use_latest=bool(args.gripper_use_latest),
        loss_type=str(args.chunk_loss),
    )
    # CLI uses nargs="+" so args.demo_dir is always a list[Path].
    # Pass it through as-is — DemoBuffer.from_dir handles list-vs-single.
    bc_cfg = BCConfig(
        demo_dir=list(args.demo_dir),
        obs_dim=obs_dim,
        action_dim=action_dim,
        success_only=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        actor_logstd_init=args.actor_logstd_init,
        device=args.device,
        action_chunking=chunking_cfg,
    )
    trainer_cfg = TrainerConfig(
        run_name=args.run_name,
        # BCTrainer never reads ``config.total_steps`` (BC has no env
        # rollout) — leaving it at 0 makes the field's irrelevance
        # to BC explicit rather than implying a step budget.
        total_steps=0,
        seed=args.seed,
        log_dir=Path(args.log_dir),
    )
    trainer = BCTrainer(env=env, config=trainer_cfg, bc_config=bc_cfg)

    print(
        f"[train_bc] demos={args.demo_dir} -> {out_root}/bc.pt "
        f"epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
        f"chunking={args.chunking} chunk_size={args.chunk_size} "
        f"loss={args.chunk_loss} alpha={args.inference_alpha} "
        f"gripper_use_latest={args.gripper_use_latest}",
        flush=True,
    )
    trainer.train()
    out_path = out_root / "bc.pt"
    trainer.save(out_path)
    print(f"[train_bc] saved {out_path} + norm.json + bc_meta.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
