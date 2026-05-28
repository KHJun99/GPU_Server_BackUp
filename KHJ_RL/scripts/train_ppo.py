"""PPO fine-tune script (BC -> PPO transfer for Phase 1).

Boots Isaac Sim once, constructs the cube-lift env, loads a BC
checkpoint into :class:`khj_rl.training.ppo.PPOTrainer`, runs the
training loop, and writes the final ``ppo.pt`` next to the BC
artifacts. The 5 BC->PPO transfer guards from CLAUDE.md ## Phase 1
결정 사항 7 are baked in here:

  1) Network sharing  — PPOTrainer uses the same ActorCritic class.
  2) actor logstd init = -1.0 (PPOConfig default).
  3) Critic warm-up   — first ``critic_warmup_iters`` iters freeze
     the actor and only update the critic so garbage advantages
     don't wreck the BC weights.
  4) Low LR after BC  — PPOConfig default 1e-4 (vs 3e-4 baseline).
  5) Frozen obs norm  — ``trainer.load_bc()`` swaps the running
     normalizer for the BC dataset's frozen mean/std.

Before training we also run ``assert_obs_alignment`` (the env reset
obs matches the cfg layout) and, when the BC artifacts include a
sidecar normalizer, ``assert_obs_distribution`` (env reset obs sits
inside ±sigma_tolerance σ of the BC training distribution). Both
guards bail loudly rather than letting a sim2real / wrapper drift
silently corrupt PPO from step 0 — the same class of failure that
hit ``jabis_sim_v2``.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \\
        python scripts/train_ppo.py \\
            --bc-ckpt runs/stage0_bc/bc.pt \\
            --run-name stage0_ppo \\
            --total-steps 1000000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# CUDA / pinocchio bootstrap mirrors launch_viewer.py so the env init
# path is identical to demo collection and the viewer. Pinocchio MUST
# be imported before AppLauncher boots (Assimp ABI conflict).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("KHJ_RL_SIM_DEVICE", "cuda:0")

import pinocchio  # noqa: F401, E402

from isaaclab.app import AppLauncher  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bc-ckpt", type=Path, default=None,
        help="BC checkpoint produced by train_bc.py. If omitted, PPO "
             "starts from random init (smoke test only — Phase 1 expects "
             "the BC handoff).",
    )
    parser.add_argument("--run-name", type=str, default="stage0_ppo")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--critic-warmup-iters", type=int, default=15)
    parser.add_argument(
        "--ent-coef", type=float, default=0.01,
        help="PPO entropy bonus weight. Default 0.01 (CleanRL standard). "
             "Set to 0 to freeze BC's actor_logstd from drifting upward — "
             "useful when BC handoff is fragile and entropy is observed "
             "rising over training.",
    )
    parser.add_argument(
        "--actor-logstd-override", type=float, default=None,
        help="If set, overrides the actor_logstd value loaded from the BC "
             "ckpt. Use to shrink exploration noise when the BC pretrain "
             "is precise (e.g. -2.5 → std ≈ 0.082) so stochastic PPO "
             "sampling doesn't drag the policy off the demo manifold.",
    )
    parser.add_argument(
        "--dense", action="store_true",
        help="Enable A-option dense shaping (cfg.dense_reward.enabled=True). "
             "Defaults wired in DenseRewardCfg per docs/dense_reward_design.md. "
             "Activates bounded+annealed+phase-gated dense reward to break "
             "the sparse-signal bottleneck observed at BC 19 percent.",
    )
    parser.add_argument(
        "--video-every-steps", type=int, default=0,
        help="If > 0, sample a rollout video every N env steps "
             "(builds an extra eval env with viewer_camera enabled). "
             "Default 0 = video sampling disabled.",
    )
    parser.add_argument(
        "--video-rollout-episodes", type=int, default=2,
        help="Number of episodes captured per video sample.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-dir", type=Path, default=Path("runs"),
        help="Parent directory for run outputs; ckpt lands at "
             "{log_dir}/{run_name}/ppo.pt.",
    )
    parser.add_argument(
        "--skip-distribution-check", action="store_true",
        help="Skip the BC distribution assert_obs_distribution check at "
             "startup (still runs the structural assert_obs_alignment).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # enable_cameras must be ON whenever the env carries a viewer
    # camera — otherwise AppLauncher refuses to spawn the sensor and
    # the env init raises (the WebRTC path uses it too).
    sim_app = AppLauncher(
        headless=True,
        enable_cameras=args.video_every_steps > 0,
    ).app

    # All khj_rl imports happen post-AppLauncher (carb bootstrap).
    import numpy as np

    from khj_rl.envs import CubeLiftEnv, CubeLiftEnvCfg
    from khj_rl.eval.obs_alignment import (
        assert_obs_alignment,
        assert_obs_distribution,
    )
    from khj_rl.eval.video_sampler import VideoSampler
    from khj_rl.training.base import TrainerConfig
    from khj_rl.training.ppo import PPOConfig, PPOTrainer

    cfg = CubeLiftEnvCfg()
    if args.dense:
        cfg.dense_reward.enabled = True
        print(
            f"[train_ppo] DENSE SHAPING ENABLED — alpha0={cfg.dense_reward.alpha0} "
            f"decay_steps={cfg.dense_reward.decay_steps} "
            f"(weights: ee_cube={cfg.dense_reward.w_ee_cube_dist} "
            f"lift={cfg.dense_reward.w_lifted_bonus} "
            f"cube_goal={cfg.dense_reward.w_cube_goal_dist} "
            f"released={cfg.dense_reward.w_released_bonus})",
            flush=True,
        )
    # ``include_viewer_camera=True`` is only needed when we plan to
    # sample videos — otherwise we skip the extra render cost. The
    # video_sampler reads env._viewer_camera directly and silently
    # no-ops when the sensor is missing.
    env = CubeLiftEnv(cfg, include_viewer_camera=args.video_every_steps > 0)

    # ---- Phase 1 guards (CLAUDE.md ## Phase 1 결정 사항 8) ----
    print("[train_ppo] running obs_alignment guard...", flush=True)
    assert_obs_alignment(env, cfg)

    out_root = Path(args.log_dir) / args.run_name
    ppo_cfg = PPOConfig(
        obs_dim=cfg.obs_total_size,
        action_dim=cfg.action_size,
        num_steps=args.num_steps,
        lr=args.lr,
        critic_warmup_iters=args.critic_warmup_iters,
        ent_coef=args.ent_coef,
    )
    trainer_cfg = TrainerConfig(
        run_name=args.run_name,
        total_steps=args.total_steps,
        seed=args.seed,
        log_dir=Path(args.log_dir),
    )
    video_sampler = None
    if args.video_every_steps > 0:
        video_dir = out_root / "videos"
        video_sampler = VideoSampler(
            output_dir=video_dir,
            every_steps=args.video_every_steps,
            rollout_episodes=args.video_rollout_episodes,
        )
        print(
            f"[train_ppo] video_sampler enabled: every "
            f"{args.video_every_steps} steps -> {video_dir}",
            flush=True,
        )

    trainer = PPOTrainer(
        env=env,
        config=trainer_cfg,
        ppo_config=ppo_cfg,
        video_sampler=video_sampler,
    )

    if args.bc_ckpt is not None:
        print(f"[train_ppo] loading BC checkpoint {args.bc_ckpt}", flush=True)
        trainer.load_bc(Path(args.bc_ckpt))
        if args.actor_logstd_override is not None:
            import torch as _torch
            with _torch.no_grad():
                trainer.network.actor_logstd.data.fill_(
                    float(args.actor_logstd_override)
                )
            print(
                f"[train_ppo] actor_logstd overridden to "
                f"{args.actor_logstd_override} (std≈"
                f"{float(_torch.tensor(args.actor_logstd_override).exp()):.4f})",
                flush=True,
            )
        if not args.skip_distribution_check:
            norm = trainer.normalizer
            print(
                "[train_ppo] running obs_distribution guard (BC frozen "
                "mean/std) ...",
                flush=True,
            )
            assert_obs_distribution(
                env, cfg, mean=np.asarray(norm.mean), std=np.asarray(norm.std),
            )
    else:
        print(
            "[train_ppo] WARNING: no --bc-ckpt; PPO starts from random init "
            "(Phase 1 normally expects BC pretrain).",
            flush=True,
        )

    print(
        f"[train_ppo] starting PPO: total_steps={args.total_steps} "
        f"num_steps/iter={args.num_steps} lr={args.lr} "
        f"critic_warmup_iters={args.critic_warmup_iters}",
        flush=True,
    )
    trainer.train()

    out_path = out_root / "ppo.pt"
    trainer.save(out_path)
    print(f"[train_ppo] saved {out_path}", flush=True)

    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
