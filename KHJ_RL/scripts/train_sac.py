"""SACfD training script (Card 1 of the 2026-05-19 decision tree).

Boots Isaac Sim once, loads a 1-step BC checkpoint into the SAC actor via
surgical transfer, builds the demo replay buffer from the same NPZ dirs
used for BC, and runs the training loop. Mirrors ``train_ppo.py``'s
guards and CLI flags so the operator can hot-swap between trainers
without learning a new interface.

Usage (Phase B smoke)::

    OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y CUDA_VISIBLE_DEVICES=0 \\
        python scripts/train_sac.py \\
            --bc-ckpt runs/stage01_bc_1step_v1/bc.pt \\
            --demo-dir runs/demos/stage0 runs/demos/stage0_extra \\
                       runs/demos/stage1 runs/demos/stage1_extra runs/demos/stage1_extra2 \\
            --run-name stage0_sacfd_smoke \\
            --total-steps 50000 \\
            --learning-starts 5000 \\
            --video-every-steps 0

Phase C full adds ``--total-steps 500000 --video-every-steps 5000``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("KHJ_RL_SIM_DEVICE", "cuda:0")

import pinocchio  # noqa: F401, E402

from isaaclab.app import AppLauncher  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bc-ckpt", type=Path, default=None,
        help="1-step BC checkpoint produced by train_bc.py --no-chunking. "
             "Chunked checkpoints are rejected by SACfDTrainer.load_bc_actor. "
             "In Card C' mode (--no-bc-loss + --her) this can be omitted — "
             "actor starts from random SAC init.",
    )
    p.add_argument(
        "--demo-dir", type=Path, nargs="+", required=True,
        help="Demo NPZ directories for the demo replay buffer. Pass the "
             "same ones used for BC pretrain so the actor and Q-net see "
             "the same trajectory distribution.",
    )
    p.add_argument("--run-name", type=str, default="stage0_sacfd")
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument(
        "--critic-warmup-steps", type=int, default=2_000,
        help="F12: number of SAC update steps after learning_starts where "
             "actor/alpha optimizers are skipped. Critic learns Q for the "
             "BC-driven trajectories before actor is allowed to chase it.",
    )
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--alpha-lr", type=float, default=3e-4)
    p.add_argument("--init-alpha", type=float, default=0.2)
    p.add_argument(
        "--target-entropy", type=float, default=None,
        help="SAC auto-alpha target entropy. Default None → -action_dim "
             "(=-6). Raise toward 0 (e.g. -3) to keep more exploration "
             "in sparse-reward env, which is the Card A tweak.",
    )
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--bc-loss-weight", type=float, default=1.0)
    p.add_argument("--bc-anneal-steps", type=int, default=200_000)
    p.add_argument(
        "--no-q-filter", dest="q_filter", action="store_false",
        help="Disable Q-filter on the BC auxiliary loss; apply unconditionally.",
    )
    p.set_defaults(q_filter=True)
    p.add_argument(
        "--no-bc-loss", dest="bc_loss_enabled", action="store_false",
        help="Card C': disable actor BC auxiliary loss entirely (anchor 0).",
    )
    p.set_defaults(bc_loss_enabled=True)
    p.add_argument(
        "--no-entropy-term", dest="no_entropy_term", action="store_true",
        help="F6 / TD3-like: remove entropy bonus from Bellman target_v "
             "and actor_loss. Use when LOG_STD_MAX is tight (e.g. -1.6) "
             "and squashed Gaussian log π > 0 causes the entropy term "
             "to subtract from Q, dragging it negative.",
    )
    p.set_defaults(no_entropy_term=False)
    p.add_argument(
        "--no-bc-transfer", dest="bc_transfer", action="store_false",
        help="Card C': skip BC weight transfer to SAC actor. Actor starts "
             "from SAC random init.",
    )
    p.set_defaults(bc_transfer=True)
    p.add_argument(
        "--her", dest="her_enabled", action="store_true",
        help="Card C': enable Hindsight Experience Replay buffer.",
    )
    p.set_defaults(her_enabled=False)
    p.add_argument("--her-k-future", type=int, default=4)
    p.add_argument("--her-ratio-initial", type=float, default=0.4)
    p.add_argument("--her-ratio-final", type=float, default=0.3)
    p.add_argument("--her-ratio-switch-step", type=int, default=50_000)
    p.add_argument("--her-max-episodes", type=int, default=6_000)
    # G+ task curriculum
    p.add_argument(
        "--task-level", type=int, default=0,
        help="G+ sub-task curriculum level: 0=lift-only, 1=lift+place, "
             "2=full PnP. cfg.curriculum.task_level passes through to "
             "env reward + termination logic.",
    )
    p.add_argument(
        "--curriculum-stage", type=int, default=0,
        help="cfg.curriculum.current_stage_idx (cube xy spawn radius). "
             "G+ adds 2cm at index 0.",
    )
    # Card D — dense reward (existing DenseRewardCfg in cfg.py, see
    # docs/dense_reward_design.md). When --her is on we typically
    # zero out the goal-dependent dense terms so HER handles those.
    p.add_argument(
        "--dense", action="store_true",
        help="Enable cfg.dense_reward (A-option shaping). For Card D "
             "we additionally let the per-term weights be overridden "
             "via --dense-w-* flags.",
    )
    p.add_argument("--dense-alpha0", type=float, default=None)
    p.add_argument("--dense-decay-steps", type=int, default=None)
    p.add_argument("--dense-w-ee-cube", type=float, default=None)
    p.add_argument("--dense-w-lifted", type=float, default=None)
    p.add_argument("--dense-w-cube-goal", type=float, default=None)
    p.add_argument("--dense-w-released", type=float, default=None)
    p.add_argument("--demo-ratio-initial", type=float, default=0.5)
    p.add_argument("--demo-ratio-final", type=float, default=0.25)
    p.add_argument("--demo-ratio-switch-step", type=int, default=50_000)
    p.add_argument("--update-every", type=int, default=1)
    p.add_argument("--log-every-steps", type=int, default=2048)
    p.add_argument(
        "--video-every-steps", type=int, default=0,
        help="If > 0, sample a rollout video every N env steps.",
    )
    p.add_argument("--video-rollout-episodes", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--log-dir", type=Path, default=Path("runs"),
        help="Parent directory for run outputs; ckpt lands at "
             "{log_dir}/{run_name}/sac.pt.",
    )
    p.add_argument(
        "--skip-distribution-check", action="store_true",
        help="Skip the BC distribution assert_obs_distribution check at "
             "startup (still runs the structural assert_obs_alignment).",
    )
    p.add_argument(
        "--resume", type=Path, default=None,
        help="Resume from a saved sac.pt checkpoint (loads actor, critic, "
             "critic_target, log_alpha, normalizer). Skips BC weight transfer.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    sim_app = AppLauncher(
        headless=True,
        enable_cameras=args.video_every_steps > 0,
    ).app

    import numpy as np

    from khj_rl.envs import CubeLiftEnv, CubeLiftEnvCfg
    from khj_rl.eval.obs_alignment import (
        assert_obs_alignment,
        assert_obs_distribution,
    )
    from khj_rl.eval.video_sampler import VideoSampler
    from khj_rl.training.base import TrainerConfig
    from khj_rl.training.demo_buffer import DemoBuffer, DemoReplay
    from khj_rl.training.her import HERBuffer, HERConfig
    from khj_rl.training.normalizer import ObsNormalizer
    from khj_rl.training.sac import SACConfig, SACfDTrainer

    cfg = CubeLiftEnvCfg()
    cfg.curriculum.task_level = int(args.task_level)
    cfg.curriculum.current_stage_idx = int(args.curriculum_stage)
    print(
        f"[train_sac] curriculum: task_level={cfg.curriculum.task_level} "
        f"stage_idx={cfg.curriculum.current_stage_idx} "
        f"(spawn_side={cfg.curriculum.side_length_m[cfg.curriculum.current_stage_idx]}m)",
        flush=True,
    )
    if args.dense:
        cfg.dense_reward.enabled = True
        if args.dense_alpha0 is not None:
            cfg.dense_reward.alpha0 = float(args.dense_alpha0)
        if args.dense_decay_steps is not None:
            cfg.dense_reward.decay_steps = int(args.dense_decay_steps)
        if args.dense_w_ee_cube is not None:
            cfg.dense_reward.w_ee_cube_dist = float(args.dense_w_ee_cube)
        if args.dense_w_lifted is not None:
            cfg.dense_reward.w_lifted_bonus = float(args.dense_w_lifted)
        if args.dense_w_cube_goal is not None:
            cfg.dense_reward.w_cube_goal_dist = float(args.dense_w_cube_goal)
        if args.dense_w_released is not None:
            cfg.dense_reward.w_released_bonus = float(args.dense_w_released)
        print(
            f"[train_sac] DENSE enabled: alpha0={cfg.dense_reward.alpha0} "
            f"decay_steps={cfg.dense_reward.decay_steps} "
            f"w_ee_cube={cfg.dense_reward.w_ee_cube_dist} "
            f"w_lifted={cfg.dense_reward.w_lifted_bonus} "
            f"w_cube_goal={cfg.dense_reward.w_cube_goal_dist} "
            f"w_released={cfg.dense_reward.w_released_bonus}",
            flush=True,
        )
    env = CubeLiftEnv(cfg, include_viewer_camera=args.video_every_steps > 0)

    print("[train_sac] running obs_alignment guard...", flush=True)
    assert_obs_alignment(env, cfg)

    # Normalizer source: when bc_ckpt is given, reuse its frozen mean/std
    # (matches CLAUDE.md decision 7 step 5). In Card C' mode (no bc_ckpt),
    # fit normalizer fresh from demo obs + hypothetical HER-relabeled
    # target_delta samples so the HER batches don't land on a distribution
    # the normalizer never saw (Codex REVISE 2).
    if args.bc_ckpt is not None:
        bc_norm_path = Path(args.bc_ckpt).with_name("norm.json")
        print(f"[train_sac] loading BC normalizer {bc_norm_path}", flush=True)
        normalizer = ObsNormalizer.load(bc_norm_path)
    else:
        print(
            "[train_sac] no bc_ckpt — fitting normalizer from demo obs + "
            "HER hypothetical relabel samples",
            flush=True,
        )
        demo_buffer, demo_buf_stats = DemoBuffer.from_dir(
            list(args.demo_dir), success_only=True
        )
        normalizer = ObsNormalizer.fit(demo_buffer.obs)
        # Mix in hypothetical relabeled target_delta values: for K random
        # (ep, t, t') triples, the relabeled obs differs from the demo obs
        # only in indices 17:20 (target_delta_base). Build synthetic obs
        # rows whose target_delta channel is the relabel value and
        # everything else copies a sampled demo obs, then re-fit.
        rng_init = np.random.default_rng(args.seed)
        K = min(50_000, int(demo_buffer.obs.shape[0]))
        idx = rng_init.integers(0, demo_buffer.obs.shape[0], size=K)
        synth = demo_buffer.obs[idx].copy()
        # Sample t' offset uniform in [1, 40] step within episode (matches
        # rough HER future horizon for our 160-step episodes). For each
        # row we don't have per-ep span lookup here — use a global cube
        # path average shift. cube_xyz lives at indices 14:17. delta_xyz
        # = cube_xyz[idx_off] - cube_xyz[idx] where idx_off = idx + shift
        # clipped to buffer length.
        shifts = rng_init.integers(1, 40, size=K)
        idx_off = np.clip(idx + shifts, 0, demo_buffer.obs.shape[0] - 1)
        cube_now = demo_buffer.obs[idx, 14:17]
        cube_future = demo_buffer.obs[idx_off, 14:17]
        new_target_delta = cube_future - cube_now
        synth[:, 17:20] = new_target_delta
        # Refit on demo + synth concatenated.
        combined = np.concatenate([demo_buffer.obs, synth], axis=0)
        normalizer = ObsNormalizer.fit(combined)
        del demo_buffer, combined, synth

    if args.bc_ckpt is not None and not args.skip_distribution_check:
        print(
            "[train_sac] running obs_distribution guard (BC frozen mean/std)...",
            flush=True,
        )
        assert_obs_distribution(
            env, cfg,
            mean=np.asarray(normalizer.mean),
            std=np.asarray(normalizer.std),
        )

    # Demo replay — converts (obs, actions, rewards) NPZ into SAC tuples.
    print(f"[train_sac] loading demo replay from {args.demo_dir}", flush=True)
    demo_replay = DemoReplay.from_dir(list(args.demo_dir), success_only=True)
    print(
        f"[train_sac] demo_replay: {demo_replay.stats.n_episodes} ep / "
        f"{demo_replay.stats.n_transitions} transitions",
        flush=True,
    )

    # HER buffer (Card C').
    her_buffer = None
    if args.her_enabled:
        her_cfg = HERConfig(
            enabled=True,
            strategy="future",
            k_future=int(args.her_k_future),
            max_episodes=int(args.her_max_episodes),
        )
        her_buffer = HERBuffer(
            cfg=her_cfg,
            env_cfg_goal_radius_xy_m=cfg.goal.radius_xy_m,
            env_cfg_success=cfg.success,
            gripper_open_threshold=cfg.robot.gripper_open_threshold,
        )
        print(
            f"[train_sac] HER enabled: strategy=future k={args.her_k_future} "
            f"max_eps={args.her_max_episodes}",
            flush=True,
        )

    out_root = Path(args.log_dir) / args.run_name

    sac_cfg = SACConfig(
        obs_dim=cfg.obs_total_size,
        action_dim=cfg.action_size,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        init_alpha=args.init_alpha,
        target_entropy=args.target_entropy,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        critic_warmup_steps=int(args.critic_warmup_steps),
        update_every=args.update_every,
        demo_ratio_initial=args.demo_ratio_initial,
        demo_ratio_final=args.demo_ratio_final,
        demo_ratio_switch_step=args.demo_ratio_switch_step,
        bc_loss_weight=args.bc_loss_weight,
        bc_loss_q_filter=bool(args.q_filter),
        bc_anneal_steps=args.bc_anneal_steps,
        bc_loss_enabled=bool(args.bc_loss_enabled),
        no_entropy_term=bool(args.no_entropy_term),
        her_enabled=bool(args.her_enabled),
        her_ratio_initial=float(args.her_ratio_initial),
        her_ratio_final=float(args.her_ratio_final),
        her_ratio_switch_step=int(args.her_ratio_switch_step),
        log_every_steps=args.log_every_steps,
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
            f"[train_sac] video_sampler enabled: every "
            f"{args.video_every_steps} steps -> {video_dir}",
            flush=True,
        )

    trainer = SACfDTrainer(
        env=env,
        config=trainer_cfg,
        sac_config=sac_cfg,
        demo_replay=demo_replay,
        normalizer=normalizer,
        video_sampler=video_sampler,
        her_buffer=her_buffer,
    )

    if args.resume is not None:
        print(f"[train_sac] resuming from {args.resume}", flush=True)
        trainer.load(Path(args.resume))
    elif args.bc_ckpt is not None and args.bc_transfer:
        print(f"[train_sac] loading BC actor weights from {args.bc_ckpt}", flush=True)
        trainer.load_bc_actor(Path(args.bc_ckpt))
    else:
        print(
            "[train_sac] no BC weight transfer — actor starts from SAC random init "
            "(Card C')",
            flush=True,
        )

    print(
        f"[train_sac] starting SACfD: total_steps={args.total_steps} "
        f"learning_starts={args.learning_starts} batch={args.batch_size} "
        f"bc_loss_enabled={args.bc_loss_enabled} "
        f"bc_w={args.bc_loss_weight} bc_anneal={args.bc_anneal_steps} "
        f"q_filter={args.q_filter} her={args.her_enabled} "
        f"demo_ratio={args.demo_ratio_initial}→{args.demo_ratio_final}"
        f"@{args.demo_ratio_switch_step}",
        flush=True,
    )
    trainer.train()

    out_path = out_root / "sac.pt"
    trainer.save(out_path)
    print(f"[train_sac] saved {out_path}", flush=True)

    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
