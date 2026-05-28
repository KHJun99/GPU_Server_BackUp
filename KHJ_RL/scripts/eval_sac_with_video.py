"""Eval a SAC actor checkpoint and save a rollout video.

Loads ``sac.pt`` (state_dict produced by SACfDTrainer.save), constructs
GaussianActor with matching arch, normalizes obs via the ckpt's
norm.json sidecar, then rolls 4 episodes and writes an MP4 via the
existing VideoSampler.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y CUDA_VISIBLE_DEVICES=1 \\
        python scripts/eval_sac_with_video.py \\
            --ckpt runs/g_plus_sacfd_L0_smoke/sac.pt \\
            --episodes 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import pinocchio  # noqa: F401, E402  (pre-AppLauncher Assimp ABI fix)

from isaaclab.app import AppLauncher  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <ckpt_dir>/videos_eval/")
    p.add_argument("--task-level", type=int, default=0)
    p.add_argument("--curriculum-stage", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--deterministic", action="store_true",
        help="Use actor.mean_action (clamp(mean)) instead of actor.sample. "
             "Default = stochastic to match SAC online rollout.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    sim_app = AppLauncher(headless=True, enable_cameras=True).app

    import numpy as np
    import torch

    from khj_rl.envs import CubeLiftEnv, CubeLiftEnvCfg
    from khj_rl.eval.video_sampler import VideoSampler
    from khj_rl.training.normalizer import ObsNormalizer
    from khj_rl.training.sac_network import GaussianActor

    cfg = CubeLiftEnvCfg()
    cfg.seed = args.seed
    cfg.curriculum.current_stage_idx = int(args.curriculum_stage)
    cfg.curriculum.task_level = int(args.task_level)
    env = CubeLiftEnv(cfg, include_viewer_camera=True)

    device = torch.device("cuda:0")
    ckpt = torch.load(args.ckpt, map_location=device)
    obs_dim = int(ckpt.get("obs_dim", cfg.obs_total_size))
    action_dim = int(ckpt.get("action_dim", cfg.action_size))
    hidden_dim = int(ckpt.get("hidden_dim", 256))

    actor = GaussianActor(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim
    ).to(device)
    # SACfDTrainer.save() stores actor under "actor" key.
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    normalizer = ObsNormalizer.load(args.ckpt.with_name("norm.json"))
    out_dir = args.out_dir or args.ckpt.parent / "videos_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval_video] ckpt={args.ckpt}", flush=True)
    print(f"[eval_video] out_dir={out_dir}", flush=True)
    print(f"[eval_video] episodes={args.episodes} deterministic={args.deterministic}", flush=True)

    def policy(obs_np: np.ndarray) -> np.ndarray:
        obs = torch.from_numpy(
            normalizer.normalize_np(np.asarray(obs_np, dtype=np.float32)[None, :])
        ).to(device)
        with torch.no_grad():
            if args.deterministic:
                action = actor.mean_action(obs)
            else:
                action, _ = actor.sample(obs)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)

    sampler = VideoSampler(
        output_dir=out_dir,
        every_steps=1,
        rollout_episodes=int(args.episodes),
        fps=20,
    )
    # Trigger a single capture: pass step=1 with _last_capture_step=-1.
    out_path = sampler.maybe_capture(step=1, env=env, policy=policy)
    if out_path is None:
        print("[eval_video] FAILED to write video (no viewer camera?)", flush=True)
        sim_app.close()
        return 1
    print(f"[eval_video] saved: {out_path}", flush=True)
    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
