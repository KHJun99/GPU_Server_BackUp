"""Capture rollout videos for a trained checkpoint.

Used to inspect what the policy actually *does* (reward hacking check —
the 97.9% / 0% / dense-shaping cube-crush failures all looked fine in
numbers; the videos exposed them). Builds the env with the third-person
viewer camera, runs N episodes through the same ChunkBuffer + ACT path
as ``eval_policy.py``, and writes one MP4 per episode at policy rate.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \\
        python scripts/rollout_video.py \\
            --ckpt runs/stage0_bc_act_k12/bc.pt \\
            --episodes 6 \\
            --out-dir runs/stage0_bc_act_k12/eval_videos
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Match eval_policy.py bootstrap (pinocchio + CUDA env vars before kit).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("KHJ_RL_SIM_DEVICE", "cuda:0")

import pinocchio  # noqa: F401, E402

from isaaclab.app import AppLauncher  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Defaults to <ckpt parent>/eval_videos",
    )
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--curriculum-stage", type=int, default=0,
        help="Index into CurriculumCfg.side_length_m (0=5cm, 1=10cm, 2=15cm, "
             "3=20cm). Matches eval_policy.py so the video sample comes from "
             "the same distribution as the corresponding eval run.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    # Enable the renderer — viewer camera needs it.
    sim_app = AppLauncher(headless=True, enable_cameras=True).app

    import imageio.v2 as imageio
    import numpy as np
    import torch

    from khj_rl.envs import CubeLiftEnv, CubeLiftEnvCfg
    from khj_rl.eval.obs_alignment import assert_obs_alignment
    from khj_rl.training.network import ActorCritic, ChunkedActorCritic
    from khj_rl.training.normalizer import ObsNormalizer

    out_dir = args.out_dir or args.ckpt.parent / "eval_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CubeLiftEnvCfg()
    cfg.seed = args.seed
    cfg.curriculum.current_stage_idx = int(args.curriculum_stage)
    env = CubeLiftEnv(cfg, include_viewer_camera=True)
    assert_obs_alignment(env, cfg)

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    obs_dim = int(ckpt.get("obs_dim", cfg.obs_total_size))
    action_dim = int(ckpt.get("action_dim", cfg.action_size))
    hidden_dim = int(ckpt.get("hidden_dim", 256))
    actor_logstd_init = float(ckpt.get("actor_logstd_init", -1.0))
    chunking_enabled = bool(ckpt.get("chunking_enabled", False))
    chunk_size = int(ckpt.get("chunk_size", 0))
    inference_alpha = float(ckpt.get("inference_alpha", 0.2))
    gripper_use_latest = bool(ckpt.get("gripper_use_latest", True))

    if chunking_enabled:
        net = ChunkedActorCritic(
            obs_dim=obs_dim, action_dim=action_dim, chunk_size=chunk_size,
            hidden_dim=hidden_dim, actor_logstd_init=actor_logstd_init,
        ).to(device)
    else:
        net = ActorCritic(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim,
            actor_logstd_init=actor_logstd_init,
        ).to(device)
    net.load_state_dict(ckpt["actor_critic"])
    net.eval()
    normalizer = ObsNormalizer.load(args.ckpt.with_name("norm.json"))

    print(
        f"[rollout_video] ckpt={args.ckpt} episodes={args.episodes} "
        f"chunking={chunking_enabled} k={chunk_size} alpha={inference_alpha}",
        flush=True,
    )

    class ChunkBuffer:
        """Same temporal-ensemble logic as eval_policy.py — kept inline so
        the two scripts diverge only when needed."""

        def __init__(self, k, alpha, gripper_latest):
            self.k, self.alpha, self.gripper_latest = int(k), float(alpha), bool(gripper_latest)
            self._buf: list[tuple[int, np.ndarray]] = []

        def reset(self):
            self._buf.clear()

        def push(self, t, chunk):
            self._buf.append((int(t), np.asarray(chunk, dtype=np.float32)))
            self._buf = [(o, c) for (o, c) in self._buf if t - o < self.k]

        def get_action(self, t):
            contribs = []
            latest_origin = -1
            latest_action = None
            for origin, chunk in self._buf:
                offset = t - origin
                if 0 <= offset < self.k:
                    w = float(np.exp(-self.alpha * offset))
                    contribs.append((w, chunk[offset]))
                    if origin > latest_origin:
                        latest_origin = origin
                        latest_action = chunk[offset]
            total_w = sum(w for w, _ in contribs)
            action = sum(w * a for w, a in contribs) / total_w
            if self.gripper_latest and latest_action is not None:
                action = action.copy()
                action[-1] = latest_action[-1]
            return action.astype(np.float32, copy=False)

    chunk_buf = (
        ChunkBuffer(chunk_size, inference_alpha, gripper_use_latest)
        if chunking_enabled else None
    )

    def policy(obs_np, t):
        obs_t = torch.from_numpy(
            normalizer.normalize_np(np.asarray(obs_np, dtype=np.float32))
        ).to(device).unsqueeze(0)
        with torch.no_grad():
            if chunking_enabled:
                chunk = net.actor_chunks(obs_t).squeeze(0).cpu().numpy().astype(np.float32)
                chunk_buf.push(t, chunk)
                return chunk_buf.get_action(t)
            return net.actor_mean(obs_t).squeeze(0).cpu().numpy().astype(np.float32)

    def read_rgb():
        # Mirror video_sampler._read_camera_rgb robustness (some frames
        # come back empty / float / RGBA on the first tick after reset).
        cam = env._viewer_camera
        data = getattr(cam, "data", None)
        if data is None:
            return None
        out = getattr(data, "output", None)
        if out is None or "rgb" not in out:
            return None
        rgb = out["rgb"]
        try:
            tensor = rgb[0]
        except Exception:
            return None
        arr = tensor.detach().cpu().numpy()
        if arr.ndim != 3:
            return None
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        return arr

    summary: list[tuple[int, bool, dict]] = []
    for ep_id in range(args.episodes):
        obs = env.reset()
        if chunk_buf is not None:
            chunk_buf.reset()
        frames: list[np.ndarray] = []
        per: dict = {}
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy(obs, steps)
            obs, _r, terminated, truncated, info = env.step(action)
            steps += 1
            per = info.get("success_per_condition", per)
            rgb = read_rgb()
            if rgb is not None:
                frames.append(rgb)

        success = bool(terminated)
        tag = "succ" if success else "fail"
        stem = f"ep{ep_id:03d}_{tag}_steps{steps}"
        out_path = out_dir / f"{stem}.mp4"
        thumb_dir = out_dir / "thumbnails"
        if frames:
            imageio.mimsave(out_path, frames, fps=args.fps, macro_block_size=1)
            # Codex review (overnight): keep thumbnail extraction in the same
            # script so the operator never has to chase a separate analysis
            # pass. first / middle / last gives enough trajectory shape to
            # spot reward-hacking attractors at a glance.
            thumb_dir.mkdir(parents=True, exist_ok=True)
            n = len(frames)
            for label, idx in (("first", 0), ("middle", n // 2), ("last", n - 1)):
                imageio.imwrite(thumb_dir / f"{stem}_{label}.png", frames[idx])
        print(
            f"[rollout_video] ep={ep_id:03d} steps={steps} success={success} "
            f"per_cond={ {k: int(bool(v)) for k, v in per.items()} } "
            f"-> {out_path.name}",
            flush=True,
        )
        summary.append((ep_id, success, per))

    n_succ = sum(1 for _, ok, _ in summary if ok)
    print(
        f"\n[rollout_video] {n_succ}/{args.episodes} success in this sample. "
        f"Videos -> {out_dir}",
        flush=True,
    )
    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
