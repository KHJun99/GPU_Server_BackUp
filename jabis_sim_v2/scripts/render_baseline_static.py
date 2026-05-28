"""Zero-action render — robot 이 baseline 자세 유지하는지 시각 확인용.

정책 inference 안 하고 zero action 으로 sim step. robot 은 actuator 가 default
(baseline) 유지하려고 함 → 영상 모든 frame 에서 baseline 자세 보임.
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--video_length", type=int, default=90)
parser.add_argument("--output_dir", type=str, default="videos/sim_match_check")
parser.add_argument("--tag", type=str, default="baseline_static")
parser.add_argument("--fps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import jabis_sim_v2  # noqa: F401, E402
from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import SoArm101CubeLiftEnvCfg_VIDEO  # noqa: E402


def grab(env, sensor_name: str) -> np.ndarray:
    rgb = env.unwrapped.scene.sensors[sensor_name].data.output["rgb"][0].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    return rgb


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)
    env_cfg = SoArm101CubeLiftEnvCfg_VIDEO()
    env_cfg.scene.num_envs = 1
    env = gym.make("Jabis-V2-CubeLift-Video-v0", cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    device = env.unwrapped.device

    env.unwrapped.reset()
    action_dim = env.unwrapped.action_manager.total_action_dim
    zero_action = torch.zeros(1, action_dim, device=device)

    robot = env.unwrapped.scene["robot"]
    print(f"[VIDEO_DBG] robot.joint_names = {robot.data.joint_names}", flush=True)
    print(f"[VIDEO_DBG] robot.default_joint_pos = "
          f"{[round(x, 4) for x in robot.data.default_joint_pos[0].cpu().tolist()]}",
          flush=True)
    print(f"[VIDEO_DBG] robot.joint_pos (after reset) = "
          f"{[round(x, 4) for x in robot.data.joint_pos[0].cpu().tolist()]}",
          flush=True)

    frames_top, frames_diag = [], []
    for step in range(args_cli.video_length):
        env.step(zero_action)
        frames_top.append(grab(env, "top_camera"))
        frames_diag.append(grab(env, "diag_camera"))
        if step % 30 == 0:
            jp = [round(x, 4) for x in robot.data.joint_pos[0].cpu().tolist()]
            print(f"  step {step}/{args_cli.video_length}  joint_pos={jp}", flush=True)

    for view, frames in [("top", frames_top), ("diag", frames_diag)]:
        out_path = os.path.join(args_cli.output_dir, f"{args_cli.tag}_{view}.mp4")
        imageio.mimsave(out_path, frames, fps=args_cli.fps)
        print(f"[DONE] {view} → {out_path}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
