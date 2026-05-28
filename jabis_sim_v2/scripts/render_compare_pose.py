"""Render single frame at user-specified pose. baseline vs learning-default 비교용.

특정 joint_pos 를 직접 write_joint_state_to_sim 로 강제 적용 후 1 frame capture.
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--pose", type=str, required=True, choices=["baseline", "learning_default", "extreme"])
parser.add_argument("--tag", type=str, default="pose")
parser.add_argument("--output_dir", type=str, default="/home/j-k14d101/jabis_sim_v2/videos/sim_match_check")
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


POSES = {
    "baseline": [1.6157, -1.74, 1.685, 1.2413, 0.0499, 0.5258, 0.0, 0.0, 0.0, 0.0],
    "learning_default": [0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "extreme": [1.5708, 0.0, 1.5708, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


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
    robot = env.unwrapped.scene["robot"]
    mirror = env.unwrapped.scene["mirror_arm"]

    # 강제 joint pos write
    target_pos = torch.tensor(POSES[args_cli.pose], device=device, dtype=torch.float32).unsqueeze(0)
    target_vel = torch.zeros_like(target_pos)
    robot.write_joint_state_to_sim(target_pos, target_vel)
    mirror.write_joint_state_to_sim(target_pos, target_vel)
    print(f"[FORCED] robot.joint_pos = {[round(x, 4) for x in robot.data.joint_pos[0].cpu().tolist()]}",
          flush=True)

    # 카메라 update 위해 1 step (action 무시)
    action_dim = env.unwrapped.action_manager.total_action_dim
    zero_action = torch.zeros(1, action_dim, device=device)
    for _ in range(3):
        env.step(zero_action)
    print(f"[AFTER STEP] robot.joint_pos = {[round(x, 4) for x in robot.data.joint_pos[0].cpu().tolist()]}",
          flush=True)

    for view in ("top_camera", "diag_camera"):
        img = grab(env, view)
        out = os.path.join(args_cli.output_dir, f"{args_cli.tag}_{args_cli.pose}_{view.split('_')[0]}.png")
        imageio.imwrite(out, img)
        print(f"[DONE] {out}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
