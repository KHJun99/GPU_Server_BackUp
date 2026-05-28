"""Render PPO policy to mp4 — v1 pattern (TiledCamera + imageio, no RecordVideo).

Avoids gym.wrappers.RecordVideo which triggers omni.kit.widget.viewport hydra
engine init that hangs on headless jupyter07. Instead, grabs frames directly
from TiledCamera sensors and writes mp4 via imageio.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--output_dir", type=str, default="videos/ppo_99")
parser.add_argument("--tag", type=str, default="ppo99")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--views", type=str, default="top,diag")
parser.add_argument(
    "--task",
    type=str,
    default="lift",
    choices=["lift", "pnp"],
    help="lift = Jabis-V2-CubeLift-Video-v0; pnp = Jabis-V2-CubePickPlace-Video-v0",
)
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

from rsl_rl.modules import ActorCritic  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import jabis_sim_v2  # noqa: F401, E402  (registers tasks)
from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import (  # noqa: E402
    SoArm101CubeLiftEnvCfg_VIDEO,
    SoArm101PickPlaceEnvCfg_VIDEO,
)

TASK_REGISTRY = {
    "lift": ("Jabis-V2-CubeLift-Video-v0", SoArm101CubeLiftEnvCfg_VIDEO),
    "pnp": ("Jabis-V2-CubePickPlace-Video-v0", SoArm101PickPlaceEnvCfg_VIDEO),
}


def grab_rgb(env, sensor_name: str) -> np.ndarray:
    sensors = env.unwrapped.scene.sensors
    if sensor_name not in sensors:
        raise KeyError(f"sensor '{sensor_name}' not in scene.sensors "
                       f"(available: {list(sensors.keys())})")
    rgb = sensors[sensor_name].data.output["rgb"][0].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8) if rgb.max() <= 1.0 \
              else rgb.astype(np.uint8)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    return rgb


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)
    view_names = [v.strip() for v in args_cli.views.split(",") if v.strip()]

    task_id, env_cfg_cls = TASK_REGISTRY[args_cli.task]
    env_cfg = env_cfg_cls()
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = 1

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    device = env.unwrapped.device

    obs, _ = env.get_observations()
    print(f"[INFO] obs.shape={tuple(obs.shape)}  device={device}", flush=True)

    ac = ActorCritic(
        num_actor_obs=obs.shape[1],
        num_critic_obs=obs.shape[1],
        num_actions=env.unwrapped.action_manager.total_action_dim,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        init_noise_std=1.0,
    ).to(device)

    blob = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    ac.load_state_dict(blob["model_state_dict"])
    ac.eval()
    print(f"[INFO] loaded ckpt iter={blob.get('iter', '?')}  → {args_cli.checkpoint}", flush=True)

    env.unwrapped.seed(args_cli.seed)
    env.unwrapped.reset()
    obs, _ = env.get_observations()

    frames_by_view = {v: [] for v in view_names}
    for step in range(args_cli.video_length):
        with torch.no_grad():
            action = ac.act_inference(obs)
        obs, _, _, _ = env.step(action)

        for v in view_names:
            frames_by_view[v].append(grab_rgb(env, f"{v}_camera"))

        if step % 50 == 0:
            print(f"  step {step}/{args_cli.video_length}", flush=True)

    for v, fr in frames_by_view.items():
        out_path = os.path.join(
            args_cli.output_dir, f"{args_cli.tag}_{v}_seed{args_cli.seed}.mp4"
        )
        imageio.mimsave(out_path, fr, fps=args_cli.fps)
        print(f"[DONE] view={v} → {out_path}  frames={len(fr)}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
