"""Evaluate BC policy on PnP task (in_bin metric)."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--bc_init", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--cube_x", type=float, default=0.16)
parser.add_argument("--cube_y", type=float, default=0.0)
parser.add_argument("--randomize_spawn", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"] + sys.argv[1:])

launcher = AppLauncher(args)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import (  # noqa: E402
    SoArm101PickPlaceEnvCfg,
)
from jabis_sim_v2.tasks.cube_lift.mdp import events as cube_events  # noqa: E402


BIN_X_MIN, BIN_X_MAX = -0.045, 0.105
BIN_Y_MIN, BIN_Y_MAX = 0.18, 0.31


class BCActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 128, 64)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU())
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


def main():
    cfg = SoArm101PickPlaceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = 100.0
    if not args.randomize_spawn:
        cfg.events.reset_cube = EventTerm(
            func=cube_events.reset_cube_grid,
            mode="reset",
            params={
                "x_vals": [args.cube_x],
                "y_vals": [args.cube_y],
                "envs_per_point": args.num_envs,
            },
        )

    env_raw = ManagerBasedRLEnv(cfg=cfg)
    env = RslRlVecEnvWrapper(env_raw, clip_actions=1.0)
    device = env.unwrapped.device

    blob = torch.load(args.bc_init, map_location=device, weights_only=False)
    arch = blob["model_arch"]
    actor = BCActor(
        arch["obs_dim"], arch["act_dim"], tuple(arch["hidden_dims"])
    ).to(device)
    actor.net.load_state_dict(blob["actor_state_dict"])
    actor.eval()
    obs_mean = blob["obs_mean"].to(device)
    obs_std = blob["obs_std"].to(device)

    print(
        f"[BC eval] num_envs={args.num_envs}, steps={args.steps}, "
        f"cube=({args.cube_x:.3f},{args.cube_y:+.3f}), randomize={args.randomize_spawn}",
        flush=True,
    )

    env_origins = env_raw.scene.env_origins
    cube = env_raw.scene["cube"]

    env_raw.reset()
    obs, _ = env.get_observations()

    with torch.no_grad():
        for step in range(args.steps):
            obs_norm = (obs - obs_mean) / obs_std
            action = actor(obs_norm).clamp(-1.0, 1.0)
            obs, _, _, _ = env.step(action)
            if step % 200 == 0:
                cl = cube.data.root_pos_w - env_origins
                print(
                    f"  step {step}/{args.steps}  cube_z med={cl[:, 2].median():.3f}",
                    flush=True,
                )

    cl_final = cube.data.root_pos_w - env_origins
    fx, fy, fz = cl_final[:, 0], cl_final[:, 1], cl_final[:, 2]
    in_bin_xy = (
        (fx >= BIN_X_MIN) & (fx <= BIN_X_MAX) & (fy >= BIN_Y_MIN) & (fy <= BIN_Y_MAX)
    )
    in_bin = in_bin_xy & (fz < 0)

    print(f"\n[BC eval] === Result ===", flush=True)
    print(
        f"  cube xyz median: ({fx.median():.3f}, {fy.median():+.3f}, {fz.median():.3f})",
        flush=True,
    )
    print(
        f"  in_bin (xy + z<0): {int(in_bin.sum())}/{args.num_envs} "
        f"({100 * in_bin.float().mean():.1f}%)",
        flush=True,
    )
    print(
        f"  in_bin_xy:         {int(in_bin_xy.sum())}/{args.num_envs} "
        f"({100 * in_bin_xy.float().mean():.1f}%)",
        flush=True,
    )

    env.close()


main()
launcher.app.close()
