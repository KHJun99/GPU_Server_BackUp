"""Evaluate trained BC actor in sim (no PPO yet)."""

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--bc_init", required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--n_trials", type=int, default=3)
parser.add_argument("--success_z", type=float, default=0.07)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(['--headless'] + sys.argv[1:])

launcher = AppLauncher(args)

import torch
from isaaclab.envs import ManagerBasedRLEnv
from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import SoArm101CubeLiftEnvCfg

# BC actor 정의 (train_bc.py 와 동일)
import torch.nn as nn

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
    # load BC
    blob = torch.load(args.bc_init, map_location="cpu", weights_only=False)
    arch = blob["model_arch"]
    print(f"[INFO] BC arch: {arch}", flush=True)
    actor = BCActor(arch["obs_dim"], arch["act_dim"], tuple(arch["hidden_dims"]))
    actor.net.load_state_dict(blob["actor_state_dict"])
    actor.eval()
    obs_mean = blob.get("obs_mean")
    obs_std = blob.get("obs_std")
    if obs_mean is not None:
        print(f"[norm] applying normalization (mean shape {obs_mean.shape})", flush=True)

    # env
    cfg = SoArm101CubeLiftEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=cfg)
    device = env.device
    actor = actor.to(device)

    print(f"[INFO] running {args.n_trials} trials × {args.num_envs} envs", flush=True)

    total_success = 0
    total_episodes = 0
    z_max_all = []

    for trial in range(args.n_trials):
        obs_dict, _ = env.reset()
        obs_tensor = obs_dict["policy"]

        z_max_episode = torch.zeros(args.num_envs, device=device)
        env_origins_z = env.scene.env_origins[:, 2]

        for step in range(300):  # 1 episode
            with torch.no_grad():
                obs_in = obs_tensor
                if obs_mean is not None:
                    obs_in = (obs_tensor - obs_mean.to(device)) / obs_std.to(device)
                action = actor(obs_in).clamp(-1.0, 1.0)
                obs_dict, _, terminated, truncated, _ = env.step(action)
                obs_tensor = obs_dict["policy"]
                cube_z = env.scene["cube"].data.root_pos_w[:, 2] - env_origins_z
                z_max_episode = torch.maximum(z_max_episode, cube_z)

        success = (z_max_episode >= args.success_z).int()
        total_success += int(success.sum())
        total_episodes += args.num_envs
        z_max_np = z_max_episode.cpu().numpy()
        z_max_all.extend(z_max_np.tolist())
        print(f"trial {trial}: z_max={z_max_np.round(3)}  s={int(success.sum())}/{args.num_envs}", flush=True)

    rate = 100 * total_success / total_episodes
    print(f"\n[BC eval] total: {total_success}/{total_episodes} ({rate:.1f}%)", flush=True)
    print(f"[BC eval] z_max mean: {sum(z_max_all)/len(z_max_all):.3f}, max: {max(z_max_all):.3f}", flush=True)

    env.close()


main()
launcher.app.close()
