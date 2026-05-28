"""Eval residual policy: load checkpoint, run sim."""

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--bc_init", required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--n_trials", type=int, default=3)
parser.add_argument("--success_z", type=float, default=0.07)
parser.add_argument("--residual_scale", type=float, default=0.1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"] + sys.argv[1:])

launcher = AppLauncher(args)

import torch
import torch.nn as nn
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.modules import ActorCritic
from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import SoArm101CubeLiftEnvCfg


def build_bc_mlp(obs_dim, act_dim, hidden_dims=(256, 128, 64)):
    layers = []
    prev = obs_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ELU())
        prev = h
    layers.append(nn.Linear(prev, act_dim))
    return nn.Sequential(*layers)


class ResidualActor(nn.Module):
    def __init__(self, residual_mlp, bc_actor, scale=0.1):
        super().__init__()
        self.residual = residual_mlp
        self.bc = bc_actor
        self.scale = scale

    def forward(self, obs):
        bc_action = self.bc(obs).detach()
        residual = self.residual(obs)
        return (bc_action + residual * self.scale).clamp(-1.0, 1.0)


def main():
    env_cfg = SoArm101CubeLiftEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    device = env.unwrapped.device

    env.unwrapped.reset()
    obs, _ = env.get_observations()
    num_obs = obs.shape[1]
    num_actions = env.unwrapped.action_manager.total_action_dim

    # Build ActorCritic with ResidualActor wrapped
    ac = ActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        init_noise_std=0.3,
    ).to(device)

    # Build BC actor (same arch as during training)
    bc_blob = torch.load(args.bc_init, map_location="cpu", weights_only=False)
    arch = bc_blob["model_arch"]
    bc_actor = build_bc_mlp(arch["obs_dim"], arch["act_dim"], tuple(arch["hidden_dims"]))
    bc_actor.load_state_dict(bc_blob["actor_state_dict"])
    bc_actor.eval()
    bc_actor = bc_actor.to(device)

    # Wrap ActorCritic actor
    original_actor = ac.actor
    ac.actor = ResidualActor(original_actor, bc_actor, scale=args.residual_scale).to(device)

    # Load checkpoint (state_dict 에 actor.residual.* + actor.bc.* 둘 다 있어서 strict load OK)
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ac.load_state_dict(blob["model_state_dict"])
    ac.eval()

    print(f"[INFO] running {args.n_trials} trials × {args.num_envs} envs", flush=True)

    total_s, total_n = 0, 0
    z_max_all = []
    env_origins_z = env.unwrapped.scene.env_origins[:, 2]

    for trial in range(args.n_trials):
        env.unwrapped.reset()
        obs, _ = env.get_observations()
        z_max = torch.zeros(args.num_envs, device=device)
        for step in range(300):
            with torch.no_grad():
                action = ac.act_inference(obs)
                obs, _, _, _ = env.step(action)
                cube_z = env.unwrapped.scene["cube"].data.root_pos_w[:, 2] - env_origins_z
                z_max = torch.maximum(z_max, cube_z)
        s = int((z_max >= args.success_z).sum())
        total_s += s
        total_n += args.num_envs
        z_max_all.extend(z_max.cpu().numpy().tolist())
        print(f"trial {trial}: s={s}/{args.num_envs}  z_max_mean={float(z_max.mean()):.3f}", flush=True)

    rate = 100 * total_s / total_n
    print(f"", flush=True)
    print(f"[Residual eval] {args.checkpoint}", flush=True)
    print(f"[Residual eval] total: {total_s}/{total_n} ({rate:.1f}%)", flush=True)
    print(f"[Residual eval] z_max mean: {sum(z_max_all)/len(z_max_all):.3f}", flush=True)

    env.close()


main()
launcher.app.close()
