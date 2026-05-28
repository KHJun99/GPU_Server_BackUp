"""Evaluate trained PPO actor in sim."""

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="Path to model_*.pt")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--n_trials", type=int, default=3)
parser.add_argument("--success_z", type=float, default=0.07)
parser.add_argument("--deterministic", action="store_true", default=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(['--headless'] + sys.argv[1:])

launcher = AppLauncher(args)

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.modules import ActorCritic

from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import SoArm101CubeLiftEnvCfg


def main():
    # env
    env_cfg = SoArm101CubeLiftEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    device = env.unwrapped.device

    env.unwrapped.reset()
    obs, _ = env.get_observations()
    num_obs = obs.shape[1]
    num_actions = env.unwrapped.action_manager.total_action_dim

    print(f"[INFO] num_obs={num_obs}  num_actions={num_actions}", flush=True)

    # build ActorCritic (must match training cfg)
    actor_critic = ActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        init_noise_std=1.0,
    ).to(device)

    # load checkpoint
    print(f"[INFO] loading {args.checkpoint}", flush=True)
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    actor_critic.load_state_dict(blob["model_state_dict"])
    actor_critic.eval()

    print(f"[INFO] running {args.n_trials} trials × {args.num_envs} envs", flush=True)

    total_success = 0
    total_episodes = 0
    z_max_all = []

    env_origins_z = env.unwrapped.scene.env_origins[:, 2]

    for trial in range(args.n_trials):
        env.unwrapped.reset()
        obs, _ = env.get_observations()
        z_max_episode = torch.zeros(args.num_envs, device=device)

        for step in range(300):
            with torch.no_grad():
                # deterministic action (mean of distribution)
                action = actor_critic.act_inference(obs) if args.deterministic else actor_critic.act(obs)
                obs, _, _, _ = env.step(action)
                cube_z = env.unwrapped.scene["cube"].data.root_pos_w[:, 2] - env_origins_z
                z_max_episode = torch.maximum(z_max_episode, cube_z)

        success = (z_max_episode >= args.success_z).int()
        total_success += int(success.sum())
        total_episodes += args.num_envs
        z_max_np = z_max_episode.cpu().numpy()
        z_max_all.extend(z_max_np.tolist())
        print(f"trial {trial}: z_max={z_max_np.round(3)}  s={int(success.sum())}/{args.num_envs}", flush=True)

    rate = 100 * total_success / total_episodes
    print(f"\n[PPO eval] {args.checkpoint}", flush=True)
    print(f"[PPO eval] total: {total_success}/{total_episodes} ({rate:.1f}%)", flush=True)
    print(f"[PPO eval] z_max mean: {sum(z_max_all)/len(z_max_all):.3f}, max: {max(z_max_all):.3f}", flush=True)

    env.close()


main()
launcher.app.close()
