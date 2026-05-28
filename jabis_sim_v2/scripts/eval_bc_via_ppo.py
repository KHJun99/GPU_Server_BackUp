"""Eval BC weights via PPO ActorCritic wrapper (no training)."""
import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--bc_init", required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--n_trials", type=int, default=2)
parser.add_argument("--success_z", type=float, default=0.07)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(['--headless'] + sys.argv[1:])

launcher = AppLauncher(args)

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.modules import ActorCritic
from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import SoArm101CubeLiftEnvCfg


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

    ac = ActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        init_noise_std=0.3,
    ).to(device)

    bc = torch.load(args.bc_init, map_location=device, weights_only=False)
    missing, unexpected = ac.actor.load_state_dict(bc["actor_state_dict"], strict=False)
    print(f"[INFO] BC loaded: missing={missing}, unexpected={unexpected}", flush=True)
    ac.eval()

    env_origins_z = env.unwrapped.scene.env_origins[:, 2]

    total = 0
    succ = 0
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
        succ += s
        total += args.num_envs
        print(f"trial {trial}: s={s}/{args.num_envs}  z_max_mean={float(z_max.mean()):.3f}  z_max_max={float(z_max.max()):.3f}", flush=True)

    print(f"\n[BC via PPO] total: {succ}/{total} ({100*succ/total:.1f}%)", flush=True)
    env.close()


main()
launcher.app.close()
