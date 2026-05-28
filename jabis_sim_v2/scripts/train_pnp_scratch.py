"""PnP PPO scratch (no warmstart) — clean sim setup, multi-waypoint reward."""

import argparse
import sys
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iter", type=int, default=20000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--init_noise_std", type=float, default=1.0)
parser.add_argument("--learning_rate", type=float, default=5e-4)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--num_steps_per_env", type=int, default=8)
parser.add_argument("--num_learning_epochs", type=int, default=10)
parser.add_argument("--num_mini_batches", type=int, default=16)
parser.add_argument("--bc_warmstart", type=str, default=None,
                    help="BC bc_init.pt path (optional). actor 만 load.")
parser.add_argument("--log_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"] + sys.argv[1:])

launcher = AppLauncher(args)

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from rsl_rl.runners import OnPolicyRunner

from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import SoArm101PickPlaceEnvCfg


def main():
    env_cfg = SoArm101PickPlaceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    if args.log_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = f"logs/rsl_rl/pnp_scratch_{ts}"
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"[INFO] log_dir: {args.log_dir}", flush=True)
    print(f"[INFO] num_envs={args.num_envs}, max_iter={args.max_iter}", flush=True)
    print(f"[INFO] lr={args.learning_rate}", flush=True)
    if args.bc_warmstart:
        print(f"[INFO] BC warmstart: {args.bc_warmstart}", flush=True)
    else:
        print(f"[INFO] PPO scratch (no warmstart)", flush=True)

    agent_cfg = RslRlOnPolicyRunnerCfg(
        seed=args.seed,
        device="cuda:0",
        num_steps_per_env=args.num_steps_per_env,
        max_iterations=args.max_iter,
        save_interval=args.save_interval,
        experiment_name="v2_pnp_scratch",
        empirical_normalization=False,
        clip_actions=1.0,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=args.init_noise_std,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=args.entropy_coef,
            num_learning_epochs=args.num_learning_epochs,
            num_mini_batches=args.num_mini_batches,
            learning_rate=args.learning_rate,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=args.log_dir, device=agent_cfg.device)

    if args.bc_warmstart:
        print(f"[INFO] loading BC warmstart...", flush=True)
        blob = torch.load(args.bc_warmstart, map_location=agent_cfg.device, weights_only=False)
        # BC checkpoint: actor_state_dict (BCActor MLP keys '0.weight' ...)
        actor_sd = blob["actor_state_dict"]
        runner.alg.policy.actor.load_state_dict(actor_sd)
        print(f"[INFO] BC actor loaded (critic + std fresh)", flush=True)

    print(f"[INFO] starting PnP PPO training: {args.max_iter} iterations", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


main()
launcher.app.close()
