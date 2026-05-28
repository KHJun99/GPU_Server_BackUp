"""PPO training for v2 cube lift, with optional BC warmstart.

Usage:
  # PPO from scratch
  python scripts/train.py --num_envs 64 --max_iter 500

  # PPO with BC warmstart
  python scripts/train.py --num_envs 64 --max_iter 500 --bc_init tasks/bc_init_v2.pt
"""

import argparse
import sys
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iter", type=int, default=500)
parser.add_argument("--bc_init", type=str, default=None, help="Path to bc_init.pt")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log_dir", type=str, default=None,
                    help="Log dir (default: logs/rsl_rl/v2_<timestamp>)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(['--headless'] + sys.argv[1:])

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

from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import SoArm101CubeLiftEnvCfg


def make_agent_cfg(seed, max_iter):
    """Build PPO agent cfg (v1 lift settings)."""
    cfg = RslRlOnPolicyRunnerCfg(
        seed=seed,
        device="cuda:0",
        num_steps_per_env=24,
        max_iterations=max_iter,
        save_interval=50,
        experiment_name="v2_lift",
        empirical_normalization=False,
        clip_actions=1.0,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=0.1,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=16,
            learning_rate=1.0e-5,
            schedule="adaptive",
            gamma=0.98,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )
    return cfg


def main():
    # env
    env_cfg = SoArm101CubeLiftEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    # log dir
    if args.log_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = f"logs/rsl_rl/v2_{ts}"
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"[INFO] log_dir: {args.log_dir}", flush=True)

    # agent
    agent_cfg = make_agent_cfg(args.seed, args.max_iter)

    # runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=args.log_dir, device=agent_cfg.device)

    # BC warmstart
    if args.bc_init:
        print(f"[INFO] BC init from: {args.bc_init}", flush=True)
        bc = torch.load(args.bc_init, map_location="cpu", weights_only=False)
        actor_sd = bc.get("actor_state_dict", bc)
        target = runner.alg.policy.actor
        missing, unexpected = target.load_state_dict(actor_sd, strict=False)
        print(f"[BC init] loaded {len(actor_sd)} params; missing={list(missing)}; unexpected={list(unexpected)}", flush=True)
        if missing or unexpected:
            print("[BC init] WARNING: state_dict keys mismatch — partial init", flush=True)

    # train
    print(f"[INFO] starting training: {args.max_iter} iterations", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


main()
launcher.app.close()
