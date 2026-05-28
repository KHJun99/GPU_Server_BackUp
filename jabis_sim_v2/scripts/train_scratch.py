"""Pure PPO from scratch — large exploration, long training, big batch."""

import argparse
import sys
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--max_iter", type=int, default=5000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--init_noise_std", type=float, default=1.0)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--log_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"] + sys.argv[1:])

launcher = AppLauncher(args)

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from rsl_rl.runners import OnPolicyRunner

from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import SoArm101CubeLiftEnvCfg


def main():
    env_cfg = SoArm101CubeLiftEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    if args.log_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = f"logs/rsl_rl/scratch_{ts}"
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"[INFO] log_dir: {args.log_dir}", flush=True)
    print(f"[INFO] num_envs={args.num_envs}, max_iter={args.max_iter}", flush=True)
    print(f"[INFO] init_noise_std={args.init_noise_std}, lr={args.learning_rate}", flush=True)

    agent_cfg = RslRlOnPolicyRunnerCfg(
        seed=args.seed,
        device="cuda:0",
        num_steps_per_env=24,
        max_iterations=args.max_iter,
        save_interval=args.save_interval,
        experiment_name="v2_scratch",
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
            num_learning_epochs=5,
            num_mini_batches=16,  # mini_batch ~1500 for num_envs 1024
            learning_rate=args.learning_rate,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=args.log_dir, device=agent_cfg.device)

    print(f"[INFO] starting training: {args.max_iter} iterations", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


main()
launcher.app.close()
