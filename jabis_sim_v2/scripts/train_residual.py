"""Residual PPO training: BC actor frozen + residual policy learned.

action = clamp(BC(obs) + scale * residual(obs), -1, 1)

Critic learned from scratch.
"""

import argparse
import sys
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iter", type=int, default=500)
parser.add_argument("--bc_init", type=str, required=True)
parser.add_argument("--residual_scale", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"] + sys.argv[1:])

launcher = AppLauncher(args)

import torch
import torch.nn as nn
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from rsl_rl.runners import OnPolicyRunner

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
    """BC (frozen) + Residual MLP."""

    def __init__(self, residual_mlp, bc_actor, scale=0.1):
        super().__init__()
        self.residual = residual_mlp
        self.bc = bc_actor
        self.scale = scale

    def forward(self, obs):
        bc_action = self.bc(obs).detach()
        residual = self.residual(obs)
        return (bc_action + residual * self.scale).clamp(-1.0, 1.0)


def make_agent_cfg(seed, max_iter):
    return RslRlOnPolicyRunnerCfg(
        seed=seed,
        device="cuda:0",
        num_steps_per_env=24,
        max_iterations=max_iter,
        save_interval=50,
        experiment_name="v2_residual",
        empirical_normalization=False,
        clip_actions=1.0,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=0.3,
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
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.98,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )


def main():
    env_cfg = SoArm101CubeLiftEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    if args.log_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = f"logs/rsl_rl/residual_{ts}"
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"[INFO] log_dir: {args.log_dir}", flush=True)

    agent_cfg = make_agent_cfg(args.seed, args.max_iter)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=args.log_dir, device=agent_cfg.device)

    print(f"[INFO] loading BC from: {args.bc_init}", flush=True)
    bc_blob = torch.load(args.bc_init, map_location="cpu", weights_only=False)
    arch = bc_blob["model_arch"]
    bc_actor = build_bc_mlp(arch["obs_dim"], arch["act_dim"], tuple(arch["hidden_dims"]))
    bc_actor.load_state_dict(bc_blob["actor_state_dict"])
    bc_actor.eval()
    for p in bc_actor.parameters():
        p.requires_grad = False
    bc_actor = bc_actor.to(agent_cfg.device)
    print("[INFO] BC actor loaded, frozen", flush=True)

    original_actor = runner.alg.policy.actor
    residual_actor = ResidualActor(original_actor, bc_actor, scale=args.residual_scale)
    runner.alg.policy.actor = residual_actor
    print(f"[INFO] Residual actor active, scale={args.residual_scale}", flush=True)
    n_trainable = sum(p.numel() for p in runner.alg.policy.parameters() if p.requires_grad)
    print(f"[INFO] Trainable params: {n_trainable}", flush=True)

    print(f"[INFO] starting training: {args.max_iter} iterations", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


main()
launcher.app.close()
