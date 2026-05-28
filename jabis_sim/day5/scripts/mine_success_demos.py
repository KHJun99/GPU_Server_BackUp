"""Mine successful (obs, action) trajectories from a trained PPO checkpoint.

Runs the given rsl_rl checkpoint in the SO-ARM101 cube-lift Play env, tracks per-env
cube z, and saves trajectories where cube_z_max >= --success_z. These trajectories
are intended to seed BC warmstart for v7.

Pattern adapted from src/isaac_so_arm101/scripts/rsl_rl/play.py.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Mine successful demos from a trained PPO checkpoint.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel envs.")
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Lift-Cube-Play-v0", help="Task ID.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point."
)
parser.add_argument("--target_episodes", type=int, default=200, help="Stop after this many successes.")
parser.add_argument(
    "--max_total_episodes",
    type=int,
    default=10000,
    help="Hard ceiling on episodes attempted (per env_idx-wise reset, summed).",
)
parser.add_argument("--success_z", type=float, default=0.04, help="cube z_max threshold for success.")
parser.add_argument("--output", type=str, required=True, help="Where to save the .pt demo file.")
parser.add_argument(
    "--stochastic",
    action="store_true",
    default=False,
    help="Sample actions from policy distribution (more diverse but lower success rate).",
)
parser.add_argument("--seed", type=int, default=None, help="Env seed.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric (uses USD I/O)."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

if not args_cli.checkpoint:
    raise SystemExit("--checkpoint is required (path to model_*.pt).")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic
    policy_nn.eval()

    if args_cli.stochastic:
        def act(obs):
            return policy_nn.act(obs)
    else:
        def act(obs):
            return policy_nn.act_inference(obs)

    device = env.unwrapped.device
    n_envs = env.unwrapped.num_envs

    obs = env.get_observations()  # TensorDict, kept as-is for policy_nn
    obs_tensor = obs["policy"]  # actual (n_envs, obs_dim) tensor for buffering
    obs_dim = obs_tensor.shape[1]
    print(f"[INFO] obs_dim={obs_dim}, n_envs={n_envs}")
    print(f"[INFO] success_z={args_cli.success_z}, target={args_cli.target_episodes}, stochastic={args_cli.stochastic}")

    episode_obs = [[] for _ in range(n_envs)]
    episode_act = [[] for _ in range(n_envs)]
    episode_z_max = torch.zeros(n_envs, device=device)

    demos_obs: list[torch.Tensor] = []
    demos_act: list[torch.Tensor] = []
    demos_meta: list[dict] = []
    successes = 0
    finished_episodes = 0
    start_time = time.time()

    object_asset = env.unwrapped.scene["object"]
    env_origins_z = env.unwrapped.scene.env_origins[:, 2]

    print("[INFO] Mining...")
    while successes < args_cli.target_episodes and finished_episodes < args_cli.max_total_episodes:
        with torch.inference_mode():
            actions = act(obs)
            cube_z_local = (object_asset.data.root_pos_w[:, 2] - env_origins_z).detach().clone()

        for i in range(n_envs):
            episode_obs[i].append(obs_tensor[i].detach().cpu().clone())
            episode_act[i].append(actions[i].detach().cpu().clone())
        episode_z_max = torch.maximum(episode_z_max, cube_z_local)

        with torch.inference_mode():
            obs, _, dones, _ = env.step(actions)
            obs_tensor = obs["policy"]
            done_indices = torch.nonzero(dones, as_tuple=False).squeeze(-1).tolist()

        for env_idx in done_indices:
            finished_episodes += 1
            z_max = float(episode_z_max[env_idx])
            ep_len = len(episode_obs[env_idx])
            if z_max >= args_cli.success_z and ep_len > 0:
                demos_obs.append(torch.stack(episode_obs[env_idx]))
                demos_act.append(torch.stack(episode_act[env_idx]))
                demos_meta.append({"length": ep_len, "z_max": z_max, "env_idx": env_idx})
                successes += 1
                if successes % 20 == 0 or successes == args_cli.target_episodes:
                    elapsed = time.time() - start_time
                    rate = successes / max(finished_episodes, 1) * 100.0
                    print(
                        f"[INFO] successes={successes}/{args_cli.target_episodes} "
                        f"finished={finished_episodes} rate={rate:.1f}% z_max(last)={z_max:.3f} "
                        f"len={ep_len} elapsed={elapsed:.0f}s"
                    )
            episode_obs[env_idx] = []
            episode_act[env_idx] = []
            episode_z_max[env_idx] = 0.0

    elapsed = time.time() - start_time
    rate = successes / max(finished_episodes, 1) * 100.0
    print(
        f"\n[DONE] successes={successes} finished_episodes={finished_episodes} "
        f"rate={rate:.2f}% elapsed={elapsed:.0f}s"
    )

    if successes == 0:
        print("[WARN] no successes — saving empty file is skipped. Try --stochastic or another checkpoint.")
    else:
        os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
        torch.save(
            {
                "obs": demos_obs,
                "act": demos_act,
                "meta": demos_meta,
                "checkpoint": resume_path,
                "task": args_cli.task,
                "obs_dim": obs_dim,
                "act_dim": int(demos_act[0].shape[1]),
                "success_z": args_cli.success_z,
                "stochastic": args_cli.stochastic,
                "n_total_episodes_attempted": finished_episodes,
            },
            args_cli.output,
        )
        print(f"[INFO] saved {successes} demo trajectories to {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
