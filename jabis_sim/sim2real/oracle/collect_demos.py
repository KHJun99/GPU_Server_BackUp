"""Collect oracle demonstrations for SO-ARM101 cube lift.

Runs ``OraclePolicy`` inside Isaac-SO-ARM101-Lift-Cube-Play-v0, buffers per-env
trajectories, and saves successful ones (``cube_z_max >= success_z``).

Output format mirrors mine_success_demos.py for downstream ingestion:
    {
        "obs":  list[Tensor(T_i, obs_dim)],
        "act":  list[Tensor(T_i, act_dim)],
        "meta": list[dict(length, z_max, env_idx, final_state)],
        "task": str,
        "obs_dim": int, "act_dim": int,
        "success_z": float,
        "source": "oracle_v1",
        "n_total_episodes_attempted": int,
        "saturation_per_joint_pct": list[float],
        "params": dict,
    }
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect oracle demos via DifferentialIK.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Lift-Cube-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--target_episodes", type=int, default=100)
parser.add_argument("--max_total_episodes", type=int, default=2000)
parser.add_argument("--success_z", type=float, default=0.10)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)

# Oracle params (mirrors OraclePolicy defaults; tunable for debugging)
parser.add_argument("--scale", type=float, default=1.5,
                    help="JointPositionAction scale used in env config.")
parser.add_argument("--reach_above_dz", type=float, default=0.10)
parser.add_argument("--descend_dz", type=float, default=0.005)
parser.add_argument("--lift_dz", type=float, default=0.10)
parser.add_argument("--reach_dist", type=float, default=0.02)
parser.add_argument("--descend_z_dist", type=float, default=0.005)
parser.add_argument("--close_steps", type=int, default=8)
parser.add_argument("--max_ee_step", type=float, default=0.02)

parser.add_argument("--debug_first_steps", type=int, default=0,
                    help="Print env-0 diagnostic for the first N steps.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# Oracle module is in the same directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_policy import OraclePolicy  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    device = env.unwrapped.device
    n_envs = env.unwrapped.num_envs
    robot = env.unwrapped.scene["robot"]
    object_asset = env.unwrapped.scene["object"]

    # Provide MOVE_TO_GOAL target: command_manager.get_command("object_pose")
    # returns (N, 7) [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z] in robot
    # base frame for UniformPoseCommand. We forward only the position to the oracle.
    def goal_provider() -> torch.Tensor:
        cmd = env.unwrapped.command_manager.get_command("object_pose")
        return cmd[:, 0:3].detach()

    oracle = OraclePolicy(
        num_envs=n_envs,
        device=device,
        scale=args_cli.scale,
        reach_above_dz=args_cli.reach_above_dz,
        descend_dz=args_cli.descend_dz,
        lift_dz=args_cli.lift_dz,
        reach_dist=args_cli.reach_dist,
        descend_z_dist=args_cli.descend_z_dist,
        close_steps=args_cli.close_steps,
        max_ee_step=args_cli.max_ee_step,
        success_z=args_cli.success_z,
    )
    oracle.setup(robot=robot, object_asset=object_asset, target_pos_b_provider=goal_provider)

    print(f"[INFO] arm_ids={oracle.arm_ids}  ee_idx={oracle.ee_idx}  n_envs={n_envs}")
    print(f"[INFO] default_arm[0]={oracle.default_arm[0].cpu().tolist()}")
    print(f"[INFO] scale={oracle.scale}  success_z={args_cli.success_z}  "
          f"target={args_cli.target_episodes}")

    obs = env.get_observations()
    obs_tensor = obs["policy"]
    obs_dim = obs_tensor.shape[1]

    episode_obs = [[] for _ in range(n_envs)]
    episode_act = [[] for _ in range(n_envs)]
    episode_z_max = torch.zeros(n_envs, device=device)

    demos_obs: list[torch.Tensor] = []
    demos_act: list[torch.Tensor] = []
    demos_meta: list[dict] = []
    successes = 0
    finished_episodes = 0
    step_count = 0

    saturation_count = torch.zeros(5, device=device)
    saturation_total = 0
    start_time = time.time()

    env_origins_z = env.unwrapped.scene.env_origins[:, 2]

    print("[INFO] Collecting...")
    while successes < args_cli.target_episodes and finished_episodes < args_cli.max_total_episodes:
        action, info = oracle.compute()

        # Pre-clamp arm_raw saturation rate (computed before clamp inside to_action).
        # Recompute here for diagnostics.
        with torch.no_grad():
            jt = info["joint_target"]
            arm_raw_unclamped = (jt - oracle.default_arm) / oracle.scale
            sat = (arm_raw_unclamped.abs() > 1.0).float()
            saturation_count += sat.sum(dim=0).detach()
            saturation_total += n_envs

            cube_z_local = (object_asset.data.root_pos_w[:, 2] - env_origins_z).detach().clone()

        for i in range(n_envs):
            episode_obs[i].append(obs_tensor[i].detach().cpu().clone())
            episode_act[i].append(action[i].detach().cpu().clone())
        episode_z_max = torch.maximum(episode_z_max, cube_z_local)

        if step_count < args_cli.debug_first_steps:
            print(
                f"[DBG step={step_count}] s[0]={int(oracle.states[0])} "
                f"ee={info['ee_pos_b'][0].cpu().tolist()} "
                f"cube={info['cube_pos_b'][0].cpu().tolist()} "
                f"tgt={info['target_pos_b'][0].cpu().tolist()} "
                f"close_step={int(oracle.close_step[0])} z_max={float(episode_z_max[0]):.3f}"
            )

        with torch.no_grad():
            obs, _, dones, _ = env.step(action)
            obs_tensor = obs["policy"]
            done_indices = torch.nonzero(dones, as_tuple=False).squeeze(-1).tolist()

        step_count += 1

        for env_idx in done_indices:
            finished_episodes += 1
            z_max = float(episode_z_max[env_idx])
            ep_len = len(episode_obs[env_idx])
            if z_max >= args_cli.success_z and ep_len > 0:
                demos_obs.append(torch.stack(episode_obs[env_idx]))
                demos_act.append(torch.stack(episode_act[env_idx]))
                demos_meta.append({
                    "length": ep_len,
                    "z_max": z_max,
                    "env_idx": int(env_idx),
                    "final_state": int(oracle.states[env_idx]),
                })
                successes += 1
                if successes % 10 == 0 or successes == args_cli.target_episodes:
                    elapsed = time.time() - start_time
                    rate = successes / max(finished_episodes, 1) * 100.0
                    print(
                        f"[INFO] succ={successes}/{args_cli.target_episodes} "
                        f"finished={finished_episodes} rate={rate:.1f}% "
                        f"z_max={z_max:.3f} len={ep_len} elapsed={elapsed:.0f}s"
                    )
            episode_obs[env_idx] = []
            episode_act[env_idx] = []
            episode_z_max[env_idx] = 0.0

        # Reset oracle state for any envs that ended.
        if done_indices:
            oracle.reset(done_indices)

    elapsed = time.time() - start_time
    rate = successes / max(finished_episodes, 1) * 100.0
    sat_pct = (saturation_count / max(saturation_total, 1) * 100.0).cpu().tolist()
    print(
        f"\n[DONE] succ={successes} finished={finished_episodes} "
        f"rate={rate:.2f}% steps={step_count} elapsed={elapsed:.0f}s"
    )
    print(f"[DONE] per-joint saturation rate (%): {[f'{s:.1f}' for s in sat_pct]}")

    if successes == 0:
        print("[WARN] no successes — saving skipped.")
    else:
        os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
        torch.save(
            {
                "obs": demos_obs,
                "act": demos_act,
                "meta": demos_meta,
                "task": args_cli.task,
                "obs_dim": obs_dim,
                "act_dim": int(demos_act[0].shape[1]),
                "success_z": args_cli.success_z,
                "source": "oracle_v1",
                "n_total_episodes_attempted": finished_episodes,
                "saturation_per_joint_pct": sat_pct,
                "params": {
                    "scale": args_cli.scale,
                    "reach_above_dz": args_cli.reach_above_dz,
                    "descend_dz": args_cli.descend_dz,
                    "lift_dz": args_cli.lift_dz,
                    "reach_dist": args_cli.reach_dist,
                    "descend_z_dist": args_cli.descend_z_dist,
                    "close_steps": args_cli.close_steps,
                    "max_ee_step": args_cli.max_ee_step,
                },
            },
            args_cli.output,
        )
        print(f"[INFO] saved {successes} demos → {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
