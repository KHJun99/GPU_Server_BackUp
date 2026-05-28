"""Validation-only driver for OraclePolicy.

Runs ``OraclePolicy`` inside Isaac-SO-ARM101-Lift-Cube-Play-v0 for a fixed
number of episodes and reports:

  * success rate (cube_z_max >= success_z)
  * state-at-done histogram split by success/failure
  * per-joint saturation rate

Does not save demos. Use ``collect_demos.py`` for that. Designed for Phase 1-1
sanity checks.

Usage examples
--------------

  # 1env-1ep sanity (with state transition trace):
  CUDA_VISIBLE_DEVICES=1 uv run python validate_oracle.py \\
      --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \\
      --num_envs 1 --num_episodes 1 --debug_first_steps 200 --headless

  # 100ep success-rate measurement:
  CUDA_VISIBLE_DEVICES=1 uv run python validate_oracle.py \\
      --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \\
      --num_envs 16 --num_episodes 100 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Validate OraclePolicy without saving demos.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Lift-Cube-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_episodes", type=int, default=100,
                    help="Exact total number of episodes to run before reporting.")
parser.add_argument("--success_z", type=float, default=0.10)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)

parser.add_argument("--scale", type=float, default=1.5)
parser.add_argument("--reach_above_dz", type=float, default=0.10)
parser.add_argument("--descend_dz", type=float, default=0.005)
parser.add_argument("--lift_dz", type=float, default=0.10)
parser.add_argument("--reach_dist", type=float, default=0.02)
parser.add_argument("--descend_z_dist", type=float, default=0.005)
parser.add_argument("--close_steps", type=int, default=8)
parser.add_argument("--max_ee_step", type=float, default=0.02)

parser.add_argument("--debug_first_steps", type=int, default=0,
                    help="Print step-level state/ee/cube/target trace for the first N steps.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaac_so_arm101.tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_policy import OraclePolicy, APPROACH, DESCEND, CLOSE, LIFT, MOVE_TO_GOAL  # noqa: E402


STATE_NAMES = {APPROACH: "APPROACH", DESCEND: "DESCEND", CLOSE: "CLOSE",
               LIFT: "LIFT", MOVE_TO_GOAL: "MOVE_TO_GOAL"}


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
    print(f"[INFO] scale={oracle.scale}  success_z={args_cli.success_z}  "
          f"num_episodes={args_cli.num_episodes}")

    obs = env.get_observations()
    _ = obs["policy"]

    episode_z_max = torch.zeros(n_envs, device=device)
    episode_len = torch.zeros(n_envs, dtype=torch.long, device=device)

    success_count = 0
    fail_count = 0
    finished = 0

    success_state_hist = [0] * 5
    fail_state_hist = [0] * 5

    saturation_count = torch.zeros(5, device=device)
    saturation_total = 0
    step_count = 0
    start_time = time.time()

    env_origins_z = env.unwrapped.scene.env_origins[:, 2]

    print("[INFO] Validating...")
    while finished < args_cli.num_episodes:
        action, info = oracle.compute()

        with torch.no_grad():
            jt = info["joint_target"]
            arm_raw_unclamped = (jt - oracle.default_arm) / oracle.scale
            sat = (arm_raw_unclamped.abs() > 1.0).float()
            saturation_count += sat.sum(dim=0).detach()
            saturation_total += n_envs

            cube_z_local = (object_asset.data.root_pos_w[:, 2] - env_origins_z).detach().clone()

        episode_z_max = torch.maximum(episode_z_max, cube_z_local)
        episode_len += 1

        if step_count < args_cli.debug_first_steps:
            print(
                f"[DBG step={step_count}] s[0]={STATE_NAMES[int(oracle.states[0])]:<13} "
                f"ee={[f'{v:.3f}' for v in info['ee_pos_b'][0].cpu().tolist()]} "
                f"cube={[f'{v:.3f}' for v in info['cube_pos_b'][0].cpu().tolist()]} "
                f"tgt={[f'{v:.3f}' for v in info['target_pos_b'][0].cpu().tolist()]} "
                f"close_step={int(oracle.close_step[0])} z_max={float(episode_z_max[0]):.3f}"
            )

        with torch.no_grad():
            obs, _, dones, _ = env.step(action)
            done_indices = torch.nonzero(dones, as_tuple=False).squeeze(-1).tolist()

        step_count += 1

        for env_idx in done_indices:
            if finished >= args_cli.num_episodes:
                break
            finished += 1
            z_max = float(episode_z_max[env_idx])
            ep_len = int(episode_len[env_idx])
            final_state = int(oracle.states[env_idx])

            if z_max >= args_cli.success_z:
                success_count += 1
                success_state_hist[final_state] += 1
            else:
                fail_count += 1
                fail_state_hist[final_state] += 1

            if finished % 10 == 0 or finished == args_cli.num_episodes:
                elapsed = time.time() - start_time
                rate = success_count / max(finished, 1) * 100.0
                print(
                    f"[INFO] finished={finished}/{args_cli.num_episodes} "
                    f"succ={success_count} fail={fail_count} rate={rate:.1f}% "
                    f"last z_max={z_max:.3f} len={ep_len} state={STATE_NAMES[final_state]} "
                    f"elapsed={elapsed:.0f}s"
                )

            episode_z_max[env_idx] = 0.0
            episode_len[env_idx] = 0

        if done_indices:
            oracle.reset(done_indices)

    elapsed = time.time() - start_time
    rate = success_count / max(finished, 1) * 100.0
    sat_pct = (saturation_count / max(saturation_total, 1) * 100.0).cpu().tolist()

    print()
    print("=" * 60)
    print(f"[DONE] episodes={finished}  steps={step_count}  elapsed={elapsed:.0f}s")
    print(f"[DONE] success={success_count} fail={fail_count} success_rate={rate:.2f}%")
    print(f"[DONE] per-joint saturation (%): {[f'{s:.1f}' for s in sat_pct]}")
    print()
    print("[STATE HISTOGRAM] (count by state at done)")
    for s in range(5):
        print(f"  {STATE_NAMES[s]:<13} success={success_state_hist[s]:3d}  fail={fail_state_hist[s]:3d}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
