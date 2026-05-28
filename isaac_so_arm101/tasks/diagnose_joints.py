"""Diagnostic driver — log per-step joint positions/velocities, oracle action vector,
target/current ee, cube position, and oracle state to CSV.

Designed for session-2 hypothesis testing (H1/H2/H3). Reads OraclePolicy/env without
modifying any of them. Writes a single CSV row per env step.

After ``--force_close_step`` (default 100), if oracle is still stuck in DESCEND, the
gripper channel of the action vector is overridden to -1.0 ("force_close" mode) for
the remaining steps so we can observe whether the 5 finger joints respond to a CLOSE
command even when the natural state-machine transition never fires.

Usage
-----
  CUDA_VISIBLE_DEVICES=1 uv run python /home/j-k14d101/isaac_so_arm101/tasks/diagnose_joints.py \\
      --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \\
      --num_envs 1 --max_steps 200 \\
      --output /home/j-k14d101/isaac_so_arm101/tasks/diagnose_joints.csv \\
      --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Diagnose joint activations.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Lift-Cube-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--max_steps", type=int, default=200)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--force_close_step", type=int, default=100,
                    help="From this step on, if oracle is still in DESCEND, override "
                         "action[:, 5] = -1.0 (CLOSE command) so finger response can "
                         "be observed even if natural transition never fires.")

parser.add_argument("--scale", type=float, default=1.5)
parser.add_argument("--reach_above_dz", type=float, default=0.10)
parser.add_argument("--descend_dz", type=float, default=0.005)
parser.add_argument("--lift_dz", type=float, default=0.10)
parser.add_argument("--reach_dist", type=float, default=0.02)
parser.add_argument("--descend_z_dist", type=float, default=0.005)
parser.add_argument("--close_steps", type=int, default=8)
parser.add_argument("--max_ee_step", type=float, default=0.02)
parser.add_argument("--success_z", type=float, default=0.10)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaac_so_arm101.tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

sys.path.insert(0, "/home/j-k14d101/jabis_sim/sim2real/oracle")
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

    n_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    object_asset = env.unwrapped.scene["object"]

    def goal_provider() -> torch.Tensor:
        cmd = env.unwrapped.command_manager.get_command("object_pose")
        return cmd[:, 0:3].detach()

    oracle = OraclePolicy(
        num_envs=n_envs, device=device,
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

    joint_names = list(robot.data.joint_names)
    n_joints = len(joint_names)
    print(f"[INFO] joint_names ({n_joints}): {joint_names}")
    print(f"[INFO] arm_ids={oracle.arm_ids}  ee_idx={oracle.ee_idx}  n_envs={n_envs}")
    print(f"[INFO] writing CSV to: {args_cli.output}")
    print(f"[INFO] force_close_step={args_cli.force_close_step}  max_steps={args_cli.max_steps}")

    header = ["step", "state", "force_close"]
    for jn in joint_names:
        header.append(f"jpos_{jn}")
    for jn in joint_names:
        header.append(f"jvel_{jn}")
    for i in range(6):
        header.append(f"action_{i}")
    header += ["target_ee_x", "target_ee_y", "target_ee_z",
               "current_ee_x", "current_ee_y", "current_ee_z",
               "cube_x", "cube_y", "cube_z"]

    out_dir = os.path.dirname(args_cli.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fp = open(args_cli.output, "w", newline="")
    writer = csv.writer(fp)
    writer.writerow(header)

    _ = env.get_observations()

    last_step = -1
    print("[INFO] Diagnosing...")
    for step_idx in range(args_cli.max_steps):
        # 1) compute oracle action + info from CURRENT robot state.
        action, info = oracle.compute()

        # 2) Decide whether to force-close gripper.
        force_close = False
        if step_idx >= args_cli.force_close_step:
            current_state = int(oracle.states[0])
            if current_state == DESCEND:
                action[:, 5] = -1.0
                force_close = True

        # 3) Snapshot CURRENT state (start-of-step) — joint pos/vel reflect state BEFORE
        #    the action is applied. This pairs with action that will be applied this step.
        state_now = int(oracle.states[0])
        joint_pos_now = robot.data.joint_pos[0].detach().cpu().tolist()
        joint_vel_now = robot.data.joint_vel[0].detach().cpu().tolist()
        action_now = action[0].detach().cpu().tolist()
        target_ee = info["target_pos_b"][0].detach().cpu().tolist()
        current_ee = info["ee_pos_b"][0].detach().cpu().tolist()
        cube = info["cube_pos_b"][0].detach().cpu().tolist()

        row = [step_idx, STATE_NAMES[state_now], int(force_close)]
        row += joint_pos_now
        row += joint_vel_now
        row += action_now
        row += target_ee
        row += current_ee
        row += cube
        writer.writerow(row)
        last_step = step_idx

        # 4) Apply the action.
        with torch.no_grad():
            _, _, dones, _ = env.step(action)
            done_indices = torch.nonzero(dones, as_tuple=False).squeeze(-1).tolist()

        if 0 in done_indices:
            print(f"[INFO] env-0 done at end of step {step_idx}, stopping.")
            oracle.reset(done_indices)
            break

    fp.close()
    print(f"[DONE] wrote {last_step + 1} rows to {args_cli.output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
