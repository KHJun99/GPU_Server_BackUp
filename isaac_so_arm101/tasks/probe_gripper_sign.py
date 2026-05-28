"""Gripper sign probe — drive arm to default (raw zeros) and toggle gripper command
between -1 and +1 to isolate which direction is "close" for each finger joint.

Records per-step: 10 joint pos, 10 joint vel, 10 joint applied_torque, 6-dim raw
action, plus optional finger-tip distance for sign cross-check (codex caveat).

Usage:
  CUDA_VISIBLE_DEVICES=1 uv run python probe_gripper_sign.py \\
      --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \\
      --num_envs 1 --hold_steps 50 --headless \\
      --output /home/j-k14d101/isaac_so_arm101/tasks/probe_gripper_sign.csv
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Probe gripper sign by isolated close/open commands.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Lift-Cube-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--hold_steps", type=int, default=50,
                    help="Steps to hold close (-1) and then open (+1) commands.")
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)

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


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    n_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.data.joint_names)
    n_joints = len(joint_names)
    print(f"[INFO] joint_names ({n_joints}): {joint_names}")

    # Locate finger body indices for tip-distance cross-check (codex caveat).
    body_names = list(robot.data.body_names)
    print(f"[INFO] body_names ({len(body_names)}): {body_names}")
    left_distal_idx = body_names.index("left_distal_link") if "left_distal_link" in body_names else -1
    right_distal_idx = body_names.index("right_distal_link") if "right_distal_link" in body_names else -1
    print(f"[INFO] left_distal_link body idx={left_distal_idx}  right_distal_link={right_distal_idx}")

    header = ["step", "phase", "raw_grip_cmd"]
    for jn in joint_names:
        header.append(f"jpos_{jn}")
    for jn in joint_names:
        header.append(f"jvel_{jn}")
    for jn in joint_names:
        header.append(f"jtorque_{jn}")
    header += ["tip_distance"]

    out_dir = os.path.dirname(args_cli.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fp = open(args_cli.output, "w", newline="")
    writer = csv.writer(fp)
    writer.writerow(header)

    _ = env.get_observations()

    # Build a fixed action: arm 5 channels = 0 (use_default_offset=True so this means
    # "stay at default joint pos"), gripper channel toggled per phase.
    action = torch.zeros((n_envs, 6), device=device)

    print(f"[INFO] hold_steps={args_cli.hold_steps}  total={args_cli.hold_steps * 2}")
    print(f"[INFO] writing to: {args_cli.output}")

    for step_idx in range(args_cli.hold_steps * 2):
        if step_idx < args_cli.hold_steps:
            phase = "close"
            grip_cmd = -1.0
        else:
            phase = "open"
            grip_cmd = 1.0
        action[:, :5] = 0.0
        action[:, 5] = grip_cmd

        # Snapshot CURRENT state (before applying).
        joint_pos_now = robot.data.joint_pos[0].detach().cpu().tolist()
        joint_vel_now = robot.data.joint_vel[0].detach().cpu().tolist()
        # Torque snapshot (some IsaacLab versions name this differently).
        try:
            torque_now = robot.data.applied_torque[0].detach().cpu().tolist()
        except AttributeError:
            torque_now = [float("nan")] * n_joints

        tip_dist = float("nan")
        if left_distal_idx >= 0 and right_distal_idx >= 0:
            try:
                pos_w = robot.data.body_pos_w[0]
                d = pos_w[left_distal_idx] - pos_w[right_distal_idx]
                tip_dist = float(torch.norm(d))
            except (AttributeError, RuntimeError):
                pass

        row = [step_idx, phase, grip_cmd]
        row += joint_pos_now + joint_vel_now + torque_now + [tip_dist]
        writer.writerow(row)

        with torch.no_grad():
            _, _, _, _ = env.step(action)

    fp.close()
    print(f"[DONE] wrote {args_cli.hold_steps * 2} rows")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
