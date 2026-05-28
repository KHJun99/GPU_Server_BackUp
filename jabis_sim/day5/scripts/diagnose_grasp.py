"""Grasp diagnostic for SoArm101 lift policy.

Runs N envs for N_STEPS using a trained rsl_rl checkpoint and records:
  - left_proximal joint angle per step
  - raw policy action[:, 5] (gripper command) per step
  - ee <-> cube distance per step

Then prints:
  - close-attempt count    (|left_proximal| > 0.1)
  - policy close-command   (action[:,5] < 0)
  - left_proximal min/max/mean/std
  - mean closure when ee within 5 cm of cube

Optionally records a video of the same run.

Usage (from ~/isaac_so_arm101):
  CUDA_VISIBLE_DEVICES=1 conda run -n isaaclab python \
    /home/j-k14d101/jabis_sim/day5/scripts/diagnose_grasp.py \
    --task Isaac-Lift-Cube-SoArm101-v0 \
    --num_envs 4 --steps 400 --video \
    --checkpoint /home/j-k14d101/isaac_so_arm101/logs/rsl_rl/lift/2026-05-08_13-58-09/model_750.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=400)
parser.add_argument("--output_dir", type=str,
                    default="/home/j-k14d101/jabis_sim/day5/logs")
parser.add_argument("--close_thresh", type=float, default=0.1,
                    help="|left_proximal| > thresh counts as close attempt")
parser.add_argument("--near_thresh", type=float, default=0.05,
                    help="ee<->cube distance below this counts as 'near'")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Rest of imports must come after AppLauncher
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaac_so_arm101.tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg) -> None:  # type: ignore[no-untyped-def]
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    if not args_cli.checkpoint:
        raise SystemExit("--checkpoint is required (provided by rsl_rl cli_args)")
    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        ts = time.strftime("%H%M%S")
        video_dir = os.path.join(args_cli.output_dir, f"grasp_diag_{ts}")
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[INFO] video -> {video_dir}")

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["object"]
    ee_frame = env.unwrapped.scene["ee_frame"]
    joint_names: list[str] = list(robot.data.joint_names)
    if "left_proximal" not in joint_names:
        raise RuntimeError(f"left_proximal not in robot joints: {joint_names}")
    lp_idx = joint_names.index("left_proximal")

    GRIPPER_JOINTS = ["left_proximal", "left_distal", "right_proximal",
                      "right_distal", "gripper"]
    gripper_idx: dict[str, int | None] = {
        j: (joint_names.index(j) if j in joint_names else None)
        for j in GRIPPER_JOINTS
    }
    print(f"[INFO] joint_names ({len(joint_names)}): {joint_names}")
    print(f"[INFO] gripper joint indices: {gripper_idx}")
    print(f"[INFO] num_envs={args_cli.num_envs}, steps={args_cli.steps}")

    proximal_log: list[torch.Tensor] = []
    grip_action_log: list[torch.Tensor] = []
    distance_log: list[torch.Tensor] = []
    gripper_logs: dict[str, list[torch.Tensor]] = {
        j: [] for j, idx in gripper_idx.items() if idx is not None
    }
    cube_z_log: list[torch.Tensor] = []
    cube_vz_log: list[torch.Tensor] = []
    cube_xy_log: list[torch.Tensor] = []
    first_lift: dict[str, float | int] | None = None
    LIFT_Z_THRESH = 0.04
    LIFT_VZ_THRESH = 0.05

    obs = env.get_observations()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        proximal_log.append(robot.data.joint_pos[:, lp_idx].detach().clone().cpu())
        grip_action_log.append(actions[:, 5].detach().clone().cpu())
        ee_w = ee_frame.data.target_pos_w[..., 0, :]
        cube_w = obj.data.root_pos_w
        dist = torch.linalg.vector_norm(ee_w - cube_w, dim=-1)
        distance_log.append(dist.detach().clone().cpu())
        for j, idx in gripper_idx.items():
            if idx is not None:
                gripper_logs[j].append(
                    robot.data.joint_pos[:, idx].detach().clone().cpu()
                )
        cube_z_step = cube_w[:, 2].detach().clone().cpu()
        cube_vz_step = obj.data.root_lin_vel_w[:, 2].detach().clone().cpu()
        cube_z_log.append(cube_z_step)
        cube_vz_log.append(cube_vz_step)
        cube_xy_log.append(cube_w[:, :2].detach().clone().cpu())

        if first_lift is None:
            lifted = (cube_z_step > LIFT_Z_THRESH).nonzero(as_tuple=False).flatten()
            if lifted.numel() > 0:
                env_id = int(lifted[0].item())
                snapshot = {
                    j: float(robot.data.joint_pos[env_id, idx].item())
                    for j, idx in gripper_idx.items() if idx is not None
                }
                first_lift = {
                    "step": step,
                    "env_id": env_id,
                    "cube_z": float(cube_z_step[env_id].item()),
                    "cube_vz": float(cube_vz_step[env_id].item()),
                    "ee_obj_dist": float(dist[env_id].item()),
                    **{f"q_{j}": v for j, v in snapshot.items()},
                }
                print(f"[INFO] first lift: step={step} env={env_id} "
                      f"cube_z={cube_z_step[env_id].item():.4f} "
                      f"snapshot={snapshot}")

    proximal = torch.stack(proximal_log)
    grip_act = torch.stack(grip_action_log)
    distance = torch.stack(distance_log)
    cube_z = torch.stack(cube_z_log)
    cube_vz = torch.stack(cube_vz_log)
    cube_xy = torch.stack(cube_xy_log)
    gripper_tensors: dict[str, torch.Tensor] = {
        j: torch.stack(v) for j, v in gripper_logs.items()
    }
    total_steps = proximal.numel()

    close_attempts = (proximal.abs() > args_cli.close_thresh).sum().item()
    policy_close_cmds = (grip_act < 0.0).sum().item()
    near_mask = distance < args_cli.near_thresh
    near_steps = int(near_mask.sum().item())
    closure_when_near = (
        proximal[near_mask].abs().mean().item() if near_steps > 0 else float("nan")
    )

    lift_step_count = int((cube_z > LIFT_Z_THRESH).sum().item())
    velocity_lift_count = int((cube_vz > LIFT_VZ_THRESH).sum().item())

    gripper_stats: dict[str, dict[str, float]] = {}
    for j, t in gripper_tensors.items():
        gripper_stats[j] = {
            "mean": float(t.mean()),
            "min":  float(t.min()),
            "max":  float(t.max()),
            "std":  float(t.std()),
        }

    summary: dict[str, object] = {
        "checkpoint": str(resume_path),
        "task": args_cli.task,
        "num_envs": int(args_cli.num_envs),
        "steps": int(args_cli.steps),
        "total_step_samples": int(total_steps),
        "close_thresh": float(args_cli.close_thresh),
        "near_thresh": float(args_cli.near_thresh),
        "close_attempt_count": int(close_attempts),
        "close_attempt_ratio": float(close_attempts / max(total_steps, 1)),
        "policy_close_cmd_count": int(policy_close_cmds),
        "policy_close_cmd_ratio": float(policy_close_cmds / max(total_steps, 1)),
        "near_step_count": near_steps,
        "near_step_ratio": float(near_steps / max(total_steps, 1)),
        "mean_closure_abs_when_near": float(closure_when_near)
            if closure_when_near == closure_when_near else None,
        "left_proximal_min": float(proximal.min()),
        "left_proximal_max": float(proximal.max()),
        "left_proximal_mean": float(proximal.mean()),
        "left_proximal_std": float(proximal.std()),
        "ee_obj_dist_min": float(distance.min()),
        "ee_obj_dist_mean": float(distance.mean()),
        "cube_z_mean": float(cube_z.mean()),
        "cube_z_min":  float(cube_z.min()),
        "cube_z_max":  float(cube_z.max()),
        "cube_vz_mean": float(cube_vz.mean()),
        "cube_vz_max":  float(cube_vz.max()),
        "lift_step_count_z_gt_0_04": lift_step_count,
        "velocity_lift_count_vz_gt_0_05": velocity_lift_count,
        "first_lift_event": first_lift,
        "gripper_stats": gripper_stats,
    }

    print("\n" + "=" * 60)
    print("GRASP DIAGNOSTIC SUMMARY")
    print("=" * 60)
    scalar_keys = [k for k, v in summary.items()
                   if not isinstance(v, (dict, list)) and v is not None
                   and k not in ("first_lift_event", "gripper_stats")]
    for k in scalar_keys:
        v = summary[k]
        if isinstance(v, float):
            print(f"  {k:32s} = {v:.4f}")
        else:
            print(f"  {k:32s} = {v}")

    print("\n" + "-" * 60)
    print("GRIPPER 4-BAR JOINT STATS")
    print("-" * 60)
    print(f"| {'joint':18s} | {'mean':>9s} | {'min':>9s} | "
          f"{'max':>9s} | {'std':>9s} |")
    print(f"| {'-'*18} | {'-'*9} | {'-'*9} | {'-'*9} | {'-'*9} |")
    for j in GRIPPER_JOINTS:
        if j in gripper_stats:
            s = gripper_stats[j]
            print(f"| {j:18s} | {s['mean']:+9.4f} | {s['min']:+9.4f} | "
                  f"{s['max']:+9.4f} | {s['std']:9.4f} |")
        else:
            print(f"| {j:18s} | {'(absent)':>9s} | "
                  f"{'-':>9s} | {'-':>9s} | {'-':>9s} |")

    print("\n" + "-" * 60)
    print("CUBE STATS")
    print("-" * 60)
    print(f"  cube_z mean/min/max         = "
          f"{summary['cube_z_mean']:.4f} / {summary['cube_z_min']:.4f} / "
          f"{summary['cube_z_max']:.4f}")
    print(f"  cube_vz mean/max            = "
          f"{summary['cube_vz_mean']:+.4f} / {summary['cube_vz_max']:+.4f}")
    print(f"  lift_step_count(z>0.04)     = {lift_step_count} / {total_steps}")
    print(f"  velocity_lift(vz>0.05)      = {velocity_lift_count} / {total_steps}")
    if first_lift is not None:
        print(f"  first_lift                  = {first_lift}")
    else:
        print("  first_lift                  = (none — cube never above 0.04 m)")

    # XY drift: per-env (max - min) of (x, y) — measures sliding/pushing
    cube_x = cube_xy[..., 0]
    cube_y = cube_xy[..., 1]
    x_drift = (cube_x.max(dim=0).values - cube_x.min(dim=0).values)
    y_drift = (cube_y.max(dim=0).values - cube_y.min(dim=0).values)
    summary["cube_x_drift_per_env_max"] = float(x_drift.max())
    summary["cube_x_drift_per_env_mean"] = float(x_drift.mean())
    summary["cube_y_drift_per_env_max"] = float(y_drift.max())
    summary["cube_y_drift_per_env_mean"] = float(y_drift.mean())
    print(f"  cube xy drift (per-env max)  = "
          f"x: max {x_drift.max():.4f} mean {x_drift.mean():.4f}, "
          f"y: max {y_drift.max():.4f} mean {y_drift.mean():.4f}")
    print("=" * 60)

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%H%M%S")
    json_path = out_dir / f"grasp_diag_{ts}.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] summary -> {json_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
