"""Robot workspace measurement: shoulder_pan 별 ee 도달 가능 위치.

목적: sim 의 실제 reach 영역 확정. URDF 27.7cm 와 PD 한계 비교.
방법:
  - 각 shoulder_pan 목표 값에 대해 robot reset → 자세 set → PD 안정화 → ee 측정.
  - 나머지 arm joints 는 work-ready (lift -0.8, elb 1.0, wrist 1.24, roll 0.05).
"""

import argparse
import sys

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_pans", type=int, default=11)
parser.add_argument("--envs_per_pan", type=int, default=4)
parser.add_argument("--pan_min", type=float, default=-1.9)
parser.add_argument("--pan_max", type=float, default=1.9)
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"] + sys.argv[1:])

launcher = AppLauncher(args)

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402

from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import (  # noqa: E402
    SoArm101PickPlaceEnvCfg,
)
from jabis_sim_v2.tasks.cube_lift.mdp import events as cube_events  # noqa: E402


def main():
    pan_vals = np.linspace(args.pan_min, args.pan_max, args.num_pans)
    total_envs = args.num_pans * args.envs_per_pan

    cfg = SoArm101PickPlaceEnvCfg()
    cfg.scene.num_envs = total_envs
    cfg.episode_length_s = 100.0
    # cube를 robot 멀리 놓아서 reach 영향 없게
    cfg.events.reset_cube = EventTerm(
        func=cube_events.reset_cube_grid,
        mode="reset",
        params={
            "x_vals": [0.8],
            "y_vals": [0.0],
            "envs_per_point": total_envs,
        },
    )

    env = ManagerBasedRLEnv(cfg=cfg)
    device = env.device

    robot = env.scene["robot"]
    arm_joint_names = [
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
    ]
    arm_ids = robot.find_joints(arm_joint_names)[0]
    arm_ids_t = torch.tensor(arm_ids, device=device, dtype=torch.long)
    ee_idx = robot.find_bodies("gripper_link")[0][0]

    env_to_pan = np.repeat(np.arange(args.num_pans), args.envs_per_pan)
    env_to_pan_t = torch.from_numpy(env_to_pan).long().to(device)

    default_arm = robot.data.default_joint_pos[:, arm_ids_t].clone()
    pan_t = torch.from_numpy(pan_vals).float().to(device)

    print(
        f"[Workspace] {args.num_pans} pan values, {args.envs_per_pan} envs/pan, "
        f"steps={args.steps}",
        flush=True,
    )
    print(f"[Workspace] pan range [{args.pan_min:.2f}, {args.pan_max:.2f}]", flush=True)

    env.reset()

    # Action: shoulder_pan = pan_per_env, others = default → raw = (target - default) / 1.5
    target_arm = default_arm.clone()
    target_arm[:, 0] = pan_t[env_to_pan_t]
    arm_action_scale = 1.5
    arm_raw = (target_arm - default_arm) / arm_action_scale
    arm_raw = arm_raw.clamp(-1.0, 1.0)
    action = torch.zeros(total_envs, 6, device=device)
    action[:, :5] = arm_raw
    action[:, -1] = 1.0  # gripper open

    for step in range(args.steps):
        env.step(action)

    ee_pos = robot.data.body_pos_w[:, ee_idx] - env.scene.env_origins
    pan_actual = robot.data.joint_pos[:, arm_ids[0]]

    print(f"\n{'pan_cmd':>8} {'pan_actual':>10} | {'ee_x':>7} {'ee_y':>7} {'ee_z':>7}", flush=True)
    print("-" * 60, flush=True)
    for i, pan in enumerate(pan_vals):
        mask = env_to_pan_t == i
        ex = ee_pos[mask, 0].median().item()
        ey = ee_pos[mask, 1].median().item()
        ez = ee_pos[mask, 2].median().item()
        pa = pan_actual[mask].median().item()
        print(
            f"{pan:+8.3f} {pa:+10.3f} | {ex:+7.3f} {ey:+7.3f} {ez:+7.3f}",
            flush=True,
        )

    env.close()


main()
launcher.app.close()
