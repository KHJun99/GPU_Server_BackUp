# -*- coding: utf-8 -*-
"""SO-ARM101 각 관절 단독 이동 → gripper world pos 측정 + matplotlib top-view"""

import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

import sys, os
sys.path.insert(0, os.path.expanduser("~/jabis_sim"))
os.makedirs(os.path.expanduser("~/jabis_sim/day4/captures"), exist_ok=True)

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

from configs.robots.so101_cfg import SO101_PINCOPEN_CFG


def settle(sim, robot, cup, target, steps, sim_dt):
    for _ in range(steps):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        cup.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(sim_cfg)

    cfg_light = sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0))
    cfg_light.func("/World/Light", cfg_light)

    cfg_floor = sim_utils.CuboidCfg(
        size=(10.0, 10.0, 0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
    )
    cfg_floor.func("/World/Floor", cfg_floor, translation=(0.0, 0.0, -0.05))

    cfg_desk = sim_utils.CuboidCfg(
        size=(0.40, 0.30, 0.02),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
    )
    cfg_desk.func("/World/Desk", cfg_desk, translation=(0.25, 0.0, 0.04))

    robot_cfg = SO101_PINCOPEN_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg=robot_cfg)

    cup_cfg = RigidObjectCfg(
        prim_path="/World/Cup",
        spawn=sim_utils.CylinderCfg(
            radius=0.025, height=0.06,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.05, 0.08)),
    )
    cup = RigidObject(cfg=cup_cfg)

    sim.reset()

    grip_idx = robot.body_names.index("left_distal_link")
    main_joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
    test_value = 0.5  # 각 관절 단독으로 +0.5 rad

    print(f"\n{'='*70}")
    print(f"SO-ARM101 관절 calibration — 각 관절 home + {test_value} rad 단독 이동")
    print(f"{'='*70}")
    print(f"Cup target: ({0.25:+.3f}, {0.05:+.3f}, 0.080)")
    print(f"Body: left_distal_link")

    # Home 측정
    settle(sim, robot, cup, robot.data.default_joint_pos, 60, sim_cfg.dt)
    home_pos = robot.data.body_pos_w[0, grip_idx].cpu().numpy().copy()
    print(f"\n[Home] gripper = ({home_pos[0]:+.4f}, {home_pos[1]:+.4f}, {home_pos[2]:+.4f})")

    results = {"home": home_pos}

    # 각 관절 단독 이동
    for jname in main_joints:
        # 항상 home으로 리셋
        target = robot.data.default_joint_pos.clone()
        idx = robot.joint_names.index(jname)
        target[:, idx] = test_value
        settle(sim, robot, cup, target, 200, sim_cfg.dt)
        pos = robot.data.body_pos_w[0, grip_idx].cpu().numpy().copy()
        delta = pos - home_pos
        results[jname] = {"pos": pos, "delta": delta}
        print(f"\n[{jname}={test_value:+.2f}]")
        print(f"  gripper = ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")
        print(f"  Δ from home = ({delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f})")
        # home 복귀
        settle(sim, robot, cup, robot.data.default_joint_pos, 80, sim_cfg.dt)

    # ==================== matplotlib top-view ====================
    print(f"\n{'='*70}")
    print("matplotlib top-view 그리는 중...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    fig, ax = plt.subplots(figsize=(10, 8))
    # Desk (top-down: 0.40 x 0.30)
    ax.add_patch(Rectangle((0.05, -0.15), 0.40, 0.30,
                            facecolor=(0.6, 0.4, 0.2), alpha=0.5, label="Desk"))
    # Cup
    ax.add_patch(Circle((0.25, 0.05), 0.025,
                         facecolor=(0.8, 0.2, 0.2), label="Cup"))
    # Robot base (대략 base_link world 위치)
    base_idx = robot.body_names.index("base_link")
    base_pos = robot.data.body_pos_w[0, base_idx].cpu().numpy()
    ax.add_patch(Circle((base_pos[0], base_pos[1]), 0.03,
                         facecolor="black", label="Robot base"))

    # Home gripper
    ax.scatter(home_pos[0], home_pos[1], color="blue", s=150,
               marker="*", zorder=10, label=f"Home gripper")

    # 각 관절 단독 이동 결과
    colors = {"shoulder_pan": "red", "shoulder_lift": "green",
              "elbow_flex": "purple", "wrist_flex": "orange"}
    for jname in main_joints:
        p = results[jname]["pos"]
        ax.annotate("", xy=(p[0], p[1]), xytext=(home_pos[0], home_pos[1]),
                    arrowprops=dict(arrowstyle="->", color=colors[jname], lw=2))
        ax.scatter(p[0], p[1], color=colors[jname], s=100, zorder=10,
                   label=f"{jname}+0.5")

    ax.set_xlabel("X (전방, m)")
    ax.set_ylabel("Y (좌측+, m)")
    ax.set_title(f"SO-ARM101 관절 단독 +{test_value} rad 영향 (top-view)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-0.2, 0.7)
    ax.set_ylim(-0.4, 0.4)

    out_path = os.path.expanduser("~/jabis_sim/day4/captures/calibration_topview.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"📊 저장: {out_path}")
    print(f"{'='*70}\n")

    # 요약 테이블
    print("=== 관절 영향 요약 (cm 단위) ===")
    print(f"{'joint':<16} {'Δx':>8} {'Δy':>8} {'Δz':>8}")
    for jname in main_joints:
        d = results[jname]["delta"] * 100
        print(f"{jname:<16} {d[0]:+8.2f} {d[1]:+8.2f} {d[2]:+8.2f}")

    sim_app.close()


if __name__ == "__main__":
    main()
