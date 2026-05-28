# -*- coding: utf-8 -*-
"""Day 4 Step 4-5 — Reach 테스트
책상 위 컵 좌표로 다축 joint target 이동, gripper 링크 world pos 측정.
"""

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import sys, os
sys.path.insert(0, os.path.expanduser("~/jabis_sim"))

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

from configs.robots.so101_cfg import SO101_PINCOPEN_CFG


def main():
    # Sim
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.2, 1.2, 0.8], target=[0.25, 0.05, 0.10])

    # Light
    cfg_light = sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0))
    cfg_light.func("/World/Light", cfg_light)

    # Floor
    cfg_floor = sim_utils.CuboidCfg(
        size=(10.0, 10.0, 0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
    )
    cfg_floor.func("/World/Floor", cfg_floor, translation=(0.0, 0.0, -0.05))

    # Desk
    cfg_desk = sim_utils.CuboidCfg(
        size=(0.40, 0.30, 0.02),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
    )
    cfg_desk.func("/World/Desk", cfg_desk, translation=(0.25, 0.0, 0.04))

    # Robot
    robot_cfg = SO101_PINCOPEN_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg=robot_cfg)

    # Cup
    cup_cfg = RigidObjectCfg(
        prim_path="/World/Cup",
        spawn=sim_utils.CylinderCfg(
            radius=0.025,
            height=0.06,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.05, 0.08)),
    )
    cup = RigidObject(cfg=cup_cfg)

    sim.reset()

    print("\n========== Reach Test Setup ==========")
    print(f"Robot joints: {robot.joint_names}")
    print(f"Robot bodies: {robot.body_names}")
    print(f"Cup pos     : {cup.data.root_pos_w[0].cpu().numpy()}")
    print("======================================\n")

    # ----------------------------------------------------------------
    # Phase 1: Home 안정화 (60 step)
    # ----------------------------------------------------------------
    print(">>> Phase 1: Home 안정화")
    for _ in range(60):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg.dt)
        cup.update(sim_cfg.dt)

    # ----------------------------------------------------------------
    # Phase 2: Reach target — 컵 위 5cm 추정 자세
    #   1차 추정값 (Day 4 가이드 표):
    #     shoulder_pan  +0.10  (좌측 5cm 방향)
    #     shoulder_lift -0.30  (어깨 내림)
    #     elbow_flex    -0.50  (팔꿈치 굽힘)
    #     wrist_flex    -0.20  (그리퍼 아래)
    # ----------------------------------------------------------------
    print("\n>>> Phase 2: Reach target 적용 (200 step)")
    target = robot.data.default_joint_pos.clone()
    reach_targets = {
        "shoulder_pan":   0.10,
        "shoulder_lift": -0.30,
        "elbow_flex":    -0.50,
        "wrist_flex":    -0.20,
    }
    for name, val in reach_targets.items():
        idx = robot.joint_names.index(name)
        target[:, idx] = val

    for i in range(200):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg.dt)
        cup.update(sim_cfg.dt)

    # ----------------------------------------------------------------
    # 결과 측정
    # ----------------------------------------------------------------
    # gripper 링크 world position 추출
    # body_names에서 그리퍼 끝쪽 링크 자동 탐색
    candidate_names = ["left_distal_link", "right_distal_link", "gripper_link", "wrist_roll_link"]
    grip_link = None
    for n in candidate_names:
        if n in robot.body_names:
            grip_link = n
            break
    if grip_link is None:
        grip_link = robot.body_names[-1]  # fallback: 마지막 link
    grip_idx = robot.body_names.index(grip_link)

    grip_pos = robot.data.body_pos_w[0, grip_idx].cpu().numpy()
    cup_pos = cup.data.root_pos_w[0].cpu().numpy()
    final_joints = robot.data.joint_pos[0].cpu().numpy()

    # 마지막 50 step joint err 측정 (정착 여부)
    joint_err = abs(final_joints - target[0].cpu().numpy()).max()

    # 합격선
    dx = grip_pos[0] - cup_pos[0]
    dy = grip_pos[1] - cup_pos[1]
    dz = grip_pos[2] - cup_pos[2]
    xy_dist = (dx**2 + dy**2)**0.5
    cup_displaced = (
        (cup_pos[0] - 0.25)**2
        + (cup_pos[1] - 0.05)**2
    )**0.5

    print("\n========== Day 4 Step 4-5 결과 ==========")
    print(f"Reach 기준 link  : {grip_link} (body idx {grip_idx})")
    print(f"  Gripper world  : ({grip_pos[0]:+.4f}, {grip_pos[1]:+.4f}, {grip_pos[2]:+.4f})")
    print(f"  Cup world      : ({cup_pos[0]:+.4f}, {cup_pos[1]:+.4f}, {cup_pos[2]:+.4f})")
    print(f"  XY 거리        : {xy_dist*100:.2f} cm  (합격선 < 8 cm)")
    print(f"  Z 차이         : {dz*100:+.2f} cm  (합격선: 컵 위 3~10 cm)")
    print(f"Cup 이동량       : {cup_displaced*1000:.2f} mm  (합격선 < 5 mm)")
    print(f"Joint 정착 err   : {joint_err:.4f} rad  (합격선 < 0.05 rad)")

    pass_xy = xy_dist < 0.08
    pass_z  = 0.03 <= dz <= 0.10
    pass_cup = cup_displaced < 0.005
    pass_joint = joint_err < 0.05

    print(f"\n  XY      : {'✅' if pass_xy    else '❌'}")
    print(f"  Z       : {'✅' if pass_z     else '❌'}")
    print(f"  Cup     : {'✅' if pass_cup   else '❌'}")
    print(f"  Joint   : {'✅' if pass_joint else '❌'}")

    if pass_xy and pass_z and pass_cup and pass_joint:
        print("\n🎉 Day 4 Step 4-5 PASS — Day 4 종료, Day 5 IK 진입 가능")
    else:
        print("\n⚠️  일부 실패 — joint target 재조정 필요 (보고 후 가이드)")
    print("=========================================\n")

    sim_app.close()


if __name__ == "__main__":
    main()
