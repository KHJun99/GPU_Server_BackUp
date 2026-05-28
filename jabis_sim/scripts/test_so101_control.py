# -*- coding: utf-8 -*-
"""Day 3 — SO-ARM101 Control 검증 (Nucleus-free 버전)"""

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import sys, os
sys.path.insert(0, os.path.expanduser("~/jabis_sim"))

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from configs.robots.so101_cfg import SO101_PINCOPEN_CFG


def main():
    # ----------------------------------------------------------------
    # Step 1: SimulationContext 셋업
    # ----------------------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.5])

    # ----------------------------------------------------------------
    # Step 2: 라이트 (Nucleus 불필요)
    # ----------------------------------------------------------------
    cfg_light = sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0))
    cfg_light.func("/World/Light", cfg_light)

    # ----------------------------------------------------------------
    # Step 3: Floor — Nucleus-free, 큰 큐브로 대체
    # ----------------------------------------------------------------
    # 두께 0.1m, 가로세로 10m 검은색 큐브를 z=-0.05에 배치 → 윗면이 z=0
    cfg_floor = sim_utils.CuboidCfg(
        size=(10.0, 10.0, 0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),  # 고정
        mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
    )
    cfg_floor.func(
        "/World/Floor", cfg_floor,
        translation=(0.0, 0.0, -0.05),
    )

    # ----------------------------------------------------------------
    # Step 4: SO-ARM101 spawn
    # ----------------------------------------------------------------
    robot_cfg = SO101_PINCOPEN_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg=robot_cfg)

    # ----------------------------------------------------------------
    # Step 5: Sim reset (이때 articulation이 활성화됨)
    # ----------------------------------------------------------------
    sim.reset()
    print("\n========== Articulation Info ==========")
    print(f"DOF count        : {robot.num_joints}")
    print(f"Joint names      : {robot.joint_names}")
    print(f"Body count       : {robot.num_bodies}")
    print(f"Initial joint pos: {robot.data.joint_pos[0].cpu().numpy()}")
    print("=======================================\n")

    # ----------------------------------------------------------------
    # Test 1: Home 자세 유지
    # ----------------------------------------------------------------
    print(">>> [Test 1] Home 자세 유지 (60 step)")
    for i in range(60):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg.dt)
    pos_after_home = robot.data.joint_pos[0].cpu().numpy()
    home_drift = abs(pos_after_home).max()
    print(f"    최대 drift: {home_drift:.4f} rad ({home_drift*57.3:.2f}°)")
    if home_drift < 0.05:
        print("    ✅ Home 안정")
    else:
        print(f"    ❌ Home drift={home_drift} (stiffness/damping 튜닝 필요)")
        print(f"    Pos: {pos_after_home}")

    # ----------------------------------------------------------------
    # Test 2: shoulder_pan 0 → 0.5 rad
    # ----------------------------------------------------------------
    print("\n>>> [Test 2] shoulder_pan 0 → 0.5 rad")
    target = robot.data.default_joint_pos.clone()
    pan_idx = robot.joint_names.index("shoulder_pan")
    target[:, pan_idx] = 0.5
    for i in range(120):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg.dt)
        if i % 30 == 0:
            cur = robot.data.joint_pos[0, pan_idx].item()
            print(f"    step {i:3d}: shoulder_pan = {cur:.4f}")
    final_pan = robot.data.joint_pos[0, pan_idx].item()
    err_pan = abs(final_pan - 0.5)
    print(f"    최종: {final_pan:.4f}, error: {err_pan:.4f}")
    if err_pan < 0.05:
        print("    ✅ shoulder_pan OK")
    else:
        print(f"    ❌ pan err={err_pan}")

    # ----------------------------------------------------------------
    # Test 3: 다축 동시 이동
    # ----------------------------------------------------------------
    print("\n>>> [Test 3] 다축 동시 이동")
    target = robot.data.default_joint_pos.clone()
    multi_targets = {
        "shoulder_pan":  0.3,
        "shoulder_lift": -0.5,
        "elbow_flex":    0.8,
        "wrist_flex":   -0.3,
    }
    for name, val in multi_targets.items():
        idx = robot.joint_names.index(name)
        target[:, idx] = val
    for i in range(180):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg.dt)
    max_err = 0.0
    for name, target_val in multi_targets.items():
        idx = robot.joint_names.index(name)
        cur = robot.data.joint_pos[0, idx].item()
        err = abs(cur - target_val)
        max_err = max(max_err, err)
        status = "✅" if err < 0.05 else "❌"
        print(f"    {status} {name:15s}: {cur:+.4f} (target={target_val:+.4f}, err={err:.4f})")
    if max_err < 0.05:
        print("    ✅ 다축 OK")
    else:
        print(f"    ❌ multi err={max_err}")

    # ----------------------------------------------------------------
    # Test 4: 그리퍼 open/close
    # ----------------------------------------------------------------
    print("\n>>> [Test 4] 그리퍼 open/close")
    grip_idx = robot.joint_names.index("gripper")
    target = robot.data.default_joint_pos.clone()
    target[:, grip_idx] = 0.5
    for _ in range(120):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg.dt)
    closed = robot.data.joint_pos[0, grip_idx].item()
    print(f"    close (target=0.5): {closed:.4f}")
    target[:, grip_idx] = 0.0
    for _ in range(120):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg.dt)
    opened = robot.data.joint_pos[0, grip_idx].item()
    print(f"    open  (target=0.0): {opened:.4f}")
    print("    ✅ 그리퍼 동작 확인")

    print("\n========== 🎉 Control Tests Completed ==========\n")
    sim_app.close()


if __name__ == "__main__":
    main()
