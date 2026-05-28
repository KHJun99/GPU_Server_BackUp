# -*- coding: utf-8 -*-
"""Day 4 — Scene 로드 + 안정성 검증
SO-ARM101 + Floor + Desk + Cup, 300 step 시뮬 후 안정성 측정.
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
    # ----------------------------------------------------------------
    # Step 1: SimulationContext + 카메라
    # ----------------------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(sim_cfg)
    # 카메라 시점: 책상 위 컵 영역에 초점
    sim.set_camera_view(eye=[1.2, 1.2, 0.8], target=[0.25, 0.05, 0.08])

    # ----------------------------------------------------------------
    # Step 2: Light (Day 3 그대로)
    # ----------------------------------------------------------------
    cfg_light = sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0))
    cfg_light.func("/World/Light", cfg_light)

    # ----------------------------------------------------------------
    # Step 3: Floor (Day 3 그대로) — z=-0.05, top=z=0
    # ----------------------------------------------------------------
    cfg_floor = sim_utils.CuboidCfg(
        size=(10.0, 10.0, 0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
    )
    cfg_floor.func("/World/Floor", cfg_floor, translation=(0.0, 0.0, -0.05))

    # ----------------------------------------------------------------
    # Step 4: Desk (NEW) — Cuboid kinematic 0.40 × 0.30 × 0.02
    #   center z=0.04 → top면 z=0.05 (robot base와 같은 평면)
    # ----------------------------------------------------------------
    cfg_desk = sim_utils.CuboidCfg(
        size=(0.40, 0.30, 0.02),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),  # 갈색
    )
    cfg_desk.func("/World/Desk", cfg_desk, translation=(0.25, 0.0, 0.04))

    # ----------------------------------------------------------------
    # Step 5: Robot (Day 3 그대로) — base z=0.05
    # ----------------------------------------------------------------
    robot_cfg = SO101_PINCOPEN_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg=robot_cfg)

    # ----------------------------------------------------------------
    # Step 6: Cup (NEW) — RigidObject Cylinder
    #   r=0.025, h=0.06 → 컵 base z = center_z - 0.03 = 0.05 (desk top)
    #   추적 가능하도록 RigidObject 인스턴스 생성
    # ----------------------------------------------------------------
    cup_cfg = RigidObjectCfg(
        prim_path="/World/Cup",
        spawn=sim_utils.CylinderCfg(
            radius=0.025,
            height=0.06,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),  # 빨강
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.05, 0.08)),
    )
    cup = RigidObject(cfg=cup_cfg)

    # ----------------------------------------------------------------
    # Step 7: sim.reset() — articulation/rigid object 활성화
    # ----------------------------------------------------------------
    sim.reset()

    print("\n========== Scene Info ==========")
    print(f"Robot DOF      : {robot.num_joints}")
    print(f"Robot bodies   : {robot.num_bodies}")
    print(f"Robot joints   : {robot.joint_names}")
    print(f"Cup init pos   : {cup.data.root_pos_w[0].cpu().numpy()}")
    print(f"Cup init z     : {cup.data.root_pos_w[0, 2].item():.4f} m")
    print("================================\n")

    # ----------------------------------------------------------------
    # Step 8: 300 step 안정성 시뮬 (robot home 유지하면서 cup 추적)
    # ----------------------------------------------------------------
    print(">>> 300 step 안정성 시뮬")
    cup_init = cup.data.root_pos_w[0].clone()

    for i in range(300):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg.dt)
        cup.update(sim_cfg.dt)
        if i % 50 == 0:
            cup_z = cup.data.root_pos_w[0, 2].item()
            print(f"    step {i:3d}: cup z = {cup_z:.4f}")

    # ----------------------------------------------------------------
    # Step 9: 결과 측정 + 합격선 검사
    # ----------------------------------------------------------------
    cup_final = cup.data.root_pos_w[0].cpu().numpy()
    cup_z_drift = abs(cup_final[2] - cup_init[2].item())
    cup_xy_drift = (
        (cup_final[0] - cup_init[0].item()) ** 2
        + (cup_final[1] - cup_init[1].item()) ** 2
    ) ** 0.5

    robot_pos = robot.data.joint_pos[0].cpu().numpy()
    robot_drift = abs(robot_pos).max()

    print("\n========== Day 4 Step 4-4 결과 ==========")
    print(f"Cup z drift  : {cup_z_drift*1000:.2f} mm   (합격선 < 5mm)")
    print(f"Cup xy drift : {cup_xy_drift*1000:.2f} mm  (합격선 < 5mm)")
    print(f"Robot drift  : {robot_drift:.4f} rad ({robot_drift*57.3:.2f}°, 합격선 < 0.05 rad ≈ 2.86°)")

    pass_cup_z  = cup_z_drift  < 0.005
    pass_cup_xy = cup_xy_drift < 0.005
    pass_robot  = robot_drift  < 0.05

    print(f"\nCup z   : {'✅' if pass_cup_z  else '❌'}")
    print(f"Cup xy  : {'✅' if pass_cup_xy else '❌'}")
    print(f"Robot   : {'✅' if pass_robot  else '❌'}")

    if pass_cup_z and pass_cup_xy and pass_robot:
        print("\n🎉 Day 4 Step 4-4 PASS — Step 4-5 reach 테스트 진입 가능")
    else:
        print("\n⚠️  실패 — 자산 좌표/물리 속성 조정 필요 (보고 양식 §실패 분기 참고)")
    print("=========================================\n")

    sim_app.close()


if __name__ == "__main__":
    main()