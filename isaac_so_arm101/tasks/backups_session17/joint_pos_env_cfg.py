# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from isaaclab.assets import RigidObjectCfg

# from isaaclab.managers NotImplementedError
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg, RigidBodyMaterialCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, CollisionPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaac_so_arm101.robots import SO_ARM100_CFG, SO_ARM101_CFG  # noqa: F401
from isaac_so_arm101.tasks.lift.lift_env_cfg import LiftEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class SoArm100LiftCubeEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set so arm as robot
        self.scene.robot = SO_ARM100_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_proximal", "right_proximal"],
            open_command_expr={"left_proximal": 0.0, "right_proximal": 0.0},
            close_command_expr={"left_proximal": -0.6, "right_proximal": 0.6},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = ["gripper"]

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, 0.015], rot=[1, 0, 0, 0]),
            spawn=CuboidCfg(
                size=(0.05, 0.05, 0.05),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=0.15),
                collision_props=CollisionPropertiesCfg(),
                visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, -0.09, 0.01],
                    ),
                ),
            ],
        )


@configclass
class SoArm100LiftCubeEnvCfg_PLAY(SoArm100LiftCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101LiftCubeEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set so arm as robot
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        # v7 ablation: scale 0.5 → 1.5 (default ±1.5 rad, action space 3배 확장)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=1.5,
            use_default_offset=True,
        )
        # PincOpen: left_proximal + right_proximal driven (codex 권고: distal/gripper 는 4-bar 자연 추종에 맡김).
        # right_proximal 부호 +0.6: URDF mimic multiplier=-1 따라 close 방향이 반대.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_proximal", "right_proximal"],
            open_command_expr={"left_proximal": 0.0, "right_proximal": 0.0},
            close_command_expr={"left_proximal": -0.6, "right_proximal": 0.6},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = ["gripper_link"]

        # Set Cube as object — v4: 3cm cube + friction 1.5 (5cm/default friction에서 위누름 자세 정체 → 측면 그립 강제)
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, 0.012], rot=[1, 0, 0, 0]),
            spawn=CuboidCfg(
                size=(0.03, 0.03, 0.03),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=0.15),
                collision_props=CollisionPropertiesCfg(),
                physics_material=RigidBodyMaterialCfg(
                    static_friction=1.5,
                    dynamic_friction=1.5,
                    restitution=0.0,
                ),
                visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.01, 0.0, -0.09],
                    ),
                ),
            ],
        )


@configclass
class SoArm101LiftCubeEnvCfg_PLAY(SoArm101LiftCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101LiftCubeEnvCfg_VIDEO(SoArm101LiftCubeEnvCfg_PLAY):
    """Video-recording variant — session 10: top + diag cameras (2 views)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        from isaaclab.sensors import TiledCameraCfg
        import isaaclab.sim as sim_utils

        # View 1: top-down (book 위 1.2m, OpenGL convention forward -Z)
        # OpenGL: forward -Z, up +Y. Identity rot → forward=-Z (down) ✓
        self.scene.top_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/top_camera",
            update_period=0.0,
            data_types=["rgb"],
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.2, 0.0, 1.2),
                rot=(1.0, 0.0, 0.0, 0.0),  # identity → -Z forward (top-down)
                convention="opengl",
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 10.0),
            ),
            width=640,
            height=480,
        )

        # View 2: diag_front (대각선 우측 위, look_at (0.2, 0, 0.05))
        # quaternion 추정: pos→target 방향 + up=(0,0,1). 첫 frame 검증 후 조정 가능.
        # OpenGL forward=-Z + up=+Y. forward 를 (-0.5,-0.5,-0.45) 정규화로.
        # 대략적 quat (w,x,y,z) — 첫 frame test 후 조정
        self.scene.diag_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/diag_camera",
            update_period=0.0,
            data_types=["rgb"],
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.7, 0.5, 0.5),
                # look_at (0.2, 0, 0.05) — numpy R→quat computed (world up=+Z)
                rot=(0.3354, 0.1841, 0.4446, 0.8099),
                convention="opengl",
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 10.0),
            ),
            width=640,
            height=480,
        )
