"""Concrete cube lift env: SO-ARM101 (one arm) + JointPosition action."""
from isaaclab.assets import ArticulationCfg
from isaaclab.envs.mdp import JointPositionActionCfg, BinaryJointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm

from . import mdp

from isaac_so_arm101.robots import SO_ARM101_CFG  # v1 robot asset 재활용 (코드 X)

from .env_cfg import CubeLiftEnvCfg


@configclass
class SoArm101CubeLiftEnvCfg(CubeLiftEnvCfg):
    """Concrete cube lift env with SO-ARM101."""

    def __post_init__(self):
        super().__post_init__()

        # baseline rest pose — 사용자 제공 home 각도 (deg → rad 변환).
        # shoulder_lift, elbow_flex 는 URDF hard limit 초과로 clamp.
        #   shoulder_pan  +92.57° → +1.6157 (limit ±1.91986)
        #   shoulder_lift -109.27° → -1.9075 → -1.745 (URDF hard limit ±1.745, clamp)
        #   elbow_flex    +106.64° → +1.8612 → +1.69 (URDF hard limit ±1.69, clamp)
        #   wrist_flex    +71.12° → +1.2413
        #   wrist_roll    +2.86° → +0.0499
        #   gripper       30.12° → +0.5258
        # baseline rest pose — 시연환경 follower_rest_pose.json 값을 sim radian 으로
        # 변환. shoulder_lift, elbow_flex 는 URDF hard limit 으로 미세 clamp.
        # 사진 (sim_match_check/그리퍼 자세.jpg) 의 "Z" 모양 접힌 자세 매칭.
        # Ready pose: shoulder_pan 0 (정면 +x, action 양방향 균등).
        # Power-on 자세 (Z-fold)는 실물 환경에만 — sim은 ready 부터 시작 (RL/Oracle 학습 phase).
        baseline_joint_pos = {
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.8,
            "elbow_flex": 1.0,
            "wrist_flex": 1.2413,
            "wrist_roll": 0.0499,
            "gripper": 0.5258,
            "left_proximal": 0.0,
        }

        # robot (with stronger gripper PD for cube grip)
        # base init_rot = identity (양팔 마주봄 layout: robot 의 +x 가 sim +x).
        # baseline shoulder_pan +1.6157 (90°) 으로 회전 → forward 사용자 반대쪽.
        robot_cfg = SO_ARM101_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos=baseline_joint_pos,
                joint_vel={".*": 0.0},
            ),
        )
        robot_cfg.init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=baseline_joint_pos,
            joint_vel={".*": 0.0},
        )
        # gripper stiffness 강화 (cube grip 위해) — arm 은 SO_ARM101_CFG default.
        robot_cfg.actuators["gripper"].stiffness = 500.0
        robot_cfg.actuators["gripper"].damping = 80.0
        robot_cfg.actuators["finger_distal"].stiffness = 50.0
        robot_cfg.actuators["finger_distal"].damping = 10.0
        # arm actuator 는 SO_ARM101_CFG default 유지 (실물 매칭) — override 제거.
        self.scene.robot = robot_cfg

        # mirror_arm — 시연환경 왼팔 (visual-only).
        # base init_rot z=π (마주봄 layout). 양팔 동일 baseline_joint_pos.
        mirror_cfg = SO_ARM101_CFG.replace(
            prim_path="{ENV_REGEX_NS}/MirrorArm",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(1.10, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
                joint_pos=baseline_joint_pos,
                joint_vel={".*": 0.0},
            ),
        )
        self.scene.mirror_arm = mirror_cfg

        # action space — scale 1.5 (실물 매칭). baseline shoulder_pan 0 이라 양방향 ±1.5 균등.
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=1.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_proximal", "left_distal", "right_proximal", "right_distal"],  # no mimic, all 4 explicit
            open_command_expr={
                "left_proximal": 0.0, "left_distal": 0.0,
                "right_proximal": 0.0, "right_distal": 0.0,
            },
            close_command_expr={
                "left_proximal": -0.8, "left_distal": +0.8,
                "right_proximal": +0.8, "right_distal": -0.8,
            },  # all 4 fingers explicit (no mimic)
        )



@configclass
class OracleCubeLiftEnvCfg(SoArm101CubeLiftEnvCfg):
    """Same env but with arm_action scale=1.0 for Oracle demo collection.

    Reason: Oracle outputs absolute joint targets via IK.
    Training/eval uses scale=1.5 (v1 lesson).
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.scale = 1.0



@configclass
class SoArm101CubeLiftEnvCfg_VIDEO(SoArm101CubeLiftEnvCfg):
    """Video-recording variant — single env, top + diag cameras."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1

        from isaaclab.sensors import TiledCameraCfg
        import isaaclab.sim as sim_utils

        self.scene.top_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/top_camera",
            update_period=0.0,
            data_types=["rgb"],
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.2, 0.0, 1.2),
                rot=(1.0, 0.0, 0.0, 0.0),  # identity → OpenGL -Z forward (top-down)
                convention="opengl",
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 10.0),
            ),
            width=1280,
            height=720,
        )

        self.scene.diag_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/diag_camera",
            update_period=0.0,
            data_types=["rgb"],
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.9, 0.6, 0.75),
                rot=(0.389165, 0.220342, 0.440683, 0.77833),  # look_at (0.1, 0, 0.15), up=+Z
                convention="opengl",
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=14.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 10.0),
            ),
            width=1280,
            height=720,
        )


@configclass
class PnPRewardsCfg:
    """PnP rewards (B) — dense reward 강화, placed sparse 약화."""
    reach = RewTerm(func=mdp.rewards_pnp.reach_cube, weight=1.0, params={"std": 0.1})
    lifted = RewTerm(func=mdp.rewards_pnp.cube_lifted_above, weight=10.0, params={"threshold": 0.07})
    lift_height = RewTerm(func=mdp.rewards_pnp.cube_lift_height, weight=30.0, params={"threshold": 0.07})
    to_goal = RewTerm(func=mdp.rewards_pnp.cube_to_goal_distance, weight=30.0, params={"std": 0.20})
    placed = RewTerm(func=mdp.rewards_pnp.cube_at_goal, weight=100.0, params={"threshold": 0.05})
    drop = RewTerm(func=mdp.rewards_pnp.cube_dropped, weight=-3.0)
    action_rate = RewTerm(func=mdp.rewards_pnp.action_rate_penalty, weight=-0.01)


@configclass
class SoArm101PickPlaceEnvCfg(SoArm101CubeLiftEnvCfg):
    """Pick & Place: pick cube → transport → place at fixed target (0.35, 0.15, 0.05)."""

    rewards: PnPRewardsCfg = PnPRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 8.0  # 300 step


@configclass
class SoArm101PickPlaceEnvCfg_VIDEO(SoArm101PickPlaceEnvCfg):
    """Video-recording PnP variant — single env, top + diag cameras, goal marker."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1

        from isaaclab.sensors import TiledCameraCfg
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.sim.spawners.shapes import CuboidCfg

        # visual-only flat pad at PnP target (0.35, 0.15, 0.05)
        self.scene.goal_marker = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/goal_marker",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.80, 0.0, 0.0025]),
            spawn=CuboidCfg(
                size=(0.06, 0.06, 0.005),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.85, 0.2)),
            ),
        )

        self.scene.top_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/top_camera",
            update_period=0.0,
            data_types=["rgb"],
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.25, 0.08, 1.2),
                rot=(1.0, 0.0, 0.0, 0.0),  # identity → OpenGL -Z forward (top-down)
                convention="opengl",
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 10.0),
            ),
            width=1280,
            height=720,
        )

        self.scene.diag_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/diag_camera",
            update_period=0.0,
            data_types=["rgb"],
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.9, 0.6, 0.75),
                rot=(0.389165, 0.220342, 0.440683, 0.77833),  # look_at (0.1, 0, 0.15), up=+Z
                convention="opengl",
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=14.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 10.0),
            ),
            width=1280,
            height=720,
        )
