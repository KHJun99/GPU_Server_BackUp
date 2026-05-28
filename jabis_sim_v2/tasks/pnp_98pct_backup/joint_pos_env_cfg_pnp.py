"""Concrete cube lift env: SO-ARM101 (one arm) + JointPosition action."""
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

        # robot (with stronger gripper PD for cube grip)
        robot_cfg = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override gripper stiffness — original (60, 20) too soft for cube grip
        # mimic joints (distal) also strengthened
        robot_cfg.actuators["gripper"].stiffness = 500.0
        robot_cfg.actuators["gripper"].damping = 80.0
        robot_cfg.actuators["finger_distal"].stiffness = 50.0
        robot_cfg.actuators["finger_distal"].damping = 10.0
        self.scene.robot = robot_cfg

        # action space (v1 lesson: arm scale 1.5)
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
    """Rewards for Pick & Place task."""
    reach = RewTerm(func=mdp.rewards_pnp.reach_cube, weight=1.0, params={"std": 0.1})
    lifted = RewTerm(func=mdp.rewards_pnp.cube_lifted_above, weight=10.0, params={"threshold": 0.05})
    to_goal = RewTerm(func=mdp.rewards_pnp.cube_to_goal_distance, weight=15.0, params={"std": 0.15})
    placed = RewTerm(func=mdp.rewards_pnp.cube_at_goal, weight=300.0, params={"threshold": 0.05})
    drop = RewTerm(func=mdp.rewards_pnp.cube_dropped, weight=-3.0)
    action_rate = RewTerm(func=mdp.rewards_pnp.action_rate_penalty, weight=-0.05)


@configclass
class SoArm101PickPlaceEnvCfg(SoArm101CubeLiftEnvCfg):
    """Pick & Place: pick cube → transport → place at fixed target (0.35, 0.15, 0.05)."""

    rewards: PnPRewardsCfg = PnPRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 5.0  # 300 step


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
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.35, 0.15, 0.0025]),
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
