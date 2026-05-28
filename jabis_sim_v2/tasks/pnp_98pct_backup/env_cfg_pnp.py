"""Cube lift environment for SO-ARM101 (v2 RL track).

v1 12-session lessons:
- success_z 0.10 (training/eval 동일, Mode A 회피)
- mimic joint USD 수정본 사용 (v1 asset 참조)
- action scale 1.5 (v7+ 검증)
"""
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.utils import configclass

from . import mdp


# =============================================================================
# Scene
# =============================================================================

@configclass
class CubeLiftSceneCfg(InteractiveSceneCfg):
    """Scene: robot + cube + ground plane."""

    # robot: 하위 cfg에서 채움
    robot: ArticulationCfg = MISSING

    # cube (3cm, 측면 grip 유도)
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.24, 0.0, 0.05]),
        spawn=CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=3.0,
                dynamic_friction=3.0,
            ),
        ),
    )

    # ground
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )

    # light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


# =============================================================================
# MDP — Actions, Observations, Rewards, Terminations
# =============================================================================

@configclass
class ActionsCfg:
    """Action space: 5 arm joints + 1 gripper binary."""
    arm_action: object = MISSING
    gripper_action: object = MISSING


@configclass
class ObservationsCfg:
    """Observation: 36-dim (v1 호환)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """36-dim obs (v1 호환): joint(14) + cube(6) + action(7) + extras."""
        joint_pos = ObsTerm(func=mdp.observations.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.observations.joint_vel)
        cube_pos = ObsTerm(func=mdp.observations.cube_position)
        cube_vel = ObsTerm(func=mdp.observations.cube_velocity)
        actions = ObsTerm(func=mdp.observations.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms (v1 lessons: lift bonus 강하게, action_rate 작게)."""
    reach = RewTerm(func=mdp.rewards.reach_cube, weight=1.0, params={"std": 0.1})
    lift = RewTerm(func=mdp.rewards.cube_height, weight=5.0)  # ONLY z>0.04 (anti-trap)
    success = RewTerm(func=mdp.rewards.cube_lifted, weight=200.0, params={"threshold": 0.07})  # big bonus
    drop = RewTerm(func=mdp.rewards.cube_dropped, weight=-2.0)
    action_rate = RewTerm(func=mdp.rewards.action_rate_penalty, weight=-0.05)


@configclass
class TerminationsCfg:
    """Termination conditions."""
    time_out = DoneTerm(func=mdp.time_out_term, time_out=True)


@configclass
class EventsCfg:
    """Reset on each episode."""
    reset_cube = EventTerm(
        func=mdp.events.reset_cube_position,
        mode="reset",
        params={"pose_range": {"x": (-0.05, 0.10), "y": (-0.10, 0.10), "z": (0.0, 0.0)}},
    )
    reset_robot = EventTerm(func=mdp.events.reset_joint_default, mode="reset")


# =============================================================================
# Top-level cfg
# =============================================================================

@configclass
class CubeLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Base cube lift env (abstract, robot 미지정)."""

    # Scene
    scene: CubeLiftSceneCfg = CubeLiftSceneCfg(num_envs=1024, env_spacing=2.5)

    # MDP
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        # episode length 5초
        self.decimation = 2
        self.episode_length_s = 5.0

        # sim
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation

        # success threshold (v1 Mode A lesson)
        self.success_z = 0.10
