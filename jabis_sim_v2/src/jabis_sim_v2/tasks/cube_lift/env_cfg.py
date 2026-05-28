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
    """Scene: robot + cube + table + mat + ground plane.

    시연환경 매칭 (S14P31D101): SO-101 양팔이 책상 좌우 끝에서 마주봄.
    본 sim 은 right-arm 한 팔만 학습 — robot base 는 env-local (0,0,0) 에
    고정.

    좌표 매핑 (책상 앉은 사용자 기준, top-down image up = 사용자 가까운):
      - 사용자 오른쪽 = sim -x
      - 사용자 가까운 (사용자 쪽) = sim +y
      - 사용자 위쪽 (책상 안쪽) = sim -y
      - robot forward (sim +x) = 사용자 왼쪽 (양팔 마주봄)

    책상 specs:
      - 가로 1.20m (양팔 마주봄 방향, sim ±x)
      - 세로 0.60m (사용자 앞-뒤, sim ±y)
      - 높이 0.70m (상판 두께 0.04m + 다리 0.66m)
      - right-front corner = 사용자 시점 오른쪽 가까운 모서리 (sim x=-0.045, y=+0.18)
      - 상판 표면 z=-0.001 (매트보다 1mm 아래)
      - 다리 4개 visual-only, 검정

    매트 specs (양팔 사이 작업 영역):
      - 가로 1.10m, 세로 0.50m, 두께 1.6mm
      - 표면 z=0 + collision → cube 가 매트 위에 안착 (학습 분포 보존)
    """

    # robot: 하위 cfg에서 채움 (학습 대상)
    robot: ArticulationCfg = MISSING

    # mirror_arm: 양팔 마주봄 시연환경의 왼팔 — 순수 visual.
    # 학습 obs/action 에서 명시적으로 제외, default joint pos 에 stationary.
    mirror_arm: ArticulationCfg = MISSING

    # cube (5cm, sim 학습용 — 시연환경 3cm 와 augmentation 으로 추후 매칭).
    # init pos = sweet spot (work-ready ready pose, shoulder_pan=0)
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.18, 0.0, 0.05]),
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

    # table 상판 — corner 위치는 right-front (-0.045, +0.18) [사용자 시점 오른쪽 가까운],
    # left-back (+1.155, -0.42) [사용자 시점 왼쪽 먼]. center = (+0.555, -0.12).
    # 상판 두께 0.04m → 표면 z=-0.001 (매트 표면 z=0 보다 1mm 아래; z-fighting 회피).
    # 학습 분포의 cube 안착 surface 는 매트 (z=0). 매트 밖에서는 책상 상판이 받침.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.555, -0.12, -0.021]),
        spawn=CuboidCfg(
            size=(1.20, 0.60, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.675, 0.604, 0.475),  # RGB(172, 154, 121) 우드 톤
                roughness=0.7,
            ),
        ),
    )

    # table legs — 4 corner 안쪽 5cm. visual-only (collision 미부착).
    # 굵기 0.04 × 0.04, 길이 0.66m. top z=-0.04 (상판 bottom), bottom z=-0.70.
    # 명명: RF = right-front (사용자 시점 오른쪽 가까운 = robot 옆 사용자 쪽 다리).
    table_leg_rf = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegRF",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.005, 0.13, -0.37]),
        spawn=CuboidCfg(
            size=(0.04, 0.04, 0.66),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.05, 0.05), roughness=0.7,
            ),
        ),
    )
    table_leg_rb = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegRB",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.005, -0.37, -0.37]),
        spawn=CuboidCfg(
            size=(0.04, 0.04, 0.66),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.05, 0.05), roughness=0.7,
            ),
        ),
    )
    table_leg_lf = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegLF",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[1.105, 0.13, -0.37]),
        spawn=CuboidCfg(
            size=(0.04, 0.04, 0.66),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.05, 0.05), roughness=0.7,
            ),
        ),
    )
    table_leg_lb = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegLB",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[1.105, -0.37, -0.37]),
        spawn=CuboidCfg(
            size=(0.04, 0.04, 0.66),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.05, 0.05), roughness=0.7,
            ),
        ),
    )

    # mat — 양팔 사이 작업 영역. 가로 1.10m × 세로 0.50m × 두께 1.6mm.
    # 표면 (top) z=0 → 학습 분포 (cube spawn z=0.05) 와 정합. bottom z=-0.0016.
    # collision 부착 — cube 가 매트 위 z=0 에 안착 (학습 시 ground z=0 과 동일).
    # 매트 영역 밖 (책상 가장자리 부근) 으로 cube 가 갈 일이 없음 (cube spawn
    # range 가 매트 안에 fully 들어감).
    mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.55, -0.12, -0.0008]),
        spawn=CuboidCfg(
            size=(1.10, 0.50, 0.0016),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.651, 0.631, 0.616),  # RGB(166, 161, 157) 따뜻한 그레이
                roughness=0.9,
            ),
        ),
    )

    # trash_bin — hollow box, 4 walls + 1 bottom. PnP target (추후 사용).
    # 책상 가로 면 (사용자 가까운 라인 sim_y = +0.18) 외부에 매달림.
    # 외부 영역: sim_x [-0.045, +0.105], sim_y [+0.18, +0.31], sim_z [-0.13, 0].
    # 입구 size 15 × 13cm (가로 sim_x × 세로 sim_y), 깊이 13cm. 입구 z=0
    # → 책상/매트 표면과 평평. 한 모서리 = right-front corner (-0.045, +0.18).
    # 책상 right edge 에서 책상 사용자 가까운 라인을 따라 sim +x 방향으로 15cm 길이.
    # 벽 두께 3mm → 내부 14.4 × 12.4 × 12.7cm.
    # collision 부착 — PnP target. robot reach 안에서 cube drop 가능.
    # 색상: RGB(128, 164, 209).
    trash_bin_bottom = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TrashBinBottom",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.03, 0.245, -0.1285]),
        spawn=CuboidCfg(
            size=(0.15, 0.13, 0.003),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.502, 0.643, 0.820), roughness=0.5,
            ),
        ),
    )
    trash_bin_wall_xmin = AssetBaseCfg(  # 책상 right edge 와 인접한 벽 (sim_x = -0.045)
        prim_path="{ENV_REGEX_NS}/TrashBinWallXmin",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.0435, 0.245, -0.0635]),
        spawn=CuboidCfg(
            size=(0.003, 0.13, 0.127),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.502, 0.643, 0.820), roughness=0.5,
            ),
        ),
    )
    trash_bin_wall_xmax = AssetBaseCfg(  # 책상 안쪽 끝 벽 (sim_x = +0.105)
        prim_path="{ENV_REGEX_NS}/TrashBinWallXmax",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.1035, 0.245, -0.0635]),
        spawn=CuboidCfg(
            size=(0.003, 0.13, 0.127),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.502, 0.643, 0.820), roughness=0.5,
            ),
        ),
    )
    trash_bin_wall_ymin = AssetBaseCfg(  # 책상 가로 면과 인접한 벽 (sim_y = +0.18)
        prim_path="{ENV_REGEX_NS}/TrashBinWallYmin",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.03, 0.3085, -0.0635]),
        spawn=CuboidCfg(
            size=(0.144, 0.003, 0.127),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.502, 0.643, 0.820), roughness=0.5,
            ),
        ),
    )
    trash_bin_wall_ymax = AssetBaseCfg(  # 사용자 가까운 쪽 외부 끝 벽 (sim_y = +0.31)
        prim_path="{ENV_REGEX_NS}/TrashBinWallYmax",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.03, 0.1815, -0.0635]),
        spawn=CuboidCfg(
            size=(0.144, 0.003, 0.127),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.502, 0.643, 0.820), roughness=0.5,
            ),
        ),
    )

    # ground — 책상 전체 높이 0.70m 매칭하여 z=-0.70.
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.70]),
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
        params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.05, 0.05), "z": (0.0, 0.0)}},
    )
    reset_robot = EventTerm(func=mdp.events.reset_joint_default, mode="reset")
    # mirror_arm 도 매 에피소드 default 로 reset — visual stationary 보장.
    reset_mirror = EventTerm(
        func=mdp.events.reset_joint_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("mirror_arm")},
    )


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
