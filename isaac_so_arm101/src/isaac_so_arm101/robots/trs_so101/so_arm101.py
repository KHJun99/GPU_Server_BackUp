"""SO-ARM101 + PincOpen Articulation Configuration
Jabis project - PincOpen 통합 버전 (Day 5)

Reconstructed at session 7 start from tasks/lessons.md §6.1 patch unit
(UsdFileCfg using mimic-converted USD with 3 actuator groups). Pre-fix backup at
so_arm101.py.session7_pre.
"""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

USD_PATH = "/home/j-k14d101/jabis_sim/usd/so101_pincopen.usd"

SO_ARM101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 1.57,
            "wrist_roll": 0.0,
            "gripper": 0.0,
            "left_proximal": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
            effort_limit_sim=2.5,
            velocity_limit_sim=1.5,
            stiffness={
                "shoulder_pan": 200.0,
                "shoulder_lift": 170.0,
                "elbow_flex": 120.0,
                "wrist_flex": 80.0,
                "wrist_roll": 50.0,
            },
            damping={
                "shoulder_pan": 80.0,
                "shoulder_lift": 65.0,
                "elbow_flex": 45.0,
                "wrist_flex": 30.0,
                "wrist_roll": 20.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_proximal", "right_proximal"],
            effort_limit_sim=2.5,
            velocity_limit_sim=1.5,
            stiffness=60.0,
            damping=20.0,
        ),
        "finger_distal": ImplicitActuatorCfg(
            joint_names_expr=["left_distal", "right_distal"],
            effort_limit_sim=2.5,
            velocity_limit_sim=1.5,
            stiffness=5.0,
            damping=2.0,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
