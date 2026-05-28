# -*- coding: utf-8 -*-
"""
SO-ARM101 + PincOpen Articulation Configuration for Isaac Lab v2.0.2
Jabis project — Day 3
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

USD_PATH = "/home/j-k14d101/jabis_sim/usd/so101_pincopen.usd"

SO101_PINCOPEN_CFG = ArticulationCfg(
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
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan":   0.0,
            "shoulder_lift":  0.0,
            "elbow_flex":     0.0,
            "wrist_flex":     0.0,
            "wrist_roll":     0.0,
            "gripper":        0.0,
            "left_proximal":  0.0,
            "right_proximal": 0.0,
            "left_distal":    0.0,
            "right_distal":   0.0,
        },
    ),
    actuators={
        "main_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan", "shoulder_lift",
                "elbow_flex", "wrist_flex", "wrist_roll",
            ],
            effort_limit=50.0,
            velocity_limit=10.0,
            stiffness=200.0,
            damping=20.0,
        ),
        "gripper_main": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit=10.0,
            velocity_limit=5.0,
            stiffness=100.0,
            damping=10.0,
        ),
        "gripper_linkage": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_proximal", "right_proximal",
                "left_distal", "right_distal",
            ],
            effort_limit=5.0,
            velocity_limit=5.0,
            stiffness=50.0,
            damping=5.0,
        ),
    },
)
