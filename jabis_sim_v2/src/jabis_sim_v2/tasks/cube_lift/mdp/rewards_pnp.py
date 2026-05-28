"""Reward functions for Pick & Place task — v2: lift high required."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject, Articulation


# Fixed target = trash bin center (env_cfg.py 의 trash_bin 위치)
TARGET_POS = torch.tensor([0.03, 0.245, 0.0])

# Lift threshold — cube 들어야 transport 시작
LIFT_HIGH_THRESHOLD = 0.07  # 7cm (3cm cube 안정 z=0.015 + 5.5cm lift)

# Trash bin volume (env_cfg.py 매칭)
BIN_X_MIN, BIN_X_MAX = -0.045, 0.105
BIN_Y_MIN, BIN_Y_MAX = 0.18, 0.31


def reach_cube(env, std=0.1,
               robot_cfg=SceneEntityCfg("robot"),
               cube_cfg=SceneEntityCfg("cube")):
    """Phase 1: ee → cube. Always-on."""
    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_pos = robot.data.body_pos_w[:, -1]
    cube_pos = cube.data.root_pos_w
    dist = torch.norm(ee_pos - cube_pos, dim=-1)
    return 1.0 - torch.tanh(dist / std)


def cube_lifted_above(env, threshold=LIFT_HIGH_THRESHOLD,
                      cube_cfg=SceneEntityCfg("cube")):
    """Phase 2 binary: cube z > 12cm — 진짜 높이 들었나."""
    cube: RigidObject = env.scene[cube_cfg.name]
    env_origins_z = env.scene.env_origins[:, 2]
    cube_z = cube.data.root_pos_w[:, 2] - env_origins_z
    return (cube_z > threshold).float()


def cube_lift_height(env, threshold=LIFT_HIGH_THRESHOLD,
                     cube_cfg=SceneEntityCfg("cube")):
    """Phase 2 proportional: cube z above threshold — 진짜 높이 비례 reward."""
    cube: RigidObject = env.scene[cube_cfg.name]
    env_origins_z = env.scene.env_origins[:, 2]
    cube_z = cube.data.root_pos_w[:, 2] - env_origins_z
    return torch.clamp(cube_z - threshold, min=0.0)


def cube_to_goal_distance(env, std=0.15,
                          cube_cfg=SceneEntityCfg("cube")):
    """Phase 3: cube → goal. Only when cube is lifted HIGH (z > 12cm)."""
    cube: RigidObject = env.scene[cube_cfg.name]
    env_origins = env.scene.env_origins
    cube_local = cube.data.root_pos_w - env_origins

    target = TARGET_POS.to(cube_local.device)
    dist = torch.norm(cube_local - target, dim=-1)

    # 진짜 변경 — 진짜 높이 들어야 to_goal reward
    lifted_high_mask = (cube_local[:, 2] > LIFT_HIGH_THRESHOLD).float()
    return lifted_high_mask * (1.0 - torch.tanh(dist / std))


def cube_at_goal(env, threshold=0.05,
                 cube_cfg=SceneEntityCfg("cube")):
    """Phase 4: cube in trash bin — sparse success.

    cube xy in bin volume AND cube z < 0 (책상 표면 아래 = bin 안 떨어짐).
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    env_origins = env.scene.env_origins
    cube_local = cube.data.root_pos_w - env_origins

    fx, fy, fz = cube_local[:, 0], cube_local[:, 1], cube_local[:, 2]
    in_bin_xy = (
        (fx >= BIN_X_MIN) & (fx <= BIN_X_MAX) & (fy >= BIN_Y_MIN) & (fy <= BIN_Y_MAX)
    )
    in_bin_z = fz < 0.0
    return (in_bin_xy & in_bin_z).float()


def cube_dropped(env, cube_cfg=SceneEntityCfg("cube")):
    """Penalty: cube가 책상/매트/bin 외부로 떨어짐."""
    cube: RigidObject = env.scene[cube_cfg.name]
    env_origins = env.scene.env_origins
    cube_local = cube.data.root_pos_w - env_origins
    fx, fy, fz = cube_local[:, 0], cube_local[:, 1], cube_local[:, 2]
    # bin 안 떨어진 건 OK
    in_bin_xy = (
        (fx >= BIN_X_MIN) & (fx <= BIN_X_MAX) & (fy >= BIN_Y_MIN) & (fy <= BIN_Y_MAX)
    )
    # 매트 영역 (대략) 안 떨어진 건 OK
    on_mat = (fx >= 0.0) & (fx <= 1.10) & (fy >= -0.37) & (fy <= 0.13) & (fz > -0.05)
    # 책상 아래로 떨어진 + bin/mat 외 = drop
    dropped = (fz < -0.20) & ~in_bin_xy
    return dropped.float()


def action_rate_penalty(env):
    """Smoothness penalty."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1)
