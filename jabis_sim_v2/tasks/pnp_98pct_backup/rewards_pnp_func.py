"""Reward functions for Pick & Place task."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject, Articulation


# Fixed target location (책상 위, local frame)
TARGET_POS = torch.tensor([0.35, 0.15, 0.05])


def reach_cube(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Phase 1: ee → cube. Always-on."""
    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_pos = robot.data.body_pos_w[:, -1]
    cube_pos = cube.data.root_pos_w
    dist = torch.norm(ee_pos - cube_pos, dim=-1)
    return 1.0 - torch.tanh(dist / std)


def cube_lifted_above(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Phase 2: cube z > threshold (lifted)."""
    cube: RigidObject = env.scene[cube_cfg.name]
    env_origins_z = env.scene.env_origins[:, 2]
    cube_z = cube.data.root_pos_w[:, 2] - env_origins_z
    return (cube_z > threshold).float()


def cube_to_goal_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.15,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Phase 3: cube → goal. Only active when cube is lifted."""
    cube: RigidObject = env.scene[cube_cfg.name]
    env_origins = env.scene.env_origins
    cube_local = cube.data.root_pos_w - env_origins

    target = TARGET_POS.to(cube_local.device)
    dist = torch.norm(cube_local - target, dim=-1)

    lifted_mask = (cube_local[:, 2] > 0.04).float()
    return lifted_mask * (1.0 - torch.tanh(dist / std))


def cube_at_goal(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Phase 4: cube placed at goal — sparse success.

    Cube within threshold of target AND near ground (placed).
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    env_origins = env.scene.env_origins
    cube_local = cube.data.root_pos_w - env_origins

    target = TARGET_POS.to(cube_local.device)
    dist = torch.norm(cube_local - target, dim=-1)

    near_target = (dist < threshold).float()
    near_ground = (cube_local[:, 2] < 0.08).float()
    return near_target * near_ground


def cube_dropped(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Penalty: cube fell to floor."""
    cube: RigidObject = env.scene[cube_cfg.name]
    env_origins_z = env.scene.env_origins[:, 2]
    cube_z = cube.data.root_pos_w[:, 2] - env_origins_z
    return (cube_z < 0.03).float()


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Smoothness penalty."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1)
