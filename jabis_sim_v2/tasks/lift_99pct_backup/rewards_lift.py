"""Reward functions for cube lift (v2 — anti-shaping-trap)."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject, Articulation


def cube_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    """Lift reward — ONLY when cube z > 0.04 (avoid hold-only local optimum).

    Previously: always rewarded cube z, so policy learned to hold cube at low z forever.
    Now: zero reward until cube clears threshold, then proportional.
    """
    cube: RigidObject = env.scene[asset_cfg.name]
    z = cube.data.root_pos_w[:, 2]
    return torch.clamp(z - 0.04, min=0.0)  # 0 if z<0.04, else (z-0.04)


def reach_cube(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Distance reward: end-effector → cube."""
    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_pos = robot.data.body_pos_w[:, -1]
    cube_pos = cube.data.root_pos_w
    dist = torch.norm(ee_pos - cube_pos, dim=-1)
    return 1.0 - torch.tanh(dist / std)


def cube_lifted(
    env: ManagerBasedRLEnv,
    threshold: float = 0.10,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Sparse success reward."""
    cube: RigidObject = env.scene[cube_cfg.name]
    return (cube.data.root_pos_w[:, 2] > threshold).float()


def cube_dropped(
    env: ManagerBasedRLEnv,
    drop_threshold: float = 0.01,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Penalty: cube below initial height + threshold (= dropped)."""
    cube: RigidObject = env.scene[cube_cfg.name]
    # cube initial height ~ 0.04m (on desk + half size). Dropped if z < 0.03
    return (cube.data.root_pos_w[:, 2] < 0.03).float()


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action changes."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1)
