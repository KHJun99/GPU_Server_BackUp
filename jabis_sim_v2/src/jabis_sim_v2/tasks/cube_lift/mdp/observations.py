"""Observation functions for cube lift."""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject, Articulation


def cube_position(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    """Cube position (xyz) in world frame."""
    cube: RigidObject = env.scene[asset_cfg.name]
    return cube.data.root_pos_w - env.scene.env_origins


def cube_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    """Cube linear velocity (xyz)."""
    cube: RigidObject = env.scene[asset_cfg.name]
    return cube.data.root_lin_vel_w


def joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint position relative to default (5 arm + 2 gripper)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_pos - robot.data.default_joint_pos


def joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_vel


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Last action taken."""
    return env.action_manager.action


def ee_position_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """End-effector position in base frame (env origin)."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_idx = robot.body_names.index("gripper_link")
    ee_pos_w = robot.data.body_pos_w[:, ee_idx]
    return ee_pos_w - env.scene.env_origins


def ee_to_cube(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Relative vector from ee to cube (base frame)."""
    robot: Articulation = env.scene["robot"]
    cube: RigidObject = env.scene["cube"]
    ee_idx = robot.body_names.index("gripper_link")
    ee_pos = robot.data.body_pos_w[:, ee_idx]
    cube_pos = cube.data.root_pos_w
    return cube_pos - ee_pos


def ee_to_cube_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distance from ee to cube (scalar, useful signal)."""
    rel = ee_to_cube(env)
    return rel.norm(dim=-1, keepdim=True)
