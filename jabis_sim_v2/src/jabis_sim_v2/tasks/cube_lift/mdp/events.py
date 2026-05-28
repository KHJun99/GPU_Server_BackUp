"""Event (reset) functions for cube lift."""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject, Articulation


def reset_cube_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
):
    """Reset cube to random position within pose_range."""
    cube: RigidObject = env.scene[asset_cfg.name]

    # default state
    root_state = cube.data.default_root_state[env_ids].clone()

    # random offset within range
    n = len(env_ids)
    for i, key in enumerate(["x", "y", "z"]):
        low, high = pose_range.get(key, (0.0, 0.0))
        root_state[:, i] += torch.empty(n, device=env.device).uniform_(low, high)

    # add env origin
    root_state[:, :3] += env.scene.env_origins[env_ids]

    cube.write_root_state_to_sim(root_state, env_ids=env_ids)


def reset_cube_grid(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    x_vals: list,
    y_vals: list,
    envs_per_point: int,
    z_init: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
):
    """Deterministic grid spawn for reach-map sweeps."""
    cube: RigidObject = env.scene[asset_cfg.name]
    root_state = cube.data.default_root_state[env_ids].clone()

    n_y = len(y_vals)
    n_pts = len(x_vals) * n_y

    env_ids_int = env_ids.long()
    grid_idx = (env_ids_int // envs_per_point) % n_pts
    grid_x_idx = grid_idx // n_y
    grid_y_idx = grid_idx % n_y

    x_t = torch.tensor(x_vals, device=env.device, dtype=root_state.dtype)[grid_x_idx]
    y_t = torch.tensor(y_vals, device=env.device, dtype=root_state.dtype)[grid_y_idx]

    root_state[:, 0] = x_t
    root_state[:, 1] = y_t
    root_state[:, 2] = z_init
    root_state[:, :3] += env.scene.env_origins[env_ids]

    cube.write_root_state_to_sim(root_state, env_ids=env_ids)


def reset_joint_default(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robot joints to default position."""
    robot: Articulation = env.scene[asset_cfg.name]
    default_pos = robot.data.default_joint_pos[env_ids]
    default_vel = robot.data.default_joint_vel[env_ids]
    robot.write_joint_state_to_sim(default_pos, default_vel, env_ids=env_ids)
