"""Cube lift task registration."""
import gymnasium as gym

gym.register(
    id="Jabis-V2-CubeLift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101CubeLiftEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="Jabis-V2-CubeLift-Video-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101CubeLiftEnvCfg_VIDEO"},
    disable_env_checker=True,
)

gym.register(
    id="Jabis-V2-CubePickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="Jabis-V2-CubePickPlace-Video-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceEnvCfg_VIDEO"},
    disable_env_checker=True,
)
