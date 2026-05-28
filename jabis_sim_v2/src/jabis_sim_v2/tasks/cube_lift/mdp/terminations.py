"""Termination functions."""
import torch
from isaaclab.envs import ManagerBasedRLEnv


def time_out_term(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check episode length and return True if exceeded."""
    max_length = int(env.max_episode_length)
    return env.episode_length_buf >= max_length
