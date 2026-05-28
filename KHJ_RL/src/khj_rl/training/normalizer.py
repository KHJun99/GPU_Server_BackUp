"""Frozen observation normalizer.

CLAUDE.md ## Phase 1 결정 사항 7: the obs mean/std used by PPO must be
the same one computed from the BC demonstrations, with **no running mean
update** during PPO. Reason: running normalization shifts the obs
distribution while PPO is fine-tuning, and the BC-pretrained actor was
trained against a fixed distribution — letting the distribution drift
silently destroys the BC weights in a few iterations.

This wrapper computes mean/std once (from the BC dataset) and then is
read-only for the rest of training. Saved/loaded alongside the network
checkpoint so the rollout time, BC time, and PPO time all see the same
obs statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class ObsNormalizer:
    """``(obs - mean) / std`` with frozen statistics."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        if mean.shape != std.shape:
            raise ValueError(f"mean/std shape mismatch: {mean.shape} vs {std.shape}")
        # std clip avoids divide-by-zero on constant-valued obs components
        # (e.g. normalized_t at episode start, or a degenerate seed).
        self._mean = mean
        self._std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    @classmethod
    def fit(cls, obs: np.ndarray) -> "ObsNormalizer":
        """Compute mean/std from a (N, obs_dim) batch of observations."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 2:
            raise ValueError(f"expected (N, D) obs, got shape {obs.shape}")
        mean = obs.mean(axis=0)
        std = obs.std(axis=0)
        return cls(mean=mean, std=std)

    def normalize_np(self, obs: np.ndarray) -> np.ndarray:
        return ((obs - self._mean) / self._std).astype(np.float32)

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        mean_t = torch.as_tensor(self._mean, device=obs.device, dtype=obs.dtype)
        std_t = torch.as_tensor(self._std, device=obs.device, dtype=obs.dtype)
        return (obs - mean_t) / std_t

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def std(self) -> np.ndarray:
        return self._std

    # ---- persistence -----------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # JSON sidecar — small, human-readable, easy to diff across runs.
        with path.open("w") as f:
            json.dump(
                {"mean": self._mean.tolist(), "std": self._std.tolist()},
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> "ObsNormalizer":
        with Path(path).open("r") as f:
            payload = json.load(f)
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )
