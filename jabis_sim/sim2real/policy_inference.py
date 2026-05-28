"""rsl_rl ActorCritic policy inference wrapper (PyTorch only, IsaacLab-free)."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SoArm101LiftPolicy:
    """Deterministic actor-only inference wrapper for SO-ARM101 lift policy."""

    OBS_DIM: int = 36
    ACTION_DIM: int = 6
    HIDDEN_DIMS: tuple[int, ...] = (256, 128, 64)

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.device: torch.device = torch.device(device)
        self.actor: nn.Sequential = self._build_actor()
        self._load_weights(Path(checkpoint_path))
        self.actor.to(self.device).eval()
        logger.info(
            "SoArm101LiftPolicy ready: obs_dim=%d action_dim=%d hidden=%s device=%s",
            self.OBS_DIM, self.ACTION_DIM, self.HIDDEN_DIMS, self.device,
        )

    def _build_actor(self) -> nn.Sequential:
        dims: list[int] = [self.OBS_DIM, *self.HIDDEN_DIMS, self.ACTION_DIM]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ELU())
        return nn.Sequential(*layers)

    def _load_weights(self, ckpt_path: Path) -> None:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if "model_state_dict" not in ckpt:
            raise KeyError(
                f"checkpoint missing 'model_state_dict' (got keys: {list(ckpt.keys())})"
            )
        full_sd = ckpt["model_state_dict"]
        prefix = "actor."
        actor_sd = {
            k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)
        }
        if not actor_sd:
            raise RuntimeError("no 'actor.*' keys found in checkpoint state_dict")
        try:
            self.actor.load_state_dict(actor_sd, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"actor state_dict mismatch (built {self.HIDDEN_DIMS} MLP, "
                f"checkpoint had keys {sorted(actor_sd.keys())}): {exc}"
            ) from exc
        logger.info(
            "loaded actor weights from %s (iter=%s, %d tensors)",
            ckpt_path, ckpt.get("iter", "?"), len(actor_sd),
        )

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            raise TypeError(f"obs must be torch.Tensor, got {type(obs).__name__}")
        if not torch.isfinite(obs).all():
            raise ValueError("obs contains NaN/Inf — refusing to run policy")
        if obs.dim() == 1:
            if obs.shape[0] != self.OBS_DIM:
                raise ValueError(f"obs shape {tuple(obs.shape)} != ({self.OBS_DIM},)")
            obs = obs.unsqueeze(0)
            squeeze = True
        elif obs.dim() == 2:
            if obs.shape[-1] != self.OBS_DIM:
                raise ValueError(
                    f"obs last dim {obs.shape[-1]} != {self.OBS_DIM}"
                )
            squeeze = False
        else:
            raise ValueError(f"obs must be 1D or 2D, got {obs.dim()}D")

        obs_in = obs.to(device=self.device, dtype=torch.float32)
        action = self.actor(obs_in)
        if action.shape[-1] != self.ACTION_DIM:
            raise RuntimeError(
                f"actor output last dim {action.shape[-1]} != {self.ACTION_DIM}"
            )
        return action.squeeze(0) if squeeze else action
