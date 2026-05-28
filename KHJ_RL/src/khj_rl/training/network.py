"""Shared actor-critic network for BC pretrain + PPO fine-tune.

CLAUDE.md ## Phase 1 결정 사항 7 mandates that BC and PPO use the *same*
``ActorCritic`` class so the BC-trained weights drop into PPO without a
shape mismatch. Keep this file dependency-free of trainer logic so both
``ppo.py`` and ``bc.py`` can import it without circular references.

Architecture: two-layer MLP trunk (tanh) → linear actor head producing
the Gaussian mean over the action (Method B post-M2: 4-D EE-delta+gripper.
Pre-M2: 6-D joint-delta), plus a separate linear critic head producing a
scalar value. ``actor_logstd`` is a global parameter (not state-conditional)
initialised to -1.0 so the initial exploration std ≈ 0.37 — small enough
that BC weights are not immediately destroyed by random actions when PPO
fine-tune starts. ``action_dim`` is passed in by the caller so the same
class trains both action layouts without modification.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _layer_init(layer: nn.Linear, std: float = 2**0.5, bias: float = 0.0) -> nn.Linear:
    """Orthogonal init — CleanRL default. Helps actor/critic both learn fast."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Continuous-action actor-critic with a fixed Gaussian policy std.

    Parameters
    ----------
    obs_dim : int
        Length of the normalized observation vector. Method B post-M2: 29-D.
        Pre-M2: 31-D (last_action 6→4 was the only change).
    action_dim : int
        Length of the normalized action vector. Method B post-M2: 4-D
        ([Δx, Δy, Δz, g]). Pre-M2: 6-D (5 joint-delta + 1 gripper).
    hidden_dim : int
        Width of each MLP hidden layer. Two layers by convention.
    actor_logstd_init : float
        Initial value of the learned ``log(std)``. -1.0 → std ≈ 0.37, which
        is small enough that PPO doesn't kick a BC-pretrained actor off
        the demo manifold on the first rollout (CLAUDE.md guidance).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        actor_logstd_init: float = -1.0,
    ) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            # Smaller std for the action head — CleanRL convention; keeps
            # initial outputs near zero so the un-trained actor doesn't
            # saturate the policy at the action limits.
            _layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_logstd = nn.Parameter(
            torch.full((action_dim,), float(actor_logstd_init))
        )

    def actor_mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(action, log_prob, entropy, value)``.

        If ``action`` is given, ``log_prob`` is computed for it (PPO update
        path); otherwise an action is sampled from the policy (rollout
        path). This matches the CleanRL ppo_continuous_action API one-to-one
        so anyone familiar with that reference can read this trainer.
        """
        mean = self.actor_mean(obs)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        # sum over action dims so log_prob is per-step, not per-component
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs)
        return action, log_prob, entropy, value


class ChunkedActorCritic(nn.Module):
    """ACT-style chunked actor + 1-step critic.

    The actor head outputs ``action_dim * chunk_size`` logits which reshape
    to ``(..., chunk_size, action_dim)``. ``actor_mean`` returns chunk[0] so
    callers expecting the 1-step ``ActorCritic`` API (PPO, eval) keep
    working without changes — the chunk dimension is opt-in via
    ``actor_chunks``. ``actor_logstd`` keeps shape ``(action_dim,)`` and is
    used only by PPO (BC training is L1/L2 deterministic).

    Codex review (2026-05-18) reflected:
      - A: ``actor_mean`` asserts the trailing shape == ``action_dim`` before
        returning, killing silent reshape errors.
      - I: ``chunk_size`` is captured as a buffer so checkpoint surgery can
        round-trip it.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int = 256,
        actor_logstd_init: float = -1.0,
    ) -> None:
        super().__init__()
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self._action_dim = int(action_dim)
        self._chunk_size = int(chunk_size)

        self.actor = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(
                nn.Linear(hidden_dim, action_dim * self._chunk_size), std=0.01
            ),
        )
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_logstd = nn.Parameter(
            torch.full((action_dim,), float(actor_logstd_init))
        )

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def actor_chunks(self, obs: torch.Tensor) -> torch.Tensor:
        """Return (..., chunk_size, action_dim) action sequence."""
        flat = self.actor(obs)
        return flat.view(*flat.shape[:-1], self._chunk_size, self._action_dim)

    def actor_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """1-step facade: return chunk[..., 0, :] so PPO/eval just see the
        familiar (B, action_dim) mean. Codex (A): assert trailing shape."""
        chunks = self.actor_chunks(obs)
        mean = chunks[..., 0, :]
        assert mean.shape[-1] == self._action_dim, (
            f"actor_mean trailing dim={mean.shape[-1]} != action_dim={self._action_dim}"
        )
        return mean

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same 1-step signature as :class:`ActorCritic` — PPO transfer compat."""
        mean = self.actor_mean(obs)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs)
        return action, log_prob, entropy, value
