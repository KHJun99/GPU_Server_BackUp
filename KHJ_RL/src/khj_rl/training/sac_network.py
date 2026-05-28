"""SAC actor + critic networks (Phase 1 SACfD).

Distinct from :class:`khj_rl.training.network.ActorCritic` because SAC needs:
  - State-conditional log-std on the actor (PPO uses a global parameter).
  - Tanh squashing with the Jacobian log-prob correction (PPO uses raw
    Gaussian + clip).
  - Twin Q networks for the standard double-Q overestimation fix.

The actor's trunk + mean_head deliberately mirrors the shapes of
``ActorCritic.actor`` (Linear(31, 256) → Tanh → Linear(256, 256) → Tanh →
Linear(256, 6)) so weights from a 1-step BC checkpoint can be transferred
without a remapping step. Both networks use Tanh activations — BC weight
transfer is exact (ReLU was here before F9-fix; the activation mismatch
was root-cause of lift=0% despite bc_loss decreasing normally).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Standard SAC log-std clamp range (Haarnoja et al. 2018 / CleanRL) is
# (-20, 2). F7: tighten LOG_STD_MAX to log(0.05)=-3.0 — F5/F6 at std=0.20
# still produced lift=0% (tanh squash differs from BC sweep's clamp).
# BC sweep at std=0.05 gives lift_history=79%, so this cap puts the actor
# in the near-deterministic-BC region. Combined with F6's --no-entropy-term
# (TD3-like), narrow std no longer backfires via squashed Gaussian log π.
LOG_STD_MIN = -20.0
LOG_STD_MAX = -3.0


def _layer_init(layer: nn.Linear, gain: float = math.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class GaussianActor(nn.Module):
    """Squashed Gaussian policy with state-conditional log-std.

    Forward returns ``(mean, log_std)``. ``sample`` returns
    ``(action, log_prob)`` with the tanh Jacobian correction applied so SAC's
    entropy term is unbiased. ``mean_action`` returns ``tanh(mean)`` for
    deterministic evaluation.

    The ``logstd_head`` is intentionally a separate linear (not a global
    parameter) so SAC can learn state-dependent exploration. During BC
    pretrain it can be frozen at a fixed init via :meth:`freeze_logstd`.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self._action_dim = int(action_dim)
        self.trunk = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.mean_head = _layer_init(nn.Linear(hidden_dim, action_dim), gain=0.01)
        self.logstd_head = _layer_init(nn.Linear(hidden_dim, action_dim), gain=0.01)
        # Bias logstd_head so initial std ≈ exp(-1.0) ≈ 0.37 regardless of
        # the trunk's pre-activation magnitude (CLAUDE.md Phase 1 결정 7).
        nn.init.constant_(self.logstd_head.bias, -1.0)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def freeze_logstd(self) -> None:
        for p in self.logstd_head.parameters():
            p.requires_grad_(False)

    def unfreeze_logstd(self) -> None:
        for p in self.logstd_head.parameters():
            p.requires_grad_(True)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.logstd_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(action ∈ [-1,1]^A, log_prob)`` with clamp (F8 fix).

        F8 (#27 troubleshooting addendum): switched from ``tanh(z)`` to
        ``z.clamp(-1, 1)``. The previous tanh squash compressed BC's
        transferred mean weights (trained to output [-1, 1] without
        squash) so a BC action of -1 became tanh(-1)=-0.76 in SAC, and
        SAC could never reach the boundary action (gripper close). This
        broke grasp from the start. Clamp preserves -1/+1 endpoints, so
        BC weight transfer is exact. log_prob no longer needs tanh
        jacobian correction; raw Gaussian log_prob suffices (the clamp
        boundary has measure-zero impact on continuous Gaussian density).
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = z.clamp(-1.0, 1.0)
        log_prob = normal.log_prob(z).sum(dim=-1)
        return action, log_prob

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic eval action (clamp(mean, -1, 1), no sampling)."""
        mean, _ = self.forward(obs)
        return mean.clamp(-1.0, 1.0)


class QNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Linear(obs_dim + action_dim, hidden_dim)),
            nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, 1), gain=1.0),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


class TwinQ(nn.Module):
    """Two independent Q-networks; the SAC critic uses ``min(Q1, Q2)``."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = QNet(obs_dim, action_dim, hidden_dim)
        self.q2 = QNet(obs_dim, action_dim, hidden_dim)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)
