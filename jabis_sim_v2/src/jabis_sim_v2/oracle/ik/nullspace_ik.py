"""
Nullspace Projection IK for redundant manipulators.

For 5-DOF arm solving 3-DOF position task:
  - Primary task:   reach position target (3 DOF)
  - Secondary task: maintain joint posture (2 DOF nullspace)

This solves 'finger flipping' problem in standard DLS IK where
secondary DOF gets chosen arbitrarily.

Math:
  dq_primary   = J_pos+ @ dx
  N            = I - J_pos+ @ J_pos     (nullspace projector)
  dq_secondary = N @ (q_rest - q)        (pull toward rest posture)
  dq           = dq_primary + lam * dq_secondary
"""
from __future__ import annotations

import torch


class NullspaceIK:
    """Vectorized 5-DOF position IK with posture nullspace task."""

    def __init__(
        self,
        num_envs: int,
        num_joints: int = 5,
        device: torch.device | str = "cpu",
        # DLS damping for primary task
        damping: float = 0.05,
        # secondary task gain
        nullspace_gain: float = 0.5,
        # secondary task rest pose (per-env, set later)
        rest_pose: torch.Tensor | None = None,
        # rest pose joint weights (higher = more strongly pulled)
        rest_weights: torch.Tensor | None = None,
        # per-step joint delta cap
        max_dq: float = 0.10,
    ):
        self.num_envs = int(num_envs)
        self.num_joints = int(num_joints)
        self.device = torch.device(device)
        self.damping = float(damping)
        self.nullspace_gain = float(nullspace_gain)
        self.max_dq = float(max_dq)

        if rest_pose is None:
            rest_pose = torch.zeros(num_envs, num_joints, device=self.device)
        self.rest_pose = rest_pose.to(self.device)

        if rest_weights is None:
            rest_weights = torch.ones(num_joints, device=self.device)
        self.rest_weights = rest_weights.to(self.device)

    def set_rest_pose(self, rest_pose: torch.Tensor):
        """Update secondary task target joint pose (per-env)."""
        self.rest_pose = rest_pose.to(self.device)

    def compute(
        self,
        target_pos_b: torch.Tensor,  # [N, 3] desired EE pos in base frame
        current_pos_b: torch.Tensor,  # [N, 3] current EE pos in base frame
        jacobian: torch.Tensor,  # [N, 6, num_joints] base-frame jacobian (6-DOF)
        current_joint_pos: torch.Tensor,  # [N, num_joints]
    ) -> torch.Tensor:
        """One IK step. Returns target joint position."""
        # 1) position error
        dx = target_pos_b - current_pos_b  # [N, 3]
        # cap step
        dx_norm = dx.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        cap_scale = (self.max_dq / dx_norm).clamp(max=1.0)
        dx = dx * cap_scale

        # 2) position-part of jacobian
        J_pos = jacobian[:, :3, :]  # [N, 3, 5]

        # 3) DLS pseudoinverse
        # J+ = J^T (J J^T + lam^2 I)^-1
        lam2 = self.damping ** 2
        N, D, K = J_pos.shape  # D=3, K=num_joints
        I3 = torch.eye(D, device=self.device).unsqueeze(0).expand(N, -1, -1)
        JJt = torch.bmm(J_pos, J_pos.transpose(1, 2))  # [N, 3, 3]
        JJt_damped = JJt + lam2 * I3
        JJt_inv = torch.linalg.inv(JJt_damped)
        J_pinv = torch.bmm(J_pos.transpose(1, 2), JJt_inv)  # [N, 5, 3]

        # 4) primary task
        dq_primary = torch.bmm(J_pinv, dx.unsqueeze(-1)).squeeze(-1)  # [N, 5]

        # 5) nullspace projector
        # N = I - J+ J
        IK_dim = K
        I_K = torch.eye(IK_dim, device=self.device).unsqueeze(0).expand(N, -1, -1)
        N_proj = I_K - torch.bmm(J_pinv, J_pos)  # [N, 5, 5]

        # 6) secondary task: pull toward rest pose
        q_err = self.rest_pose - current_joint_pos  # [N, 5]
        q_err_weighted = q_err * self.rest_weights.unsqueeze(0)
        dq_secondary = torch.bmm(N_proj, q_err_weighted.unsqueeze(-1)).squeeze(-1)

        # 7) total dq + cap
        dq = dq_primary + self.nullspace_gain * dq_secondary
        dq_norm = dq.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        cap_scale = (self.max_dq / dq_norm).clamp(max=1.0)
        dq = dq * cap_scale

        # 8) joint target
        joint_target = current_joint_pos + dq
        return joint_target
