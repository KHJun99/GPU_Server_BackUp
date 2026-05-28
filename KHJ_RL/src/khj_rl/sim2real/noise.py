"""Per-step obs / action noise injected to close the sim2real gap.

Applied at obs post-processing only — the reward / success path uses GT
pose so noise can't leak into the learning signal (CLAUDE.md ## Phase 1
결정 사항 6). Phase 1 fields:

- joint encoder gaussian (rad / rad·s)
- cube xy/z gaussian (m) + low-rate outlier jumps (m)
- per-camera visibility dropout used as a cube-pose dropout

Legacy isotropic fields (``obs_object_pos_std``, ``obs_object_quat_std``)
are kept for back-compat with Phase 0 callers but unused by Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseModel:
    """Phase 1 sim2real noise. Stateless — caller supplies an ``rng``."""

    # Joint encoder noise.
    obs_joint_pos_std: float = 0.0       # radians
    obs_joint_vel_std: float = 0.0       # rad/s

    # Cube pose noise (xy and z split — z is noisier on depth-from-image).
    obs_object_pos_xy_std: float = 0.0   # meters
    obs_object_pos_z_std: float = 0.0    # meters
    obs_object_outlier_prob: float = 0.0
    obs_object_outlier_max: float = 0.0  # meters, ± uniform jump amplitude
    obs_visibility_dropout_prob: float = 0.0

    # Action path.
    action_std: float = 0.0
    action_latency_steps: int = 0

    # Legacy Phase 0 fields (kept for back-compat; not used in Phase 1).
    obs_object_pos_std: float = 0.0
    obs_object_quat_std: float = 0.0

    def is_active(self) -> bool:
        scalars = (
            self.obs_joint_pos_std,
            self.obs_joint_vel_std,
            self.obs_object_pos_xy_std,
            self.obs_object_pos_z_std,
            self.obs_object_outlier_prob,
            self.obs_object_outlier_max,
            self.obs_visibility_dropout_prob,
            self.action_std,
            self.obs_object_pos_std,
            self.obs_object_quat_std,
        )
        return any(v != 0.0 for v in scalars) or self.action_latency_steps > 0

    # ---- apply hooks (env calls these in compute_obs only) ---------------

    def apply_obs_joint_pos(self, joint_pos: np.ndarray, rng) -> np.ndarray:
        if self.obs_joint_pos_std <= 0.0:
            return joint_pos
        noise = rng.normal(0.0, self.obs_joint_pos_std, size=joint_pos.shape)
        return (joint_pos + noise).astype(joint_pos.dtype)

    def apply_obs_joint_vel(self, joint_vel: np.ndarray, rng) -> np.ndarray:
        if self.obs_joint_vel_std <= 0.0:
            return joint_vel
        noise = rng.normal(0.0, self.obs_joint_vel_std, size=joint_vel.shape)
        return (joint_vel + noise).astype(joint_vel.dtype)

    def apply_obs_object_pos(
        self, xyz: np.ndarray, rng
    ) -> tuple[np.ndarray, bool]:
        """Returns ``(noisy_xyz, dropout_occurred)``.

        The caller decides what to do on ``dropout_occurred=True`` — typically
        substitute a last-known pose. Keeping that policy out of NoiseModel
        keeps this class stateless and easy to reason about.
        """
        xyz = np.asarray(xyz, dtype=np.float64).copy()
        dropout = (
            self.obs_visibility_dropout_prob > 0.0
            and float(rng.random()) < self.obs_visibility_dropout_prob
        )
        sigmas = np.array(
            [
                self.obs_object_pos_xy_std,
                self.obs_object_pos_xy_std,
                self.obs_object_pos_z_std,
            ]
        )
        if (sigmas > 0.0).any():
            xyz = xyz + rng.normal(0.0, sigmas)
        if (
            self.obs_object_outlier_prob > 0.0
            and self.obs_object_outlier_max > 0.0
            and float(rng.random()) < self.obs_object_outlier_prob
        ):
            jump = rng.uniform(
                -self.obs_object_outlier_max,
                +self.obs_object_outlier_max,
                size=3,
            )
            xyz = xyz + jump
        return xyz, dropout
