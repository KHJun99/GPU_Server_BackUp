"""Sim → physical right-arm pose transform.

The Jabis sim oracle outputs poses in the simulated arm's base frame, which
has zero rotation about world Z. The physical right arm's base frame is
rotated by **π about world Z** relative to the sim base frame (see memory
#23 / d456_perception_design.md). This module composes that fixed Z-π
parent-frame rotation onto sim outputs.

Mathematical contract
---------------------
- Position: ``(x, y, z) → (-x, -y, z)`` (rotating a position vector by Rz(π)).
- Quaternion: ``q_right = q_pi_z ⊗ q_sim``, where
  ``q_pi_z = (0, 0, 1, 0)`` in ``(qx, qy, qz, qw)`` order. This is a
  **left** multiplication, which corresponds to a **parent-frame**
  composition: ``R_right = Rz(π) · R_sim``. Right multiplication would be a
  body-frame composition and is not what we want here.

Quaternion convention
---------------------
- All quaternions are 4-vectors in ``(qx, qy, qz, qw)`` order (Isaac/Hamilton
  convention with the scalar last).
- Hamilton product:
    ``(q1 ⊗ q2).w = w1 w2 - v1 · v2``
    ``(q1 ⊗ q2).v = w1 v2 + w2 v1 + v1 × v2``
- Inputs may be non-unit; the public functions normalize internally.
- A near-zero quaternion (``‖q‖ < 1e-12``) raises ``ValueError`` — there is
  no meaningful rotation to apply.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# z축 π 회전을 나타내는 단위 quaternion. (qx, qy, qz, qw) 순서.
# axis = (0, 0, 1), angle = π → q = (sin(π/2)·axis, cos(π/2)) = (0, 0, 1, 0).
Q_PI_Z: np.ndarray = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)


def _validate_vec3(v, name: str) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be shape (3,); got {arr.shape}")
    return arr


def _validate_quat(q, name: str) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64).reshape(-1)
    if arr.shape != (4,):
        raise ValueError(
            f"{name} must be shape (4,) in (qx,qy,qz,qw) order; got {arr.shape}"
        )
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError(f"{name} has zero norm; not a valid quaternion")
    return arr


def quaternion_multiply(q1, q2) -> np.ndarray:
    """Hamilton product ``q1 ⊗ q2``; both in (qx, qy, qz, qw) order.

    Inputs are validated for shape but **not normalized**, and the output is
    **not normalized** either. Pass already-unit quaternions if a unit
    result is required, OR prefer the public wrappers
    (``sim_to_right_arm_quat`` / ``sim_to_right_arm_pose``) which normalize
    internally. Chaining many products on non-unit input drifts numerically.
    """
    a = _validate_quat(q1, "q1")
    b = _validate_quat(q2, "q2")
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    w = w1 * w2 - (x1 * x2 + y1 * y2 + z1 * z2)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([x, y, z, w], dtype=np.float64)


def sim_to_right_arm_pos(sim_pos) -> np.ndarray:
    """Apply z-axis π rotation to a position vector.

    ``(x, y, z) → (-x, -y, z)``.
    """
    p = _validate_vec3(sim_pos, "sim_pos")
    return np.array([-p[0], -p[1], p[2]], dtype=np.float64)


def sim_to_right_arm_quat(sim_quat) -> np.ndarray:
    """Compose Z-π parent-frame rotation onto a sim-frame quaternion.

    Returns the unit quaternion ``q_right = Q_PI_Z ⊗ normalize(q_sim)``.
    """
    q = _validate_quat(sim_quat, "sim_quat")
    q_unit = q / np.linalg.norm(q)
    out = quaternion_multiply(Q_PI_Z, q_unit)
    out = out / np.linalg.norm(out)
    return out


def sim_to_right_arm_pose(sim_pos, sim_quat) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper that converts position and quaternion together."""
    return sim_to_right_arm_pos(sim_pos), sim_to_right_arm_quat(sim_quat)
