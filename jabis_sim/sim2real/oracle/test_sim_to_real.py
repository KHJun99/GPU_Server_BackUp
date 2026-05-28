"""Unit tests for sim_to_real_right_arm.

Critical test: ``test_left_multiplication_matches_matrix_composition`` proves
the implementation uses LEFT multiplication (parent-frame) by composing a
non-trivial rotation that does not commute with Rz(π) — Codex review 5/11.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Make sibling module importable when running ``pytest`` from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim_to_real_right_arm import (  # noqa: E402
    Q_PI_Z,
    quaternion_multiply,
    sim_to_right_arm_pose,
    sim_to_right_arm_pos,
    sim_to_right_arm_quat,
)

ATOL = 1e-9
IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def _quat_equal_up_to_sign(a: np.ndarray, b: np.ndarray, atol: float = ATOL) -> bool:
    """Quaternions q and -q represent the same rotation."""
    return np.allclose(a, b, atol=atol) or np.allclose(a, -b, atol=atol)


def _quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Convert (qx, qy, qz, qw) unit quaternion to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# ------------------------------ Position ------------------------------ #


def test_position_at_origin():
    assert np.allclose(sim_to_right_arm_pos([0, 0, 0]), [0, 0, 0], atol=ATOL)


def test_position_x_axis():
    """sim의 +x 0.2 m → 실물의 -x 0.2 m"""
    assert np.allclose(sim_to_right_arm_pos([0.2, 0, 0]), [-0.2, 0, 0], atol=ATOL)


def test_position_y_axis():
    """sim의 +y 0.1 m → 실물의 -y 0.1 m"""
    assert np.allclose(sim_to_right_arm_pos([0, 0.1, 0]), [0, -0.1, 0], atol=ATOL)


def test_position_z_unchanged():
    """z 축은 변하지 않음"""
    assert np.allclose(sim_to_right_arm_pos([0, 0, 0.15]), [0, 0, 0.15], atol=ATOL)


def test_position_general():
    res = sim_to_right_arm_pos([0.3, -0.2, 0.05])
    assert np.allclose(res, [-0.3, 0.2, 0.05], atol=ATOL)


def test_position_invalid_shape_raises():
    with pytest.raises(ValueError):
        sim_to_right_arm_pos([1.0, 2.0])
    with pytest.raises(ValueError):
        sim_to_right_arm_pos([1.0, 2.0, 3.0, 4.0])


# ----------------------------- Quaternion ----------------------------- #


def test_quaternion_identity_to_pi_z():
    """identity quaternion (0,0,0,1)에 z-π 회전 적용 → ±(0,0,1,0)"""
    res = sim_to_right_arm_quat(IDENTITY_QUAT)
    assert _quat_equal_up_to_sign(res, Q_PI_Z), f"got {res}"


def test_quaternion_double_pi_returns_identity():
    """π 회전을 두 번 적용하면 ±identity (quaternion sign ambiguity)"""
    once = sim_to_right_arm_quat(IDENTITY_QUAT)
    twice = sim_to_right_arm_quat(once)
    assert _quat_equal_up_to_sign(twice, IDENTITY_QUAT), f"got {twice}"


def test_quaternion_normalization():
    """Non-unit quaternion → normalized output (Codex review)."""
    non_unit = np.array([0.0, 0.0, 0.0, 2.0])  # ||q|| = 2
    res = sim_to_right_arm_quat(non_unit)
    assert np.isclose(np.linalg.norm(res), 1.0, atol=ATOL)
    assert _quat_equal_up_to_sign(res, Q_PI_Z)


def test_quaternion_zero_raises():
    """Zero-norm quaternion is not a valid rotation (Codex review)."""
    with pytest.raises(ValueError, match="zero norm"):
        sim_to_right_arm_quat([0.0, 0.0, 0.0, 0.0])


def test_quaternion_invalid_shape_raises():
    with pytest.raises(ValueError):
        sim_to_right_arm_quat([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        sim_to_right_arm_quat([0.0, 0.0, 0.0, 1.0, 0.0])


def test_left_multiplication_matches_matrix_composition():
    """**Critical test (Codex review 5/11):** prove LEFT multiplication.

    Use a +90° rotation about the x-axis — this does NOT commute with Rz(π).
    Expected: ``R_actual == Rz(π) · Rx(π/2)`` (parent-frame), and explicitly
    NOT ``Rx(π/2) · Rz(π)`` (body-frame). If both happened to be equal the
    test would be a no-op, so we assert they differ.
    """
    s = np.sin(np.pi / 4)
    c = np.cos(np.pi / 4)
    quat_x90 = np.array([s, 0.0, 0.0, c])  # +90° about +x

    Rz_pi = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    Rx_90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    R_expected_left = Rz_pi @ Rx_90   # parent-frame composition (correct)
    R_expected_right = Rx_90 @ Rz_pi  # body-frame composition (incorrect)

    # Sanity: rotations must NOT commute, otherwise the test cannot tell
    # left from right.
    assert not np.allclose(R_expected_left, R_expected_right, atol=1e-3), (
        "test setup error: rotations commute"
    )

    out_quat = sim_to_right_arm_quat(quat_x90)
    R_actual = _quat_to_rot_matrix(out_quat)
    assert np.allclose(R_actual, R_expected_left, atol=ATOL), (
        f"left-mul mismatch:\n actual=\n{R_actual}\n expected=\n{R_expected_left}"
    )


# ---------------------- quaternion_multiply itself --------------------- #


def test_quaternion_multiply_identity_is_left_neutral():
    q = np.array([0.1, -0.2, 0.3, 0.9273618])  # arbitrary non-unit-but-close
    out = quaternion_multiply(IDENTITY_QUAT, q)
    assert np.allclose(out, q, atol=ATOL)


def test_quaternion_multiply_identity_is_right_neutral():
    q = np.array([0.1, -0.2, 0.3, 0.9273618])
    out = quaternion_multiply(q, IDENTITY_QUAT)
    assert np.allclose(out, q, atol=ATOL)


def test_quaternion_multiply_non_commutative():
    """Sanity: q1 ⊗ q2 ≠ q2 ⊗ q1 for non-trivial inputs."""
    q1 = np.array([np.sin(np.pi / 4), 0, 0, np.cos(np.pi / 4)])  # x90
    q2 = Q_PI_Z  # z180
    a = quaternion_multiply(q1, q2)
    b = quaternion_multiply(q2, q1)
    assert not np.allclose(a, b, atol=1e-3)


# ------------------------------- Pose --------------------------------- #


def test_pose_convenience_matches_individual_calls():
    pos = [0.3, -0.2, 0.05]
    quat = [0.0, 0.0, 0.0, 1.0]
    out_pos, out_quat = sim_to_right_arm_pose(pos, quat)
    assert np.allclose(out_pos, sim_to_right_arm_pos(pos), atol=ATOL)
    assert np.allclose(out_quat, sim_to_right_arm_quat(quat), atol=ATOL)


def test_pose_with_nontrivial_rotation():
    """Ensure pose wrapper applies both transforms consistently."""
    pos = [0.1, 0.2, 0.3]
    s = np.sin(np.pi / 4)
    c = np.cos(np.pi / 4)
    quat = [s, 0.0, 0.0, c]
    out_pos, out_quat = sim_to_right_arm_pose(pos, quat)
    assert np.allclose(out_pos, [-0.1, -0.2, 0.3], atol=ATOL)
    # Output quat must be unit
    assert np.isclose(np.linalg.norm(out_quat), 1.0, atol=ATOL)
