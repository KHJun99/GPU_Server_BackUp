"""Home pose top-down verification (per docs/method_b_design.md §6 M0.5 Step F).

The home pose ``cfg.robot.joint_init`` is the null-space bias target the IK
falls toward whenever it has redundancy left after solving the position
task. For Method B (top-down PnP, no orientation task) the entire
top-down invariant on the fingers therefore hinges on this one pose being
top-down — if it ever drifts, every grasp attempt drifts with it.

The published invariant (docs §6 Step F, cfg.py:51 comment "down_align =
0.96") is:

    cos(angle(left_distal_local_X, world_neg_Z)) >= 0.96     # ≈ 16.3°

This is enforced symmetrically for left_distal_link and right_distal_link
under the env's documented FK assumption (gripper joint = 0, all four
4-bar mimic joints = 0 — same as ``env._fk_ee_pose``).

Failure modes this regression catches:
- Someone retunes joint_init for a different workspace altitude and loses
  the top-down orientation silently (sim still runs, just with horizontal
  grasp posture → cube ejection on close).
- URDF re-conversion changes link frame orientation so the same q gives
  a different ee pose.
- left and right fingers drift apart in alignment (asymmetric grasp).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

from khj_rl.envs.cube_lift.cfg import CubeLiftEnvCfg


URDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "converted"
    / "urdf"
    / "so101_pincopen_gripper.urdf"
)

# Published invariant (docs/method_b_design.md §6 M0.5 Step F).
DOWN_ALIGN_MIN = 0.96  # cos(angle); 0.96 ≈ 16.26°.

# Symmetry tolerance: left and right finger local-+X projections onto
# world -Z must agree to within this cosine difference. 4-bar geometry is
# symmetric under URDF mirroring so any discrepancy at q=0 indicates a
# URDF / mimic mismatch we want to see.
LEFT_RIGHT_COSINE_AGREEMENT = 1e-3


@pytest.fixture(scope="module")
def cfg() -> CubeLiftEnvCfg:
    return CubeLiftEnvCfg()


@pytest.fixture(scope="module")
def pin_model() -> pin.Model:
    assert URDF_PATH.exists(), f"URDF missing: {URDF_PATH}"
    return pin.buildModelFromUrdf(str(URDF_PATH))


@pytest.fixture(scope="module")
def pin_data(pin_model: pin.Model) -> pin.Data:
    return pin_model.createData()


def _q_at_home(pin_model: pin.Model, cfg: CubeLiftEnvCfg) -> np.ndarray:
    """Build the full nq=10 q vector at home pose.

    Mirrors ``env._fk_ee_pose``: arm slots take joint_init, gripper and
    all four mimic passive joints are pinned at 0. That is the
    documented assumption — the top-down check is meaningful only at
    this canonical configuration.
    """
    q = np.zeros(pin_model.nq, dtype=np.float64)
    for slot, name in enumerate(cfg.robot.joint_names):
        qi = pin_model.joints[pin_model.getJointId(name)].idx_q
        q[qi] = float(cfg.robot.joint_init[slot])
    q[pin_model.joints[pin_model.getJointId(cfg.robot.gripper_joint_name)].idx_q] = 0.0
    for name in ("left_proximal", "left_distal", "right_proximal", "right_distal"):
        qi = pin_model.joints[pin_model.getJointId(name)].idx_q
        q[qi] = 0.0
    return q


def _finger_long_axis_world(
    pin_model: pin.Model, pin_data: pin.Data, frame_name: str
) -> np.ndarray:
    """Return the world-frame unit vector of the finger's local +X axis.

    Local +X is the finger long-axis (cfg.py:51 design note). For a
    top-down grasp posture we want this vector pointing roughly toward
    world -Z, i.e. down.
    """
    fid = pin_model.getFrameId(frame_name)
    R = np.asarray(pin_data.oMf[fid].rotation, dtype=np.float64)
    return R @ np.array([1.0, 0.0, 0.0], dtype=np.float64)


def _down_align(axis_world: np.ndarray) -> float:
    """Cosine similarity against world -Z. 1.0 = exactly down."""
    world_neg_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return float(np.dot(axis_world, world_neg_z))


class TestHomePoseTopDown:
    def test_left_finger_points_down(
        self,
        pin_model: pin.Model,
        pin_data: pin.Data,
        cfg: CubeLiftEnvCfg,
    ) -> None:
        q = _q_at_home(pin_model, cfg)
        pin.framesForwardKinematics(pin_model, pin_data, q)
        axis = _finger_long_axis_world(pin_model, pin_data, "left_distal_link")
        da = _down_align(axis)
        assert da >= DOWN_ALIGN_MIN, (
            f"left_distal_link local +X is no longer top-down at "
            f"cfg.robot.joint_init: down_align={da:.4f} < {DOWN_ALIGN_MIN}. "
            f"axis in world = {axis.round(4).tolist()}. "
            f"joint_init may have been retuned without verifying the "
            f"top-down invariant (cfg.py:51 design note)."
        )

    def test_right_finger_points_down(
        self,
        pin_model: pin.Model,
        pin_data: pin.Data,
        cfg: CubeLiftEnvCfg,
    ) -> None:
        q = _q_at_home(pin_model, cfg)
        pin.framesForwardKinematics(pin_model, pin_data, q)
        axis = _finger_long_axis_world(pin_model, pin_data, "right_distal_link")
        da = _down_align(axis)
        assert da >= DOWN_ALIGN_MIN, (
            f"right_distal_link local +X is no longer top-down: "
            f"down_align={da:.4f} < {DOWN_ALIGN_MIN}. "
            f"axis in world = {axis.round(4).tolist()}."
        )

    def test_left_right_symmetric(
        self,
        pin_model: pin.Model,
        pin_data: pin.Data,
        cfg: CubeLiftEnvCfg,
    ) -> None:
        """At gripper=0 + all mimics=0, left/right fingers must align identically.

        The 4-bar URDF is mirrored across the gripper x-z plane, so the
        long-axis direction (local +X expressed in world) should be the
        same vector for both fingers when gripper and mimics are at 0.
        A drift here points at URDF asymmetry that would silently bias
        grasps.
        """
        q = _q_at_home(pin_model, cfg)
        pin.framesForwardKinematics(pin_model, pin_data, q)
        left = _finger_long_axis_world(pin_model, pin_data, "left_distal_link")
        right = _finger_long_axis_world(pin_model, pin_data, "right_distal_link")
        # Both unit-norm rotations applied to (1,0,0) so dot equals cos(angle).
        cosine_diff = float(abs(np.dot(left, right) - 1.0))
        assert cosine_diff < LEFT_RIGHT_COSINE_AGREEMENT, (
            f"left/right finger long-axis disagree at home: "
            f"left={left.round(4).tolist()} right={right.round(4).tolist()} "
            f"cosine_diff={cosine_diff:.2e} > {LEFT_RIGHT_COSINE_AGREEMENT:.2e}"
        )
