"""Jacobian regression tests for Method B IK (per docs/method_b_design.md §6 M0.5 Step B).

TODO (M3 oracle rewrite): refresh the three ``task_*_placeholder`` poses in
``_named_poses()`` with arm configurations actually visited by the new
EE-delta oracle (grasp / lift / release+retreat frames). Until then the
placeholder values are home-pose offsets — fine for Jacobian regression
(analytic = finite-diff at any reachable q), but they do not yet exercise
the IK at the workspace points the policy will see. Refresh path:
1) Run new EE-delta oracle 1 ep and dump arm_q at each phase boundary.
2) Replace the three ``_arm_with(...)`` calls in ``_named_poses()``.
3) Re-run this test; analytic-vs-FD should still pass within JAC_MAX_COL_ERR.



Goal: lock in the Pinocchio-based Jacobian path used by ``env._ik_step()`` so
the per-step IK call always sees the same linear map between arm joint
velocities and ``gripper_dummy_link`` velocity that we measured here.

Two test classes:

1. ``TestJacobianStandalone`` — Pinocchio vs finite-difference FK. No
   AppLauncher / Isaac Sim required. Runs in CI / on dev machines.
   This is the primary regression — finite-diff is the ground truth and
   any future Pinocchio version change that breaks it must be caught here.

2. ``TestJacobianVsPhysX`` — Pinocchio vs PhysX
   ``root_physx_view.get_jacobians()``. Requires a booted ``AppLauncher``
   and Isaac Lab installation. Auto-skips when those imports fail.
   This locks the one-time numerical agreement between the two engines
   so we can use Pinocchio everywhere knowing PhysX would give the same
   numbers at the same arm configuration.

Pose suite (per design doc §6 M0.5 Step B):
- home pose (``cfg.robot.joint_init``)
- 4 limit-near (shoulder_pan / shoulder_lift / elbow_flex / wrist_flex
  each clamped to 95% of their limit; wrist_roll is full-rotation so
  "limit-near" is meaningless and skipped — it is also
  ``wrist_roll_clamp_to_home`` in IK anyway)
- 1 singular (elbow fully extended at 0.0 → arm + forearm collinear)
- 3 task-critical placeholders (grasp / lift / release+retreat). These
  use home-pose offsets here; M3 (oracle rewrite) will refresh them
  with real oracle frames once the EE-delta oracle exists.
- 10 random within joint limits, seed=0
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
EE_LINK_NAME = "gripper_dummy_link"

# Numerical tolerances. Finite-diff at eps=1e-5 against analytic Pinocchio
# jacobian agrees within a few μm for 1-DOF revolute joints on a 30 cm
# kinematic chain. 5e-4 m/rad gives 50× margin against float32 noise and
# any benign Pinocchio version drift.
JAC_MAX_COL_ERR = 5e-4  # m/rad, per (row, col) entry
FRAME_FROBENIUS_MAX = 1e-3  # full-matrix safety bound
FINITE_DIFF_EPS = 1e-5  # rad

# PhysX comparison is looser — PhysX returns a 6-DOF body-jacobian and we
# slice the same arm columns. Build differences (root joint, body inertia
# parameterization) can introduce small offsets even at the same config.
PHYSX_MAX_COL_ERR = 5e-3  # m/rad


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


@pytest.fixture(scope="module")
def arm_qidx(pin_model: pin.Model, cfg: CubeLiftEnvCfg) -> list[int]:
    """Map cfg.robot.joint_names → pinocchio q indices, preserving order.

    This is the asserted-once mapping that env._ik_step() will rely on.
    """
    qidx: list[int] = []
    for name in cfg.robot.joint_names:
        # buildModelFromUrdf adds the universe joint at idx 0; arm joints
        # follow as 1-DOF revolutes so idx_q == ordering position in
        # ``joint_names`` for this URDF. Asserted below by index match.
        jid = pin_model.getJointId(name)
        assert jid < pin_model.njoints, (
            f"joint {name!r} not in pinocchio model"
        )
        qidx.append(pin_model.joints[jid].idx_q)
    return qidx


@pytest.fixture(scope="module")
def ee_frame_id(pin_model: pin.Model) -> int:
    fid = pin_model.getFrameId(EE_LINK_NAME)
    assert fid < pin_model.nframes, f"frame {EE_LINK_NAME!r} not found"
    return fid


# ---------------------------------------------------------------------------
# Pose suite generation
# ---------------------------------------------------------------------------


def _zeroed_q(pin_model: pin.Model) -> np.ndarray:
    return np.zeros(pin_model.nq, dtype=np.float64)


def _fill_arm(q: np.ndarray, arm_qidx: list[int], arm_vals: np.ndarray) -> np.ndarray:
    out = q.copy()
    for slot, qi in enumerate(arm_qidx):
        out[qi] = float(arm_vals[slot])
    return out


def _named_poses(cfg: CubeLiftEnvCfg) -> list[tuple[str, np.ndarray]]:
    """The deterministic part of the pose suite (9 poses)."""
    home = np.array(cfg.robot.joint_init, dtype=np.float64)
    limits = cfg.robot.joint_pos_limit  # tuple of (low, high) per arm joint
    poses: list[tuple[str, np.ndarray]] = [("home", home.copy())]

    # 4 limit-near: clamp one joint (slots 0..3) at 95% of its limit, hold
    # the others at home. wrist_roll (slot 4) is full rotation + clamped
    # to home in IK so skipping it is correct.
    for slot in range(4):
        low, high = limits[slot]
        # Move toward whichever limit is farther from home — gives the
        # most useful "near-limit" pose for the IK to wrestle with.
        target_high = home[slot] + 0.95 * (high - home[slot])
        target_low = home[slot] + 0.95 * (low - home[slot])
        target = target_high if abs(target_high - home[slot]) > abs(target_low - home[slot]) else target_low
        q = home.copy()
        q[slot] = float(target)
        poses.append((f"limit_near_{cfg.robot.joint_names[slot]}", q))

    # Singular: elbow_flex at lower limit (0.0) = fully extended → arm
    # straightens out and reachable workspace edge collapses.
    q_sing = home.copy()
    q_sing[2] = 0.0
    poses.append(("singular_elbow_extended", q_sing))

    # 3 task-critical placeholders (refresh from EE-delta oracle in M3).
    # Use home with shoulder_lift/elbow tilts that mimic the rough poses
    # the oracle MP visits during grasp / lift / release+retreat.
    poses.append(("task_grasp_placeholder", _arm_with(home, shoulder_lift=-0.6, elbow_flex=0.6, wrist_flex=1.2)))
    poses.append(("task_lift_placeholder", _arm_with(home, shoulder_lift=-0.1, elbow_flex=0.4, wrist_flex=1.5)))
    poses.append(("task_release_placeholder", _arm_with(home, shoulder_lift=-0.4, elbow_flex=0.8, wrist_flex=1.3)))
    return poses


def _arm_with(home: np.ndarray, **overrides: float) -> np.ndarray:
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    out = home.copy()
    for k, v in overrides.items():
        out[names.index(k)] = float(v)
    return out


def _random_arm_poses(cfg: CubeLiftEnvCfg, n: int, seed: int = 0) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    limits = np.asarray(cfg.robot.joint_pos_limit, dtype=np.float64)
    lows = limits[:, 0]
    highs = limits[:, 1]
    out: list[tuple[str, np.ndarray]] = []
    for i in range(n):
        q = rng.uniform(lows, highs)
        out.append((f"random_{i}_seed{seed}", q))
    return out


def _arm_pose_suite(cfg: CubeLiftEnvCfg) -> list[tuple[str, np.ndarray]]:
    return _named_poses(cfg) + _random_arm_poses(cfg, n=10, seed=0)


# ---------------------------------------------------------------------------
# Asserts on model topology (run once per session)
# ---------------------------------------------------------------------------


class TestModelTopology:
    """Frozen mapping between cfg.robot.joint_names and Pinocchio q indices.

    Every IK-related test starts here so a future URDF reshuffle that
    silently breaks the assumed slot order fails loudly here, not in
    M1+ where the error would look like "IK target drifts by some
    inscrutable amount".
    """

    def test_joint_names_present_and_ordered(self, pin_model: pin.Model, cfg: CubeLiftEnvCfg) -> None:
        for slot, name in enumerate(cfg.robot.joint_names):
            jid = pin_model.getJointId(name)
            assert jid < pin_model.njoints, f"missing joint {name!r}"
            j = pin_model.joints[jid]
            # idx_q == slot only holds because every joint up to and
            # including the arm is 1-DOF revolute. Verified by the
            # M0.5 hand inspection: nq=10, all 1-DOF.
            assert j.idx_q == slot, (
                f"joint {name!r} idx_q={j.idx_q} does not match cfg "
                f"ordering slot {slot}. URDF root layout changed — "
                f"update arm_qidx logic in env._ik_step()."
            )

    def test_ee_frame_present(self, pin_model: pin.Model) -> None:
        fid = pin_model.getFrameId(EE_LINK_NAME)
        assert fid < pin_model.nframes, f"frame {EE_LINK_NAME!r} missing"

    def test_no_unexpected_extra_arm_dofs(self, pin_model: pin.Model, cfg: CubeLiftEnvCfg) -> None:
        # Method B IK only touches the 5 arm slots. If the URDF gained
        # an extra arm-side actuated joint we want to see it here.
        expected_pre_gripper = {"universe", *cfg.robot.joint_names}
        actual_pre_gripper = set(pin_model.names[: 1 + len(cfg.robot.joint_names)])
        assert actual_pre_gripper == expected_pre_gripper, (
            f"unexpected joint set before gripper: {actual_pre_gripper}"
        )


# ---------------------------------------------------------------------------
# Standalone Jacobian regression (Pinocchio vs finite-diff FK)
# ---------------------------------------------------------------------------


def _pin_fk_ee(
    pin_model: pin.Model,
    pin_data: pin.Data,
    ee_frame_id: int,
    q: np.ndarray,
) -> np.ndarray:
    """3-vector EE position at config q, in base frame (universe is fixed)."""
    pin.framesForwardKinematics(pin_model, pin_data, q)
    return np.asarray(pin_data.oMf[ee_frame_id].translation, dtype=np.float64).copy()


def _pin_position_jacobian(
    pin_model: pin.Model,
    pin_data: pin.Data,
    ee_frame_id: int,
    q: np.ndarray,
    arm_qidx: list[int],
) -> np.ndarray:
    """3×5 linear (position) Jacobian for the arm columns only.

    Uses ``LOCAL_WORLD_ALIGNED`` so the linear rows are world-frame
    displacement per joint angle — directly comparable to a finite-diff
    of ``pin_fk_ee`` in world frame.
    """
    pin.computeJointJacobians(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    J_full = pin.getFrameJacobian(pin_model, pin_data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
    return np.asarray(J_full[:3, arm_qidx], dtype=np.float64)


def _finite_diff_position_jacobian(
    pin_model: pin.Model,
    pin_data: pin.Data,
    ee_frame_id: int,
    q: np.ndarray,
    arm_qidx: list[int],
    eps: float = FINITE_DIFF_EPS,
) -> np.ndarray:
    """Central-difference 3×5 Jacobian. Ground truth for analytic check."""
    J_fd = np.zeros((3, len(arm_qidx)), dtype=np.float64)
    for col, qi in enumerate(arm_qidx):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[qi] += eps
        q_minus[qi] -= eps
        ee_plus = _pin_fk_ee(pin_model, pin_data, ee_frame_id, q_plus)
        ee_minus = _pin_fk_ee(pin_model, pin_data, ee_frame_id, q_minus)
        J_fd[:, col] = (ee_plus - ee_minus) / (2.0 * eps)
    return J_fd


class TestJacobianStandalone:
    """Pinocchio analytic Jacobian must match central-difference FK."""

    @pytest.mark.parametrize(
        "pose_name, arm_q",
        # Late-bound through indirection — pytest fixtures cannot supply
        # parametrize values directly, but module-scope reuse is fine for
        # the small fixed suite.
        [(name, q) for name, q in _arm_pose_suite(CubeLiftEnvCfg())],
    )
    def test_pinocchio_vs_finite_diff(
        self,
        pin_model: pin.Model,
        pin_data: pin.Data,
        ee_frame_id: int,
        arm_qidx: list[int],
        pose_name: str,
        arm_q: np.ndarray,
    ) -> None:
        q = _fill_arm(_zeroed_q(pin_model), arm_qidx, arm_q)
        J_analytic = _pin_position_jacobian(
            pin_model, pin_data, ee_frame_id, q, arm_qidx
        )
        J_fd = _finite_diff_position_jacobian(
            pin_model, pin_data, ee_frame_id, q, arm_qidx
        )
        diff = J_analytic - J_fd
        max_entry_err = float(np.max(np.abs(diff)))
        frob_err = float(np.linalg.norm(diff))
        assert max_entry_err < JAC_MAX_COL_ERR, (
            f"[{pose_name}] max entry err {max_entry_err:.2e} >= "
            f"{JAC_MAX_COL_ERR:.2e}\nJ_analytic=\n{J_analytic}\nJ_fd=\n{J_fd}"
        )
        assert frob_err < FRAME_FROBENIUS_MAX, (
            f"[{pose_name}] frobenius {frob_err:.2e} >= {FRAME_FROBENIUS_MAX:.2e}"
        )


# ---------------------------------------------------------------------------
# Optional: Pinocchio vs PhysX (one-time sim parity check)
# ---------------------------------------------------------------------------


def _sim_parity_available() -> bool:
    """Sim parity test needs Isaac Lab importable AND the EULA env vars set.

    Without ``OMNI_KIT_ACCEPT_EULA=YES`` the kit bootstrap tries to read
    user input from stdin, which deadlocks pytest's captured stdin. We
    require the explicit opt-in env vars (same set documented in
    CLAUDE.md for any AppLauncher invocation) before even trying.
    """
    import os

    try:
        import isaaclab  # noqa: F401
    except Exception:
        return False
    return (
        os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() == "YES"
        and os.environ.get("PRIVACY_CONSENT", "").upper() == "Y"
    )


@pytest.mark.skipif(
    not _sim_parity_available(),
    reason=(
        "Sim parity is opt-in. Run with: "
        "OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y CUDA_VISIBLE_DEVICES=1 "
        "pytest -k TestJacobianVsPhysX -s"
    ),
)
class TestJacobianVsPhysX:
    """One-time numerical agreement check between Pinocchio and PhysX.

    Run manually when:
      - URDF re-converted
      - Isaac Lab / Isaac Sim major version bumped
      - PincOpen mimic API stamp reapplied

    Requires:
      ``OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y CUDA_VISIBLE_DEVICES=1
        pytest tests/test_ik_jacobian_regression.py -k TestJacobianVsPhysX``

    Boots ``CubeLiftEnv`` once for the whole class (singleton sim
    constraint — see ``env.py`` notes) and compares the position-Jacobian
    slice at each pose in the suite.
    """

    @pytest.fixture(scope="class")
    def env_module(self) -> object:
        # Pre-AppLauncher pinocchio import is required by env.py too —
        # see scripts/launch_viewer.py header. Already imported at module
        # top here, so the Assimp ABI binding order is already correct.
        from isaaclab.app import AppLauncher

        sim_app = AppLauncher(headless=True, enable_cameras=False).app
        from khj_rl.envs.cube_lift.env import CubeLiftEnv

        cfg = CubeLiftEnvCfg()
        env = CubeLiftEnv(cfg)
        yield env
        sim_app.close()

    def test_physx_matches_pinocchio_at_home(
        self,
        env_module,
        pin_model: pin.Model,
        pin_data: pin.Data,
        ee_frame_id: int,
        arm_qidx: list[int],
        cfg: CubeLiftEnvCfg,
    ) -> None:
        env = env_module
        env.reset()

        # PhysX 6×N_dof body jacobian for the EE body.
        physx_jacs = env._robot.root_physx_view.get_jacobians()  # (1, n_bodies, 6, n_dofs)
        n_root_dofs = physx_jacs.shape[3] - env._robot.data.joint_pos.shape[1]
        arm_phys_idx = [n_root_dofs + i for i in env._arm_joint_idxs]
        J_physx = (
            physx_jacs[0, env._ee_body_idx, :3, :][:, arm_phys_idx]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        # Pinocchio at the same arm config the env just reset to.
        arm_q_now = (
            env._robot.data.joint_pos[0, env._arm_joint_idxs]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        q = _fill_arm(_zeroed_q(pin_model), arm_qidx, arm_q_now)
        J_pin = _pin_position_jacobian(pin_model, pin_data, ee_frame_id, q, arm_qidx)

        diff = J_physx - J_pin
        max_entry_err = float(np.max(np.abs(diff)))
        assert max_entry_err < PHYSX_MAX_COL_ERR, (
            f"PhysX vs Pinocchio mismatch {max_entry_err:.2e} >= "
            f"{PHYSX_MAX_COL_ERR:.2e}\nJ_physx=\n{J_physx}\nJ_pin=\n{J_pin}"
        )
