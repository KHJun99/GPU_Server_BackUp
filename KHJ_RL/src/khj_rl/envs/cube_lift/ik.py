"""Pure-function IK step for Method B (per docs/method_b_design.md §3).

``compute_ik_step`` takes the current arm configuration + a normalized
3-D EE delta and returns the next arm joint target plus an audit dict.
It is intentionally:

- **Pure (no env / sim state)**: every input is passed explicitly so the
  function is testable standalone via ``pytest`` (no AppLauncher boot).
  See ``tests/test_ik_step.py``.
- **DLS + null-space biased**: position-only 3-D task; null space pulls
  the arm toward ``home_pose`` so the redundant DOFs (2 in a 5-DOF arm
  on a 3-D position task) don't drift to joint limits over many steps.
- **Multi-layer guarded**: immediate AND (large q_dot + large x_err),
  immediate OR (jitter / jerk p95 over a rolling window), cumulative
  truncate (consecutive fail counter), slow divergence warning. Each
  fail carries a categorical ``fail_reason`` for diagnostics.
- **Frame-consistent**: FK on ``q_target`` is recomputed before each
  return so ``commanded_ee_delta_actual = FK(q_target) - FK(q_arm)``
  is the post-clamp / post-wrist-roll-clamp expected EE motion. This
  is the definition of "commanded delta" the next step's jitter residual
  is computed against (docs §2).

The IK step does NOT issue any sim-side writes — it returns the joint
target the caller (env) sets via ``set_joint_position_target``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np
import pinocchio as pin

from khj_rl.envs.cube_lift.cfg import EEControlCfg


# ---------------------------------------------------------------------------
# Audit state — single instance per env, reset per episode
# ---------------------------------------------------------------------------


@dataclass
class IKAuditState:
    """Episode-scoped IK history. Caller resets at episode boundary.

    All deques are bounded by the longest relevant window so memory
    stays O(1) per env regardless of episode length.
    """

    # Counters
    ik_fail_count: int = 0
    consecutive_ik_fails: int = 0
    slow_diverge_count: int = 0
    fail_reasons: dict[str, int] = field(default_factory=dict)

    # Histories for guards. Length cap is set in ``reset_for_episode``
    # so a single ``IKAuditState`` can be reused across episodes that
    # use different cfg windows.
    ee_residual_history: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    q_jerk_norm_history: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    x_err_history: Deque[float] = field(default_factory=lambda: deque(maxlen=64))

    # Last-step state for delta / jerk computation. ``None`` on the
    # first step of an episode so the corresponding metrics return 0.
    last_ee_pos: np.ndarray | None = None
    last_q_arm: np.ndarray | None = None
    last_q_dot: np.ndarray | None = None
    last_commanded_delta: np.ndarray | None = None
    last_valid_q_target: np.ndarray | None = None

    def reset_for_episode(self, cfg: EEControlCfg) -> None:
        """Clear all per-episode state, resize history caps to current cfg."""
        cap = max(cfg.ik_jitter_window, cfg.ik_slow_diverge_window, 4) + 2
        self.ik_fail_count = 0
        self.consecutive_ik_fails = 0
        self.slow_diverge_count = 0
        self.fail_reasons = {}
        self.ee_residual_history = deque(maxlen=cap)
        self.q_jerk_norm_history = deque(maxlen=cap)
        self.x_err_history = deque(maxlen=cap)
        self.last_ee_pos = None
        self.last_q_arm = None
        self.last_q_dot = None
        self.last_commanded_delta = None
        self.last_valid_q_target = None


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


@dataclass
class IKStepResult:
    q_target_arm: np.ndarray             # (n_arm,) joint targets, post-clamp
    fallback_used: bool                  # any guard tripped this step
    fail_reason: str | None              # canonical category if fallback
    truncate_episode: bool               # consecutive fails ≥ cfg threshold
    audit: dict[str, float]              # per-step diagnostics for env / logger


# ---------------------------------------------------------------------------
# Fail-reason taxonomy (docs §4.1)
# ---------------------------------------------------------------------------

FAIL_JOINT_LIMIT = "joint_limit"
FAIL_SINGULAR = "singular"
FAIL_UNREACHABLE = "unreachable"
FAIL_LARGE_QDOT = "large_qdot"
FAIL_JITTER = "jitter"

# det(J J^T) below this counts as singular config (3x3 PSD matrix in
# position-only task).
_SINGULAR_DET_THRESHOLD = 1e-4
# ||x_err|| above this counts as unreachable even after one IK iter.
_UNREACHABLE_X_ERR_M = 0.1
# Joint limit "touched" tolerance — clamp leaves q exactly on the wall.
_JOINT_LIMIT_TOL = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_full_q(pin_model: pin.Model, q_arm: np.ndarray, arm_qidx: list[int]) -> np.ndarray:
    """Pad arm-only q (n_arm,) up to full nq with gripper + mimic zeroed.

    Matches the convention in ``env._fk_ee_pose`` so FK done here yields
    the same EE pose the env's other FK calls would produce.
    """
    q = np.zeros(pin_model.nq, dtype=np.float64)
    for slot, qi in enumerate(arm_qidx):
        q[qi] = float(q_arm[slot])
    return q


def _fk_ee_pos(
    pin_model: pin.Model, pin_data: pin.Data, q_full: np.ndarray, ee_frame_id: int
) -> np.ndarray:
    """3-vector EE position in base frame (universe is fixed in this URDF)."""
    pin.framesForwardKinematics(pin_model, pin_data, q_full)
    return np.asarray(pin_data.oMf[ee_frame_id].translation, dtype=np.float64).copy()


def _position_jacobian(
    pin_model: pin.Model,
    pin_data: pin.Data,
    q_full: np.ndarray,
    ee_frame_id: int,
    arm_qidx: list[int],
) -> np.ndarray:
    """3×n_arm linear Jacobian in world frame (LOCAL_WORLD_ALIGNED rows 0:3)."""
    pin.computeJointJacobians(pin_model, pin_data, q_full)
    pin.updateFramePlacements(pin_model, pin_data)
    J_full = pin.getFrameJacobian(pin_model, pin_data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
    return np.asarray(J_full[:3, arm_qidx], dtype=np.float64)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_ik_step(
    *,
    pin_model: pin.Model,
    pin_data: pin.Data,
    q_arm_now: np.ndarray,                  # (n_arm,) current arm joint angles, rad
    ee_delta_normalized: np.ndarray,        # (3,) policy output Δxyz in [-1, 1]
    arm_qidx: list[int],                    # pinocchio q indices for arm slots
    ee_frame_id: int,                       # frame id of EE link (gripper_dummy_link)
    arm_joint_limits: np.ndarray,           # (n_arm, 2) (low, high) per slot, rad
    home_pose: np.ndarray,                  # (n_arm,) joint_init for null-space bias
    wrist_roll_slot: int,                   # which slot index is wrist_roll (clamp target)
    cfg: EEControlCfg,
    audit_state: IKAuditState,
    dt_s: float = 0.05,                     # 1 / policy_rate_hz, used for q_dot / q_jerk
) -> IKStepResult:
    """One IK step.

    Caller (``env._ik_step``) is responsible for:
      1) keeping a single ``IKAuditState`` across the episode (call
         ``reset_for_episode`` at the episode boundary, not here),
      2) translating the returned ``q_target_arm`` into a
         ``set_joint_position_target`` call on the articulation,
      3) reading ``truncate_episode`` and force-terminating the episode
         when True.
    """
    n_arm = len(arm_qidx)
    assert q_arm_now.shape == (n_arm,)
    assert ee_delta_normalized.shape == (3,)
    assert arm_joint_limits.shape == (n_arm, 2)
    assert home_pose.shape == (n_arm,)
    assert 0 <= wrist_roll_slot < n_arm

    q_arm_now = np.asarray(q_arm_now, dtype=np.float64)
    ee_delta_normalized = np.clip(
        np.asarray(ee_delta_normalized, dtype=np.float64), -1.0, 1.0
    )

    # ---- (1) Current EE pose via FK -------------------------------------
    q_full_now = _build_full_q(pin_model, q_arm_now, arm_qidx)
    ee_pos_now = _fk_ee_pos(pin_model, pin_data, q_full_now, ee_frame_id)

    # ---- (2) Jitter residual from previous step's commanded delta ------
    jitter_residual_now = 0.0
    if audit_state.last_ee_pos is not None and audit_state.last_commanded_delta is not None:
        expected_ee = audit_state.last_ee_pos + audit_state.last_commanded_delta
        jitter_residual_now = float(np.linalg.norm(ee_pos_now - expected_ee))
        audit_state.ee_residual_history.append(jitter_residual_now)

    # ---- (3) q_dot, q_jerk (rad/s, rad/s²) ------------------------------
    q_dot_now = np.zeros(n_arm)
    q_jerk_norm_now = 0.0
    if audit_state.last_q_arm is not None:
        q_dot_now = (q_arm_now - audit_state.last_q_arm) / dt_s
        if audit_state.last_q_dot is not None:
            q_jerk = (q_dot_now - audit_state.last_q_dot) / dt_s
            q_jerk_norm_now = float(np.linalg.norm(q_jerk))
            audit_state.q_jerk_norm_history.append(q_jerk_norm_now)

    # ---- (4) Build target EE (1st clamp via ee_delta_max_m) -------------
    ee_delta_m = ee_delta_normalized * float(cfg.ee_delta_max_m)
    # ``ee_delta_normalized`` is already in [-1, 1] so per-component cap
    # holds by construction; norm cap not enforced here (axis-aligned
    # cap is the documented design — see docs §3 step 3).
    target_ee = ee_pos_now + ee_delta_m

    # ---- (5) DLS + null-space bias --------------------------------------
    J = _position_jacobian(pin_model, pin_data, q_full_now, ee_frame_id, arm_qidx)
    x_err = (target_ee - ee_pos_now).astype(np.float64)
    x_err_norm = float(np.linalg.norm(x_err))
    lam2 = float(cfg.dls_lambda) ** 2
    JJT = J @ J.T
    JJT_det = float(np.linalg.det(JJT))
    JJT_damp = JJT + lam2 * np.eye(3)
    # Primary task — DLS.
    primary = J.T @ np.linalg.solve(JJT_damp, x_err)
    # Null-space secondary — pull toward home.
    N = np.eye(n_arm) - J.T @ np.linalg.solve(JJT_damp, J)
    secondary = float(cfg.null_bias_gain) * (home_pose - q_arm_now)
    q_dot_solve = primary + N @ secondary
    q_target_raw = q_arm_now + q_dot_solve

    # ---- (6) wrist_roll hard clamp --------------------------------------
    if cfg.wrist_roll_clamp_to_home:
        q_target_raw = q_target_raw.copy()
        q_target_raw[wrist_roll_slot] = float(home_pose[wrist_roll_slot])

    # ---- (7) joint target delta cap (2nd clamp) --------------------------
    q_delta_solve = q_target_raw - q_arm_now
    max_delta = float(cfg.ik_max_joint_delta_rad_per_step)
    q_delta_capped = np.clip(q_delta_solve, -max_delta, max_delta)
    q_target_capped = q_arm_now + q_delta_capped

    # ---- (8) joint soft-limit clamp -------------------------------------
    q_target_clamped = np.clip(q_target_capped, arm_joint_limits[:, 0], arm_joint_limits[:, 1])

    # ---- (9) commanded_ee_delta_actual = FK(q_target) - FK(q_arm) -------
    q_full_target = _build_full_q(pin_model, q_target_clamped, arm_qidx)
    ee_pos_expected = _fk_ee_pos(pin_model, pin_data, q_full_target, ee_frame_id)
    commanded_ee_delta_actual = ee_pos_expected - ee_pos_now

    # ---- (10) divergence guards -----------------------------------------
    fail_reason: str | None = None
    q_delta_norm_max = float(np.max(np.abs(q_delta_capped)))

    # Immediate AND: cap hit AND target still far. "Cap hit" detected by
    # the pre-cap delta exceeding the cap (a clean clamp leaves q_delta
    # at exactly max_delta in that direction).
    pre_cap_max = float(np.max(np.abs(q_dot_solve)))
    cap_hit = pre_cap_max > max_delta
    if cap_hit and x_err_norm > float(cfg.ik_max_position_error_m):
        # Categorize the specific physical cause if visible.
        if JJT_det < _SINGULAR_DET_THRESHOLD:
            fail_reason = FAIL_SINGULAR
        elif x_err_norm > _UNREACHABLE_X_ERR_M:
            fail_reason = FAIL_UNREACHABLE
        elif np.any(np.abs(q_target_capped - arm_joint_limits[:, 0]) < _JOINT_LIMIT_TOL) or \
             np.any(np.abs(q_target_capped - arm_joint_limits[:, 1]) < _JOINT_LIMIT_TOL):
            fail_reason = FAIL_JOINT_LIMIT
        else:
            fail_reason = FAIL_LARGE_QDOT

    # Immediate OR: jitter / jerk p95 over rolling window. Only active
    # when threshold > 0 (set after M0.5 calibration; 0.0 = disabled).
    if fail_reason is None and cfg.ik_jitter_residual_p95_max_m > 0.0:
        if len(audit_state.ee_residual_history) >= cfg.ik_jitter_window:
            recent_residuals = list(audit_state.ee_residual_history)[-cfg.ik_jitter_window:]
            jitter_p95 = float(np.percentile(recent_residuals, 95))
            if jitter_p95 > cfg.ik_jitter_residual_p95_max_m:
                fail_reason = FAIL_JITTER

    if fail_reason is None and cfg.ik_jerk_p95_max_rad_s2 > 0.0:
        if len(audit_state.q_jerk_norm_history) >= cfg.ik_jitter_window:
            recent_jerks = list(audit_state.q_jerk_norm_history)[-cfg.ik_jitter_window:]
            jerk_p95 = float(np.percentile(recent_jerks, 95))
            if jerk_p95 > cfg.ik_jerk_p95_max_rad_s2:
                fail_reason = FAIL_JITTER

    fallback_used = fail_reason is not None

    # Fallback: hold last valid q_target if any guard tripped.
    if fallback_used:
        if audit_state.last_valid_q_target is not None:
            q_target_final = audit_state.last_valid_q_target.copy()
            # Recompute commanded delta for the held target — for jitter
            # accounting next step we want this to be the delta the env
            # actually issues.
            q_full_held = _build_full_q(pin_model, q_target_final, arm_qidx)
            ee_pos_held = _fk_ee_pos(pin_model, pin_data, q_full_held, ee_frame_id)
            commanded_ee_delta_actual = ee_pos_held - ee_pos_now
        else:
            # First-step fallback (no prior valid target): hold current q.
            q_target_final = q_arm_now.copy()
            commanded_ee_delta_actual = np.zeros(3, dtype=np.float64)
        audit_state.consecutive_ik_fails += 1
        audit_state.ik_fail_count += 1
        audit_state.fail_reasons[fail_reason] = (
            audit_state.fail_reasons.get(fail_reason, 0) + 1
        )
    else:
        q_target_final = q_target_clamped
        audit_state.consecutive_ik_fails = 0
        audit_state.last_valid_q_target = q_target_final.copy()

    # Slow divergence — monotone increasing x_err over rolling window.
    audit_state.x_err_history.append(x_err_norm)
    slow_diverge_now = False
    if len(audit_state.x_err_history) >= cfg.ik_slow_diverge_window:
        recent_x_err = list(audit_state.x_err_history)[-cfg.ik_slow_diverge_window:]
        # Strict monotone increase across the whole window.
        if np.all(np.diff(np.asarray(recent_x_err)) > 0):
            slow_diverge_now = True
            audit_state.slow_diverge_count += 1

    truncate_episode = (
        audit_state.consecutive_ik_fails >= cfg.ik_consecutive_fail_truncate
    )

    # ---- (11) update audit_state for next step's delta computations -----
    audit_state.last_ee_pos = ee_pos_now.copy()
    audit_state.last_commanded_delta = commanded_ee_delta_actual.copy()
    audit_state.last_q_arm = q_arm_now.copy()
    audit_state.last_q_dot = q_dot_now.copy()

    audit = {
        "ee_pos_x": float(ee_pos_now[0]),
        "ee_pos_y": float(ee_pos_now[1]),
        "ee_pos_z": float(ee_pos_now[2]),
        "x_err_norm_m": x_err_norm,
        "q_delta_pre_cap_max_rad": pre_cap_max,
        "q_delta_post_cap_max_rad": q_delta_norm_max,
        "jitter_residual_m": jitter_residual_now,
        "q_jerk_norm_rad_s2": q_jerk_norm_now,
        "JJT_det": JJT_det,
        "consecutive_ik_fails": int(audit_state.consecutive_ik_fails),
        "ik_fail_count": int(audit_state.ik_fail_count),
        "slow_diverge_count": int(audit_state.slow_diverge_count),
        "slow_diverge_now": bool(slow_diverge_now),
        "fail_reason": fail_reason or "",
        "commanded_ee_delta_x": float(commanded_ee_delta_actual[0]),
        "commanded_ee_delta_y": float(commanded_ee_delta_actual[1]),
        "commanded_ee_delta_z": float(commanded_ee_delta_actual[2]),
    }

    return IKStepResult(
        q_target_arm=q_target_final.astype(np.float32),
        fallback_used=fallback_used,
        fail_reason=fail_reason,
        truncate_episode=truncate_episode,
        audit=audit,
    )
