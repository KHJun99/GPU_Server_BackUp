"""Reward-hacking negative scenarios (per docs/method_b_design.md §6 M0.5 Step C).

Each test forces a synthetic episode state sequence through
``MultiConditionSuccess`` and asserts the judge rejects it (final
``SuccessReport.success == False``). Each scenario also asserts which
specific condition fired, so a future relaxation of any single check
that silently re-opens a known hacking pattern fails loudly here.

Scope (v7 plan §6 M0.5 Step C): the v7 plan lists 12 scenarios. This
file implements the 6 that the **current** 5-condition success.py can
already reject without the dormant ``SuccessGuardCfg`` sub-checks:

  1. Horizontal cube slide to goal (jabis_sim_v2 rolling hack)        → cond #4 vel_stab
  2. Cube reaches goal without ever being lifted                       → cond #1 lift_history
  3. Cube briefly in-radius then bounces out                           → cond #2 stable_placement
  4. Visibility loss at the success-check step                         → cond #5 visual_agreement
  7. Placement at goal but cube has non-zero velocity                  → cond #4 vel_stab
  9. Gripper never opens (no release)                                  → cond #3 release_retreat

The remaining v7 scenarios (#5 stuck cube, #6 terminal IK fail, #8 lift
then drop+slide, #10 release with no contact separation, #11 IK fail
outside terminal, #12 cube tipped at goal) require sub-checks in
``SuccessGuardCfg`` (contact_separation_min_m, ik_fail_mask_*,
cube_max_tip_angle_rad, gripper_match_*) that are still dormant. Those
land in M2 alongside the success.py rewrite.

A separate ``TestBaselineSanity::test_baseline_episode_passes`` confirms
the baseline trajectory builder itself does produce success — otherwise
the negative tests could be trivially passing by accident.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from khj_rl.envs.cube_lift.cfg import CubeLiftEnvCfg
from khj_rl.envs.cube_lift.success import (
    MultiConditionSuccess,
    SuccessReport,
    SuccessState,
)


# ---------------------------------------------------------------------------
# Episode geometry — picks values that satisfy / violate the 5 conditions
# cleanly so each test asserts a single failure mode.
# ---------------------------------------------------------------------------

CFG = CubeLiftEnvCfg()
GOAL_XYZ = np.asarray(CFG.goal.pos_xyz_m, dtype=np.float64)
GOAL_XY = GOAL_XYZ[:2]
PLACE_Z = 0.023  # cube center at rest on table ≈ table_top + size/2 + clearance
LIFT_Z_CLEAR = CFG.success.lift_z_m + 0.02  # 0.10 m — well above 0.08 threshold
GRIPPER_CLOSED = 0.20  # ≤ lift_gripper_max_open (0.85)
GRIPPER_OPEN = 0.95     # ≥ gripper_open_threshold (0.8)
ZERO_VEL = np.zeros(3, dtype=np.float64)
TINY_VEL = np.array([0.02, 0.0, 0.0], dtype=np.float64)  # ≤ cube_v_max (0.05)
TINY_OMEGA = np.array([0.0, 0.0, 0.1], dtype=np.float64)  # ≤ cube_omega_max (~0.52)

# Total episode length matches CFG.control.max_steps (160 @ 20 Hz, 8 s).
EPISODE_LEN = CFG.control.max_steps
LIFT_LATCH_LEN = CFG.success.lift_hold_steps         # 20
PLACE_DWELL_LEN = CFG.success.place_hold_steps       # 20


def _default_state(**overrides) -> SuccessState:
    """Build a SuccessState with all-passing defaults, override per scenario.

    Defaults: cube at goal, both cameras see it, gripper closed (lift posture),
    EE on top of cube, zero velocities. Each scenario flips just the fields
    relevant to its hack pattern so the assertion can pin the failing cond.
    """
    base = {
        "cube_xyz_m": np.array([GOAL_XYZ[0], GOAL_XYZ[1], PLACE_Z], dtype=np.float64),
        "cube_lin_vel_m_s": ZERO_VEL.copy(),
        "cube_ang_vel_rad_s": ZERO_VEL.copy(),
        "ee_xyz_m": np.array([GOAL_XYZ[0], GOAL_XYZ[1], PLACE_Z + 0.01], dtype=np.float64),
        "gripper_opening": GRIPPER_CLOSED,
        "visibility_top": True,
        "visibility_wrist": True,
    }
    base.update(overrides)
    return SuccessState(**base)


def _judge() -> MultiConditionSuccess:
    return MultiConditionSuccess(
        goal=CFG.goal,
        success_cfg=CFG.success,
        gripper_open_threshold=CFG.robot.gripper_open_threshold,
    )


def _run(states: list[SuccessState]) -> SuccessReport:
    """Drive a sequence through the judge, return the FINAL report."""
    j = _judge()
    j.reset()
    report = SuccessReport(success=False, per_condition={})
    for s in states:
        report = j.update(s)
    return report


# ---------------------------------------------------------------------------
# Trajectory building blocks
# ---------------------------------------------------------------------------


def _lift_sequence(n: int = LIFT_LATCH_LEN) -> list[SuccessState]:
    """``n`` consecutive frames that satisfy every lift_history sub-condition.

    Used as the lift prefix in scenarios that need lift_history to latch
    so the rejection later in the episode is unambiguously caused by a
    different condition.
    """
    # Cube lifted, EE on top, gripper closed, near-zero velocity.
    s = _default_state(
        cube_xyz_m=np.array([GOAL_XYZ[0] - 0.05, GOAL_XYZ[1], LIFT_Z_CLEAR], dtype=np.float64),
        ee_xyz_m=np.array([GOAL_XYZ[0] - 0.05, GOAL_XYZ[1], LIFT_Z_CLEAR + 0.02], dtype=np.float64),
        gripper_opening=GRIPPER_CLOSED,
    )
    return [s] * n


def _place_sequence(n: int = PLACE_DWELL_LEN) -> list[SuccessState]:
    """``n`` consecutive frames satisfying stable_placement (in_radius + z_ok)."""
    s = _default_state(
        cube_xyz_m=np.array([GOAL_XYZ[0], GOAL_XYZ[1], PLACE_Z], dtype=np.float64),
        # Cube still being held — gripper closed, EE close. Release happens later.
        ee_xyz_m=np.array([GOAL_XYZ[0], GOAL_XYZ[1], PLACE_Z + 0.01], dtype=np.float64),
        gripper_opening=GRIPPER_CLOSED,
    )
    return [s] * n


def _release_retreat_sequence(n: int) -> list[SuccessState]:
    """``n`` frames where gripper is open and EE has retreated from cube."""
    cube_at_goal = np.array([GOAL_XYZ[0], GOAL_XYZ[1], PLACE_Z], dtype=np.float64)
    ee_above = cube_at_goal + np.array([0.0, 0.0, 0.10], dtype=np.float64)
    s = _default_state(
        cube_xyz_m=cube_at_goal,
        ee_xyz_m=ee_above,
        gripper_opening=GRIPPER_OPEN,
    )
    return [s] * n


def _baseline_successful_episode() -> list[SuccessState]:
    """Full 160-step trajectory that passes all 5 conditions at the terminal step.

    Phases (step counts):
      [0, 20)   lift (20 step, latches lift_history)
      [20, 80)  carry (60 step, lifted, in transit — no constraint enforced here)
      [80, 140) place dwell (60 step, > place_hold_steps so stable_placement holds)
      [140, 160) release + retreat (20 step, gripper open + EE away)

    Total = 20 + 60 + 60 + 20 = 160 = EPISODE_LEN.
    """
    lift = _lift_sequence(20)
    # Carry: cube lifted but moving toward goal; pad with the same lift frames
    # so velocity/visibility stay clean.
    carry = _lift_sequence(60)
    place = _place_sequence(60)
    release = _release_retreat_sequence(20)
    seq = lift + carry + place + release
    assert len(seq) == EPISODE_LEN, f"baseline length {len(seq)} != {EPISODE_LEN}"
    return seq


# ---------------------------------------------------------------------------
# Meta-sanity: baseline must actually succeed, otherwise negative tests are
# trivially passing.
# ---------------------------------------------------------------------------


class TestBaselineSanity:
    def test_baseline_episode_passes(self) -> None:
        report = _run(_baseline_successful_episode())
        assert report.success, (
            f"baseline episode must produce success=True, but per_condition={report.per_condition}"
        )


# ---------------------------------------------------------------------------
# v7 negative scenarios — now-implementable subset (6 of 12)
# ---------------------------------------------------------------------------


class TestNegativeScenarios:
    """All 6 negative scenarios must produce success=False at the final step."""

    def test_scenario_1_horizontal_slide_to_goal(self) -> None:
        """v7 #1: cube slides horizontally to goal (jabis_sim_v2 rolling hack).

        Cube stays on table the whole time (cube_z = PLACE_Z), moves at
        v_xy = 0.10 m/s ≫ cube_v_max=0.05, ends at goal xy. No lift, no
        grasp, no release ever happens.

        Expected rejection: cond #4 vel_stab (and cond #1 lift_history,
        cond #3 release_retreat both fail too).
        """
        moving_vel = np.array([0.10, 0.0, 0.0], dtype=np.float64)
        seq = [
            _default_state(
                cube_xyz_m=np.array([GOAL_XYZ[0], GOAL_XYZ[1], PLACE_Z], dtype=np.float64),
                cube_lin_vel_m_s=moving_vel,
                ee_xyz_m=np.array([0.0, 0.0, 0.30], dtype=np.float64),  # EE far away (no grasp)
                gripper_opening=0.5,
            )
            for _ in range(EPISODE_LEN)
        ]
        report = _run(seq)
        assert not report.success, f"per_condition={report.per_condition}"
        assert not report.per_condition["velocity_stability"], (
            "vel_stab must reject horizontal slide; it is the direct block "
            "against jabis_sim_v2 rolling hack"
        )
        assert not report.per_condition["lift_history"], (
            "lift_history must also fail (cube never lifted)"
        )

    def test_scenario_2_no_lift_goal_reached(self) -> None:
        """v7 #2: cube reaches goal without ever being lifted.

        Cube starts at goal already (or arrives via slide), comes to rest.
        vel_stab passes, stable_placement passes, but lift_history never
        latches because cube_z stays at PLACE_Z (< lift_z_m=0.08).

        Expected rejection: cond #1 lift_history.
        """
        seq = [
            _default_state(
                cube_xyz_m=np.array([GOAL_XYZ[0], GOAL_XYZ[1], PLACE_Z], dtype=np.float64),
                ee_xyz_m=np.array([0.0, 0.0, 0.30], dtype=np.float64),
                gripper_opening=GRIPPER_OPEN,
            )
            for _ in range(EPISODE_LEN)
        ]
        report = _run(seq)
        assert not report.success
        assert not report.per_condition["lift_history"], (
            "lift_history must reject no-lift trajectory (cube_z < lift_z_m)"
        )

    def test_scenario_3_bounce_in_and_out_of_goal(self) -> None:
        """v7 #3: cube briefly enters goal radius then is out at terminal.

        place_history is a rolling 20-step deque of (in_radius AND z_ok).
        stable_placement is True only if all 20 are True. The bounce hack:
        cube briefly satisfies placement then leaves the goal radius,
        and at the success-check step is no longer there. We construct:

          lift(20) + carry(60) + briefly_in(5) + out_until_end(75)

        End-state: cube parked just outside goal radius, at rest, gripper
        open, EE retreated. All 4 other conditions pass at terminal so
        the failure is unambiguously stable_placement.

        Expected rejection: cond #2 stable_placement.
        """
        lift = _lift_sequence(20)
        carry = _lift_sequence(60)
        briefly_in = _place_sequence(5)
        # Cube parked 10 cm outside goal (radius_xy_m=0.03), at rest, gripper
        # opened, EE retreated. Every other terminal-evaluated condition
        # (lift_history latched earlier, release_retreat, vel_stab,
        # visual_agreement) stays True so the only failing cond is placement.
        cube_out = np.array([GOAL_XYZ[0] + 0.10, GOAL_XYZ[1], PLACE_Z], dtype=np.float64)
        ee_away = cube_out + np.array([0.0, 0.0, 0.10], dtype=np.float64)
        out_until_end = [
            _default_state(
                cube_xyz_m=cube_out,
                ee_xyz_m=ee_away,
                gripper_opening=GRIPPER_OPEN,
            )
            for _ in range(75)
        ]
        seq = lift + carry + briefly_in + out_until_end
        assert len(seq) == EPISODE_LEN
        report = _run(seq)
        assert not report.success, f"per_condition={report.per_condition}"
        assert not report.per_condition["stable_placement"], (
            "stable_placement must reject when cube is out of goal radius at "
            "terminal regardless of any prior in-radius visit"
        )
        # Sanity: only stable_placement fails. Lift latched earlier, gripper
        # open + EE retreated = release_retreat True, stationary cube = vel_stab,
        # both cameras default True = visual_agreement.
        for k in ("lift_history", "release_retreat", "velocity_stability", "visual_agreement"):
            assert report.per_condition[k], (
                f"isolated stable_placement violation should leave {k} True; "
                f"got per_condition={report.per_condition}"
            )

    def test_scenario_4_visibility_loss_at_terminal(self) -> None:
        """v7 #4: successful trajectory but cameras lose cube at success-check step.

        Run the baseline episode but flip ``visibility_top=False`` at the
        very last frame. All other 4 conditions still hold at that step.

        Expected rejection: cond #5 visual_agreement.
        """
        seq = _baseline_successful_episode()
        # Replace the last frame with one missing top visibility.
        seq[-1] = replace(seq[-1], visibility_top=False)
        report = _run(seq)
        assert not report.success
        assert not report.per_condition["visual_agreement"], (
            "visual_agreement must reject when either camera loses the cube"
        )
        # Sanity: all other conditions still pass at terminal.
        for k in ("lift_history", "stable_placement", "release_retreat", "velocity_stability"):
            assert report.per_condition[k], (
                f"baseline-derived terminal should keep {k} True; got "
                f"per_condition={report.per_condition}"
            )

    def test_scenario_7_high_velocity_at_correct_placement(self) -> None:
        """v7 #7: cube is at goal with correct geometry but moving too fast.

        Baseline trajectory through placement, but at the terminal step
        the cube has ``v = 0.10 m/s`` (above ``cube_v_max = 0.05``). All
        other 4 conditions still hold.

        Expected rejection: cond #4 vel_stab. This is the targeted
        re-test of the jabis_sim_v2 hacking failure mode.
        """
        seq = _baseline_successful_episode()
        seq[-1] = replace(
            seq[-1],
            cube_lin_vel_m_s=np.array([0.10, 0.0, 0.0], dtype=np.float64),
        )
        report = _run(seq)
        assert not report.success
        assert not report.per_condition["velocity_stability"]
        # Sanity: other 4 still pass — the only failure is vel_stab.
        for k in ("lift_history", "stable_placement", "release_retreat", "visual_agreement"):
            assert report.per_condition[k], (
                f"isolated vel_stab violation should leave {k} True; got "
                f"per_condition={report.per_condition}"
            )

    def test_scenario_9_gripper_never_opens(self) -> None:
        """v7 #9: cube is placed correctly but gripper stays closed the whole time.

        Lift sequence + carry + place dwell pass cleanly, but the release
        phase keeps ``gripper_opening = GRIPPER_CLOSED`` instead of
        opening to GRIPPER_OPEN. release_retreat requires
        ``gripper_opening >= gripper_open_threshold (0.8)`` so it fails
        even if the EE retreats away.

        Expected rejection: cond #3 release_retreat.
        """
        lift = _lift_sequence(20)
        carry = _lift_sequence(60)
        place = _place_sequence(60)
        # Replace the release phase with a "fake retreat" where the EE pulls
        # away but the gripper is never opened. cube velocity stays zero
        # (cube stays at goal even with closed gripper because we're
        # constructing forced states, not simulating physics).
        fake_release = [
            _default_state(
                cube_xyz_m=np.array([GOAL_XYZ[0], GOAL_XYZ[1], PLACE_Z], dtype=np.float64),
                ee_xyz_m=np.array([GOAL_XYZ[0], GOAL_XYZ[1], PLACE_Z + 0.10], dtype=np.float64),
                gripper_opening=GRIPPER_CLOSED,  # ← stays closed
            )
            for _ in range(20)
        ]
        seq = lift + carry + place + fake_release
        assert len(seq) == EPISODE_LEN
        report = _run(seq)
        assert not report.success
        assert not report.per_condition["release_retreat"], (
            "release_retreat must reject when gripper never opens "
            "(released = gripper_opening >= 0.8)"
        )
