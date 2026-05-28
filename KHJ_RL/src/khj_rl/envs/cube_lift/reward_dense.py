"""Dense reward shaping for Phase 1 cube-lift (A option).

Activated via cfg.dense_reward.enabled=True. Bounded + annealed +
phase-gated; see docs/dense_reward_design.md for the safety contract
and Codex review (2026-05-17).

DISABLED by default (cfg.dense_reward.enabled=False). Activation is
the result of the explicit decision recorded in CLAUDE.md 결정사항 5
(A 옵션 도입 결정 — 2026-05-17).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from khj_rl.envs.cube_lift.cfg import DenseRewardCfg, GoalCfg, SuccessCfg
from khj_rl.envs.cube_lift.success import SuccessState


@dataclass
class DenseRewardReport:
    """Step-level dense reward + per-term breakdown for logging.

    per_term keys MUST match the cfg field stem names so a dashboard
    can correlate weight -> term -> outcome without code duplication.
    """

    total: float
    per_term: dict[str, float]


def compute_dense_reward(
    state: SuccessState,
    goal: GoalCfg,
    success_cfg: SuccessCfg,
    dense_cfg: DenseRewardCfg,
    gripper_open_threshold: float,
    lift_history: bool = False,
) -> DenseRewardReport:
    """Step dense reward (pre-anneal). All terms safe by construction:
    bounded, phase-gated, saturating; see docs/dense_reward_design.md.
    """
    cube = state.cube_xyz_m
    ee = state.ee_xyz_m

    d_ee_cube = float(np.linalg.norm(ee - cube))
    lifted = float(cube[2] >= success_cfg.lift_z_m)

    goal_xy = np.asarray(goal.pos_xyz_m[:2], dtype=np.float32)
    d_cube_goal_xy_raw = float(np.linalg.norm(cube[:2] - goal_xy))
    # Saturate so a far-away cube right after lift doesn't dominate the
    # signal — encourages "move toward goal until close" without
    # over-reacting on the initial transport segment.
    d_cube_goal_xy = min(d_cube_goal_xy_raw, dense_cfg.dist_cap)

    # released = success #3 base condition (gripper open + EE retreated).
    released_now = float(
        state.gripper_opening >= gripper_open_threshold
        and d_ee_cube >= success_cfg.retreat_distance_m
    )
    # Codex review (2026-05-17) added gates: require cube near goal AND
    # cube stable so trivial release (gripper open + walk away) does
    # NOT score points unless the placement is actually done.
    near_goal = float(d_cube_goal_xy_raw < dense_cfg.near_goal_radius_xy)
    stable = float(
        float(np.linalg.norm(state.cube_lin_vel_m_s)) < dense_cfg.velocity_stable_lin
        and float(np.linalg.norm(state.cube_ang_vel_rad_s)) < dense_cfg.velocity_stable_ang
    )

    terms = {
        # ↓ EE -> cube attraction. Weight 0.5 (down from 1.0) so the
        #   approach signal doesn't dominate the lifted_bonus.
        "ee_cube_dist": -dense_cfg.w_ee_cube_dist * d_ee_cube,
        # ↓ Per-step lift maintenance. Weight 0.03 (down from 0.1) per
        #   Codex 2026-05-17 — larger weights produce a hover attractor
        #   that was the predicted main failure mode.
        "lifted_bonus": dense_cfg.w_lifted_bonus * lifted,
        # ↓ Goal attraction once lifted. Saturated by dist_cap and
        #   gated by `* lifted` so cube-rolling never scores.
        "cube_goal_dist": -dense_cfg.w_cube_goal_dist * d_cube_goal_xy * lifted,
        # ↓ Released bonus only when (lift_history + release + near goal + stable)
        #   all hold. lift_history gate prevents trivial exploit where agent
        #   does nothing when cube spawns near goal.
        "released_bonus": dense_cfg.w_released_bonus * float(lift_history) * released_now * near_goal * stable,
    }
    return DenseRewardReport(total=sum(terms.values()), per_term=terms)


def dense_alpha(step: int, dense_cfg: DenseRewardCfg) -> float:
    """Linear anneal: dense_cfg.alpha0 at step 0 -> 0 at decay_steps.

    Linear (vs exponential) so the late-training reward is exactly pure
    sparse — eliminates residual shaping bias on the converged policy.
    Codex 2026-05-17 set decay_steps shorter than total_steps so the
    final ~200k env steps verify the policy under sparse-only.
    """
    if dense_cfg.decay_steps <= 0:
        return 0.0
    frac = max(0.0, 1.0 - step / dense_cfg.decay_steps)
    return dense_cfg.alpha0 * frac
