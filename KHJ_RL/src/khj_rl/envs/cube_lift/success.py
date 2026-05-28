"""Phase 1 multi-condition success criterion.

Five conditions must all hold simultaneously for +1 reward (CLAUDE.md
## Phase 1 결정 사항 5). The judge owns only rolling history buffers, so
the env can construct one instance and call ``update()`` each step.

Condition #4 (velocity stability) is the direct block against the
jabis_sim_v2 "rolling cube" reward-hacking failure mode.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

from khj_rl.envs.cube_lift.cfg import GoalCfg, SuccessCfg


@dataclass
class SuccessState:
    """Per-step state the judge consumes. All in robot base frame.

    Built by the env from GT pose so the success signal is never contaminated
    by perception noise.
    """

    cube_xyz_m: np.ndarray            # (3,)
    cube_lin_vel_m_s: np.ndarray      # (3,)
    cube_ang_vel_rad_s: np.ndarray    # (3,)
    ee_xyz_m: np.ndarray              # (3,)
    gripper_opening: float            # 0..1
    visibility_top: bool              # GT-pose visibility proxy (env decides)
    visibility_wrist: bool


@dataclass
class SuccessReport:
    success: bool
    per_condition: dict[str, bool]


class MultiConditionSuccess:
    """Stateful judge that owns only rolling history buffers."""

    def __init__(
        self,
        goal: GoalCfg,
        success_cfg: SuccessCfg,
        gripper_open_threshold: float,
    ) -> None:
        self._goal = goal
        self._cfg = success_cfg
        self._gripper_open_threshold = gripper_open_threshold
        # #1 lift_history is LATCHING: once the cube spends `lift_hold_steps`
        # consecutive steps at z >= lift_z_m the condition flips to True
        # for the rest of the episode. The earlier "rolling window" form
        # was structurally incompatible with PnP — stable_placement
        # requires the cube to be at goal (i.e. NOT lifted) at the
        # episode end, so the two could never be True at the same step.
        # Latching captures the spirit of "the cube was actually picked
        # up" while letting stable_placement / release_retreat reflect
        # the final-state requirements.
        self._lift_history: Deque[bool] = deque(maxlen=success_cfg.lift_hold_steps)
        self._lift_ever_latched: bool = False
        self._place_history: Deque[bool] = deque(maxlen=success_cfg.place_hold_steps)

    def reset(self) -> None:
        self._lift_history.clear()
        self._lift_ever_latched = False
        self._place_history.clear()

    def update(self, state: SuccessState) -> SuccessReport:
        per: dict[str, bool] = {}

        # #1 lift history — latching once the rolling buffer has logged
        # ``lift_hold_steps`` consecutive lifted frames. Once latched
        # we stop checking the buffer (and stop appending to it) — the
        # condition can never become False again within the episode.
        if not self._lift_ever_latched:
            # F8 / #27 anti-pressing v1: cube_z alone is hackable. Require
            # gripper grasping conditions to be true at the same step.
            # F15 anti-pressing v2 (post F14 STEP 5 squash hack): also require
            # cube velocity stability — real lift keeps cube xy stationary +
            # no rotation, squash hack bounces cube horizontally + tumbles.
            cube_above = float(state.cube_xyz_m[2]) >= self._cfg.lift_z_m
            ee_cube_dist = float(np.linalg.norm(
                np.asarray(state.ee_xyz_m, dtype=np.float64)
                - np.asarray(state.cube_xyz_m, dtype=np.float64)
            ))
            ee_close = ee_cube_dist <= self._cfg.lift_ee_cube_max_dist_m
            gripper_closed = (
                float(state.gripper_opening) <= self._cfg.lift_gripper_max_open
            )
            v_xy = float(np.linalg.norm(
                np.asarray(state.cube_lin_vel_m_s, dtype=np.float64)[:2]
            ))
            omega = float(np.linalg.norm(
                np.asarray(state.cube_ang_vel_rad_s, dtype=np.float64)
            ))
            cube_stable_xy = v_xy <= self._cfg.lift_cube_v_xy_max
            cube_not_tumbling = omega <= self._cfg.lift_cube_omega_max
            lifted_now = (
                cube_above and ee_close and gripper_closed
                and cube_stable_xy and cube_not_tumbling
            )
            self._lift_history.append(lifted_now)
            if (
                len(self._lift_history) == self._cfg.lift_hold_steps
                and all(self._lift_history)
            ):
                self._lift_ever_latched = True
        per["lift_history"] = self._lift_ever_latched

        # #2 stable placement.
        # 2026-05-20 high-release patch (codex review): also require the
        # cube to be physically lowered to table_top level (cube_z <=
        # place_z_max_m). An in-air gripper open lets the cube free-fall
        # through the goal radius but cube_z stays above the cap until
        # impact, so this cap blocks that.
        goal_xy = np.asarray(self._goal.pos_xyz_m[:2], dtype=np.float64)
        cube_xy = np.asarray(state.cube_xyz_m[:2], dtype=np.float64)
        in_radius = float(np.linalg.norm(cube_xy - goal_xy)) <= self._goal.radius_xy_m
        cube_z = float(state.cube_xyz_m[2])
        z_ok = cube_z <= self._cfg.place_z_max_m
        self._place_history.append(in_radius and z_ok)
        per["stable_placement"] = (
            len(self._place_history) == self._cfg.place_hold_steps
            and all(self._place_history)
        )

        # #3 release + retreat.
        released = state.gripper_opening >= self._gripper_open_threshold
        ee_dist = float(np.linalg.norm(
            np.asarray(state.ee_xyz_m, dtype=np.float64)
            - np.asarray(state.cube_xyz_m, dtype=np.float64)
        ))
        retreated = ee_dist >= self._cfg.retreat_distance_m
        per["release_retreat"] = released and retreated

        # #4 velocity stability (rolling-block).
        v = float(np.linalg.norm(state.cube_lin_vel_m_s))
        w = float(np.linalg.norm(state.cube_ang_vel_rad_s))
        per["velocity_stability"] = (
            v <= self._cfg.cube_v_max and w <= self._cfg.cube_omega_max
        )

        # #5 visual agreement (env supplies the proxy bools).
        if self._cfg.require_visual_agreement:
            per["visual_agreement"] = bool(state.visibility_top and state.visibility_wrist)
        else:
            per["visual_agreement"] = True

        return SuccessReport(success=all(per.values()), per_condition=per)
