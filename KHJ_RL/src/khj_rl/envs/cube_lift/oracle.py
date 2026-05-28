"""MP-based oracle policy for BC demo collection.

6-phase scripted policy (approach → descend → grasp → lift → move → place
→ retreat) driving the SO-ARM101 5-DOF arm via an Isaac Lab
``DifferentialIKController`` in damped-least-squares mode and the
PincOpen gripper through the same 6-D normalized action the policy will
later receive (5 arm deltas + 1 gripper open/close).

The oracle owns no sim state; it holds a reference to a constructed
``CubeLiftEnv`` and reads GT poses + jacobian directly from the
articulation. Demos collected this way feed BC pretrain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from khj_rl.envs.cube_lift.cfg import CubeLiftEnvCfg


class OraclePolicy(Protocol):
    def reset(self, obs: np.ndarray, info: dict) -> None: ...
    def act(self, obs: np.ndarray, info: dict) -> np.ndarray: ...


@dataclass(frozen=True)
class _PhaseSpec:
    """One MP phase: target generator, gripper command, exit conditions."""

    name: str
    gripper_open: bool
    # (cube_xyz, goal_xyz) -> target xyz (in robot base frame).
    target_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    # Phase exits when the EE is within this many meters of `target_fn`'s output
    # AND `min_steps` have passed; or when `max_steps` is hit (timeout fallback
    # so a stuck phase doesn't freeze the whole episode).
    exit_distance_m: float
    max_steps: int
    min_steps: int = 1


# Heights are in robot base frame. lift_z = 0.12 sits comfortably above
# success.py's lift_z_m=0.08; retreat_z=0.18 keeps the EE 6cm above the
# released cube (matches success retreat_distance_m=0.06 floor).
_LIFT_Z = 0.12
_RETREAT_Z = 0.18

# Phase targets are expressed in *fingertip-center* world coords. Oracle
# converts those to the dummy_link (IK EE) target at runtime by
# subtracting the current dummy→fingertip offset, recomputed each step
# via Pinocchio FK. Static offsets stopped working as soon as the wrist
# rotated a few degrees during approach — fingers ended up one cube
# edge ahead of the cube.


def _build_phases() -> tuple[_PhaseSpec, ...]:
    # target_fn returns the *fingertip-center* desired position in the
    # robot base frame. Oracle.act() translates that to the dummy_link
    # target each step by subtracting the current FK-derived offset.
    return (
        _PhaseSpec(
            name="approach",
            gripper_open=True,
            # Fingertip 8cm above cube. Home pose already puts the
            # fingertip near (0.25, 0, 0.1) so this is a few-cm tweak.
            target_fn=lambda cube, goal: np.array(
                [cube[0], cube[1], cube[2] + 0.08], dtype=np.float32
            ),
            exit_distance_m=0.03,
            max_steps=60,
        ),
        _PhaseSpec(
            name="descend",
            gripper_open=True,
            # Drive fingertip-center to cube center. With the updated
            # home pose (sl=-0.70, ef=+1.00, wf=+0.40) the FK sweep
            # confirms this is reachable: home puts fingers at z≈0.09
            # and the arm has plenty of pitch left to descend ~7 cm
            # without hitting joint limits. Fingertips end up on the
            # cube's vertical faces (not on top), so the grasp close
            # wraps the cube SIDES instead of catching the upper edge
            # corner.
            target_fn=lambda cube, goal: np.array(
                [cube[0], cube[1], cube[2]], dtype=np.float32
            ),
            exit_distance_m=0.015,
            max_steps=60,
            min_steps=15,
        ),
        _PhaseSpec(
            name="grasp",
            gripper_open=False,
            # Arm holds the fingertip at cube center; only the gripper
            # PD moves. The effort-limited actuator (5 N·m, matching
            # real STS3215 stall) stops the close on cube contact so
            # the fingers settle at whatever spread the object needs.
            target_fn=lambda cube, goal: np.array(
                [cube[0], cube[1], cube[2]], dtype=np.float32
            ),
            exit_distance_m=0.04,
            max_steps=80,
            min_steps=40,
        ),
        _PhaseSpec(
            name="lift",
            gripper_open=False,
            target_fn=lambda cube, goal: np.array(
                [cube[0], cube[1], _LIFT_Z], dtype=np.float32
            ),
            exit_distance_m=0.015,
            # min_steps holds the cube at LIFT_Z long enough that the
            # success.py rolling-window collects 20 consecutive
            # ``z >= lift_z_m`` (= 8 cm) frames before lift exits.
            # Without this, lift would finish in ~25 steps and the
            # cube spent only ~12-16 frames above 8 cm before move +
            # place dropped it — short of the latch window. 30 steps
            # = 1.5 s of pure dwell at z = 12 cm is enough to fill
            # the buffer with margin.
            max_steps=60,
            min_steps=30,
        ),
        _PhaseSpec(
            name="move",
            gripper_open=False,
            target_fn=lambda cube, goal: np.array(
                [goal[0], goal[1], _LIFT_Z], dtype=np.float32
            ),
            exit_distance_m=0.015,
            # Light dwell at goal-XY high pose so the cube doesn't
            # plunge straight from lift into place — gives the
            # velocity_stability success cond breathing room too.
            max_steps=40,
            min_steps=10,
        ),
        _PhaseSpec(
            name="place",
            gripper_open=False,
            target_fn=lambda cube, goal: np.array(
                [goal[0], goal[1], goal[2]], dtype=np.float32
            ),
            exit_distance_m=0.015,
            max_steps=30,
        ),
        _PhaseSpec(
            name="retreat",
            gripper_open=True,
            # Retreat off to the side so the EE doesn't sit directly
            # above the placed cube — the success.py top-camera
            # visibility proxy returns False when the EE is within
            # 5 cm xy AND above the cube, which the old "straight
            # up over goal" retreat triggered 100% of the time.
            # Stepping 8 cm back along -X keeps the EE inside the
            # 30 cm wrist-cam radius (the other side of the proxy)
            # while clearing the top-cam occlusion test.
            target_fn=lambda cube, goal: np.array(
                [goal[0] - 0.08, goal[1], _RETREAT_Z], dtype=np.float32
            ),
            exit_distance_m=0.02,
            max_steps=30,
            min_steps=15,
        ),
    )


class MotionPlanningOracle:
    """6-phase MP oracle. Outputs 6-D normalized action (5 arm + 1 gripper)."""

    def __init__(self, env, dls_lambda: float = 0.1, null_bias_gain: float = 0.15) -> None:
        """Oracle with custom DLS + null-space bias IK.

        Parameters
        ----------
        env : CubeLiftEnv (lazy ref to avoid cycle)
        dls_lambda : float
            Damped least-squares regularization. 0.01 was too low (IK
            blew up to >11 rad near singularities); 0.1 stable; 0.5
            very conservative but slow.
        null_bias_gain : float
            Secondary-task gain pulling joints toward ``joint_init`` in
            the null space of the position task. The 5-DOF arm has 2
            redundancy directions w.r.t. the 3-D position target; without
            this term DLS picks the min-norm solution and the redundant
            DOFs drift toward soft limits over the 40+ descend steps,
            wedging the arm and tanking grasp success. 0.5 = strong
            enough to keep approach end pose near home; tune lower if
            the bias fights position tracking too hard.
        """
        import torch

        self._env = env
        self._torch = torch
        self._phases = _build_phases()
        self._dls_lambda = float(dls_lambda)
        self._null_bias_gain = float(null_bias_gain)
        self._device = env._sim.device
        # Home pose tensor for the null-space bias target.
        self._q_home = torch.tensor(
            list(env.cfg.robot.joint_init), device=self._device, dtype=torch.float32
        )
        # Cache mimic joint indices so we can log them at phase transitions
        # to check that PincOpen's 4-bar actually follows the gripper joint.
        mimic_names = ["left_proximal", "left_distal", "right_proximal", "right_distal"]
        self._mimic_idxs, _ = env._robot.find_joints(mimic_names, preserve_order=True)
        self._phase = 0
        self._steps_in_phase = 0

    # ---- OraclePolicy contract ------------------------------------------

    def reset(self, obs: np.ndarray, info: dict) -> None:
        self._phase = 0
        self._steps_in_phase = 0

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        torch = self._torch
        env = self._env
        cfg: CubeLiftEnvCfg = env.cfg

        # 1) Read GT (oracle bypasses NoiseModel — it needs ground truth).
        cube_xyz = env._sim_cube_xyz_gt()
        dummy_pos = env._sim_ee_pose_gt()[:3]  # IK EE = gripper_dummy_link
        # Current fingertip center via Pinocchio FK at the live arm config.
        # Used both for the phase-exit check (fingertip distance to target)
        # and to translate fingertip-space phase targets into dummy-space
        # IK targets dynamically (the wrist orientation that determines
        # the offset shifts as the arm moves).
        arm_q_now = env._sim_joint_pos_gt()
        finger_pos = env._fk_fingertip_center(arm_q_now)
        offset_dummy_to_finger = finger_pos - dummy_pos
        goal = np.asarray(cfg.goal.pos_xyz_m, dtype=np.float32)

        # 2) Resolve current phase target (fingertip frame), gripper cmd,
        #    advance conditions. ``target_xyz`` is what we drive the IK
        #    EE (dummy) toward; ``finger_target`` is what we measure
        #    phase-exit against.
        spec = self._phases[self._phase]
        finger_target = spec.target_fn(cube_xyz, goal)
        target_xyz = finger_target - offset_dummy_to_finger
        gripper_norm = 1.0 if spec.gripper_open else -1.0

        ee_to_target = float(np.linalg.norm(finger_pos - finger_target))
        reached = ee_to_target <= spec.exit_distance_m
        held = self._steps_in_phase >= spec.min_steps
        timed_out = self._steps_in_phase >= spec.max_steps
        if self._phase < len(self._phases) - 1 and held and (reached or timed_out):
            gripper_pos = float(
                env._robot.data.joint_pos[0, env._gripper_joint_idx]
                .detach()
                .cpu()
                .item()
            )
            mimic_pos = (
                env._robot.data.joint_pos[0, self._mimic_idxs]
                .detach()
                .cpu()
                .numpy()
                .round(3)
                .tolist()
            )
            print(
                f"[oracle] phase {self._phase} '{spec.name}' end "
                f"steps={self._steps_in_phase} finger_to_target={ee_to_target:.4f}m "
                f"(exit<= {spec.exit_distance_m:.3f}) reached={reached} "
                f"timed_out={timed_out} finger={finger_pos.round(3).tolist()} "
                f"dummy={dummy_pos.round(3).tolist()} "
                f"target_finger={finger_target.round(3).tolist()} "
                f"cube={cube_xyz.round(3).tolist()} "
                f"gripper={gripper_pos:.3f} mimic(L_prox,L_dist,R_prox,R_dist)={mimic_pos}",
                flush=True,
            )
            self._phase += 1
            self._steps_in_phase = 0
            spec = self._phases[self._phase]
            finger_target = spec.target_fn(cube_xyz, goal)
            target_xyz = finger_target - offset_dummy_to_finger
            gripper_norm = 1.0 if spec.gripper_open else -1.0
        else:
            self._steps_in_phase += 1

        # Alias for the IK code below (which still works in dummy space).
        ee_pos = dummy_pos

        # 3) IK: custom DLS + null-space bias toward joint_init.
        #    5-DOF arm × 3-D position task = redundancy 2. Stock DLS picks
        #    the min-norm solution and lets redundant DOFs drift toward
        #    joint limits over many steps. Adding a null-space term
        #    k * (q_home - q) projected onto null(J) keeps the arm near
        #    its home configuration without disturbing the EE task.
        device = self._device
        # PhysX raw jacobian shape (num_envs, num_bodies, 6, num_dofs). For
        # a floating-base articulation the leading 6 dof columns are the
        # root 6-DoF; arm joint columns start at index 6.
        jacobians = env._robot.root_physx_view.get_jacobians()
        arm_idx = list(env._arm_joint_idxs)
        n_root_dofs = jacobians.shape[3] - env._robot.data.joint_pos.shape[1]
        joint_cols = [n_root_dofs + i for i in arm_idx]
        # Linear (position) jacobian only — rows 0:3.
        jac_lin = jacobians[0, env._ee_body_idx, :3, :][:, joint_cols]  # (3, n_arm)
        q_arm = env._robot.data.joint_pos[0, arm_idx]  # (n_arm,)
        ee_pos_t = torch.tensor(ee_pos, device=device, dtype=torch.float32)
        target_t = torch.tensor(target_xyz, device=device, dtype=torch.float32)
        x_err = target_t - ee_pos_t  # (3,)

        # Damped least-squares primary task: q_dot = J^T (J J^T + λ²I)⁻¹ x_err
        lam2 = self._dls_lambda ** 2
        JJT = jac_lin @ jac_lin.T  # (3, 3)
        JJT_damped = JJT + lam2 * torch.eye(3, device=device, dtype=torch.float32)
        # solve(A, b) avoids explicit inverse
        primary = jac_lin.T @ torch.linalg.solve(JJT_damped, x_err)  # (n_arm,)

        # Null-space secondary task: pull toward q_home.
        # N = I - J^T (J J^T + λ²I)⁻¹ J   (n_arm × n_arm projector)
        n_arm = jac_lin.shape[1]
        I_q = torch.eye(n_arm, device=device, dtype=torch.float32)
        N = I_q - jac_lin.T @ torch.linalg.solve(JJT_damped, jac_lin)
        secondary = self._null_bias_gain * (self._q_home - q_arm)
        q_dot = primary + N @ secondary

        # The Isaac Lab DLS controller returned absolute joint targets, so
        # the rest of the oracle expects shape (1, n_arm). Convert q_dot
        # (joint-space increment over one IK call) into an absolute target.
        target_joint_arm = (q_arm + q_dot).unsqueeze(0)  # (1, n_arm)
        # Keep the (1, n_arm) view of current joints for the diagnostic
        # printing below — it indexes [0] expecting the old DLS shape.
        joint_pos_arm = q_arm.unsqueeze(0)

        # Clip IK output to per-joint soft limits — even with null-space
        # bias, large EE errors at the start of a phase can still push
        # primary past a limit. Without clip the env clamps silently and
        # the arm pins itself at the boundary.
        limits = env._robot.data.soft_joint_pos_limits[:, arm_idx, :]
        lo = limits[..., 0]
        hi = limits[..., 1]
        target_joint_arm = torch.clamp(target_joint_arm, lo, hi)

        # 4) Arm joint delta → normalized [-1, 1] (env un-normalizes on step).
        cur_arm_np = joint_pos_arm[0].detach().cpu().numpy()
        target_arm_np = target_joint_arm[0].detach().cpu().numpy()
        delta = target_arm_np - cur_arm_np

        # Per-step diagnostic, throttled. The first step in each phase
        # gives the full shape sanity print; subsequent steps emit one
        # compact line every 5 steps so we can watch IK target evolve
        # vs cur_arm without flooding the log at 20 Hz.
        if self._steps_in_phase == 0:
            print(
                f"[oracle.ik] phase={spec.name} ENTRY "
                f"ee={ee_pos.round(3).tolist()} "
                f"target={target_xyz.round(3).tolist()} "
                f"d={ee_to_target:.4f}m "
                f"cur_arm={cur_arm_np.round(3).tolist()} "
                f"ik_target_arm={target_arm_np.round(3).tolist()} "
                f"|delta|max={float(np.abs(delta).max()):.4f}",
                flush=True,
            )
        elif self._steps_in_phase % 5 == 0:
            print(
                f"[oracle.ik] phase={spec.name} step={self._steps_in_phase} "
                f"ee={ee_pos.round(3).tolist()} d={ee_to_target:.4f} "
                f"cur_arm={cur_arm_np.round(3).tolist()} "
                f"ik_tgt={target_arm_np.round(3).tolist()} "
                f"|delta|max={float(np.abs(delta).max()):.4f}",
                flush=True,
            )

        arm_norm = np.clip(delta / cfg.robot.action_delta_max, -1.0, 1.0).astype(np.float32)

        # Stash for the viewer overlay (frame painter reads these via
        # env._oracle_dbg_*). Lets the live viewer caption each frame
        # with phase + ee-to-target without re-querying the IK.
        env._oracle_dbg_phase = spec.name
        env._oracle_dbg_step = self._steps_in_phase
        env._oracle_dbg_ee_to_target = ee_to_target

        return np.concatenate([arm_norm, [gripper_norm]]).astype(np.float32)

    # ---- Helpers ---------------------------------------------------------

    @property
    def phase(self) -> int:
        return self._phase

    @property
    def phase_name(self) -> str:
        return self._phases[self._phase].name

    @property
    def done(self) -> bool:
        last = len(self._phases) - 1
        return (
            self._phase >= last
            and self._steps_in_phase >= self._phases[last].min_steps
        )
