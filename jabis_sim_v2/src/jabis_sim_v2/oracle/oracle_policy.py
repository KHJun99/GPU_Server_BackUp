"""Oracle 5-state policy for cube lift (v2, rewritten from v1 lessons).

V1 patterns applied:
1. EE pose in base frame (not world)
2. Jacobian transformed by base rotation inverse
3. EE step capped per timestep (max_ee_step=0.02m)
4. CLOSE state freezes joints (no IK during contact)
5. Gripper raw action +1=open / -1=close (matches BinaryJointPositionAction)

State machine (multi-waypoint + shoulder_pan rotation, v4):
  APPROACH(0) → DESCEND(1) → CLOSE(2) → LIFT(3)
               → ROTATE_PAN(4) : shoulder_pan을 bin 방향으로 회전 (cube grip 유지)
               → TRANSPORT(5)  : ee → (bin_xy, lift_z) — bin 위까지 미세 조정
               → LOWER(6)      : ee z → release_z — bin 입구 위로 하강
               → RELEASE(7)    : gripper open + 자세 유지
               → DONE(8)       : release_steps 끝나면 정지
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .ik import NullspaceIK

# State IDs
APPROACH = 0
DESCEND = 1
CLOSE = 2
LIFT = 3
ROTATE_PAN = 4
TRANSPORT = 5
LOWER = 6
RELEASE = 7
DONE = 8

# Backward-compat: old code uses MOVE_TO_GOAL — now points to TRANSPORT (=5)
MOVE_TO_GOAL = TRANSPORT

DEFAULT_ARM_JOINT_RE = ("shoulder_.*", "elbow_flex", "wrist_.*")


# Backward-compat aliases
class OracleState:
    APPROACH = APPROACH
    DESCEND = DESCEND
    CLOSE = CLOSE
    LIFT = LIFT
    ROTATE_PAN = ROTATE_PAN
    TRANSPORT = TRANSPORT
    LOWER = LOWER
    RELEASE = RELEASE
    DONE = DONE
    MOVE_TO_GOAL = TRANSPORT


@dataclass
class OracleCfg:
    arm_joint_names: tuple = DEFAULT_ARM_JOINT_RE
    ee_body_name: str = "gripper_link"

    # action scale (must match env's JointPositionActionCfg scale)
    arm_action_scale: float = 1.5
    arm_action_scales: tuple = (1.5, 1.5, 1.5, 1.5, 1.5)

    # waypoint offsets
    reach_above_dz: float = 0.10
    descend_dz: float = 0.07  # 원래 default (50% grip 성공)
    lift_dz: float = 0.15
    finger_x_offset: float = 0.006

    # transition thresholds
    reach_dist: float = 0.02
    descend_z_dist: float = 0.02  # 2cm tolerance
    close_steps: int = 60

    # multi-waypoint bin drop (v3)
    release_z: float = 0.15  # bin 입구 z=0 위 15cm (drop 시 ee z) — 책상에서 robot reach 한계로 인해 높이 유지
    bin_xy_tol: float = 0.10  # TRANSPORT→LOWER: ee xy 와 bin xy 거리 임계 (10cm — robot reach 한계 보상)
    bin_z_tol: float = 0.03  # LOWER→RELEASE: ee z 와 release_z 거리 임계
    release_steps: int = 30  # RELEASE 유지 step (gripper open + cube 자유낙하)
    transport_max_steps: int = 200  # TRANSPORT timeout — bin xy 못 도달 시 강제 LOWER
    lower_max_steps: int = 80  # LOWER timeout — bin z 못 도달 시 강제 RELEASE

    # ROTATE_PAN (v4) — cube 잡고 shoulder_pan 만 점진적 회전 (IK 우회)
    # Actuator 강화 (effort 20, velocity 10) 후 점진적 명령으로 cube grip 유지.
    rotate_pan_speed: float = 0.01  # rad/step (~0.57°/step) — cube grip 유지용 slow rotate
    rotate_pan_tol: float = 0.05
    rotate_max_steps: int = 500  # 90° = 157 step, 180° = 314 step

    # LIFT minimum dwell — cube 가 책상에서 충분히 들리도록
    lift_min_steps: int = 30  # LIFT state 최소 step
    lift_ee_z_tol: float = 0.05  # ee_z 가 lift_target_z 의 5cm 이내면 LIFT 완료

    # safety
    max_ee_step: float = 0.5  # 원래 default — 실물 매칭

    # success
    success_z: float = 0.07

    # NullspaceIK tuning (Phase 2 ablation)
    ik_damping: float = 0.05
    ik_max_dq: float = 0.30
    ik_nullspace_gain: float = 0.2


class OraclePolicy:
    """Per-env state machine for cube pick & lift."""

    NUM_ARM_JOINTS = 5
    NUM_ACTION_DIMS = 6  # 5 arm + 1 gripper

    def __init__(self, env, cfg: OracleCfg | None = None):
        self.env = env
        self.cfg = cfg or OracleCfg()
        self.num_envs = env.num_envs
        self.device = env.device

        self.robot = env.scene["robot"]
        self.object_asset = env.scene["cube"]

        # joint / body indices
        self.arm_ids = self.robot.find_joints(list(self.cfg.arm_joint_names))[0]
        self.arm_ids_t = torch.tensor(self.arm_ids, device=self.device, dtype=torch.long)
        self.ee_idx = self.robot.find_bodies(self.cfg.ee_body_name)[0][0]
        self.ee_jacobi_idx = self.ee_idx - 1

        self.default_arm = self.robot.data.default_joint_pos[:, self.arm_ids_t].clone()

        # state buffers
        self.states = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.close_step = torch.zeros_like(self.states)
        self.frozen_close_joint = torch.zeros(
            self.num_envs, self.NUM_ARM_JOINTS, device=self.device
        )
        self.lift_target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        # per-env goal xy in base frame (optional, for transport measurement)
        self._goal_xy_b = None  # (num_envs, 2) or None
        self.cube_z_max = torch.zeros(self.num_envs, device=self.device)
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # gradual close (0→1 over N steps in CLOSE state)
        self.close_progress = torch.zeros(self.num_envs, device=self.device)
        # release timer (RELEASE state 동안 gripper open + 자세 유지)
        self.release_step = torch.zeros_like(self.states)
        # transport timer (TRANSPORT 못 끝나면 강제 LOWER)
        self.transport_step = torch.zeros_like(self.states)
        # lower timer (LOWER 못 끝나면 강제 RELEASE)
        self.lower_step = torch.zeros_like(self.states)
        # LIFT step counter
        self.lift_step = torch.zeros_like(self.states)
        # ROTATE_PAN buffers
        self.rotate_step = torch.zeros_like(self.states)
        # shoulder_pan target per env (set by set_goal_xy)
        self._rotate_pan_target = torch.zeros(self.num_envs, device=self.device)
        # frozen arm pose at LIFT→ROTATE_PAN entry (used during rotate)
        self.frozen_rotate_arm = torch.zeros(
            self.num_envs, self.NUM_ARM_JOINTS, device=self.device
        )

        # backward-compat alias
        self.state = self.states  # same tensor reference

        # NullspaceIK: 5-DOF position task + 2-DOF posture nullspace
        # rest_pose = default arm pose (keeps wrist_flex at default)
        # rest_weights: shoulder/elbow free, wrist heavily pulled to default (grip 정렬 유지)
        rest_weights = torch.tensor([0.02, 0.02, 0.02, 1.0, 1.0], device=self.device)

        self._ik = NullspaceIK(
            num_envs=self.num_envs,
            num_joints=self.NUM_ARM_JOINTS,
            device=self.device,
            damping=self.cfg.ik_damping,
            nullspace_gain=self.cfg.ik_nullspace_gain,
            rest_pose=self.default_arm.clone(),
            rest_weights=rest_weights,
            max_dq=self.cfg.ik_max_dq,
        )

    def set_goal_xy(self, goal_xy_b: torch.Tensor):
        """Set per-env goal xy in base frame.

        Sequence: LIFT → ROTATE_PAN (shoulder_pan을 goal 향함) → TRANSPORT
                → LOWER → RELEASE → DONE.

        shoulder_pan target = atan2(-goal_y, goal_x)
          (URDF convention: shoulder_pan 0 → forward +x;
                            +1.5708 → forward -y;
                            -1.5708 → forward +y.
           즉 robot forward = (cos(s_pan), -sin(s_pan)).
           goal 방향 향하려면 s_pan = atan2(-goal_y, goal_x).)
        """
        assert goal_xy_b.shape == (self.num_envs, 2)
        self._goal_xy_b = goal_xy_b.to(self.device)
        self._rotate_pan_target = torch.atan2(
            -goal_xy_b[:, 1], goal_xy_b[:, 0]
        ).to(self.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, list):
            if len(env_ids) == 0:
                return
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        self.states[env_ids] = APPROACH
        self.close_step[env_ids] = 0
        self.frozen_close_joint[env_ids] = 0.0
        self.lift_target_pos_b[env_ids] = 0.0
        self.cube_z_max[env_ids] = 0.0
        self.success[env_ids] = False
        self.close_progress[env_ids] = 0.0
        self.release_step[env_ids] = 0
        self.transport_step[env_ids] = 0
        self.lower_step[env_ids] = 0
        self.rotate_step[env_ids] = 0
        self.lift_step[env_ids] = 0
        self.frozen_rotate_arm[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_cube_pos_w(self) -> torch.Tensor:
        return self.object_asset.data.root_pos_w

    def _get_ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        pose = self.robot.data.body_pose_w[:, self.ee_idx]
        return pose[:, 0:3], pose[:, 3:7]

    def _compute_target_pos_b(
        self, ee_pos_b: torch.Tensor, cube_pos_b: torch.Tensor
    ) -> torch.Tensor:
        """Per-state EE target in base frame (multi-waypoint v3)."""
        target = ee_pos_b.clone()
        m_appr = self.states == APPROACH
        m_desc = self.states == DESCEND
        m_lift = self.states == LIFT
        m_trans = self.states == TRANSPORT
        m_lower = self.states == LOWER
        m_release_or_done = (self.states == RELEASE) | (self.states == DONE)

        approach_offset = torch.tensor(
            [0.0, 0.0, self.cfg.reach_above_dz], device=self.device
        )
        descend_offset = torch.tensor(
            [0.0, 0.0, self.cfg.descend_dz], device=self.device
        )
        finger_x_offset = torch.tensor(
            [self.cfg.finger_x_offset, 0.0, 0.0], device=self.device
        )

        target[m_appr] = cube_pos_b[m_appr] + approach_offset + finger_x_offset
        target[m_desc] = cube_pos_b[m_desc] + descend_offset + finger_x_offset
        target[m_lift] = self.lift_target_pos_b[m_lift]

        if self._goal_xy_b is not None:
            # TRANSPORT: above bin at lift z
            trans_target = self.lift_target_pos_b.clone()
            trans_target[:, 0] = self._goal_xy_b[:, 0]
            trans_target[:, 1] = self._goal_xy_b[:, 1]
            target[m_trans] = trans_target[m_trans]

            # LOWER: above bin at release_z (낮은 z)
            lower_target = trans_target.clone()
            lower_target[:, 2] = self.cfg.release_z
            target[m_lower] = lower_target[m_lower]

            # RELEASE / DONE: stay at release waypoint
            target[m_release_or_done] = lower_target[m_release_or_done]
        else:
            # 외부 goal 없으면 모두 lift 자리 유지
            target[m_trans] = self.lift_target_pos_b[m_trans]
            target[m_lower] = self.lift_target_pos_b[m_lower]
            target[m_release_or_done] = self.lift_target_pos_b[m_release_or_done]

        return target

    def _step_state_machine(
        self,
        ee_pos_b: torch.Tensor,
        cube_pos_b: torch.Tensor,
        joint_pos_arm: torch.Tensor,
        target_pos_b: torch.Tensor,
    ):
        m_appr = self.states == APPROACH
        m_desc = self.states == DESCEND
        m_close = self.states == CLOSE
        m_lift = self.states == LIFT
        m_rotate = self.states == ROTATE_PAN
        m_trans = self.states == TRANSPORT
        m_lower = self.states == LOWER
        m_release = self.states == RELEASE

        dist3 = torch.norm(target_pos_b - ee_pos_b, dim=-1)
        dz_abs = (target_pos_b[:, 2] - ee_pos_b[:, 2]).abs()
        cube_z = cube_pos_b[:, 2]

        # TRANSPORT→LOWER: ee xy 가 bin xy 에 도달
        if self._goal_xy_b is not None:
            ee_xy_to_bin = torch.norm(ee_pos_b[:, :2] - self._goal_xy_b, dim=-1)
        else:
            ee_xy_to_bin = torch.full_like(dist3, 1.0)  # never triggers

        # ROTATE_PAN→TRANSPORT: shoulder_pan 이 target 각도 도달
        current_pan = joint_pos_arm[:, 0]  # shoulder_pan = 0번 joint (arm_joint_names 순서)
        pan_diff_abs = (self._rotate_pan_target - current_pan).abs()

        t_0_1 = m_appr & (dist3 < self.cfg.reach_dist)
        t_1_2 = m_desc & (dz_abs < self.cfg.descend_z_dist)
        t_2_3 = m_close & (self.close_step >= self.cfg.close_steps)
        # LIFT → ROTATE_PAN: cube grip + ee_z 가 lift target z 도달 + 최소 step
        ee_z_to_lift = (self.lift_target_pos_b[:, 2] - ee_pos_b[:, 2]).abs()
        t_3_4 = m_lift & (cube_z >= self.cfg.success_z) & (
            (ee_z_to_lift < self.cfg.lift_ee_z_tol)
            | (self.lift_step >= self.cfg.lift_min_steps * 3)
        ) & (self.lift_step >= self.cfg.lift_min_steps)
        # ROTATE_PAN → TRANSPORT: 회전 완료 OR timeout
        t_4_5 = m_rotate & (
            (pan_diff_abs < self.cfg.rotate_pan_tol)
            | (self.rotate_step >= self.cfg.rotate_max_steps)
        )
        # TRANSPORT → LOWER: xy 도달 OR timeout
        t_5_6 = m_trans & (
            (ee_xy_to_bin < self.cfg.bin_xy_tol)
            | (self.transport_step >= self.cfg.transport_max_steps)
        )
        t_6_7 = m_lower & (
            (dz_abs < self.cfg.bin_z_tol)
            | (self.lower_step >= self.cfg.lower_max_steps)
        )
        t_7_8 = m_release & (self.release_step >= self.cfg.release_steps)

        # Lock joints at DESCEND→CLOSE entry
        if t_1_2.any():
            self.frozen_close_joint = torch.where(
                t_1_2.unsqueeze(-1),
                joint_pos_arm.detach().clone(),
                self.frozen_close_joint,
            )
        # Lock lift target at CLOSE→LIFT entry
        if t_2_3.any():
            new_lift = ee_pos_b.detach().clone()
            new_lift[:, 2] = new_lift[:, 2] + self.cfg.lift_dz
            self.lift_target_pos_b = torch.where(
                t_2_3.unsqueeze(-1), new_lift, self.lift_target_pos_b
            )
        # Lock arm pose at LIFT→ROTATE_PAN entry
        if t_3_4.any():
            self.frozen_rotate_arm = torch.where(
                t_3_4.unsqueeze(-1),
                joint_pos_arm.detach().clone(),
                self.frozen_rotate_arm,
            )
        # On ROTATE_PAN→TRANSPORT entry: re-lock lift_target_pos_b at new ee pos
        # (회전 후 ee 가 base 기준 새 위치 → TRANSPORT/LOWER target 갱신 필요)
        if t_4_5.any():
            new_lift_after_rotate = ee_pos_b.detach().clone()
            self.lift_target_pos_b = torch.where(
                t_4_5.unsqueeze(-1),
                new_lift_after_rotate,
                self.lift_target_pos_b,
            )

        self.close_step = torch.where(
            m_close, self.close_step + 1, torch.zeros_like(self.close_step)
        )
        # gradual close: state >= CLOSE 면 close. RELEASE 부터 다시 open.
        self.close_progress = torch.where(
            m_close,
            (self.close_progress + 0.025).clamp(max=1.0),
            torch.where(
                (self.states > CLOSE) & (self.states < RELEASE),
                torch.ones_like(self.close_progress),
                torch.zeros_like(self.close_progress),
            ),
        )
        # timers
        self.release_step = torch.where(
            m_release, self.release_step + 1, torch.zeros_like(self.release_step)
        )
        self.lift_step = torch.where(
            m_lift, self.lift_step + 1, self.lift_step
        )
        self.rotate_step = torch.where(
            m_rotate, self.rotate_step + 1, self.rotate_step
        )
        self.transport_step = torch.where(
            m_trans, self.transport_step + 1, self.transport_step
        )
        self.lower_step = torch.where(
            m_lower, self.lower_step + 1, self.lower_step
        )

        new_states = self.states.clone()
        new_states = torch.where(t_0_1, torch.full_like(new_states, DESCEND), new_states)
        new_states = torch.where(t_1_2, torch.full_like(new_states, CLOSE), new_states)
        new_states = torch.where(t_2_3, torch.full_like(new_states, LIFT), new_states)
        new_states = torch.where(t_3_4, torch.full_like(new_states, ROTATE_PAN), new_states)
        new_states = torch.where(t_4_5, torch.full_like(new_states, TRANSPORT), new_states)
        new_states = torch.where(t_5_6, torch.full_like(new_states, LOWER), new_states)
        new_states = torch.where(t_6_7, torch.full_like(new_states, RELEASE), new_states)
        new_states = torch.where(t_7_8, torch.full_like(new_states, DONE), new_states)
        self.states = new_states
        self.state = self.states

        self.cube_z_max = torch.maximum(self.cube_z_max, cube_z)
        self.success = self.success | (cube_z >= self.cfg.success_z)

    def _to_action(self, joint_target_arm: torch.Tensor) -> torch.Tensor:
        """Convert joint target → env raw action (5 arm + 1 gripper).

        Gripper schedule:
          state < CLOSE        → open (+1)
          CLOSE ≤ state < RELEASE → close (-1, gradual via close_progress)
          state ≥ RELEASE      → open (+1, drop cube)
        """
        scales = torch.tensor(self.cfg.arm_action_scales, device=self.device)
        arm_raw = (joint_target_arm - self.default_arm) / scales  # broadcast (n_envs, 5)
        arm_raw = arm_raw.clamp(-1.0, 1.0)

        gripper_open_pre = self.states < CLOSE
        gripper_open_post = self.states >= RELEASE
        gripper_open_mask = gripper_open_pre | gripper_open_post

        close_value = -self.close_progress.clamp(max=1.0)
        gripper_raw = torch.where(
            gripper_open_mask,
            torch.tensor(1.0, device=self.device),
            close_value,
        ).unsqueeze(-1)
        return torch.cat([arm_raw, gripper_raw], dim=-1)

    # ------------------------------------------------------------------
    # Compute (per step)
    # ------------------------------------------------------------------
    def compute_action(self) -> torch.Tensor:
        from isaaclab.utils.math import (
            matrix_from_quat,
            quat_inv,
            subtract_frame_transforms,
        )

        with torch.no_grad():
            # 1) base-frame EE & cube
            ee_pose_w = self.robot.data.body_pose_w[:, self.ee_idx]
            root_pose_w = self.robot.data.root_pose_w
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
            )
            cube_pos_w = self.object_asset.data.root_pos_w
            cube_pos_b = cube_pos_w - root_pose_w[:, 0:3]

            # 2) per-state target
            target_pos_b = self._compute_target_pos_b(ee_pos_b, cube_pos_b)

            # 3) cap ee delta
            delta = target_pos_b - ee_pos_b
            norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            cap_scale = (self.cfg.max_ee_step / norm).clamp(max=1.0)
            target_capped = ee_pos_b + delta * cap_scale

            # 4) jacobian (base-frame)
            jac = self.robot.root_physx_view.get_jacobians()[
                :, self.ee_jacobi_idx, :, self.arm_ids_t
            ].clone()
            base_rot_inv = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
            jac[:, :3, :] = torch.bmm(base_rot_inv, jac[:, :3, :])
            jac[:, 3:, :] = torch.bmm(base_rot_inv, jac[:, 3:, :])
            joint_pos_arm = self.robot.data.joint_pos[:, self.arm_ids_t]

            # 5) IK (nullspace projection)
            joint_target = self._ik.compute(
                target_pos_b=target_capped,
                current_pos_b=ee_pos_b,
                jacobian=jac,
                current_joint_pos=joint_pos_arm,
            )

            # 6a) freeze CLOSE
            close_mask = self.states == CLOSE
            if close_mask.any():
                joint_target = torch.where(
                    close_mask.unsqueeze(-1), self.frozen_close_joint, joint_target
                )

            # 6b) ROTATE_PAN: IK 우회 — shoulder_pan 점진적 명령 (cube grip 유지용)
            rotate_mask = self.states == ROTATE_PAN
            if rotate_mask.any():
                current_pan = joint_pos_arm[:, 0]
                diff = self._rotate_pan_target - current_pan
                step = torch.clamp(
                    diff,
                    -self.cfg.rotate_pan_speed,
                    self.cfg.rotate_pan_speed,
                )
                new_pan = current_pan + step
                rotate_target = self.frozen_rotate_arm.clone()
                rotate_target[:, 0] = new_pan
                joint_target = torch.where(
                    rotate_mask.unsqueeze(-1), rotate_target, joint_target
                )

            # 7) action
            action = self._to_action(joint_target)

            # 8) advance state machine for next step
            self._step_state_machine(ee_pos_b, cube_pos_b, joint_pos_arm, target_pos_b)

        return action

    def is_done(self) -> torch.Tensor:
        return self.states == DONE
