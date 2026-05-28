"""SO-ARM101 cube-lift oracle policy.

5-state vectorized state machine driven by IsaacLab's DifferentialIKController:
    APPROACH → DESCEND → CLOSE → LIFT → MOVE_TO_GOAL

The class is split so that:
  * Construction + state-buffer management + state transitions are pure PyTorch
    (testable on CPU without IsaacLab — see smoke_test.py).
  * `compute()` and `setup()` lazily import IsaacLab pieces, so the module can
    be imported in a plain Python environment.

Action space matches the env in ``isaac_so_arm101.tasks.lift.joint_pos_env_cfg`` —
specifically ``JointPositionActionCfg(scale=<scale>, use_default_offset=True)``
for the 5 arm joints, plus a ``BinaryJointPositionAction`` for the gripper. Demos
produced here are therefore drop-in compatible with the rsl_rl actor.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch

# State IDs
APPROACH = 0
DESCEND = 1
CLOSE = 2
LIFT = 3
MOVE_TO_GOAL = 4

DEFAULT_ARM_JOINT_RE = ("shoulder_.*", "elbow_flex", "wrist_.*")


class OraclePolicy:
    """Vectorized oracle for cube-lift demo collection.

    Args:
        num_envs: number of parallel environments.
        device: torch device.
        scale: action scale matching ``JointPositionActionCfg.scale``. Used to
            convert joint targets back to PPO's raw action space:
            ``raw = (joint_target - default_arm) / scale``. v6 task uses 1.5.
        reach_above_dz / descend_dz / lift_dz: vertical offsets in metres.
        reach_dist / descend_z_dist: 0→1 / 1→2 transition thresholds.
        close_steps: number of timesteps to hold the gripper closed before LIFT.
        max_ee_step: cap on per-step ee target delta to avoid IK joint jumps that
            would saturate ``raw`` outside ``[-1, +1]`` after scaling.
        success_z: cube z (env-local) that triggers LIFT → MOVE_TO_GOAL transition.
    """

    NUM_ARM_JOINTS = 5
    NUM_ACTION_DIMS = 6  # 5 arm + 1 binary gripper

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str = "cpu",
        scale: float = 1.5,
        reach_above_dz: float = 0.10,
        descend_dz: float = 0.005,
        lift_dz: float = 0.10,
        reach_dist: float = 0.02,
        descend_z_dist: float = 0.005,
        close_steps: int = 8,
        max_ee_step: float = 0.02,
        success_z: float = 0.10,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.scale = float(scale)
        self.params = {
            "reach_above_dz": reach_above_dz,
            "descend_dz": descend_dz,
            "lift_dz": lift_dz,
            "reach_dist": reach_dist,
            "descend_z_dist": descend_z_dist,
            "close_steps": int(close_steps),
            "max_ee_step": max_ee_step,
            "success_z": success_z,
        }

        # State buffers (per env)
        self.states = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.close_step = torch.zeros_like(self.states)
        self.frozen_close_joint = torch.zeros(self.num_envs, self.NUM_ARM_JOINTS, device=self.device)
        self.lift_target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)

        # Lazy-initialised IsaacLab handles
        self._ik = None
        self.robot = None
        self.object_asset = None
        self.target_pos_b_provider: Optional[Callable[[], torch.Tensor]] = None
        self.arm_ids: list[int] = []
        self.arm_ids_t: Optional[torch.Tensor] = None
        self.ee_idx: int = -1
        self.ee_jacobi_idx: int = -1
        self.default_arm: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Setup / lifecycle
    # ------------------------------------------------------------------
    def setup(
        self,
        robot,
        object_asset,
        target_pos_b_provider: Optional[Callable[[], torch.Tensor]] = None,
        ee_body_name: str = "gripper_link",
        arm_joint_re: Iterable[str] = DEFAULT_ARM_JOINT_RE,
    ) -> None:
        """Resolve robot/body/joint indices. Must be called once after the env
        has been built."""
        self.robot = robot
        self.object_asset = object_asset
        self.target_pos_b_provider = target_pos_b_provider

        self.arm_ids = robot.find_joints(list(arm_joint_re))[0]
        self.arm_ids_t = torch.tensor(self.arm_ids, device=self.device, dtype=torch.long)
        self.ee_idx = robot.find_bodies(ee_body_name)[0][0]
        self.ee_jacobi_idx = self.ee_idx - 1
        self.default_arm = robot.data.default_joint_pos[:, self.arm_ids_t].clone()

        if self.default_arm.shape != (self.num_envs, self.NUM_ARM_JOINTS):
            raise RuntimeError(
                f"default_arm shape {tuple(self.default_arm.shape)} does not match "
                f"({self.num_envs}, {self.NUM_ARM_JOINTS}). Check robot config."
            )

    def _ensure_ik(self) -> None:
        if self._ik is not None:
            return
        # Lazy import — IsaacLab not required for CPU smoke testing.
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        ik_cfg = DifferentialIKControllerCfg(
            command_type="position", use_relative_mode=False, ik_method="dls"
        )
        self._ik = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

    def reset(self, env_indices: torch.Tensor | list[int] | None) -> None:
        """Reset state buffers for the given env indices."""
        if env_indices is None:
            return
        if isinstance(env_indices, list):
            if len(env_indices) == 0:
                return
            env_indices = torch.tensor(env_indices, device=self.device, dtype=torch.long)
        if env_indices.numel() == 0:
            return
        self.states[env_indices] = APPROACH
        self.close_step[env_indices] = 0
        self.frozen_close_joint[env_indices] = 0.0
        self.lift_target_pos_b[env_indices] = 0.0

    # ------------------------------------------------------------------
    # Pure-Python helpers (CPU testable)
    # ------------------------------------------------------------------
    def compute_target_pos_b(
        self,
        ee_pos_b: torch.Tensor,
        cube_pos_b: torch.Tensor,
        target_pos_b_goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-state ee target in base frame (CLOSE returns current ee — frozen
        joint hold is handled separately in compute())."""
        target = ee_pos_b.clone()
        m_appr = self.states == APPROACH
        m_desc = self.states == DESCEND
        m_lift = self.states == LIFT
        m_goal = self.states == MOVE_TO_GOAL

        approach_offset = torch.tensor(
            [0.0, 0.0, self.params["reach_above_dz"]], device=self.device
        )
        descend_offset = torch.tensor(
            [0.0, 0.0, self.params["descend_dz"]], device=self.device
        )
        target[m_appr] = cube_pos_b[m_appr] + approach_offset
        target[m_desc] = cube_pos_b[m_desc] + descend_offset
        target[m_lift] = self.lift_target_pos_b[m_lift]

        if m_goal.any():
            if target_pos_b_goal is None:
                # No goal supplied — fall back to held lift target.
                target[m_goal] = self.lift_target_pos_b[m_goal]
            else:
                target[m_goal] = target_pos_b_goal[m_goal]

        return target

    def step_state_machine(
        self,
        ee_pos_b: torch.Tensor,
        cube_pos_b: torch.Tensor,
        joint_pos_arm: torch.Tensor,
        target_pos_b: torch.Tensor,
    ) -> None:
        """Apply transitions and update frozen joint / lift target buffers.

        Pure PyTorch — no IsaacLab dependency, callable from smoke tests.
        """
        m_appr = self.states == APPROACH
        m_desc = self.states == DESCEND
        m_close = self.states == CLOSE
        m_lift = self.states == LIFT

        dist3 = torch.norm(target_pos_b - ee_pos_b, dim=-1)
        dz_abs = (target_pos_b[:, 2] - ee_pos_b[:, 2]).abs()
        cube_z = cube_pos_b[:, 2]

        t_0_1 = m_appr & (dist3 < self.params["reach_dist"])
        t_1_2 = m_desc & (dz_abs < self.params["descend_z_dist"])
        t_2_3 = m_close & (self.close_step >= self.params["close_steps"])
        t_3_4 = m_lift & (cube_z >= self.params["success_z"])

        # Lock close-time joint pose at DESCEND → CLOSE entry.
        if t_1_2.any():
            self.frozen_close_joint = torch.where(
                t_1_2.unsqueeze(-1),
                joint_pos_arm.detach().clone(),
                self.frozen_close_joint,
            )
        # Lock lift target at CLOSE → LIFT entry.
        if t_2_3.any():
            new_lift = ee_pos_b.detach().clone()
            new_lift[:, 2] = new_lift[:, 2] + self.params["lift_dz"]
            self.lift_target_pos_b = torch.where(
                t_2_3.unsqueeze(-1), new_lift, self.lift_target_pos_b
            )

        # close_step counter
        self.close_step = torch.where(
            m_close, self.close_step + 1, torch.zeros_like(self.close_step)
        )

        new_states = self.states.clone()
        new_states = torch.where(t_0_1, torch.full_like(new_states, DESCEND), new_states)
        new_states = torch.where(t_1_2, torch.full_like(new_states, CLOSE), new_states)
        new_states = torch.where(t_2_3, torch.full_like(new_states, LIFT), new_states)
        new_states = torch.where(t_3_4, torch.full_like(new_states, MOVE_TO_GOAL), new_states)
        self.states = new_states

    def to_action(
        self,
        joint_target_arm: torch.Tensor,
        clamp: bool = True,
    ) -> torch.Tensor:
        """Convert joint position target to PPO raw action (5 arm + 1 gripper)."""
        if self.default_arm is None:
            raise RuntimeError("setup() must be called before to_action().")
        arm_raw = (joint_target_arm - self.default_arm) / self.scale
        if clamp:
            arm_raw = arm_raw.clamp(-1.0, 1.0)
        gripper_open_mask = self.states < CLOSE  # APPROACH, DESCEND
        gripper_raw = torch.where(
            gripper_open_mask,
            torch.tensor(1.0, device=self.device),
            torch.tensor(-1.0, device=self.device),
        ).unsqueeze(-1)
        return torch.cat([arm_raw, gripper_raw], dim=-1)

    # ------------------------------------------------------------------
    # IsaacLab-dependent compute step
    # ------------------------------------------------------------------
    def compute(self) -> tuple[torch.Tensor, dict]:
        """Run one step of the oracle. Reads robot + object data from IsaacLab,
        runs IK + state machine, and returns ``(action, info)``.

        Action shape: ``(num_envs, 6)`` already in PPO raw space and clamped.
        Info contains diagnostic tensors (cube_pos_b, ee_pos_b, target_pos_b,
        states, close_step) for the current step.
        """
        if self.robot is None or self.object_asset is None:
            raise RuntimeError("setup() must be called before compute().")
        self._ensure_ik()

        from isaaclab.utils.math import (
            matrix_from_quat,
            quat_inv,
            subtract_frame_transforms,
        )

        with torch.no_grad():
            # 1) Frames
            ee_pose_w = self.robot.data.body_pose_w[:, self.ee_idx]
            root_pose_w = self.robot.data.root_pose_w
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
            )
            cube_pos_w = self.object_asset.data.root_pos_w
            # Identity base rotation per SO_ARM101_CFG (rot=(1,0,0,0)).
            cube_pos_b = cube_pos_w - root_pose_w[:, 0:3]

            target_pos_b_goal = None
            if self.target_pos_b_provider is not None:
                target_pos_b_goal = self.target_pos_b_provider()

            # 2) Per-state target (in base frame)
            target_pos_b = self.compute_target_pos_b(ee_pos_b, cube_pos_b, target_pos_b_goal)

            # 3) Per-step ee delta cap so the resulting joint target stays inside
            #    PPO action range (default ± scale rad).
            delta = target_pos_b - ee_pos_b
            norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            cap_scale = (self.params["max_ee_step"] / norm).clamp(max=1.0)
            target_capped = ee_pos_b + delta * cap_scale

            # 4) Jacobian (transformed to base frame)
            jac = self.robot.root_physx_view.get_jacobians()[
                :, self.ee_jacobi_idx, :, self.arm_ids_t
            ].clone()
            base_rot_inv = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
            jac[:, :3, :] = torch.bmm(base_rot_inv, jac[:, :3, :])
            jac[:, 3:, :] = torch.bmm(base_rot_inv, jac[:, 3:, :])
            joint_pos_arm = self.robot.data.joint_pos[:, self.arm_ids_t]

            # 5) IK
            self._ik.set_command(command=target_capped, ee_quat=ee_quat_b)
            joint_target = self._ik.compute(ee_pos_b, ee_quat_b, jac, joint_pos_arm)

            # 6) Override CLOSE with frozen joint target (no IK during contact).
            close_mask = self.states == CLOSE
            if close_mask.any():
                joint_target = torch.where(
                    close_mask.unsqueeze(-1), self.frozen_close_joint, joint_target
                )

            # 7) Convert to PPO action
            action = self.to_action(joint_target, clamp=True)

            # 8) Apply state transitions for the *next* step.
            self.step_state_machine(ee_pos_b, cube_pos_b, joint_pos_arm, target_pos_b)

        info = {
            "ee_pos_b": ee_pos_b,
            "cube_pos_b": cube_pos_b,
            "target_pos_b": target_pos_b,
            "joint_target": joint_target,
            "states": self.states.clone(),
            "close_step": self.close_step.clone(),
        }
        return action, info
