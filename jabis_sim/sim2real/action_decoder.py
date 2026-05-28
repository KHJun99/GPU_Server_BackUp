"""SO-ARM101 raw policy action → servo command decoder (PyTorch only)."""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class SoArm101ActionDecoder:
    """Replicates the IsaacLab JointPositionAction + BinaryJointPositionAction mapping
    used at training time, so a raw policy output (6,) can be turned into joint
    target positions for the real SO-ARM101.
    """

    DEFAULT_ARM_JOINT_ORDER: tuple[str, ...] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )
    DEFAULT_GRIPPER_JOINT: str = "left_proximal"
    DEFAULT_JOINT_POS: dict[str, float] = {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 1.57,
        "wrist_roll": 0.0,
        "left_proximal": 0.0,
    }
    JOINT_LIMITS: dict[str, tuple[float, float]] = {
        "shoulder_pan":   (-3.14, 3.14),
        "shoulder_lift":  (-3.14, 3.14),
        "elbow_flex":     (-1.57, 3.14),
        "wrist_flex":     (-3.14, 3.14),
        "wrist_roll":     (-3.14, 3.14),
        "left_proximal":  (-0.70, 0.00),
    }

    def __init__(
        self,
        default_joint_pos: dict[str, float] | None = None,
        arm_scale: float = 0.5,
        gripper_open: float = 0.0,
        gripper_close: float = -0.6,
        arm_joint_order: tuple[str, ...] | None = None,
        gripper_joint: str | None = None,
        joint_limits: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.arm_joint_order: tuple[str, ...] = (
            tuple(arm_joint_order) if arm_joint_order is not None
            else self.DEFAULT_ARM_JOINT_ORDER
        )
        self.gripper_joint: str = gripper_joint or self.DEFAULT_GRIPPER_JOINT
        self.default_joint_pos: dict[str, float] = (
            dict(default_joint_pos) if default_joint_pos is not None
            else dict(self.DEFAULT_JOINT_POS)
        )
        self.joint_limits: dict[str, tuple[float, float]] = (
            {k: (float(lo), float(hi)) for k, (lo, hi) in joint_limits.items()}
            if joint_limits is not None
            else {k: tuple(v) for k, v in self.JOINT_LIMITS.items()}
        )
        all_joints = (*self.arm_joint_order, self.gripper_joint)
        missing = [j for j in all_joints if j not in self.default_joint_pos]
        if missing:
            raise KeyError(f"default_joint_pos missing keys: {missing}")
        missing_lim = [j for j in all_joints if j not in self.joint_limits]
        if missing_lim:
            raise KeyError(f"joint_limits missing keys: {missing_lim}")
        for j, (lo, hi) in self.joint_limits.items():
            if lo > hi:
                raise ValueError(f"joint_limits[{j}] has lo > hi: ({lo}, {hi})")
        self.arm_scale: float = float(arm_scale)
        self.gripper_open: float = float(gripper_open)
        self.gripper_close: float = float(gripper_close)
        if len(self.arm_joint_order) != 5:
            raise ValueError(
                f"arm_joint_order must have 5 entries, got {len(self.arm_joint_order)}"
            )
        logger.info(
            "SoArm101ActionDecoder: scale=%.3f open=%.3f close=%.3f order=%s",
            self.arm_scale, self.gripper_open, self.gripper_close, self.arm_joint_order,
        )
        logger.info("joint_limits: %s", self.joint_limits)

    def decode(self, action: torch.Tensor) -> dict[str, float]:
        if not isinstance(action, torch.Tensor):
            raise TypeError(f"action must be torch.Tensor, got {type(action).__name__}")
        if action.dim() != 1 or action.shape[0] != 6:
            raise ValueError(
                f"decode() expects a single action of shape (6,), got "
                f"{tuple(action.shape)}. For batched policy output (B, 6), "
                "index a single sample first: decoder.decode(action[i])."
            )

        a = action.detach().to(dtype=torch.float32, device="cpu")
        raw_cmd: dict[str, float] = {}
        for i, joint in enumerate(self.arm_joint_order):
            offset = self.default_joint_pos[joint]
            raw_cmd[joint] = float(self.arm_scale * a[i].item() + offset)
        raw_cmd[self.gripper_joint] = (
            self.gripper_open if a[5].item() >= 0.0 else self.gripper_close
        )
        return self._clamp(raw_cmd)

    def _clamp(self, cmd: dict[str, float]) -> dict[str, float]:
        clamped: dict[str, float] = {}
        for joint, value in cmd.items():
            lo, hi = self.joint_limits[joint]
            new_value = lo if value < lo else hi if value > hi else value
            if new_value != value:
                logger.warning(
                    "clamped %s: %+.4f -> %+.4f (limit [%+.2f, %+.2f])",
                    joint, value, new_value, lo, hi,
                )
            clamped[joint] = new_value
        return clamped
