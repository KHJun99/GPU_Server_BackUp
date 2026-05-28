"""CubeLiftEnv — Phase 1 EnvLike adapter wired to an Isaac Lab scene.

The env owns its own ``SimulationContext`` + ``InteractiveScene`` (single
``num_envs=1``); callers must boot ``AppLauncher`` before constructing it.
The decimation loop (``physics_dt × 6 → policy_dt``) reproduces the
``DirectRLEnv`` step pattern: ``set_joint_position_target`` →
``scene.write_data_to_sim`` → ``sim.step`` → ``scene.update(dt)``.

Reward / success uses GT pose from the same buffers; the NoiseModel is
applied only on the obs path so the success signal stays clean
(CLAUDE.md ## Phase 1 결정 사항 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from khj_rl.envs.cube_lift.cfg import CubeLiftEnvCfg
from khj_rl.envs.cube_lift.reward_dense import (
    compute_dense_reward,
    dense_alpha,
)
from khj_rl.envs.cube_lift.success import (
    MultiConditionSuccess,
    SuccessReport,
    SuccessState,
)


# EE link the policy's end-effector frame attaches to. `gripper_dummy_link`
# is the gripper-joint child near the PincOpen base — it sits ~5cm behind
# the actual fingertips, so phase targets that want fingers wrapping the
# cube need a +Z offset (oracle adds it via per-phase target_fn). We tried
# pinning EE at `left_distal_link` (fingertip) but it moves with the
# 4-bar mimic and the IK couldn't keep up; gripper_dummy_link is stable.
_EE_LINK_NAME = "gripper_dummy_link"

# Physics rate = policy_rate × decimation. Phase 1 결정 사항: 20 Hz × 6 = 120 Hz.
_DECIMATION = 6


class CubeLiftEnv:
    """SO-ARM101 cube-lift env. Conforms to ``khj_rl.training.base.EnvLike``.

    Requires an active Isaac Sim AppLauncher context — instantiate after
    booting ``isaaclab.app.AppLauncher``.
    """

    def __init__(self, cfg: CubeLiftEnvCfg, *, include_viewer_camera: bool = False) -> None:
        # Lazy isaaclab imports (carb is only valid after AppLauncher boot).
        import os

        import torch
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.utils.math import quat_apply

        from khj_rl.envs.cube_lift.scene import cube_lift_scene_cfg

        self.cfg = cfg
        self._torch = torch
        # Cache the quaternion-rotate helper for ``_fk_fingertip_center``
        # (called every oracle step). Lazy-imported here because
        # isaaclab.utils.math is only valid after the AppLauncher boot
        # that this env's caller is required to have already done.
        self._quat_apply = quat_apply

        # NOTE: ``SimulationContext`` is a Carb/Omniverse singleton — only one
        # ``CubeLiftEnv`` may exist per process. Phase 1 runs single-env so
        # this is fine; the multi-env path lands with the Phase 2
        # ``ManagerBasedRLEnv`` migration (see CLAUDE.md 결정 사항 1).
        # Physics dt = policy dt / decimation.
        self._physics_dt = cfg.control.dt / _DECIMATION
        # Device resolution:
        # - Headless training/testing isolates with ``CUDA_VISIBLE_DEVICES=1``
        #   and uses ``cuda:0`` (the only visible GPU).
        # - Livestream/render paths conflict with CUDA_VISIBLE_DEVICES
        #   (Omniverse logs ``Skipping NVIDIA GPU due CUDA being in bad
        #   state`` and the viewport never activates), so the viewer
        #   script unsets that var and points us directly at ``cuda:1``
        #   via ``KHJ_RL_SIM_DEVICE``.
        device = os.environ.get("KHJ_RL_SIM_DEVICE", "cuda:0")
        # use_fabric=True — without fabric Isaac Lab does not mirror PhysX
        # state back into USD, so the WebRTC viewport (which renders from
        # USD) stayed frozen on the env.reset pose forever. The bundled
        # omni.physx.fabric-106.3.2 has an ABI mismatch (v0.2 vs core
        # v1.2) so Kit silently fell back to the working 106.5.3 fabric
        # only AFTER we relaxed its exact-pin dependency on
        # omni.physx==106.5.3 (our build ships 106.5.7). The patched
        # extension.toml is documented in the commit message accompanying
        # this change.
        self._sim = SimulationContext(
            SimulationCfg(
                dt=self._physics_dt,
                render_interval=_DECIMATION,
                device=device,
                use_fabric=True,
            )
        )
        scene_cfg = cube_lift_scene_cfg(cfg, include_viewer_camera=include_viewer_camera)
        self._scene = InteractiveScene(scene_cfg)
        # First reset spawns prims and builds physics views.
        self._sim.reset()

        self._robot = self._scene["robot"]
        self._cube = self._scene["cube"]
        self._viewer_camera = (
            self._scene["viewer_camera"] if include_viewer_camera else None
        )
        # CameraCfg.OffsetCfg with convention="world" left the sensor at
        # world origin (cam_pos_w=[0,0,0], quat=[0,0,0,0]), so every
        # captured frame showed the same dark slice from inside the robot
        # base. set_world_poses_from_view + USD xform stamp both failed to
        # update the rendered output on this version. Fall back to Isaac
        # Sim's canonical viewport API which DOES drive the render product.
        if self._viewer_camera is not None:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(
                eye=(0.7, 0.5, 0.5),
                target=(0.2, 0.0, 0.05),
                camera_prim_path=self._viewer_camera.cfg.prim_path,
            )

        # Resolve body / joint indices once (names → USD prim indices).
        ee_ids, _ = self._robot.find_bodies(_EE_LINK_NAME)
        if not ee_ids:
            raise RuntimeError(
                f"body '{_EE_LINK_NAME}' not found in articulation; "
                f"available: {self._robot.body_names}"
            )
        self._ee_body_idx = ee_ids[0]
        l_dist_ids, _ = self._robot.find_bodies("left_distal_link")
        r_dist_ids, _ = self._robot.find_bodies("right_distal_link")
        if not l_dist_ids or not r_dist_ids:
            raise RuntimeError(
                "left_distal_link / right_distal_link not found in articulation; "
                f"available: {self._robot.body_names}"
            )
        self._left_distal_body_idx = l_dist_ids[0]
        self._right_distal_body_idx = r_dist_ids[0]
        self._arm_joint_idxs, _ = self._robot.find_joints(
            list(cfg.robot.joint_names), preserve_order=True
        )
        gripper_idxs, _ = self._robot.find_joints([cfg.robot.gripper_joint_name])
        if not gripper_idxs:
            raise RuntimeError(
                f"joint '{cfg.robot.gripper_joint_name}' not found; "
                f"available: {self._robot.joint_names}"
            )
        self._gripper_joint_idx = gripper_idxs[0]
        # 4-bar mimic relations (solved from the PincOpen mujoco model's
        # <equality> block). PhysX MimicJointAPI is supposed to enforce
        # these, but in this isaacsim 4.5 install the fabric extension's
        # ABI mismatch (IPhysxPrivate v0.2 vs core v1.2) lets the mimic
        # be ignored — the 4-bar joints drift freely and the cube never
        # gets grasped. We force the relation in step() by writing the
        # joint state every physics tick.
        self._mimic_joint_idxs, _ = self._robot.find_joints(
            ["left_proximal", "left_distal", "right_proximal", "right_distal"],
            preserve_order=True,
        )
        # joint = multiplier * gripper. Magnitudes (±0.5) come from the
        # PincOpen ROS2 driver's xacro <mimic multiplier=...>; signs are
        # the same mirror pattern (left/right and proximal/distal each
        # paired as opposites) we derived from the mujoco equality block.
        # Earlier ±1.0 made the fingers swing twice as far and rake the
        # cube sideways instead of converging on it.
        self._mimic_signs = torch.tensor(
            [+0.5, -0.5, -0.5, +0.5],
            device=self._sim.device,
            dtype=torch.float32,
        )

        # One-time structural dump for IK debugging. The jacobian below is
        # the raw PhysX view; ``arm_idx`` are joint_pos indices. The script
        # `scripts/oracle.py` builds IK by treating jacobian column
        # ``n_root_dofs + joint_pos_idx`` as the column for that joint —
        # which holds only if joint_pos order matches the jacobian's DOF
        # order. The first three rows (linear) of each "arm" column must
        # therefore be non-zero for joints that actually move the EE.
        jac = self._robot.root_physx_view.get_jacobians()
        n_root_dofs = jac.shape[3] - self._robot.data.joint_pos.shape[1]
        print(
            f"[env.diag] joint_names={list(self._robot.joint_names)}",
            flush=True,
        )
        print(
            f"[env.diag] body_names={list(self._robot.body_names)} "
            f"ee_body_idx={self._ee_body_idx}",
            flush=True,
        )
        print(
            f"[env.diag] arm_joint_idxs={list(self._arm_joint_idxs)} "
            f"gripper_idx={self._gripper_joint_idx} "
            f"mimic_idxs={list(self._mimic_joint_idxs)}",
            flush=True,
        )
        print(
            f"[env.diag] jac.shape={tuple(jac.shape)} n_root_dofs={n_root_dofs}",
            flush=True,
        )
        for slot, jidx in enumerate(self._arm_joint_idxs):
            col = n_root_dofs + jidx
            lin = jac[0, self._ee_body_idx, :3, col].detach().cpu().numpy()
            lo = float(self._robot.data.soft_joint_pos_limits[0, jidx, 0])
            hi = float(self._robot.data.soft_joint_pos_limits[0, jidx, 1])
            print(
                f"[env.diag] arm slot {slot} (jp_idx={jidx} -> jac_col={col}): "
                f"d_ee/d_q = {lin.round(4).tolist()} "
                f"limits=[{lo:.3f}, {hi:.3f}] rad",
                flush=True,
            )
        # Gripper limit print — we extended the URDF/USD limits from
        # ±0.77 → ±1.5 (the closed pose needs joint ≤ -1.0). If
        # Isaac Lab's articulation parser dropped the USD edit, this
        # print shows the old ±0.77 and target clamping eats the close
        # command before PD ever sees it.
        g_lo = float(
            self._robot.data.soft_joint_pos_limits[0, self._gripper_joint_idx, 0]
        )
        g_hi = float(
            self._robot.data.soft_joint_pos_limits[0, self._gripper_joint_idx, 1]
        )
        print(
            f"[env.diag] gripper (jp_idx={self._gripper_joint_idx}): "
            f"limits=[{g_lo:.3f}, {g_hi:.3f}] rad (PRE-OVERRIDE)",
            flush=True,
        )
        # USD physics revolute joints store limits in DEGREES, not radians
        # (per the USDPhysics schema convention). The URDF→USD converter
        # carries the URDF ``<limit lower="-0.77" upper="0.77">`` straight
        # through as-is, but Isaac Lab parses the result as degrees:
        # ±0.77° ≈ ±0.013 rad after the soft-limit factor lands on the
        # 4-bar joints, ±0.026 rad on the gripper main joint after a
        # round-trip — every joint here was clamped to a sub-degree
        # micro-range, so PhysX swallowed every gripper command before
        # the PD ever saw it and ``_force_mimic_coupling`` kept getting
        # clamped back to a flat range as soon as the next sim step
        # ran. Override every gripper-side joint at runtime so the
        # PincOpen 4-bar can actually swing.
        #
        # Override the gripper main + 4-bar mimic joint soft limits with
        # the cfg-defined target ranges (see cfg.RobotCfg). USD physics
        # parses URDF radian limits as degrees → 1.5 rad → 1.5° → 0.026
        # rad, which clamps every gripper command to a sub-degree
        # micro-range. The cfg fields are the single source of truth
        # for gripper joint range — env.step's action mapping reads
        # the same fields so soft limit and command range can never
        # drift apart.
        gripper_id = self._gripper_joint_idx
        mimic_ids = list(self._mimic_joint_idxs)
        g_lo = float(cfg.robot.gripper_joint_target_low)
        g_hi = float(cfg.robot.gripper_joint_target_high)
        m_lo = float(cfg.robot.mimic_joint_target_low)
        m_hi = float(cfg.robot.mimic_joint_target_high)
        gripper_limits = torch.tensor(
            [[[g_lo, g_hi]]], device=self._sim.device, dtype=torch.float32
        )
        mimic_limits = torch.tensor(
            [[[m_lo, m_hi]] * len(mimic_ids)],
            device=self._sim.device,
            dtype=torch.float32,
        )
        self._robot.write_joint_position_limit_to_sim(
            gripper_limits, joint_ids=[gripper_id], warn_limit_violation=False,
        )
        self._robot.write_joint_position_limit_to_sim(
            mimic_limits, joint_ids=mimic_ids, warn_limit_violation=False,
        )
        for jidx in [gripper_id, *mimic_ids]:
            lo = float(self._robot.data.soft_joint_pos_limits[0, jidx, 0])
            hi = float(self._robot.data.soft_joint_pos_limits[0, jidx, 1])
            name = self._robot.joint_names[jidx]
            print(
                f"[env.diag] gripper-side joint '{name}' (jp_idx={jidx}): "
                f"limits=[{lo:.3f}, {hi:.3f}] rad (POST-OVERRIDE)",
                flush=True,
            )

        # Per-step state.
        self._success = MultiConditionSuccess(
            goal=cfg.goal,
            success_cfg=cfg.success,
            gripper_open_threshold=cfg.robot.gripper_open_threshold,
        )
        self._noise = cfg.noise.to_noise_model()
        self._rng = np.random.default_rng(cfg.seed)
        self._step_idx = 0
        # Global step counter for dense-reward annealing. Trainer must
        # call ``set_global_step`` each rollout step; dense_alpha() reads
        # this. Plain 0 default keeps the env usable from collect_demos
        # / eval without the trainer wiring (those paths leave dense
        # disabled in cfg anyway).
        self._global_step = 0
        self._last_action = np.zeros(cfg.action_size, dtype=np.float32)
        self._last_known_cube_xyz: np.ndarray | None = None
        # G+ hover-attractor tracking: when lift_history latches we record
        # the step and the cube xy at that moment. The level-0 force_term
        # rule + the post-lift drift metric both read from these.
        self._lift_latched_step: int = -1
        self._lift_latched_cube_xy: np.ndarray | None = None
        self._hover_steps_post_lift: int = 0

        # Pinocchio FK setup. The URDF root joint is fixed (universe is the
        # only joint before shoulder_pan), so pinocchio's "world" frame
        # coincides with the robot base — FK output is already in base
        # frame, which is exactly what ee_pose_base expects.
        import pinocchio as pin  # lazy: pinocchio import is slow

        urdf_path = (
            Path(__file__).resolve().parents[4]
            / "assets"
            / "converted"
            / "urdf"
            / "so101_pincopen_gripper.urdf"
        )
        self._pin_model = pin.buildModelFromUrdf(str(urdf_path))
        self._pin_data = self._pin_model.createData()
        self._pin_ee_frame_id = self._pin_model.getFrameId(_EE_LINK_NAME)
        # Map pinocchio joint name -> q vector index for ordered fill-in.
        self._pin_joint_qidx: dict[str, int] = {}
        for joint_id in range(1, self._pin_model.njoints):  # skip "universe"
            j = self._pin_model.joints[joint_id]
            self._pin_joint_qidx[self._pin_model.names[joint_id]] = j.idx_q
        # Pre-compute the q indices for the arm joints in the same order
        # _fk_ee_pose receives them (cfg.robot.joint_names order).
        self._pin_arm_qidx = [
            self._pin_joint_qidx[name] for name in cfg.robot.joint_names
        ]
        self._pin_gripper_qidx = self._pin_joint_qidx[cfg.robot.gripper_joint_name]
        self._pin_mimic_qidx_signs = [
            (self._pin_joint_qidx["left_proximal"], +0.5),
            (self._pin_joint_qidx["left_distal"], -0.5),
            (self._pin_joint_qidx["right_proximal"], -0.5),
            (self._pin_joint_qidx["right_distal"], +0.5),
        ]
        # Frame ids for the two PincOpen fingertips — used to compute a
        # dynamic dummy→fingertip-center offset in oracle (the static
        # offset assumed at the home pose drifts as the wrist rotates).
        self._pin_left_distal_fid = self._pin_model.getFrameId("left_distal_link")
        self._pin_right_distal_fid = self._pin_model.getFrameId("right_distal_link")


    # ---- EnvLike contract ------------------------------------------------

    @property
    def observation_space(self) -> Any:
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.obs_total_size,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> Any:
        # Normalized [-1, 1] for all action dims; env.step un-normalizes.
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.cfg.action_size,),
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        import sys as _sys

        def _t(m):
            print(f"[reset.trace] {m}", flush=True, file=_sys.stderr)
        _t("ENTER")
        self._success.reset()
        self._step_idx = 0
        self._last_action = np.zeros(self.cfg.action_size, dtype=np.float32)
        self._last_known_cube_xyz = None
        # G+ hover-attractor tracking reset
        self._lift_latched_step = -1
        self._lift_latched_cube_xy = None
        self._hover_steps_post_lift = 0

        torch = self._torch

        # 1) Robot home pose: start from articulation default, overwrite the
        #    arm + gripper subset we own.
        _t("before default_joint_pos clone")
        default_pos = self._robot.data.default_joint_pos.clone()
        for slot, jidx in enumerate(self._arm_joint_idxs):
            default_pos[:, jidx] = float(self.cfg.robot.joint_init[slot])
        default_pos[:, self._gripper_joint_idx] = 0.0  # closed at start
        default_vel = torch.zeros_like(default_pos)
        _t("before write_joint_state_to_sim")
        self._robot.write_joint_state_to_sim(position=default_pos, velocity=default_vel)
        _t("before write_root_pose_to_sim")
        # Reset root pose / velocity so the articulation lands at the origin.
        default_root_state = self._robot.data.default_root_state.clone()
        self._robot.write_root_pose_to_sim(default_root_state[:, :7])
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:])

        # 2) Cube spawn: random xy inside the active curriculum stage around
        # the goal. Stage index lives in cfg.curriculum so eval scripts can
        # advance the curriculum without code edits.
        _t("before cube spawn")
        stage_idx = int(self.cfg.curriculum.current_stage_idx)
        side = float(self.cfg.curriculum.side_length_m[stage_idx])
        cube_xy = self._rng.uniform(low=-side / 2, high=side / 2, size=2)
        cube_xy += np.asarray(self.cfg.goal.pos_xyz_m[:2])
        cube_pose = torch.tensor(
            [
                [
                    float(cube_xy[0]),
                    float(cube_xy[1]),
                    float(self.cfg.cube.spawn_z_m),
                    1.0,  # quat w
                    0.0,
                    0.0,
                    0.0,
                ]
            ],
            device=self._sim.device,
        )
        self._cube.write_root_pose_to_sim(cube_pose)
        self._cube.write_root_velocity_to_sim(
            torch.zeros((1, 6), device=self._sim.device)
        )

        # 3) Push writes to sim, settle, then refresh data buffers.
        _t("before scene.write_data_to_sim")
        self._scene.write_data_to_sim()
        _t("before sim.forward")
        self._sim.forward()
        _t("before scene.update")
        self._scene.update(dt=self._physics_dt)

        _t("before compute_obs")
        obs = self._compute_obs()
        _t("EXIT")
        return obs

    def set_global_step(self, step: int) -> None:
        """Trainer-side hook to drive dense-reward annealing.

        The PPO trainer calls this each rollout step so ``dense_alpha``
        sees the *learning-progress* step, not the within-episode step
        counter. No-op outside dense_reward.enabled paths — leaving
        this unwired is safe.
        """
        self._global_step = int(step)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        torch = self._torch

        # Un-normalize: arm = current + delta; gripper = absolute.
        arm_delta = action[: self.cfg.robot.n_joints] * self.cfg.robot.action_delta_max
        gripper_low = self.cfg.robot.gripper_joint_target_low
        gripper_high = self.cfg.robot.gripper_joint_target_high
        gripper_offset = 0.5 * (gripper_low + gripper_high)
        gripper_scale = 0.5 * (gripper_high - gripper_low)
        gripper_target = gripper_offset + float(action[-1]) * gripper_scale

        target = self._robot.data.joint_pos.clone()
        for slot, jidx in enumerate(self._arm_joint_idxs):
            target[:, jidx] = target[:, jidx] + float(arm_delta[slot])
        target[:, self._gripper_joint_idx] = gripper_target
        for slot, jidx in enumerate(self._mimic_joint_idxs):
            sign = float(self._mimic_signs[slot].item())
            target[:, jidx] = sign * gripper_target
        limits = self._robot.data.soft_joint_pos_limits
        target = torch.clamp(target, limits[..., 0], limits[..., 1])

        self._robot.set_joint_position_target(target)

        # Decimation loop: physics 120 Hz × 6 = policy 20 Hz. The renderer
        # is enabled only on the last physics tick so the viewer's camera
        # buffer refreshes once per policy step; the other 5 ticks run
        # ``render=False`` to keep training throughput up. When the env is
        # constructed without ``include_viewer_camera`` the renderer still
        # has nothing wired to its output, so the cost is minimal.
        for tick in range(_DECIMATION):
            self._scene.write_data_to_sim()
            render = tick == _DECIMATION - 1
            self._sim.step(render=render)
            self._scene.update(dt=self._physics_dt)

        self._last_action = action.copy()
        self._step_idx += 1

        obs = self._compute_obs()
        report, gt_state = self._compute_success()

        # G+ task-level reward decomposition (docs/g_plus_design.md §4).
        # The full-PnP 5-condition AND is too strict for sparse RL to find
        # the first success under a weak BC actor (today's 5 fails). For
        # task_level 0/1 we emit the sub-task sparse reward instead so the
        # actor gets a learning signal it can actually reach.
        task_level = int(self.cfg.curriculum.task_level)
        lifted_now = bool(report.per_condition["lift_history"])
        # Latch tracking — first step lift_history flips True is when we
        # start the hover guard timer.
        if lifted_now and self._lift_latched_step < 0:
            self._lift_latched_step = self._step_idx
            self._lift_latched_cube_xy = gt_state.cube_xyz_m[:2].astype(
                np.float32, copy=True
            )
        if lifted_now:
            self._hover_steps_post_lift += 1

        if task_level == 0:
            sparse = float(lifted_now)
        elif task_level == 1:
            # 2026-05-20 throw-hack patch: lift_history latches, so without a
            # velocity gate the policy can lift for 1s then THROW the cube
            # into the goal radius and still collect reward. Adding
            # velocity_stability blocks the throw — cube must come to rest
            # inside the goal before episode terminates.
            sparse = float(
                lifted_now
                and report.per_condition["stable_placement"]
                and report.per_condition["velocity_stability"]
            )
        else:
            sparse = float(report.success)
        # G+ hover-attractor metric: post-lift cube xy drift.
        cube_xy_drift_post_lift = 0.0
        if self._lift_latched_cube_xy is not None:
            cube_xy_drift_post_lift = float(
                np.linalg.norm(
                    gt_state.cube_xyz_m[:2].astype(np.float32)
                    - self._lift_latched_cube_xy
                )
            )
        info = {
            "success": report.success,
            "success_per_condition": dict(report.per_condition),
            "gt_cube_xyz_m": gt_state.cube_xyz_m.copy(),
            "step": self._step_idx,
            "task_level": task_level,
            "lift_latched_step": int(self._lift_latched_step),
            "hover_steps_post_lift": int(self._hover_steps_post_lift),
            "cube_xy_drift_post_lift": cube_xy_drift_post_lift,
            "her_raw": {
                "cube_xyz_m": gt_state.cube_xyz_m.astype(np.float32).copy(),
                "cube_lin_vel_m_s": gt_state.cube_lin_vel_m_s.astype(np.float32).copy(),
                "cube_ang_vel_rad_s": gt_state.cube_ang_vel_rad_s.astype(np.float32).copy(),
                "ee_xyz_m": gt_state.ee_xyz_m.astype(np.float32).copy(),
                "gripper_opening": float(gt_state.gripper_opening),
                "visibility_top": bool(gt_state.visibility_top),
                "visibility_wrist": bool(gt_state.visibility_wrist),
                "goal_xyz_m": np.asarray(self.cfg.goal.pos_xyz_m, dtype=np.float32),
            },
        }
        if self.cfg.dense_reward.enabled:
            # Dense shaping (A option, 2026-05-17). All weights / gates
            # live in cfg.dense_reward; see docs/dense_reward_design.md.
            # _global_step is advanced by the trainer at each rollout
            # step so the linear anneal tracks learning progress, not
            # within-episode position.
            dense_report = compute_dense_reward(
                gt_state,
                self.cfg.goal,
                self.cfg.success,
                self.cfg.dense_reward,
                gripper_open_threshold=self.cfg.robot.gripper_open_threshold,
                lift_history=(self._lift_latched_step >= 0),
            )
            alpha = dense_alpha(self._global_step, self.cfg.dense_reward)
            reward = sparse + alpha * dense_report.total
            info["dense_reward_raw"] = dense_report.total
            info["dense_alpha"] = alpha
            info["dense_terms"] = dense_report.per_term
        else:
            reward = sparse
        # Episode termination:
        #   - level 0: terminate on lift_history (sub-task success) OR
        #     force_terminate when actor exceeds max_post_lift_steps after
        #     latch (hover-attractor block, docs/g_plus_design.md §4.1).
        #   - level 1: terminate on (lift AND placement).
        #   - level 2: terminate on full 5-condition success (original).
        # The sub-task-level success flag is what feeds `terminated`;
        # report.success (full 5-cond) is still kept in info for logging.
        if task_level == 0:
            sub_success = lifted_now
            force_term = (
                self._lift_latched_step >= 0
                and self._step_idx
                >= self._lift_latched_step + int(self.cfg.curriculum.max_post_lift_steps)
            )
            terminated = bool(sub_success) or bool(force_term)
            info["force_terminated"] = bool(force_term and not sub_success)
        elif task_level == 1:
            # Match the sparse-reward condition above so an episode only
            # terminates when the cube is lifted, placed AND at rest —
            # never on a throw-into-goal trajectory.
            terminated = bool(
                lifted_now
                and report.per_condition["stable_placement"]
                and report.per_condition["velocity_stability"]
            )
            info["force_terminated"] = False
        else:
            terminated = bool(report.success)
            info["force_terminated"] = False
        truncated = self._step_idx >= self.cfg.control.max_steps
        return obs, reward, terminated, truncated, info

    # ---- Obs / success ---------------------------------------------------

    def _compute_obs(self) -> np.ndarray:
        joint_pos_gt = self._sim_joint_pos_gt()
        joint_vel_gt = self._sim_joint_vel_gt()
        cube_xyz_gt = self._sim_cube_xyz_gt()
        gripper_state = np.asarray([self._sim_gripper_opening_gt()], dtype=np.float32)

        # NoiseModel touches only obs-side joint encoders + cube pose.
        joint_pos_obs = self._noise.apply_obs_joint_pos(joint_pos_gt, self._rng).astype(np.float32)
        joint_vel_obs = self._noise.apply_obs_joint_vel(joint_vel_gt, self._rng).astype(np.float32)
        noisy_xyz, dropout = self._noise.apply_obs_object_pos(
            cube_xyz_gt.astype(np.float64), self._rng
        )
        if dropout and self._last_known_cube_xyz is not None:
            cube_xyz_obs = self._last_known_cube_xyz.astype(np.float32)
        else:
            cube_xyz_obs = noisy_xyz.astype(np.float32)
        if not dropout:
            self._last_known_cube_xyz = cube_xyz_obs.copy()

        # EE pose: Phase 1 uses sim GT (joint_pos input ignored). Pinocchio FK
        # from noisy_joint_pos is the planned upgrade — until then, obs ee_pose
        # is GT (a known sim2real leak documented in CLAUDE.md결정 사항 4).
        ee_pose_base = self._fk_ee_pose(joint_pos_obs).astype(np.float32)

        goal_xyz = np.asarray(self.cfg.goal.pos_xyz_m, dtype=np.float32)
        target_delta = (goal_xyz - cube_xyz_obs).astype(np.float32)
        normalized_t = np.asarray(
            [self._step_idx / max(1, self.cfg.control.max_steps)], dtype=np.float32
        )

        parts: dict[str, np.ndarray] = {
            "joint_pos": joint_pos_obs,
            "joint_vel": joint_vel_obs,
            "ee_pose_base": ee_pose_base,
            "cube_xyz_base": cube_xyz_obs,
            "target_delta_base": target_delta,
            "gripper_state": gripper_state,
            "last_action": self._last_action,
            "normalized_t": normalized_t,
        }
        return np.concatenate([parts[k] for k in self.cfg.obs_keys], axis=0).astype(np.float32)

    def _compute_success(self) -> tuple[SuccessReport, SuccessState]:
        cube_xyz = self._sim_cube_xyz_gt().astype(np.float64)
        # Reward path: GT joint_pos → FK helper. obs path used noisy joint_pos.
        ee_xyz = self._fk_ee_pose(self._sim_joint_pos_gt()).astype(np.float64)[:3]
        state = SuccessState(
            cube_xyz_m=cube_xyz,
            cube_lin_vel_m_s=self._sim_cube_lin_vel_gt().astype(np.float64),
            cube_ang_vel_rad_s=self._sim_cube_ang_vel_gt().astype(np.float64),
            ee_xyz_m=ee_xyz,
            gripper_opening=float(self._sim_gripper_opening_gt()),
            visibility_top=self._visibility_top_proxy(cube_xyz, ee_xyz),
            visibility_wrist=self._visibility_wrist_proxy(cube_xyz, ee_xyz),
        )
        return self._success.update(state), state

    # ---- Visibility proxies (GT-pose only, NoiseModel-independent) -------

    def _visibility_top_proxy(self, cube_xyz: np.ndarray, ee_xyz: np.ndarray) -> bool:
        # Phase 1: top camera always sees the cube unless EE is directly above
        # within 5cm — simple occlusion rule, not a real raycast.
        xy_dist = float(np.linalg.norm(cube_xyz[:2] - ee_xyz[:2]))
        return not (xy_dist < 0.05 and ee_xyz[2] > cube_xyz[2])

    def _visibility_wrist_proxy(self, cube_xyz: np.ndarray, ee_xyz: np.ndarray) -> bool:
        # Phase 1: wrist camera sees the cube if EE is within ~30cm.
        dist = float(np.linalg.norm(cube_xyz - ee_xyz))
        return dist <= 0.30

    # ---- Sim GT accessors (articulation / cube buffers) ------------------

    def _sim_joint_pos_gt(self) -> np.ndarray:
        return (
            self._robot.data.joint_pos[0, self._arm_joint_idxs]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _sim_joint_vel_gt(self) -> np.ndarray:
        return (
            self._robot.data.joint_vel[0, self._arm_joint_idxs]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _sim_cube_xyz_gt(self) -> np.ndarray:
        # Single env: world frame == base frame == env origin (no offset).
        return (
            self._cube.data.root_pos_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _sim_cube_lin_vel_gt(self) -> np.ndarray:
        return (
            self._cube.data.root_lin_vel_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _sim_cube_ang_vel_gt(self) -> np.ndarray:
        return (
            self._cube.data.root_ang_vel_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _sim_ee_pose_gt(self) -> np.ndarray:
        """GT end-effector pose direct from the sim's body state buffer.

        Returns xyz (3) + quat[w, x, y, z] (4) of the EE body in world frame.
        This is the ground-truth path used by reward / success / oracle code;
        the obs path goes through ``_fk_ee_pose`` (Phase 1 also reads sim GT
        there, but will swap to a Pinocchio FK from noisy joint_pos).
        """
        pos = (
            self._robot.data.body_pos_w[0, self._ee_body_idx]
            .detach()
            .cpu()
            .numpy()
        )
        quat = (
            self._robot.data.body_quat_w[0, self._ee_body_idx]
            .detach()
            .cpu()
            .numpy()
        )
        pose = np.empty(7, dtype=np.float32)
        pose[:3] = pos.astype(np.float32)
        pose[3:] = quat.astype(np.float32)
        return pose

    def _sim_gripper_opening_gt(self) -> float:
        gripper_pos = float(
            self._robot.data.joint_pos[0, self._gripper_joint_idx].detach().cpu().item()
        )
        lo = float(self._robot.data.soft_joint_pos_limits[0, self._gripper_joint_idx, 0])
        hi = float(self._robot.data.soft_joint_pos_limits[0, self._gripper_joint_idx, 1])
        span = hi - lo
        if span <= 1e-6:
            return 0.0
        return float(np.clip((gripper_pos - lo) / span, 0.0, 1.0))

    # NOTE: ``_force_mimic_coupling`` removed. The earlier approach
    # (write joint state for the 4-bar passive joints every physics
    # tick to multiplier × gripper) fought PhysX's contact resolver:
    # cube tried to push fingers apart, our write snapped them back
    # in, and the loop produced a high-frequency cube vibration with
    # no visible finger contact. The 4-bar is now driven by a normal
    # PD actuator group (see ``articulation.py`` "mimic" actuator)
    # and ``env.step`` sets ``set_joint_position_target`` for those
    # joints alongside the arm + gripper.

    def _fk_fingertip_center(self, joint_pos: np.ndarray) -> np.ndarray:
        """World-frame midpoint of the two PincOpen fingertip CONTACT
        points (not the distal-link body origins).

        The URDF's left/right_distal_link body origin sits at the
        proximal-distal hinge, but the actual finger contact surface
        is at the FAR end of the distal mesh — STL bounding box shows
        the tip occupies x ∈ [0.005, 0.036], y ∈ [-0.013, -0.005] in
        the link's local frame (mirror Y for the right). If we drove
        IK to put the body-origin midpoint at the cube, the visible
        fingertips ended up on the cube's near face and the close
        shoved the cube toward the robot. Picking a representative
        tip-contact point in local coords (0.025, ±0.009, 0.010) and
        rotating it into world via the body quaternion gives the
        actual grasp contact location.

        ``joint_pos`` is accepted for API compatibility (legacy
        callers pass GT arm angles) but no longer consulted.
        """
        del joint_pos  # unused; kept for callsite compatibility
        torch = self._torch
        # Local-frame tip-contact offset (link x = along finger, link y
        # = inward toward the other finger). Picked roughly at the
        # midpoint of the gripping pad. Right finger mirrors Y.
        offset_local_left = torch.tensor(
            [0.025, -0.009, 0.010], device=self._sim.device, dtype=torch.float32
        )
        offset_local_right = torch.tensor(
            [0.025, +0.009, 0.010], device=self._sim.device, dtype=torch.float32
        )
        l_pos = self._robot.data.body_pos_w[0, self._left_distal_body_idx]
        r_pos = self._robot.data.body_pos_w[0, self._right_distal_body_idx]
        l_quat = self._robot.data.body_quat_w[0, self._left_distal_body_idx]
        r_quat = self._robot.data.body_quat_w[0, self._right_distal_body_idx]
        l_tip = l_pos + self._quat_apply(l_quat.unsqueeze(0), offset_local_left.unsqueeze(0))[0]
        r_tip = r_pos + self._quat_apply(r_quat.unsqueeze(0), offset_local_right.unsqueeze(0))[0]
        return ((l_tip + r_tip) * 0.5).detach().cpu().numpy().astype(np.float32)

    def _fk_ee_pose(self, joint_pos: np.ndarray) -> np.ndarray:
        """SO-ARM101 forward kinematics: arm joint_pos (5,) -> EE pose (7,).

        Pure function of the 5-arm-joint vector (in ``cfg.robot.joint_names``
        order). Builds the full 10-DOF pinocchio q internally: arm slots
        take the input, the gripper main joint is pinned at 0 (its motion
        rotates the gripper assembly relative to the wrist but doesn't
        translate ``gripper_dummy_link`` significantly — small accuracy
        cost we accept for sim2real cleanliness), and the four passive
        4-bar joints follow the gripper via the same ±0.5 multiplier
        ``_force_mimic_coupling`` enforces in physics.

        Returns xyz + quat[w, x, y, z] of ``gripper_dummy_link`` in the
        pinocchio model's root frame, which is the robot base frame
        (URDF root joint is fixed). Same 7-D layout as
        ``_sim_ee_pose_gt`` so reward / oracle code can be retargeted
        without obs-shape changes.
        """
        import pinocchio as pin

        arm = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
        if arm.shape[0] != len(self._pin_arm_qidx):
            raise ValueError(
                f"_fk_ee_pose expected arm (5,), got shape {arm.shape}"
            )
        q = np.zeros(self._pin_model.nq, dtype=np.float64)
        for slot, qidx in enumerate(self._pin_arm_qidx):
            q[qidx] = float(arm[slot])
        # Gripper / mimic = 0 — the obs path's FK should not depend on the
        # gripper state (it's a separate obs component). Reward path's
        # ee_xyz also uses [:3] only, which gripper joint barely affects.
        q[self._pin_gripper_qidx] = 0.0
        for qidx, _sign in self._pin_mimic_qidx_signs:
            q[qidx] = 0.0

        pin.framesForwardKinematics(self._pin_model, self._pin_data, q)
        oMf = self._pin_data.oMf[self._pin_ee_frame_id]
        pos = np.asarray(oMf.translation, dtype=np.float32)
        # pin.Quaternion(matrix) returns (x, y, z, w); reorder to (w, x, y, z).
        quat_xyzw = np.asarray(pin.Quaternion(oMf.rotation).coeffs(), dtype=np.float32)
        pose = np.empty(7, dtype=np.float32)
        pose[:3] = pos
        pose[3] = quat_xyzw[3]
        pose[4:7] = quat_xyzw[:3]
        return pose
