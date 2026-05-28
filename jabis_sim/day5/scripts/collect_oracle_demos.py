"""Oracle scripted demo collection for SO-ARM101 cube lift (Phase 1'' of B-3 BC warmstart).

State machine: REACH_ABOVE → DESCEND → CLOSE → LIFT.
- Position-only IK (5-DOF arm cannot satisfy 6-DOF pose).
- CLOSE: frozen joint target locked at DESCEND exit (no IK drift during contact).
- LIFT:  ee target locked at CLOSE exit + (0,0,+0.10) (no receding target).
- TRACK: omitted — LIFT held until episode timeout, success = cube_z >= 0.10.

Saves (obs_36, action_6) per step for episodes where cube_z_max >= success_z.
Output is compatible with mine_success_demos.py format so the BC trainer can ingest both.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect oracle scripted demos via DifferentialIK.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Lift-Cube-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--target_episodes", type=int, default=200)
parser.add_argument("--max_total_episodes", type=int, default=2000)
parser.add_argument("--success_z", type=float, default=0.10)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)

# state-machine thresholds (CLI-tunable for debugging)
parser.add_argument("--reach_dist", type=float, default=0.02, help="0→1 transition threshold (m).")
parser.add_argument("--descend_z_dist", type=float, default=0.005, help="1→2 transition |dz| threshold (m).")
parser.add_argument("--close_steps", type=int, default=8, help="2→3 transition: hold close N steps.")
parser.add_argument("--reach_above_dz", type=float, default=0.10, help="REACH_ABOVE z offset above cube (m).")
parser.add_argument("--descend_dz", type=float, default=0.005, help="DESCEND z offset above cube top (m).")
parser.add_argument("--lift_dz", type=float, default=0.10, help="LIFT delta z above CLOSE pose (m).")
parser.add_argument("--max_ee_step", type=float, default=0.02, help="Per-step ee position delta cap (m).")
parser.add_argument("--debug_first_steps", type=int, default=0, help="Print debug for first N steps.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


# State enum
S_REACH_ABOVE = 0
S_DESCEND = 1
S_CLOSE = 2
S_LIFT = 3


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    device = env.unwrapped.device
    n_envs = env.unwrapped.num_envs
    robot = env.unwrapped.scene["robot"]
    object_asset = env.unwrapped.scene["object"]

    # Resolve ee body & arm joint indices
    arm_joint_names_re = ["shoulder_.*", "elbow_flex", "wrist_.*"]
    arm_ids = robot.find_joints(arm_joint_names_re)[0]
    arm_ids_t = torch.tensor(arm_ids, device=device, dtype=torch.long)
    ee_idx = robot.find_bodies("gripper_link")[0][0]
    ee_jacobi_idx = ee_idx - 1

    print(f"[INFO] arm_ids={arm_ids}  ee_idx={ee_idx}  ee_jacobi_idx={ee_jacobi_idx}  n_envs={n_envs}")
    print(f"[INFO] arm joints: {[robot.joint_names[i] for i in arm_ids]}")
    print(f"[INFO] ee body: {robot.body_names[ee_idx]}")

    default_arm = robot.data.default_joint_pos[:, arm_ids_t].clone()  # (N, 5)
    print(f"[INFO] default_arm[0] = {default_arm[0].cpu().tolist()}")

    # IK controller — position-only
    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik = DifferentialIKController(ik_cfg, num_envs=n_envs, device=device)

    # State buffers (per env)
    states = torch.zeros(n_envs, dtype=torch.long, device=device)  # all start REACH_ABOVE
    close_step = torch.zeros(n_envs, dtype=torch.long, device=device)
    frozen_close_joint = torch.zeros(n_envs, len(arm_ids), device=device)
    lift_target_pos_b = torch.zeros(n_envs, 3, device=device)

    # Episode buffers
    obs = env.get_observations()
    obs_tensor = obs["policy"]
    obs_dim = obs_tensor.shape[1]
    print(f"[INFO] obs_dim={obs_dim}, target={args_cli.target_episodes}, success_z={args_cli.success_z}")

    episode_obs = [[] for _ in range(n_envs)]
    episode_act = [[] for _ in range(n_envs)]
    episode_z_max = torch.zeros(n_envs, device=device)

    demos_obs: list[torch.Tensor] = []
    demos_act: list[torch.Tensor] = []
    demos_meta: list[dict] = []
    successes = 0
    finished_episodes = 0
    step_count = 0
    saturation_count = torch.zeros(5, device=device)  # per-joint count of clamped actions
    saturation_total = 0
    start_time = time.time()

    print("[INFO] Collecting...")

    while successes < args_cli.target_episodes and finished_episodes < args_cli.max_total_episodes:
        with torch.no_grad():
            # 1) Frames
            ee_pose_w = robot.data.body_pose_w[:, ee_idx]
            root_pose_w = robot.data.root_pose_w
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
            )
            cube_pos_w = object_asset.data.root_pos_w
            # base rotation is identity per SO_ARM101_CFG → use simple subtract
            cube_pos_b = cube_pos_w - root_pose_w[:, 0:3]

            cube_z_local = (cube_pos_w[:, 2] - env.unwrapped.scene.env_origins[:, 2]).clone()

            # 2) Per-state ee target position (base frame)
            target_pos_b = ee_pos_b.clone()
            mask_reach = states == S_REACH_ABOVE
            mask_descend = states == S_DESCEND
            mask_close = states == S_CLOSE
            mask_lift = states == S_LIFT

            offset_reach = torch.tensor([0.0, 0.0, args_cli.reach_above_dz], device=device)
            offset_descend = torch.tensor([0.0, 0.0, args_cli.descend_dz], device=device)
            target_pos_b[mask_reach] = cube_pos_b[mask_reach] + offset_reach
            target_pos_b[mask_descend] = cube_pos_b[mask_descend] + offset_descend
            # CLOSE: target unused (frozen joint hold below)
            target_pos_b[mask_lift] = lift_target_pos_b[mask_lift]

            # 3) Jacobian (base-frame transform)
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_ids_t].clone()
            base_rot_inv = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
            jacobian[:, :3, :] = torch.bmm(base_rot_inv, jacobian[:, :3, :])
            jacobian[:, 3:, :] = torch.bmm(base_rot_inv, jacobian[:, 3:, :])
            joint_pos_arm = robot.data.joint_pos[:, arm_ids_t]

            # 4a) Cap position_error to prevent IK from emitting joint jumps that saturate the
            #     PPO action range (±0.5 rad/step under scale=0.5). Per-step ee delta ≤ max_ee_step.
            delta_target = target_pos_b - ee_pos_b
            delta_norm = delta_target.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            scale = (args_cli.max_ee_step / delta_norm).clamp(max=1.0)
            target_capped = ee_pos_b + delta_target * scale

            # 4b) IK compute (position-only — needs ee_quat for orientation hold)
            ik.set_command(command=target_capped, ee_quat=ee_quat_b)
            joint_target = ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos_arm)

            # 5) Override CLOSE with frozen target
            if mask_close.any():
                joint_target = torch.where(mask_close.unsqueeze(-1), frozen_close_joint, joint_target)

            # 6) Action conversion
            arm_raw = (joint_target - default_arm) / 0.5
            arm_clamped = arm_raw.clamp(-1.0, 1.0)
            saturated = (arm_raw.abs() > 1.0).to(torch.float)
            saturation_count += saturated.sum(dim=0).detach()
            saturation_total += n_envs

            gripper_open_mask = states < S_CLOSE
            gripper_raw = torch.where(
                gripper_open_mask,
                torch.tensor(1.0, device=device),
                torch.tensor(-1.0, device=device),
            ).unsqueeze(-1)
            action_for_env = torch.cat([arm_clamped, gripper_raw], dim=-1)

        # 7) Buffer (outside inference_mode for storage tensors)
        for i in range(n_envs):
            episode_obs[i].append(obs_tensor[i].detach().cpu().clone())
            episode_act[i].append(action_for_env[i].detach().cpu().clone())
        episode_z_max = torch.maximum(episode_z_max, cube_z_local)

        with torch.no_grad():
            # 8) State transitions BEFORE stepping (use current ee pose vs target_pos_b)
            dist3 = torch.norm(target_pos_b - ee_pos_b, dim=-1)
            dz_abs = (target_pos_b[:, 2] - ee_pos_b[:, 2]).abs()

            t_0_1 = mask_reach & (dist3 < args_cli.reach_dist)
            t_1_2 = mask_descend & (dz_abs < args_cli.descend_z_dist)
            t_2_3 = mask_close & (close_step >= args_cli.close_steps)

            # Lock frozen joint target at DESCEND exit
            if t_1_2.any():
                frozen_close_joint = torch.where(
                    t_1_2.unsqueeze(-1), joint_pos_arm.detach().clone(), frozen_close_joint
                )
            # Lock lift target at CLOSE exit
            if t_2_3.any():
                new_lift = ee_pos_b.detach().clone()
                new_lift[:, 2] = new_lift[:, 2] + args_cli.lift_dz
                lift_target_pos_b = torch.where(t_2_3.unsqueeze(-1), new_lift, lift_target_pos_b)

            # Update close_step
            close_step = torch.where(mask_close, close_step + 1, torch.zeros_like(close_step))

            # Apply transitions
            states = torch.where(t_0_1, torch.full_like(states, S_DESCEND), states)
            states = torch.where(t_1_2, torch.full_like(states, S_CLOSE), states)
            states = torch.where(t_2_3, torch.full_like(states, S_LIFT), states)

            # 9) env.step
            obs, _, dones, _ = env.step(action_for_env)
            obs_tensor = obs["policy"]
            done_indices = torch.nonzero(dones, as_tuple=False).squeeze(-1).tolist()

        # Debug: print first env state for the first N steps
        if step_count < args_cli.debug_first_steps:
            print(
                f"[DBG step={step_count}] state[0]={int(states[0])} "
                f"ee_b={ee_pos_b[0].cpu().tolist()} cube_b={cube_pos_b[0].cpu().tolist()} "
                f"target={target_pos_b[0].cpu().tolist()} dist={float((target_pos_b[0]-ee_pos_b[0]).norm()):.3f} "
                f"sat={[int(x) for x in (arm_raw[0].abs() > 1.0).tolist()]}"
            )

        # 10) Done handling
        step_count += 1
        for env_idx in done_indices:
            finished_episodes += 1
            z_max = float(episode_z_max[env_idx])
            ep_len = len(episode_obs[env_idx])
            if z_max >= args_cli.success_z and ep_len > 0:
                demos_obs.append(torch.stack(episode_obs[env_idx]))
                demos_act.append(torch.stack(episode_act[env_idx]))
                demos_meta.append({
                    "length": ep_len, "z_max": z_max, "env_idx": env_idx,
                    "final_state": int(states[env_idx]),
                })
                successes += 1
                if successes % 10 == 0 or successes == args_cli.target_episodes:
                    elapsed = time.time() - start_time
                    rate = successes / max(finished_episodes, 1) * 100.0
                    print(
                        f"[INFO] succ={successes}/{args_cli.target_episodes} "
                        f"finished={finished_episodes} rate={rate:.1f}% "
                        f"z_max={z_max:.3f} len={ep_len} elapsed={elapsed:.0f}s"
                    )
            episode_obs[env_idx] = []
            episode_act[env_idx] = []
            episode_z_max[env_idx] = 0.0
            states[env_idx] = S_REACH_ABOVE
            close_step[env_idx] = 0
            frozen_close_joint[env_idx] = 0.0
            lift_target_pos_b[env_idx] = 0.0

    # Summary
    elapsed = time.time() - start_time
    rate = successes / max(finished_episodes, 1) * 100.0
    sat_rate_pj = (saturation_count / max(saturation_total, 1) * 100.0).cpu().tolist()
    print(
        f"\n[DONE] succ={successes} finished={finished_episodes} rate={rate:.2f}% "
        f"steps={step_count} elapsed={elapsed:.0f}s"
    )
    print(f"[DONE] per-joint saturation rate (%): {[f'{r:.1f}' for r in sat_rate_pj]}")

    if successes == 0:
        print("[WARN] no successes — saving skipped. Check IK/state-machine logs.")
    else:
        os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
        torch.save(
            {
                "obs": demos_obs,
                "act": demos_act,
                "meta": demos_meta,
                "task": args_cli.task,
                "obs_dim": obs_dim,
                "act_dim": int(demos_act[0].shape[1]),
                "success_z": args_cli.success_z,
                "source": "oracle_ik_state_machine",
                "saturation_per_joint_pct": sat_rate_pj,
                "n_total_episodes_attempted": finished_episodes,
                "params": {
                    "reach_dist": args_cli.reach_dist,
                    "descend_z_dist": args_cli.descend_z_dist,
                    "close_steps": args_cli.close_steps,
                    "reach_above_dz": args_cli.reach_above_dz,
                    "descend_dz": args_cli.descend_dz,
                    "lift_dz": args_cli.lift_dz,
                },
            },
            args_cli.output,
        )
        print(f"[INFO] saved {successes} demos to {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
