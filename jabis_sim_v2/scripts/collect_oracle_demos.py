"""Oracle demo collection for BC pretrain.

Cube spawn fixed at sweet spot (0.16, 0.00). Oracle 으로 1200 step 실행.
Success episode (cube in trash bin) 만 저장.
"""

import argparse
import os
import sys

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256, help="parallel envs per trial")
parser.add_argument("--n_trials", type=int, default=2, help="trials (if not enough success)")
parser.add_argument("--target_demos", type=int, default=200)
parser.add_argument("--cube_x", type=float, default=0.16)
parser.add_argument("--cube_y", type=float, default=0.0)
parser.add_argument("--goal_x", type=float, default=0.03)
parser.add_argument("--goal_y", type=float, default=0.245)
parser.add_argument("--episode_steps", type=int, default=1200)
parser.add_argument("--lift_dz", type=float, default=0.15)
parser.add_argument("--out_dir", type=str, default="demos/oracle_pnp")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"] + sys.argv[1:])

launcher = AppLauncher(args)

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from jabis_sim_v2.oracle.oracle_policy import (  # noqa: E402
    DONE, OracleCfg, OraclePolicy,
)
from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import (  # noqa: E402
    SoArm101PickPlaceEnvCfg,
)
from jabis_sim_v2.tasks.cube_lift.mdp import events as cube_events  # noqa: E402


BIN_X_MIN, BIN_X_MAX = -0.045, 0.105
BIN_Y_MIN, BIN_Y_MAX = 0.18, 0.31


def main():
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = SoArm101PickPlaceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = 100.0  # avoid auto-reset
    cfg.events.reset_cube = EventTerm(
        func=cube_events.reset_cube_grid,
        mode="reset",
        params={
            "x_vals": [args.cube_x],
            "y_vals": [args.cube_y],
            "envs_per_point": args.num_envs,
        },
    )

    env_raw = ManagerBasedRLEnv(cfg=cfg)
    env = RslRlVecEnvWrapper(env_raw, clip_actions=1.0)
    device = env.unwrapped.device

    oracle = OraclePolicy(
        env_raw,
        OracleCfg(
            lift_dz=args.lift_dz,
            ik_damping=0.02,
            ik_max_dq=1.0,
            ik_nullspace_gain=0.5,
        ),
    )
    goal_per_env = (
        torch.tensor([[args.goal_x, args.goal_y]], device=device)
        .expand(args.num_envs, -1).float().contiguous()
    )
    oracle.set_goal_xy(goal_per_env)

    env_origins = env_raw.scene.env_origins
    cube = env_raw.scene["cube"]

    print(
        f"[Demo] num_envs={args.num_envs}, target={args.target_demos}, "
        f"cube=({args.cube_x:.3f},{args.cube_y:+.3f}), goal=({args.goal_x:.3f},{args.goal_y:+.3f})",
        flush=True,
    )

    all_demos_obs = []   # list of (T, obs_dim)
    all_demos_act = []   # list of (T, act_dim)
    n_saved = 0

    for trial in range(args.n_trials):
        env_raw.reset()
        oracle.reset()
        obs, _ = env.get_observations()

        traj_obs = []  # list of (num_envs, obs_dim) per step
        traj_act = []

        for step in range(args.episode_steps):
            action = oracle.compute_action()
            traj_obs.append(obs.detach().cpu().numpy())
            traj_act.append(action.detach().cpu().numpy())
            obs, _, _, _ = env.step(action)
            if step % 200 == 0:
                n_done = int((oracle.states == DONE).sum())
                print(
                    f"  trial {trial} step {step}/{args.episode_steps}  DONE={n_done}/{args.num_envs}",
                    flush=True,
                )

        # final cube position
        cl_final = cube.data.root_pos_w - env_origins
        fx, fy, fz = cl_final[:, 0], cl_final[:, 1], cl_final[:, 2]
        in_bin_xy = (fx >= BIN_X_MIN) & (fx <= BIN_X_MAX) & (fy >= BIN_Y_MIN) & (fy <= BIN_Y_MAX)
        in_bin = in_bin_xy & (fz < 0)

        success_mask = in_bin.cpu().numpy()
        n_trial_success = int(success_mask.sum())
        print(
            f"  trial {trial}: {n_trial_success}/{args.num_envs} success",
            flush=True,
        )

        # stack trajectories: (num_envs, T, dim)
        traj_obs_arr = np.stack(traj_obs, axis=1)  # (num_envs, T, obs_dim)
        traj_act_arr = np.stack(traj_act, axis=1)  # (num_envs, T, act_dim)

        # filter success only
        success_idx = np.where(success_mask)[0]
        for idx in success_idx:
            if n_saved >= args.target_demos:
                break
            np.savez(
                os.path.join(args.out_dir, f"demo_{n_saved:04d}.npz"),
                obs=traj_obs_arr[idx],   # (T, obs_dim)
                action=traj_act_arr[idx],  # (T, act_dim)
                cube_x=args.cube_x,
                cube_y=args.cube_y,
                goal_x=args.goal_x,
                goal_y=args.goal_y,
            )
            n_saved += 1

        print(f"  total saved: {n_saved}/{args.target_demos}", flush=True)
        if n_saved >= args.target_demos:
            break

    print(f"\n[Demo] saved {n_saved} demos to {args.out_dir}/", flush=True)
    print(f"[Demo] obs dim: {traj_obs_arr.shape[-1]}, action dim: {traj_act_arr.shape[-1]}", flush=True)
    print(f"[Demo] episode length: {traj_obs_arr.shape[1]} steps", flush=True)

    env.close()


main()
launcher.app.close()
