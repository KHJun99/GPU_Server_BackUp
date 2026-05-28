"""Transport map: cube를 sweet spot에 fixed spawn, Oracle goal 위치를 grid로 변동.

목적: cube를 잡은 후 어디까지 옮길 수 있는지 (transport feasibility) 측정.
RL의 PnP target (trash bin) 이 reach 안에 있는지 검증.
"""

import argparse
import json
import sys

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--grid_x_n", type=int, default=7)
parser.add_argument("--grid_y_n", type=int, default=7)
parser.add_argument("--envs_per_point", type=int, default=8)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--goal_x_min", type=float, default=-0.10)
parser.add_argument("--goal_x_max", type=float, default=0.20)
parser.add_argument("--goal_y_min", type=float, default=-0.20)
parser.add_argument("--goal_y_max", type=float, default=0.30)
parser.add_argument("--cube_x", type=float, default=0.045, help="fixed cube spawn x")
parser.add_argument("--cube_y", type=float, default=-0.20, help="fixed cube spawn y")
parser.add_argument("--lift_dz", type=float, default=0.15)
parser.add_argument("--success_z", type=float, default=0.10)
parser.add_argument("--reach_tol", type=float, default=0.05, help="goal 도달 거리 임계")
parser.add_argument("--ik_damping", type=float, default=0.05)
parser.add_argument("--ik_max_dq", type=float, default=0.30)
parser.add_argument("--ik_nullspace_gain", type=float, default=0.2)
parser.add_argument("--out", type=str, default="/tmp/transport_map.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"] + sys.argv[1:])

launcher = AppLauncher(args)

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402

from jabis_sim_v2.oracle.oracle_policy import OracleCfg, OraclePolicy  # noqa: E402
from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import (  # noqa: E402
    SoArm101PickPlaceEnvCfg,
)
from jabis_sim_v2.tasks.cube_lift.mdp import events as cube_events  # noqa: E402


def main():
    # goal grid
    gx_vals = np.linspace(args.goal_x_min, args.goal_x_max, args.grid_x_n)
    gy_vals = np.linspace(args.goal_y_min, args.goal_y_max, args.grid_y_n)
    goal_points = np.array([(x, y) for x in gx_vals for y in gy_vals])
    n_pts = len(goal_points)
    total_envs = n_pts * args.envs_per_point

    print(
        f"[Transport] cube fixed at ({args.cube_x:.3f}, {args.cube_y:.3f}), "
        f"goal grid {args.grid_x_n}x{args.grid_y_n} = {n_pts} pts, "
        f"{args.envs_per_point} envs/pt, total {total_envs}, steps={args.steps}",
        flush=True,
    )
    print(
        f"[Transport] goal x ∈ [{args.goal_x_min:.3f}, {args.goal_x_max:.3f}], "
        f"y ∈ [{args.goal_y_min:.3f}, {args.goal_y_max:.3f}]",
        flush=True,
    )

    cfg = SoArm101PickPlaceEnvCfg()
    cfg.scene.num_envs = total_envs
    cfg.episode_length_s = 100.0
    # cube spawn fixed (grid 함수에 x_vals=[cube_x], y_vals=[cube_y])
    cfg.events.reset_cube = EventTerm(
        func=cube_events.reset_cube_grid,
        mode="reset",
        params={
            "x_vals": [args.cube_x],
            "y_vals": [args.cube_y],
            "envs_per_point": total_envs,  # 모든 env 같은 cube 위치
        },
    )

    env = ManagerBasedRLEnv(cfg=cfg)
    device = env.device

    oracle = OraclePolicy(
        env,
        OracleCfg(
            lift_dz=args.lift_dz,
            success_z=args.success_z,
            ik_damping=args.ik_damping,
            ik_max_dq=args.ik_max_dq,
            ik_nullspace_gain=args.ik_nullspace_gain,
        ),
    )

    # env_id → goal point mapping
    env_to_goal = np.repeat(np.arange(n_pts), args.envs_per_point)
    env_to_goal_t = torch.from_numpy(env_to_goal).long().to(device)
    goal_xy_t = torch.from_numpy(goal_points).float().to(device)  # (n_pts, 2)
    goal_per_env = goal_xy_t[env_to_goal_t]  # (total_envs, 2)
    oracle.set_goal_xy(goal_per_env)

    env_origins = env.scene.env_origins
    cube = env.scene["cube"]

    env.reset()
    oracle.reset()

    cl0 = cube.data.root_pos_w - env_origins
    print(
        f"[Transport] cube xy after reset (first 3 envs): "
        f"{cl0[:3, :2].cpu().tolist()}",
        flush=True,
    )

    max_state = torch.zeros(total_envs, dtype=torch.long, device=device)
    max_z = torch.zeros(total_envs, device=device)

    for step in range(args.steps):
        action = oracle.compute_action()
        env.step(action)
        max_state = torch.maximum(max_state, oracle.states)
        cl = cube.data.root_pos_w - env_origins
        max_z = torch.maximum(max_z, cl[:, 2])
        if step % 50 == 0:
            print(f"  step {step}/{args.steps}", flush=True)

    # final cube xy
    final_cube_xy = (cube.data.root_pos_w - env_origins)[:, :2]
    # distance from final cube to goal
    goal_dist = torch.norm(final_cube_xy - goal_per_env, dim=-1)

    print("\n[Transport] per-goal results", flush=True)
    print(
        f"{'gx':>7} {'gy':>7} | LIFT  GOAL  >12cm | dist_med  reached  hover_z",
        flush=True,
    )
    print("-" * 80, flush=True)

    per_point = []
    for i, (gx, gy) in enumerate(goal_points):
        mask = env_to_goal_t == i
        lift_pct = (max_state[mask] >= 3).float().mean().item() * 100
        goal_pct = (max_state[mask] >= 4).float().mean().item() * 100
        cm12_pct = (max_z[mask] >= 0.12).float().mean().item() * 100
        dist_med = goal_dist[mask].median().item()
        reached_pct = (goal_dist[mask] < args.reach_tol).float().mean().item() * 100
        hover_z_med = max_z[mask].median().item()
        per_point.append(
            {
                "goal_x": float(gx),
                "goal_y": float(gy),
                "lift_pct": float(lift_pct),
                "goal_state_pct": float(goal_pct),
                "cm12_pct": float(cm12_pct),
                "dist_median": float(dist_med),
                "reached_pct": float(reached_pct),
                "hover_z_median": float(hover_z_med),
            }
        )
        print(
            f"{gx:7.3f} {gy:+7.3f} | {lift_pct:4.0f}%  {goal_pct:4.0f}%  {cm12_pct:4.0f}% | "
            f"{dist_med:.3f}m   {reached_pct:4.0f}%   {hover_z_med:.3f}",
            flush=True,
        )

    result = {
        "cube_x": args.cube_x,
        "cube_y": args.cube_y,
        "goal_grid_x": gx_vals.tolist(),
        "goal_grid_y": gy_vals.tolist(),
        "envs_per_point": args.envs_per_point,
        "steps": args.steps,
        "reach_tol": args.reach_tol,
        "per_point": per_point,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Transport] saved -> {args.out}", flush=True)

    env.close()


main()
launcher.app.close()
