"""PnP success rate across cube spawn grid (K1) + bin variability (K3).

Cube spawn 을 grid 로 두고 fixed trash bin 으로 transport+drop 시도.
각 spawn 위치에서 trash bin 안착 비율 측정.
"""

import argparse
import json
import sys

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--cube_x_n", type=int, default=4)
parser.add_argument("--cube_y_n", type=int, default=4)
parser.add_argument("--cube_x_min", type=float, default=0.025)
parser.add_argument("--cube_x_max", type=float, default=0.065)
parser.add_argument("--cube_y_min", type=float, default=-0.26)
parser.add_argument("--cube_y_max", type=float, default=-0.14)
parser.add_argument("--envs_per_point", type=int, default=8)
parser.add_argument("--goal_x", type=float, default=0.03)
parser.add_argument("--goal_y", type=float, default=0.245)
parser.add_argument("--transport_steps", type=int, default=1000)
parser.add_argument("--drop_steps", type=int, default=200)
parser.add_argument("--lift_dz", type=float, default=0.20)
parser.add_argument("--ik_damping", type=float, default=0.02)
parser.add_argument("--ik_max_dq", type=float, default=1.0)
parser.add_argument("--ik_nullspace_gain", type=float, default=0.5)
parser.add_argument("--out", type=str, default="/tmp/pnp_success.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"] + sys.argv[1:])

launcher = AppLauncher(args)

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402

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
    x_vals = np.linspace(args.cube_x_min, args.cube_x_max, args.cube_x_n)
    y_vals = np.linspace(args.cube_y_min, args.cube_y_max, args.cube_y_n)
    spawn_points = np.array([(x, y) for x in x_vals for y in y_vals])
    n_pts = len(spawn_points)
    total_envs = n_pts * args.envs_per_point

    cfg = SoArm101PickPlaceEnvCfg()
    cfg.scene.num_envs = total_envs
    cfg.episode_length_s = 100.0
    cfg.events.reset_cube = EventTerm(
        func=cube_events.reset_cube_grid,
        mode="reset",
        params={
            "x_vals": x_vals.tolist(),
            "y_vals": y_vals.tolist(),
            "envs_per_point": args.envs_per_point,
        },
    )

    env = ManagerBasedRLEnv(cfg=cfg)
    device = env.device

    oracle = OraclePolicy(
        env,
        OracleCfg(
            lift_dz=args.lift_dz,
            ik_damping=args.ik_damping,
            ik_max_dq=args.ik_max_dq,
            ik_nullspace_gain=args.ik_nullspace_gain,
        ),
    )
    goal_per_env = (
        torch.tensor([[args.goal_x, args.goal_y]], device=device)
        .expand(total_envs, -1).float().contiguous()
    )
    oracle.set_goal_xy(goal_per_env)

    env_to_spawn = np.repeat(np.arange(n_pts), args.envs_per_point)
    env_to_spawn_t = torch.from_numpy(env_to_spawn).long().to(device)

    env_origins = env.scene.env_origins
    cube = env.scene["cube"]

    env.reset()
    oracle.reset()

    print(
        f"[PnP] cube spawn {args.cube_x_n}x{args.cube_y_n}={n_pts} pts, "
        f"{args.envs_per_point} envs/pt, total {total_envs}",
        flush=True,
    )
    print(
        f"[PnP] cube x ∈ [{args.cube_x_min:.3f}, {args.cube_x_max:.3f}], "
        f"y ∈ [{args.cube_y_min:.3f}, {args.cube_y_max:.3f}]",
        flush=True,
    )
    print(f"[PnP] bin goal ({args.goal_x:.3f}, {args.goal_y:+.3f})", flush=True)

    total_steps = args.transport_steps + args.drop_steps
    for step in range(total_steps):
        action = oracle.compute_action()
        env.step(action)
        if step % 200 == 0:
            n_done = int((oracle.states == DONE).sum())
            print(
                f"  step {step}/{total_steps}  DONE={n_done}/{total_envs}",
                flush=True,
            )

    cl_final = cube.data.root_pos_w - env_origins
    fx, fy, fz = cl_final[:, 0], cl_final[:, 1], cl_final[:, 2]

    in_bin_xy = (fx >= BIN_X_MIN) & (fx <= BIN_X_MAX) & (fy >= BIN_Y_MIN) & (fy <= BIN_Y_MAX)
    in_bin = in_bin_xy & (fz < 0)

    print(f"\n[PnP] === Overall ===", flush=True)
    print(
        f"  in_bin (xy+z<0): {int(in_bin.sum())}/{total_envs} "
        f"({100 * in_bin.float().mean():.1f}%)",
        flush=True,
    )
    print(
        f"  in_bin_xy only:  {int(in_bin_xy.sum())}/{total_envs} "
        f"({100 * in_bin_xy.float().mean():.1f}%)",
        flush=True,
    )

    print(f"\n[PnP] === Per-spawn success ===", flush=True)
    print(f"{'cube_x':>8} {'cube_y':>8} | {'in_bin':>8} | {'in_bin_xy':>10}", flush=True)
    print("-" * 50, flush=True)
    per_point = []
    for i, (sx, sy) in enumerate(spawn_points):
        mask = env_to_spawn_t == i
        n_envs = int(mask.sum())
        n_in = int(in_bin[mask].sum())
        n_xy = int(in_bin_xy[mask].sum())
        rate_in = 100 * n_in / n_envs
        rate_xy = 100 * n_xy / n_envs
        per_point.append(
            {
                "x": float(sx),
                "y": float(sy),
                "in_bin_pct": float(rate_in),
                "in_bin_xy_pct": float(rate_xy),
                "n_envs": n_envs,
            }
        )
        print(
            f"{sx:8.3f} {sy:+8.3f} | {rate_in:5.1f}% | {rate_xy:6.1f}%",
            flush=True,
        )

    with open(args.out, "w") as f:
        json.dump(
            {
                "cube_grid_x": x_vals.tolist(),
                "cube_grid_y": y_vals.tolist(),
                "goal": [args.goal_x, args.goal_y],
                "per_point": per_point,
            },
            f,
            indent=2,
        )
    print(f"\n[PnP] saved -> {args.out}", flush=True)

    env.close()


main()
launcher.app.close()
