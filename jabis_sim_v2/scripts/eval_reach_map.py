"""SO-ARM101 reach map — cube을 grid에 두고 Oracle reach 성공률 측정.

각 grid point에 envs_per_point env 할당, cube 위치 강제 설정.
Oracle을 steps 만큼 돌려서 각 env가 어디까지 state 진입했는지 측정.
출력:
  - per-point reach 성공률 (DESCEND/CLOSE/LIFT 도달 %)
  - per-point >12cm lift 성공률
  - JSON 저장 (heatmap 그리기용)
"""

import argparse
import json
import sys

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--grid_n", type=int, default=7)
parser.add_argument("--grid_x_n", type=int, default=None, help="override grid_n for x axis")
parser.add_argument("--grid_y_n", type=int, default=None, help="override grid_n for y axis")
parser.add_argument("--envs_per_point", type=int, default=8)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--x_min", type=float, default=0.15)
parser.add_argument("--x_max", type=float, default=0.39)
parser.add_argument("--y_min", type=float, default=-0.12)
parser.add_argument("--y_max", type=float, default=0.12)
parser.add_argument("--lift_dz", type=float, default=0.15)
parser.add_argument("--out", type=str, default="/tmp/reach_map.json")
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
    x_n = args.grid_x_n if args.grid_x_n is not None else args.grid_n
    y_n = args.grid_y_n if args.grid_y_n is not None else args.grid_n
    x_vals = np.linspace(args.x_min, args.x_max, x_n)
    y_vals = np.linspace(args.y_min, args.y_max, y_n)
    grid_points = np.array([(x, y) for x in x_vals for y in y_vals])  # (N, 2)
    n_points = len(grid_points)
    total_envs = n_points * args.envs_per_point

    print(
        f"[ReachMap] grid {x_n}x{y_n} = {n_points} pts, "
        f"{args.envs_per_point} envs/pt, total {total_envs} envs, steps={args.steps}",
        flush=True,
    )
    print(
        f"[ReachMap] x range [{args.x_min:.3f}, {args.x_max:.3f}] step {(args.x_max - args.x_min) / max(x_n - 1, 1):.3f}",
        flush=True,
    )
    print(
        f"[ReachMap] y range [{args.y_min:.3f}, {args.y_max:.3f}] step {(args.y_max - args.y_min) / max(y_n - 1, 1):.3f}",
        flush=True,
    )

    cfg = SoArm101PickPlaceEnvCfg()
    cfg.scene.num_envs = total_envs
    cfg.episode_length_s = 100.0  # avoid auto-reset during sweep
    # replace random reset event with deterministic grid-spawn
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

    oracle = OraclePolicy(env, OracleCfg(lift_dz=args.lift_dz))

    env_origins = env.scene.env_origins
    cube = env.scene["cube"]

    # env_id → grid point mapping (matches reset_cube_grid logic)
    env_to_grid = np.repeat(np.arange(n_points), args.envs_per_point)
    env_to_grid_t = torch.from_numpy(env_to_grid).long().to(device)

    env.reset()
    oracle.reset()

    # debug: verify cube placement
    cl = cube.data.root_pos_w - env_origins
    print(f"[ReachMap] sample cube xy after reset (first 5 envs):", flush=True)
    for i in range(min(5, total_envs)):
        gi = env_to_grid[i]
        print(
            f"  env {i}: grid_pt {gi} = ({grid_points[gi][0]:.3f}, {grid_points[gi][1]:+.3f})  "
            f"actual cube_local = ({cl[i, 0].item():.3f}, {cl[i, 1].item():+.3f}, {cl[i, 2].item():.3f})",
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

    print("\n[ReachMap] per-point results", flush=True)
    print(f"{'x':>7} {'y':>7} | DESC+  CLOSE+  LIFT+  >12cm | max_z med", flush=True)
    print("-" * 70, flush=True)

    per_point = []
    for i, (x, y) in enumerate(grid_points):
        mask = env_to_grid_t == i
        desc_pct = (max_state[mask] >= 1).float().mean().item() * 100
        close_pct = (max_state[mask] >= 2).float().mean().item() * 100
        lift_pct = (max_state[mask] >= 3).float().mean().item() * 100
        cm12_pct = (max_z[mask] >= 0.12).float().mean().item() * 100
        mz_med = max_z[mask].median().item()
        per_point.append(
            {
                "x": float(x),
                "y": float(y),
                "descend_pct": float(desc_pct),
                "close_pct": float(close_pct),
                "lift_pct": float(lift_pct),
                "cm12_pct": float(cm12_pct),
                "max_z_median": float(mz_med),
            }
        )
        print(
            f"{x:7.3f} {y:+7.3f} | {desc_pct:4.0f}%  {close_pct:5.0f}%  {lift_pct:4.0f}%  {cm12_pct:5.0f}% | {mz_med:.3f}",
            flush=True,
        )

    result = {
        "grid_x": x_vals.tolist(),
        "grid_y": y_vals.tolist(),
        "envs_per_point": args.envs_per_point,
        "steps": args.steps,
        "lift_dz": args.lift_dz,
        "per_point": per_point,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[ReachMap] saved -> {args.out}", flush=True)

    env.close()


main()
launcher.app.close()
