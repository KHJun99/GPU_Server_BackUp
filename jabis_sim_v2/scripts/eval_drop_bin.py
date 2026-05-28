"""Drop bin physics test: cube를 sweet spot에 spawn, Oracle 로 trash bin 위까지
transport 후 gripper open. cube 가 trash bin 안에 떨어지는지 측정.

목적: sim 환경의 PnP task 물리적 가능성 검증.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--cube_x", type=float, default=0.03, help="cube spawn (sweet spot)")
parser.add_argument("--cube_y", type=float, default=-0.15)
parser.add_argument("--goal_x", type=float, default=0.03, help="trash bin center xy")
parser.add_argument("--goal_y", type=float, default=0.245)
parser.add_argument("--transport_steps", type=int, default=300)
parser.add_argument("--drop_steps", type=int, default=200)
parser.add_argument("--lift_dz", type=float, default=0.15)
parser.add_argument("--ik_damping", type=float, default=0.05)
parser.add_argument("--ik_max_dq", type=float, default=0.30)
parser.add_argument("--ik_nullspace_gain", type=float, default=0.2)
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


# trash bin volume (env_cfg.py 기준)
BIN_X_MIN, BIN_X_MAX = -0.045, 0.105
BIN_Y_MIN, BIN_Y_MAX = 0.18, 0.31
BIN_Z_TOP = 0.0  # 입구 z
BIN_Z_BOTTOM = -0.13  # 바닥 z


def main():
    cfg = SoArm101PickPlaceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = 100.0
    cfg.events.reset_cube = EventTerm(
        func=cube_events.reset_cube_grid,
        mode="reset",
        params={
            "x_vals": [args.cube_x],
            "y_vals": [args.cube_y],
            "envs_per_point": args.num_envs,
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
        .expand(args.num_envs, -1)
        .float()
        .contiguous()
    )
    oracle.set_goal_xy(goal_per_env)

    env_origins = env.scene.env_origins
    cube = env.scene["cube"]
    robot = env.scene["robot"]
    ee_idx = oracle.ee_idx

    env.reset()
    oracle.reset()

    print(
        f"[DropBin] cube ({args.cube_x:.3f}, {args.cube_y:+.3f}) → "
        f"bin goal ({args.goal_x:.3f}, {args.goal_y:+.3f})  "
        f"num_envs={args.num_envs}  transport={args.transport_steps}  drop={args.drop_steps}",
        flush=True,
    )

    # transport phase (Oracle drives ee → goal_xy at lift_z)
    arm_ids_t = oracle.arm_ids_t
    print(f"  rotate_pan_target = {oracle._rotate_pan_target[0].item():.3f} rad", flush=True)
    for step in range(args.transport_steps):
        action = oracle.compute_action()
        env.step(action)
        if step % 50 == 0:
            cl = cube.data.root_pos_w - env_origins
            ee = robot.data.body_pos_w[:, ee_idx] - env_origins
            jpos = robot.data.joint_pos[:, arm_ids_t]
            pan = jpos[:, 0].median().item()
            slift = jpos[:, 1].median().item()
            eflex = jpos[:, 2].median().item()
            print(
                f"  step {step}/{args.transport_steps}  "
                f"cube=({cl[:, 0].median():.3f},{cl[:, 1].median():+.3f},{cl[:, 2].median():.3f}) "
                f"ee=({ee[:, 0].median():.3f},{ee[:, 1].median():+.3f},{ee[:, 2].median():.3f}) "
                f"pan/lift/elb=({pan:.3f},{slift:.3f},{eflex:.3f}) "
                f"st={int(oracle.states.float().median().item())}",
                flush=True,
            )

    cl = cube.data.root_pos_w - env_origins
    print(f"\n[DropBin] === After transport ===", flush=True)
    print(
        f"  cube xy median: ({cl[:, 0].median():.3f}, {cl[:, 1].median():+.3f})",
        flush=True,
    )
    print(f"  cube z median: {cl[:, 2].median():.3f}m", flush=True)
    print(
        f"  cube z range: [{cl[:, 2].min():.3f}, {cl[:, 2].max():.3f}]m",
        flush=True,
    )
    print(
        f"  oracle states (median): {oracle.states.float().median().item():.0f} "
        f"(0=APPROACH, 1=DESCEND, 2=CLOSE, 3=LIFT, 4=MOVE_TO_GOAL)",
        flush=True,
    )

    # drop phase — multi-waypoint Oracle 이 RELEASE state 에서 gripper 자동 open
    for step in range(args.drop_steps):
        action = oracle.compute_action()
        env.step(action)
        if step % 50 == 0:
            cl = cube.data.root_pos_w - env_origins
            from jabis_sim_v2.oracle.oracle_policy import (
                ROTATE_PAN, TRANSPORT, LOWER, RELEASE, DONE,
            )
            n_rot = int((oracle.states == ROTATE_PAN).sum())
            n_trans = int((oracle.states == TRANSPORT).sum())
            n_lower = int((oracle.states == LOWER).sum())
            n_release = int((oracle.states == RELEASE).sum())
            n_done = int((oracle.states == DONE).sum())
            print(
                f"  drop step {step}/{args.drop_steps}  "
                f"cube xyz=({cl[:, 0].median():.3f}, {cl[:, 1].median():+.3f}, {cl[:, 2].median():.3f})  "
                f"RT/TR/LO/RE/DO={n_rot}/{n_trans}/{n_lower}/{n_release}/{n_done}",
                flush=True,
            )

    cl_final = cube.data.root_pos_w - env_origins
    fx, fy, fz = cl_final[:, 0], cl_final[:, 1], cl_final[:, 2]

    # ee final position (at start of drop = transport end)
    ee_final = robot.data.body_pos_w[:, ee_idx] - env_origins
    eex, eey, eez = ee_final[:, 0], ee_final[:, 1], ee_final[:, 2]
    ee_in_bin_xy = (eex >= BIN_X_MIN) & (eex <= BIN_X_MAX) & (eey >= BIN_Y_MIN) & (eey <= BIN_Y_MAX)
    print(f"\n[DropBin] === EE final position ===", flush=True)
    print(f"  ee xyz median: ({eex.median():.3f}, {eey.median():+.3f}, {eez.median():.3f})", flush=True)
    print(f"  ee x range: [{eex.min():.3f}, {eex.max():.3f}]", flush=True)
    print(f"  ee y range: [{eey.min():.3f}, {eey.max():.3f}]", flush=True)
    print(f"  ee_in_bin_xy: {int(ee_in_bin_xy.sum())}/{args.num_envs} ({100*ee_in_bin_xy.float().mean():.1f}%)", flush=True)

    # bin classification
    in_bin_xy = (fx >= BIN_X_MIN) & (fx <= BIN_X_MAX) & (fy >= BIN_Y_MIN) & (fy <= BIN_Y_MAX)
    on_table = (fz > -0.01) & (fz < 0.10)  # 책상 표면 (z=-0.001) 근처
    in_bin_z = (fz > BIN_Z_BOTTOM - 0.01) & (fz < BIN_Z_TOP)  # bin 내부 (z<0)
    in_bin = in_bin_xy & in_bin_z
    on_floor = fz < -0.30  # 책상 아래로 떨어짐

    print(f"\n[DropBin] === Final cube position ({args.drop_steps} drop steps) ===", flush=True)
    print(
        f"  xyz median: ({fx.median():.3f}, {fy.median():+.3f}, {fz.median():.3f})",
        flush=True,
    )
    print(f"  x range: [{fx.min():.3f}, {fx.max():.3f}]", flush=True)
    print(f"  y range: [{fy.min():.3f}, {fy.max():.3f}]", flush=True)
    print(f"  z range: [{fz.min():.3f}, {fz.max():.3f}]", flush=True)
    print(f"", flush=True)
    print(
        f"  in_bin_xy:  {int(in_bin_xy.sum())}/{args.num_envs} "
        f"({100 * in_bin_xy.float().mean():.1f}%)",
        flush=True,
    )
    print(
        f"  in_bin (xy+z<0): {int(in_bin.sum())}/{args.num_envs} "
        f"({100 * in_bin.float().mean():.1f}%)",
        flush=True,
    )
    print(
        f"  on_table (z>~0): {int(on_table.sum())}/{args.num_envs} "
        f"({100 * on_table.float().mean():.1f}%)",
        flush=True,
    )
    print(
        f"  on_floor (z<-0.3): {int(on_floor.sum())}/{args.num_envs} "
        f"({100 * on_floor.float().mean():.1f}%)",
        flush=True,
    )

    env.close()


main()
launcher.app.close()
