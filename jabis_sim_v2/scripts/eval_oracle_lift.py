"""Oracle policy 환경 검증 — sim에서 진짜 N cm lift 가능한지 측정.

목적: RL 재학습 전에 SoArm101 + cube + PnP env 의 물리적 lift 한계 확인.
Oracle은 5-state machine (APPROACH→DESCEND→CLOSE→LIFT→MOVE_TO_GOAL).
lift_dz 를 키워서 cube z 가 실제로 어디까지 올라가는지 추적.
"""

import argparse
import statistics
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--n_trials", type=int, default=3)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--lift_dz", type=float, default=0.12)
parser.add_argument("--success_z", type=float, default=0.10)
parser.add_argument("--close_steps", type=int, default=60)
parser.add_argument("--finger_x_offset", type=float, default=0.006)
parser.add_argument("--tag", type=str, default="baseline")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"] + sys.argv[1:])

launcher = AppLauncher(args)

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from jabis_sim_v2.oracle.oracle_policy import OracleCfg, OraclePolicy  # noqa: E402
from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import (  # noqa: E402
    SoArm101PickPlaceEnvCfg,
)


STATE_NAMES = ["APPROACH", "DESCEND", "CLOSE", "LIFT", "MOVE_TO_GOAL"]
THRESHOLDS = [0.05, 0.07, 0.10, 0.12]


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = min(int(len(sorted_vals) * p), len(sorted_vals) - 1)
    return sorted_vals[idx]


def main():
    cfg = SoArm101PickPlaceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed

    env = ManagerBasedRLEnv(cfg=cfg)
    device = env.device

    oracle_cfg = OracleCfg(
        lift_dz=args.lift_dz,
        success_z=args.success_z,
        close_steps=args.close_steps,
        finger_x_offset=args.finger_x_offset,
    )
    oracle = OraclePolicy(env, oracle_cfg)

    env_origins = env.scene.env_origins
    cube = env.scene["cube"]

    print(
        f"[Oracle eval] tag={args.tag}  lift_dz={args.lift_dz:.3f}  success_z={args.success_z:.3f}  "
        f"close_steps={args.close_steps}  num_envs={args.num_envs}  n_trials={args.n_trials}  steps={args.steps}",
        flush=True,
    )

    all_max_z = []
    all_final_xy_x = []
    all_final_xy_y = []
    state_dist_accum = [0] * 5
    threshold_hits = {t: 0 for t in THRESHOLDS}
    total_n = 0
    lift_reached_total = 0
    goal_reached_total = 0

    for trial in range(args.n_trials):
        env.reset()
        oracle.reset()
        max_z = torch.zeros(args.num_envs, device=device)
        final_xy = torch.zeros(args.num_envs, 2, device=device)

        for step in range(args.steps):
            action = oracle.compute_action()
            env.step(action)
            cl = cube.data.root_pos_w - env_origins
            max_z = torch.maximum(max_z, cl[:, 2])
            final_xy = cl[:, :2]

        # state distribution at end of trial
        for s in range(5):
            state_dist_accum[s] += int((oracle.states == s).sum())
        lift_reached_trial = int((oracle.states >= 3).sum())
        goal_reached_trial = int((oracle.states >= 4).sum())
        lift_reached_total += lift_reached_trial
        goal_reached_total += goal_reached_trial

        for thr in THRESHOLDS:
            threshold_hits[thr] += int((max_z > thr).sum())

        mz_list = max_z.cpu().tolist()
        all_max_z.extend(mz_list)
        all_final_xy_x.extend(final_xy[:, 0].cpu().tolist())
        all_final_xy_y.extend(final_xy[:, 1].cpu().tolist())
        total_n += args.num_envs

        print(
            f"trial {trial}: max_z med={statistics.median(mz_list):.3f}  "
            f">5cm={int((max_z > 0.05).sum())}/{args.num_envs}  "
            f">10cm={int((max_z > 0.10).sum())}/{args.num_envs}  "
            f">12cm={int((max_z > 0.12).sum())}/{args.num_envs}  "
            f"LIFT+={lift_reached_trial}/{args.num_envs}  "
            f"GOAL={goal_reached_trial}/{args.num_envs}",
            flush=True,
        )

    sorted_z = sorted(all_max_z)
    print("", flush=True)
    print(f"[Oracle eval] === SUMMARY (lift_dz={args.lift_dz:.3f}) ===", flush=True)
    print(f"[Oracle eval] max_z median: {statistics.median(all_max_z):.3f} m", flush=True)
    print(f"[Oracle eval] max_z mean:   {statistics.mean(all_max_z):.3f} m", flush=True)
    print(f"[Oracle eval] max_z p90:    {percentile(sorted_z, 0.9):.3f} m", flush=True)
    for thr in THRESHOLDS:
        n = threshold_hits[thr]
        print(
            f"[Oracle eval] >{int(thr * 100):2d}cm: {n}/{total_n} ({100 * n / total_n:.1f}%)",
            flush=True,
        )
    print("[Oracle eval] state dist (last step):", flush=True)
    for s, name in enumerate(STATE_NAMES):
        n = state_dist_accum[s]
        print(f"  {name:12s}: {n}/{total_n} ({100 * n / total_n:.1f}%)", flush=True)
    print(
        f"[Oracle eval] LIFT reached:         {lift_reached_total}/{total_n} "
        f"({100 * lift_reached_total / total_n:.1f}%)",
        flush=True,
    )
    print(
        f"[Oracle eval] MOVE_TO_GOAL reached: {goal_reached_total}/{total_n} "
        f"({100 * goal_reached_total / total_n:.1f}%)",
        flush=True,
    )
    fx_med = statistics.median(all_final_xy_x)
    fy_med = statistics.median(all_final_xy_y)
    print(
        f"[Oracle eval] final cube xy median: ({fx_med:.3f}, {fy_med:.3f})  "
        f"(target=(0.80, 0.0))",
        flush=True,
    )

    env.close()


main()
launcher.app.close()
