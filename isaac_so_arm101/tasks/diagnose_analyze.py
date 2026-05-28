"""Analyze diagnose_joints CSV and print A.3 / A.4 / B.1 tables.

Pure pandas/numpy. No Isaac dependency.

Tables produced
---------------
A.3 — Joint Activation Table
    For each of the 10 joints: min_pos, max_pos, total_delta (sum of |dpos|),
    first_movement_step (smallest step where |jpos - jpos_step0| > 1e-3 rad).
    Joints with total_delta < 1e-3 rad are flagged DEAD.

A.4 — CLOSE finger response
    First 5 steps where state == CLOSE OR force_close == 1.
    Columns: step, left_proximal, left_distal, right_proximal, right_distal,
    gripper, action_5 (gripper command).

B.1 — Arm action / EE delta statistics (APPROACH state, env step granular,
    decimation already collapsed inside env.step()).
    For each of action[0..4] (5 arm channels):
      - |raw_action| stats (mean/min/max/std)
      - |target_delta| (target_ee - current_ee) stats
      - |actual_delta| (current_ee_t+1 - current_ee_t) stats
      - actual / target ratio stats
"""

import argparse
import sys

import numpy as np
import pandas as pd


def fmt(x: float, w: int = 10) -> str:
    if abs(x) < 1e-6:
        return f"{0.0:>{w}.6f}"
    if abs(x) >= 100 or abs(x) < 0.001:
        return f"{x:>{w}.3e}"
    return f"{x:>{w}.6f}"


def joint_activation_table(df: pd.DataFrame) -> None:
    print("\n[A.3 Joint Activation Table]")
    print(f"{'joint_name':<18} | {'min_pos':>10} | {'max_pos':>10} | {'total_delta':>12} | "
          f"{'first_move':>10} | DEAD?")
    print("-" * 80)

    jpos_cols = [c for c in df.columns if c.startswith("jpos_")]
    for col in jpos_cols:
        name = col[len("jpos_"):]
        series = df[col].to_numpy()
        first_val = series[0]
        diffs = np.abs(np.diff(series))
        total_delta = float(diffs.sum())
        min_p = float(series.min())
        max_p = float(series.max())

        moved_mask = np.abs(series - first_val) > 1e-3
        if moved_mask.any():
            first_move = int(np.argmax(moved_mask))
        else:
            first_move = -1
        dead = total_delta < 1e-3
        dead_str = "DEAD" if dead else ""
        first_move_str = "(none)" if first_move < 0 else str(first_move)
        print(f"{name:<18} | {fmt(min_p):>10} | {fmt(max_p):>10} | {fmt(total_delta):>12} | "
              f"{first_move_str:>10} | {dead_str}")


def close_finger_response(df: pd.DataFrame) -> None:
    print("\n[A.4 CLOSE state finger response — first 5 steps]")
    finger_cols = ["jpos_left_proximal", "jpos_left_distal", "jpos_right_proximal",
                   "jpos_right_distal", "jpos_gripper"]
    sub = df[(df["state"] == "CLOSE") | (df["force_close"] == 1)].copy()
    if len(sub) == 0:
        print("(empty: oracle never entered CLOSE and force_close was never triggered)")
        return

    sub = sub.head(5)
    has_force = (sub["force_close"] == 1).any()
    natural = (sub["state"] == "CLOSE").any()
    src = "natural" if natural and not has_force else ("force_close" if has_force and not natural
                                                      else "mixed")
    print(f"(source={src})")
    print(f"{'step':>4} | {'l_prox':>10} | {'l_dist':>10} | {'r_prox':>10} | "
          f"{'r_dist':>10} | {'gripper':>10} | {'cmd_grip':>10} | {'state':>8} | force")
    print("-" * 100)
    for _, row in sub.iterrows():
        print(f"{int(row['step']):>4} | {fmt(row['jpos_left_proximal']):>10} | "
              f"{fmt(row['jpos_left_distal']):>10} | {fmt(row['jpos_right_proximal']):>10} | "
              f"{fmt(row['jpos_right_distal']):>10} | {fmt(row['jpos_gripper']):>10} | "
              f"{fmt(row['action_5']):>10} | {row['state']:>8} | {int(row['force_close'])}")


def action_ee_stats(df: pd.DataFrame) -> None:
    print("\n[B.1 Arm Action / EE Delta Statistics — APPROACH state]")
    print("(env step granularity. decimation N is already collapsed inside env.step.)")
    sub = df[df["state"] == "APPROACH"].copy().reset_index(drop=True)
    if len(sub) < 2:
        print(f"(skipped: only {len(sub)} APPROACH rows, need >=2)")
        return

    arm_action = np.abs(sub[[f"action_{i}" for i in range(5)]].to_numpy())
    target = sub[["target_ee_x", "target_ee_y", "target_ee_z"]].to_numpy()
    current = sub[["current_ee_x", "current_ee_y", "current_ee_z"]].to_numpy()
    target_delta = np.linalg.norm(target - current, axis=1)
    actual_delta = np.linalg.norm(np.diff(current, axis=0), axis=1)
    target_delta_paired = target_delta[:-1]

    eps = 1e-9
    ratio = actual_delta / np.maximum(target_delta_paired, eps)

    print(f"\n  rows={len(sub)} (APPROACH)")
    print(f"  {'metric':<24} | {'mean':>12} | {'min':>12} | {'max':>12} | {'std':>12}")
    print("  " + "-" * 84)
    for i in range(5):
        v = arm_action[:, i]
        print(f"  |action_{i}| (arm)         | {fmt(v.mean()):>12} | {fmt(v.min()):>12} | "
              f"{fmt(v.max()):>12} | {fmt(v.std()):>12}")
    print(f"  {'|target_delta_3d|':<24} | {fmt(target_delta.mean()):>12} | "
          f"{fmt(target_delta.min()):>12} | {fmt(target_delta.max()):>12} | "
          f"{fmt(target_delta.std()):>12}")
    print(f"  {'|actual_delta_3d|':<24} | {fmt(actual_delta.mean()):>12} | "
          f"{fmt(actual_delta.min()):>12} | {fmt(actual_delta.max()):>12} | "
          f"{fmt(actual_delta.std()):>12}")
    print(f"  {'ratio (actual/target)':<24} | {fmt(ratio.mean()):>12} | "
          f"{fmt(ratio.min()):>12} | {fmt(ratio.max()):>12} | {fmt(ratio.std()):>12}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"loaded {len(df)} rows from {args.csv}")
    print(f"states observed: {sorted(df['state'].unique().tolist())}")
    print(f"force_close rows: {int((df['force_close'] == 1).sum())}")

    joint_activation_table(df)
    close_finger_response(df)
    action_ee_stats(df)


if __name__ == "__main__":
    main()
