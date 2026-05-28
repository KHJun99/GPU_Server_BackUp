"""H2 35× squash decomposition + H6 torque saturation analysis.

Reads diagnose_h2_force CSV. Pure pandas/numpy (no Isaac).

H2 decomposition stages (codex order)
-------------------------------------
1) clip:           |clipped_action| / |pre_action|
2) scale to joint: |joint_target_delta| / (|clipped_action| × scale=1.5)
   (joint_target = applied_target for that joint, delta = applied_target_t - jpos_{t-1})
3) tracking:       |joint_pos_delta| / |joint_target_delta|
4) ee aggregate:   |actual_ee_delta_3d| / |target_ee_delta_3d|

H6 torque saturation
--------------------
For each joint: mean / max |applied_torque|, saturation % = fraction of steps
where |torque| > 0.95 × effort_limit. effort_limit per joint:
  arm joints (shoulder_*, elbow_flex, wrist_*): 1.9
  gripper joints (left_proximal, right_proximal): 2.5
  others (gripper joint, distals): no actuator, expect torque ≈ 0
"""

import argparse

import numpy as np
import pandas as pd


ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIP_DRIVEN = ["left_proximal", "right_proximal"]
PASSIVE = ["gripper", "left_distal", "right_distal"]
ALL_JOINTS = ARM_JOINTS + ["gripper", "left_proximal", "right_proximal", "left_distal", "right_distal"]
EFFORT_LIMIT = {j: 1.9 for j in ARM_JOINTS}
EFFORT_LIMIT.update({j: 2.5 for j in GRIP_DRIVEN})
SCALE = 1.5  # arm_action.scale per env cfg


def fmt(x: float, w: int = 12) -> str:
    if not np.isfinite(x):
        return f"{'NaN':>{w}}"
    if abs(x) < 1e-9:
        return f"{0.0:>{w}.6f}"
    if abs(x) >= 100 or abs(x) < 1e-4:
        return f"{x:>{w}.3e}"
    return f"{x:>{w}.6f}"


def stats(arr: np.ndarray) -> tuple[float, float, float, float]:
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return float(arr.mean()), float(arr.min()), float(arr.max()), float(arr.std())


def h2_decomposition(df: pd.DataFrame) -> None:
    print("\n[H2 35× Decomposition — APPROACH state, env step granularity]")
    sub = df[df["state"] == "APPROACH"].copy().reset_index(drop=True)
    if len(sub) < 2:
        print(f"(skipped: only {len(sub)} APPROACH rows)")
        return

    pre = np.abs(sub[[f"pre_action_{i}" for i in range(5)]].to_numpy())
    clipped = np.abs(sub[[f"clipped_action_{i}" for i in range(5)]].to_numpy())

    # Stage 1: clip ratio (per-channel)
    pre_norm = np.linalg.norm(pre, axis=1)
    clipped_norm = np.linalg.norm(clipped, axis=1)
    eps = 1e-9
    r1 = clipped_norm / np.maximum(pre_norm, eps)

    # Stage 2: applied_target delta vs scaled clipped_action
    # joint_target should match: default_offset + clipped_action * scale
    # We approximate by checking: for arm joints, applied_target - jpos_{t-1} ≈ clipped × scale
    # Use 3-D norm in joint-arm space.
    arm_target_t = sub[[f"applied_target_{j}" for j in ARM_JOINTS]].to_numpy()
    arm_jpos_t = sub[[f"jpos_{j}" for j in ARM_JOINTS]].to_numpy()
    # joint_target_delta_t = applied_target[t] - jpos[t]  (target departure from current)
    joint_target_delta = arm_target_t - arm_jpos_t
    # scaled action ≈ clipped × 1.5
    scaled_norm = np.linalg.norm(clipped * SCALE, axis=1)
    jt_delta_norm = np.linalg.norm(joint_target_delta, axis=1)
    r2 = jt_delta_norm / np.maximum(scaled_norm, eps)

    # Stage 3: tracking — joint_pos_delta / joint_target_delta
    # joint_pos_delta = jpos[t+1] - jpos[t]
    arm_jpos_next = arm_jpos_t[1:]
    arm_jpos_curr = arm_jpos_t[:-1]
    jpos_delta_norm = np.linalg.norm(arm_jpos_next - arm_jpos_curr, axis=1)
    jt_delta_paired = jt_delta_norm[:-1]
    r3 = jpos_delta_norm / np.maximum(jt_delta_paired, eps)

    # Stage 4: ee aggregate
    target_ee = sub[["target_ee_x", "target_ee_y", "target_ee_z"]].to_numpy()
    current_ee = sub[["current_ee_x", "current_ee_y", "current_ee_z"]].to_numpy()
    target_ee_delta = np.linalg.norm(target_ee - current_ee, axis=1)
    actual_ee_delta = np.linalg.norm(np.diff(current_ee, axis=0), axis=1)
    target_ee_paired = target_ee_delta[:-1]
    r4 = actual_ee_delta / np.maximum(target_ee_paired, eps)

    print(f"  rows={len(sub)} APPROACH steps")
    print(f"  {'stage':<48} | {'mean':>12} | {'min':>12} | {'max':>12} | {'std':>12}")
    print("  " + "-" * 108)
    for label, r in [
        ("(1) |clipped| / |pre_action|", r1),
        ("(2) |joint_target_delta| / (|clipped| × scale=1.5)", r2),
        ("(3) |jpos_delta| / |joint_target_delta|  (tracking)", r3),
        ("(4) |actual_ee_delta| / |target_ee_delta| (ee total)", r4),
    ]:
        m, mn, mx, sd = stats(r)
        print(f"  {label:<48} | {fmt(m):>12} | {fmt(mn):>12} | {fmt(mx):>12} | {fmt(sd):>12}")


def h6_torque_saturation(df: pd.DataFrame) -> None:
    print("\n[H6 Torque Saturation per Joint]")
    print(f"  {'joint':<18} | {'effort_limit':>12} | {'mean|t|':>12} | "
          f"{'max|t|':>12} | {'sat%>0.95':>12}")
    print("  " + "-" * 80)
    for j in ALL_JOINTS:
        col = f"applied_torque_{j}"
        if col not in df.columns:
            print(f"  {j:<18} | (missing)")
            continue
        t = np.abs(df[col].to_numpy())
        finite_t = t[np.isfinite(t)]
        limit = EFFORT_LIMIT.get(j, None)
        sat_str = "(no actuator)"
        if limit is not None and len(finite_t) > 0:
            sat = (finite_t > 0.95 * limit).mean() * 100.0
            sat_str = f"{sat:.1f}%"
        m = float(finite_t.mean()) if len(finite_t) else float("nan")
        mx = float(finite_t.max()) if len(finite_t) else float("nan")
        limit_str = f"{limit:.1f}" if limit is not None else "(passive)"
        print(f"  {j:<18} | {limit_str:>12} | {fmt(m):>12} | {fmt(mx):>12} | {sat_str:>12}")


def integrity_checks(df: pd.DataFrame) -> None:
    print("\n[Integrity checks]")
    print(f"  rows: {len(df)}")
    print(f"  states: {sorted(df['state'].unique().tolist())}")
    print(f"  force_close rows: {int((df['force_close'] == 1).sum())}")
    nan_torque_cols = []
    for j in ALL_JOINTS:
        col = f"applied_torque_{j}"
        if col in df.columns:
            if df[col].isna().all():
                nan_torque_cols.append(j)
    if nan_torque_cols:
        print(f"  applied_torque all-NaN for: {nan_torque_cols}")
    nan_target_cols = []
    for j in ALL_JOINTS:
        col = f"applied_target_{j}"
        if col in df.columns:
            if df[col].isna().all():
                nan_target_cols.append(j)
    if nan_target_cols:
        print(f"  applied_target all-NaN for: {nan_target_cols}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"loaded {len(df)} rows from {args.csv}")
    integrity_checks(df)
    h2_decomposition(df)
    h6_torque_saturation(df)


if __name__ == "__main__":
    main()
