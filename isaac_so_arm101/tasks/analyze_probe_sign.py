"""Analyze gripper-sign probe CSV. Reports per-finger direction during close vs
open phase, plus tip distance change for cross-check (codex caveat)."""

import argparse
import numpy as np
import pandas as pd


FINGERS = ["left_proximal", "left_distal", "right_proximal", "right_distal", "gripper"]


def fmt(x: float, w: int = 11) -> str:
    if not np.isfinite(x):
        return f"{'NaN':>{w}}"
    if abs(x) < 1e-6:
        return f"{0.0:>{w}.6f}"
    if abs(x) >= 100 or abs(x) < 1e-4:
        return f"{x:>{w}.3e}"
    return f"{x:>{w}.6f}"


def direction(start: float, end: float, threshold: float = 5e-3) -> str:
    delta = end - start
    if abs(delta) < threshold:
        return "static"
    return "(+) up" if delta > 0 else "(-) down"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"loaded {len(df)} rows from {args.csv}")
    close_df = df[df["phase"] == "close"].reset_index(drop=True)
    open_df = df[df["phase"] == "open"].reset_index(drop=True)
    print(f"  close phase: {len(close_df)} rows  open phase: {len(open_df)} rows")
    print(f"  raw_grip_cmd close mean: {close_df['raw_grip_cmd'].mean():.2f}  "
          f"open mean: {open_df['raw_grip_cmd'].mean():.2f}")

    print("\n[Sign Probe — close phase]")
    print(f"{'joint':<18} | {'start':>11} | {'end':>11} | {'delta':>11} | "
          f"{'mean|t|':>11} | direction")
    print("-" * 90)
    for j in FINGERS:
        col_pos = f"jpos_{j}"
        col_t = f"jtorque_{j}"
        if col_pos not in close_df.columns:
            continue
        start = float(close_df[col_pos].iloc[0])
        end = float(close_df[col_pos].iloc[-1])
        delta = end - start
        if col_t in close_df.columns:
            mean_t = float(np.abs(close_df[col_t]).mean())
        else:
            mean_t = float("nan")
        dirn = direction(start, end)
        print(f"{j:<18} | {fmt(start)} | {fmt(end)} | {fmt(delta)} | "
              f"{fmt(mean_t)} | {dirn}")

    print("\n[Sign Probe — open phase]")
    print(f"{'joint':<18} | {'start':>11} | {'end':>11} | {'delta':>11} | "
          f"{'mean|t|':>11} | direction")
    print("-" * 90)
    for j in FINGERS:
        col_pos = f"jpos_{j}"
        col_t = f"jtorque_{j}"
        if col_pos not in open_df.columns:
            continue
        start = float(open_df[col_pos].iloc[0])
        end = float(open_df[col_pos].iloc[-1])
        delta = end - start
        if col_t in open_df.columns:
            mean_t = float(np.abs(open_df[col_t]).mean())
        else:
            mean_t = float("nan")
        dirn = direction(start, end)
        print(f"{j:<18} | {fmt(start)} | {fmt(end)} | {fmt(delta)} | "
              f"{fmt(mean_t)} | {dirn}")

    print("\n[Tip distance cross-check — finger separation]")
    print(f"{'phase':<8} | {'start':>11} | {'end':>11} | {'delta':>11} | grasp meaning")
    print("-" * 70)
    if "tip_distance" in df.columns:
        for label, sub in [("close", close_df), ("open", open_df)]:
            s = float(sub["tip_distance"].iloc[0])
            e = float(sub["tip_distance"].iloc[-1])
            d = e - s
            meaning = "tips closer (grasp)" if d < 0 else ("tips farther (release)" if d > 0 else "no change")
            print(f"{label:<8} | {fmt(s)} | {fmt(e)} | {fmt(d)} | {meaning}")

    # Verdict logic
    print("\n[Verdict]")
    static_close = []
    moving_close = {}
    for j in FINGERS:
        col = f"jpos_{j}"
        if col not in close_df.columns:
            continue
        delta = float(close_df[col].iloc[-1] - close_df[col].iloc[0])
        if abs(delta) < 5e-3:
            static_close.append(j)
        else:
            moving_close[j] = delta

    if static_close:
        print(f"  static joints during close phase: {static_close}")
        print(f"  -> H5 scenario C indicated (passive joints need actuator)")
    else:
        print(f"  all 5 fingers moved during close phase.")

    if moving_close:
        signs = {j: ("+" if d > 0 else "-") for j, d in moving_close.items()}
        print(f"  close-phase sign pattern: {signs}")
        unique_signs = set(signs.values())
        if len(unique_signs) == 1:
            print(f"  -> all moving fingers same direction. H5 scenario A (simple broadcast) OK.")
        else:
            print(f"  -> mixed directions. H5 scenario B (sign-aware broadcast) needed.")


if __name__ == "__main__":
    main()
