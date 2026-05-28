"""Compare session 14 PPO checkpoints (std=0.02) vs session 13 baselines."""

import numpy as np
import pandas as pd
from pathlib import Path

T = Path("/home/j-k14d101/isaac_so_arm101/tasks")


def safe_mean(s):
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.mean()) if len(s) else float("nan")


def fmt(v, w=7):
    if not np.isfinite(v):
        return f"{'nan':>{w}}"
    if abs(v) >= 100:
        return f"{v:>{w}.1f}"
    if abs(v) > 0 and abs(v) < 0.001:
        return f"{v:>{w}.2e}"
    return f"{v:>{w}.4f}"


def main():
    iters = [0, 200, 500, 999]
    dfs = {}
    for it in iters:
        p = T / f"diagnose_session14_iter{it}.csv"
        if p.exists():
            dfs[f"iter{it}"] = pd.read_csv(p)
    for name in ["oracle", "bc", "ppo"]:
        p = T / f"diagnose_{name}_session13.csv"
        if p.exists():
            dfs[f"s13_{name}"] = pd.read_csv(p)

    order = ["s13_oracle", "s13_bc", "s13_ppo", "iter0", "iter200", "iter500", "iter999"]
    labels = ["s13 Or", "s13 BC", "s13 PPO", "s14 i0", "s14 i200", "s14 i500", "s14 i999"]
    widths = [8, 8, 8, 8, 8, 8, 8]

    header = f"{'metric':<32}"
    for lab, w in zip(labels, widths):
        header += " | " + f"{lab:>{w}}"
    print(header)
    print("-" * len(header))

    metrics = [
        ("success rate (%)", lambda d: d["success"].mean() * 100),
        ("ee_to_cube_min mean (m)", lambda d: d["ee_to_cube_min"].mean()),
        ("ee_to_cube_min < 0.05 (%)", lambda d: (d["ee_to_cube_min"] < 0.05).mean() * 100),
        ("ee_to_cube_min < 0.02 (%)", lambda d: (d["ee_to_cube_min"] < 0.02).mean() * 100),
        ("ep with close (%)", lambda d: (d["first_close_step"] >= 0).mean() * 100),
        ("ee_dist_at_close mean", lambda d: safe_mean(d["ee_to_cube_at_close"])),
        ("cube_vel_max after close", lambda d: safe_mean(d["cube_vel_max_after_close"])),
        ("z_at_grasp_max", lambda d: safe_mean(d["z_at_grasp_max"])),
        ("z_max mean", lambda d: d["z_max"].mean()),
    ]
    for mname, fn in metrics:
        line = f"{mname:<32}"
        for k, w in zip(order, widths):
            if k in dfs:
                line += " | " + fmt(fn(dfs[k]), w)
            else:
                line += " | " + " " * w
        print(line)

    print()
    print("=== Key gate: ee_to_cube_min < 5cm (target ≥ 50%) ===")
    for k in order:
        if k in dfs:
            v = (dfs[k]["ee_to_cube_min"] < 0.05).mean() * 100
            print(f"  {k:<15} : {v:5.1f}%")

    print()
    print("=== Distribution (iter 0 vs iter 999) ===")
    if "iter0" in dfs and "iter999" in dfs:
        d0 = dfs["iter0"]
        d9 = dfs["iter999"]
        print(
            f"  iter0   ee_to_cube_min: μ={d0['ee_to_cube_min'].mean():.4f}  σ={d0['ee_to_cube_min'].std():.4f}  min={d0['ee_to_cube_min'].min():.4f}  max={d0['ee_to_cube_min'].max():.4f}"
        )
        print(
            f"  iter999 ee_to_cube_min: μ={d9['ee_to_cube_min'].mean():.4f}  σ={d9['ee_to_cube_min'].std():.4f}  min={d9['ee_to_cube_min'].min():.4f}  max={d9['ee_to_cube_min'].max():.4f}"
        )
        diff = (d9["ee_to_cube_min"] - d0["ee_to_cube_min"]).abs()
        print(f"  |iter999 - iter0| mean diff: {diff.mean():.4f}  (0 = identical trajectory)")
        # check seed alignment via cube_init
        seed_diff = (d9["cube_init_x"] - d0["cube_init_x"]).abs().mean()
        print(f"  cube_init_x diff: {seed_diff:.5f}  (0 = same seed)")


if __name__ == "__main__":
    main()
