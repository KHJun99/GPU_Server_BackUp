"""Analyze diagnose_policy CSVs and produce summary tables + scatter plots."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

TASKS = Path("/home/j-k14d101/isaac_so_arm101/tasks")


def safe_mean(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def safe_count_lt(arr, thr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    return int((arr < thr).sum())


def fmt(x, w=10):
    if not np.isfinite(x):
        return f"{'NaN':>{w}}"
    if abs(x) >= 100 or (abs(x) > 0 and abs(x) < 0.001):
        return f"{x:>{w}.3e}"
    return f"{x:>{w}.4f}"


def load(name):
    p = TASKS / f"diagnose_{name}_session13.csv"
    df = pd.read_csv(p)
    return df


def policy_table(dfs):
    print("\n[Policy Behavior Comparison — 100 ep]")
    print(f"{'metric':<32} | {'Oracle':>10} | {'BC':>10} | {'PPO':>10}")
    print("-" * 75)
    for metric, fn, fmt_fn in [
        ("success rate (%)", lambda d: d["success"].mean() * 100, fmt),
        ("z_max mean", lambda d: d["z_max"].mean(), fmt),
        ("z_max max", lambda d: d["z_max"].max(), fmt),
        ("ee_to_cube_min mean (m)", lambda d: d["ee_to_cube_min"].mean(), fmt),
        ("ee_to_cube_min < 0.02 (%)", lambda d: (d["ee_to_cube_min"] < 0.02).mean() * 100, fmt),
        ("ee_to_cube_min < 0.05 (%)", lambda d: (d["ee_to_cube_min"] < 0.05).mean() * 100, fmt),
        ("first_close_step mean", lambda d: d.loc[d["first_close_step"] >= 0, "first_close_step"].mean(), fmt),
        ("ep with close attempt (%)", lambda d: (d["first_close_step"] >= 0).mean() * 100, fmt),
        ("gripper_closed_steps mean", lambda d: d["gripper_closed_steps"].mean(), fmt),
        ("ee_dist_at_close mean (m)", lambda d: safe_mean(d["ee_to_cube_at_close"]), fmt),
        ("z_at_first_close mean (m)", lambda d: safe_mean(d["z_at_first_close"]), fmt),
        ("cube_vel_max_after_close", lambda d: safe_mean(d["cube_vel_max_after_close"]), fmt),
        ("z_at_grasp_max mean (m)", lambda d: safe_mean(d["z_at_grasp_max"]), fmt),
    ]:
        vals = [fn(df) for df in dfs]
        print(f"{metric:<32} | {fmt_fn(vals[0])} | {fmt_fn(vals[1])} | {fmt_fn(vals[2])}")


def success_vs_fail(df, label):
    print(f"\n[{label}: Success vs Fail cube_init xy]")
    s = df[df["success"] == 1]
    f = df[df["success"] == 0]
    print(f"  n_success={len(s)}  n_fail={len(f)}")
    if len(s):
        print(f"  success cube_x: μ={s['cube_init_x'].mean():.4f}  σ={s['cube_init_x'].std():.4f}")
        print(f"  success cube_y: μ={s['cube_init_y'].mean():.4f}  σ={s['cube_init_y'].std():.4f}")
    if len(f):
        print(f"  fail    cube_x: μ={f['cube_init_x'].mean():.4f}  σ={f['cube_init_x'].std():.4f}")
        print(f"  fail    cube_y: μ={f['cube_init_y'].mean():.4f}  σ={f['cube_init_y'].std():.4f}")


def scatter_plot(dfs, names, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, df, name in zip(axes, dfs, names):
        s = df[df["success"] == 1]
        f = df[df["success"] == 0]
        ax.scatter(f["cube_init_x"], f["cube_init_y"], c="tab:red", s=15, alpha=0.5, label=f"fail ({len(f)})")
        ax.scatter(s["cube_init_x"], s["cube_init_y"], c="tab:green", s=25, alpha=0.9, label=f"success ({len(s)})")
        ax.set_xlabel("cube init x")
        ax.set_ylabel("cube init y")
        ax.set_title(f"{name}: success={len(s)}/{len(df)}")
        ax.axvline(0.2, color="gray", lw=0.5, ls="--")
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Cube init xy — success vs fail", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n[DONE] saved → {out_png}")


def main():
    dfs = [load("oracle"), load("bc"), load("ppo")]
    names = ["Oracle", "BC", "PPO"]
    policy_table(dfs)
    for df, name in zip(dfs, names):
        success_vs_fail(df, name)
    scatter_plot(dfs, names, TASKS / "session13_success_vs_fail.png")


if __name__ == "__main__":
    main()
