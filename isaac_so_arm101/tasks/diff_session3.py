"""Quantitative trace diff between two diagnose_joints CSVs (session 2 vs session 3).

Aligns rows by step index. For every numeric column reports max abs diff, mean abs
diff, and the count of steps where |diff| > 1e-6.

Usage:
  python diff_session3.py --csv_a <path> --csv_b <path>
"""
import argparse

import numpy as np
import pandas as pd


def fmt(x: float, w: int = 12) -> str:
    if abs(x) < 1e-9:
        return f"{0.0:>{w}.6f}"
    if abs(x) >= 100 or abs(x) < 1e-4:
        return f"{x:>{w}.3e}"
    return f"{x:>{w}.6f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_a", required=True, help="baseline CSV (session 2)")
    p.add_argument("--csv_b", required=True, help="comparison CSV (session 3)")
    args = p.parse_args()

    da = pd.read_csv(args.csv_a)
    db = pd.read_csv(args.csv_b)

    n = min(len(da), len(db))
    da = da.iloc[:n].reset_index(drop=True)
    db = db.iloc[:n].reset_index(drop=True)

    print(f"comparing {n} rows ({args.csv_a} vs {args.csv_b})")
    print(f"\n[Session 2 vs 3 Trace Diff] (|diff| threshold = 1e-6)")
    print(f"{'column':<28} | {'max_abs_diff':>12} | {'mean_abs_diff':>13} | {'nonzero_steps':>13}")
    print("-" * 80)

    skip = {"step", "state", "force_close"}
    for col in da.columns:
        if col in skip or col not in db.columns:
            continue
        a = da[col].to_numpy()
        b = db[col].to_numpy()
        if a.dtype == object or b.dtype == object:
            equal = (a == b)
            print(f"{col:<28} | {('SAME' if equal.all() else 'DIFFER'):>12} | "
                  f"{'-':>13} | {(~equal).sum():>13}")
            continue
        diff = np.abs(a - b)
        max_d = float(diff.max()) if len(diff) else 0.0
        mean_d = float(diff.mean()) if len(diff) else 0.0
        nz = int((diff > 1e-6).sum())
        marker = " *" if nz > 0 else "  "
        print(f"{col:<28}{marker}| {fmt(max_d):>12} | {fmt(mean_d):>13} | {nz:>13}")


if __name__ == "__main__":
    main()
