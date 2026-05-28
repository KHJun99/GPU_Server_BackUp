"""Session 13~17 ablation evolution plot."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

T = Path("/home/j-k14d101/isaac_so_arm101/tasks")


def safe_mean(s):
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.mean()) if len(s) else float("nan")


def metrics_from(p: Path) -> dict:
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    return {
        "succ": d["success"].mean() * 100,
        "ee5": (d["ee_to_cube_min"] < 0.05).mean() * 100,
        "ee_min": d["ee_to_cube_min"].mean() * 100,
        "close_dist": safe_mean(d["ee_to_cube_at_close"]) * 100,
    }


def main():
    sources = [
        ("s13 Oracle", T / "diagnose_oracle_session13.csv"),
        ("s13 BC v8", T / "diagnose_bc_session13.csv"),
        ("s13 PPO best", T / "diagnose_ppo_session13.csv"),
    ]
    for it in [0, 100, 200, 400, 500, 999]:
        sources.append((f"s14 i{it}", T / f"diagnose_session14_iter{it}.csv"))
    for it in [0, 50, 100, 150, 199]:
        sources.append((f"s15 i{it}", T / f"diagnose_session15_iter{it}.csv"))
    for it in [0, 200, 500, 1000, 1500, 1999]:
        sources.append((f"s17 i{it}", T / f"diagnose_session17_iter{it}.csv"))

    rows = []
    for label, p in sources:
        m = metrics_from(p)
        if m:
            rows.append((label, m))

    labels = [r[0] for r in rows]
    succ = [r[1]["succ"] for r in rows]
    ee5 = [r[1]["ee5"] for r in rows]
    ee_min = [r[1]["ee_min"] for r in rows]
    close = [r[1]["close_dist"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    bars = ax.bar(range(len(labels)), succ, color=["green" if s >= 25 else "orange" if s >= 15 else "red" for s in succ])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Success Rate Evolution")
    ax.axhline(15, color="gray", lw=0.5, ls="--", label="14-15% saddle")
    ax.axhline(25, color="green", lw=0.5, ls="--", label="Path A target")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    for i, v in enumerate(succ):
        ax.text(i, v + 0.5, f"{v:.0f}", ha="center", fontsize=7)

    ax = axes[0, 1]
    bars = ax.bar(range(len(labels)), ee5, color=["green" if s >= 50 else "orange" if s >= 25 else "red" for s in ee5])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ee→cube < 5cm reach rate (%)")
    ax.set_title("Reach Rate Evolution (saddle indicator)")
    ax.axhline(50, color="green", lw=0.5, ls="--", label="reward fix gate")
    ax.axhline(86, color="purple", lw=0.5, ls="--", label="Oracle (86%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    for i, v in enumerate(ee5):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=7)

    ax = axes[1, 0]
    ax.bar(range(len(labels)), ee_min, color="steelblue")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ee→cube min mean (cm)")
    ax.set_title("ee_to_cube_min mean — lower is better")
    ax.axhline(4.6, color="purple", lw=0.5, ls="--", label="Oracle (4.6cm)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    valid = [(l, c) for l, c in zip(labels, close) if np.isfinite(c)]
    ax.bar(range(len(valid)), [c for _, c in valid], color="coral")
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels([l for l, _ in valid], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ee_dist_at_close mean (cm)")
    ax.set_title("ee_dist_at_close — close timing accuracy (lower better)")
    ax.axhline(4.8, color="purple", lw=0.5, ls="--", label="Oracle (4.8cm)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Sessions 13~17 Ablation Evolution", fontsize=14)
    fig.tight_layout()
    out = Path("/home/j-k14d101/jabis_sim/day5/videos/presentation/ablation_evolution.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(T / "ablation_evolution.png", dpi=150, bbox_inches="tight")
    print(f"saved → {out}")
    print(f"saved → {T / 'ablation_evolution.png'}")
    print(f"\n{'label':<15} | {'succ%':>6} | {'ee5%':>6} | {'ee_min':>7} | {'clos':>7}")
    print("-" * 50)
    for l, s, e, em, c in zip(labels, succ, ee5, ee_min, close):
        cstr = f"{c:>7.2f}" if np.isfinite(c) else "    nan"
        print(f"{l:<15} | {s:>6.1f} | {e:>6.1f} | {em:>7.2f} | {cstr}")


if __name__ == "__main__":
    main()
