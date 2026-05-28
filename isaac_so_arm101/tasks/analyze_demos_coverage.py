"""Analyze demos init cube position coverage.

obs idx 20=cube_x, 21=cube_y, 22=cube_z (constant 0.012). Identifies whether the
402 demos cover the cube init region broadly or narrowly.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", required=True)
    p.add_argument("--out_png", required=True)
    args = p.parse_args()

    d = torch.load(args.demos, map_location="cpu", weights_only=False)
    n = len(d["obs"])
    init_xy = np.array([[float(d["obs"][i][0][20]), float(d["obs"][i][0][21])] for i in range(n)])
    z_max = np.array([m["z_max"] for m in d["meta"]])
    final_state = np.array([m["final_state"] for m in d["meta"]])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: cube init xy scatter, color by z_max
    sc = axes[0].scatter(init_xy[:, 0], init_xy[:, 1], c=z_max, cmap="viridis",
                         s=12, alpha=0.7)
    axes[0].set_xlabel("cube init x (obs idx 20)")
    axes[0].set_ylabel("cube init y (obs idx 21)")
    axes[0].set_title(f"Cube init xy distribution (n={n})")
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].axvline(0.2, color="gray", lw=0.5, ls="--", label="default 0.2")
    axes[0].legend()
    plt.colorbar(sc, ax=axes[0], label="z_max")

    # Right: histograms
    axes[1].hist(init_xy[:, 0], bins=30, alpha=0.6, label=f"x  μ={init_xy[:,0].mean():.3f} σ={init_xy[:,0].std():.3f}")
    axes[1].hist(init_xy[:, 1], bins=30, alpha=0.6, label=f"y  μ={init_xy[:,1].mean():.3f} σ={init_xy[:,1].std():.3f}")
    axes[1].set_xlabel("position (m)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Cube init position histograms")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Demos coverage — {Path(args.demos).name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150, bbox_inches="tight")
    print(f"[DONE] saved → {args.out_png}")

    # Stats
    print(f"\n[Stats]")
    print(f"  n demos: {n}")
    print(f"  cube_x: mean={init_xy[:,0].mean():.4f}  std={init_xy[:,0].std():.4f}  "
          f"range=[{init_xy[:,0].min():.3f}, {init_xy[:,0].max():.3f}]")
    print(f"  cube_y: mean={init_xy[:,1].mean():.4f}  std={init_xy[:,1].std():.4f}  "
          f"range=[{init_xy[:,1].min():.3f}, {init_xy[:,1].max():.3f}]")
    print(f"  z_max:  mean={z_max.mean():.4f}  range=[{z_max.min():.3f}, {z_max.max():.3f}]")
    print(f"  final_state distribution: {dict(zip(*np.unique(final_state, return_counts=True)))}")


if __name__ == "__main__":
    main()
