"""Plot session 9 / session 10 ablation curves.

Reads eval logs for v0 (session 9), v1 (session 10 entropy 0.005), v2 (session 10
entropy 0.001 + lr 3e-4) and produces a 1×3 figure: success / action_std / MSE-vs-BC.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASKS_DIR = Path("/home/j-k14d101/isaac_so_arm101/tasks")
ITERS = [0, 50, 100, 200, 500, 999]


def parse_eval_log(path: Path) -> dict:
    txt = path.read_text() if path.exists() else ""
    out = {}
    m = re.search(r"success_rate=([\d.]+)%", txt)
    if m:
        out["success"] = float(m.group(1))
    m = re.search(r"entropy proxy \(mean action_std\) = ([\d.eE+-]+)", txt)
    if m:
        out["std"] = float(m.group(1))
    m = re.search(r"action MSE vs BC = ([\d.eE+-]+)", txt)
    if m:
        out["mse"] = float(m.group(1))
    return out


def collect_run(prefix: str, iters):
    """prefix examples: 'ppo_eval_session9_iter', 'eval_session10_v1_iter', 'eval_session10_v2_iter'."""
    rows = []
    for it in iters:
        p = TASKS_DIR / f"{prefix}{it}.log"
        d = parse_eval_log(p)
        if d:
            d["iter"] = it
            rows.append(d)
    return rows


def main():
    v0 = collect_run("ppo_eval_session9_iter", ITERS)
    v1 = collect_run("eval_session10_v1_iter", [0, 100, 200, 500, 999])
    v2 = collect_run("eval_session10_v2_iter", [0, 100, 200, 500, 999])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for label, run, color in [
        ("v0 (ent=0.02, sess9)", v0, "tab:red"),
        ("v1 (ent=0.005, sess10)", v1, "tab:orange"),
        ("v2 (ent=0.001+lr3e-4)", v2, "tab:blue"),
    ]:
        if not run:
            continue
        xs = [r["iter"] for r in run]
        axes[0].plot(xs, [r.get("success", float("nan")) for r in run],
                     "-o", label=label, color=color)
        axes[1].plot(xs, [r.get("std", float("nan")) for r in run],
                     "-o", label=label, color=color)
        axes[2].plot(xs, [r.get("mse", float("nan")) for r in run],
                     "-o", label=label, color=color)

    # references
    axes[0].axhline(y=14.0, color="gray", ls="--", label="BC baseline 14%")
    axes[0].axhline(y=25.0, color="green", ls=":", label="gate 25%")

    axes[0].set_xlabel("iter")
    axes[0].set_ylabel("success rate %")
    axes[0].set_title("Success rate (sz=0.10)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("iter")
    axes[1].set_ylabel("mean action_std")
    axes[1].set_yscale("log")
    axes[1].set_title("action_std (entropy proxy)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].set_xlabel("iter")
    axes[2].set_ylabel("action MSE vs BC")
    axes[2].set_yscale("log")
    axes[2].set_title("Distance from BC (KL proxy)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle("PPO ablation: entropy_coef impact (BC saddle exit)", fontsize=12)
    fig.tight_layout()

    out = TASKS_DIR / "ablation_session10_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[DONE] saved → {out}")

    # Also print summary table
    print("\n[SUMMARY]")
    print(f"{'iter':>5} | {'v0 succ':>8} {'v0 std':>8} | "
          f"{'v1 succ':>8} {'v1 std':>8} | {'v2 succ':>8} {'v2 std':>8}")
    print("-" * 80)
    for it in [0, 100, 200, 500, 999]:
        def find(run):
            for r in run:
                if r["iter"] == it:
                    return r.get("success", "—"), r.get("std", "—")
            return "—", "—"
        s0, std0 = find(v0)
        s1, std1 = find(v1)
        s2, std2 = find(v2)
        def fnum(x):
            return f"{x:.2f}" if isinstance(x, float) else str(x)
        print(f"{it:>5} | {fnum(s0):>8} {fnum(std0):>8} | "
              f"{fnum(s1):>8} {fnum(std1):>8} | {fnum(s2):>8} {fnum(std2):>8}")


if __name__ == "__main__":
    main()
