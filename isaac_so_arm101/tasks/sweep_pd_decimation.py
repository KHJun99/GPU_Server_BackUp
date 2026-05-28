"""PD stiffness × decimation × damping ablation sweep.

Patches so_arm101.py and lift_env_cfg.py in-place per combo, runs a short 50-step
diagnose_h2_force, computes PD tracking ratio (stage 3) and torque saturation %,
then restores the cfg files. Outputs CSV with one row per combo.

Designed to run AFTER session-6 option-4 measurements (serial). Roughly 18 combos
× ~25s = ~8 min.

Usage:
  CUDA_VISIBLE_DEVICES=1 uv run python tasks/sweep_pd_decimation.py
"""

import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROBOT_CFG = Path("/home/j-k14d101/isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/so_arm101.py")
ENV_CFG = Path("/home/j-k14d101/isaac_so_arm101/src/isaac_so_arm101/tasks/lift/lift_env_cfg.py")
OUTPUT_CSV = Path("/home/j-k14d101/isaac_so_arm101/tasks/sweep_results.csv")
TMP_CSV_DIR = Path("/tmp/sweep_csvs")
TMP_CSV_DIR.mkdir(exist_ok=True)

STIFFNESS_MULS = [1.0, 2.0, 5.0]
DECIMATIONS = [4, 2, 1]
DAMPING_MULS = [1.0, 0.7]

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
EFFORT_LIMIT = 1.9  # baseline (so_arm101 currently has 2.5 from session 5; we treat 1.9 as nominal)


def _patch_named_dict(text: str, dict_name: str, mul: float) -> str:
    """Find `<dict_name>={ ... }` and multiply each numeric value (after a colon)
    inside the braces. Avoids touching init_state.joint_pos and other dicts."""
    pattern = re.compile(rf'({dict_name}\s*=\s*\{{)([^}}]*)(\}})')

    def replace(m: re.Match) -> str:
        body = m.group(2)
        body_new = re.sub(r'(:\s*)([\d.]+)',
                          lambda mm: f"{mm.group(1)}{float(mm.group(2)) * mul:.6f}",
                          body)
        return m.group(1) + body_new + m.group(3)

    return pattern.sub(replace, text, count=1)


def patch_robot_cfg(text: str, stiff_mul: float, damp_mul: float) -> str:
    """Multiply each numeric value in the `stiffness={...}` and `damping={...}` dicts
    inside so_arm101.py. Targets the first occurrence of each (which is the arm
    actuator)."""
    out = _patch_named_dict(text, "stiffness", stiff_mul)
    out = _patch_named_dict(out, "damping", damp_mul)
    return out


def patch_env_decimation(text: str, decimation: int) -> str:
    return re.sub(r"self\.decimation\s*=\s*\d+", f"self.decimation = {decimation}", text)


def run_one_combo(stiff: float, damp: float, dec: int, robot_orig: str, env_orig: str) -> dict:
    """Returns dict with ratio_pd, torque_max, saturation_pct."""
    print(f"\n[combo] stiff×{stiff} damp×{damp} dec={dec}", flush=True)

    # Apply patches.
    ROBOT_CFG.write_text(patch_robot_cfg(robot_orig, stiff, damp))
    ENV_CFG.write_text(patch_env_decimation(env_orig, dec))

    csv_path = TMP_CSV_DIR / f"sweep_s{stiff}_d{damp}_dec{dec}.csv"
    stderr_path = TMP_CSV_DIR / "stderr.log"

    cmd = (
        f"cd /home/j-k14d101/jabis_sim/sim2real/oracle && "
        f"CUDA_VISIBLE_DEVICES=1 timeout 90 uv run python "
        f"/home/j-k14d101/isaac_so_arm101/tasks/diagnose_h2_force.py "
        f"--task Isaac-SO-ARM101-Lift-Cube-Play-v0 --num_envs 1 --max_steps 50 "
        f"--output_csv {csv_path} --output_stderr {stderr_path} --headless"
    )
    t0 = time.time()
    rc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(f"  ran in {elapsed:.1f}s, exit={rc.returncode}", flush=True)
    if rc.returncode != 0 or not csv_path.exists():
        return {"ratio_pd": float("nan"), "torque_max": float("nan"),
                "saturation_pct": float("nan"), "elapsed": elapsed,
                "error": rc.stderr[-200:] if rc.stderr else ""}

    return analyze_csv(csv_path) | {"elapsed": elapsed, "error": ""}


def analyze_csv(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    sub = df[df["state"] == "APPROACH"].reset_index(drop=True)
    if len(sub) < 2:
        return {"ratio_pd": float("nan"), "torque_max": float("nan"), "saturation_pct": float("nan")}

    # Stage 3: |jpos_delta| / |joint_target_delta|
    arm_target = sub[[f"applied_target_{j}" for j in ARM_JOINTS]].to_numpy()
    arm_jpos = sub[[f"jpos_{j}" for j in ARM_JOINTS]].to_numpy()
    jt_delta = np.linalg.norm(arm_target - arm_jpos, axis=1)
    jpos_delta = np.linalg.norm(np.diff(arm_jpos, axis=0), axis=1)
    ratio = float((jpos_delta / np.maximum(jt_delta[:-1], 1e-9)).mean())

    arm_torque = sub[[f"applied_torque_{j}" for j in ARM_JOINTS]].to_numpy()
    torque_max = float(np.abs(arm_torque).max())
    sat_pct = float((np.abs(arm_torque) > 0.95 * EFFORT_LIMIT).mean() * 100.0)

    return {"ratio_pd": ratio, "torque_max": torque_max, "saturation_pct": sat_pct}


def main():
    robot_orig = ROBOT_CFG.read_text()
    env_orig = ENV_CFG.read_text()
    print(f"backed up {ROBOT_CFG.name} ({len(robot_orig)} bytes), {ENV_CFG.name} ({len(env_orig)} bytes)", flush=True)

    fp = open(OUTPUT_CSV, "w", newline="")
    writer = csv.writer(fp)
    writer.writerow(["stiff_mul", "damping_mul", "decimation",
                     "ratio_pd", "torque_max", "saturation_pct", "elapsed_s", "error"])

    try:
        for stiff in STIFFNESS_MULS:
            for damp in DAMPING_MULS:
                for dec in DECIMATIONS:
                    res = run_one_combo(stiff, damp, dec, robot_orig, env_orig)
                    writer.writerow([stiff, damp, dec, res["ratio_pd"], res["torque_max"],
                                     res["saturation_pct"], res["elapsed"], res["error"][:80]])
                    fp.flush()
                    print(f"  → ratio={res['ratio_pd']:.4f}  tq_max={res['torque_max']:.3f}  "
                          f"sat={res['saturation_pct']:.1f}%", flush=True)
    finally:
        ROBOT_CFG.write_text(robot_orig)
        ENV_CFG.write_text(env_orig)
        print("\n[restore] cfg files restored", flush=True)
        fp.close()
    print(f"[DONE] sweep results → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
