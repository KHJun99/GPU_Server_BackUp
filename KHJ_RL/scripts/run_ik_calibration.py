"""Method B M3.3 + M0.5 Step D/E — IK calibration via oracle rollouts.

Boots Isaac Lab, runs the new EE-delta MotionPlanningOracle for
N_calibration + N_validation episodes, and computes:

  D)  ``ee_delta_max_m`` recommendation
        Per-phase statistics (median / p90 / p95 / max) of the oracle's
        *pre-clip* natural EE delta magnitude. Recommended cap is the
        global-max p95 across phases (v7 plan §6 M0.5 Step D).

  E)  ``T_jitter``, ``T_jerk`` calibration + validation split
        Calibration set: p95 of jitter_residual_m and q_jerk_norm_rad_s2
        across all calibration-episode steps. Recommended thresholds =
        calibration p95 × ``--threshold-margin`` (default 2.5).
        Validation set: simulate enabling the OR-guard with the new
        thresholds and report trip rate. PASS if trip_rate < 0.01.

  M3) New EE-delta oracle eval (sanity)
        success rate, IK fail rate, fail_reason distribution, terminal
        window IK fail count, vel_stab violation rate.

Outputs:

  runs/calibration/ik_thresholds.json    — machine-readable recommendations
  runs/calibration/ik_calibration_report.txt — human summary

Usage:
  OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y CUDA_VISIBLE_DEVICES=1 \\
    python scripts/run_ik_calibration.py \\
      --n-calibration 100 --n-validation 100

After running, manually update ``cfg.EEControlCfg`` with the recommended
``ee_delta_max_m`` / ``ik_jitter_residual_p95_max_m`` /
``ik_jerk_p95_max_rad_s2`` from the JSON. The script intentionally does
NOT auto-patch cfg — every threshold change should be human-reviewed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pinocchio  # noqa: F401, E402  (pre-AppLauncher Assimp ABI fix)

from isaaclab.app import AppLauncher  # noqa: E402  (must be first; isaaclab bootstrap)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-calibration", type=int, default=100,
                   help="Episodes for jitter/jerk p95 calibration.")
    p.add_argument("--n-validation", type=int, default=100,
                   help="Episodes for validating proposed thresholds.")
    p.add_argument("--out-dir", type=str, default="runs/calibration",
                   help="Output directory for JSON + report.")
    p.add_argument("--seed-cal-start", type=int, default=0)
    p.add_argument("--seed-val-start", type=int, default=100)
    p.add_argument("--threshold-margin", type=float, default=2.5,
                   help="T = calibration_p95 * margin (v7 plan §4.1, Step E).")
    p.add_argument("--terminal-window", type=int, default=20,
                   help="Match cfg.success_guard.terminal_window_steps.")
    p.add_argument("--curriculum-stage", type=int, default=0,
                   help="Index into CurriculumCfg.side_length_m (0=2cm narrow start).")
    p.add_argument("--task-level", type=int, default=2,
                   help="Force full PnP (5-cond AND) so vel_stab is evaluated.")
    return p.parse_args()


def _run_one_episode(env, oracle, seed: int) -> dict:
    """Run a single episode and return per-step + episode summary records.

    The episode terminates on env.terminated (task success) or
    env.truncated (max_steps or IK consecutive truncate). Records are
    appended every step regardless of guard state — guards are
    *disabled* in cfg at calibration time (T=0), so every step's audit
    fields are clean measurements.
    """
    # Re-seed env's spawn RNG so calibration / validation episodes are
    # reproducible by seed (not by call order).
    env._rng = np.random.default_rng(seed)
    obs = env.reset()
    oracle.reset(obs, {})

    steps: list[dict] = []
    terminated = False
    truncated = False
    info: dict = {}
    while not (terminated or truncated):
        action = oracle.act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        natural = getattr(env, "_oracle_dbg_natural_delta_m", None)
        if natural is None:
            natural_mag = 0.0
        else:
            natural_mag = float(np.linalg.norm(natural))
        ik = info.get("ik", {})
        cmd_mag = float(np.linalg.norm([
            ik.get("commanded_ee_delta_x", 0.0),
            ik.get("commanded_ee_delta_y", 0.0),
            ik.get("commanded_ee_delta_z", 0.0),
        ]))
        steps.append({
            "phase": str(getattr(env, "_oracle_dbg_phase", "")),
            "natural_delta_mag_m": natural_mag,
            "commanded_delta_mag_m": cmd_mag,
            "jitter_residual_m": float(ik.get("jitter_residual_m", 0.0)),
            "q_jerk_norm_rad_s2": float(ik.get("q_jerk_norm_rad_s2", 0.0)),
            "x_err_norm_m": float(ik.get("x_err_norm_m", 0.0)),
            "fallback_used": bool(info.get("ik_fallback_used", False)),
            "fail_reason": str(info.get("ik_fail_reason", "") or ""),
            "ik_truncate": bool(info.get("ik_truncate_episode", False)),
            "slow_diverge_now": bool(ik.get("slow_diverge_now", False)),
        })

    final_per_cond = info.get("success_per_condition", {})
    return {
        "seed": int(seed),
        "n_steps": len(steps),
        "success": bool(info.get("success", False)),
        "terminated_task_success": bool(terminated),
        "truncated_by_ik_or_time": bool(truncated),
        "ik_truncated": bool(info.get("ik_truncate_episode", False)),
        "per_condition_terminal": {k: bool(v) for k, v in final_per_cond.items()},
        "steps": steps,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _summarize_per_phase(episodes: list[dict], field: str) -> dict[str, dict[str, float]]:
    """Group ``steps[i][field]`` by ``steps[i]['phase']`` and compute stats."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for ep in episodes:
        for s in ep["steps"]:
            buckets[s["phase"]].append(float(s[field]))
    out: dict[str, dict[str, float]] = {}
    for phase, vals in sorted(buckets.items()):
        if not vals:
            continue
        out[phase] = {
            "n": len(vals),
            "median": _percentile(vals, 50),
            "p90": _percentile(vals, 90),
            "p95": _percentile(vals, 95),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
        }
    return out


def _episode_metrics(episodes: list[dict], terminal_window: int) -> dict:
    """Compute episode-level + step-aggregate stats for M3 oracle eval."""
    total_steps = sum(ep["n_steps"] for ep in episodes)
    total_eps = len(episodes)
    if total_eps == 0 or total_steps == 0:
        return {
            "n_episodes": 0, "n_steps": 0,
            "success_rate": 0.0,
            "per_condition_terminal_rate": {},
            "ik_fail_rate": 0.0,
            "fail_reason_counts": {},
            "fail_reason_rate": {},
            "terminal_window_ik_fail_eps": 0,
            "terminal_window_ik_fail_rate": 0.0,
            "vel_stab_violation_eps": 0,
            "vel_stab_violation_rate": 0.0,
            "ik_truncate_eps": 0,
            "ik_truncate_rate": 0.0,
        }
    successes = sum(1 for ep in episodes if ep["success"])
    success_rate = successes / total_eps

    # Per-condition terminal rate.
    cond_keys: set[str] = set()
    for ep in episodes:
        cond_keys.update(ep["per_condition_terminal"].keys())
    per_cond_rate = {
        k: sum(1 for ep in episodes if ep["per_condition_terminal"].get(k, False)) / total_eps
        for k in sorted(cond_keys)
    }

    # Step-aggregate fail info.
    fail_count = 0
    fail_reasons: Counter = Counter()
    for ep in episodes:
        for s in ep["steps"]:
            if s["fallback_used"]:
                fail_count += 1
                fail_reasons[s["fail_reason"] or "unknown"] += 1
    ik_fail_rate = fail_count / total_steps if total_steps > 0 else 0.0
    fail_reason_rate = {k: v / total_steps for k, v in fail_reasons.items()}

    # Terminal-window IK fail (last ``terminal_window`` steps of each ep).
    twfe = 0
    for ep in episodes:
        tail = ep["steps"][-terminal_window:]
        if any(s["fallback_used"] for s in tail):
            twfe += 1
    twfe_rate = twfe / total_eps

    # vel_stab violation at terminal.
    vsv = sum(
        1 for ep in episodes
        if not ep["per_condition_terminal"].get("velocity_stability", True)
    )
    vsv_rate = vsv / total_eps

    # IK truncation.
    itr = sum(1 for ep in episodes if ep["ik_truncated"])
    itr_rate = itr / total_eps

    return {
        "n_episodes": total_eps,
        "n_steps": total_steps,
        "success_rate": success_rate,
        "per_condition_terminal_rate": per_cond_rate,
        "ik_fail_rate": ik_fail_rate,
        "fail_reason_counts": dict(fail_reasons),
        "fail_reason_rate": fail_reason_rate,
        "terminal_window_ik_fail_eps": twfe,
        "terminal_window_ik_fail_rate": twfe_rate,
        "vel_stab_violation_eps": vsv,
        "vel_stab_violation_rate": vsv_rate,
        "ik_truncate_eps": itr,
        "ik_truncate_rate": itr_rate,
    }


def _compute_thresholds(
    cal_episodes: list[dict],
    val_episodes: list[dict],
    margin: float,
) -> dict:
    """E step: derive T_jitter / T_jerk and validate trip rate."""
    cal_jitter = [s["jitter_residual_m"] for ep in cal_episodes for s in ep["steps"]]
    cal_jerk = [s["q_jerk_norm_rad_s2"] for ep in cal_episodes for s in ep["steps"]]
    jitter_p95 = _percentile(cal_jitter, 95)
    jerk_p95 = _percentile(cal_jerk, 95)
    T_jitter = float(jitter_p95 * margin)
    T_jerk = float(jerk_p95 * margin)

    # Validate: simulate "if these thresholds had been active" on val set.
    val_steps = [s for ep in val_episodes for s in ep["steps"]]
    n_val = len(val_steps)
    jitter_trips = sum(1 for s in val_steps if s["jitter_residual_m"] > T_jitter)
    jerk_trips = sum(1 for s in val_steps if s["q_jerk_norm_rad_s2"] > T_jerk)
    jitter_rate = jitter_trips / n_val if n_val > 0 else 0.0
    jerk_rate = jerk_trips / n_val if n_val > 0 else 0.0

    return {
        "margin": margin,
        "cal_n_steps": len(cal_jitter),
        "cal_jitter_residual_p95_m": jitter_p95,
        "cal_q_jerk_p95_rad_s2": jerk_p95,
        "recommended_T_jitter_m": T_jitter,
        "recommended_T_jerk_rad_s2": T_jerk,
        "val_n_steps": n_val,
        "val_jitter_trip_rate": jitter_rate,
        "val_jerk_trip_rate": jerk_rate,
        "val_jitter_pass": jitter_rate < 0.01,
        "val_jerk_pass": jerk_rate < 0.01,
    }


def _format_report(
    args: argparse.Namespace,
    d_stats: dict,
    e_stats: dict,
    m3_cal: dict,
    m3_val: dict,
) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Method B IK calibration report — M0.5 D/E + M3.3 oracle eval")
    lines.append("=" * 72)
    lines.append(f"n_calibration={args.n_calibration} (seeds {args.seed_cal_start}..{args.seed_cal_start+args.n_calibration-1})")
    lines.append(f"n_validation ={args.n_validation} (seeds {args.seed_val_start}..{args.seed_val_start+args.n_validation-1})")
    lines.append(f"curriculum_stage={args.curriculum_stage}  task_level={args.task_level}")
    lines.append(f"threshold_margin={args.threshold_margin}  terminal_window={args.terminal_window}")
    lines.append("")

    # D
    lines.append("-" * 72)
    lines.append("D) ee_delta_max_m recommendation — per-phase natural delta")
    lines.append("-" * 72)
    lines.append(f"{'phase':<24} {'n':>6} {'median(m)':>10} {'p90(m)':>10} {'p95(m)':>10} {'max(m)':>10}")
    global_p95s: list[float] = []
    for phase, st in d_stats.items():
        lines.append(
            f"{phase:<24} {st['n']:>6d} "
            f"{st['median']:>10.4f} {st['p90']:>10.4f} "
            f"{st['p95']:>10.4f} {st['max']:>10.4f}"
        )
        global_p95s.append(st["p95"])
    recommended_ee_delta = float(max(global_p95s)) if global_p95s else 0.0
    lines.append("")
    lines.append(f"  -> recommended ee_delta_max_m = max(per-phase p95) = {recommended_ee_delta:.4f} m")
    lines.append( "     (current cfg.EEControlCfg.ee_delta_max_m = 0.015)")
    lines.append("")

    # E
    lines.append("-" * 72)
    lines.append("E) T_jitter / T_jerk calibration + validation")
    lines.append("-" * 72)
    lines.append(f"  calibration n_steps             = {e_stats['cal_n_steps']}")
    lines.append(f"  jitter_residual p95             = {e_stats['cal_jitter_residual_p95_m']:.6f} m")
    lines.append(f"  q_jerk p95                      = {e_stats['cal_q_jerk_p95_rad_s2']:.4f} rad/s^2")
    lines.append(f"  -> recommended T_jitter (×{args.threshold_margin}) = {e_stats['recommended_T_jitter_m']:.6f} m")
    lines.append(f"  -> recommended T_jerk   (×{args.threshold_margin}) = {e_stats['recommended_T_jerk_rad_s2']:.4f} rad/s^2")
    lines.append("")
    lines.append(f"  validation n_steps              = {e_stats['val_n_steps']}")
    lines.append(f"  jitter trip rate (target <1%)   = {e_stats['val_jitter_trip_rate']*100:.2f}%  "
                 f"{'PASS' if e_stats['val_jitter_pass'] else 'FAIL'}")
    lines.append(f"  jerk   trip rate (target <1%)   = {e_stats['val_jerk_trip_rate']*100:.2f}%  "
                 f"{'PASS' if e_stats['val_jerk_pass'] else 'FAIL'}")
    lines.append("")

    # M3
    for name, metrics in (("calibration", m3_cal), ("validation", m3_val)):
        lines.append("-" * 72)
        lines.append(f"M3) Oracle eval — {name} set")
        lines.append("-" * 72)
        lines.append(f"  n_episodes              = {metrics['n_episodes']}")
        lines.append(f"  n_steps                 = {metrics['n_steps']}")
        lines.append(f"  success_rate            = {metrics['success_rate']*100:.1f}%")
        lines.append(f"  vel_stab_violation_eps  = {metrics['vel_stab_violation_eps']} / {metrics['n_episodes']} "
                     f"({metrics['vel_stab_violation_rate']*100:.1f}%)")
        lines.append(f"  ik_fail_rate            = {metrics['ik_fail_rate']*100:.2f}%")
        lines.append(f"  ik_truncate_eps         = {metrics['ik_truncate_eps']} ({metrics['ik_truncate_rate']*100:.1f}%)")
        lines.append(f"  terminal_window_ik_fail = {metrics['terminal_window_ik_fail_eps']} eps "
                     f"({metrics['terminal_window_ik_fail_rate']*100:.1f}%)")
        if metrics["fail_reason_counts"]:
            lines.append("  fail_reason distribution (counts):")
            for reason, count in sorted(metrics["fail_reason_counts"].items(),
                                         key=lambda kv: -kv[1]):
                lines.append(f"    {reason:<14} {count:>6d} "
                             f"({metrics['fail_reason_rate'][reason]*100:.3f}%)")
        else:
            lines.append("  fail_reason distribution: (no IK fallbacks)")
        lines.append("  per-condition terminal rate:")
        for k, v in metrics["per_condition_terminal_rate"].items():
            lines.append(f"    {k:<22} {v*100:>5.1f}%")
        lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    sim_app = AppLauncher(headless=True, enable_cameras=False).app

    # Post-bootstrap imports.
    from khj_rl.envs import CubeLiftEnv, CubeLiftEnvCfg
    from khj_rl.envs.cube_lift.oracle import MotionPlanningOracle

    cfg = CubeLiftEnvCfg()
    cfg.curriculum.current_stage_idx = int(args.curriculum_stage)
    cfg.curriculum.task_level = int(args.task_level)
    env = CubeLiftEnv(cfg)
    oracle = MotionPlanningOracle(env)

    cal_episodes: list[dict] = []
    val_episodes: list[dict] = []

    print(f"[calib] running {args.n_calibration} calibration episodes...", flush=True)
    for i in range(args.n_calibration):
        seed = args.seed_cal_start + i
        ep = _run_one_episode(env, oracle, seed)
        cal_episodes.append(ep)
        if (i + 1) % 10 == 0:
            print(f"[calib]   cal {i+1}/{args.n_calibration} (success_so_far="
                  f"{sum(1 for e in cal_episodes if e['success'])}/{i+1})", flush=True)

    print(f"[calib] running {args.n_validation} validation episodes...", flush=True)
    for i in range(args.n_validation):
        seed = args.seed_val_start + i
        ep = _run_one_episode(env, oracle, seed)
        val_episodes.append(ep)
        if (i + 1) % 10 == 0:
            print(f"[calib]   val {i+1}/{args.n_validation} (success_so_far="
                  f"{sum(1 for e in val_episodes if e['success'])}/{i+1})", flush=True)

    # D — per-phase natural delta stats (calibration set only, since we
    # just need the oracle's natural distribution).
    d_stats = _summarize_per_phase(cal_episodes, "natural_delta_mag_m")
    recommended_ee_delta = float(max((s["p95"] for s in d_stats.values()), default=0.0))

    # E — jitter / jerk thresholds.
    e_stats = _compute_thresholds(cal_episodes, val_episodes, args.threshold_margin)

    # M3 — oracle eval (both sets reported separately).
    m3_cal = _episode_metrics(cal_episodes, args.terminal_window)
    m3_val = _episode_metrics(val_episodes, args.terminal_window)

    # Write outputs.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ik_thresholds.json"
    report_path = out_dir / "ik_calibration_report.txt"

    payload = {
        "args": vars(args),
        "D_ee_delta_max_m_per_phase": d_stats,
        "D_recommended_ee_delta_max_m": recommended_ee_delta,
        "E_thresholds": e_stats,
        "M3_oracle_eval_calibration": m3_cal,
        "M3_oracle_eval_validation": m3_val,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    report = _format_report(args, d_stats, e_stats, m3_cal, m3_val)
    report_path.write_text(report)
    print(report, flush=True)
    print(f"\n[calib] wrote {json_path}")
    print(f"[calib] wrote {report_path}")
    print("\n[calib] Next step: review the report, then manually update")
    print("[calib] cfg.EEControlCfg in src/khj_rl/envs/cube_lift/cfg.py:")
    print(f"[calib]   ee_delta_max_m              = {recommended_ee_delta:.4f}")
    print(f"[calib]   ik_jitter_residual_p95_max_m = {e_stats['recommended_T_jitter_m']:.6f}")
    print(f"[calib]   ik_jerk_p95_max_rad_s2      = {e_stats['recommended_T_jerk_rad_s2']:.4f}")

    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
