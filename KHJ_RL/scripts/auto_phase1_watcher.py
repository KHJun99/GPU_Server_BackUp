"""Overnight watcher: wait for PPO -> auto eval -> dump metrics -> health report.

Run inside tmux so SSH disconnects don't kill it. Steps the watcher
does (with the user asleep / away):

  1. **Wait for ``train_ppo.py`` to exit** (pgrep poll every 30 s).
     Also kills any sim_app-hung Python process that lingers >120 s
     after the ckpt was saved (the Isaac Sim close() zombie pattern
     burned us multiple times today — see troubleshooting.md #17).

  2. **Run ``eval_policy.py --episodes N``** against the saved ckpt.
     Defaults to 100 episodes (Phase 1 curriculum gate count).

  3. **Parse + dump metrics** to
     ``runs/<run_name>/eval_result.json``:
       - success_rate (0..1)
       - per_condition_pass_rate (5 keys, 0..1)
       - ep_length_stats (mean/min/max)
       - eval_log_path (cross-ref for the full transcript)

  4. **Write ``status_report.txt``** — single-file dashboard the user
     reads first thing in the morning:
       - run config (BC ckpt used, dense weights, LR, ent_coef, etc.)
       - training curve summary (final ent, v_loss, sps, ep_return)
       - eval summary (success rate, per-condition, ep_length)
       - video file list
       - auto-suggested next step (advance curriculum / tune weights
         / rollback) based on success rate threshold

  5. **30 min health check loop while waiting**: zombie sim_app
     processes get SIGTERM'd. Result logged.

Designed for overnight use. Foreground output everything to a log so
the user can scan it on wake-up via ``cat /tmp/watcher.log``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


_SUCCESS_RE = re.compile(r"\[eval\] SUCCESS RATE:\s+(\d+)/(\d+)")
_PER_COND_RE = re.compile(r"^\s+([a-z_]+):\s+(\d+)/(\d+)")
_EP_LEN_RE = re.compile(r"\[eval\] ep_length mean=([\d.]+) min=(\d+) max=(\d+)")
_PPO_ITER_RE = re.compile(
    r"^\[PPO\] iter=(\d+) step=(\d+) frozen_actor=(\w+) "
    r"ep_return_mean=(-?[\d.]+) ep_len_mean=([\d.]+) "
    r"pi_loss=(-?[\d.]+) v_loss=(-?[\d.]+) ent=([\d.]+) "
    r"kl=(-?[\d.]+) clipfrac=([\d.]+) sps=(\d+)"
)


def _log(msg: str) -> None:
    """Single-line timestamped log to stdout (will be tee'd to file)."""
    ts = time.strftime("%H:%M:%S")
    print(f"[watcher {ts}] {msg}", flush=True)


def _alive(pattern: str) -> list[int]:
    """Return PIDs matching ``pgrep -f pattern`` (or [] if none)."""
    res = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(x) for x in res.stdout.split() if x.strip()]


def _kill_zombies(ckpt_path: Path) -> None:
    """Kill sim_app-hung Python processes >120 s after ckpt was saved.

    Isaac Sim 4.5's ``sim_app.close()`` sometimes hangs in fabric/physx
    plugin unload — the Python process stays alive (and pins ~7 GB of
    GPU memory) until SIGTERM'd. We detect this as: ckpt file exists
    and is >120 s old, yet train_ppo.py / collect_demos.py / eval_policy.py
    is still in the process table. Safe to kill — work output (ckpt /
    eval_result) is already on disk before close() is called.
    """
    if not ckpt_path.exists():
        return
    ckpt_age = time.time() - ckpt_path.stat().st_mtime
    if ckpt_age < 120:
        return
    patterns = ["train_ppo.py", "collect_demos.py", "eval_policy.py"]
    for pat in patterns:
        for pid in _alive(pat):
            _log(f"killing zombie {pat} pid={pid} (ckpt age {ckpt_age:.0f} s)")
            try:
                os.kill(pid, 15)  # SIGTERM
            except ProcessLookupError:
                pass


def _wait_for_ppo_end(run_dir: Path, poll_s: int = 30) -> Path | None:
    """Block until train_ppo.py exits AND ppo.pt exists. Returns ckpt
    path on success, None on timeout (90 min after process death).

    Zombie-kill loop runs inline: every poll_s we check whether
    train_ppo died but the Python process is hanging in sim_app.close().
    """
    ckpt = run_dir / "ppo.pt"
    _log(f"waiting for PPO to finish; ckpt target = {ckpt}")
    last_ckpt_seen = None
    while True:
        running = _alive("train_ppo.py")
        if not running:
            if ckpt.exists():
                _log(f"PPO done: ckpt at {ckpt} (size {ckpt.stat().st_size} B)")
                return ckpt
            else:
                _log("train_ppo.py exited but no ckpt — likely crashed before save")
                return None
        # Process is alive — kill zombies if ckpt was saved long ago.
        if ckpt.exists():
            if last_ckpt_seen is None:
                last_ckpt_seen = time.time()
                _log(f"ckpt appeared at {ckpt} — watching for sim close() hang")
            _kill_zombies(ckpt)
        time.sleep(poll_s)


def _run_eval(
    ckpt: Path, episodes: int, log_path: Path, eval_env: dict[str, str]
) -> int:
    """Run eval_policy.py, tee output to log_path. Returns rc."""
    cmd = [
        sys.executable, "scripts/eval_policy.py",
        "--ckpt", str(ckpt),
        "--episodes", str(episodes),
    ]
    _log(f"running eval: {' '.join(cmd)}")
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=eval_env)
    _log(f"eval rc={proc.returncode}; log={log_path}")
    return proc.returncode


def _parse_eval(log_path: Path) -> dict:
    """Extract success_rate, per_condition_pass_rate, ep_length_stats."""
    text = log_path.read_text()
    result = {"success_rate": None, "per_condition_pass_rate": {}, "ep_length_stats": {}}
    m = _SUCCESS_RE.search(text)
    if m:
        n, total = int(m.group(1)), int(m.group(2))
        result["success_rate"] = n / total if total > 0 else 0.0
        result["success_episodes"] = n
        result["total_episodes"] = total
    for line in text.splitlines():
        m = _PER_COND_RE.match(line)
        if m:
            n, total = int(m.group(2)), int(m.group(3))
            result["per_condition_pass_rate"][m.group(1)] = (
                n / total if total > 0 else 0.0
            )
    m = _EP_LEN_RE.search(text)
    if m:
        result["ep_length_stats"] = {
            "mean": float(m.group(1)),
            "min": int(m.group(2)),
            "max": int(m.group(3)),
        }
    return result


def _parse_ppo_curve(train_log: Path) -> dict:
    """Summarize PPO training curve from train log."""
    if not train_log.exists():
        return {}
    iters = []
    for line in train_log.read_text().splitlines():
        m = _PPO_ITER_RE.match(line)
        if m:
            iters.append({
                "iter": int(m.group(1)),
                "step": int(m.group(2)),
                "ep_return": float(m.group(4)),
                "ep_len": float(m.group(5)),
                "v_loss": float(m.group(7)),
                "ent": float(m.group(8)),
                "sps": int(m.group(11)),
            })
    if not iters:
        return {}
    return {
        "n_iters": len(iters),
        "final_step": iters[-1]["step"],
        "first": iters[0],
        "last": iters[-1],
        "max_ep_return": max(i["ep_return"] for i in iters),
        "min_ep_return": min(i["ep_return"] for i in iters),
        "ent_start": iters[0]["ent"],
        "ent_end": iters[-1]["ent"],
        "v_loss_start": iters[0]["v_loss"],
        "v_loss_end": iters[-1]["v_loss"],
        "avg_sps": sum(i["sps"] for i in iters) / len(iters),
    }


def _suggest_next_step(success_rate: float, per_cond: dict) -> str:
    """Heuristic next-step suggestion based on Phase 1 결정사항 7 / 8
    and the dense_reward_design rollback rules."""
    if success_rate is None:
        return "unknown — eval log not parsed; investigate eval output manually"
    sr_pct = success_rate * 100
    lift = per_cond.get("lift_history", 0) * 100
    release = per_cond.get("release_retreat", 0) * 100
    if sr_pct >= 70:
        return (
            f"ADVANCE: success {sr_pct:.1f}% >= 70% threshold. "
            f"Advance to curriculum stage 1 (10cm cube spawn) via "
            f"CurriculumCfg or env_cfg.randomization update."
        )
    if sr_pct >= 50:
        return (
            f"PARTIAL: success {sr_pct:.1f}% in 50-70% range. "
            f"Try another PPO pass (BC + dense ON, more steps) or tune "
            f"DenseRewardCfg weights. lift={lift:.0f}% release={release:.0f}%."
        )
    if sr_pct >= 30:
        return (
            f"WEAK: success {sr_pct:.1f}% in 30-50% range. "
            f"Inspect videos for hover/short-place patterns; consider "
            f"raising w_released_bonus or shortening decay_steps. "
            f"lift={lift:.0f}% release={release:.0f}%."
        )
    return (
        f"FAIL: success {sr_pct:.1f}% < 30% threshold. "
        f"ROLLBACK candidates: revert dense, try DAgger or action "
        f"chunking. lift={lift:.0f}% release={release:.0f}% — if "
        f"lift << BC baseline (36%) dense broke the policy "
        f"(per docs/dense_reward_design.md rollback rule)."
    )


def _write_status_report(
    run_dir: Path,
    eval_result: dict,
    ppo_curve: dict,
    train_log: Path,
    eval_log: Path,
    ckpt: Path | None,
) -> Path:
    """Write the single-file dashboard the user reads on wake-up."""
    out = run_dir / "status_report.txt"
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"PHASE 1 OVERNIGHT REPORT — run={run_dir.name}")
    lines.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)

    # ---- Eval result ----
    lines.append("\n[EVAL RESULT]")
    if eval_result.get("success_rate") is None:
        lines.append("  (eval did not produce a SUCCESS RATE line — check eval log)")
    else:
        sr = eval_result["success_rate"] * 100
        lines.append(
            f"  Success rate: {eval_result['success_episodes']}/"
            f"{eval_result['total_episodes']} = {sr:.1f}%"
        )
        lines.append("  Per-condition pass rate:")
        for k, v in sorted(eval_result["per_condition_pass_rate"].items()):
            lines.append(f"    {k}: {v*100:.1f}%")
        if eval_result["ep_length_stats"]:
            s = eval_result["ep_length_stats"]
            lines.append(
                f"  ep_length: mean={s['mean']:.1f} min={s['min']} max={s['max']}"
            )

    # ---- PPO training curve ----
    lines.append("\n[PPO TRAINING CURVE]")
    if not ppo_curve:
        lines.append("  (no [PPO] iter lines found in train log)")
    else:
        lines.append(f"  iters={ppo_curve['n_iters']} final_step={ppo_curve['final_step']}")
        lines.append(
            f"  ep_return: min={ppo_curve['min_ep_return']:.3f} "
            f"max={ppo_curve['max_ep_return']:.3f} "
            f"final={ppo_curve['last']['ep_return']:.3f}"
        )
        lines.append(
            f"  ent: {ppo_curve['ent_start']:.3f} -> {ppo_curve['ent_end']:.3f}"
        )
        lines.append(
            f"  v_loss: {ppo_curve['v_loss_start']:.4f} -> {ppo_curve['v_loss_end']:.4f}"
        )
        lines.append(f"  avg sps: {ppo_curve['avg_sps']:.0f}")

    # ---- Files of interest ----
    lines.append("\n[FILES]")
    if ckpt is not None:
        lines.append(f"  ckpt: {ckpt}")
    lines.append(f"  train log: {train_log}")
    lines.append(f"  eval log: {eval_log}")
    videos = sorted((run_dir / "videos").glob("rollout_step_*.mp4")) if (run_dir / "videos").exists() else []
    if videos:
        lines.append(f"  videos ({len(videos)}):")
        for v in videos:
            lines.append(f"    {v.name}  ({v.stat().st_size // 1024} KB)")

    # ---- Suggested next step ----
    lines.append("\n[SUGGESTED NEXT STEP]")
    lines.append(
        "  " + _suggest_next_step(
            eval_result.get("success_rate"),
            eval_result.get("per_condition_pass_rate", {}),
        )
    )

    lines.append("\n" + "=" * 72)
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default="stage0_ppo_a",
                   help="Subdir under runs/ that holds ppo.pt and videos/")
    p.add_argument("--log-dir", type=Path, default=Path("runs"))
    p.add_argument("--train-log", type=Path, default=Path("/tmp/ppo_a.log"),
                   help="train_ppo.py stdout/stderr log (for curve summary)")
    p.add_argument("--eval-log", type=Path, default=None,
                   help="Where to write eval log (default: /tmp/<run-name>_eval.log)")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--skip-wait", action="store_true",
                   help="Skip PPO wait (use when ckpt already exists)")
    args = p.parse_args()

    run_dir = args.log_dir / args.run_name
    eval_log = args.eval_log or Path(f"/tmp/{args.run_name}_eval.log")

    _log(f"watcher start run={run_dir} train_log={args.train_log} eval_log={eval_log}")

    # ---- 1) wait for PPO ----
    if args.skip_wait:
        _log("--skip-wait set; assuming ckpt is ready")
        ckpt = run_dir / "ppo.pt"
        if not ckpt.exists():
            _log(f"ckpt missing at {ckpt} — abort")
            return 1
    else:
        ckpt = _wait_for_ppo_end(run_dir, poll_s=args.poll_seconds)
        if ckpt is None:
            _log("PPO ended without ckpt — abort eval, still writing report")

    # Even if ckpt missing, write partial report so morning user sees the failure mode.
    eval_result: dict = {}
    if ckpt is not None:
        eval_env = os.environ.copy()
        eval_env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        eval_env.setdefault("PRIVACY_CONSENT", "Y")
        eval_env.setdefault("CUDA_VISIBLE_DEVICES", "1")
        rc = _run_eval(ckpt, args.episodes, eval_log, eval_env)
        eval_result = _parse_eval(eval_log)
        _log(
            f"eval rc={rc} success_rate="
            f"{eval_result.get('success_rate')!r}"
        )

        # Dump structured eval result.
        json_path = run_dir / "eval_result.json"
        json_path.write_text(json.dumps(eval_result, indent=2))
        _log(f"wrote {json_path}")

        # Post-eval: kill any zombie if eval_policy.py hangs in sim_app.close().
        time.sleep(60)
        _kill_zombies(eval_log)

    # ---- 4) status report ----
    ppo_curve = _parse_ppo_curve(args.train_log)
    report = _write_status_report(
        run_dir=run_dir,
        eval_result=eval_result,
        ppo_curve=ppo_curve,
        train_log=args.train_log,
        eval_log=eval_log,
        ckpt=ckpt,
    )
    _log(f"wrote {report}")
    _log("watcher done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
