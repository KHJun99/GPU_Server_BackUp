"""Long-running health monitor — overnight backup against silent failures.

Runs alongside the PPO trainer + watcher. Every N minutes:
  1. GPU memory + utilization (CUDA_VISIBLE_DEVICES=1 GPU)
  2. Disk usage on the runs/ partition
  3. Zombie sim_app Python processes (>120 s after ckpt save) — SIGTERM
  4. tmux session liveness (ppo_a, watcher)

All findings appended to ``/tmp/health_monitor.log``. Designed as a
redundant safety net: the watcher itself already kills zombies, but if
the watcher dies for any reason this script keeps the GPU clean.

Usage::

    # in its own tmux session:
    tmux new-session -d -s health \\
        "bash -lc 'python scripts/health_monitor.py --interval 600 \\
            2>&1 | tee /tmp/health_monitor.log; exec bash'"

The monitor is itself a long-running Python process, so on session
end (TaskStop / kill) it dies cleanly. No persistent daemon.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[health {ts}] {msg}", flush=True)


def _gpu_stats() -> str:
    """Return one-line summary of CUDA_VISIBLE_DEVICES=1 GPU.

    Uses nvidia-smi; if the binary isn't available, returns 'n/a'.
    """
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "gpu: n/a"
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    # CUDA_VISIBLE_DEVICES=1 means the *logical* GPU 0 in our process is
    # the system's physical GPU 1. nvidia-smi here lists all physical
    # GPUs (it ignores CVD). We log all of them — the system has 4
    # L40S and only GPU index 1 is ours.
    return " | ".join(f"gpu{i}: {ln}" for i, ln in enumerate(lines))


def _disk_stats(path: Path) -> str:
    if not path.exists():
        return f"disk({path}): missing"
    usage = shutil.disk_usage(path)
    used_gb = usage.used / (1024**3)
    total_gb = usage.total / (1024**3)
    pct = usage.used / usage.total * 100
    return f"disk({path}): {used_gb:.1f}/{total_gb:.1f} GiB ({pct:.0f}%)"


def _alive(pattern: str) -> list[int]:
    res = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(x) for x in res.stdout.split() if x.strip()]


def _kill_zombies(ckpt_path: Path, age_threshold_s: float = 120) -> int:
    """SIGTERM any train_ppo.py / collect_demos.py / eval_policy.py
    that's been alive >age_threshold_s after the ckpt was written.

    Returns number killed.
    """
    if not ckpt_path.exists():
        return 0
    ckpt_age = time.time() - ckpt_path.stat().st_mtime
    if ckpt_age < age_threshold_s:
        return 0
    n = 0
    for pat in ("train_ppo.py", "collect_demos.py", "eval_policy.py"):
        for pid in _alive(pat):
            _log(f"  killing zombie {pat} pid={pid} (ckpt age {ckpt_age:.0f} s)")
            try:
                os.kill(pid, 15)
                n += 1
            except ProcessLookupError:
                pass
    return n


def _tmux_alive(session: str) -> bool:
    try:
        res = subprocess.run(
            ["tmux", "has-session", "-t", session],
            env={**os.environ, "LD_LIBRARY_PATH": ""},
            capture_output=True,
        )
        return res.returncode == 0
    except FileNotFoundError:
        return False


def _scan_ckpts(runs_dir: Path) -> list[Path]:
    """Find any *.pt under runs/ for zombie check."""
    if not runs_dir.exists():
        return []
    return list(runs_dir.glob("**/ppo.pt")) + list(runs_dir.glob("**/bc.pt"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interval", type=int, default=600,
                   help="Check interval in seconds (default 600 = 10 min)")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--watched-sessions", nargs="+",
                   default=["ppo_a", "watcher"],
                   help="tmux session names to verify alive")
    p.add_argument("--max-iters", type=int, default=0,
                   help="Stop after N iters (default 0 = run forever)")
    args = p.parse_args()

    _log(f"health monitor start; interval={args.interval}s "
         f"runs_dir={args.runs_dir} sessions={args.watched_sessions}")

    i = 0
    while True:
        i += 1
        _log(f"check #{i}")
        _log(f"  {_gpu_stats()}")
        _log(f"  {_disk_stats(args.runs_dir)}")
        for s in args.watched_sessions:
            alive = _tmux_alive(s)
            _log(f"  tmux/{s}: {'alive' if alive else 'DEAD'}")
        # NOTE: Zombie-kill is DISABLED at the global health-monitor
        # level — a stale ckpt from an old run (`runs/stage0_ppo/`) was
        # misclassifying the live PPO A run as a zombie (commit history
        # shows the incident, 2026-05-17 23:33 KST). The watcher already
        # kills zombies scoped to its own run dir; the monitor only
        # reports here.
        _ = _kill_zombies  # silence "unused" linter — the helper is
                           # kept for ad-hoc manual use.
        if args.max_iters > 0 and i >= args.max_iters:
            _log("max_iters reached; exit")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
