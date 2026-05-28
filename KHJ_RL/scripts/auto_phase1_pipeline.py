"""Auto pipeline: wait for collect+PPO → BC retrain → eval → PPO retrain.

Bundles the next-step plan into one runnable so the operator can launch
it inside tmux and walk away. Steps:

  1. Wait for ``collect_demos.py`` and ``train_ppo.py`` processes to exit
     (so GPU + carb singleton are free).
  2. Run BC retrain over all ``--demo-dirs`` merged into one buffer.
  3. Eval the BC ckpt (deterministic, ``--eval-episodes`` rollouts).
  4. Parse the success rate from the eval log.
  5. If rate >= ``--success-threshold``: launch PPO retrain (D-option
     hyperparams baked in — ent_coef=0, lr=5e-5, critic_warmup=5).
     Otherwise: bail with a pointer at ``docs/dense_reward_design.md``.

Designed to run inside tmux so SSH disconnects don't kill the pipeline.

Usage::

    tmux new-session -d -s pipeline \\
        "bash -lc 'CUDA_VISIBLE_DEVICES=1 python scripts/auto_phase1_pipeline.py \\
            --demo-dirs runs/demos/stage0 runs/demos/stage0_extra \\
            --bc-run-name stage0_bc_3k \\
            --ppo-run-name stage0_ppo_d2 \\
            2>&1 | tee /tmp/pipeline.log; echo DONE; exec bash'"
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def _alive(pattern: str) -> bool:
    """True if any process matches ``pgrep -f pattern``."""
    return bool(
        subprocess.run(["pgrep", "-f", pattern], capture_output=True)
        .stdout.strip()
    )


def _wait_done(pattern: str, label: str, poll_s: int = 60) -> None:
    """Block until no process matches ``pattern``. Idempotent — returns
    immediately if nothing was alive to begin with."""
    if not _alive(pattern):
        print(f"[pipeline] {label}: already done (or never started)", flush=True)
        return
    print(f"[pipeline] waiting for {label} ({pattern}) to finish...", flush=True)
    while _alive(pattern):
        time.sleep(poll_s)
    print(f"[pipeline] {label} done", flush=True)


_RATE_RE = re.compile(r"\[eval\] SUCCESS RATE:\s+(\d+)/(\d+)")


def _parse_success_rate(log_text: str) -> float | None:
    """Returns success rate ∈ [0, 1] or None if line not found."""
    m = _RATE_RE.search(log_text)
    if not m:
        return None
    n, total = int(m.group(1)), int(m.group(2))
    return n / total if total > 0 else 0.0


def _run(cmd: list[str], log_path: Path | None = None) -> int:
    """Run command with shared env. If log_path given, tee stdout+stderr to it."""
    pretty = " ".join(cmd)
    print(f"[pipeline] $ {pretty}", flush=True)
    if log_path is None:
        return subprocess.run(cmd).returncode
    # tee: subprocess writes to file AND we print summary on completion.
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"[pipeline] {cmd[0]} exited rc={proc.returncode} log={log_path}", flush=True)
    return proc.returncode


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Auto pipeline: wait for collect+PPO -> BC retrain -> eval -> "
            "PPO retrain. See module docstring for full step list."
        ),
    )
    p.add_argument("--demo-dirs", nargs="+", required=True, type=Path,
                   help="One or more demo directories merged for BC retrain.")
    p.add_argument("--bc-run-name", default="stage0_bc_3k")
    p.add_argument("--bc-epochs", type=int, default=100,
                   help="More epochs than the 1k-demo run (50) since the "
                        "merged buffer is larger and BC under-fitting is the "
                        "main suspect from the 15 percent baseline.")
    p.add_argument("--bc-batch-size", type=int, default=256)
    p.add_argument("--bc-lr", type=float, default=3e-4)
    p.add_argument("--ppo-run-name", default="stage0_ppo_d2")
    p.add_argument("--ppo-total-steps", type=int, default=500_000)
    p.add_argument("--ppo-num-steps", type=int, default=2048)
    p.add_argument("--ppo-lr", type=float, default=5e-5,
                   help="D-option hyperparam: low LR protects BC weights.")
    p.add_argument("--ppo-ent-coef", type=float, default=0.0,
                   help="D-option hyperparam: no entropy bonus, freeze "
                        "actor_logstd from drifting upward.")
    p.add_argument("--ppo-critic-warmup", type=int, default=5,
                   help="D-option hyperparam: shorten critic warmup so "
                        "critic doesn't collapse to a 0-constant.")
    p.add_argument("--ppo-video-every-steps", type=int, default=0,
                   help="0 = let pipeline pick (total_steps/8). Override "
                        "if you want a specific cadence.")
    p.add_argument("--success-threshold", type=float, default=0.30,
                   help="Min BC eval success rate (0..1) to trigger PPO "
                        "retrain. 0.30 is the codex-suggested floor; below "
                        "this, PPO still won't catch sparse signal often "
                        "enough.")
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--log-dir", type=Path, default=Path("runs"))
    p.add_argument("--skip-wait", action="store_true",
                   help="Skip waiting for existing collect_extra / PPO D "
                        "processes (use if pipeline runs standalone).")
    p.add_argument("--skip-ppo", action="store_true",
                   help="Run BC + eval only; never launch PPO regardless "
                        "of success rate. Use for debugging the pipeline.")
    args = p.parse_args()

    python = sys.executable

    # ---- 1) wait for prior work -----------------------------------------
    if not args.skip_wait:
        _wait_done("collect_demos.py", "collect_extra")
        _wait_done("train_ppo.py", "PPO D (or other train_ppo runs)")

    # Sanity: at least one demo dir must contain NPZs (cheap pre-check
    # so we don't burn time booting BC on an empty buffer).
    total_npz = 0
    for d in args.demo_dirs:
        n = len(list(Path(d).glob("*.npz")))
        print(f"[pipeline] {d}: {n} NPZ files", flush=True)
        total_npz += n
    if total_npz == 0:
        print("[pipeline] ERROR: no NPZ files across all --demo-dirs", flush=True)
        return 1

    # ---- 2) BC retrain --------------------------------------------------
    bc_cmd = [
        python, "scripts/train_bc.py",
        "--demo-dir", *[str(d) for d in args.demo_dirs],
        "--run-name", args.bc_run_name,
        "--epochs", str(args.bc_epochs),
        "--batch-size", str(args.bc_batch_size),
        "--lr", str(args.bc_lr),
        "--log-dir", str(args.log_dir),
    ]
    rc = _run(bc_cmd)
    if rc != 0:
        print(f"[pipeline] BC retrain failed (rc={rc}); aborting", flush=True)
        return rc

    bc_ckpt = args.log_dir / args.bc_run_name / "bc.pt"
    if not bc_ckpt.exists():
        print(f"[pipeline] ERROR: BC ckpt missing at {bc_ckpt}", flush=True)
        return 1

    # ---- 3) BC eval -----------------------------------------------------
    # eval_policy.py boots Isaac Sim itself; we set the env vars it
    # expects via subprocess env (not via `env ...` prefix) so the log
    # file captures clean Python output.
    eval_env = os.environ.copy()
    eval_env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    eval_env["PRIVACY_CONSENT"] = "Y"
    eval_env.setdefault("CUDA_VISIBLE_DEVICES", "1")

    eval_log = Path(f"/tmp/{args.bc_run_name}_eval.log")
    eval_cmd = [
        python, "scripts/eval_policy.py",
        "--ckpt", str(bc_ckpt),
        "--episodes", str(args.eval_episodes),
    ]
    print(f"[pipeline] $ {' '.join(eval_cmd)} > {eval_log}", flush=True)
    with eval_log.open("w") as f:
        proc = subprocess.run(eval_cmd, stdout=f, stderr=subprocess.STDOUT, env=eval_env)
    print(f"[pipeline] eval exited rc={proc.returncode} log={eval_log}", flush=True)
    if proc.returncode != 0:
        print(f"[pipeline] eval failed; aborting", flush=True)
        return proc.returncode

    log_text = eval_log.read_text()
    rate = _parse_success_rate(log_text)
    if rate is None:
        print(
            f"[pipeline] ERROR: could not parse SUCCESS RATE from {eval_log}",
            flush=True,
        )
        return 1
    print(f"[pipeline] BC eval success rate: {rate*100:.1f}%", flush=True)

    # ---- 4) decide ------------------------------------------------------
    if args.skip_ppo:
        print(f"[pipeline] --skip-ppo set; stopping after BC eval", flush=True)
        return 0
    if rate < args.success_threshold:
        print(
            f"[pipeline] BC {rate*100:.1f}% < threshold "
            f"{args.success_threshold*100:.0f}% — PPO retrain skipped.\n"
            f"            Consider A option (dense shaping); see "
            f"docs/dense_reward_design.md",
            flush=True,
        )
        return 0

    # ---- 5) PPO retrain -------------------------------------------------
    video_every = args.ppo_video_every_steps or max(
        1, args.ppo_total_steps // 8
    )
    ppo_cmd = [
        python, "scripts/train_ppo.py",
        "--bc-ckpt", str(bc_ckpt),
        "--run-name", args.ppo_run_name,
        "--total-steps", str(args.ppo_total_steps),
        "--num-steps", str(args.ppo_num_steps),
        "--lr", str(args.ppo_lr),
        "--ent-coef", str(args.ppo_ent_coef),
        "--critic-warmup-iters", str(args.ppo_critic_warmup),
        "--video-every-steps", str(video_every),
        "--video-rollout-episodes", "2",
        "--log-dir", str(args.log_dir),
    ]
    rc = _run(ppo_cmd)
    print(f"[pipeline] PPO retrain done (rc={rc})", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
