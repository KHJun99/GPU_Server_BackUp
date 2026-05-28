"""Phase 1 result analysis dashboard.

Bundles three things into one markdown report so the operator can
diagnose a PPO run without clicking through 10 separate mp4 files:

  1. **Eval per-episode breakdown** — parsed from eval log, every
     episode's 5-condition pass/fail in a single table. Reveals
     *where the failures cluster* (e.g. lift always 0 → policy
     dropped grasping entirely).

  2. **PPO training curve summary** — final iter snapshot + curve
     extremes (max/min ep_return, ent start/end, v_loss start/end).

  3. **Video thumbnails** — for each rollout mp4 in run_dir/videos/,
     extract first/middle/last frames as PNG so the report
     embeds them inline. Reveals policy *behaviour drift* (BC PnP
     → hover → press) across training without opening any media
     player.

Output: ``runs/<run_name>/result_analysis.md`` + a ``thumbnails/``
subdir with the PNG frames.

Usage::

    python scripts/analyze_phase1_results.py --run-name stage0_ppo_a
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


_EP_RE = re.compile(
    r"\[eval\] ep=(\d+) steps=(\d+) term=(\w+) trunc=(\w+) "
    r"success=(\w+) per_cond=\{([^}]+)\}"
)
_PPO_RE = re.compile(
    r"^\[PPO\] iter=(\d+) step=(\d+) frozen_actor=(\w+) "
    r"ep_return_mean=(-?[\d.]+) ep_len_mean=([\d.]+) "
    r"pi_loss=(-?[\d.]+) v_loss=(-?[\d.]+) ent=([\d.]+) "
    r"kl=(-?[\d.]+) clipfrac=([\d.]+) sps=(\d+)"
)


def _parse_eval_episodes(eval_log: Path) -> list[dict]:
    """Return list of {ep, steps, term, trunc, success, per_cond}."""
    if not eval_log.exists():
        return []
    out: list[dict] = []
    for line in eval_log.read_text().splitlines():
        m = _EP_RE.search(line)
        if not m:
            continue
        per_cond_str = m.group(6)
        per_cond: dict[str, int] = {}
        for pair in per_cond_str.split(","):
            if ":" not in pair:
                continue
            k, v = pair.split(":", 1)
            k = k.strip().strip("'\"")
            v = v.strip()
            try:
                per_cond[k] = int(v)
            except ValueError:
                pass
        out.append({
            "ep": int(m.group(1)),
            "steps": int(m.group(2)),
            "term": m.group(3) == "True",
            "trunc": m.group(4) == "True",
            "success": m.group(5) == "True",
            "per_cond": per_cond,
        })
    return out


def _parse_ppo_iters(train_log: Path) -> list[dict]:
    if not train_log.exists():
        return []
    out: list[dict] = []
    for line in train_log.read_text().splitlines():
        m = _PPO_RE.match(line)
        if not m:
            continue
        out.append({
            "iter": int(m.group(1)),
            "step": int(m.group(2)),
            "frozen_actor": m.group(3) == "True",
            "ep_return": float(m.group(4)),
            "ep_len": float(m.group(5)),
            "pi_loss": float(m.group(6)),
            "v_loss": float(m.group(7)),
            "ent": float(m.group(8)),
            "kl": float(m.group(9)),
            "clipfrac": float(m.group(10)),
            "sps": int(m.group(11)),
        })
    return out


def _extract_video_thumbnails(
    video_dir: Path, out_dir: Path
) -> list[tuple[Path, list[Path]]]:
    """For each mp4 in video_dir, extract first/middle/last frame.

    Returns ``[(video_path, [first_png, middle_png, last_png]), ...]``.
    Frames written to ``out_dir/<stem>_first.png`` etc.

    Uses imageio.v2 (already a project dep via video_sampler) so we
    don't need a system ffmpeg binary.
    """
    if not video_dir.exists():
        return []

    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[analyze] imageio not available — skipping thumbnails", flush=True)
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[Path, list[Path]]] = []
    for mp4 in sorted(video_dir.glob("rollout_step_*.mp4")):
        try:
            reader = imageio.get_reader(str(mp4))
            frames = []
            for frame in reader:
                frames.append(frame)
            reader.close()
        except Exception as e:
            print(f"[analyze] failed to read {mp4.name}: {e}", flush=True)
            continue
        if not frames:
            continue
        n = len(frames)
        picks = [(0, "first"), (n // 2, "middle"), (n - 1, "last")]
        png_paths: list[Path] = []
        for idx, label in picks:
            png = out_dir / f"{mp4.stem}_{label}.png"
            imageio.imwrite(str(png), frames[idx])
            png_paths.append(png)
        results.append((mp4, png_paths))
        print(f"[analyze] {mp4.name}: {n} frames -> 3 thumbs", flush=True)
    return results


def _failure_pattern(episodes: list[dict]) -> dict[str, dict]:
    """For each condition, compute pass rate AND co-occurrence with success.

    Returns dict[cond_name] -> {pass_rate, pass_count, fail_when_others_pass}.

    ``fail_when_others_pass`` = #episodes where every OTHER condition was
    True but this one was False. Identifies the bottleneck condition.
    """
    if not episodes:
        return {}
    cond_keys: set[str] = set()
    for ep in episodes:
        cond_keys.update(ep["per_cond"].keys())
    total = len(episodes)
    out: dict[str, dict] = {}
    for k in sorted(cond_keys):
        pass_count = sum(1 for ep in episodes if ep["per_cond"].get(k, 0) == 1)
        bottleneck_count = sum(
            1
            for ep in episodes
            if ep["per_cond"].get(k, 0) == 0
            and all(ep["per_cond"].get(o, 0) == 1 for o in cond_keys if o != k)
        )
        out[k] = {
            "pass_rate": pass_count / total if total else 0.0,
            "pass_count": pass_count,
            "fail_when_others_pass": bottleneck_count,
        }
    return out


def _build_report(
    run_dir: Path,
    episodes: list[dict],
    ppo_iters: list[dict],
    thumbnails: list[tuple[Path, list[Path]]],
) -> str:
    L: list[str] = []
    L.append(f"# Phase 1 result analysis — {run_dir.name}")
    L.append("")

    # ---- Eval overview ----
    L.append("## Eval — overall")
    if not episodes:
        L.append("_(no eval episodes parsed)_")
    else:
        n_total = len(episodes)
        n_success = sum(1 for ep in episodes if ep["success"])
        avg_steps = sum(ep["steps"] for ep in episodes) / n_total
        L.append(f"- success: **{n_success}/{n_total} = {n_success / n_total * 100:.1f}%**")
        L.append(f"- ep_length mean: {avg_steps:.1f}")
        L.append("")

        # ---- Per-condition pass / bottleneck ----
        L.append("### Per-condition pass rate + bottleneck count")
        L.append("")
        L.append("`fail_when_others_pass` = episodes where every OTHER condition was True")
        L.append("but this one wasn't. **High value = this condition is the bottleneck**.")
        L.append("")
        L.append("| condition | pass rate | bottleneck count |")
        L.append("|---|---|---|")
        patt = _failure_pattern(episodes)
        for k, v in patt.items():
            L.append(
                f"| {k} | {v['pass_count']}/{n_total} = "
                f"{v['pass_rate'] * 100:.1f}% | {v['fail_when_others_pass']} |"
            )
        L.append("")

        # ---- Episode-by-episode trail (compact) ----
        L.append("### Episode trail (s = success, x = fail)")
        L.append("")
        cond_keys = sorted(patt.keys())
        L.append("```")
        L.append("ep   ok  steps  " + "  ".join(f"{k[:3]:>3}" for k in cond_keys))
        for ep in episodes:
            mark = "s" if ep["success"] else "x"
            vals = "  ".join(
                f"{ep['per_cond'].get(k, 0):>3}" for k in cond_keys
            )
            L.append(f"{ep['ep']:>3}   {mark}   {ep['steps']:>3}  {vals}")
        L.append("```")
        L.append("")

    # ---- PPO training curve ----
    L.append("## PPO training curve")
    if not ppo_iters:
        L.append("_(no PPO iter lines parsed)_")
    else:
        first, last = ppo_iters[0], ppo_iters[-1]
        L.append(f"- iters: {len(ppo_iters)} (final_step={last['step']})")
        L.append(
            f"- ep_return: min={min(i['ep_return'] for i in ppo_iters):.4f} "
            f"max={max(i['ep_return'] for i in ppo_iters):.4f} "
            f"final={last['ep_return']:.4f}"
        )
        L.append(f"- ent: {first['ent']:.4f} → {last['ent']:.4f}")
        L.append(f"- v_loss: {first['v_loss']:.4f} → {last['v_loss']:.4f}")
        L.append(f"- avg sps: {np.mean([i['sps'] for i in ppo_iters]):.1f}")
        L.append("")

        L.append("### Curve (every Nth iter)")
        L.append("```")
        L.append(
            f"{'iter':>4} {'step':>7} {'frozen':>6} {'ep_ret':>8} "
            f"{'ent':>6} {'v_loss':>7}"
        )
        n = len(ppo_iters)
        stride = max(1, n // 30)  # ~30 rows
        for i in ppo_iters[::stride]:
            frozen = "Y" if i["frozen_actor"] else "N"
            L.append(
                f"{i['iter']:>4} {i['step']:>7} {frozen:>6} "
                f"{i['ep_return']:>8.4f} {i['ent']:>6.4f} {i['v_loss']:>7.4f}"
            )
        if ppo_iters[-1]["iter"] % stride != 0:
            i = ppo_iters[-1]
            frozen = "Y" if i["frozen_actor"] else "N"
            L.append(
                f"{i['iter']:>4} {i['step']:>7} {frozen:>6} "
                f"{i['ep_return']:>8.4f} {i['ent']:>6.4f} {i['v_loss']:>7.4f}"
            )
        L.append("```")
        L.append("")

    # ---- Video thumbnails ----
    L.append("## Video thumbnails")
    if not thumbnails:
        L.append("_(no videos / imageio unavailable)_")
    else:
        L.append(
            "First / middle / last frame of each rollout. Compare across "
            "step to see policy behaviour drift."
        )
        L.append("")
        for video, pngs in thumbnails:
            L.append(f"### `{video.name}`")
            row = " | ".join(
                f"![{p.stem}]({p.relative_to(video.parent.parent)})" for p in pngs
            )
            L.append(f"| first | middle | last |")
            L.append(f"|---|---|---|")
            L.append(f"| {row} |")
            L.append("")

    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True,
                   help="Subdir under runs/ (e.g. stage0_ppo_a)")
    p.add_argument("--log-dir", type=Path, default=Path("runs"))
    p.add_argument("--train-log", type=Path, default=None,
                   help="train_ppo.py log; defaults to /tmp/<run_name>.log "
                        "or /tmp/ppo_a.log fallback")
    p.add_argument("--eval-log", type=Path, default=None,
                   help="eval_policy.py log; defaults to "
                        "/tmp/<run_name>_eval.log")
    args = p.parse_args()

    run_dir = args.log_dir / args.run_name
    train_log = args.train_log or Path(f"/tmp/ppo_a.log")
    eval_log = args.eval_log or Path(f"/tmp/{args.run_name}_eval.log")

    print(f"[analyze] run_dir={run_dir}", flush=True)
    print(f"[analyze] train_log={train_log}", flush=True)
    print(f"[analyze] eval_log={eval_log}", flush=True)

    episodes = _parse_eval_episodes(eval_log)
    print(f"[analyze] parsed {len(episodes)} eval episodes", flush=True)
    ppo_iters = _parse_ppo_iters(train_log)
    print(f"[analyze] parsed {len(ppo_iters)} PPO iters", flush=True)

    thumbnails = _extract_video_thumbnails(
        video_dir=run_dir / "videos",
        out_dir=run_dir / "thumbnails",
    )

    report = _build_report(run_dir, episodes, ppo_iters, thumbnails)
    out_path = run_dir / "result_analysis.md"
    out_path.write_text(report)
    print(f"[analyze] wrote {out_path} ({len(report)} chars)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
