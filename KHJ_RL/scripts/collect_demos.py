"""Collect oracle MP demos for BC pretrain.

Runs the 6-phase MotionPlanningOracle against CubeLiftEnv, dropping each
successful episode into ``DemoBuffer`` under ``runs/demos/{stage}/``.
Failed episodes are still counted in the log but not saved.

Method B post-M2 note: actions stored here are 4-D
``[Δx, Δy, Δz, g_normalized]`` (env-side IK converts to joint targets).
Pre-M2 6-D joint-delta demos are 폐기됨. Run M3.3 calibration first so
``cfg.EEControlCfg.ee_delta_max_m`` reflects the oracle's measured natural
speed, otherwise the clipped EE-delta distribution may differ from what
M5 BC expects to imitate.

Usage:

    # M4 full collect (post-M3.3 calibration):
    OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y CUDA_VISIBLE_DEVICES=1 \\
        python scripts/collect_demos.py --episodes 2000 --stage stage0 \\
            --max-success 1000 --curriculum-stage 0

    # smoke test:
    OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y CUDA_VISIBLE_DEVICES=1 \\
        python scripts/collect_demos.py --episodes 5 --stage stage0
"""

from __future__ import annotations

import argparse
import sys

import pinocchio  # noqa: F401, E402  (pre-AppLauncher Assimp ABI fix)

from isaaclab.app import AppLauncher  # noqa: E402  (must be first; isaaclab bootstrap)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--stage", type=str, default="stage0")
    parser.add_argument("--out-root", type=str, default="runs/demos")
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Added to cfg.seed when stamping each episode's record metadata.",
    )
    parser.add_argument(
        "--curriculum-stage",
        type=int,
        default=0,
        help="Index into CurriculumCfg.side_length_m. G+ adds a 2cm narrow "
             "start at index 0, so the layout is 0=2cm, 1=5cm, 2=10cm, "
             "3=15cm, 4=20cm. Determines the cube xy-spawn radius.",
    )
    parser.add_argument(
        "--max-success",
        type=int,
        default=0,
        help="If > 0, stop early once this many successful demos have been "
             "written. Useful for time-bounded collects on slow stages "
             "(stage 1 = ~22 s/ep so 1000 attempts ≈ 6 h).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sim_app = AppLauncher(headless=True, enable_cameras=False).app

    # Imports happen *after* AppLauncher boot (carb bootstrap dependency).
    from khj_rl.data import DemoBuffer, EpisodeRecord
    from khj_rl.envs import CubeLiftEnv, CubeLiftEnvCfg
    from khj_rl.envs.cube_lift.oracle import MotionPlanningOracle

    cfg = CubeLiftEnvCfg()
    cfg.curriculum.current_stage_idx = int(args.curriculum_stage)
    # Demos always collect full PnP (5-condition AND) trajectories so they
    # remain reusable across G+ task levels — level-0 demos would otherwise
    # be truncated by the level-0 force_terminate guard and lose
    # place/release/retreat segments.
    cfg.curriculum.task_level = 2
    env = CubeLiftEnv(cfg)
    oracle = MotionPlanningOracle(env)
    buf = DemoBuffer(args.out_root)

    n_success = 0
    n_total = 0
    for ep_id in range(args.episodes):
        obs = env.reset()
        oracle.reset(obs, {})
        rec = EpisodeRecord()
        info: dict = {}
        success = False
        terminated = False

        last_term = False
        last_trunc = False
        while True:
            action = oracle.act(obs, info)
            next_obs, reward, term, trunc, info = env.step(action)
            rec.add(obs, action, reward)
            obs = next_obs
            last_term = bool(term)
            last_trunc = bool(trunc)
            if term:
                terminated = True
                success = True
                break
            if trunc:
                break

        rec.finish(terminated=terminated, success_flag=success)
        n_total += 1
        if success:
            path = buf.write(
                args.stage,
                ep_id,
                rec,
                cfg,
                seed=cfg.seed + args.seed_offset + ep_id,
            )
            n_success += 1
            print(f"[EP {ep_id:03d}] success ({len(rec)} steps) -> {path.name}", flush=True)
            if args.max_success > 0 and n_success >= args.max_success:
                print(
                    f"[collect] reached --max-success={args.max_success}; "
                    f"stopping after {n_total} attempts",
                    flush=True,
                )
                break
        else:
            per = info.get("success_per_condition", {}) if info else {}
            per_str = " ".join(f"{k}={int(v)}" for k, v in per.items())
            print(
                f"[EP {ep_id:03d}] fail (steps={len(rec)}, final phase={oracle.phase_name}, "
                f"term={last_term} trunc={last_trunc}) per_cond[{per_str}]",
                flush=True,
            )

    print(f"\n[SUMMARY] collected {n_success}/{n_total} success episodes", flush=True)
    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
