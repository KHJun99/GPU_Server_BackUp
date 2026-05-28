"""Measure reach accuracy of a BC or SAC policy.

For each rollout episode, records the minimum finger-to-cube xy distance
the policy ever achieves during the episode. Then bins episodes by cube
spawn distance from goal and reports the reach error distribution per
bin. Lets us see whether the lift_history ceiling is "reach is bad
everywhere" vs "reach is bad only when cube is far from goal".

Usage::

    OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y CUDA_VISIBLE_DEVICES=1 \\
        python scripts/measure_reach.py \\
            --ckpt runs/g_plus_sacfd_L2_full/sac.pt \\
            --episodes 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import pinocchio  # noqa: F401, E402

from isaaclab.app import AppLauncher  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--task-level", type=int, default=2)
    p.add_argument("--curriculum-stage", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    sim_app = AppLauncher(headless=True, enable_cameras=False).app

    import collections

    import numpy as np
    import torch

    from khj_rl.envs import CubeLiftEnv, CubeLiftEnvCfg
    from khj_rl.training.normalizer import ObsNormalizer

    cfg = CubeLiftEnvCfg()
    cfg.seed = args.seed
    cfg.curriculum.current_stage_idx = int(args.curriculum_stage)
    cfg.curriculum.task_level = int(args.task_level)
    env = CubeLiftEnv(cfg)

    device = torch.device("cuda:0")
    ckpt = torch.load(args.ckpt, map_location=device)
    is_sac = "actor" in ckpt and "actor_critic" not in ckpt

    obs_dim = int(ckpt.get("obs_dim", cfg.obs_total_size))
    action_dim = int(ckpt.get("action_dim", cfg.action_size))
    hidden_dim = int(ckpt.get("hidden_dim", 256))

    if is_sac:
        from khj_rl.training.sac_network import GaussianActor
        actor = GaussianActor(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim
        ).to(device)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()

        def policy(obs_norm):
            obs_t = torch.from_numpy(obs_norm[None]).to(device).float()
            with torch.no_grad():
                return actor.mean_action(obs_t).squeeze(0).cpu().numpy().astype(np.float32)
    else:
        from khj_rl.training.network import ActorCritic
        actor_logstd_init = float(ckpt.get("actor_logstd_init", -1.0))
        net = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            actor_logstd_init=actor_logstd_init,
        ).to(device)
        net.load_state_dict(ckpt["actor_critic"])
        net.eval()

        def policy(obs_norm):
            obs_t = torch.from_numpy(obs_norm[None]).to(device).float()
            with torch.no_grad():
                return net.actor_mean(obs_t).squeeze(0).cpu().numpy().astype(np.float32)

    normalizer = ObsNormalizer.load(args.ckpt.with_name("norm.json"))

    goal_xy = np.asarray(cfg.goal.pos_xyz_m[:2], dtype=np.float32)

    print(
        f"[measure_reach] ckpt={args.ckpt} is_sac={is_sac} "
        f"episodes={args.episodes} stage={args.curriculum_stage} "
        f"task_level={args.task_level}",
        flush=True,
    )

    # (spawn_dist_from_goal_m, min_finger_to_cube_xy_m, lifted)
    results: list[tuple[float, float, bool]] = []
    for ep_id in range(args.episodes):
        obs = env.reset()
        # cube spawn xy is whatever the env just sampled and committed to sim.
        cube_spawn_xy = env._sim_cube_xyz_gt()[:2].astype(np.float64).copy()
        spawn_dist = float(np.linalg.norm(cube_spawn_xy - goal_xy.astype(np.float64)))

        min_finger_cube_xy = float("inf")
        lifted = False
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = policy(normalizer.normalize_np(obs.astype(np.float32)))
            obs, _r, terminated, truncated, info = env.step(action)
            cube_xy = np.asarray(info["gt_cube_xyz_m"][:2], dtype=np.float64)
            ee_xy = np.asarray(info["her_raw"]["ee_xyz_m"][:2], dtype=np.float64)
            d = float(np.linalg.norm(ee_xy - cube_xy))
            if d < min_finger_cube_xy:
                min_finger_cube_xy = d
            if info["success_per_condition"].get("lift_history"):
                lifted = True

        results.append((spawn_dist, min_finger_cube_xy, lifted))
        if ep_id % 20 == 0:
            print(
                f"[measure_reach] ep={ep_id:03d} "
                f"spawn_dist={spawn_dist*100:.2f}cm "
                f"min_finger_cube_xy={min_finger_cube_xy*1000:.2f}mm "
                f"lifted={lifted}",
                flush=True,
            )

    # Aggregate by spawn distance bin (1cm bins).
    bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0),
            (4.0, 5.0), (5.0, 6.0), (6.0, 7.0), (7.0, 8.0)]
    binned: dict[tuple[float, float], list[tuple[float, bool]]] = collections.defaultdict(list)
    for spawn_dist, min_d, lifted in results:
        for lo, hi in bins:
            if lo / 100.0 <= spawn_dist < hi / 100.0:
                binned[(lo, hi)].append((min_d, lifted))
                break

    print("\n[measure_reach] reach accuracy by cube-to-goal spawn distance:", flush=True)
    print("  bin (cm)   |  n  |  reach_xy mean (mm) |  median (mm) |  worst (mm) |  lifted_rate", flush=True)
    print("  ---------- | --- | ------------------- | ------------ | ----------- | ------------", flush=True)
    for (lo, hi), values in sorted(binned.items()):
        if not values:
            continue
        ds = np.array([v[0] for v in values])
        lifts = np.array([v[1] for v in values])
        print(
            f"  [{lo:>4.1f}, {hi:<4.1f}] | {len(ds):>3} | "
            f"{ds.mean()*1000:>19.1f} | {np.median(ds)*1000:>12.1f} | "
            f"{ds.max()*1000:>11.1f} | {lifts.mean()*100:>10.1f}%",
            flush=True,
        )

    # Overall.
    all_ds = np.array([r[1] for r in results])
    all_lifts = np.array([r[2] for r in results])
    print(
        f"\n[measure_reach] OVERALL n={len(all_ds)} "
        f"reach_xy mean={all_ds.mean()*1000:.1f}mm "
        f"median={np.median(all_ds)*1000:.1f}mm "
        f"worst={all_ds.max()*1000:.1f}mm "
        f"lifted_rate={all_lifts.mean()*100:.1f}%",
        flush=True,
    )

    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
