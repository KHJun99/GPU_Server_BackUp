"""Collect oracle demonstrations for v2 cube lift task.

Saves only successful episodes (cube_z_max >= success_z) for BC training.

Usage:
  python scripts/collect_demos.py --num_envs 64 --target_episodes 100 --output tasks/demos.pt
"""

import argparse
import os
import time
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--target_episodes", type=int, default=100)
parser.add_argument("--max_total_episodes", type=int, default=2000)
parser.add_argument("--success_z", type=float, default=0.07)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(['--headless'] + sys.argv[1:])

launcher = AppLauncher(args)

import torch
from isaaclab.envs import ManagerBasedRLEnv

from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import SoArm101CubeLiftEnvCfg
from jabis_sim_v2.oracle import OraclePolicy


def main():
    cfg = SoArm101CubeLiftEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        cfg.seed = args.seed

    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()

    oracle = OraclePolicy(env)
    oracle.reset()

    device = env.device
    n_envs = env.num_envs

    # buffers
    episode_obs = [[] for _ in range(n_envs)]
    episode_act = [[] for _ in range(n_envs)]
    episode_z_max = torch.zeros(n_envs, device=device)

    demos_obs = []
    demos_act = []
    demos_meta = []
    successes = 0
    finished = 0
    step_count = 0
    start = time.time()

    # initial obs
    obs_dict = env.observation_manager.compute()
    obs_tensor = obs_dict["policy"]
    obs_dim = obs_tensor.shape[1]
    print(f"[INFO] obs_dim={obs_dim} n_envs={n_envs} success_z={args.success_z} target={args.target_episodes}", flush=True)

    while successes < args.target_episodes and finished < args.max_total_episodes:
        action = oracle.compute_action()

        # record obs+act per env
        for i in range(n_envs):
            episode_obs[i].append(obs_tensor[i].detach().cpu().clone())
            episode_act[i].append(action[i].detach().cpu().clone())

        # step
        with torch.no_grad():
            obs_dict, _, terminated, truncated, _ = env.step(action)
            obs_tensor = obs_dict["policy"]
            dones = terminated | truncated
            if step_count % 50 == 0:
                print(f"[DBG step_raw] term_sum={int(terminated.sum())} trunc_sum={int(truncated.sum())} dones_sum={int(dones.sum())}", flush=True)

        # update z_max
        cube_z = oracle._get_cube_pos_w()[:, 2] - env.scene.env_origins[:, 2]
        episode_z_max = torch.maximum(episode_z_max, cube_z)

        done_indices = torch.nonzero(dones, as_tuple=False).squeeze(-1).tolist()
        if isinstance(done_indices, int):
            done_indices = [done_indices]
        for env_idx in done_indices:
            finished += 1
            z_max = float(episode_z_max[env_idx])
            ep_len = len(episode_obs[env_idx])
            if finished <= 20 or finished % 10 == 0:
                print(f"[DBG done] finished={finished} env={env_idx} z_max={z_max:.3f} len={ep_len} succ={'YES' if z_max >= args.success_z else 'no'}", flush=True)
            if z_max >= args.success_z and ep_len > 0:
                demos_obs.append(torch.stack(episode_obs[env_idx]))
                demos_act.append(torch.stack(episode_act[env_idx]))
                demos_meta.append({
                    "length": ep_len,
                    "z_max": z_max,
                    "env_idx": int(env_idx),
                })
                successes += 1
                if successes % 10 == 0 or successes == args.target_episodes:
                    elapsed = time.time() - start
                    rate = successes / max(finished, 1) * 100.0
                    print(f"[INFO] succ={successes}/{args.target_episodes}  finished={finished}  rate={rate:.1f}%  z_max={z_max:.3f}  elapsed={elapsed:.0f}s", flush=True)
            episode_obs[env_idx] = []
            episode_act[env_idx] = []
            episode_z_max[env_idx] = 0.0

        if done_indices:
            oracle.reset(torch.tensor(done_indices, device=device, dtype=torch.long))

        step_count += 1

        # DEBUG: every 50 steps print current state
        if step_count % 50 == 0:
            cube_z_now = (oracle._get_cube_pos_w()[:, 2] - env.scene.env_origins[:, 2])
            n_lift = int((cube_z_now > 0.055).sum())
            print(f"[DBG step={step_count}] z_max_alive_max={float(episode_z_max.max()):.3f} z_max_mean={float(episode_z_max.mean()):.3f} cube_lifted_now={n_lift}/{n_envs} dones_so_far={finished} succ_so_far={successes}", flush=True)

    elapsed = time.time() - start
    rate = successes / max(finished, 1) * 100.0
    print(f"\n[DONE] succ={successes} finished={finished} rate={rate:.2f}% steps={step_count} elapsed={elapsed:.0f}s", flush=True)

    if successes == 0:
        print("[WARN] no successes — save skipped.")
    else:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        torch.save({
            "obs": demos_obs,
            "act": demos_act,
            "meta": demos_meta,
            "obs_dim": obs_dim,
            "act_dim": int(demos_act[0].shape[1]),
            "success_z": args.success_z,
            "source": "oracle_v2",
            "n_total_episodes_attempted": finished,
        }, args.output)
        print(f"[INFO] saved {successes} demos → {args.output}", flush=True)

    env.close()


main()
launcher.app.close()
