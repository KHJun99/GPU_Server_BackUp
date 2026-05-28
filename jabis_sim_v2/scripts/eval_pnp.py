"""Eval PPO policy on Pick & Place — measure cube at target (mid-episode)."""

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--n_trials", type=int, default=3)
parser.add_argument("--success_dist", type=float, default=0.05)
parser.add_argument("--measure_step", type=int, default=250,
                    help="step at which to measure placed (avoid end-reset)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"] + sys.argv[1:])

launcher = AppLauncher(args)

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.modules import ActorCritic
from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import SoArm101PickPlaceEnvCfg
from jabis_sim_v2.tasks.cube_lift.mdp.rewards_pnp import TARGET_POS


def main():
    cfg = SoArm101PickPlaceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    device = env.unwrapped.device

    env.unwrapped.reset()
    obs, _ = env.get_observations()
    num_obs = obs.shape[1]
    num_actions = env.unwrapped.action_manager.total_action_dim

    ac = ActorCritic(num_actor_obs=num_obs, num_critic_obs=num_obs, num_actions=num_actions,
                     actor_hidden_dims=[256,128,64], critic_hidden_dims=[256,128,64],
                     activation="elu", init_noise_std=1.0).to(device)
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ac.load_state_dict(blob["model_state_dict"])
    ac.eval()
    print(f"[INFO] loaded: {args.checkpoint}", flush=True)

    target = TARGET_POS.to(device)
    env_origins = env.unwrapped.scene.env_origins
    cube = env.unwrapped.scene["cube"]

    total_placed_at_measure = 0
    total_reached_anytime = 0
    total_holding_stable = 0  # placed for >= 50 consecutive steps
    total_n = 0
    min_dists = []

    for trial in range(args.n_trials):
        env.unwrapped.reset()
        obs, _ = env.get_observations()
        reached_anytime = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        consecutive_placed = torch.zeros(args.num_envs, dtype=torch.int, device=device)
        max_consec = torch.zeros(args.num_envs, dtype=torch.int, device=device)
        min_dist = torch.full((args.num_envs,), 999.0, device=device)
        placed_at_measure = None

        for step in range(300):
            with torch.no_grad():
                action = ac.act_inference(obs)
                obs, _, _, _ = env.step(action)
                cl = cube.data.root_pos_w - env_origins
                dist = torch.norm(cl - target, dim=-1)
                min_dist = torch.minimum(min_dist, dist)
                placed_now = (dist < args.success_dist) & (cl[:, 2] < 0.08)
                reached_anytime = reached_anytime | placed_now

                consecutive_placed = torch.where(placed_now, consecutive_placed + 1, torch.zeros_like(consecutive_placed))
                max_consec = torch.maximum(max_consec, consecutive_placed)

                if step == args.measure_step:
                    placed_at_measure = placed_now.clone()

        total_placed_at_measure += int(placed_at_measure.sum())
        total_reached_anytime += int(reached_anytime.sum())
        total_holding_stable += int((max_consec >= 50).sum())
        total_n += args.num_envs
        min_dists.extend(min_dist.cpu().tolist())

        print(f"trial {trial}: placed_at_step{args.measure_step}={int(placed_at_measure.sum())}/{args.num_envs}, "
              f"reached_anytime={int(reached_anytime.sum())}/{args.num_envs}, "
              f"holding>=50steps={int((max_consec >= 50).sum())}/{args.num_envs}", flush=True)

    import statistics
    print(f"", flush=True)
    print(f"[PnP eval] {args.checkpoint}", flush=True)
    print(f"[PnP eval] placed at step {args.measure_step}: {total_placed_at_measure}/{total_n} ({100*total_placed_at_measure/total_n:.1f}%)", flush=True)
    print(f"[PnP eval] reached anytime:            {total_reached_anytime}/{total_n} ({100*total_reached_anytime/total_n:.1f}%)", flush=True)
    print(f"[PnP eval] holding stable (>=50 step): {total_holding_stable}/{total_n} ({100*total_holding_stable/total_n:.1f}%)", flush=True)
    print(f"[PnP eval] min dist median:            {statistics.median(min_dists):.3f} m", flush=True)

    env.close()


main()
launcher.app.close()
