"""Evaluate MoG BC actor in sim."""

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--bc_init", required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--n_trials", type=int, default=3)
parser.add_argument("--success_z", type=float, default=0.07)
parser.add_argument("--mode", default="argmax", choices=["argmax", "sample"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(['--headless'] + sys.argv[1:])

launcher = AppLauncher(args)

import torch
from isaaclab.envs import ManagerBasedRLEnv
from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import SoArm101CubeLiftEnvCfg

# import MoGActor from train script
sys.path.insert(0, '/home/j-k14d101/jabis_sim_v2/scripts')
from train_bc_mog import MoGActor


def main():
    blob = torch.load(args.bc_init, map_location="cpu", weights_only=False)
    arch = blob["model_arch"]
    print(f"[INFO] MoG arch: {arch}", flush=True)
    actor = MoGActor(arch["obs_dim"], arch["act_dim"], tuple(arch["hidden_dims"]), num_modes=arch["num_modes"])
    actor.load_state_dict(blob["model_state_dict"])
    actor.eval()

    cfg = SoArm101CubeLiftEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=cfg)
    device = env.device
    actor = actor.to(device)

    print(f"[INFO] mode={args.mode}, {args.n_trials} trials × {args.num_envs} envs", flush=True)

    total_success = 0
    total_episodes = 0
    z_max_all = []

    for trial in range(args.n_trials):
        obs_dict, _ = env.reset()
        obs_tensor = obs_dict["policy"]
        z_max_episode = torch.zeros(args.num_envs, device=device)
        env_origins_z = env.scene.env_origins[:, 2]

        for step in range(300):
            with torch.no_grad():
                action = actor.act_inference(obs_tensor, mode=args.mode).clamp(-1.0, 1.0)
                obs_dict, _, _, _, _ = env.step(action)
                obs_tensor = obs_dict["policy"]
                cube_z = env.scene["cube"].data.root_pos_w[:, 2] - env_origins_z
                z_max_episode = torch.maximum(z_max_episode, cube_z)

        success = (z_max_episode >= args.success_z).int()
        total_success += int(success.sum())
        total_episodes += args.num_envs
        z_max_np = z_max_episode.cpu().numpy()
        z_max_all.extend(z_max_np.tolist())
        print(f"trial {trial}: s={int(success.sum())}/{args.num_envs}  z_max_mean={float(z_max_episode.mean()):.3f}", flush=True)

    rate = 100 * total_success / total_episodes
    print(f"\n[MoG eval] mode={args.mode}", flush=True)
    print(f"[MoG eval] total: {total_success}/{total_episodes} ({rate:.1f}%)", flush=True)
    print(f"[MoG eval] z_max mean: {sum(z_max_all)/len(z_max_all):.3f}, max: {max(z_max_all):.3f}", flush=True)

    env.close()


main()
launcher.app.close()
