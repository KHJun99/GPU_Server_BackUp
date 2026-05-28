"""Episode 끝났을 때 cube가 어디에 있는지 측정."""
import argparse, sys, statistics
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--n_trials", type=int, default=3)
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"] + sys.argv[1:])
launcher = AppLauncher(args)

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.modules import ActorCritic
from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import SoArm101PickPlaceEnvCfg

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

env_origins = env.unwrapped.scene.env_origins
cube = env.unwrapped.scene["cube"]

all_x, all_y, all_z = [], [], []
for trial in range(args.n_trials):
    env.unwrapped.reset()
    obs, _ = env.get_observations()
    for step in range(args.steps):
        with torch.no_grad():
            action = ac.act_inference(obs)
            obs, _, _, _ = env.step(action)
    cl = (cube.data.root_pos_w - env_origins).cpu()
    all_x.extend(cl[:, 0].tolist())
    all_y.extend(cl[:, 1].tolist())
    all_z.extend(cl[:, 2].tolist())
    print(f"trial {trial}: cube final xy median=({statistics.median(cl[:,0].tolist()):.3f}, {statistics.median(cl[:,1].tolist()):.3f}, {statistics.median(cl[:,2].tolist()):.3f})", flush=True)

print(f"", flush=True)
print(f"[final pos] x median={statistics.median(all_x):.3f}, mean={statistics.mean(all_x):.3f}, range=[{min(all_x):.3f}, {max(all_x):.3f}]", flush=True)
print(f"[final pos] y median={statistics.median(all_y):.3f}, mean={statistics.mean(all_y):.3f}, range=[{min(all_y):.3f}, {max(all_y):.3f}]", flush=True)
print(f"[final pos] z median={statistics.median(all_z):.3f}, mean={statistics.mean(all_z):.3f}", flush=True)
env.close()
launcher.app.close()
