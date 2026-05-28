"""이전 PnP 모델의 lift height 측정 — 진짜 몇 cm 드는가."""

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--n_trials", type=int, default=3)
parser.add_argument("--env", default="new", choices=["new", "old"],
                    help="new=joint_pos_env_cfg (12cm), old=env_cfg (7cm)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"] + sys.argv[1:])

launcher = AppLauncher(args)

import torch
import statistics
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.modules import ActorCritic

if args.env == "new":
    from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import SoArm101PickPlaceEnvCfg as Cfg
else:
    from jabis_sim_v2.tasks.cube_lift.env_cfg import SoArm101CubeLiftEnvCfg as Cfg

cfg = Cfg()
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
print(f"[INFO] env: {args.env}", flush=True)

env_origins = env.unwrapped.scene.env_origins
cube = env.unwrapped.scene["cube"]

all_max_z = []
all_final_z = []
total_n = 0
threshold_5cm_hit = 0
threshold_7cm_hit = 0
threshold_10cm_hit = 0
threshold_12cm_hit = 0

for trial in range(args.n_trials):
    env.unwrapped.reset()
    obs, _ = env.get_observations()
    max_z = torch.zeros(args.num_envs, device=device)
    final_z = torch.zeros(args.num_envs, device=device)

    for step in range(300):
        with torch.no_grad():
            action = ac.act_inference(obs)
            obs, _, _, _ = env.step(action)
            cl = cube.data.root_pos_w - env_origins
            cube_z = cl[:, 2]
            max_z = torch.maximum(max_z, cube_z)
            final_z = cube_z

    threshold_5cm_hit += int((max_z > 0.05).sum())
    threshold_7cm_hit += int((max_z > 0.07).sum())
    threshold_10cm_hit += int((max_z > 0.10).sum())
    threshold_12cm_hit += int((max_z > 0.12).sum())
    all_max_z.extend(max_z.cpu().tolist())
    all_final_z.extend(final_z.cpu().tolist())
    total_n += args.num_envs

    print(f"trial {trial}: max_z median={statistics.median(max_z.cpu().tolist()):.3f}, "
          f">5cm={int((max_z>0.05).sum())}/{args.num_envs}, "
          f">7cm={int((max_z>0.07).sum())}/{args.num_envs}, "
          f">12cm={int((max_z>0.12).sum())}/{args.num_envs}", flush=True)

print(f"", flush=True)
print(f"[lift height eval] {args.checkpoint}", flush=True)
print(f"[lift height eval] env: {args.env}", flush=True)
print(f"[lift height eval] max_z median:  {statistics.median(all_max_z):.3f} m", flush=True)
print(f"[lift height eval] max_z mean:    {statistics.mean(all_max_z):.3f} m", flush=True)
print(f"[lift height eval] max_z p90:     {sorted(all_max_z)[int(len(all_max_z)*0.9)]:.3f} m", flush=True)
print(f"[lift height eval] reached >5cm:  {threshold_5cm_hit}/{total_n} ({100*threshold_5cm_hit/total_n:.1f}%)", flush=True)
print(f"[lift height eval] reached >7cm:  {threshold_7cm_hit}/{total_n} ({100*threshold_7cm_hit/total_n:.1f}%)", flush=True)
print(f"[lift height eval] reached >10cm: {threshold_10cm_hit}/{total_n} ({100*threshold_10cm_hit/total_n:.1f}%)", flush=True)
print(f"[lift height eval] reached >12cm: {threshold_12cm_hit}/{total_n} ({100*threshold_12cm_hit/total_n:.1f}%)", flush=True)

env.close()
launcher.app.close()
