"""BC actor standalone evaluation — runs BC policy in lift task for N episodes
with optional action repeat (sim 100Hz env, BC trained on 50Hz dataset).

Reports success rate (cube_z_max >= success_z), state-at-done histogram, z_max
distribution stats.

Usage:
  CUDA_VISIBLE_DEVICES=1 uv run python tasks/eval_bc_session8.py \\
      --bc_actor tasks/bc_actor_session8_v1.pt \\
      --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \\
      --num_envs 16 --num_episodes 100 --action_repeat 2 \\
      --output tasks/bc_eval_session8.log --headless
"""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate BC actor (success rate + z_max).")
parser.add_argument("--bc_actor", required=True)
parser.add_argument("--task", default="Isaac-SO-ARM101-Lift-Cube-Play-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--action_repeat", type=int, default=2,
                    help="Repeat each BC action this many env steps (sim2real freq).")
parser.add_argument("--success_z", type=float, default=0.10)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaac_so_arm101.tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

sys.path.insert(0, "/home/j-k14d101/jabis_sim/sim2real/oracle")
from train_bc import BCActor  # noqa: E402


def load_bc(path: str, device: torch.device) -> BCActor:
    payload = torch.load(path, map_location=device, weights_only=False)
    sd = payload.get("actor_state_dict", payload)
    arch = payload.get("model_arch", {})
    obs_dim = arch.get("obs_dim", 36)
    act_dim = arch.get("act_dim", 6)
    hidden_dims = tuple(arch.get("hidden_dims", (256, 128, 64)))
    actor = BCActor(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=hidden_dims).to(device)
    # save_bc_init writes actor.net.state_dict() (no "net." prefix) to be
    # compatible with rsl_rl actor MLP loading.
    actor.net.load_state_dict(sd)
    actor.eval()
    return actor


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    n_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    object_asset = env.unwrapped.scene["object"]
    env_origins_z = env.unwrapped.scene.env_origins[:, 2]

    actor = load_bc(args_cli.bc_actor, device)
    print(f"[INFO] BC actor loaded from {args_cli.bc_actor}, action_repeat={args_cli.action_repeat}")
    print(f"[INFO] task={args_cli.task}  num_envs={n_envs}  num_episodes={args_cli.num_episodes}")

    obs = env.get_observations()

    success_count = 0
    fail_count = 0
    finished = 0
    z_max_all: list[float] = []
    last_action: torch.Tensor | None = None
    step_in_repeat = 0
    step_count = 0

    episode_z_max = torch.zeros(n_envs, device=device)
    episode_len = torch.zeros(n_envs, dtype=torch.long, device=device)
    start_time = time.time()

    print("[INFO] Evaluating...")
    while finished < args_cli.num_episodes:
        obs_t = obs if isinstance(obs, torch.Tensor) else obs["policy"]
        if step_in_repeat == 0 or last_action is None:
            with torch.no_grad():
                last_action = actor(obs_t.to(device))
        action_to_apply = last_action

        with torch.no_grad():
            cube_z_local = (object_asset.data.root_pos_w[:, 2] - env_origins_z).detach()
        episode_z_max = torch.maximum(episode_z_max, cube_z_local)
        episode_len += 1

        with torch.no_grad():
            obs, _, dones, _ = env.step(action_to_apply)
            done_indices = torch.nonzero(dones, as_tuple=False).squeeze(-1).tolist()

        step_in_repeat = (step_in_repeat + 1) % args_cli.action_repeat
        step_count += 1

        for env_idx in done_indices:
            if finished >= args_cli.num_episodes:
                break
            finished += 1
            z_max = float(episode_z_max[env_idx])
            z_max_all.append(z_max)
            if z_max >= args_cli.success_z:
                success_count += 1
            else:
                fail_count += 1
            episode_z_max[env_idx] = 0.0
            episode_len[env_idx] = 0

            if finished % 10 == 0 or finished == args_cli.num_episodes:
                elapsed = time.time() - start_time
                rate = success_count / max(finished, 1) * 100.0
                print(
                    f"[INFO] finished={finished}/{args_cli.num_episodes} "
                    f"succ={success_count} fail={fail_count} rate={rate:.1f}% "
                    f"last z_max={z_max:.3f} elapsed={elapsed:.0f}s"
                )

    elapsed = time.time() - start_time
    rate = success_count / max(finished, 1) * 100.0
    z_arr = sorted(z_max_all)
    z_min = z_arr[0] if z_arr else 0.0
    z_max = z_arr[-1] if z_arr else 0.0
    z_mean = sum(z_arr) / len(z_arr) if z_arr else 0.0
    z_med = z_arr[len(z_arr) // 2] if z_arr else 0.0
    z_q25 = z_arr[len(z_arr) // 4] if z_arr else 0.0
    z_q75 = z_arr[(3 * len(z_arr)) // 4] if z_arr else 0.0

    print()
    print("=" * 60)
    print(f"[DONE] BC episodes={finished}  steps={step_count}  elapsed={elapsed:.0f}s")
    print(f"[DONE] success={success_count} fail={fail_count} success_rate={rate:.2f}%")
    print(f"[DONE] z_max: min={z_min:.3f}  q25={z_q25:.3f}  median={z_med:.3f}  "
          f"q75={z_q75:.3f}  mean={z_mean:.3f}  max={z_max:.3f}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
