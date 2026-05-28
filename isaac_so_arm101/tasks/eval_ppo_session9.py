"""PPO checkpoint standalone evaluation.

Loads a rsl_rl OnPolicyRunner checkpoint, runs N episodes, reports:
  - success rate (cube_z_max >= success_z)
  - z_max distribution
  - PPO action_mean entropy proxy (mean of action_std)
  - action MSE vs BC actor (KL proxy, requires --bc_actor)

Usage:
  CUDA_VISIBLE_DEVICES=1 uv run python tasks/eval_ppo_session9.py \\
      --checkpoint logs/rsl_rl/lift/<run>/model_50.pt \\
      --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \\
      --num_envs 16 --num_episodes 100 --action_repeat 2 \\
      --bc_actor tasks/bc_actor_session8_v1.pt \\
      --output tasks/ppo_eval_iter50.log --headless
"""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a PPO checkpoint.")
parser.add_argument("--ppo_ckpt", required=True, help="path to rsl_rl model_<N>.pt")
parser.add_argument("--task", default="Isaac-SO-ARM101-Lift-Cube-Play-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--action_repeat", type=int, default=2)
parser.add_argument("--success_z", type=float, default=0.10)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--bc_actor", default=None,
                    help="optional BC actor for action-MSE comparison")
parser.add_argument("--disable_fabric", action="store_true", default=False)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaac_so_arm101.tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

sys.path.insert(0, "/home/j-k14d101/jabis_sim/sim2real/oracle")
from train_bc import BCActor  # noqa: E402


def load_bc(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    sd = payload.get("actor_state_dict", payload)
    arch = payload.get("model_arch", {})
    actor = BCActor(
        obs_dim=arch.get("obs_dim", 36),
        act_dim=arch.get("act_dim", 6),
        hidden_dims=tuple(arch.get("hidden_dims", (256, 128, 64))),
    ).to(device)
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

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(device))
    runner.load(args_cli.ppo_ckpt)
    runner.alg.policy.eval()
    print(f"[INFO] PPO checkpoint loaded: {args_cli.ppo_ckpt}")

    bc = None
    if args_cli.bc_actor is not None:
        bc = load_bc(args_cli.bc_actor, device)
        print(f"[INFO] BC actor loaded for action-MSE proxy: {args_cli.bc_actor}")

    print(f"[INFO] task={args_cli.task}  num_envs={n_envs}  episodes={args_cli.num_episodes}  "
          f"action_repeat={args_cli.action_repeat}")

    # Reset
    obs = env.get_observations()
    success_count = 0
    fail_count = 0
    finished = 0
    z_max_all: list[float] = []
    action_mse_total = 0.0
    action_mse_n = 0
    entropy_proxy_total = 0.0
    entropy_proxy_n = 0

    last_action: torch.Tensor | None = None
    step_in_repeat = 0
    step_count = 0

    episode_z_max = torch.zeros(n_envs, device=device)
    start_time = time.time()

    print("[INFO] Evaluating...")
    while finished < args_cli.num_episodes:
        obs_t = obs if isinstance(obs, torch.Tensor) else obs["policy"]
        # rsl_rl ActorCritic.act_inference expects an obs dict.
        ppo_obs = {"policy": obs_t.to(device)}
        if step_in_repeat == 0 or last_action is None:
            with torch.no_grad():
                last_action = runner.alg.policy.act_inference(ppo_obs)
                # entropy proxy = mean of policy noise std (scalar or log-std)
                policy = runner.alg.policy
                if hasattr(policy, "std") and isinstance(policy.std, torch.Tensor):
                    std_val = float(policy.std.detach().mean())
                elif hasattr(policy, "log_std"):
                    std_val = float(torch.exp(policy.log_std).detach().mean())
                else:
                    std_val = float("nan")
                entropy_proxy_total += std_val
                entropy_proxy_n += 1
                # action MSE vs BC
                if bc is not None:
                    bc_action = bc(obs_t.to(device))
                    mse = ((last_action - bc_action) ** 2).mean()
                    action_mse_total += float(mse)
                    action_mse_n += 1
        action_to_apply = last_action

        with torch.no_grad():
            cube_z_local = (object_asset.data.root_pos_w[:, 2] - env_origins_z).detach()
        episode_z_max = torch.maximum(episode_z_max, cube_z_local)

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
            if finished % 20 == 0 or finished == args_cli.num_episodes:
                elapsed = time.time() - start_time
                rate = success_count / max(finished, 1) * 100.0
                print(
                    f"[INFO] finished={finished}/{args_cli.num_episodes} "
                    f"succ={success_count} rate={rate:.1f}% "
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
    entropy_avg = entropy_proxy_total / max(entropy_proxy_n, 1)
    action_mse_avg = action_mse_total / max(action_mse_n, 1) if bc is not None else float("nan")

    print()
    print("=" * 60)
    print(f"[DONE] PPO ckpt={args_cli.ppo_ckpt}")
    print(f"[DONE] episodes={finished}  steps={step_count}  elapsed={elapsed:.0f}s")
    print(f"[DONE] success={success_count} fail={fail_count} success_rate={rate:.2f}%")
    print(f"[DONE] z_max: min={z_min:.3f}  q25={z_q25:.3f}  median={z_med:.3f}  "
          f"q75={z_q75:.3f}  mean={z_mean:.3f}  max={z_max:.3f}")
    print(f"[DONE] entropy proxy (mean action_std) = {entropy_avg:.4f}")
    if bc is not None:
        print(f"[DONE] action MSE vs BC = {action_mse_avg:.6f}  (KL proxy)")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
