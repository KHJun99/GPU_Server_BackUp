"""Episode-level policy behavior diagnostic.

Runs N episodes of an oracle/BC/PPO policy and records, per episode:
  - cube_init_xy (obs idx 20, 21)
  - z_max (cube z relative to env origin)
  - ee_to_cube_min: min |ee_pos_b - cube_pos_b| over the episode
  - ee_to_cube_at_close: distance at the FIRST step where gripper command < -0.5
  - gripper_closed_steps: count of steps where action[5] < -0.5
  - first_close_step: index of first close (-1 if never)
  - z_at_first_close: cube z at first close
  - cube_vel_max_after_close: max |cube_lin_vel| AFTER first close (slip proxy)
  - z_at_grasp: cube z that the gripper "grabs" (max z achieved while closed)
  - action_std_episode: mean std across action dims & steps (raw policy stochasticity)

Output: CSV (one row per episode).

Usage:
  CUDA_VISIBLE_DEVICES=1 uv run python tasks/diagnose_policy.py \\
      --policy bc --bc_actor tasks/bc_actor_session8_v1.pt \\
      --num_envs 16 --num_episodes 100 --action_repeat 2 \\
      --output tasks/diagnose_bc_session13.csv --headless
"""

import argparse
import csv
import os
import sys

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Diagnose policy behavior per episode.")
parser.add_argument("--policy", choices=["oracle", "bc", "ppo"], required=True)
parser.add_argument("--bc_actor", default=None)
parser.add_argument("--ppo_ckpt", default=None)
parser.add_argument("--task", default="Isaac-SO-ARM101-Lift-Cube-Play-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--action_repeat", type=int, default=2)
parser.add_argument("--success_z", type=float, default=0.10)
parser.add_argument("--gripper_close_thr", type=float, default=-0.5,
                    help="action[5] < threshold counts as close attempt")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--output", required=True)
parser.add_argument("--disable_fabric", action="store_true", default=False)

# oracle params
parser.add_argument("--scale", type=float, default=1.5)
parser.add_argument("--reach_above_dz", type=float, default=0.05)
parser.add_argument("--descend_dz", type=float, default=0.020)
parser.add_argument("--lift_dz", type=float, default=0.10)
parser.add_argument("--reach_dist", type=float, default=0.04)
parser.add_argument("--descend_z_dist", type=float, default=0.025)
parser.add_argument("--close_steps", type=int, default=8)
parser.add_argument("--max_ee_step", type=float, default=0.02)

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
from oracle_policy import OraclePolicy  # noqa: E402
from train_bc import BCActor  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402


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
    robot = env.unwrapped.scene["robot"]
    object_asset = env.unwrapped.scene["object"]
    env_origins = env.unwrapped.scene.env_origins  # (n_envs, 3)

    # Build policy
    oracle = None
    actor = None
    runner = None
    if args_cli.policy == "oracle":
        def goal_provider() -> torch.Tensor:
            cmd = env.unwrapped.command_manager.get_command("object_pose")
            return cmd[:, 0:3].detach()
        oracle = OraclePolicy(
            num_envs=n_envs, device=device,
            scale=args_cli.scale, reach_above_dz=args_cli.reach_above_dz,
            descend_dz=args_cli.descend_dz, lift_dz=args_cli.lift_dz,
            reach_dist=args_cli.reach_dist, descend_z_dist=args_cli.descend_z_dist,
            close_steps=args_cli.close_steps, max_ee_step=args_cli.max_ee_step,
            success_z=args_cli.success_z,
        )
        oracle.setup(robot=robot, object_asset=object_asset,
                     target_pos_b_provider=goal_provider)
        ee_idx = oracle.ee_idx
    elif args_cli.policy == "bc":
        actor = load_bc(args_cli.bc_actor, device)
        ee_idx = robot.find_bodies("gripper_link")[0][0]
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(device))
        runner.load(args_cli.ppo_ckpt)
        runner.alg.policy.eval()
        ee_idx = robot.find_bodies("gripper_link")[0][0]

    # CSV header
    header = [
        "ep", "success", "z_max", "cube_init_x", "cube_init_y", "cube_init_z",
        "ee_to_cube_min", "ee_to_cube_at_close", "first_close_step",
        "gripper_closed_steps", "z_at_first_close", "cube_vel_max_after_close",
        "z_at_grasp_max", "ep_length",
    ]
    if args_cli.policy == "oracle":
        header.append("final_state")
    os.makedirs(os.path.dirname(args_cli.output) or ".", exist_ok=True)
    fp = open(args_cli.output, "w", newline="")
    writer = csv.writer(fp)
    writer.writerow(header)

    obs = env.get_observations()
    last_action: torch.Tensor | None = None
    step_in_repeat = 0

    # Per-env episode trackers
    ep_step_count = torch.zeros(n_envs, dtype=torch.long, device=device)
    z_max_buf = torch.zeros(n_envs, device=device)
    cube_init = torch.zeros((n_envs, 3), device=device)
    cube_init_set = torch.zeros(n_envs, dtype=torch.bool, device=device)
    ee_to_cube_min = torch.full((n_envs,), 1e9, device=device)
    first_close_step = torch.full((n_envs,), -1, dtype=torch.long, device=device)
    ee_dist_at_close = torch.zeros(n_envs, device=device)
    z_at_first_close = torch.zeros(n_envs, device=device)
    closed_count = torch.zeros(n_envs, dtype=torch.long, device=device)
    cube_vel_after_close_max = torch.zeros(n_envs, device=device)
    z_at_grasp_max = torch.zeros(n_envs, device=device)

    finished = 0
    rows: list[list] = []

    print(f"[INFO] policy={args_cli.policy}  task={args_cli.task}  num_envs={n_envs}  "
          f"target eps={args_cli.num_episodes}  action_repeat={args_cli.action_repeat}")

    while finished < args_cli.num_episodes:
        # Snapshot start-of-step state
        # cube position in env-local frame:
        cube_pos_w = object_asset.data.root_pos_w[:, :3]  # (n_envs, 3) world
        cube_pos_l = cube_pos_w - env_origins  # env-local (~ base frame for the cube)
        # ee position in robot base frame: body_pose_w - env_origins
        ee_pos_w = robot.data.body_pose_w[:, ee_idx, :3]
        ee_pos_l = ee_pos_w - env_origins
        cube_vel = object_asset.data.root_lin_vel_w  # world frame ok for magnitude

        # First step of episode: capture cube_init
        first = ~cube_init_set & (ep_step_count == 0)
        if first.any():
            cube_init[first] = cube_pos_l[first]
            cube_init_set[first] = True

        # Track z_max
        z_max_buf = torch.maximum(z_max_buf, cube_pos_l[:, 2])

        # Compute action
        if step_in_repeat == 0 or last_action is None:
            obs_t = obs if isinstance(obs, torch.Tensor) else obs["policy"]
            obs_t = obs_t.to(device)
            with torch.no_grad():
                if args_cli.policy == "oracle":
                    last_action, _ = oracle.compute()
                elif args_cli.policy == "bc":
                    last_action = actor(obs_t)
                else:
                    last_action = runner.alg.policy.act_inference({"policy": obs_t})

        # Track ee→cube distance min
        ee_to_cube = torch.norm(ee_pos_l - cube_pos_l, dim=1)
        ee_to_cube_min = torch.minimum(ee_to_cube_min, ee_to_cube)

        # Gripper close attempts (action[:, 5] < threshold)
        is_close = last_action[:, 5] < args_cli.gripper_close_thr
        closed_count += is_close.long()

        # First close step + ee_dist at that step + cube z at that step
        first_close = is_close & (first_close_step < 0)
        if first_close.any():
            first_close_step[first_close] = ep_step_count[first_close]
            ee_dist_at_close[first_close] = ee_to_cube[first_close]
            z_at_first_close[first_close] = cube_pos_l[first_close, 2]

        # cube vel after close + z at grasp
        after_close = first_close_step >= 0
        if after_close.any():
            vmag = torch.norm(cube_vel[after_close], dim=1)
            sub_idx = after_close.nonzero(as_tuple=True)[0]
            cube_vel_after_close_max[sub_idx] = torch.maximum(
                cube_vel_after_close_max[sub_idx], vmag
            )
            cur_z_closed = cube_pos_l[after_close, 2]
            z_at_grasp_max[sub_idx] = torch.maximum(z_at_grasp_max[sub_idx], cur_z_closed)

        ep_step_count += 1

        with torch.no_grad():
            obs, _, dones, _ = env.step(last_action)
            done_indices = torch.nonzero(dones, as_tuple=False).squeeze(-1).tolist()

        step_in_repeat = (step_in_repeat + 1) % args_cli.action_repeat

        for env_idx in done_indices:
            if finished >= args_cli.num_episodes:
                break
            finished += 1
            zm = float(z_max_buf[env_idx])
            success = int(zm >= args_cli.success_z)
            row = [
                finished, success, zm,
                float(cube_init[env_idx, 0]),
                float(cube_init[env_idx, 1]),
                float(cube_init[env_idx, 2]),
                float(ee_to_cube_min[env_idx]),
                float(ee_dist_at_close[env_idx]) if first_close_step[env_idx] >= 0 else float("nan"),
                int(first_close_step[env_idx]),
                int(closed_count[env_idx]),
                float(z_at_first_close[env_idx]) if first_close_step[env_idx] >= 0 else float("nan"),
                float(cube_vel_after_close_max[env_idx]) if first_close_step[env_idx] >= 0 else float("nan"),
                float(z_at_grasp_max[env_idx]) if first_close_step[env_idx] >= 0 else float("nan"),
                int(ep_step_count[env_idx]),
            ]
            if args_cli.policy == "oracle":
                row.append(int(oracle.states[env_idx]))
            rows.append(row)

            # Reset trackers for this env (env auto-resets)
            ep_step_count[env_idx] = 0
            z_max_buf[env_idx] = 0.0
            cube_init[env_idx] = 0.0
            cube_init_set[env_idx] = False
            ee_to_cube_min[env_idx] = 1e9
            first_close_step[env_idx] = -1
            ee_dist_at_close[env_idx] = 0.0
            z_at_first_close[env_idx] = 0.0
            closed_count[env_idx] = 0
            cube_vel_after_close_max[env_idx] = 0.0
            z_at_grasp_max[env_idx] = 0.0

            if finished % 20 == 0 or finished == args_cli.num_episodes:
                rate = sum(r[1] for r in rows) / len(rows) * 100.0
                print(f"[INFO] finished={finished}/{args_cli.num_episodes} rate={rate:.1f}% "
                      f"last z_max={zm:.3f}")

        if done_indices and oracle is not None:
            oracle.reset(done_indices)

    for r in rows:
        writer.writerow(r)
    fp.close()
    print(f"[DONE] wrote {len(rows)} rows to {args_cli.output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
