"""H2 force-tracing diagnostic — log per-step pre_action / clipped_action / scaled_action /
joint_target / applied_torque / joint_pos / joint_vel to CSV. Also captures stderr lines
mentioning "mimic" to a separate file.

Designed for next-session H2 root-cause analysis. The 35× ee-delta squash observed in
session 2 needs decomposition: where exactly does the command lose magnitude between
oracle output and actual joint motion?

DO NOT RUN THIS SCRIPT IN SESSION 3. Just compile-check.

Usage (NEXT SESSION ONLY)
-------------------------
  CUDA_VISIBLE_DEVICES=1 uv run python /home/j-k14d101/isaac_so_arm101/tasks/diagnose_h2_force.py \\
      --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \\
      --num_envs 1 --max_steps 200 \\
      --output_csv /home/j-k14d101/isaac_so_arm101/tasks/diagnose_h2_force.csv \\
      --output_stderr /home/j-k14d101/isaac_so_arm101/tasks/diagnose_h2_stderr.log \\
      --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="H2 force-tracing diagnostic.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Lift-Cube-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--max_steps", type=int, default=200)
parser.add_argument("--output_csv", type=str, required=True)
parser.add_argument("--output_stderr", type=str, required=True,
                    help="Path to capture stderr lines mentioning 'mimic'.")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)

parser.add_argument("--scale", type=float, default=1.5)
parser.add_argument("--reach_above_dz", type=float, default=0.10)
parser.add_argument("--descend_dz", type=float, default=0.005)
parser.add_argument("--lift_dz", type=float, default=0.10)
parser.add_argument("--reach_dist", type=float, default=0.02)
parser.add_argument("--descend_z_dist", type=float, default=0.005)
parser.add_argument("--close_steps", type=int, default=8)
parser.add_argument("--max_ee_step", type=float, default=0.02)
parser.add_argument("--success_z", type=float, default=0.10)
parser.add_argument("--force_close_step", type=int, default=100)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

# Redirect stderr through a tee that filters for "mimic" lines.
import io  # noqa: E402

class MimicTee:
    """File-like wrapper that copies all writes to original stderr and additionally
    appends any line containing 'mimic' (case-insensitive) to a separate file."""

    def __init__(self, original, mimic_log_path: str):
        self.original = original
        self.mimic_fp = open(mimic_log_path, "w")
        self._buffer = io.StringIO()

    def write(self, s: str) -> int:
        n = self.original.write(s)
        self._buffer.write(s)
        # Flush by line.
        buf = self._buffer.getvalue()
        if "\n" in buf:
            head, _, tail = buf.rpartition("\n")
            for line in head.split("\n"):
                if "mimic" in line.lower():
                    self.mimic_fp.write(line + "\n")
                    self.mimic_fp.flush()
            self._buffer = io.StringIO()
            self._buffer.write(tail)
        return n

    def flush(self) -> None:
        self.original.flush()
        self.mimic_fp.flush()

    def fileno(self) -> int:
        # Required by faulthandler.enable() and similar low-level callers.
        return self.original.fileno()

    def isatty(self) -> bool:
        try:
            return self.original.isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self.original, name)


os.makedirs(os.path.dirname(args_cli.output_stderr) or ".", exist_ok=True)
sys.stderr = MimicTee(sys.stderr, args_cli.output_stderr)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaac_so_arm101.tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

sys.path.insert(0, "/home/j-k14d101/jabis_sim/sim2real/oracle")
from oracle_policy import OraclePolicy, APPROACH, DESCEND, CLOSE, LIFT, MOVE_TO_GOAL  # noqa: E402

STATE_NAMES = {APPROACH: "APPROACH", DESCEND: "DESCEND", CLOSE: "CLOSE",
               LIFT: "LIFT", MOVE_TO_GOAL: "MOVE_TO_GOAL"}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    n_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    object_asset = env.unwrapped.scene["object"]

    arm_action_term = env.unwrapped.action_manager.get_term("arm_action")
    grip_action_term = env.unwrapped.action_manager.get_term("gripper_action")
    clip_actions_val = float(agent_cfg.clip_actions) if agent_cfg.clip_actions is not None else None

    # codex caveat: API attributes vary across IsaacLab versions. Dump for visibility.
    print(f"[DBG] arm_action_term attrs: {sorted([a for a in dir(arm_action_term) if not a.startswith('_')])[:25]}")
    print(f"[DBG] grip_action_term attrs: {sorted([a for a in dir(grip_action_term) if not a.startswith('_')])[:25]}")
    print(f"[DBG] grip_action_term cfg.joint_names: {getattr(grip_action_term.cfg, 'joint_names', '(missing)')}")
    print(f"[DBG] grip_action_term action_dim: {getattr(grip_action_term, 'action_dim', '(missing)')}")
    print(f"[DBG] robot.data attrs: {sorted([a for a in dir(robot.data) if 'torque' in a.lower() or 'action' in a.lower()])}")

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
    oracle.setup(robot=robot, object_asset=object_asset, target_pos_b_provider=goal_provider)

    joint_names = list(robot.data.joint_names)
    n_joints = len(joint_names)
    print(f"[INFO] joint_names ({n_joints}): {joint_names}")
    print(f"[INFO] arm_action_term: {type(arm_action_term).__name__}")
    print(f"[INFO] grip_action_term: {type(grip_action_term).__name__}")
    print(f"[INFO] clip_actions: {clip_actions_val}")
    print(f"[INFO] csv: {args_cli.output_csv}")
    print(f"[INFO] mimic stderr log: {args_cli.output_stderr}")

    header = ["step", "state", "force_close"]
    for i in range(6):
        header.append(f"pre_action_{i}")
    for i in range(6):
        header.append(f"clipped_action_{i}")
    for j in joint_names:
        header.append(f"applied_target_{j}")
    for j in joint_names:
        header.append(f"applied_torque_{j}")
    for j in joint_names:
        header.append(f"jpos_{j}")
    for j in joint_names:
        header.append(f"jvel_{j}")
    header += ["target_ee_x", "target_ee_y", "target_ee_z",
               "current_ee_x", "current_ee_y", "current_ee_z",
               "cube_x", "cube_y", "cube_z"]

    out_dir = os.path.dirname(args_cli.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fp = open(args_cli.output_csv, "w", newline="")
    writer = csv.writer(fp)
    writer.writerow(header)

    _ = env.get_observations()

    print("[INFO] H2 force-tracing started...")
    last_step = -1
    for step_idx in range(args_cli.max_steps):
        action, info = oracle.compute()

        force_close = False
        if step_idx >= args_cli.force_close_step:
            current_state = int(oracle.states[0])
            if current_state == DESCEND:
                action[:, 5] = -1.0
                force_close = True

        pre_action = action[0].detach().cpu().tolist()
        if clip_actions_val is not None:
            clipped = action.clamp(-clip_actions_val, clip_actions_val)
        else:
            clipped = action.clone()
        clipped_action = clipped[0].detach().cpu().tolist()

        state_now = int(oracle.states[0])
        target_ee = info["target_pos_b"][0].detach().cpu().tolist()
        current_ee = info["ee_pos_b"][0].detach().cpu().tolist()
        cube = info["cube_pos_b"][0].detach().cpu().tolist()

        with torch.no_grad():
            _, _, dones, _ = env.step(action)
            done_indices = torch.nonzero(dones, as_tuple=False).squeeze(-1).tolist()

        # codex caveat: processed_actions / _joint_names attribute names vary by version.
        # Use hasattr fallback chain.
        applied_target = [float("nan")] * n_joints
        for term, attr in [(arm_action_term, "arm"), (grip_action_term, "grip")]:
            target_t = None
            for cand in ("processed_actions", "_processed_actions", "raw_actions", "_raw_actions"):
                if hasattr(term, cand):
                    target_t = getattr(term, cand)
                    break
            term_joint_names = None
            for cand in ("_joint_names", "joint_names"):
                if hasattr(term, cand):
                    term_joint_names = getattr(term, cand)
                    break
            if target_t is None or term_joint_names is None:
                continue
            for k, jn in enumerate(joint_names):
                if jn in term_joint_names:
                    idx = list(term_joint_names).index(jn)
                    try:
                        applied_target[k] = float(target_t[0, idx])
                    except (IndexError, TypeError):
                        pass

        applied_torque = [float("nan")] * n_joints
        for cand in ("applied_torque", "_applied_torque", "computed_torque", "joint_torque"):
            if hasattr(robot.data, cand):
                try:
                    applied_torque = getattr(robot.data, cand)[0].detach().cpu().tolist()
                    break
                except (AttributeError, IndexError, TypeError):
                    continue

        joint_pos_now = robot.data.joint_pos[0].detach().cpu().tolist()
        joint_vel_now = robot.data.joint_vel[0].detach().cpu().tolist()

        row = [step_idx, STATE_NAMES[state_now], int(force_close)]
        row += pre_action
        row += clipped_action
        row += applied_target
        row += applied_torque
        row += joint_pos_now
        row += joint_vel_now
        row += target_ee + current_ee + cube
        writer.writerow(row)
        last_step = step_idx

        if 0 in done_indices:
            print(f"[INFO] env-0 done at step {step_idx}, stopping.")
            oracle.reset(done_indices)
            break

    fp.close()
    sys.stderr.flush()
    print(f"[DONE] wrote {last_step + 1} rows to {args_cli.output_csv}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
