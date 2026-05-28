"""Runtime introspection — confirm what cfg the env *actually* uses, after editable
install + import + Hydra + AppLauncher. Read-only. No modifications.

Outputs:
  [1] env.cfg class hierarchy (mro) + source file
  [2] Robot articulation: joint_names, num_joints, actuators (key, joint_names,
      effort/stiffness/damping per actuator)
  [3] Action manager terms: cfg attributes (joint_names, scale, clip,
      open/close_command_expr), runtime buffers (raw_actions, processed_actions,
      action_dim, _joint_ids/_joint_names)
  [4] Total action dim, action space
"""

import argparse
import inspect
import sys

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Lift-Cube-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaac_so_arm101.tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def _safe_repr(v, w: int = 80) -> str:
    s = repr(v)
    if len(s) > w:
        return s[:w] + "..."
    return s


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped

    print()
    print("=" * 70)
    print("[1] env.cfg class hierarchy + source")
    print("=" * 70)
    cfg_cls = type(unwrapped.cfg)
    print(f"class: {cfg_cls.__module__}.{cfg_cls.__name__}")
    try:
        print(f"source: {inspect.getfile(cfg_cls)}")
    except TypeError:
        print("source: (builtin)")
    print(f"mro:")
    for c in cfg_cls.__mro__:
        try:
            src = inspect.getfile(c)
        except TypeError:
            src = "(builtin)"
        print(f"  {c.__module__}.{c.__name__}  <- {src}")

    print()
    print("=" * 70)
    print("[2] Robot articulation")
    print("=" * 70)
    robot = unwrapped.scene["robot"]
    print(f"joint_names ({len(robot.joint_names)}): {robot.joint_names}")
    print(f"num_joints: {robot.num_joints}")
    print(f"actuators keys: {list(robot.actuators.keys())}")
    for name, act in robot.actuators.items():
        print(f"  --- actuator '{name}' (class {type(act).__name__}) ---")
        # Some attrs are on the actuator instance, others on its cfg.
        for attr in ("joint_names", "joint_indices"):
            v = getattr(act, attr, "(missing)")
            print(f"    {attr}: {v}")
        cfg = getattr(act, "cfg", None)
        if cfg is not None:
            for attr in ("joint_names_expr", "effort_limit", "effort_limit_sim",
                         "velocity_limit", "velocity_limit_sim", "stiffness", "damping"):
                if hasattr(cfg, attr):
                    print(f"    cfg.{attr}: {_safe_repr(getattr(cfg, attr))}")

    print()
    print("=" * 70)
    print("[3] Action manager terms")
    print("=" * 70)
    am = unwrapped.action_manager
    print(f"active terms: {am.active_terms}")
    print(f"total_action_dim: {am.total_action_dim}")

    for term_name in am.active_terms:
        term = am.get_term(term_name)
        print(f"\n  --- term '{term_name}' ---")
        print(f"    class: {type(term).__module__}.{type(term).__name__}")
        try:
            print(f"    source: {inspect.getfile(type(term))}")
        except TypeError:
            print("    source: (builtin)")
        cfg = getattr(term, "cfg", None)
        if cfg is not None:
            for attr in ("joint_names", "scale", "clip", "use_default_offset",
                         "open_command_expr", "close_command_expr"):
                if hasattr(cfg, attr):
                    print(f"    cfg.{attr}: {_safe_repr(getattr(cfg, attr))}")
        print(f"    action_dim: {getattr(term, 'action_dim', '(missing)')}")
        # Resolved internal joint names/ids — pre-cfg-rename and post.
        for cand in ("_joint_names", "joint_names"):
            if hasattr(term, cand):
                print(f"    {cand}: {_safe_repr(getattr(term, cand))}")
                break
        for cand in ("_joint_ids", "joint_ids", "joint_indices"):
            if hasattr(term, cand):
                print(f"    {cand}: {_safe_repr(getattr(term, cand))}")
                break
        # Runtime buffers.
        for cand in ("raw_actions", "_raw_actions", "processed_actions",
                     "_processed_actions", "_open_command", "_close_command"):
            if hasattr(term, cand):
                v = getattr(term, cand)
                if hasattr(v, "shape"):
                    print(f"    {cand} shape: {tuple(v.shape)}")
                else:
                    print(f"    {cand}: {_safe_repr(v)}")
        # Show all non-private attrs once for completeness.
        attrs = sorted(a for a in dir(term) if not a.startswith("__"))
        print(f"    [all attrs]: {attrs}")

    print()
    print("=" * 70)
    print("[4] action_space / total dim")
    print("=" * 70)
    print(f"env.action_space: {env.action_space}")
    print(f"unwrapped.action_space: {unwrapped.action_space}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
