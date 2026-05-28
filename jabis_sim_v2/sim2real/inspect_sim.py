"""Sim inspection - joint ordering, action mapping, frame ground truth.

학습 시 sim env 가 어떤 ordering / frame 으로 obs 를 만들었는지 확인.
Phase 2 obs_builder 의 ground truth.

실행 후 marker 사이 출력만 추출:
    python inspect_sim.py 2>&1 \\
        | awk '/JABIS_INSPECT_BEGIN/{flag=1} flag; /JABIS_INSPECT_END/{flag=0}' \\
        | tee inspection_raw.txt
"""

import argparse
import sys


def _p(msg):
    print(msg, flush=True)
    sys.stderr.flush()


_p("[stage] importing AppLauncher")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])

_p("[stage] launching app")
launcher = AppLauncher(args)
_p("[stage] app launched")

# IsaacLab modules can only be imported after AppLauncher.
_p("[stage] importing ManagerBasedRLEnv")
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

_p("[stage] importing env cfg")
from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import (  # noqa: E402
    SoArm101PickPlaceEnvCfg,
)
_p("[stage] env cfg imported")


def _to_list(v):
    if hasattr(v, "cpu"):
        v = v.cpu().tolist()
    elif hasattr(v, "tolist"):
        v = v.tolist()
    return v


def main():
    _p("[stage] building cfg")
    cfg = SoArm101PickPlaceEnvCfg()
    cfg.scene.num_envs = 1
    _p("[stage] cfg built; creating env")
    env = ManagerBasedRLEnv(cfg=cfg)
    _p("[stage] env created; resetting")
    env.reset()
    _p("[stage] env reset done; getting robot")
    robot = env.scene["robot"]
    _p("[stage] robot ok")

    print("=====JABIS_INSPECT_BEGIN=====", flush=True)

    # 1. joint ordering + defaults (obs idx 0~9 ground truth)
    names = list(robot.data.joint_names)
    default = robot.data.default_joint_pos[0].cpu().numpy().tolist()
    print("JOINT_NAMES_AND_DEFAULTS:")
    for i, (n, d) in enumerate(zip(names, default)):
        print(f"  [{i:2d}] {n:25s} default={d:+.4f}")
    print(f"  total={len(names)}")

    # 2. arm_action / gripper_action joint mapping
    arm = env.action_manager.get_term("arm_action")
    grip = env.action_manager.get_term("gripper_action")

    print("ARM_ACTION:")
    for attr in ("_joint_names", "joint_names", "_joint_ids", "joint_ids"):
        if hasattr(arm, attr):
            print(f"  arm.{attr} = {_to_list(getattr(arm, attr))}")

    print("GRIPPER_ACTION:")
    for attr in (
        "_joint_names", "joint_names",
        "_joint_ids", "joint_ids",
        "_open_command", "open_command",
        "_close_command", "close_command",
    ):
        if hasattr(grip, attr):
            print(f"  gripper.{attr} = {_to_list(getattr(grip, attr))}")

    # 3. frame info
    env_origin = env.scene.env_origins[0].cpu().numpy().tolist()
    robot_root = robot.data.root_pos_w[0].cpu().numpy().tolist()
    cube_root = env.scene["cube"].data.root_pos_w[0].cpu().numpy().tolist()
    print(f"ENV_ORIGINS[0]: {env_origin}")
    print(f"ROBOT_ROOT_POS_W[0]: {robot_root}")
    print(f"CUBE_ROOT_POS_W[0]: {cube_root}")

    diff = [robot_root[i] - env_origin[i] for i in range(3)]
    cube_local = [cube_root[i] - env_origin[i] for i in range(3)]
    print(f"ROBOT_ROOT - ENV_ORIGIN: {diff}")
    print(f"CUBE - ENV_ORIGIN (= obs cube_pos if hypothesis holds): {cube_local}")

    # 4. obs dim
    obs_dict = env.observation_manager.compute()
    policy_obs = obs_dict["policy"]
    print(f"OBS_SHAPE: {tuple(policy_obs.shape)}")
    print(f"OBS_VALUES[0]: {policy_obs[0].cpu().numpy().tolist()}")

    # 5. action dim
    print(f"ACTION_DIM: {env.action_manager.total_action_dim}")

    # 6. sim timing
    sim_dt = float(env.cfg.sim.dt)
    decimation = int(env.cfg.decimation)
    step_dt = sim_dt * decimation
    print(f"SIM_DT: {sim_dt}, DECIMATION: {decimation}, STEP_DT: {step_dt}")

    # physics_dt attribute may or may not exist on env (informational)
    for attr in ("physics_dt", "step_dt"):
        if hasattr(env, attr):
            print(f"  env.{attr} = {getattr(env, attr)}")

    print("=====JABIS_INSPECT_END=====")

    env.close()
    launcher.app.close()


if __name__ == "__main__":
    main()
