"""Debug: cfg 의 baseline_joint_pos 가 sim runtime 의 robot.data.default_joint_pos
에 실제로 반영되는지 확인."""
import sys
from isaaclab.app import AppLauncher

import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])

launcher = AppLauncher(args)

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from jabis_sim_v2.tasks.cube_lift.joint_pos_env_cfg import (  # noqa: E402
    SoArm101CubeLiftEnvCfg,
)


def main():
    print("=====DEBUG_BEGIN=====", flush=True)
    cfg = SoArm101CubeLiftEnvCfg()
    cfg.scene.num_envs = 1
    print(f"[cfg] robot.init_state.joint_pos = {cfg.scene.robot.init_state.joint_pos}",
          flush=True)
    print(f"[cfg] robot.init_state.pos = {cfg.scene.robot.init_state.pos}",
          flush=True)
    print(f"[cfg] mirror_arm.init_state.joint_pos = "
          f"{cfg.scene.mirror_arm.init_state.joint_pos}", flush=True)

    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    robot = env.scene["robot"]
    print(f"[runtime] robot.joint_names = {robot.data.joint_names}", flush=True)
    default = robot.data.default_joint_pos[0].cpu().numpy().tolist()
    joints = robot.data.joint_pos[0].cpu().numpy().tolist()
    print(f"[runtime] robot.default_joint_pos = {[round(x, 4) for x in default]}",
          flush=True)
    print(f"[runtime] robot.joint_pos (after reset) = "
          f"{[round(x, 4) for x in joints]}", flush=True)

    mirror = env.scene["mirror_arm"]
    m_default = mirror.data.default_joint_pos[0].cpu().numpy().tolist()
    m_joints = mirror.data.joint_pos[0].cpu().numpy().tolist()
    print(f"[runtime] mirror.default_joint_pos = {[round(x, 4) for x in m_default]}",
          flush=True)
    print(f"[runtime] mirror.joint_pos (after reset) = "
          f"{[round(x, 4) for x in m_joints]}", flush=True)
    print("=====DEBUG_END=====", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    launcher.app.close()
