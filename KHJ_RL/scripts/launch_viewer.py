"""Live-stream the CubeLiftEnv via Isaac Sim WebRTC 2.x.

Boots the sim with ``AppLauncher(headless=True, livestream=2)`` so a
browser or the NVIDIA Omniverse Streaming Client can connect to watch
the scene without an X display. Runs an infinite reset → episode loop
driven by either the MP oracle or random actions, so you can eyeball
where the oracle is failing (current symptom: episodes stuck in `move`).

Server side (this script):
    OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y CUDA_VISIBLE_DEVICES=1 \\
        python scripts/launch_viewer.py --mode oracle

    # quit with Ctrl+C

Ports opened by Isaac Sim WebRTC 2.x (must be reachable from the client):
    - 8211 (TCP) HTTP signaling
    - 49100 (TCP) signaling
    - 47995-48012, 49100-49200 (UDP) media
    Check / open them on the GPU server's firewall before connecting.

Client side: NVIDIA Omniverse Streaming Client (browser direct-access
to 8211 returns AccessDenied — that endpoint is signaling-only, not a
hosted client page).
    1) Download Omniverse Streaming Client (Win/Mac/Linux) from
       https://docs.omniverse.nvidia.com/streaming-client/
    2) Launch it, enter ``<server-ip>``, hit Connect.
    3) The actual signaling port is printed by Isaac Sim at boot —
       look for a "WebRTC ..." line in the script's stdout if the
       client fails to connect.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# GPU isolation — this host has 4× L40S; we own GPU 1. CUDA_VISIBLE_DEVICES
# alone doesn't fully isolate: Omniverse's renderer enumerates GPUs via EGL
# *and* NVIDIA's runtime, sees all 4, and starts multi-GPU rendering even
# when CUDA only sees one. That causes 49 ms inter-GPU latency and
# write_joint_state_to_sim hangs. Hide GPU 1 on all three visibility paths
# so EGL/Vulkan/CUDA agree on a single device, and disable X11 in the EGL
# layer (jupyter07 is headless — DISPLAY is unset; OMNI_EGL_NO_X11 stops
# the renderer from probing /tmp/.X11-unix and producing GLFW warnings).
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("KHJ_RL_SIM_DEVICE", "cuda:0")

# Import pinocchio BEFORE AppLauncher: pinocchio's libhpp-fcl.so links
# against system Assimp, but isaacsim ships its own Assimp build whose
# C++ name-mangling for ``Assimp::IOSystem::CurrentDirectory`` differs.
# If isaacsim's Assimp loads first, pinocchio's import fails with an
# undefined-symbol error. Pre-loading pinocchio binds against the system
# Assimp before kit's loader gets to it.
import pinocchio  # noqa: F401, E402

from isaaclab.app import AppLauncher  # bootstrap  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("oracle", "random", "zero"),
        default="oracle",
        help="Action source: MP oracle (default), random, or zero action.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=-1,
        help="Number of episodes (-1 = loop forever until Ctrl+C).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    # livestream=2 → WebRTC 2.x. The isaaclab `.python.rendering.kit`
    # experience left the WebRTC stream connected but produced a black
    # frame (viewport window never registered — UI only showed View /
    # About, no viewport widget). NVIDIA ships a dedicated streaming
    # experience inside isaacsim that wires up viewport + WebRTC for
    # exactly this case; use it.
    _EXPERIENCE = (
        "/home/j-k14d101/.conda/envs/khj-rl/lib/python3.10/site-packages/"
        "isaacsim/apps/isaacsim.exp.full.streaming.kit"
    )
    # Single-GPU. CUDA_VISIBLE_DEVICES=1 already hides the other three
    # GPUs from CUDA, but Omniverse still tries multi-GPU rendering and
    # warns about the unreachable devices, so keep the kit-side flags
    # too — they map the one visible GPU as renderer device 0.
    _KIT_ARGS = (
        "--/renderer/multiGpu/enabled=false "
        "--/renderer/multiGpu/maxGpuCount=1 "
        "--/renderer/activeGpu=1 "
        # Force PhysX to write joint/body transforms back to USD every
        # step. Default is False because fabric handles this faster; on
        # our install fabric crashes (ABI mismatch) so without this
        # setting the streaming viewport only sees the initial reset
        # pose — arm motion never appears.
        "--/physics/updateToUsd=true "
        "--/physics/updateParticlesToUsd=true "
        "--/physics/updateForceSensorsToUsd=true "
        "--/physics/outputVelocitiesLocalSpace=false"
    )
    sim_app = AppLauncher(
        headless=True,
        livestream=2,
        enable_cameras=True,
        experience=_EXPERIENCE,
        kit_args=_KIT_ARGS,
    ).app
    print(
        "[viewer] Sim livestream ready. Connect a WebRTC 2.x client to "
        "this host (browser at http://<server>:8211 or Omniverse Streaming "
        "Client). Ctrl+C to quit.",
        flush=True,
    )

    # Lazy imports after AppLauncher boot.
    import numpy as np
    from isaacsim.core.utils.viewports import set_camera_view

    from khj_rl.envs import CubeLiftEnv, CubeLiftEnvCfg
    from khj_rl.envs.cube_lift.oracle import MotionPlanningOracle

    cfg = CubeLiftEnvCfg()
    print("[trace] BEFORE env init", flush=True)
    env = CubeLiftEnv(cfg)
    print("[trace] AFTER env init", flush=True)
    # Default viewport camera looks at world origin from far away and
    # the cube-lift scene falls outside the frame (black streamed image).
    # Move it to a corner-of-table angle so robot + cube + goal are all
    # visible; tweak after seeing the result if needed.
    set_camera_view(eye=(0.7, 0.5, 0.5), target=(0.2, 0.0, 0.05))
    print("[trace] AFTER set_camera_view", flush=True)
    oracle = MotionPlanningOracle(env) if args.mode == "oracle" else None
    print("[trace] AFTER oracle init", flush=True)

    ep = 0
    try:
        while args.episodes < 0 or ep < args.episodes:
            print(f"[trace] BEFORE env.reset ep={ep}", flush=True)
            obs = env.reset()
            print(f"[trace] AFTER env.reset ep={ep}", flush=True)
            if oracle is not None:
                oracle.reset(obs, {})
            info: dict = {}
            phase_seen = "approach"
            for step in range(cfg.control.max_steps):
                if args.mode == "oracle":
                    action = oracle.act(obs, info)
                elif args.mode == "random":
                    action = env.action_space.sample()
                else:
                    action = np.zeros(cfg.action_size, dtype=np.float32)

                obs, reward, term, trunc, info = env.step(action)
                if oracle is not None and oracle.phase_name != phase_seen:
                    phase_seen = oracle.phase_name
                    print(f"[viewer] ep={ep} step={step} → phase={phase_seen}", flush=True)
                if term or trunc:
                    per = info.get("success_per_condition", {})
                    per_str = " ".join(f"{k}={int(v)}" for k, v in per.items())
                    print(
                        f"[viewer] ep={ep} done (steps={step + 1} "
                        f"term={term} trunc={trunc} reward={reward}) "
                        f"per_cond[{per_str}]",
                        flush=True,
                    )
                    break
            ep += 1
            # Small pause so the operator sees the final pose before reset.
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[viewer] Ctrl+C — shutting down.", flush=True)
    finally:
        sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
