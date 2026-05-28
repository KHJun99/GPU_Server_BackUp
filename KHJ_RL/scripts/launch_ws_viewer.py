"""HTTP MJPEG viewer for CubeLiftEnv.

The original WebRTC path (``scripts/launch_viewer.py``) segfaults in
``isaacsim.asset.importer.urdf`` on this NVIDIA 4.5 install — a race we
can't patch externally.

The follow-up WebSocket version of this script worked in raw SSH/network
setups but broke under VSCode's port-forwarding tunnel: that proxy strips
the ``Connection: Upgrade`` header, so the browser's WS handshake never
reaches our server intact and we get an immediate ``empty Connection
header`` rejection. Switching the transport to **MJPEG over plain HTTP**
sidesteps the issue entirely — no Upgrade negotiation, just a long-lived
``multipart/x-mixed-replace`` response that every browser and every HTTP
proxy already understands.

Single port, single file. Open the forwarded URL in any browser
(``http://localhost:8765/`` once VSCode port-forwards 8765); the HTML
page is served at ``/`` and the live JPEG stream at ``/stream``.

Server:
    CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \\
        python scripts/launch_ws_viewer.py --mode oracle --port 8765

Multi-GPU host note: this host has 4 L40S. PhysX runs on cuda:0 via
``CUDA_VISIBLE_DEVICES=1`` masking, but the Omniverse renderer enumerates
GPUs via NVML directly and ignores that mask — so we pin the renderer to
physical GPU 1 via ``--/renderer/activeGpu=1`` to keep PhysX and rendering
on the same device (otherwise rendering deadlocks against the other
users' lerobot training pinning GPU 0 at 100% util).
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("KHJ_RL_SIM_DEVICE", "cuda:0")

import pinocchio  # noqa: F401, E402  (pre-AppLauncher, see launch_viewer.py)

from isaaclab.app import AppLauncher  # noqa: E402


_HTML_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>KHJ_RL CubeLift Viewer</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; padding: 16px; background: #111; color: #ddd;
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  h1 { margin: 0 0 8px; font-size: 14px; font-weight: 500; color: #888;
       font-family: ui-monospace, Menlo, monospace; }
  img { display: block; max-width: 100%; background: #000;
        border: 1px solid #333; border-radius: 4px; image-rendering: -webkit-optimize-contrast; }
  .stats { margin-top: 8px; font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: #888; }
</style></head>
<body>
  <h1>KHJ_RL CubeLift -- polling <code>/latest.jpg</code> at 20 Hz</h1>
  <img id="viewport" alt="awaiting frames..." />
  <div class="stats" id="stats">starting...</div>
<script>
  // Poll a single-shot endpoint instead of using multipart/x-mixed-replace.
  // VSCode port-forwarding buffers/rejects multipart responses, so we
  // drive the img from individual fetches - each a normal request/response
  // that every HTTP proxy understands.
  const img = document.getElementById("viewport");
  const stats = document.getElementById("stats");
  let frameCount = 0, lastTick = performance.now(), lastCount = 0;
  let inFlight = false;
  async function tick() {
    if (inFlight) return;  // drop frames when the previous fetch hasn't returned yet
    inFlight = true;
    try {
      const res = await fetch(`/latest.jpg?t=${Date.now()}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const prev = img.src;
      img.onload = () => {
        if (prev && prev.startsWith("blob:")) URL.revokeObjectURL(prev);
      };
      img.src = url;
      frameCount += 1;
    } catch (e) {
      stats.textContent = `error: ${e.message}`;
    } finally {
      inFlight = false;
    }
  }
  setInterval(tick, 50);  // 20 Hz - matches sim policy rate
  setInterval(() => {
    const now = performance.now();
    const fps = (frameCount - lastCount) / ((now - lastTick) / 1000);
    lastTick = now; lastCount = frameCount;
    stats.textContent = `fps: ${fps.toFixed(1)} | frames: ${frameCount}`;
  }, 1000);
</script>
</body></html>
""".encode("utf-8")


# Shared state — updated by the sim thread, read by the HTTP stream handler.
_state = {
    "jpeg": None,
    "frame_idx": 0,
    "stop": False,
}
_state_lock = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    """Serve the HTML page at `/` and frames at `/latest.jpg` (poll) or
    `/stream` (multipart MJPEG, when the network path allows it)."""

    # HTTP/1.1 so VSCode's devtunnels.ms reverse proxy doesn't reject our
    # responses. The Python stdlib default of HTTP/1.0 produced 504 Gateway
    # Timeout when accessed through devtunnels' public URL; HTTP/1.1 +
    # explicit Content-Length / Connection: close fixes it.
    protocol_version = "HTTP/1.1"

    # The default log_message dumps a line per request — at 20 frames/s the
    # terminal gets unreadable.
    def log_message(self, *args, **kwargs):  # noqa: ANN001  (stdlib signature)
        return

    def do_HEAD(self) -> None:  # noqa: N802  (stdlib casing)
        # Reverse proxies often probe with HEAD before forwarding GET.
        # Replying with the same headers (minus body) avoids the proxy
        # marking us as offline.
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_HTML_PAGE)))
            self.send_header("Connection", "close")
            self.end_headers()
            return
        if self.path.startswith("/latest.jpg"):
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Connection", "close")
            self.end_headers()
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802  (stdlib casing)
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_HTML_PAGE)))
            self.end_headers()
            self.wfile.write(_HTML_PAGE)
            return
        if self.path == "/stream":
            self._stream_mjpeg()
            return
        if self.path.startswith("/latest.jpg"):
            self._serve_latest_jpg()
            return
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"not found\n")

    def _serve_latest_jpg(self) -> None:
        """Single-shot JPEG of the most recent frame (polled by the browser)."""
        with _state_lock:
            jpeg = _state["jpeg"]
        if jpeg is None:
            # Sim hasn't produced its first frame yet — return 204 No Content
            # so the client's <img> keeps its previous (or empty) src and just
            # retries on the next tick.
            self.send_response(204)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(jpeg)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _stream_mjpeg(self) -> None:
        boundary = b"frame"
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={boundary.decode()}"
        )
        self.end_headers()
        last_sent = -1
        try:
            while not _state["stop"]:
                with _state_lock:
                    jpeg = _state["jpeg"]
                    idx = _state["frame_idx"]
                if jpeg is None or idx == last_sent:
                    time.sleep(0.02)
                    continue
                last_sent = idx
                # Each frame: --boundary, Content-Type, Content-Length, blank, bytes.
                hdr = (
                    b"\r\n--"
                    + boundary
                    + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode()
                    + b"\r\n\r\n"
                )
                self.wfile.write(hdr)
                self.wfile.write(jpeg)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("oracle", "random", "zero", "sweep"),
        default="oracle",
        help="Action source. 'sweep' drives each arm joint with a sine wave"
        " (different frequencies) to sanity-check the render pipeline -"
        " if the arm visibly waves around, the viewer + sim wiring is fine"
        " and any 'no movement' symptom in oracle mode is purely an IK issue.",
    )
    p.add_argument("--port", type=int, default=8765, help="HTTP port.")
    p.add_argument(
        "--host",
        default="0.0.0.0",
        help="HTTP bind address (0.0.0.0 = all interfaces).",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=70,
        help="JPEG quality 1-100 (lower = smaller, higher = sharper).",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=-1,
        help="Episode count (-1 = loop forever until Ctrl+C).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    sim_app = AppLauncher(
        headless=True,
        enable_cameras=True,
        kit_args=(
            "--/renderer/multiGpu/enabled=false "
            "--/renderer/multiGpu/maxGpuCount=1 "
            "--/renderer/activeGpu=1"
        ),
    ).app
    print("[viewer] kit booted", flush=True)

    import cv2
    import numpy as np

    from khj_rl.envs import CubeLiftEnv, CubeLiftEnvCfg
    from khj_rl.envs.cube_lift.oracle import MotionPlanningOracle

    cfg = CubeLiftEnvCfg()
    env = CubeLiftEnv(cfg, include_viewer_camera=True)
    oracle = MotionPlanningOracle(env) if args.mode == "oracle" else None
    print(f"[viewer] env ready (mode={args.mode})", flush=True)

    def _capture_frame() -> None:
        cam = env._viewer_camera
        if cam is None:
            return
        # Prefer the replicator annotator if available; the Camera class's
        # data.output['rgb'] doesn't refresh on this isaacsim 4.5 build.
        rep_annot = getattr(env, "_rep_rgb_annot", None)
        if rep_annot is not None:
            data = rep_annot.get_data()
            if data is None or (hasattr(data, "size") and data.size == 0):
                return
            rgb = np.asarray(data)
        else:
            cam.update(env._physics_dt, force_recompute=True)
            rgb_buf = cam.data.output.get("rgb")
            if rgb_buf is None:
                return
            rgb = rgb_buf[0].detach().cpu().numpy()
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # Per-frame index overlay so we can tell at-a-glance whether what the
        # browser displays is fresh or a cached/repeated frame (devtunnels
        # ignores the cache-buster query param on some requests). Burns the
        # number into the pixels so any proxy caching is immediately visible.
        idx_text = f"#{_state['frame_idx']:06d}"
        # Pull oracle phase/step/distance if the oracle stashed them
        # (oracle.act sets these every step; absent for random/zero modes).
        phase = getattr(env, "_oracle_dbg_phase", "n/a")
        ostep = getattr(env, "_oracle_dbg_step", 0)
        d = getattr(env, "_oracle_dbg_ee_to_target", float("nan"))
        info_text = f"{phase} step={ostep} d={d*100:.1f}cm" if oracle is not None else ""
        for txt, y in ((idx_text, 36), (info_text, 70)):
            if not txt:
                continue
            cv2.putText(
                bgr, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 4, cv2.LINE_AA,
            )
            cv2.putText(
                bgr, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2, cv2.LINE_AA,
            )
        ok, buf = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
        )
        if not ok:
            return
        with _state_lock:
            _state["jpeg"] = buf.tobytes()
            _state["frame_idx"] += 1

    server = ThreadingHTTPServer((args.host, args.port), _Handler)

    def _serve():
        print(
            f"[viewer] listening on http://{args.host}:{args.port}/  "
            f"(VSCode → forward port {args.port} → browser to http://localhost:{args.port}/)",
            flush=True,
        )
        server.serve_forever()

    server_thread = threading.Thread(target=_serve, daemon=True)
    server_thread.start()

    # Sweep mode runs a self-contained loop that bypasses env.step entirely:
    # each tick we (1) write absolute joint positions, (2) set them as the
    # actuator target so the PD loop doesn't fight us, (3) step physics +
    # render, (4) push the frame. env.step would otherwise re-read a stale
    # joint_pos buffer and overwrite our sweep value with the previous
    # target, hiding all motion.
    if args.mode == "sweep":
        import torch as _torch
        env.reset()
        # Enumerate every Camera prim on the stage so we can see which
        # one Isaac Lab actually rendered + read pose. Sometimes
        # CameraCfg wraps a deeper prim; data.pos_w may not be the right
        # field if so.
        from pxr import UsdGeom  # noqa
        stage = env._sim.stage
        print("[viewer] === stage camera prims ===", flush=True)
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                xf = UsdGeom.Xformable(prim)
                xform = xf.ComputeLocalToWorldTransform(0)
                t = xform.ExtractTranslation()
                print(f"[viewer]   {prim.GetPath()}: world_pos=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})", flush=True)
        cam = env._viewer_camera
        cam_pos_w = cam.data.pos_w[0].detach().cpu().numpy().round(3).tolist()
        ee_pos_w = env._robot.data.body_pos_w[0, env._ee_body_idx].detach().cpu().numpy().round(3).tolist()
        print(
            f"[viewer] data.pos_w={cam_pos_w} ee_w={ee_pos_w} "
            f"viewer_cam_prim={cam.cfg.prim_path}",
            flush=True,
        )

        # TOP-DOWN camera, identity rotation = looks down -Z.
        from pxr import Gf, UsdGeom
        cam_prim = stage.GetPrimAtPath(cam.cfg.prim_path)
        xf = UsdGeom.Xformable(cam_prim)
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(0.25, 0.0, 1.5))
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                prec = op.GetPrecision()
                q = Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0)) if prec == UsdGeom.XformOp.PrecisionDouble else Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
                op.Set(q)
        env._scene.write_data_to_sim()
        env._sim.step(render=True)
        env._scene.update(dt=env._physics_dt)
        print("[viewer] TOP-DOWN cam set at (0.25, 0, 1.5)", flush=True)

        # === REPLICATOR PATH: bypass Camera class entirely ===
        # The Camera class's data.output['rgb'] was never refreshing —
        # every pose write produced cam_mean=52.3 even with use_fabric=True,
        # sim.forward(), cam.update(force_recompute=True), and sim_app.update()
        # called explicitly. Create our own render product + annotator
        # against the same camera prim; this is the canonical headless
        # capture path in Isaac Sim and is independent of the Camera sensor.
        import omni.replicator.core as rep
        render_product = rep.create.render_product(
            cam.cfg.prim_path, resolution=(640, 480)
        )
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach([render_product])
        print(f"[viewer] replicator render product attached to {cam.cfg.prim_path}", flush=True)

        # Replace _capture_frame's RGB source with the annotator. We mutate
        # the closure-captured state inside the function by stashing the
        # annotator on env so _capture_frame can find it.
        env._rep_rgb_annot = rgb_annot
        # Drive replicator one step so the annotator has data.
        rep.orchestrator.step()
        print("[viewer] sweep mode: 4 discrete poses, 5s each", flush=True)
        # 4 visibly distinct arm configurations (well inside soft limits).
        poses = [
            ("home",       [ 0.0, -0.5,  1.0,  0.5, 0.0]),
            ("pan-right",  [ 1.5, -0.5,  1.0,  0.5, 0.0]),
            ("pan-left",   [-1.5, -0.5,  1.0,  0.5, 0.0]),
            ("arm-up",     [ 0.0,  1.0,  0.0, -1.0, 0.0]),
        ]
        HOLD_STEPS = 100  # 5 s at policy rate (20 Hz)
        try:
            while True:
                for name, joints in poses:
                    desired = env._robot.data.joint_pos.clone()
                    for slot, jidx in enumerate(env._arm_joint_idxs):
                        desired[:, jidx] = float(joints[slot])
                    env._robot.write_joint_state_to_sim(
                        position=desired,
                        velocity=_torch.zeros_like(desired),
                    )
                    env._robot.set_joint_position_target(desired)
                    # CRUCIAL: env.reset uses sim.forward() to sync physics
                    # state -> USD prims so the renderer sees the new
                    # joint configuration. Without it, write_joint_state
                    # only updates the PhysX state and the render product
                    # keeps showing the previous arm pose -- which is why
                    # cam_mean was identical across every pose.
                    env._scene.write_data_to_sim()
                    env._sim.forward()
                    env._scene.update(dt=env._physics_dt)
                    actual = env._robot.data.joint_pos[0, env._arm_joint_idxs].detach().cpu().numpy().round(3).tolist()
                    cam = env._viewer_camera
                    rgb = cam.data.output.get("rgb") if cam.data.output else None
                    cam_mean = float(rgb[0].detach().cpu().float().mean()) if rgb is not None else -1
                    print(
                        f"[viewer] pose='{name}' actual={actual} cam_mean={cam_mean:.1f}",
                        flush=True,
                    )
                    for _ in range(HOLD_STEPS):
                        env._scene.write_data_to_sim()
                        env._sim.step(render=True)
                        env._scene.update(dt=env._physics_dt)
                        env._force_mimic_coupling()
                        # Pump replicator orchestrator + Kit message loop so
                        # the annotator pulls a fresh frame from the render
                        # product. sim.step(render=True) updated the render
                        # product but the annotator's get_data() needs the
                        # orchestrator step to actually return new bytes.
                        import omni.replicator.core as rep
                        rep.orchestrator.step()
                        sim_app.update()
                        _capture_frame()
        except KeyboardInterrupt:
            pass
        finally:
            with _state_lock:
                _state["stop"] = True
            server.shutdown()
            sim_app.close()
        return 0

    ep = 0
    try:
        while args.episodes < 0 or ep < args.episodes:
            obs = env.reset()
            if oracle is not None:
                oracle.reset(obs, {})
            _capture_frame()
            info: dict = {}
            phase_seen = "approach" if oracle is not None else "n/a"
            for step in range(cfg.control.max_steps):
                if args.mode == "oracle":
                    action = oracle.act(obs, info)
                elif args.mode == "random":
                    action = env.action_space.sample()
                else:
                    action = np.zeros(cfg.action_size, dtype=np.float32)
                obs, reward, term, trunc, info = env.step(action)
                _capture_frame()
                if oracle is not None and oracle.phase_name != phase_seen:
                    phase_seen = oracle.phase_name
                    print(
                        f"[viewer] ep={ep} step={step} → phase={phase_seen}",
                        flush=True,
                    )
                if term or trunc:
                    print(
                        f"[viewer] ep={ep} done "
                        f"(steps={step + 1} term={term} trunc={trunc} reward={reward})",
                        flush=True,
                    )
                    break
            ep += 1
            for _ in range(10):
                _capture_frame()
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[viewer] Ctrl+C — shutting down.", flush=True)
    finally:
        with _state_lock:
            _state["stop"] = True
        server.shutdown()
        sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
