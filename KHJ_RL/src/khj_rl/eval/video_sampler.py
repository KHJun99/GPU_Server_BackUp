"""Periodic rollout video capture.

Mandatory Phase 1 guard against the jabis_sim_v2 reward-hacking failure
mode (success rate climbed while the policy was rolling the cube; the
metric never caught it because nobody was looking at the rendered
output). The trainer calls ``VideoSampler.maybe_capture`` every step;
when ``step % every_steps == 0`` the sampler runs ``rollout_episodes``
episodes through the eval env, scrapes the third-person ``viewer_camera``
sensor each policy step, and writes the result to MP4 at the policy
rate (20 Hz) so the operator can scrub through it.

The eval env passed in here MUST be constructed with
``include_viewer_camera=True`` — without the camera sensor we silently
skip the sample rather than crash the training loop (operator can
notice an empty output directory).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Type alias for the policy callable. Takes a flat obs ndarray and
# returns a flat action ndarray. Both BC and PPO actor wrappers expose
# this interface so the sampler doesn't need a torch import.
Policy = Callable[[np.ndarray], np.ndarray]


@dataclass
class VideoSampler:
    output_dir: Path
    every_steps: int = 5000
    rollout_episodes: int = 4
    fps: int = 20
    # Tracks the last step at which we captured. Threshold-based
    # trigger (vs ``step % every_steps == 0``) because PPO advances
    # ``global_step`` in chunks of ``num_steps`` (default 2048), so a
    # modulo check against 5000 would never line up exactly and the
    # sampler would silently never fire.
    _last_capture_step: int = -1

    def maybe_capture(
        self,
        step: int,
        env: Any,
        policy: Policy,
    ) -> Path | None:
        """If ``step`` triggers a capture, run rollouts and save MP4.

        Returns the written file path, or ``None`` if the step doesn't
        trigger a sample or the env has no viewer camera.
        """
        if step <= 0:
            return None
        if self._last_capture_step >= 0 and step - self._last_capture_step < self.every_steps:
            return None
        cam = getattr(env, "_viewer_camera", None)
        if cam is None:
            # Silently skip — caller built the env without a viewer
            # camera. We don't want to bring down training over this.
            return None

        # Lazy import so importing this module never pulls imageio /
        # ffmpeg into a process that doesn't intend to write video.
        import imageio.v2 as imageio

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"rollout_step_{step:08d}.mp4"

        frames: list[np.ndarray] = []
        for _ in range(self.rollout_episodes):
            obs = env.reset()
            done = False
            while not done:
                action = policy(obs)
                obs, _reward, term, trunc, _info = env.step(action)
                rgb = self._read_camera_rgb(cam)
                if rgb is not None:
                    frames.append(rgb)
                done = bool(term) or bool(trunc)

        if not frames:
            return None

        # macro_block_size=1 lets ffmpeg accept any resolution (default
        # 16-pixel block alignment would refuse the 640x480 camera
        # output and crash the writer mid-training).
        imageio.mimsave(out_path, frames, fps=self.fps, macro_block_size=1)
        self._last_capture_step = step
        return out_path

    @staticmethod
    def _read_camera_rgb(cam: Any) -> np.ndarray | None:
        """Return an HxWx3 uint8 RGB frame from an Isaac Lab Camera sensor,
        or ``None`` if the buffer isn't populated yet (the very first
        frame after reset can be empty before the renderer commits)."""
        data = getattr(cam, "data", None)
        if data is None:
            return None
        out = getattr(data, "output", None)
        if out is None or "rgb" not in out:
            return None
        rgb = out["rgb"]
        # Sensor buffer shape (num_envs, H, W, C). Take env 0; drop the
        # alpha channel if present.
        try:
            tensor = rgb[0]
        except Exception:
            return None
        arr = tensor.detach().cpu().numpy()
        if arr.ndim != 3:
            return None
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            # Camera sometimes returns float in [0, 1]; convert.
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        return arr
