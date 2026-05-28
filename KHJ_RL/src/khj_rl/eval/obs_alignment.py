"""Guard: env reset obs layout must match the single-source spec in ``env_cfg``.

The jabis_sim_v2 incident was caused by sim2real wrapper baseline-joint
constants drifting from the env baseline, sending the policy radians off
its training distribution. This guard runs at training session start and at
every eval, comparing the env's reset obs against ``CubeLiftEnvCfg.obs_keys``
/ ``obs_sizes`` — the same source of truth the policy was built against.
"""

from __future__ import annotations

import numpy as np

from khj_rl.envs.cube_lift.cfg import CubeLiftEnvCfg


def assert_obs_alignment(
    env,
    cfg: CubeLiftEnvCfg,
    atol: float = 1e-3,
) -> None:
    """Raises ``ValueError`` if the env's reset obs doesn't match ``cfg``.

    Checks:
    - return type is ``np.ndarray``
    - dtype is ``float32``
    - flat shape equals ``cfg.obs_total_size`` (sum of obs_sizes over obs_keys)
    - all finite (no NaN / inf)

    ``atol`` reserved for per-key range checks once a recorded policy spec
    can be passed in alongside ``cfg``.
    """
    obs = env.reset()
    if not isinstance(obs, np.ndarray):
        raise ValueError(
            f"env.reset() returned {type(obs).__name__}, expected np.ndarray"
        )
    if obs.dtype != np.float32:
        raise ValueError(
            f"env.reset() obs dtype is {obs.dtype}, expected float32 "
            f"(matches policy input dtype)"
        )
    expected_len = cfg.obs_total_size
    if obs.shape != (expected_len,):
        raise ValueError(
            f"env.reset() obs shape {obs.shape} != expected ({expected_len},). "
            f"obs_keys={cfg.obs_keys}, obs_sizes={cfg.obs_sizes}"
        )
    if not np.isfinite(obs).all():
        raise ValueError("env.reset() obs contains non-finite values")
    # atol reserved.
    _ = atol


def expected_obs_size(cfg: CubeLiftEnvCfg) -> int:
    return cfg.obs_total_size


def assert_obs_distribution(
    env,
    cfg: CubeLiftEnvCfg,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    n_samples: int = 16,
    sigma_tolerance: float = 6.0,
    degenerate_abs_tol: float = 1e-3,
    degenerate_std_threshold: float = 1e-6,
) -> None:
    """Verify the env's reset obs lives inside the policy's training
    distribution (BC dataset's mean / std).

    ``mean`` and ``std`` come from the frozen ``ObsNormalizer`` written
    at the end of BC pretrain — they capture the obs distribution the
    policy was trained against. At each training session start (and at
    every eval) we sample ``n_samples`` resets and check that every
    obs dim sits inside ``mean ± sigma_tolerance * std``. ``6 σ`` is
    deliberately generous (we don't want to crash training over noise
    tails); a hit is almost certainly a real distribution shift, e.g.
    a sim2real wrapper rewriting an env constant the policy never saw.

    Dims with std < ``degenerate_std_threshold`` (e.g. ``normalized_t``
    is always 0 at reset) get an absolute-tolerance check instead of a
    z-score — z-score would either divide by ~0 or get clipped to a
    no-op, both of which silently hide real drift on the very signals
    that are most diagnostic for sim2real wrappers (a wrapper that
    forgets to reset a counter would shift a "constant" obs dim).

    Raises ``ValueError`` with a per-key breakdown when the check fails
    so the operator can see which obs component drifted.
    """
    if mean.shape != (cfg.obs_total_size,) or std.shape != (cfg.obs_total_size,):
        raise ValueError(
            f"mean/std shape mismatch: got {mean.shape} / {std.shape}, "
            f"expected ({cfg.obs_total_size},)"
        )
    samples = np.stack([env.reset() for _ in range(n_samples)], axis=0)
    deviation = np.abs(samples - mean[None, :])
    is_degenerate = std <= degenerate_std_threshold
    safe_std = np.where(is_degenerate, 1.0, std)
    z = (deviation / safe_std[None, :]).max(axis=0)
    abs_dev = deviation.max(axis=0)
    # Non-degenerate dims fail when |z| > sigma_tolerance; degenerate
    # dims fail when the absolute deviation exceeds degenerate_abs_tol.
    bad = np.where(is_degenerate, abs_dev > degenerate_abs_tol, z > sigma_tolerance)
    if not np.any(bad):
        return
    # Walk obs_keys to point at the offending block — "joint_pos[3]"
    # tells the operator which observation source drifted.
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    sizes = cfg.obs_sizes
    for k in cfg.obs_keys:
        n = sizes[k]
        offsets[k] = (cursor, cursor + n)
        cursor += n
    failures: list[str] = []
    for k, (a, b) in offsets.items():
        idxs = np.where(bad[a:b])[0]
        for i in idxs:
            j = a + int(i)
            if is_degenerate[j]:
                failures.append(
                    f"  {k}[{int(i)}] (degenerate std): max |obs - mean| = "
                    f"{abs_dev[j]:.4f} > {degenerate_abs_tol} "
                    f"(mean={mean[j]:.4f}, std={std[j]:.4f})"
                )
            else:
                failures.append(
                    f"  {k}[{int(i)}]: max |z| = {z[j]:.2f} > "
                    f"{sigma_tolerance} (mean={mean[j]:.4f}, std={std[j]:.4f})"
                )
    raise ValueError(
        "env reset obs is outside the policy's BC training distribution "
        f"(across {n_samples} resets):\n" + "\n".join(failures)
    )
