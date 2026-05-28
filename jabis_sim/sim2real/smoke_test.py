"""Smoke test for SoArm101LiftPolicy + SoArm101ActionDecoder."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from action_decoder import SoArm101ActionDecoder
from policy_inference import SoArm101LiftPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smoke_test")

DEFAULT_CHECKPOINT = Path(
    "/home/j-k14d101/isaac_so_arm101/logs/rsl_rl/lift/2026-05-08_10-58-49/model_3999.pt"
)
LATENCY_ITERS = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                   help="path to model_*.pt (default: workstation training run)")
    p.add_argument("--device", type=str, default="cpu",
                   help="torch device, e.g. 'cpu' or 'cuda:0'")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(
            f"checkpoint not found: {args.checkpoint}\n"
            "pass --checkpoint /path/to/model_XXXX.pt"
        )
    logger.info("=== 1. Loading policy ===")
    policy = SoArm101LiftPolicy(args.checkpoint, device=args.device)

    logger.info("=== 2. Forward passes on dummy obs ===")
    cases: dict[str, torch.Tensor] = {
        "zeros (1D)": torch.zeros(policy.OBS_DIM),
        "ones  (1D)": torch.ones(policy.OBS_DIM),
        "randn (1D)": torch.randn(policy.OBS_DIM),
        "randn (B=4)": torch.randn(4, policy.OBS_DIM),
    }
    for name, obs in cases.items():
        action = policy(obs)
        logger.info(
            "  %s -> action shape=%s dtype=%s mean=%.4f std=%.4f",
            name, tuple(action.shape), action.dtype, action.float().mean().item(),
            action.float().std().item() if action.numel() > 1 else 0.0,
        )

    logger.info("=== 3a. Decoded servo command (zeros obs) ===")
    decoder = SoArm101ActionDecoder()
    raw = policy(torch.zeros(policy.OBS_DIM))
    cmd = decoder.decode(raw)
    for joint, value in cmd.items():
        logger.info("  %-15s = %+.4f rad", joint, value)

    logger.info("=== 3b. Decoded servo command (ones obs — expect clamps) ===")
    raw_ones = policy(torch.ones(policy.OBS_DIM))
    logger.info("  raw policy action: %s",
                ["%+.3f" % v for v in raw_ones.tolist()])
    cmd_ones = decoder.decode(raw_ones)
    for joint, value in cmd_ones.items():
        lo, hi = decoder.joint_limits[joint]
        flag = "  (at limit)" if value in (lo, hi) else ""
        logger.info("  %-15s = %+.4f rad%s", joint, value, flag)

    logger.info("=== 4. Latency (%s, %d iters, single obs) ===",
                args.device, LATENCY_ITERS)
    obs = torch.randn(policy.OBS_DIM)
    is_cuda = policy.device.type == "cuda"
    for _ in range(5):
        _ = policy(obs)
    if is_cuda:
        torch.cuda.synchronize(policy.device)
    t0 = time.perf_counter()
    for _ in range(LATENCY_ITERS):
        _ = policy(obs)
    if is_cuda:
        torch.cuda.synchronize(policy.device)
    elapsed = time.perf_counter() - t0
    mean_ms = (elapsed / LATENCY_ITERS) * 1000.0
    logger.info(
        "  total=%.3fs  mean=%.3f ms/iter  (~%.0f Hz)",
        elapsed, mean_ms, 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
    )

    logger.info("smoke test PASSED")


if __name__ == "__main__":
    main()
