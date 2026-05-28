"""Evaluate a BC or PPO checkpoint in sim.

Loads the saved ActorCritic + frozen ObsNormalizer, rolls out N
episodes through the cube-lift env, and reports the success rate +
per-condition breakdown. Phase 1 결정 사항 7 mandates ≥70% success
on 100 eval episodes before the goal-space curriculum advances —
this script produces that number.

The eval policy is **deterministic** (actor_mean only, no
exploration noise) to match what the operator will eventually
deploy on real hardware. Stochastic eval (action sampling) can be
added with --stochastic if needed for confidence-interval analysis.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \\
        python scripts/eval_policy.py \\
            --ckpt runs/stage0_bc_1k/bc.pt \\
            --episodes 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# CUDA / pinocchio bootstrap matches collect_demos.py / train_ppo.py.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("KHJ_RL_SIM_DEVICE", "cuda:0")

import pinocchio  # noqa: F401, E402

from isaaclab.app import AppLauncher  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt", type=Path, required=True,
        help="Path to bc.pt or ppo.pt; norm.json must sit alongside.",
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of eval episodes (Phase 1 curriculum gate is 100).",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Sample actions from the actor's Gaussian instead of using "
             "the mean. Default deterministic.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--curriculum-stage", type=int, default=0,
        help="Index into CurriculumCfg.side_length_m (G+: 0=2cm, 1=5cm, "
             "2=10cm, 3=15cm, 4=20cm). Selects the cube xy-spawn radius.",
    )
    parser.add_argument(
        "--task-level", type=int, default=2,
        help="G+ sub-task level (0=lift-only, 1=lift+place, 2=full PnP). "
             "Default 2 keeps the legacy 5-condition AND eval semantics.",
    )
    parser.add_argument(
        "--no-chunk-buffer", action="store_true",
        help="Disable ChunkBuffer temporal ensemble. Forces 1-step policy "
             "(actor_mean / chunk[0]) — used to measure the true PPO "
             "starting point separate from ACT inference tricks.",
    )
    parser.add_argument(
        "--policy-std-override", type=float, default=None,
        help="Override actor's trained logstd with this fixed std value "
             "(e.g. 0.05, 0.1, 0.2). When set, samples action = mean + "
             "N(0, std). Used to measure BC robustness at SAC-like noise "
             "levels separate from BC's internal trained logstd. Implies "
             "--stochastic mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sim_app = AppLauncher(headless=True, enable_cameras=False).app

    import numpy as np
    import torch

    from khj_rl.envs import CubeLiftEnv, CubeLiftEnvCfg
    from khj_rl.eval.obs_alignment import assert_obs_alignment
    from khj_rl.training.network import ActorCritic, ChunkedActorCritic
    from khj_rl.training.normalizer import ObsNormalizer

    cfg = CubeLiftEnvCfg()
    cfg.seed = args.seed
    cfg.curriculum.current_stage_idx = int(args.curriculum_stage)
    cfg.curriculum.task_level = int(args.task_level)
    env = CubeLiftEnv(cfg)
    assert_obs_alignment(env, cfg)

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    obs_dim = int(ckpt.get("obs_dim", cfg.obs_total_size))
    action_dim = int(ckpt.get("action_dim", cfg.action_size))
    hidden_dim = int(ckpt.get("hidden_dim", 256))
    is_sac_ckpt = "actor" in ckpt and "actor_critic" not in ckpt
    if is_sac_ckpt:
        from khj_rl.training.sac_network import GaussianActor
        sac_actor = GaussianActor(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim
        ).to(device)
        sac_actor.load_state_dict(ckpt["actor"])
        sac_actor.eval()
        net = None
        chunking_enabled = False
        chunk_size = 0
        inference_alpha = 0.0
        gripper_use_latest = False
        print(
            "[eval] SAC ckpt detected — using GaussianActor "
            f"(stochastic={args.stochastic})",
            flush=True,
        )
    else:
        actor_logstd_init = float(ckpt.get("actor_logstd_init", -1.0))
        # Codex I: hard-assert chunking meta — refuse to load a chunked ckpt with
        # a 1-step net or vice versa, since the state_dict shapes differ silently.
        chunking_enabled = bool(ckpt.get("chunking_enabled", False))
        chunk_size = int(ckpt.get("chunk_size", 0))
        inference_alpha = float(ckpt.get("inference_alpha", 0.2))
        gripper_use_latest = bool(ckpt.get("gripper_use_latest", True))
        if chunking_enabled and chunk_size < 1:
            raise RuntimeError(
                f"ckpt chunking_enabled=True but chunk_size={chunk_size}; meta corrupt"
            )
        if chunking_enabled:
            net = ChunkedActorCritic(
                obs_dim=obs_dim,
                action_dim=action_dim,
                chunk_size=chunk_size,
                hidden_dim=hidden_dim,
                actor_logstd_init=actor_logstd_init,
            ).to(device)
        else:
            net = ActorCritic(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                actor_logstd_init=actor_logstd_init,
            ).to(device)
        net.load_state_dict(ckpt["actor_critic"])
        net.eval()
        sac_actor = None
        print(
            f"[eval] chunking_enabled={chunking_enabled} chunk_size={chunk_size} "
            f"alpha={inference_alpha} gripper_use_latest={gripper_use_latest}",
            flush=True,
        )
    normalizer = ObsNormalizer.load(args.ckpt.with_name("norm.json"))

    class ChunkBuffer:
        """Temporal ensemble of in-flight k-step action plans.

        Codex E/F/I: zero-padding banned — only predictions that actually
        target step t contribute, weighted by exp(-alpha * offset). Reset on
        every env.reset so cross-episode plans never bleed in.
        """

        def __init__(self, k: int, alpha: float, gripper_latest: bool):
            self.k = int(k)
            self.alpha = float(alpha)
            self.gripper_latest = bool(gripper_latest)
            self._buf: list[tuple[int, np.ndarray]] = []  # (origin_t, chunk)

        def reset(self) -> None:
            self._buf.clear()

        def push(self, t: int, chunk: np.ndarray) -> None:
            self._buf.append((int(t), np.asarray(chunk, dtype=np.float32)))
            # Drop predictions whose horizon has fully expired.
            self._buf = [(o, c) for (o, c) in self._buf if t - o < self.k]

        def get_action(self, t: int) -> np.ndarray:
            contribs: list[tuple[float, np.ndarray]] = []
            latest_origin = -1
            latest_action: np.ndarray | None = None
            for origin, chunk in self._buf:
                offset = t - origin
                if 0 <= offset < self.k:
                    w = float(np.exp(-self.alpha * offset))
                    contribs.append((w, chunk[offset]))
                    if origin > latest_origin:
                        latest_origin = origin
                        latest_action = chunk[offset]
            if not contribs:
                raise RuntimeError(
                    f"ChunkBuffer empty at t={t}; push() must precede get_action()"
                )
            total_w = sum(w for w, _ in contribs)
            action = sum(w * a for w, a in contribs) / total_w
            if self.gripper_latest and latest_action is not None:
                # Last dim is the gripper command. Bypass the ensemble for it
                # so release/retreat timing isn't smeared (Codex F).
                action = action.copy()
                action[-1] = latest_action[-1]
            return action.astype(np.float32, copy=False)

    chunk_buf: ChunkBuffer | None = None
    if chunking_enabled and not args.no_chunk_buffer:
        chunk_buf = ChunkBuffer(
            k=chunk_size, alpha=inference_alpha, gripper_latest=gripper_use_latest
        )
    if args.no_chunk_buffer:
        print(
            "[eval] --no-chunk-buffer: forcing actor_mean (chunk[0]) only — "
            "ACT temporal ensemble + gripper_use_latest are disabled",
            flush=True,
        )

    def policy_step1(obs_np: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(
            normalizer.normalize_np(np.asarray(obs_np, dtype=np.float32))
        ).to(device).unsqueeze(0)
        with torch.no_grad():
            if is_sac_ckpt:
                if args.stochastic or args.policy_std_override is not None:
                    action, _ = sac_actor.sample(obs_t)
                else:
                    action = sac_actor.mean_action(obs_t)
            elif args.policy_std_override is not None:
                mean = net.actor_mean(obs_t)
                noise = torch.randn_like(mean) * args.policy_std_override
                action = (mean + noise).clamp(-1.0, 1.0)
            elif args.stochastic:
                action, _logp, _ent, _v = net.get_action_and_value(obs_t)
            else:
                action = net.actor_mean(obs_t)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def policy_chunked(obs_np: np.ndarray, t: int) -> np.ndarray:
        assert chunk_buf is not None
        obs_t = torch.from_numpy(
            normalizer.normalize_np(np.asarray(obs_np, dtype=np.float32))
        ).to(device).unsqueeze(0)
        with torch.no_grad():
            chunk = net.actor_chunks(obs_t).squeeze(0)
        chunk_np = chunk.detach().cpu().numpy().astype(np.float32)
        chunk_buf.push(t, chunk_np)
        return chunk_buf.get_action(t)

    def policy(obs_np: np.ndarray, t: int) -> np.ndarray:
        # chunk_buf is None when chunking is disabled in the ckpt OR when
        # --no-chunk-buffer is set. Both paths fall back to 1-step actor_mean.
        if chunk_buf is not None:
            return policy_chunked(obs_np, t)
        return policy_step1(obs_np)

    print(
        f"[eval] ckpt={args.ckpt} episodes={args.episodes} "
        f"stochastic={args.stochastic}",
        flush=True,
    )

    n_success = 0
    cond_counts: dict[str, int] = {}
    ep_lengths: list[int] = []
    for ep_id in range(args.episodes):
        obs = env.reset()
        # Codex H: reset the ensemble buffer between episodes so stale plans
        # from the previous rollout never feed into the next one.
        if chunk_buf is not None:
            chunk_buf.reset()
        per: dict[str, bool] = {}
        steps = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = policy(obs, steps)
            obs, _reward, terminated, truncated, info = env.step(action)
            steps += 1
            per = info.get("success_per_condition", per)
        if terminated:
            n_success += 1
        for k, v in per.items():
            cond_counts[k] = cond_counts.get(k, 0) + int(bool(v))
        ep_lengths.append(steps)
        print(
            f"[eval] ep={ep_id:03d} steps={steps} term={terminated} "
            f"trunc={truncated} success={terminated} "
            f"per_cond={ {k: int(bool(v)) for k, v in per.items()} }",
            flush=True,
        )

    rate = n_success / max(1, args.episodes)
    print(
        f"\n[eval] SUCCESS RATE: {n_success}/{args.episodes} = {rate * 100:.1f}%",
        flush=True,
    )
    print("[eval] per-condition pass rate:", flush=True)
    for k, v in sorted(cond_counts.items()):
        print(
            f"  {k}: {v}/{args.episodes} = {v / max(1, args.episodes) * 100:.1f}%",
            flush=True,
        )
    print(
        f"[eval] ep_length mean={np.mean(ep_lengths):.1f} "
        f"min={min(ep_lengths)} max={max(ep_lengths)}",
        flush=True,
    )

    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
