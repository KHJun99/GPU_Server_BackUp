"""CleanRL-style single-file PPO for Phase 1 (continuous action, single env).

Why a hand-written single-file PPO and not the ``cleanrl`` package
(CLAUDE.md ## Phase 1 결정 사항 1): the BC -> PPO bridge needs (a)
matching ``ActorCritic`` class, (b) frozen obs normalization from BC,
(c) critic warm-up window, (d) low LR. Bending an external trainer to
all four lockstep rules is more work than just writing the loop.

This trainer expects an :class:`EnvLike` whose ``reset()`` returns a 1-D
obs numpy array and ``step(action)`` returns
``(obs, reward, terminated, truncated, info)`` — the Phase 1
``CubeLiftEnv`` shape.

Phase 1 stays single-env on purpose (CLAUDE.md 결정 사항 1): the vec env
migration lands with the ManagerBasedRLEnv switch in Phase 2.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from khj_rl.training.base import EnvLike, Trainer, TrainerConfig
from khj_rl.training.network import ActorCritic, ChunkedActorCritic
from khj_rl.training.normalizer import ObsNormalizer


@dataclass
class PPOConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    actor_logstd_init: float = -1.0
    # Rollout
    num_steps: int = 2048  # env steps per update
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # Optimization
    lr: float = 1e-4  # CLAUDE.md: lower after BC pretrain
    update_epochs: int = 4
    minibatch_size: int = 256
    clip_coef: float = 0.2
    clip_vloss: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    # BC -> PPO transition guards
    critic_warmup_iters: int = 15  # actor frozen for first N PPO iters
    target_kl: float | None = None  # set to e.g. 0.02 to early-stop epochs
    # Resources
    device: str = "cuda:0"
    # Logging
    log_every: int = 1  # iterations
    extras: dict = field(default_factory=dict)


class PPOTrainer(Trainer):
    def __init__(
        self,
        env: EnvLike,
        config: TrainerConfig,
        ppo_config: PPOConfig,
        *,
        video_sampler: Any | None = None,
    ) -> None:
        super().__init__(env, config)
        self.ppo_config = ppo_config

        self._device = torch.device(ppo_config.device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self._net = ActorCritic(
            obs_dim=ppo_config.obs_dim,
            action_dim=ppo_config.action_dim,
            hidden_dim=ppo_config.hidden_dim,
            actor_logstd_init=ppo_config.actor_logstd_init,
        ).to(self._device)
        self._optim = torch.optim.Adam(self._net.parameters(), lr=ppo_config.lr)

        # Frozen normalizer; set explicitly via ``load_bc()`` or assigned
        # externally before train() runs. Default identity normalizer so
        # PPO can be smoke-tested without BC.
        self._normalizer: ObsNormalizer = ObsNormalizer(
            mean=np.zeros(ppo_config.obs_dim, dtype=np.float32),
            std=np.ones(ppo_config.obs_dim, dtype=np.float32),
        )
        # Optional video sampler hook — VideoSampler instance from
        # khj_rl.eval. The trainer calls maybe_capture(global_step,
        # env, policy) at every iteration boundary so the sampler can
        # decide internally whether to fire (every_steps modulo) and
        # which rollouts to record. After a capture the env state is
        # mid-rollout-end (the sampler's last episode terminated /
        # truncated normally), so the trainer always resets before
        # resuming the training rollout.
        self._video_sampler = video_sampler

    # ---- BC handoff -----------------------------------------------------

    def load_bc(self, bc_ckpt: Path) -> None:
        """Initialize from a BCTrainer checkpoint.

        Pulls actor + critic weights and the frozen obs mean/std. Critic
        weights come along (even though BC didn't train them) so we keep
        the random init that PPO warm-up will start from.

        If the BC ckpt was trained with ACT chunking (chunking_enabled=True
        in the meta), we replace ``self._net`` with a ``ChunkedActorCritic``
        of matching shape before loading state_dict. PPO then runs against
        ``actor_mean`` (chunk[0]) like a 1-step policy; the rest of the
        chunked head shares the backbone but isn't directly supervised by
        PPO. ChunkBuffer at eval time still gets the full chunk via
        ``actor_chunks``.
        """
        ckpt = torch.load(Path(bc_ckpt), map_location=self._device)
        chunking_enabled = bool(ckpt.get("chunking_enabled", False))
        if chunking_enabled:
            chunk_size = int(ckpt.get("chunk_size", 0))
            if chunk_size < 1:
                raise RuntimeError(
                    f"ckpt chunking_enabled=True but chunk_size={chunk_size}; "
                    "meta corrupt"
                )
            print(
                f"[PPO] BC ckpt is chunked (chunk_size={chunk_size}); "
                "swapping in ChunkedActorCritic",
                flush=True,
            )
            self._net = ChunkedActorCritic(
                obs_dim=self.ppo_config.obs_dim,
                action_dim=self.ppo_config.action_dim,
                chunk_size=chunk_size,
                hidden_dim=self.ppo_config.hidden_dim,
                actor_logstd_init=self.ppo_config.actor_logstd_init,
            ).to(self._device)
            # Optimizer must wrap the new parameter set.
            self._optim = torch.optim.Adam(
                self._net.parameters(), lr=self.ppo_config.lr
            )
        self._net.load_state_dict(ckpt["actor_critic"])
        self._normalizer = ObsNormalizer.load(Path(bc_ckpt).with_name("norm.json"))

    def set_normalizer(self, normalizer: ObsNormalizer) -> None:
        self._normalizer = normalizer

    @property
    def network(self) -> "ActorCritic | ChunkedActorCritic":
        return self._net

    @property
    def normalizer(self) -> ObsNormalizer:
        return self._normalizer

    # ---- training loop --------------------------------------------------

    def train(self) -> None:
        cfg = self.ppo_config
        total_steps = self.config.total_steps
        n_iters = max(1, total_steps // cfg.num_steps)
        print(
            f"[PPO] total_steps={total_steps} num_steps/iter={cfg.num_steps} "
            f"-> {n_iters} iters; warmup={cfg.critic_warmup_iters} "
            f"(actor frozen)",
            flush=True,
        )

        # Persistent across iterations so the (obs, done) carry over.
        next_obs_np = self.env.reset()
        if isinstance(next_obs_np, tuple):  # gymnasium-style reset
            next_obs_np = next_obs_np[0]
        next_obs = self._obs_to_tensor(next_obs_np)
        next_done = torch.zeros(1, device=self._device)

        global_step = 0
        for iteration in range(n_iters):
            # Toggle actor frozen during warm-up window.
            actor_frozen = iteration < cfg.critic_warmup_iters
            for p in self._net.actor.parameters():
                p.requires_grad_(not actor_frozen)
            self._net.actor_logstd.requires_grad_(not actor_frozen)

            # Optional periodic rollout video. The sampler decides
            # internally whether this step triggers a capture; when
            # it does, it runs its own reset/step loop. We reset
            # right after so the training rollout starts from a
            # clean state.
            if self._video_sampler is not None:
                video_path = self._video_sampler.maybe_capture(
                    step=global_step,
                    env=self.env,
                    policy=self._make_eval_policy(),
                )
                if video_path is not None:
                    print(f"[PPO] video sample -> {video_path}", flush=True)
                    next_obs_np = self.env.reset()
                    if isinstance(next_obs_np, tuple):
                        next_obs_np = next_obs_np[0]
                    next_obs = self._obs_to_tensor(next_obs_np)
                    next_done = torch.zeros(1, device=self._device)

            t0 = time.time()
            rollout = self._collect_rollout(next_obs, next_done, global_step=global_step)
            global_step += cfg.num_steps

            # Bootstrap value from the final step's next_obs
            with torch.no_grad():
                next_value = self._net.value(rollout["next_obs"])
            adv, ret = self._compute_gae(rollout, next_value)
            stats = self._update(rollout, adv, ret, actor_frozen=actor_frozen)

            next_obs = rollout["next_obs"]
            next_done = rollout["next_done"]

            elapsed = time.time() - t0
            ep_rets = rollout["episode_returns"]
            ep_lens = rollout["episode_lengths"]
            print(
                f"[PPO] iter={iteration} step={global_step} "
                f"frozen_actor={actor_frozen} "
                f"ep_return_mean={np.mean(ep_rets) if ep_rets else float('nan'):.3f} "
                f"ep_len_mean={np.mean(ep_lens) if ep_lens else float('nan'):.1f} "
                f"pi_loss={stats['policy_loss']:.4f} "
                f"v_loss={stats['value_loss']:.4f} "
                f"ent={stats['entropy']:.4f} "
                f"kl={stats['approx_kl']:.4f} "
                f"clipfrac={stats['clipfrac']:.3f} "
                f"sps={cfg.num_steps / max(1e-6, elapsed):.0f}",
                flush=True,
            )

    # ---- rollout --------------------------------------------------------

    def _collect_rollout(
        self,
        first_obs: torch.Tensor,
        first_done: torch.Tensor,
        global_step: int = 0,
    ) -> dict[str, Any]:
        cfg = self.ppo_config
        T = cfg.num_steps

        obs_buf = torch.zeros((T, cfg.obs_dim), device=self._device)
        action_buf = torch.zeros((T, cfg.action_dim), device=self._device)
        logp_buf = torch.zeros(T, device=self._device)
        value_buf = torch.zeros(T, device=self._device)
        reward_buf = torch.zeros(T, device=self._device)
        done_buf = torch.zeros(T, device=self._device)

        # Per-episode bookkeeping for logging.
        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        ep_ret = 0.0
        ep_len = 0

        next_obs = first_obs
        next_done = first_done
        for t in range(T):
            obs_buf[t] = next_obs
            done_buf[t] = next_done

            with torch.no_grad():
                action, log_prob, _ent, value = self._net.get_action_and_value(
                    next_obs.unsqueeze(0)
                )
                action = action.squeeze(0)
                value = value.squeeze(0)
            value_buf[t] = value
            action_buf[t] = action
            logp_buf[t] = log_prob.squeeze(0)

            action_np = action.detach().cpu().numpy().astype(np.float32)
            # Drive dense-reward annealing: env reads this in step() if
            # cfg.dense_reward.enabled. set_global_step is a no-op
            # otherwise (env keeps a plain int counter). Calling each
            # step keeps the anneal exact within a rollout chunk.
            if hasattr(self.env, "set_global_step"):
                self.env.set_global_step(global_step + t)
            step_out = self.env.step(action_np)
            obs_np, reward, terminated, truncated, _info = self._unpack_step(step_out)
            done = bool(terminated) or bool(truncated)

            reward_buf[t] = float(reward)
            ep_ret += float(reward)
            ep_len += 1
            if done:
                episode_returns.append(ep_ret)
                episode_lengths.append(ep_len)
                ep_ret = 0.0
                ep_len = 0
                # gymnasium-style: reset returns (obs, info)
                next_obs_np = self.env.reset()
                if isinstance(next_obs_np, tuple):
                    next_obs_np = next_obs_np[0]
                next_obs = self._obs_to_tensor(next_obs_np)
                next_done = torch.ones(1, device=self._device)
            else:
                next_obs = self._obs_to_tensor(obs_np)
                next_done = torch.zeros(1, device=self._device)

        return {
            "obs": obs_buf,
            "actions": action_buf,
            "logprobs": logp_buf,
            "values": value_buf,
            "rewards": reward_buf,
            "dones": done_buf,
            "next_obs": next_obs,
            "next_done": next_done,
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths,
        }

    # ---- GAE -----------------------------------------------------------

    def _compute_gae(
        self,
        rollout: dict[str, Any],
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.ppo_config
        T = cfg.num_steps
        rewards = rollout["rewards"]
        values = rollout["values"]
        dones = rollout["dones"]
        next_done = rollout["next_done"].squeeze()
        adv = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - next_done.item()
                next_v = next_value.item() if next_value.dim() == 0 else float(next_value)
            else:
                next_non_terminal = 1.0 - dones[t + 1].item()
                next_v = float(values[t + 1])
            delta = float(rewards[t]) + cfg.gamma * next_v * next_non_terminal - float(values[t])
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
            adv[t] = last_gae
        ret = adv + values
        return adv, ret

    # ---- update --------------------------------------------------------

    def _update(
        self,
        rollout: dict[str, Any],
        advantages: torch.Tensor,
        returns: torch.Tensor,
        actor_frozen: bool,
    ) -> dict[str, float]:
        cfg = self.ppo_config
        T = cfg.num_steps
        b_obs = rollout["obs"]
        b_actions = rollout["actions"]
        b_logprobs = rollout["logprobs"]
        b_values = rollout["values"]
        b_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        b_returns = returns

        idx = np.arange(T)
        rng = np.random.default_rng(self.config.seed)

        last_stats: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
        }
        for _ in range(cfg.update_epochs):
            rng.shuffle(idx)
            for start in range(0, T, cfg.minibatch_size):
                mb_idx = idx[start : start + cfg.minibatch_size]
                mb_idx_t = torch.as_tensor(mb_idx, device=self._device, dtype=torch.long)
                if actor_frozen:
                    # Warm-up path: critic-only, skip policy-graph build.
                    new_value = self._net.value(b_obs[mb_idx_t])
                    # Placeholders for logging — no gradient through them.
                    with torch.no_grad():
                        _a, new_logprob, entropy, _v = (
                            self._net.get_action_and_value(
                                b_obs[mb_idx_t], b_actions[mb_idx_t]
                            )
                        )
                        log_ratio = new_logprob - b_logprobs[mb_idx_t]
                        ratio = log_ratio.exp()
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()
                        clipfrac = (
                            ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                        )
                    pg_loss = torch.tensor(0.0, device=self._device)
                    entropy_loss = entropy.mean()
                else:
                    _new_action, new_logprob, entropy, new_value = (
                        self._net.get_action_and_value(
                            b_obs[mb_idx_t], b_actions[mb_idx_t]
                        )
                    )
                    log_ratio = new_logprob - b_logprobs[mb_idx_t]
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()
                        clipfrac = (
                            ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                        )
                    mb_adv = b_advantages[mb_idx_t]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(
                        ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    entropy_loss = entropy.mean()

                if cfg.clip_vloss:
                    v_unclipped = (new_value - b_returns[mb_idx_t]).pow(2)
                    v_clipped = b_values[mb_idx_t] + torch.clamp(
                        new_value - b_values[mb_idx_t],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clip = (v_clipped - b_returns[mb_idx_t]).pow(2)
                    v_loss = 0.5 * torch.max(v_unclipped, v_loss_clip).mean()
                else:
                    v_loss = 0.5 * (new_value - b_returns[mb_idx_t]).pow(2).mean()

                # During warm-up: only value head learns; pg_loss is 0 and
                # actor params are also requires_grad=False so even the
                # entropy term doesn't propagate.
                if actor_frozen:
                    loss = cfg.vf_coef * v_loss
                else:
                    loss = (
                        pg_loss
                        - cfg.ent_coef * entropy_loss
                        + cfg.vf_coef * v_loss
                    )

                self._optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), cfg.max_grad_norm)
                self._optim.step()

                last_stats = {
                    "policy_loss": float(pg_loss.detach()),
                    "value_loss": float(v_loss.detach()),
                    "entropy": float(entropy_loss.detach()),
                    "approx_kl": float(approx_kl),
                    "clipfrac": float(clipfrac),
                }

            if cfg.target_kl is not None and last_stats["approx_kl"] > cfg.target_kl:
                break
        return last_stats

    # ---- helpers -------------------------------------------------------

    def _obs_to_tensor(self, obs_np: np.ndarray) -> torch.Tensor:
        normed = self._normalizer.normalize_np(np.asarray(obs_np, dtype=np.float32))
        return torch.from_numpy(normed).to(self._device)

    def _make_eval_policy(self):
        """Return a deterministic callable for video sampling.

        Wraps ``actor_mean`` (no exploration noise) + the frozen
        normalizer so the sampler captures the policy's best guess
        at each step. The closure captures ``self._net`` and
        ``self._normalizer`` by reference so subsequent training
        updates show up in the next video sample without rebuilding
        the callable.

        The policy flips ``self._net`` to ``eval()`` mode for the
        forward pass and restores ``train()`` afterwards — the
        current ``ActorCritic`` has no Dropout/BatchNorm so the
        toggle is a no-op today, but the discipline survives a
        future layer-swap that would otherwise silently invalidate
        every captured rollout.
        """
        def policy(obs_np: np.ndarray) -> np.ndarray:
            was_training = self._net.training
            self._net.eval()
            try:
                with torch.no_grad():
                    obs_t = self._obs_to_tensor(obs_np).unsqueeze(0)
                    action = self._net.actor_mean(obs_t).squeeze(0)
            finally:
                if was_training:
                    self._net.train()
            return action.detach().cpu().numpy().astype(np.float32)
        return policy

    @staticmethod
    def _unpack_step(step_out: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        # CubeLiftEnv returns (obs, reward, terminated, truncated, info).
        # Older 4-tuple envs collapse terminated/truncated into a single
        # ``done`` — handle both.
        if len(step_out) == 5:
            obs_np, reward, terminated, truncated, info = step_out
            return obs_np, float(reward), bool(terminated), bool(truncated), info
        if len(step_out) == 4:
            obs_np, reward, done, info = step_out
            return obs_np, float(reward), bool(done), False, info
        raise ValueError(f"unexpected env.step output: {step_out!r}")

    # ---- persistence ---------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_critic": self._net.state_dict(),
                "optim": self._optim.state_dict(),
                "obs_dim": self.ppo_config.obs_dim,
                "action_dim": self.ppo_config.action_dim,
                "hidden_dim": self.ppo_config.hidden_dim,
                "ppo_config": vars(self.ppo_config),
                "seed": self.config.seed,
            },
            path,
        )
        self._normalizer.save(path.with_name("norm.json"))
        with path.with_name("ppo_meta.json").open("w") as f:
            json.dump(
                {
                    "run_name": self.config.run_name,
                    "seed": self.config.seed,
                    "ppo_config": {
                        k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(self.ppo_config).items()
                    },
                },
                f,
                indent=2,
            )

    def load(self, path: Path) -> None:
        ckpt = torch.load(Path(path), map_location=self._device)
        self._net.load_state_dict(ckpt["actor_critic"])
        if "optim" in ckpt:
            self._optim.load_state_dict(ckpt["optim"])
        norm_path = Path(path).with_name("norm.json")
        if norm_path.exists():
            self._normalizer = ObsNormalizer.load(norm_path)
