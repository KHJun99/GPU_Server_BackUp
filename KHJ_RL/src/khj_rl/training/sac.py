"""SAC + Demonstrations (SACfD) trainer for Phase 1.

Card 1 of the 2026-05-19 decision tree: 1-step BC pretrain → SAC fine-tune
with demo replay buffer + Q-filtered BC auxiliary loss. The PPO + chunked-BC
attempt (#19, #21 in docs/troubleshooting.md) failed because actor_mean
alone scored 5% — RL needs a 1-step policy whose deterministic eval is
strong by itself. SACfD adds (a) off-policy demo replay so the Q-net gets
sparse-reward signal before online success arrives, (b) Q-filtered BC
loss that anchors the actor only where the demo beats it.

This trainer matches the PPO trainer's public surface so chain scripts
and the video sampler can call ``maybe_capture(step, env, policy)`` without
caring which algorithm is running.

Architecture (see ``docs/sacfd_design.md`` + Codex review 2026-05-19):
- :class:`GaussianActor` with state-conditional log-std and tanh squash.
- :class:`TwinQ` critic; target nets get a Polyak update τ each step.
- Auto-entropy tuning (``target_entropy = -action_dim``).
- Demo + online merged minibatch; demo ratio steps from 0.5 → 0.25 at
  ``demo_ratio_switch_step``.
- BC loss is **unconditional** before ``learning_starts`` (random Q-net
  makes the Q-filter mask meaningless), Q-filtered after.
- BC loss weight anneals linearly to 0 over ``bc_anneal_steps`` (default
  200k, intentionally longer than the Codex default of 100k since the
  curriculum stage switch tends to sit around 100k).
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from khj_rl.training.base import EnvLike, Trainer, TrainerConfig
from khj_rl.training.demo_buffer import DemoReplay, OnlineReplayBuffer
from khj_rl.training.her import HERBuffer, HERConfig
from khj_rl.training.normalizer import ObsNormalizer
from khj_rl.training.sac_network import GaussianActor, TwinQ


@dataclass
class SACConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    # SAC core
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.2
    target_entropy: float | None = None  # None → -action_dim
    # Replay
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 5_000   # steps of env interaction before first update
    # F12 / CLAUDE.md decision 7.3: critic warmup — for the first
    # `critic_warmup_steps` updates *after* learning_starts the actor and
    # alpha optimizers are skipped; only critic + target soft-update run.
    # Reason: actor weights come from BC and produce useful trajectories,
    # but the critic is random init at step `learning_starts`. Letting the
    # actor train against a random Q immediately corrupts the BC behavior
    # in tens of updates (verified in F11 instrumentation: actor lost
    # grasp behavior by step 5120-5440). Warmup lets the critic learn Q
    # for the (BC + small noise) trajectories *before* actor is allowed
    # to optimize against it.
    critic_warmup_steps: int = 2_000
    update_every: int = 1          # gradient steps per env step (UTD)
    # Demo
    demo_ratio_initial: float = 0.5
    demo_ratio_final: float = 0.25
    demo_ratio_switch_step: int = 50_000
    bc_loss_weight: float = 1.0
    bc_loss_q_filter: bool = True
    bc_anneal_steps: int = 200_000
    bc_loss_enabled: bool = True  # C': set False to disable actor BC anchor entirely
    # F6: TD3-like — remove entropy term from Bellman target_v and actor_loss.
    # Reason: with squashed Gaussian + narrow LOG_STD_MAX cap (=−1.6), log π
    # becomes positive (density>1) → entropy bonus α·log π SUBTRACTS from Q
    # → Q drifts negative → no learning signal. TD3-style fixes this:
    #   target_v = min(Q1', Q2')  (no entropy)
    #   actor_loss = -min_Q_pi    (no entropy)
    no_entropy_term: bool = False
    # HER (Card C': BC anchor 제거 + 3-way replay mix)
    her_enabled: bool = False
    her_ratio_initial: float = 0.4
    her_ratio_final: float = 0.3
    her_ratio_switch_step: int = 50_000
    online_ratio_initial: float = 0.3
    online_ratio_final: float = 0.5
    online_ratio_switch_step: int = 50_000
    # Logging
    log_every_steps: int = 2048
    eval_every_steps: int = 10_000
    # Resources
    device: str = "cuda:0"
    extras: dict = field(default_factory=dict)


class SACfDTrainer(Trainer):
    def __init__(
        self,
        env: EnvLike,
        config: TrainerConfig,
        sac_config: SACConfig,
        demo_replay: DemoReplay | None,
        *,
        normalizer: ObsNormalizer,
        video_sampler: Any | None = None,
        her_buffer: HERBuffer | None = None,
    ) -> None:
        super().__init__(env, config)
        self.sac_config = sac_config
        self._device = torch.device(sac_config.device)
        self._rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)

        self._actor = GaussianActor(
            obs_dim=sac_config.obs_dim,
            action_dim=sac_config.action_dim,
            hidden_dim=sac_config.hidden_dim,
        ).to(self._device)
        self._critic = TwinQ(
            obs_dim=sac_config.obs_dim,
            action_dim=sac_config.action_dim,
            hidden_dim=sac_config.hidden_dim,
        ).to(self._device)
        self._critic_target = copy.deepcopy(self._critic).to(self._device)
        for p in self._critic_target.parameters():
            p.requires_grad_(False)

        self._actor_optim = torch.optim.Adam(
            self._actor.parameters(), lr=sac_config.actor_lr
        )
        self._critic_optim = torch.optim.Adam(
            self._critic.parameters(), lr=sac_config.critic_lr
        )

        # Auto-entropy tuning: optimize log_alpha; alpha = exp(log_alpha)
        self._target_entropy = (
            float(-sac_config.action_dim)
            if sac_config.target_entropy is None
            else float(sac_config.target_entropy)
        )
        self._log_alpha = torch.tensor(
            float(np.log(sac_config.init_alpha)),
            device=self._device,
            requires_grad=True,
        )
        self._alpha_optim = torch.optim.Adam(
            [self._log_alpha], lr=sac_config.alpha_lr
        )

        self._normalizer = normalizer
        self._video_sampler = video_sampler

        self._demo = demo_replay
        self._her = her_buffer
        self._online = OnlineReplayBuffer(
            capacity=sac_config.buffer_size,
            obs_dim=sac_config.obs_dim,
            action_dim=sac_config.action_dim,
        )
        if self._demo is not None and self._demo.stats.obs_dim != sac_config.obs_dim:
            raise RuntimeError(
                f"DemoReplay obs_dim={self._demo.stats.obs_dim} != "
                f"sac_config.obs_dim={sac_config.obs_dim}"
            )
        if sac_config.her_enabled and self._her is None:
            raise RuntimeError(
                "her_enabled=True but no HERBuffer passed to SACfDTrainer"
            )

    # ---- BC handoff -----------------------------------------------------

    def load_bc_actor(self, bc_ckpt: Path) -> None:
        """Surgical transfer from a 1-step ActorCritic BC checkpoint.

        Maps ``actor.0/2/4 → GaussianActor.trunk.0/2 + mean_head`` so we
        keep the BC weights' representation of the demo manifold.
        Both networks use Tanh activations — BC weight transfer is exact
        (F9-fix: ReLU→Tanh in GaussianActor.trunk resolved lift=0%).
        logstd_head is left at its constant -1.0 init.

        ChunkedActorCritic checkpoints are rejected — Card 1 requires a
        1-step BC ckpt (see #19 root cause).
        """
        ckpt = torch.load(Path(bc_ckpt), map_location=self._device)
        if bool(ckpt.get("chunking_enabled", False)):
            raise RuntimeError(
                "SACfD load_bc_actor refuses a chunked BC ckpt — "
                "actor_mean(chunk[0]) scored 5% in stage-0 eval (#19). "
                "Train a 1-step BC ckpt with --no-chunking."
            )
        actor_critic_state = ckpt["actor_critic"]
        # The 1-step ActorCritic.actor is a Sequential with linears at
        # indices 0, 2, 4. The GaussianActor has linears at trunk[0],
        # trunk[2], plus mean_head. Three linears → three transfers.
        try:
            w0 = actor_critic_state["actor.0.weight"]
            b0 = actor_critic_state["actor.0.bias"]
            w2 = actor_critic_state["actor.2.weight"]
            b2 = actor_critic_state["actor.2.bias"]
            w4 = actor_critic_state["actor.4.weight"]
            b4 = actor_critic_state["actor.4.bias"]
        except KeyError as e:
            raise RuntimeError(
                f"BC ckpt missing expected actor key {e}; refusing partial transfer"
            )
        with torch.no_grad():
            self._actor.trunk[0].weight.copy_(w0)
            self._actor.trunk[0].bias.copy_(b0)
            self._actor.trunk[2].weight.copy_(w2)
            self._actor.trunk[2].bias.copy_(b2)
            self._actor.mean_head.weight.copy_(w4)
            self._actor.mean_head.bias.copy_(b4)
        # Normalizer is set via constructor — caller's responsibility.
        # In Card C' mode the normalizer is fit from demo + HER samples
        # (no BC ckpt at all), so the identity guard from the original
        # SACfD codepath is intentionally relaxed here.

    # ---- core SAC update ----------------------------------------------

    def _sample_merged_batch(self, global_step: int):
        """Sample a merged batch from demo / HER / online replay.

        Order in returned batch: ``[demo | her | online]``. ``n_demo`` is
        the count at the head so the BC loss can mask cleanly to demo rows
        only (HER's relabeled actions are not demonstrations — BC should
        not pull the actor toward them).

        Ratios step at ``*_switch_step`` (single hard switch — Codex 8.3).
        """
        cfg = self.sac_config
        # Step-aware ratios.
        if global_step < cfg.demo_ratio_switch_step:
            demo_ratio = cfg.demo_ratio_initial
        else:
            demo_ratio = cfg.demo_ratio_final
        if cfg.her_enabled and self._her is not None and len(self._her) > 0:
            if global_step < cfg.her_ratio_switch_step:
                her_ratio = cfg.her_ratio_initial
            else:
                her_ratio = cfg.her_ratio_final
        else:
            her_ratio = 0.0
        # Demo can be absent in C variant — clamp to 0 if so.
        if self._demo is None:
            demo_ratio = 0.0
        online_ratio = max(0.0, 1.0 - demo_ratio - her_ratio)

        n_demo = int(round(cfg.batch_size * demo_ratio))
        n_her = int(round(cfg.batch_size * her_ratio))
        n_online = cfg.batch_size - n_demo - n_her
        if n_online < 0:
            n_online = 0
            n_her = max(0, cfg.batch_size - n_demo)

        chunks_obs, chunks_act, chunks_rew, chunks_next, chunks_done = [], [], [], [], []

        if n_demo > 0:
            d_obs, d_act, d_rew, d_next, d_done = self._demo.sample(n_demo, self._rng)
            chunks_obs.append(d_obs); chunks_act.append(d_act)
            chunks_rew.append(d_rew); chunks_next.append(d_next); chunks_done.append(d_done)
        if n_her > 0:
            h_obs, h_act, h_rew, h_next, h_done = self._her.sample(n_her, self._rng)
            chunks_obs.append(h_obs); chunks_act.append(h_act)
            chunks_rew.append(h_rew); chunks_next.append(h_next); chunks_done.append(h_done)
        if n_online > 0:
            o_obs, o_act, o_rew, o_next, o_done = self._online.sample(n_online, self._rng)
            chunks_obs.append(o_obs); chunks_act.append(o_act)
            chunks_rew.append(o_rew); chunks_next.append(o_next); chunks_done.append(o_done)

        obs_np = np.concatenate(chunks_obs, axis=0)
        act_np = np.concatenate(chunks_act, axis=0)
        rew_np = np.concatenate(chunks_rew, axis=0)
        next_np = np.concatenate(chunks_next, axis=0)
        done_np = np.concatenate(chunks_done, axis=0)

        obs = torch.from_numpy(self._normalizer.normalize_np(obs_np)).to(self._device)
        action = torch.from_numpy(act_np).to(self._device)
        reward = torch.from_numpy(rew_np).to(self._device)
        next_obs = torch.from_numpy(
            self._normalizer.normalize_np(next_np)
        ).to(self._device)
        done = torch.from_numpy(done_np).to(self._device)

        return obs, action, reward, next_obs, done, n_demo, n_her, n_online

    def _bc_loss(
        self,
        obs_demo: torch.Tensor,
        action_demo: torch.Tensor,
        global_step: int,
    ) -> torch.Tensor:
        """Q-filtered BC loss on the demo subset of the batch.

        Before ``learning_starts`` the Q-net is random — Codex 8.5 / item 4:
        skip the Q-filter and apply unconditional BC anchor. After that, use
        ``q_demo > q_actor`` (margin 0) so BC only pulls when the demo
        action is actually better than the current actor's deterministic
        mean (Vecerik 2017, Rajeswaran 2018).
        """
        cfg = self.sac_config
        # F8' (#27 fix follow-up): bc_loss must operate on the RAW mean
        # (pre-clamp), not on actor.mean_action() which is clamp(mean).
        # When mean drifts beyond [-1, 1], clamp(mean) has zero gradient
        # so BC anchor cannot pull mean back — boundary deadzone. Using
        # the raw mean gives nonzero gradient everywhere and forces mean
        # to converge exactly to demo_action. The action used in env
        # interaction is still clamp(mean + noise) so the BC sweep's
        # measured behavior at std=0.05 (lift_history 79%) carries over.
        # For Q-filter we still need the in-distribution action (=clamp);
        # use clamp(mean) only for the q_actor lookup.
        raw_mean, _ = self._actor.forward(obs_demo)
        clamped_mean = raw_mean.clamp(-1.0, 1.0)
        if cfg.bc_loss_q_filter and global_step >= cfg.learning_starts:
            with torch.no_grad():
                q_actor_1, q_actor_2 = self._critic(obs_demo, clamped_mean)
                q_actor = torch.min(q_actor_1, q_actor_2)
                q_demo_1, q_demo_2 = self._critic(obs_demo, action_demo)
                q_demo = torch.min(q_demo_1, q_demo_2)
                mask = (q_demo > q_actor).float().unsqueeze(-1)
        else:
            mask = torch.ones(
                (obs_demo.shape[0], 1), device=self._device
            )
        per_row = ((raw_mean - action_demo) ** 2).sum(dim=-1, keepdim=True)
        # F9: also expose mask.mean() for diagnostic (codex flagged Q-filter
        # mask collapsing to ~0 → BC anchor silently disabled in F8'). Stored
        # on the trainer so the periodic log line picks it up.
        self._last_bc_mask_mean = float(mask.mean().detach().item())
        return (mask * per_row).mean()

    def _bc_weight(self, global_step: int) -> float:
        cfg = self.sac_config
        if cfg.bc_anneal_steps <= 0:
            return cfg.bc_loss_weight
        frac = max(0.0, 1.0 - global_step / cfg.bc_anneal_steps)
        return cfg.bc_loss_weight * frac

    def _update(self, global_step: int) -> dict[str, float]:
        cfg = self.sac_config
        obs, action, reward, next_obs, done, n_demo, n_her, n_online = (
            self._sample_merged_batch(global_step)
        )

        # ---- Critic update --------------------------------------------
        with torch.no_grad():
            next_action, next_log_prob = self._actor.sample(next_obs)
            q1_t, q2_t = self._critic_target(next_obs, next_action)
            min_q_t = torch.min(q1_t, q2_t)
            alpha = self._log_alpha.exp()
            if cfg.no_entropy_term:
                target_v = min_q_t
            else:
                target_v = min_q_t - alpha * next_log_prob
            # F14: clamp target_v to sparse reward upper bound. Reward at
            # task_level=0 is {0, +1 at terminal}, so Q ≤ 1/(1-γ)=100 in
            # theory but realistic max ≈ γ^k_to_success ≈ 0.4-0.8. Clamping
            # to [0.0, 1.0] prevents critic bootstrap explosion that
            # caused q_o > q_d hallucination in F13.
            target_v = target_v.clamp(0.0, 1.0)
            target_q = reward + cfg.gamma * (1.0 - done) * target_v

        q1, q2 = self._critic(obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self._critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._critic_optim.step()

        # ---- Actor update ---------------------------------------------
        pi_action, pi_log_prob = self._actor.sample(obs)
        # Per-state policy logstd for diagnostic — captures the actual
        # sampling noise SAC uses online (NOT alpha; alpha is just the
        # entropy weight). With logstd_head bias=-1.0 the policy std
        # starts ≈ 0.37; collapses as actor pushes high-Q deterministic.
        with torch.no_grad():
            _, log_std_diag = self._actor.forward(obs)
            policy_std_mean = float(log_std_diag.exp().mean().item())
        q1_pi, q2_pi = self._critic(obs, pi_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        alpha = self._log_alpha.exp().detach()
        if cfg.no_entropy_term:
            actor_loss_sac = (-min_q_pi).mean()
        else:
            actor_loss_sac = (alpha * pi_log_prob - min_q_pi).mean()

        bc_w = self._bc_weight(global_step) if cfg.bc_loss_enabled else 0.0
        if n_demo > 0 and bc_w > 0.0:
            obs_demo = obs[:n_demo]
            action_demo = action[:n_demo]
            bc_loss = self._bc_loss(obs_demo, action_demo, global_step)
            actor_loss = actor_loss_sac + bc_w * bc_loss
        else:
            bc_loss = torch.zeros((), device=self._device)
            actor_loss = actor_loss_sac

        # F12: critic warmup — skip actor update for the first
        # `critic_warmup_steps` updates after learning_starts so the critic
        # sees BC-driven online trajectories long enough to learn a sane
        # Q before the actor starts chasing it.
        in_critic_warmup = (
            global_step < cfg.learning_starts + cfg.critic_warmup_steps
        )
        if not in_critic_warmup:
            self._actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._actor_optim.step()

        # ---- Alpha update ---------------------------------------------
        # F6: skip alpha auto-tune when no_entropy_term — alpha doesn't
        # affect target_v or actor_loss, so tuning it is meaningless work.
        # F12: also skip alpha tuning during critic warmup (paired with
        # actor freeze).
        if not cfg.no_entropy_term and not in_critic_warmup:
            alpha_loss = -(
                self._log_alpha.exp() * (pi_log_prob.detach() + self._target_entropy)
            ).mean()
            self._alpha_optim.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self._alpha_optim.step()
            # Alpha hard clip — floor at log(0.02) so auto-tuning cannot
            # collapse the entropy weight below this. Prevents the deterministic
            # bad-policy lock seen in F2/F3 (alpha → 0.006).
            with torch.no_grad():
                self._log_alpha.data.clamp_(min=math.log(0.02))

        # ---- Target soft update ---------------------------------------
        with torch.no_grad():
            for p, p_t in zip(
                self._critic.parameters(), self._critic_target.parameters()
            ):
                p_t.data.mul_(1.0 - cfg.tau).add_(p.data, alpha=cfg.tau)

        # Split Q by source — Codex CAUTION 4 (HER over-fit monitoring).
        q_demo_mean = float(q1[:n_demo].detach().mean().item()) if n_demo > 0 else 0.0
        q_her_mean = (
            float(q1[n_demo : n_demo + n_her].detach().mean().item())
            if n_her > 0
            else 0.0
        )
        q_online_mean = (
            float(q1[n_demo + n_her :].detach().mean().item()) if n_online > 0 else 0.0
        )
        # Reward by source — to monitor whether HER labels actually fire.
        r_demo_mean = float(reward[:n_demo].detach().mean().item()) if n_demo > 0 else 0.0
        r_her_mean = (
            float(reward[n_demo : n_demo + n_her].detach().mean().item())
            if n_her > 0
            else 0.0
        )
        r_online_mean = (
            float(reward[n_demo + n_her :].detach().mean().item())
            if n_online > 0
            else 0.0
        )
        return {
            "critic_loss": float(critic_loss.detach().item()),
            "actor_loss_sac": float(actor_loss_sac.detach().item()),
            "bc_loss": float(bc_loss.detach().item()),
            "bc_weight": float(bc_w),
            "alpha": float(self._log_alpha.exp().detach().item()),
            "q1_mean": float(q1.detach().mean().item()),
            "q_target_mean": float(target_q.detach().mean().item()),
            "q_demo_mean": q_demo_mean,
            "q_her_mean": q_her_mean,
            "q_online_mean": q_online_mean,
            "r_demo_mean": r_demo_mean,
            "r_her_mean": r_her_mean,
            "r_online_mean": r_online_mean,
            "n_demo": int(n_demo),
            "n_her": int(n_her),
            "n_online": int(n_online),
            "ent_log_prob_mean": float(pi_log_prob.detach().mean().item()),
            "policy_std_mean": policy_std_mean,
            "bc_mask_mean": float(getattr(self, "_last_bc_mask_mean", 1.0)),
        }

    # ---- env interaction ----------------------------------------------

    def _act(self, obs_np: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs = torch.from_numpy(self._normalizer.normalize_np(obs_np[None, :])).to(
            self._device
        )
        with torch.no_grad():
            if deterministic:
                action = self._actor.mean_action(obs)
            else:
                action, _ = self._actor.sample(obs)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def _make_eval_policy(self):
        """Return a numpy-only policy for VideoSampler / eval. Deterministic."""
        actor = self._actor

        def policy(obs_np: np.ndarray) -> np.ndarray:
            was_training = actor.training
            actor.eval()
            try:
                return self._act(obs_np, deterministic=True)
            finally:
                if was_training:
                    actor.train()

        return policy

    def train(self) -> None:
        cfg = self.sac_config
        total_steps = self.config.total_steps
        print(
            f"[SACfD] total_steps={total_steps} learning_starts={cfg.learning_starts} "
            f"buffer={cfg.buffer_size} batch={cfg.batch_size} "
            f"demo_ratio init={cfg.demo_ratio_initial} → final={cfg.demo_ratio_final} "
            f"@switch={cfg.demo_ratio_switch_step}; "
            f"bc_w0={cfg.bc_loss_weight} bc_anneal={cfg.bc_anneal_steps} "
            f"q_filter={cfg.bc_loss_q_filter} target_entropy={self._target_entropy}",
            flush=True,
        )

        reset_out = self.env.reset()
        if isinstance(reset_out, tuple):
            next_obs_np = reset_out[0]
            reset_info = reset_out[1] if len(reset_out) > 1 else {}
        else:
            next_obs_np = reset_out
            reset_info = {}

        # If HER is on, seed the episode staging with the first obs +
        # raw fields. The env doesn't expose her_raw at reset() — we
        # take one zero-action step? No — that mutates physics. Better:
        # call a dedicated env method, or fall back to using info from
        # the first step (next_obs at step 0 is t=1, we lose t=0). For
        # Phase 1 simplicity, we accept the off-by-one: HER episode
        # starts from the first step's pre-state by storing obs_np_t
        # before each step.
        ep_ret = 0.0
        ep_len = 0
        recent_returns: list[float] = []
        recent_lengths: list[int] = []
        last_log_step = 0
        last_video_step = 0
        # HER episode tracking — staged transitions for the in-flight ep.
        her_active = cfg.her_enabled and (self._her is not None)
        her_ep_open = False

        global_step = 0
        t_start = time.time()
        latest_stats: dict[str, float] = {}

        while global_step < total_steps:
            # ---- env step --------------------------------------------
            if global_step < cfg.learning_starts:
                action_np = self._rng.uniform(
                    low=-1.0, high=1.0, size=(cfg.action_dim,)
                ).astype(np.float32)
            else:
                action_np = self._act(next_obs_np, deterministic=False)

            if hasattr(self.env, "set_global_step"):
                self.env.set_global_step(global_step)
            obs_np_t = next_obs_np
            step_out = self.env.step(action_np)
            (
                next_obs_np,
                reward,
                terminated,
                truncated,
                info,
            ) = self._unpack_step(step_out)
            done_bool = bool(terminated) or bool(truncated)
            # Store with done=terminated (NOT truncated) so the Bellman
            # bootstrap from time-limit truncations is preserved.
            self._online.push(
                obs=obs_np_t,
                action=action_np,
                reward=float(reward),
                next_obs=next_obs_np,
                done=bool(terminated),
            )
            # HER episode push
            if her_active:
                her_raw = info.get("her_raw", None)
                if her_raw is not None:
                    if not her_ep_open:
                        # First step of an episode: seed buffer with the
                        # pre-step obs. We use the post-step her_raw as the
                        # "begin" snapshot — the obs[0] is obs_np_t (pre).
                        # cube_xyz_m for t=0 we approximate via her_raw at
                        # t=1; this is an off-by-one accepted in design.
                        self._her.begin_episode(obs_np_t, her_raw)
                        her_ep_open = True
                    self._her.push_transition(action_np, next_obs_np, her_raw)
            ep_ret += float(reward)
            ep_len += 1
            global_step += 1

            if done_bool:
                recent_returns.append(ep_ret)
                recent_lengths.append(ep_len)
                if len(recent_returns) > 32:
                    recent_returns.pop(0)
                    recent_lengths.pop(0)
                # TEMP DEBUG (F11): print per_cond + ep_len at episode end
                # to compare SAC online behavior vs BC eval. Codex Tanh fix
                # should make BC and SAC equivalent — if SAC's per_cond
                # consistently shows lift_history=0 while BC eval shows
                # lift_history=1, there is a SAC-loop-specific divergence.
                per_cond = info.get("success_per_condition", {})
                per_str = " ".join(f"{k}={int(bool(v))}" for k, v in per_cond.items())
                print(
                    f"[ep_end] global_step={global_step} ep_len={ep_len} "
                    f"ep_return={ep_ret:.3f} term={bool(terminated)} "
                    f"trunc={bool(truncated)} per_cond[{per_str}]",
                    flush=True,
                )
                if her_active and her_ep_open:
                    self._her.end_episode(terminal=bool(terminated))
                    her_ep_open = False
                ep_ret = 0.0
                ep_len = 0
                reset_out = self.env.reset()
                if isinstance(reset_out, tuple):
                    next_obs_np = reset_out[0]
                else:
                    next_obs_np = reset_out

            # ---- gradient updates ------------------------------------
            if global_step >= cfg.learning_starts:
                for _ in range(cfg.update_every):
                    latest_stats = self._update(global_step)

            # ---- logging ---------------------------------------------
            if global_step - last_log_step >= cfg.log_every_steps:
                elapsed = time.time() - t_start
                sps = global_step / max(1e-6, elapsed)
                ret_mean = (
                    float(np.mean(recent_returns))
                    if recent_returns
                    else float("nan")
                )
                len_mean = (
                    float(np.mean(recent_lengths))
                    if recent_lengths
                    else float("nan")
                )
                her_eps = self._her.n_episodes if self._her is not None else 0
                her_ok, her_tot = (
                    self._her.relabeled_success_stats()
                    if self._her is not None
                    else (0, 0)
                )
                her_rate = float(her_ok) / max(1, her_tot)
                msg = (
                    f"[SACfD] step={global_step} sps={sps:.0f} "
                    f"online={len(self._online)} her_eps={her_eps} "
                    f"her_succ_rate={her_rate:.3f} "
                    f"ep_return_mean={ret_mean:.3f} ep_len_mean={len_mean:.1f}"
                )
                if latest_stats:
                    msg += (
                        f" critic={latest_stats['critic_loss']:.4f} "
                        f"actor={latest_stats['actor_loss_sac']:.4f} "
                        f"bc={latest_stats['bc_loss']:.4f} "
                        f"bc_w={latest_stats['bc_weight']:.3f} "
                        f"bc_mask={latest_stats.get('bc_mask_mean', 1.0):.3f} "
                        f"alpha={latest_stats['alpha']:.4f} "
                        f"pi_std={latest_stats['policy_std_mean']:.4f} "
                        f"q1={latest_stats['q1_mean']:.3f} "
                        f"q_d={latest_stats['q_demo_mean']:.2f}/"
                        f"q_h={latest_stats['q_her_mean']:.2f}/"
                        f"q_o={latest_stats['q_online_mean']:.2f} "
                        f"r_h={latest_stats['r_her_mean']:.3f} "
                        f"n=d{latest_stats['n_demo']}/h{latest_stats['n_her']}/o{latest_stats['n_online']}"
                    )
                print(msg, flush=True)
                last_log_step = global_step

            # ---- video sampler ---------------------------------------
            if self._video_sampler is not None and global_step - last_video_step >= 1:
                video_path = self._video_sampler.maybe_capture(
                    step=global_step,
                    env=self.env,
                    policy=self._make_eval_policy(),
                )
                if video_path is not None:
                    print(f"[SACfD] video sample -> {video_path}", flush=True)
                    last_video_step = global_step
                    # Sampler ran its own rollout loop; reset so the training
                    # rollout continues from a clean state.
                    next_obs_np = self.env.reset()
                    if isinstance(next_obs_np, tuple):
                        next_obs_np = next_obs_np[0]
                    ep_ret = 0.0
                    ep_len = 0

    # ---- io ---------------------------------------------------------

    @staticmethod
    def _unpack_step(step_out):
        if len(step_out) == 5:
            return step_out
        # Gym (4-tuple) legacy fallback — env is gymnasium so unlikely.
        obs, reward, done, info = step_out
        return obs, reward, done, False, info

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self._actor.state_dict(),
                "critic": self._critic.state_dict(),
                "critic_target": self._critic_target.state_dict(),
                "log_alpha": self._log_alpha.detach().cpu(),
                "obs_dim": self.sac_config.obs_dim,
                "action_dim": self.sac_config.action_dim,
                "hidden_dim": self.sac_config.hidden_dim,
                "seed": self.config.seed,
            },
            path,
        )
        self._normalizer.save(path.with_name("norm.json"))
        meta = {
            "run_name": self.config.run_name,
            "seed": self.config.seed,
            "sac_config": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(self.sac_config).items()
            },
            "demo_stats": vars(self._demo.stats),
        }
        with path.with_name("sac_meta.json").open("w") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: Path) -> None:
        ckpt = torch.load(Path(path), map_location=self._device)
        self._actor.load_state_dict(ckpt["actor"])
        self._critic.load_state_dict(ckpt["critic"])
        self._critic_target.load_state_dict(ckpt["critic_target"])
        with torch.no_grad():
            self._log_alpha.copy_(ckpt["log_alpha"].to(self._device))
        self._normalizer = ObsNormalizer.load(Path(path).with_name("norm.json"))

    # ---- accessors --------------------------------------------------

    @property
    def actor(self) -> GaussianActor:
        return self._actor

    @property
    def critic(self) -> TwinQ:
        return self._critic

    @property
    def normalizer(self) -> ObsNormalizer:
        return self._normalizer
