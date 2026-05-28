"""Behavior Cloning warmstart for SO-ARM101 cube-lift PPO actor.

Trains a 36→256→128→64→6 MLP (matching ``RslRlPpoActorCriticCfg.actor_hidden_dims``
in agents/rsl_rl_ppo_cfg.py) from oracle demos, then saves an actor state_dict
that can be loaded into ``runner.alg.policy.actor`` (or .actor_critic.actor on
older rsl_rl) before PPO fine-tuning.

State-dict layout matches rsl_rl's MLP — keys are ``Linear(36,256)→ELU→
Linear(256,128)→ELU→Linear(128,64)→ELU→Linear(64,6)`` which serialize as
``0.weight 0.bias 2.weight 2.bias 4.weight 4.bias 6.weight 6.bias``. The
warmstart loader copies these into the rsl_rl actor's parameters.

Pure-Python; runnable on CPU.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


OBS_MEAN_GLOBAL = None
OBS_STD_GLOBAL = None


class BCActor(nn.Module):
    """MLP whose state_dict keys mirror rsl_rl's actor (sequential + ELU).

    Hidden dims default to [256, 128, 64], ELU activation. This intentionally
    matches ``LiftCubePPORunnerCfg.policy.actor_hidden_dims`` so that
    ``policy.actor.load_state_dict(actor_bc.state_dict())`` works without key
    renaming.
    """

    def __init__(
        self,
        obs_dim: int = 36,
        act_dim: int = 6,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        activation: nn.Module = nn.ELU,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(obs)


class DemoDataset(Dataset):
    """Flatten list-of-trajectories into (obs, act) pairs.

    Computes obs normalization stats (mean, std) and normalizes inputs.
    """

    def __init__(self, obs_list: list[torch.Tensor], act_list: list[torch.Tensor]):
        if len(obs_list) != len(act_list):
            raise ValueError("obs/act list length mismatch.")
        obs_raw = torch.cat(obs_list, dim=0).float()
        self.act = torch.cat(act_list, dim=0).float()
        if obs_raw.shape[0] != self.act.shape[0]:
            raise ValueError(
                f"obs/act timesteps mismatch: {obs_raw.shape[0]} vs {self.act.shape[0]}"
            )
        # compute normalization stats
        self.obs_mean = obs_raw.mean(dim=0)
        self.obs_std = obs_raw.std(dim=0).clamp(min=1e-3)  # avoid div by 0
        # store normalized obs
        self.obs = (obs_raw - self.obs_mean) / self.obs_std
        print(f"[norm] obs_mean range: [{float(self.obs_mean.min()):.3f}, {float(self.obs_mean.max()):.3f}]")
        print(f"[norm] obs_std  range: [{float(self.obs_std.min()):.4f}, {float(self.obs_std.max()):.3f}]")

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.obs[idx], self.act[idx]


def split_train_val(
    dataset: DemoDataset, val_fraction: float = 0.1, seed: int = 0
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    n = len(dataset)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    return torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )


def train_bc(
    obs_list: list[torch.Tensor],
    act_list: list[torch.Tensor],
    *,
    obs_dim: int = 36,
    act_dim: int = 6,
    hidden_dims: tuple[int, ...] = (256, 128, 64),
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.1,
    device: torch.device | str = "cpu",
    seed: int = 0,
    log_every: int = 1,
    progress_callback: Optional[callable] = None,
) -> tuple[BCActor, list[dict]]:
    """Train a BC actor and return ``(model, history)``.

    history entries: ``{"epoch", "train_loss", "val_loss"}``.
    """
    torch.manual_seed(seed)
    device = torch.device(device)

    actor = BCActor(obs_dim, act_dim, hidden_dims).to(device)
    opt = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    full_dataset = DemoDataset(obs_list, act_list)
    global OBS_MEAN_GLOBAL, OBS_STD_GLOBAL
    OBS_MEAN_GLOBAL = full_dataset.obs_mean
    OBS_STD_GLOBAL = full_dataset.obs_std
    train_ds, val_ds = split_train_val(full_dataset, val_fraction=val_fraction, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        actor.train()
        train_loss_sum = 0.0
        train_n = 0
        for obs_b, act_b in train_loader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            pred = actor(obs_b)
            loss = loss_fn(pred, act_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss_sum += float(loss.item()) * obs_b.shape[0]
            train_n += obs_b.shape[0]

        actor.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for obs_b, act_b in val_loader:
                obs_b = obs_b.to(device)
                act_b = act_b.to(device)
                pred = actor(obs_b)
                loss = loss_fn(pred, act_b)
                val_loss_sum += float(loss.item()) * obs_b.shape[0]
                val_n += obs_b.shape[0]

        train_loss = train_loss_sum / max(train_n, 1)
        val_loss = val_loss_sum / max(val_n, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if progress_callback is not None:
            progress_callback(epoch, train_loss, val_loss)
        elif log_every > 0 and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            print(f"[bc] epoch {epoch:3d}/{epochs}  train={train_loss:.5f}  val={val_loss:.5f}")

    return actor, history


def save_bc_init(
    actor: BCActor,
    output_path: str,
    *,
    history: Optional[list[dict]] = None,
    extra_meta: Optional[dict] = None,
) -> None:
    """Save state_dict + metadata. Matches keys for rsl_rl actor MLP."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload = {
        # ``actor_state_dict`` is the bare MLP — load with
        # ``runner.alg.policy.actor.load_state_dict(payload["actor_state_dict"])``.
        "actor_state_dict": {k: v.detach().cpu().clone() for k, v in actor.net.state_dict().items()},
        "obs_mean": OBS_MEAN_GLOBAL,
        "obs_std": OBS_STD_GLOBAL,
        "model_arch": {
            "obs_dim": actor.net[0].in_features,
            "act_dim": actor.net[-1].out_features,
            "hidden_dims": [
                actor.net[i].out_features for i in range(0, len(actor.net) - 1, 2)
            ],
            "activation": "ELU",
        },
        "history": history or [],
    }
    if extra_meta:
        payload["meta"] = extra_meta
    torch.save(payload, output_path)


def load_demos(path: str) -> tuple[list[torch.Tensor], list[torch.Tensor], dict]:
    """Load a demos.pt produced by collect_demos.py."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    obs = blob["obs"]
    act = blob["act"]
    if not obs or not act:
        raise RuntimeError(f"{path} contains no demonstrations.")
    return obs, act, blob


def main() -> None:
    p = argparse.ArgumentParser(description="Train BC warmstart from oracle demos.")
    p.add_argument("--input", required=True, help="demos.pt path (collect_demos.py output).")
    p.add_argument("--output", required=True, help="bc_init.pt path.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--device", default="cpu",
                   help="cpu or cuda. Defaults to cpu so it does not contend "
                        "with concurrent PPO training.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    obs_list, act_list, blob = load_demos(args.input)
    obs_dim = blob.get("obs_dim", obs_list[0].shape[-1])
    act_dim = blob.get("act_dim", act_list[0].shape[-1])
    total_steps = sum(o.shape[0] for o in obs_list)
    print(f"[bc] loaded {len(obs_list)} trajectories, total_steps={total_steps}, "
          f"obs_dim={obs_dim}, act_dim={act_dim}")

    actor, history = train_bc(
        obs_list, act_list,
        obs_dim=obs_dim, act_dim=act_dim,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        val_fraction=args.val_fraction, device=args.device, seed=args.seed,
    )

    save_bc_init(
        actor, args.output, history=history,
        extra_meta={
            "source_demos": args.input,
            "n_trajectories": len(obs_list),
            "total_steps": total_steps,
            "demos_meta_summary": {
                "task": blob.get("task"),
                "success_z": blob.get("success_z"),
                "source": blob.get("source"),
            },
            "training": {
                "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "weight_decay": args.weight_decay,
                "val_fraction": args.val_fraction, "seed": args.seed,
            },
        },
    )
    print(f"[bc] saved → {args.output}")


if __name__ == "__main__":
    main()
