"""BC with Mixture of Gaussians output — captures multimodal action distributions."""

from __future__ import annotations

import argparse
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class MoGActor(nn.Module):
    def __init__(self, obs_dim=32, act_dim=6, hidden_dims=(256, 128, 64), num_modes=4):
        super().__init__()
        self.act_dim = act_dim
        self.num_modes = num_modes
        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU())
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.feat_dim = prev
        self.mean_head = nn.Linear(self.feat_dim, num_modes * act_dim)
        self.log_std_head = nn.Linear(self.feat_dim, num_modes * act_dim)
        self.logit_head = nn.Linear(self.feat_dim, num_modes)

    def forward(self, obs):
        feat = self.backbone(obs)
        B = obs.shape[0]
        means = self.mean_head(feat).reshape(B, self.num_modes, self.act_dim)
        log_stds = self.log_std_head(feat).reshape(B, self.num_modes, self.act_dim).clamp(-5.0, 2.0)
        logits = self.logit_head(feat)
        return means, log_stds, logits

    def act_inference(self, obs, mode="argmax"):
        means, log_stds, logits = self.forward(obs)
        if mode == "argmax":
            idx = logits.argmax(dim=-1)
            action = means[torch.arange(means.shape[0]), idx]
        else:
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            stds = log_stds.exp()
            mean = means[torch.arange(means.shape[0]), idx]
            std = stds[torch.arange(stds.shape[0]), idx]
            action = mean + std * torch.randn_like(mean)
        return action


def mog_nll_loss(means, log_stds, logits, target):
    B, K, D = means.shape
    target_expand = target.unsqueeze(1).expand(-1, K, -1)
    stds = log_stds.exp()
    log_probs_per_dim = -0.5 * (
        ((target_expand - means) / stds).pow(2) + 2 * log_stds + math.log(2 * math.pi)
    )
    log_probs_per_mode = log_probs_per_dim.sum(dim=-1)
    log_weights = F.log_softmax(logits, dim=-1)
    log_mixture = torch.logsumexp(log_probs_per_mode + log_weights, dim=-1)
    return -log_mixture.mean()


class DemoDataset(Dataset):
    def __init__(self, obs_list, act_list):
        self.obs = torch.cat(obs_list, dim=0).float()
        self.act = torch.cat(act_list, dim=0).float()
    def __len__(self): return self.obs.shape[0]
    def __getitem__(self, idx): return self.obs[idx], self.act[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--num_modes", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    blob = torch.load(args.input, map_location="cpu", weights_only=False)
    obs_list = blob["obs"]
    act_list = blob["act"]
    obs_dim = blob.get("obs_dim", obs_list[0].shape[-1])
    act_dim = blob.get("act_dim", act_list[0].shape[-1])
    total = sum(o.shape[0] for o in obs_list)
    print(f"[mog] loaded {len(obs_list)} trajectories, total_steps={total}, obs_dim={obs_dim}, act_dim={act_dim}, K={args.num_modes}")

    actor = MoGActor(obs_dim, act_dim, num_modes=args.num_modes).to(device)
    opt = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    full = DemoDataset(obs_list, act_list)
    n_val = max(1, int(len(full) * args.val_fraction))
    n_train = len(full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    for epoch in range(1, args.epochs + 1):
        actor.train()
        tl_sum, tn = 0.0, 0
        for ob, ac in train_loader:
            ob = ob.to(device); ac = ac.to(device)
            means, log_stds, logits = actor(ob)
            loss = mog_nll_loss(means, log_stds, logits, ac)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            tl_sum += float(loss.item()) * ob.shape[0]
            tn += ob.shape[0]

        actor.eval()
        vl_sum, vn = 0.0, 0
        with torch.no_grad():
            for ob, ac in val_loader:
                ob = ob.to(device); ac = ac.to(device)
                means, log_stds, logits = actor(ob)
                loss = mog_nll_loss(means, log_stds, logits, ac)
                vl_sum += float(loss.item()) * ob.shape[0]
                vn += ob.shape[0]

        print(f"[mog] epoch {epoch:3d}/{args.epochs}  train_nll={tl_sum/tn:.4f}  val_nll={vl_sum/vn:.4f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
        "model_arch": {
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden_dims": [256, 128, 64],
            "num_modes": args.num_modes,
            "type": "MoG",
        },
    }, args.output)
    print(f"[mog] saved -> {args.output}")


if __name__ == "__main__":
    main()
