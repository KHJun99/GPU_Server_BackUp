"""Merge two demo .pt files (collect_demos.py format) with metadata
consistency check (codex caveat #5)."""

import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_a", required=True)
    p.add_argument("--in_b", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    a = torch.load(args.in_a, weights_only=False)
    b = torch.load(args.in_b, weights_only=False)

    # Consistency checks (codex caveat).
    must_match = ["task", "obs_dim", "act_dim", "success_z", "source"]
    for k in must_match:
        if a.get(k) != b.get(k):
            raise SystemExit(f"[ERROR] field '{k}' mismatch: a={a.get(k)} b={b.get(k)}")

    # params dict check (oracle hyperparams). All keys must match.
    pa, pb = a.get("params", {}), b.get("params", {})
    keys_diff = (set(pa) - set(pb)) | (set(pb) - set(pa))
    if keys_diff:
        raise SystemExit(f"[ERROR] params keys differ: {keys_diff}")
    for k in pa:
        if pa[k] != pb[k]:
            raise SystemExit(f"[ERROR] params[{k}] mismatch: a={pa[k]} b={pb[k]}")

    print(f"[OK] consistency check passed (task={a['task']}, obs_dim={a['obs_dim']}, "
          f"act_dim={a['act_dim']}, success_z={a['success_z']})")

    merged = {
        "obs": list(a["obs"]) + list(b["obs"]),
        "act": list(a["act"]) + list(b["act"]),
        "meta": list(a["meta"]) + list(b["meta"]),
        "task": a["task"],
        "obs_dim": a["obs_dim"],
        "act_dim": a["act_dim"],
        "success_z": a["success_z"],
        "source": a["source"],
        "n_total_episodes_attempted": (a.get("n_total_episodes_attempted", 0)
                                       + b.get("n_total_episodes_attempted", 0)),
        "saturation_per_joint_pct": a.get("saturation_per_joint_pct"),
        "params": pa,
    }
    torch.save(merged, args.out)
    print(f"[DONE] saved {len(merged['obs'])} demos → {args.out}")

    # Quick stats
    z_max = [m["z_max"] for m in merged["meta"]]
    lengths = [m["length"] for m in merged["meta"]]
    fs = [m["final_state"] for m in merged["meta"]]
    print(f"  z_max:  min={min(z_max):.3f}  mean={sum(z_max)/len(z_max):.3f}  max={max(z_max):.3f}")
    print(f"  length: min={min(lengths)}    mean={sum(lengths)/len(lengths):.1f}    max={max(lengths)}")
    fs_dist = {s: fs.count(s) for s in sorted(set(fs))}
    print(f"  final_state distribution: {fs_dist}")


if __name__ == "__main__":
    main()
