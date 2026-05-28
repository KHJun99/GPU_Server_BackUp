"""Downsample demos from 100Hz (sim) to 50Hz (HW match) by every-other transition.

Codex caveat #3: sim2real freq mismatch. BC dataset 50Hz downsample > inference-time
action repeat for distribution consistency.
"""

import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_demos", required=True)
    p.add_argument("--out_demos", required=True)
    p.add_argument("--stride", type=int, default=2)
    args = p.parse_args()

    d = torch.load(args.in_demos, weights_only=False)
    obs_in = d["obs"]
    act_in = d["act"]
    obs_out = [o[::args.stride].clone() for o in obs_in]
    act_out = [a[::args.stride].clone() for a in act_in]

    new_meta = []
    for m, o in zip(d["meta"], obs_out):
        m2 = dict(m)
        m2["length"] = int(o.shape[0])
        new_meta.append(m2)

    out = dict(d)
    out["obs"] = obs_out
    out["act"] = act_out
    out["meta"] = new_meta
    out["downsample_stride"] = args.stride
    out["effective_hz"] = (1.0 / 0.01) / args.stride  # 100Hz / stride
    torch.save(out, args.out_demos)

    n = len(obs_out)
    avg_len_in = sum(o.shape[0] for o in obs_in) / max(len(obs_in), 1)
    avg_len_out = sum(o.shape[0] for o in obs_out) / max(n, 1)
    print(f"[DONE] {n} demos, stride={args.stride}, "
          f"avg_len {avg_len_in:.0f} → {avg_len_out:.0f}, "
          f"effective_hz={out['effective_hz']}")
    print(f"  → {args.out_demos}")


if __name__ == "__main__":
    main()
