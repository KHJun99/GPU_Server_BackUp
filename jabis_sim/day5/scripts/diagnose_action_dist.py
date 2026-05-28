"""Analyze raw-action distribution from a saved demo file (mine_success_demos output).

Computes per-joint mean, std, and saturation rate (|raw| > 0.95) so we can decide whether
the v6 policy is hitting the action-range ceiling.
"""

import argparse
import torch

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Demo .pt file (from mine_success_demos.py).")
    parser.add_argument("--sat_threshold", type=float, default=0.95)
    args = parser.parse_args()

    blob = torch.load(args.input, map_location="cpu", weights_only=False)
    acts = blob["act"]  # list of (T, 6) tensors
    n_traj = len(acts)
    all_acts = torch.cat(acts, dim=0)  # (sum_T, 6)
    total_steps = all_acts.shape[0]
    print(f"[INFO] file={args.input}")
    print(f"[INFO] n_trajectories={n_traj}, total_steps={total_steps}, action_dim={all_acts.shape[1]}")
    print(f"[INFO] checkpoint hint: {blob.get('checkpoint', 'N/A')}")
    print(f"[INFO] success_z={blob.get('success_z')}  total_attempts={blob.get('n_total_episodes_attempted')}")

    print()
    print(f"{'joint':<16} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'sat>%g)':>10}".replace("%g", f"{args.sat_threshold:.2f}"))
    print("-" * 70)
    arm_actions = all_acts[:, :5]
    for j, name in enumerate(ARM_JOINT_NAMES):
        col = arm_actions[:, j]
        sat = float((col.abs() > args.sat_threshold).float().mean()) * 100
        print(f"{name:<16} {col.mean():>+8.3f} {col.std():>8.3f} {col.min():>+8.3f} {col.max():>+8.3f} {sat:>9.1f}%")

    gripper = all_acts[:, 5]
    open_pct = float((gripper >= 0).float().mean()) * 100
    close_pct = 100 - open_pct
    print(f"{'gripper (binary)':<16} {gripper.mean():>+8.3f} {gripper.std():>8.3f} "
          f"{gripper.min():>+8.3f} {gripper.max():>+8.3f}    open={open_pct:.1f}% close={close_pct:.1f}%")

    print()
    print("Saturation summary (per-joint, |raw|>{:.2f}):".format(args.sat_threshold))
    sats = [(arm_actions[:, j].abs() > args.sat_threshold).float().mean().item() * 100 for j in range(5)]
    max_sat = max(sats)
    avg_sat = sum(sats) / len(sats)
    print(f"  per-joint sat rates: {[f'{s:.1f}%' for s in sats]}")
    print(f"  max  saturation: {max_sat:.1f}%   avg: {avg_sat:.1f}%")

    # Decision hint
    print()
    if max_sat > 30:
        verdict = "SCALE TOO SMALL → keep scale=1.5 (or 2.0)"
    elif max_sat < 10:
        verdict = "SCALE OK → revert to 0.5 or pick 1.0; bottleneck is policy/reward"
    else:
        verdict = "AMBIGUOUS → scale ~1.0 reasonable; explore reward + IL warmstart"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
