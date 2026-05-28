# SO-ARM101 Sim2Real Inference

PyTorch-only inference wrapper for an `rsl_rl` ActorCritic policy trained on the
`SoArm101LiftCube` task. Designed for Jetson deployment — **no IsaacLab,
isaac_so_arm101, or gymnasium import**.

## Dependencies

- Python 3.10+
- `torch` (only)

That's it. `numpy` is intentionally not required.

## Files

| file | purpose |
|------|---------|
| `policy_inference.py` | `SoArm101LiftPolicy` — loads a checkpoint and runs `obs → action`. |
| `action_decoder.py`   | `SoArm101ActionDecoder` — maps raw 6-D action to servo commands. |
| `smoke_test.py`       | Loads checkpoint, runs dummy obs, measures latency. |

## Usage

```python
import torch
from policy_inference import SoArm101LiftPolicy
from action_decoder import SoArm101ActionDecoder

policy = SoArm101LiftPolicy(
    checkpoint_path="/path/to/model_3999.pt",
    device="cpu",  # or "cuda:0" on Jetson
)
decoder = SoArm101ActionDecoder()  # defaults match training-time config

# TODO: build the 36-D observation vector from real robot state.
# Training-time observation composition is not yet wired up here — must be
# replicated exactly (joint pos/vel, object pose, target pose, last action, ...)
# in the same order the policy saw during training.
obs = torch.zeros(36)

action = policy(obs)            # shape (6,)
cmd = decoder.decode(action)    # dict[str, float] of joint targets in rad
# cmd = {'shoulder_pan': ..., 'shoulder_lift': ..., 'elbow_flex': ...,
#        'wrist_flex': ..., 'wrist_roll': ..., 'left_proximal': ...}
```

## Action mapping

Replicates the training-time IsaacLab config:

- **Arm (5 joints)** — `JointPositionAction(scale=0.5, use_default_offset=True)`
  - command = `0.5 * action[i] + default_joint_pos[joint_i]`
- **Gripper (`left_proximal`)** — `BinaryJointPositionAction`
  - `action[5] >= 0` → `0.0` (open)
  - `action[5] < 0`  → `-0.6` (close)

Default joint positions come from `SO_ARM101_CFG` (`wrist_flex=1.57`, all others
`0.0`).

## ⚠️ Joint ordering caveat

The arm-joint order is **hardcoded** to:

```
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
```

This matches the user-stated training order, but the *actual* order rsl_rl saw
is determined by the USD articulation joint ordering as resolved against the
regex `["shoulder_.*", "elbow_flex", "wrist_.*"]`. There is no way to verify
this from the checkpoint alone. **If real-robot motion looks scrambled, this
ordering is the first thing to suspect** — pass an explicit
`arm_joint_order=...` to `SoArm101ActionDecoder` to override.

## Security note

`policy_inference.py` calls `torch.load(..., weights_only=False)`. This is fine
for checkpoints you trained yourself, but **never load an untrusted `.pt`
file** — it can execute arbitrary code.

## Running the smoke test

Workstation (development):

```bash
cd ~/jabis_sim/sim2real
uv run --with torch python smoke_test.py
# or, against an existing env:
conda run -n isaaclab python smoke_test.py
```

Jetson (deployment) — pass the on-device checkpoint path explicitly:

```bash
python smoke_test.py --checkpoint /opt/policies/model_3999.pt --device cpu
```

> ⚠️ Do **not** use `uv run --with torch` on Jetson. `uv` will pull a fresh
> torch wheel from PyPI which targets x86 and lacks the JetPack CUDA build —
> use the system Python with the official Jetson PyTorch wheel installed.

Expected output: actor weights load, 4 forward passes succeed (shapes
`(6,)` / `(4,6)`), decoded servo command dict, and CPU latency well under 1 ms.

## TODO

- Build a real `obs` constructor from the live robot state. The training
  observation composition lives in the IsaacLab env config and must be mirrored
  exactly (same fields, same order, same scaling).
- Sanity-check joint order on hardware before trusting `decode()` output.
