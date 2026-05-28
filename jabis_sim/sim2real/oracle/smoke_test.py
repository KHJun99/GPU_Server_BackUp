"""CPU smoke test for the oracle/BC pipeline.

No IsaacLab needed — only torch. Run:
    python smoke_test.py
or
    python -m pytest smoke_test.py -v   (if pytest is preferred)

Tests:
1. OraclePolicy state buffers initialise correctly.
2. compute_target_pos_b returns expected per-state targets.
3. step_state_machine transitions APPROACH→DESCEND→CLOSE→LIFT→MOVE_TO_GOAL
   under hand-crafted inputs.
4. to_action returns clamped (-1, +1) raw arm + ±1 gripper.
5. BCActor forward + state_dict keys match an rsl_rl-style MLP.
6. End-to-end mini training: synthetic demos → train_bc → save → reload.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

# Import from same dir
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from oracle_policy import (
    APPROACH,
    CLOSE,
    DESCEND,
    LIFT,
    MOVE_TO_GOAL,
    OraclePolicy,
)
from train_bc import BCActor, load_demos, save_bc_init, train_bc


def _ok(label: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        raise AssertionError(label)


# ---------------------------------------------------------------------------
# 1. Initialisation
# ---------------------------------------------------------------------------
def test_init() -> None:
    print("\n=== 1. OraclePolicy init ===")
    o = OraclePolicy(num_envs=4, device="cpu", scale=1.5)
    _ok("states zeros", torch.equal(o.states, torch.zeros(4, dtype=torch.long)))
    _ok("close_step zeros", torch.equal(o.close_step, torch.zeros(4, dtype=torch.long)))
    _ok("frozen_close_joint shape", o.frozen_close_joint.shape == (4, 5))
    _ok("lift_target shape", o.lift_target_pos_b.shape == (4, 3))
    _ok("scale", o.scale == 1.5)
    _ok("params", o.params["close_steps"] == 8 and o.params["success_z"] == 0.10)


# ---------------------------------------------------------------------------
# 2. compute_target_pos_b per-state
# ---------------------------------------------------------------------------
def test_target_pos() -> None:
    print("\n=== 2. compute_target_pos_b ===")
    o = OraclePolicy(num_envs=5, device="cpu", scale=1.5,
                     reach_above_dz=0.10, descend_dz=0.005)
    o.states = torch.tensor([APPROACH, DESCEND, CLOSE, LIFT, MOVE_TO_GOAL], dtype=torch.long)
    o.lift_target_pos_b[3] = torch.tensor([0.20, 0.0, 0.20])

    ee = torch.tensor([
        [0.10, 0.0, 0.20],
        [0.20, 0.0, 0.20],
        [0.20, 0.0, 0.10],
        [0.20, 0.0, 0.10],
        [0.20, 0.0, 0.20],
    ])
    cube = torch.tensor([[0.20, 0.0, 0.012]] * 5)
    goal = torch.tensor([[0.30, 0.10, 0.30]] * 5)

    target = o.compute_target_pos_b(ee, cube, goal)

    _ok("APPROACH target = cube + (0,0,0.10)",
        torch.allclose(target[0], torch.tensor([0.20, 0.0, 0.112]), atol=1e-4),
        f"got {target[0].tolist()}")
    _ok("DESCEND target = cube + (0,0,0.005)",
        torch.allclose(target[1], torch.tensor([0.20, 0.0, 0.017]), atol=1e-4),
        f"got {target[1].tolist()}")
    _ok("CLOSE target unchanged from ee (frozen handled in compute())",
        torch.allclose(target[2], ee[2]),
        f"got {target[2].tolist()}")
    _ok("LIFT target = stored lift_target",
        torch.allclose(target[3], torch.tensor([0.20, 0.0, 0.20])),
        f"got {target[3].tolist()}")
    _ok("MOVE_TO_GOAL target = goal",
        torch.allclose(target[4], goal[4]),
        f"got {target[4].tolist()}")


# ---------------------------------------------------------------------------
# 3. State machine transitions
# ---------------------------------------------------------------------------
def test_state_transitions() -> None:
    print("\n=== 3. step_state_machine transitions ===")
    o = OraclePolicy(num_envs=1, device="cpu", scale=1.5,
                     reach_dist=0.02, descend_z_dist=0.005,
                     close_steps=2, success_z=0.10)

    # 0→1: ee within reach_dist of target
    ee = torch.tensor([[0.20, 0.0, 0.112]])
    cube = torch.tensor([[0.20, 0.0, 0.012]])
    target = torch.tensor([[0.20, 0.0, 0.112]])  # dist3 = 0
    joint_pos = torch.zeros(1, 5)
    o.step_state_machine(ee, cube, joint_pos, target)
    _ok("APPROACH → DESCEND when dist3 < reach_dist", o.states.item() == DESCEND)

    # 1→2: |dz| < descend_z_dist; should also lock frozen_close_joint at this joint pose
    target_descend = torch.tensor([[0.20, 0.0, 0.017]])
    ee_descend = torch.tensor([[0.20, 0.0, 0.0175]])  # dz_abs=0.0005
    joint_at_descend = torch.tensor([[0.10, -0.30, 0.40, 1.50, 0.0]])
    o.step_state_machine(ee_descend, cube, joint_at_descend, target_descend)
    _ok("DESCEND → CLOSE when |dz| small", o.states.item() == CLOSE)
    _ok("frozen_close_joint locked at descend joints",
        torch.allclose(o.frozen_close_joint[0], joint_at_descend[0]),
        f"got {o.frozen_close_joint[0].tolist()}")

    # 2→3 needs close_step >= close_steps. close_step is incremented AFTER the
    # transition check on each call, so for close_steps=2 we need 3 calls before
    # the third call's check sees close_step=2 and transitions to LIFT:
    #   call 1 (post-CLOSE entry): check 0 < 2 → stay; close_step = 1
    #   call 2:                    check 1 < 2 → stay; close_step = 2
    #   call 3:                    check 2 ≥ 2 → LIFT
    for _ in range(3):
        o.step_state_machine(ee_descend, cube, joint_at_descend, target_descend)
    _ok("CLOSE → LIFT when close_step >= close_steps", o.states.item() == LIFT)
    _ok("lift_target_pos_b stored = ee at lift entry + (0,0,lift_dz)",
        torch.allclose(o.lift_target_pos_b[0],
                       ee_descend[0] + torch.tensor([0.0, 0.0, 0.10])),
        f"got {o.lift_target_pos_b[0].tolist()}")

    # 3→4: cube_z >= success_z
    cube_lifted = torch.tensor([[0.20, 0.0, 0.105]])
    target_lift = o.lift_target_pos_b.clone()
    o.step_state_machine(ee_descend, cube_lifted, joint_at_descend, target_lift)
    _ok("LIFT → MOVE_TO_GOAL when cube_z >= success_z",
        o.states.item() == MOVE_TO_GOAL)


# ---------------------------------------------------------------------------
# 4. to_action conversion
# ---------------------------------------------------------------------------
def test_to_action() -> None:
    print("\n=== 4. to_action ===")
    o = OraclePolicy(num_envs=2, device="cpu", scale=1.5)
    o.default_arm = torch.zeros(2, 5)  # bypass setup() for unit test
    o.states = torch.tensor([APPROACH, CLOSE], dtype=torch.long)

    joint_target = torch.tensor([
        [0.0, -0.75, 0.30, 1.50, 0.0],   # within ±1.5 → unclamped
        [0.0, -3.00, 0.30, 1.50, 0.0],   # -3.0 / 1.5 = -2.0 → clamps to -1
    ])
    action = o.to_action(joint_target, clamp=True)

    _ok("action shape (2, 6)", action.shape == (2, 6))
    _ok("env-0 arm_raw within [-1, 1]",
        action[0, :5].abs().max().item() <= 1.0 + 1e-6,
        f"max abs={float(action[0,:5].abs().max()):.3f}")
    _ok("env-0 gripper raw = +1 (APPROACH = open)",
        torch.allclose(action[0, 5], torch.tensor(1.0)))
    _ok("env-1 shoulder_lift raw clamped to -1",
        torch.allclose(action[1, 1], torch.tensor(-1.0)))
    _ok("env-1 gripper raw = -1 (CLOSE)",
        torch.allclose(action[1, 5], torch.tensor(-1.0)))

    # No-clamp variant for inspecting saturation
    action_unclamped = o.to_action(joint_target, clamp=False)
    _ok("no-clamp keeps -2.0 raw on env-1 shoulder_lift",
        torch.allclose(action_unclamped[1, 1], torch.tensor(-2.0)))


# ---------------------------------------------------------------------------
# 5. BCActor architecture & state_dict keys
# ---------------------------------------------------------------------------
def test_bcactor_arch() -> None:
    print("\n=== 5. BCActor architecture ===")
    actor = BCActor(obs_dim=36, act_dim=6, hidden_dims=(256, 128, 64))

    # Forward
    out = actor(torch.randn(8, 36))
    _ok("forward shape (8, 6)", out.shape == (8, 6))

    # state_dict keys match rsl_rl Sequential layout: 0/2/4/6 weight+bias.
    keys = sorted(actor.net.state_dict().keys())
    expected = sorted([
        "0.weight", "0.bias", "2.weight", "2.bias",
        "4.weight", "4.bias", "6.weight", "6.bias",
    ])
    _ok("state_dict keys match rsl_rl actor", keys == expected, f"got {keys}")

    # Linear sizes
    _ok("Linear(36→256)", actor.net[0].in_features == 36 and actor.net[0].out_features == 256)
    _ok("Linear(256→128)", actor.net[2].in_features == 256 and actor.net[2].out_features == 128)
    _ok("Linear(128→64)", actor.net[4].in_features == 128 and actor.net[4].out_features == 64)
    _ok("Linear(64→6)", actor.net[6].in_features == 64 and actor.net[6].out_features == 6)


# ---------------------------------------------------------------------------
# 6. End-to-end mini training
# ---------------------------------------------------------------------------
def test_e2e_training() -> None:
    print("\n=== 6. End-to-end BC training ===")

    torch.manual_seed(0)
    # Synthetic linear mapping obs → action so BC can actually fit it.
    W_true = torch.randn(36, 6) * 0.05
    b_true = torch.randn(6) * 0.02

    obs_list = []
    act_list = []
    for _ in range(8):
        T = 50
        obs = torch.randn(T, 36)
        act = obs @ W_true + b_true
        obs_list.append(obs)
        act_list.append(act)

    actor, history = train_bc(
        obs_list, act_list,
        obs_dim=36, act_dim=6, hidden_dims=(256, 128, 64),
        epochs=5, batch_size=64, lr=3e-3, weight_decay=0.0,
        val_fraction=0.1, device="cpu", seed=0, log_every=1,
    )
    first = history[0]["train_loss"]
    last = history[-1]["train_loss"]
    _ok("train_loss decreases over 5 epochs",
        last < first * 0.5,
        f"first={first:.5f} last={last:.5f}")

    # save / reload roundtrip
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bc_init.pt")
        save_bc_init(actor, path, history=history, extra_meta={"unit_test": True})
        _ok("file written", os.path.exists(path))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _ok("payload contains actor_state_dict", "actor_state_dict" in payload)
        _ok("payload contains model_arch", payload["model_arch"]["obs_dim"] == 36)

        # round-trip into a fresh BCActor and confirm forward parity
        fresh = BCActor(obs_dim=36, act_dim=6)
        fresh.net.load_state_dict(payload["actor_state_dict"])
        x = torch.randn(4, 36)
        _ok("loaded actor matches saved actor",
            torch.allclose(fresh(x), actor(x), atol=1e-6))


# ---------------------------------------------------------------------------
# 7. load_demos error handling
# ---------------------------------------------------------------------------
def test_load_demos_empty() -> None:
    print("\n=== 7. load_demos error on empty file ===")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "empty.pt")
        torch.save({"obs": [], "act": [], "meta": []}, path)
        try:
            load_demos(path)
        except RuntimeError as e:
            _ok("RuntimeError raised on empty demos", "no demonstrations" in str(e),
                f"raised: {e}")
        else:
            _ok("RuntimeError raised on empty demos", False, "did not raise")


def main() -> None:
    test_init()
    test_target_pos()
    test_state_transitions()
    test_to_action()
    test_bcactor_arch()
    test_e2e_training()
    test_load_demos_empty()
    print("\n[ALL TESTS PASSED]")


if __name__ == "__main__":
    main()
