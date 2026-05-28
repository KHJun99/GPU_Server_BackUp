"""Real-robot sensor values -> sim obs 32 dim.

Phase 2 deliverable. Coordinate-frame and dim layout are anchored on the
inspection results in inspection_results.md (env_origins == robot_root,
joint_names ordering, action mapping). This module is intentionally narrow:
it only assembles obs. Signed-axis remapping for the right-arm rot-pi joint
convention belongs to the action_executor side (Phase 3) and is NOT done here.

Conventions (do not change without re-inspecting sim):
  - JOINT_NAMES ordering matches robot.data.joint_names from sim
  - cube_pos obs is in the robot base frame (= env-local), x/y flipped vs world
    because the right-arm base is rotated pi about z relative to world.
  - cube_vel obs is hard-zeroed (training distribution: cube nearly static).
  - last_action stored is the clipped policy output (= what RslRl wrapper would
    have written into env.action_manager.action during training).
  - placeholder joint 'gripper' (idx 5) is mimic-only and is held at default
    (so joint_pos_rel[5] = 0, joint_vel[5] = 0).

Inputs to update_joint_pos are assumed to already be in the SIM joint
convention (i.e. any rot-pi sign flip on shoulder_pan has been applied by the
caller). This separation keeps obs_builder testable in isolation.
"""

from __future__ import annotations

import numpy as np

OBS_DIM = 32

# joint ordering from inspection_results.md
JOINT_NAMES = [
    "shoulder_pan",      # 0
    "shoulder_lift",     # 1
    "elbow_flex",        # 2
    "wrist_flex",        # 3
    "wrist_roll",        # 4
    "gripper",           # 5  (mimic placeholder, not actuated, held at default)
    "left_proximal",     # 6
    "right_proximal",    # 7
    "left_distal",       # 8
    "right_distal",      # 9
]
DEFAULT_JOINT_POS_ARRAY = np.array(
    [0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
)
ARM_JOINT_IDX = (0, 1, 2, 3, 4)          # arm_action targets these
GRIPPER_PLACEHOLDER_IDX = 5              # mimic, untouched
GRIPPER_FINGER_IDX = (6, 7, 8, 9)        # [left_proximal, right_proximal, left_distal, right_distal]

# inspection: gripper_action close_command_expr -> per-joint values in obs ordering
GRIPPER_OPEN = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
GRIPPER_CLOSE = np.array([-0.8, +0.8, +0.8, -0.8], dtype=np.float32)
GRIPPER_THRESHOLD = 50.0   # 0~100 servo value; >threshold -> open, else close

# right-arm base in world frame; rotation pi about z flips x and y signs
RIGHT_ARM_BASE = np.array([0.55, 0.0, 0.12], dtype=np.float32)

# inspection: cfg.sim.dt * cfg.decimation = 0.01667 (60 Hz policy/obs rate)
DEFAULT_STEP_DT = 1.0 / 60.0


class ObservationBuilder:
    """Real-robot sensor / perception -> sim obs 32 dim."""

    def __init__(self, step_dt: float = DEFAULT_STEP_DT):
        if step_dt <= 0:
            raise ValueError(f"step_dt must be positive, got {step_dt}")
        self._step_dt = float(step_dt)

        # joint state: start at sim defaults so the first joint_pos_rel = 0
        self._joint_pos_full = DEFAULT_JOINT_POS_ARRAY.copy()
        self._joint_pos_prev: np.ndarray | None = None  # zeros on first build

        # cube in sim frame
        self._cube_pos_sim = np.zeros(3, dtype=np.float32)

        # last action (clipped policy output, fed in by main_loop)
        self._last_action = np.zeros(6, dtype=np.float32)

    # ---- updates --------------------------------------------------------

    def update_joint_pos(self, arm_servo_rad: np.ndarray, gripper_pos: float) -> None:
        """Update full 10-dim joint position.

        arm_servo_rad: shape (5,), sim-convention radians for joints [0..4].
            The caller is responsible for any rot-pi sign flip.
        gripper_pos: scalar 0~100; >GRIPPER_THRESHOLD => open, else close.

        Caller contract: expected to be called exactly once per build() so that
        finite-diff joint_vel reflects one step_dt. If called multiple times
        between builds, only the last call wins -- intermediate joint states
        are silently dropped.
        """
        arm_servo_rad = np.asarray(arm_servo_rad, dtype=np.float32).reshape(-1)
        if arm_servo_rad.shape != (5,):
            raise ValueError(
                f"arm_servo_rad must be shape (5,), got {arm_servo_rad.shape}"
            )

        new_full = self._joint_pos_full.copy()
        for k, idx in enumerate(ARM_JOINT_IDX):
            new_full[idx] = arm_servo_rad[k]

        # placeholder mimic 'gripper' joint stays at default (no sensor for it)
        new_full[GRIPPER_PLACEHOLDER_IDX] = DEFAULT_JOINT_POS_ARRAY[GRIPPER_PLACEHOLDER_IDX]

        finger_values = GRIPPER_OPEN if float(gripper_pos) > GRIPPER_THRESHOLD else GRIPPER_CLOSE
        for k, idx in enumerate(GRIPPER_FINGER_IDX):
            new_full[idx] = finger_values[k]

        # advance state: prev gets the previous current, current becomes new
        self._joint_pos_prev = self._joint_pos_full
        self._joint_pos_full = new_full

    def update_cube_pos(self, cube_world_xyz: np.ndarray) -> None:
        """Convert D456 world (3,) to sim env-local (3,) and store.

        Right-arm base in world: (0.55, 0, 0.12), rotation pi about z.
        => sim_x = -(world_x - base_x), sim_y = -(world_y - base_y), sim_z = world_z - base_z
        """
        cw = np.asarray(cube_world_xyz, dtype=np.float32).reshape(-1)
        if cw.shape != (3,):
            raise ValueError(f"cube_world_xyz must be shape (3,), got {cw.shape}")

        self._cube_pos_sim = np.array(
            [
                -(cw[0] - RIGHT_ARM_BASE[0]),
                -(cw[1] - RIGHT_ARM_BASE[1]),
                 (cw[2] - RIGHT_ARM_BASE[2]),
            ],
            dtype=np.float32,
        )

    def update_last_action(self, clipped_action: np.ndarray) -> None:
        """Store the clipped policy output (= policy.get_action(...) return)."""
        a = np.asarray(clipped_action, dtype=np.float32).reshape(-1)
        if a.shape != (6,):
            raise ValueError(f"clipped_action must be shape (6,), got {a.shape}")
        self._last_action = a.copy()

    # ---- assemble -------------------------------------------------------

    def build(self) -> np.ndarray:
        joint_pos_rel = (self._joint_pos_full - DEFAULT_JOINT_POS_ARRAY).astype(np.float32)

        if self._joint_pos_prev is None:
            joint_vel = np.zeros(10, dtype=np.float32)
        else:
            joint_vel = (
                (self._joint_pos_full - self._joint_pos_prev) / self._step_dt
            ).astype(np.float32)
            # placeholder mimic joint has no real velocity signal
            joint_vel[GRIPPER_PLACEHOLDER_IDX] = 0.0

        cube_vel = np.zeros(3, dtype=np.float32)  # decision B: hard zero

        obs = np.concatenate(
            [
                joint_pos_rel,           # 10
                joint_vel,               # 10
                self._cube_pos_sim,      # 3
                cube_vel,                # 3
                self._last_action,       # 6
            ],
            dtype=np.float32,
        )
        if obs.shape != (OBS_DIM,):
            raise RuntimeError(f"obs shape must be ({OBS_DIM},), got {obs.shape}")
        return obs


# ---------------------------------------------------------------------------
# sanity check
# ---------------------------------------------------------------------------

def _fmt(arr: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+.4f}" for x in arr) + "]"


def _check(label: str, actual: np.ndarray, expected: np.ndarray, atol: float = 1e-5):
    ok = np.allclose(actual, expected, atol=atol)
    flag = "OK " if ok else "FAIL"
    print(f"  [{flag}] {label}")
    if not ok:
        print(f"        expected={_fmt(expected)}")
        print(f"        actual  ={_fmt(actual)}")
    return ok


def _sanity() -> None:
    builder = ObservationBuilder()

    # ---- dummy 1: arm at sim default, cube placed so that cube_pos_sim ~ training spawn
    # training cube_pos_sim.x lies in roughly [0.19, 0.34] (front of robot)
    # sim_x = -(world_x - 0.55) = +0.24 => world_x = 0.55 - 0.24 = 0.31 (BEHIND robot in world)
    cube_world = np.array([0.55 - 0.24, 0.0, 0.12 + 0.05], dtype=np.float32)
    expected_cube_sim = np.array([+0.24, 0.0, +0.05], dtype=np.float32)

    arm_sim_default = DEFAULT_JOINT_POS_ARRAY[list(ARM_JOINT_IDX)].copy()
    builder.update_joint_pos(arm_sim_default, gripper_pos=100.0)  # open
    builder.update_cube_pos(cube_world)
    builder.update_last_action(np.zeros(6, dtype=np.float32))
    obs1 = builder.build()

    print("[Dummy 1] arm=sim_default, cube_world=(0.31, 0, 0.17), gripper=open(100)")
    print(f"  obs.shape = {obs1.shape}")
    print(f"  joint_pos_rel = {_fmt(obs1[0:10])}")
    print(f"  joint_vel     = {_fmt(obs1[10:20])}")
    print(f"  cube_pos      = {_fmt(obs1[20:23])}")
    print(f"  cube_vel      = {_fmt(obs1[23:26])}")
    print(f"  last_action   = {_fmt(obs1[26:32])}")
    _check("joint_pos_rel == 0 (arm at default, gripper open => fingers at default)",
           obs1[0:10], np.zeros(10, dtype=np.float32))
    _check("joint_vel == 0 (first build)", obs1[10:20], np.zeros(10, dtype=np.float32))
    _check("cube_pos == expected sim frame", obs1[20:23], expected_cube_sim)
    _check("cube_vel == 0 (hard-zero)", obs1[23:26], np.zeros(3, dtype=np.float32))
    _check("last_action == 0", obs1[26:32], np.zeros(6, dtype=np.float32))

    # ---- dummy 2: feed obs into policy, observe action magnitude
    print("\n[Dummy 2] obs -> policy.get_action")
    try:
        from policy_inference import DEFAULT_CHECKPOINT, PnPPolicyInferencer

        policy = PnPPolicyInferencer(DEFAULT_CHECKPOINT, device="cpu")
        action = policy.get_action(obs1)
        print(f"  action       = {_fmt(action)}")
        print(f"  |action|max  = {float(np.abs(action).max()):+.4f}")
        print(f"  saturated?   = {bool(np.any(np.abs(action) >= 1.0 - 1e-6))}")
    except Exception as e:
        print(f"  (skipped: {e!r})")

    # ---- dummy 3: arm != default, gripper close, two updates so joint_vel != 0
    print("\n[Dummy 3] arm shifted +0.1 rad on each joint, gripper close")
    builder2 = ObservationBuilder()
    arm_v1 = arm_sim_default.copy()
    builder2.update_joint_pos(arm_v1, gripper_pos=0.0)  # close
    builder2.update_cube_pos(cube_world)
    builder2.update_last_action(np.zeros(6, dtype=np.float32))
    _ = builder2.build()  # advances state; joint_vel for next build can be nonzero

    arm_v2 = arm_v1 + 0.1
    builder2.update_joint_pos(arm_v2, gripper_pos=0.0)
    obs3 = builder2.build()

    expected_pos_rel = np.zeros(10, dtype=np.float32)
    expected_pos_rel[0:5] = 0.1  # arm joints
    expected_pos_rel[GRIPPER_PLACEHOLDER_IDX] = 0.0
    # gripper closed => finger values = GRIPPER_CLOSE, default = 0 => pos_rel = GRIPPER_CLOSE
    expected_pos_rel[6] = GRIPPER_CLOSE[0]
    expected_pos_rel[7] = GRIPPER_CLOSE[1]
    expected_pos_rel[8] = GRIPPER_CLOSE[2]
    expected_pos_rel[9] = GRIPPER_CLOSE[3]

    expected_vel = np.zeros(10, dtype=np.float32)
    expected_vel[0:5] = 0.1 / DEFAULT_STEP_DT  # finite diff
    # fingers identical between updates (both close) => vel = 0; placeholder forced 0
    print(f"  joint_pos_rel = {_fmt(obs3[0:10])}")
    print(f"  joint_vel     = {_fmt(obs3[10:20])}")
    _check("joint_pos_rel arm == +0.1, fingers == close_command",
           obs3[0:10], expected_pos_rel)
    _check("joint_vel arm == 0.1 / step_dt, fingers == 0",
           obs3[10:20], expected_vel, atol=1e-3)

    # ---- dummy 4: gripper threshold boundary + last_action round-trip
    print("\n[Dummy 4] gripper threshold boundary + last_action round-trip")
    b4 = ObservationBuilder()
    nonzero_action = np.array([0.5, -0.5, 0.25, -0.25, 0.1, -0.9], dtype=np.float32)
    b4.update_last_action(nonzero_action)
    b4.update_cube_pos(cube_world)

    # exactly at threshold -> close (condition is strict '>'). Above by eps -> open.
    b4.update_joint_pos(arm_sim_default, gripper_pos=50.0)
    obs_at = b4.build()
    _check("gripper_pos=50.0 -> close (fingers==close_command)",
           obs_at[6:10], GRIPPER_CLOSE)

    b4.update_joint_pos(arm_sim_default, gripper_pos=50.001)
    obs_above = b4.build()
    _check("gripper_pos=50.001 -> open (fingers==0)",
           obs_above[6:10], GRIPPER_OPEN)
    _check("last_action round-trip (nonzero values preserved)",
           obs_above[26:32], nonzero_action)

    print("\n[obs_builder] sanity done")


if __name__ == "__main__":
    _sanity()
