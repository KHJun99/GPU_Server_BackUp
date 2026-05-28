"""Single source of truth for the Phase 1 cube-lift environment.

Anything that other modules need to know about the env (geometry, action
limits, success thresholds, perception noise, curriculum stages) lives here
as typed dataclasses. NoiseModel wiring, EnvLike adapter, oracle script,
success criterion, and demo writer all read from one ``CubeLiftEnvCfg``
instance so there is no hidden constant elsewhere — that was the
``jabis_sim_v2`` baseline-joint sim2real drift root cause.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from khj_rl.sim2real.noise import NoiseModel
from khj_rl.sim2real.randomization import RandomizationConfig


@dataclass
class RobotCfg:
    """SO-ARM101 5-DOF arm + PincOpen gripper.

    SO-ARM101 is 5 revolute arm joints + 1 gripper drive motor. The PincOpen
    4-bar linkage adds 4 more revolute joints (left/right proximal + distal)
    that mimic the gripper main joint — those are NOT in ``joint_names`` here
    because they're either USD-level mimics or handled implicitly by PhysX.
    """

    name: str = "so_arm101_pincopen"
    n_joints: int = 5  # arm DOF the policy controls
    # Arm joint names, matching the USD prim paths under /so101_pincopen/joints/.
    joint_names: tuple[str, ...] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )
    # Gripper main joint (drives the 4-bar). USD has 4 more passive linkage
    # joints that mimic this one once the mimic API is added.
    gripper_joint_name: str = "gripper"
    # Home / neutral pose, radians. Picked via a Pinocchio FK sweep
    # constrained to: fingertip near (x≈0.25, y≈0, z≈0.09) AND the
    # left distal link's local +X axis (= finger long-axis) pointing
    # close to world -Z (i.e., fingers approach top-down). At the
    # current joint_init the cosine alignment is 0.964 (≈15.5°) —
    # invariant ``down_align >= 0.96`` is locked in
    # ``tests/test_home_pose.py`` (Method B M0.5 Step F). The
    # earlier pose (-0.70, +1.00, +0.40) reached cube center on
    # descend but the fingers approached HORIZONTALLY, so the 4-bar
    # close swung them forward and ejected the cube along +X.
    # (-0.30, +0.20, +1.40) keeps the same workspace altitude with
    # distal links pointing down (down_align = 0.96) — close motion
    # now swings inward in the horizontal plane around the cube
    # rather than driving the cube along arm-forward.
    joint_init: tuple[float, ...] = (0.0, -0.30, 0.20, 1.40, 0.0)
    # (low, high) per arm joint, radians. Rough SO-ARM101 LeRobot limits;
    # tighten with the URDF's <limit> tags during Isaac Lab wiring.
    joint_pos_limit: tuple[tuple[float, float], ...] = (
        (-3.14, 3.14),   # shoulder_pan  — full rotation
        (-1.57, 1.57),   # shoulder_lift — 90° both sides
        (0.0, 1.70),     # elbow_flex    — folds upward only
        (-1.57, 1.57),   # wrist_flex
        (-3.14, 3.14),   # wrist_roll    — full rotation
    )
    # Per-step joint-delta cap (radians/step); bounds the policy output.
    action_delta_max: float = 0.02
    # PincOpen gripper joint TARGET range (radians). action[-1] ∈ [-1, +1]
    # is linearly mapped to this range; the same range is written to the
    # articulation's soft_joint_pos_limits at env init (USD physics
    # parses URDF radian limits as degrees, so we always override at
    # runtime).
    #
    # Calibrated against the real STS3215 right-finger encoder positions
    # the user measured with calipers (2026-05-17):
    #   encoder 1175  →  fingertip spread 79 mm  →  sim gripper +0.150
    #   encoder 2031  →  fingertip spread 44 mm  →  sim gripper -0.750
    #   encoder 2887  →  fingertip spread  4 mm  →  sim gripper -1.800
    # Mapping done by a Pinocchio FK sweep over the top-down home pose:
    # solve for the gripper joint value whose left-/right-distal tip
    # midpoint (body origin + STL-bbox-derived (0.025, ±0.009, 0.010)
    # local offset, rotated by the body quaternion) matches the
    # measured spread. The mid-point matches the encoder midpoint within
    # 0.075 rad — close to linear despite the 4-bar's curved kinematics.
    #
    # At sim gripper = -1.8 the fingertip-Z sits ~6 mm above the
    # gripper=-0.5 minimum (the 4-bar arc folds back up as it closes),
    # but the rise stays inside the 0..4.5 cm cube body z range so the
    # grasp wrap is preserved.
    gripper_joint_target_low: float = -1.800
    gripper_joint_target_high: float = 0.150
    # Mimic 4-bar joint target range (radians). At gripper=-1.8 the
    # mimic targets reach ±0.9 (= 1.8 × 0.5). Cap a bit above so the
    # PD has headroom.
    mimic_joint_target_low: float = -1.0
    mimic_joint_target_high: float = 1.0
    # success-condition #3 (release + retreat) threshold on the
    # 0..1-normalized gripper_state observation: open ≥ this value
    # counts as "released".
    gripper_open_threshold: float = 0.8


@dataclass
class CubeCfg:
    """Phase 1 single-object: a 45 mm matte cube, ~50 g, lime/cyan."""

    size_m: float = 0.045
    mass_kg: float = 0.050
    color_rgba: tuple[float, float, float, float] = (0.0, 0.8, 0.5, 1.0)
    table_top_z_m: float = 0.0
    spawn_clearance_m: float = 0.001

    @property
    def spawn_z_m(self) -> float:
        return self.table_top_z_m + 0.5 * self.size_m + self.spawn_clearance_m


@dataclass
class GoalCfg:
    """Single fixed goal in robot base frame (Phase 1).

    Cube xy-start is randomized inside the curriculum's side_length; the goal
    itself stays fixed.
    """

    pos_xyz_m: tuple[float, float, float] = (0.25, 0.0, 0.05)
    # Stable-placement xy tolerance (success condition #2).
    radius_xy_m: float = 0.03


@dataclass
class ControlCfg:
    rate_hz: int = 20
    episode_seconds: float = 8.0

    @property
    def max_steps(self) -> int:
        return int(round(self.rate_hz * self.episode_seconds))

    @property
    def dt(self) -> float:
        return 1.0 / float(self.rate_hz)


@dataclass
class SuccessCfg:
    """Multi-condition success criterion (CLAUDE.md ## Phase 1 결정 사항 5).

    All five conditions must hold simultaneously for +1 reward. #4 (velocity
    stability) is the direct block against jabis_sim_v2 rolling-cube reward
    hacking.
    """

    # #1 lift history.
    lift_z_m: float = 0.08
    lift_hold_steps: int = 20  # 1 s @ 20 Hz
    # Anti-pressing guard (#27 troubleshooting): the cube_z >= lift_z_m check
    # alone is hackable — EE pressing the cube into table/edge can bounce it
    # briefly above the threshold (sim physics rebound). Require during lift:
    #   - EE close enough to cube to be plausibly grasping
    #   - gripper opening tight enough to be holding (not fully open)
    # Both conditions must hold simultaneously with cube_z >= lift_z_m before
    # the lift_history rolling buffer records a True frame.
    #
    # Threshold tuning notes (pilot demo collect, 2026-05-19):
    # - First values 0.08 / 0.7 rejected oracle MP demos — oracle's lift had
    #   ee_cube_dist > 8cm and/or gripper_opening > 0.7 at the critical step.
    # - Loosened to 0.15 / 0.85: oracle should pass while still rejecting
    #   pure-pressing (EE far above cube + gripper not closed).
    lift_ee_cube_max_dist_m: float = 0.15  # 15 cm — EE within grasp reach
    lift_gripper_max_open: float = 0.85  # 0..1 normalized — partially closed
    # F15 anti-pressing-v2 (post-#27, after observing F14 STEP 5 step 20001
    # video: actor's "grasp" was actually pressing cube against a misaligned
    # 4-bar finger and the squashed cube bounced up past lift_z_m. 3-cond AND
    # passed because gripper_opening was < 0.85 during the press too. Add
    # cube velocity stability constraints to block the squash trajectory):
    #   - lift_cube_v_xy_max: cube's horizontal velocity during lift latch
    #     must stay below threshold. Real lift keeps cube xy stationary
    #     (cube moves with EE). Squash bounces cube xy.
    #   - lift_cube_omega_max: cube must not tumble during latch.
    # Tuning (pilot 2026-05-20): oracle's grasped cube has v_xy ≤ 0.04
    # but omega ~2.2 rad/s = 126°/s sustained — cube rotates in the loose
    # 4-bar gripper during lift. Set thresholds higher than oracle observed
    # max but well below press+squash violent fling (300+°/s expected).
    lift_cube_v_xy_max: float = 0.20  # m/s — cube xy velocity ceiling
    lift_cube_omega_max: float = math.radians(270.0)  # rad/s — tumble ceiling

    # #2 stable placement (xy radius shared with GoalCfg.radius_xy_m).
    place_hold_steps: int = 20
    # 2026-05-20 high-release patch (codex review): stable_placement was
    # xy-only, so the policy could hover above goal and OPEN GRIPPER,
    # letting the cube free-fall into the radius. We require the cube to
    # be physically near the table top before counting as placed.
    # cube center at rest = table_top_z + cube/2 + clearance = 0.0235m.
    # 0.035m allows the cube center up to ~1.2cm above its resting
    # height — i.e. a hand still in contact mid-lowering counts, but a
    # released free-fall from above the gripper home pose does not.
    place_z_max_m: float = 0.035

    # #3 release + retreat.
    retreat_distance_m: float = 0.06

    # #4 velocity stability.
    cube_v_max: float = 0.05                    # m/s
    cube_omega_max: float = math.radians(30.0)  # rad/s

    # #5 visual agreement.
    require_visual_agreement: bool = True


@dataclass
class NoiseCfg:
    """Perception noise applied at obs post-processing only.

    Reward path uses GT pose so this never leaks into the success signal.
    NoiseCfg.to_noise_model() builds the runtime ``NoiseModel`` used by the env.
    """

    cube_xy_sigma_m: float = 0.004
    cube_z_sigma_m: float = 0.008
    outlier_prob: float = 0.01
    outlier_max_m: float = 0.05
    visibility_dropout_prob: float = 0.02
    joint_pos_sigma_rad: float = 0.001
    joint_vel_sigma_rad_s: float = 0.01

    def to_noise_model(self) -> NoiseModel:
        return NoiseModel(
            obs_joint_pos_std=self.joint_pos_sigma_rad,
            obs_joint_vel_std=self.joint_vel_sigma_rad_s,
            obs_object_pos_xy_std=self.cube_xy_sigma_m,
            obs_object_pos_z_std=self.cube_z_sigma_m,
            obs_object_outlier_prob=self.outlier_prob,
            obs_object_outlier_max=self.outlier_max_m,
            obs_visibility_dropout_prob=self.visibility_dropout_prob,
        )


@dataclass
class CurriculumCfg:
    """Goal-space curriculum on the cube xy-start distribution.

    G+ adds a narrow 2cm stage at index 0 (was 5cm) and a `task_level`
    axis for sub-task curriculum (lift-only → lift+place → full PnP).
    See docs/g_plus_design.md.
    """

    # Axis 1: cube xy spawn width (m). G+: 0.02 prepended for narrow start.
    side_length_m: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20)
    # Index into ``side_length_m`` selecting the active stage.
    current_stage_idx: int = 0

    # Axis 2: task complexity decomposition.
    # 0 = lift-only (reward = lift_history; force_terminate 30 step after latch)
    # 1 = lift + place (reward = lift_history AND stable_placement)
    # 2 = full 5-condition AND (original Phase 1 success)
    task_level: int = 0
    # Hover-attractor guard for level 0: episode force-terminates this many
    # steps after lift_history latches if the actor hasn't progressed to a
    # placement attempt. 30 step ≈ 1.5 s @ 20 Hz. Codex review 2026-05-19.
    max_post_lift_steps: int = 30

    advance_eval_episodes: int = 100
    advance_success_rate: float = 0.7
    stall_env_steps_per_stage: int = 1_000_000


@dataclass
class DenseRewardCfg:
    """Phase 1 A option — bounded·annealed·phase-gated dense shaping.

    Disabled by default. Activated only after CLAUDE.md 결정사항 5
    update + codex review (both completed 2026-05-17). See
    docs/dense_reward_design.md for the full safety contract and
    weight rationale.
    """

    enabled: bool = False
    alpha0: float = 0.01
    decay_steps: int = 300_000          # 200k는 pure sparse 학습 기간 (linear)
    w_ee_cube_dist: float = 0.5         # codex review: ↓ from 1.0
    w_lifted_bonus: float = 0.03        # codex review: ↓ from 0.1 (hover attractor 회피)
    w_cube_goal_dist: float = 1.0       # 거리는 min(d, dist_cap)로 saturate
    w_released_bonus: float = 0.5       # codex review: ↑ from 0.2 (조건 강화)
    dist_cap: float = 0.1               # cube_goal_dist saturate (m)
    near_goal_radius_xy: float = 0.06   # release_bonus 활성 임계 (success place_radius*2)
    velocity_stable_lin: float = 0.05   # cube 선속도 임계 (m/s)
    velocity_stable_ang: float = 0.5    # cube 각속도 임계 (rad/s)


# =============================================================================
# Method B (v7) cfg — EE-delta + env-internal IK + IL pivot. See
# docs/method_b_design.md for the single source of truth. These three
# dataclasses are wired into CubeLiftEnvCfg as fields so memory layout
# stays uniform, but the env code does NOT read them until M2 (action
# space switch). M0.5/M1 work fills in the M0.5-calibrated thresholds
# (jitter_p95, jerk_p95) and EE delta cap (ee_delta_max_m) by measurement.
# =============================================================================


@dataclass
class EEControlCfg:
    """Method B IK control parameters (per docs/method_b_design.md §5).

    All numeric thresholds use SI units in field names: _m, _m_s, _rad_s,
    _rad_s2. ``ee_delta_max_m`` and the two p95 fields are placeholders
    filled in by M0.5 calibration (Steps D and E in the design doc).
    """

    # Per-policy-step EE displacement cap (1차 클램프).
    # M0.5 Step D calibration (2026-05-23): per-phase p95 max = 0.1402m (retreat).
    # 그러나 cfg 0.015 → 0.140 변경 시 oracle SR이 100% → 0%로 무너짐 (oracle output이
    # 9.35배 증폭되어 IK 발산). cfg 갱신 = 단순 한 줄 edit 아님. oracle 재조정 필요.
    # 추가 검증 전까지 안전한 0.015 유지.
    ee_delta_max_m: float = 0.015
    # DLS regularization (oracle.py:188 검증값).
    dls_lambda: float = 0.1
    # Null-space bias gain pulling q toward q_home (oracle.py:188 검증값).
    null_bias_gain: float = 0.15
    # wrist_roll を q_home (= 0.0)로 강제. top-down 자세 직접 유지.
    wrist_roll_clamp_to_home: bool = True
    # Per-policy-step joint target delta cap (2차 클램프).
    ik_max_joint_delta_rad_per_step: float = 0.3
    # 즉시 AND 가드 두 번째 조건 (||x_err|| 임계).
    ik_max_position_error_m: float = 0.05
    # Rolling window for jitter/jerk p95.
    ik_jitter_window: int = 10
    # M0.5 Step E calibrate (2026-05-23): cal_p95 0.005797m × margin 2.5 = 0.014493m.
    # validation trip rate 0.00% (PASS).
    ik_jitter_residual_p95_max_m: float = 0.014493
    # M0.5 Step E calibrate (2026-05-23): cal_p95 17.62 rad/s² × margin 2.5 = 44.04
    # validation trip rate 1.09% FAIL (target <1%). margin 5.0 재검증 진행 중 → 결과 후 갱신.
    ik_jerk_p95_max_rad_s2: float = 0.0
    # 누적 fail이 이 step 수 연속이면 episode truncate.
    ik_consecutive_fail_truncate: int = 10
    # 느린 발산 단조 증가 판정 window.
    ik_slow_diverge_window: int = 20
    # Orientation drift (top-down quat 대비) 경고 임계.
    orientation_drift_warn_rad: float = 0.2


@dataclass
class SuccessGuardCfg:
    """Reward-hacking sub-check thresholds (per docs/method_b_design.md §4.2).

    Transit phase는 rolling tolerance, terminal window (마지막
    terminal_window_steps) 안은 매 step strict. terminal_window_steps=20
    은 placement dwell 20-step과 일관(1초).
    """

    # Sub-check rolling tolerance (transit phase only).
    visibility_transit_tolerance: int = 3
    gripper_match_transit_tolerance: int = 5
    cube_orientation_transit_tolerance: int = 3
    # cube up-axis vs world up 허용 기울임 (tipped/edge 차단).
    cube_max_tip_angle_rad: float = math.radians(30.0)
    # Terminal strict window (placement_dwell_window=20과 동기화).
    terminal_window_steps: int = 20
    # release 시점 fingertip↔cube 최소 분리 (4-bar mimic 지연 차단).
    contact_separation_min_m: float = 0.01
    # IK fail masking — transit phase.
    ik_fail_mask_consecutive: int = 5
    ik_fail_mask_ratio: float = 0.05
    # Terminal window 안 1 step IK fail로도 success=False 강제.
    ik_fail_mask_terminal_zero: bool = True


@dataclass
class AuditCfg:
    """Audit thresholds for "Hard exploit vs Audited warning" 분리
    (per docs/method_b_design.md §4.3).
    """

    # Warning bound (audited, 비차단).
    warning_mean_per_episode_max: float = 1.0
    warning_p95_per_episode_max: int = 2
    # success=True ep의 terminal-adjacent (마지막 N step) warning은 0이어야.
    warning_terminal_adjacent_window: int = 40
    warning_terminal_adjacent_max_in_success: int = 0
    # 단일 가드 category 발동률이 이 비율 초과면 자동 audit 플래그.
    warning_category_rate_audit_threshold: float = 0.01


@dataclass
class CubeLiftEnvCfg:
    """Top-level env config. Every other module reads constants from here."""

    robot: RobotCfg = field(default_factory=RobotCfg)
    cube: CubeCfg = field(default_factory=CubeCfg)
    goal: GoalCfg = field(default_factory=GoalCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    success: SuccessCfg = field(default_factory=SuccessCfg)
    noise: NoiseCfg = field(default_factory=NoiseCfg)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)
    dense_reward: DenseRewardCfg = field(default_factory=DenseRewardCfg)
    # Method B pivot cfg — dormant until M2. See docs/method_b_design.md.
    ee_control: EEControlCfg = field(default_factory=EEControlCfg)
    success_guard: SuccessGuardCfg = field(default_factory=SuccessGuardCfg)
    audit: AuditCfg = field(default_factory=AuditCfg)
    seed: int = 0

    @property
    def obs_keys(self) -> tuple[str, ...]:
        """Order of the flat obs vector. The env adapter and obs_alignment
        guard both read this — the single source of layout truth.
        """
        return (
            "joint_pos",
            "joint_vel",
            "ee_pose_base",
            "cube_xyz_base",
            "target_delta_base",
            "gripper_state",
            "last_action",
            "normalized_t",
        )

    @property
    def obs_sizes(self) -> dict[str, int]:
        return {
            "joint_pos": self.robot.n_joints,
            "joint_vel": self.robot.n_joints,
            # xyz (3) + unit quat [w, x, y, z] (4). Computed from joint_pos via
            # FK so the obs is reproducible on real hardware too.
            "ee_pose_base": 7,
            "cube_xyz_base": 3,
            "target_delta_base": 3,
            "gripper_state": 1,
            "last_action": self.action_size,
            "normalized_t": 1,
        }

    @property
    def action_size(self) -> int:
        return self.robot.n_joints + 1  # joint deltas + 1 gripper

    @property
    def obs_total_size(self) -> int:
        sizes = self.obs_sizes
        return sum(sizes[k] for k in self.obs_keys)
