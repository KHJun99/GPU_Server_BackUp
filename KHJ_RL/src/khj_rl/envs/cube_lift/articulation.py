"""Isaac Lab ArticulationCfg for SO-ARM101 + PincOpen.

The USD layer at ``assets/converted/usd/so101_pincopen.usd`` references three
sub-layers (base / physics / sensor). Actuators here drive the 5 SO-ARM101 arm
joints and the gripper main joint; the four PincOpen 4-bar passive joints
(``left_proximal`` / ``left_distal`` / ``right_proximal`` / ``right_distal``)
follow only once the PhysicsMimicJointAPI is added to the USD — that wiring
step is tracked separately from this cfg.

Default PD gains and effort limits are first-pass guesses for the Feetech
STS3215 (1:345 gearing); they MUST be tuned against the real motor before
sim2real transfer (CLAUDE.md "Isaac Lab API 변경성 경고" + STS3215 TODO).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from khj_rl.envs.cube_lift.cfg import RobotCfg

if TYPE_CHECKING:  # avoid pulling carb at module import time
    from isaaclab.assets.articulation import ArticulationCfg


# Repo-relative USD path: src/khj_rl/envs/cube_lift/articulation.py
# → parents[4] = KHJ_RL repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
SO101_PINCOPEN_USD = str(
    _REPO_ROOT / "assets" / "converted" / "usd" / "so101_pincopen.usd"
)


def soarm101_pincopen_cfg(
    robot: RobotCfg | None = None,
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    usd_path: str | None = None,
) -> "ArticulationCfg":
    # Lazy imports — isaaclab.sim / isaaclab.assets pull in carb, which is
    # only valid inside an AppLauncher-booted process. Keeping these here
    # lets path constants in this module be imported without booting kit.
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets.articulation import ArticulationCfg
    """Build an ArticulationCfg for the SO-ARM101 + PincOpen USD.

    The 5 arm joints (``RobotCfg.joint_names``) and the gripper main joint
    (``RobotCfg.gripper_joint_name``) are driven by ImplicitActuatorCfg
    groups. The 4-bar passive joints are left unactuated here — see the
    mimic-API wiring step.
    """
    robot = robot or RobotCfg()
    usd_path = usd_path or SO101_PINCOPEN_USD

    init_joint_pos: dict[str, float] = {
        name: pos for name, pos in zip(robot.joint_names, robot.joint_init)
    }
    init_joint_pos[robot.gripper_joint_name] = 0.0

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # fix_root_link=True turns base_link into a static anchor
                # (table-mount); without it the arm is a floating-base
                # chain and random/IK joint torques tip the whole robot
                # over within seconds. The 6 root DOFs still show up in
                # the PhysX jacobian (oracle accounts for the +6 offset),
                # but the root is no longer free to translate or rotate.
                fix_root_link=True,
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=init_joint_pos,
        ),
        actuators={
            # 5 SO-ARM101 arm joints — Feetech STS3215 with 1:345 reduction.
            # TODO: measure stiffness / damping / effort on the real motor.
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(robot.joint_names),
                effort_limit_sim=10.0,
                velocity_limit_sim=3.14,
                stiffness=80.0,
                damping=4.0,
            ),
            # PincOpen gripper main drive. The 4-bar passive joints follow
            # via PhysicsMimicJointAPI (USD wiring tracked separately).
            # Bumped 10× from the first guess: the original numbers couldn't
            # hold a 50g cube — finger contact moved cube by 1mm and lift
            # came up empty in smoke. TODO: still placeholder, measure on
            # the real STS3215 once available.
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[robot.gripper_joint_name],
                effort_limit_sim=5.0,
                velocity_limit_sim=1.5,
                stiffness=300.0,
                damping=60.0,
            ),
            # PincOpen 4-bar passive joints.
            #
            # History (2026-05-19): tried REMOVING this actuator entry,
            # relying solely on `PhysxMimicJointAPI:rotZ` (USD-stamped via
            # `scripts/usd_add_mimic_api.py`) for the 4-bar coupling. Pilot
            # demo collect failed: oracle MP NEVER lifted the cube — debug
            # prints showed cube_z stayed below lift_z_m the whole episode.
            # MimicAPI alone is insufficient: the constraint apparently does
            # not propagate gripper joint torque to the passive joints under
            # contact load, so the fingers go floppy and the cube slips out.
            # Restored to PD-driven actuator approach (verified working in
            # all prior demo collects). The visual "흐물흐물" asymmetric
            # deformation under load is real (#28 troubleshooting) — it's
            # the cosmetic side-effect of the drive conflict, accepted for
            # now. Anti-pressing guard in success.py (#27) is the operative
            # defense against reward hacking; rigid 4-bar would help but is
            # not required for correct success criterion.
            #
            # 2026-05-21 attempted fix (REVERTED): stamped USD DriveAPI
            # gains (stiffness=200) hoping native drive + MimicJointAPI
            # would replace this actuator. Oracle 0/5 success — USD drive
            # targetPosition=0 actively fights the mimic target so 4-bar
            # joints stay at 0 even when gripper is closed (-1.800),
            # losing grasp entirely. The conflict is fundamental: a
            # joint cannot have a drive (target=fixed value) AND a mimic
            # constraint (target=function of reference) without one
            # winning. Reverted to this entry; cosmetic asymmetry stays.
            #
            # Mimic actuator gains are MUCH stiffer than the gripper main
            # drive — the 4-bar's geometric constraint is *mechanical* on
            # the real robot, so in sim we want the PD to track multiplier
            # × gripper almost rigidly. 50 N·m effort headroom lets the PD
            # follow the target inside one physics tick.
            # 2026-05-21 option C: stiffness 2000→5000, damping 100→200 to
            # tighten finger parallelism under contact load. mimic gearing
            # was already fixed (1.0→0.5 to match env.py PD target), but
            # the user still saw asymmetric "흐물흐물" deformation when the
            # cube was squashed. Bumping PD gain forces the 4-bar joints
            # to track the mimic target more rigidly against external load.
            "mimic": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_proximal", "left_distal",
                    "right_proximal", "right_distal",
                ],
                effort_limit_sim=50.0,
                velocity_limit_sim=1.5,
                stiffness=5000.0,
                damping=200.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
