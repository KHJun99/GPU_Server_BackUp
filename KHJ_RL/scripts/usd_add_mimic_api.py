"""Stamp PhysxMimicJointAPI onto the PincOpen 4-bar passive joints.

URDF → USD conversion drops URDF ``<mimic>`` tags, so the 10 revolute
joints in ``assets/converted/usd/so101_pincopen.usd`` are independent
after the convert step (verified: 0 PhysxMimicJointAPI applications).
This script makes the four 4-bar passive joints follow the gripper main
joint so an ImplicitActuatorCfg driving only ``gripper`` makes the
fingers track — and so that grabbing the wrong joint doesn't cause a
PhysX drive conflict.

The exact mimic sign for each linkage depends on the joint axis chosen
in the original URDF; this script picks ``gearing = -1.0`` (PhysX
convention: ``joint = -gearing * reference``, i.e. ``joint = +reference``)
for every passive joint. If a finger drifts the wrong direction in sim,
flip its sign in MIMIC_DIRECTION below.

Run from the repo root:

    OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \\
        python scripts/usd_add_mimic_api.py
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.app import AppLauncher

# AppLauncher must be created before pxr is touched.
_sim_app = AppLauncher(headless=True, enable_cameras=False).app

from pxr import PhysxSchema, Usd  # noqa: E402  (after AppLauncher)


REPO_ROOT = Path(__file__).resolve().parents[1]
USD_PATH = REPO_ROOT / "assets" / "converted" / "usd" / "so101_pincopen.usd"

REFERENCE_JOINT = "/so101_pincopen/joints/gripper"

# joint name → gearing. PhysX: joint_dof - offset = -gearing * ref_dof.
# Coupling solved from the PincOpen URDF <mimic multiplier=...> attribute:
#   left_proximal  = +0.5 * gripper
#   left_distal    = -0.5 * gripper  (closes opposite to its proximal partner)
#   right_proximal = -0.5 * gripper  (mirror of left side)
#   right_distal   = +0.5 * gripper
# Magnitude 0.5 (NOT ±1.0) — earlier ±1.0 made fingers swing twice as far
# and rake the cube sideways instead of converging.
#
# Critical: this gearing MUST match ``env.py`` ``_mimic_signs`` magnitude
# and sign. If env.py applies PD with target = sign * gripper and USD
# mimic API enforces target = -gearing * gripper, the two must agree:
#   sign = -gearing  ⇒  gearing = -sign
# So env.py sign +0.5 ⇒ USD gearing -0.5, etc.
MIMIC_DIRECTION: dict[str, float] = {
    "left_proximal":  -0.5,  # joint = +0.5 * gripper
    "left_distal":    +0.5,  # joint = -0.5 * gripper
    "right_proximal": +0.5,  # joint = -0.5 * gripper
    "right_distal":   -0.5,  # joint = +0.5 * gripper
}


def _read_axis(joint_prim) -> str:
    """Return the revolute joint's principal axis ('X', 'Y', 'Z'), defaulting to 'X'."""
    attr = joint_prim.GetAttribute("physics:axis")
    if attr and attr.HasValue():
        return str(attr.Get())
    return "X"


def main() -> int:
    stage = Usd.Stage.Open(str(USD_PATH))
    if stage is None:
        raise FileNotFoundError(USD_PATH)

    applied = 0
    for j_name, gearing in MIMIC_DIRECTION.items():
        path = f"/so101_pincopen/joints/{j_name}"
        joint_prim = stage.GetPrimAtPath(path)
        if not joint_prim.IsValid():
            print(f"[WARN] joint not found: {path}")
            continue

        axis = _read_axis(joint_prim)
        instance_name = f"rot{axis}"

        mimic_api = PhysxSchema.PhysxMimicJointAPI.Apply(joint_prim, instance_name)
        mimic_api.CreateReferenceJointRel().SetTargets([REFERENCE_JOINT])
        mimic_api.CreateReferenceJointAxisAttr().Set(f"rot{axis}")
        mimic_api.CreateGearingAttr().Set(gearing)
        mimic_api.CreateOffsetAttr().Set(0.0)
        print(f"[OK]   {j_name}: axis=rot{axis}, gearing={gearing}")
        applied += 1

    stage.GetRootLayer().Save()
    print(f"[DONE] saved {USD_PATH} (applied {applied}/{len(MIMIC_DIRECTION)} mimic APIs)")
    return 0


if __name__ == "__main__":
    code = main()
    _sim_app.close()
    raise SystemExit(code)
