"""Stamp stronger PD drive gains onto the PincOpen 4-bar passive joints.

2026-05-21 — gripper rigidity fix. The URDF→USD conversion left
``drive:angular:physics:stiffness = 1.745`` (= 100°, the default URDF
friction-like value) on every 4-bar joint, which is far too weak to hold
the linkage rigid under cube contact loads. The previous workaround was
to add an Isaac Lab ``ImplicitActuatorCfg "mimic"`` entry with
stiffness=2000, but that fights the ``PhysxMimicJointAPI`` constraint
(both target the same joint with different target values), producing the
"흐물흐물" asymmetric deformation the user observed in lift videos.

Fix: stamp the USD drive gains to be strong enough on their own (drive
provides external-load stiffness, mimic API provides the joint-to-gripper
target). The Isaac Lab "mimic" actuator entry must be REMOVED so only one
control source touches the 4-bar joints.

Gains here are conservative — start at 200 / 10 (2.5× the arm joint
stiffness=80), validate with oracle demo, then bump if still floppy.

Run from the repo root::

    OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \\
        python scripts/usd_set_4bar_drive.py
"""

from __future__ import annotations

from pathlib import Path

import pinocchio  # noqa: F401, E402 (pre-AppLauncher Assimp ABI fix)

from isaaclab.app import AppLauncher

_sim_app = AppLauncher(headless=True, enable_cameras=False).app

from pxr import Usd, UsdPhysics  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
USD_PATH = REPO_ROOT / "assets" / "converted" / "usd" / "so101_pincopen.usd"

# 4-bar passive joint paths.
JOINTS = [
    "/so101_pincopen/joints/left_proximal",
    "/so101_pincopen/joints/left_distal",
    "/so101_pincopen/joints/right_proximal",
    "/so101_pincopen/joints/right_distal",
]

# Target drive gains. Arm joints use stiffness=80, damping=4; the
# gripper main drive uses 300/60. The 4-bar passive linkage needs to be
# stiffer than the gripper itself so external load doesn't deflect it
# away from the mimic-constrained pose.
NEW_STIFFNESS = 200.0
NEW_DAMPING = 10.0
NEW_MAX_FORCE = 50.0


def main() -> int:
    stage = Usd.Stage.Open(str(USD_PATH))
    if stage is None:
        raise FileNotFoundError(USD_PATH)

    applied = 0
    for jp in JOINTS:
        prim = stage.GetPrimAtPath(jp)
        if not prim.IsValid():
            print(f"[WARN] joint not found: {jp}", flush=True)
            continue

        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        # Read old values for logging.
        old_k = float(drive.GetStiffnessAttr().Get() or 0.0)
        old_d = float(drive.GetDampingAttr().Get() or 0.0)
        old_f = float(drive.GetMaxForceAttr().Get() or 0.0)

        drive.CreateStiffnessAttr().Set(NEW_STIFFNESS)
        drive.CreateDampingAttr().Set(NEW_DAMPING)
        drive.CreateMaxForceAttr().Set(NEW_MAX_FORCE)
        # Keep targetPosition / targetVelocity at their current value
        # (0.0). The mimic API enforces position; drive just provides
        # external-load stiffness on top.

        print(
            f"[OK] {jp.split('/')[-1]}: "
            f"stiffness {old_k:.3f} -> {NEW_STIFFNESS}, "
            f"damping {old_d:.4f} -> {NEW_DAMPING}, "
            f"maxForce {old_f:.1f} -> {NEW_MAX_FORCE}",
            flush=True,
        )
        applied += 1

    stage.GetRootLayer().Save()
    print(
        f"[DONE] saved {USD_PATH} (updated {applied}/{len(JOINTS)} joints)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    code = main()
    _sim_app.close()
    raise SystemExit(code)
