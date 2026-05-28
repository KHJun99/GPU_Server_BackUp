"""Inspect 4-bar passive joints — write findings to file.

Output goes to /tmp/4bar_inspect.txt so the AppLauncher log noise doesn't
drown it out.
"""

from __future__ import annotations

from pathlib import Path

import pinocchio  # noqa: F401 — pre-AppLauncher Assimp ABI fix

from isaaclab.app import AppLauncher

_sim_app = AppLauncher(headless=True, enable_cameras=False).app

from pxr import Usd  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
USD_MAIN = REPO_ROOT / "assets" / "converted" / "usd" / "so101_pincopen.usd"
USD_PHYSICS = (
    REPO_ROOT
    / "assets"
    / "converted"
    / "usd"
    / "configuration"
    / "so101_pincopen_physics.usd"
)
OUT_PATH = Path("/tmp/4bar_inspect.txt")

JOINTS = [
    "/so101_pincopen/joints/left_proximal",
    "/so101_pincopen/joints/left_distal",
    "/so101_pincopen/joints/right_proximal",
    "/so101_pincopen/joints/right_distal",
    "/so101_pincopen/joints/gripper",
]


def inspect(path: Path, out) -> None:
    out.write(f"\n========== {path.name} ==========\n")
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        out.write(f"[WARN] could not open {path}\n")
        return
    for jp in JOINTS:
        prim = stage.GetPrimAtPath(jp)
        if not prim.IsValid():
            out.write(f"  {jp}  <missing>\n")
            continue
        out.write(f"\n  {jp}  type={prim.GetTypeName()}\n")
        schemas = list(prim.GetAppliedSchemas())
        out.write(f"    schemas: {schemas if schemas else '(none)'}\n")
        for attr in prim.GetAttributes():
            n = attr.GetName()
            if any(
                k in n
                for k in [
                    "drive:",
                    "mimic",
                    "physics:stiffness",
                    "physics:damping",
                    "physics:maxForce",
                    "physics:targetPosition",
                    "physics:targetVelocity",
                    "physics:lowerLimit",
                    "physics:upperLimit",
                    "physics:axis",
                    "gearing",
                    "Offset",
                    "ReferenceJoint",
                ]
            ):
                v = attr.Get()
                if v is not None:
                    out.write(f"    {n} = {v}\n")


def main() -> int:
    with open(OUT_PATH, "w") as out:
        inspect(USD_MAIN, out)
        inspect(USD_PHYSICS, out)
    return 0


if __name__ == "__main__":
    code = main()
    _sim_app.close()
    raise SystemExit(code)
