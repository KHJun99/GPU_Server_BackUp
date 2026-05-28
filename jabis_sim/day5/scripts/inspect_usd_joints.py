"""Inspect SO-ARM101 PincOpen USD articulation joints.

Lists every PhysicsJoint, its type, drive APIs, and whether it carries any
'mimic' property (PhysxSchema.PhysxMimicJointAPI). Outputs a compact summary
table for quick verification after reconversion.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

USD = "/home/j-k14d101/jabis_sim/usd/so101_pincopen.usd"

parser = argparse.ArgumentParser()
parser.add_argument("--usd", default=USD)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics  # noqa: E402

stage = Usd.Stage.Open(args_cli.usd)
print(f"\n=== Stage: {args_cli.usd} ===")
print(f"default_prim: {stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None}")

joint_rows: list[dict[str, object]] = []
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.Joint):
        continue
    name = prim.GetName()
    type_name = prim.GetTypeName()
    has_drive = bool(prim.HasAPI(UsdPhysics.DriveAPI, "angular")) or any(
        s.GetName().startswith("drive:") for s in prim.GetAuthoredPropertiesInNamespace("drive")
    )
    applied_apis = [s for s in prim.GetAppliedSchemas()]
    mimic_apis = [a for a in applied_apis if "Mimic" in a]
    refj = ""
    mult = ""
    for prop in prim.GetAuthoredProperties():
        pn = prop.GetName()
        if pn.startswith("physxMimicJoint:") and pn.endswith(":referenceJoint"):
            tgt = prop.GetTargets() if hasattr(prop, "GetTargets") else None
            if tgt:
                refj = ",".join(str(t) for t in tgt)
        if pn.startswith("physxMimicJoint:") and pn.endswith(":gearing"):
            mult = str(prop.Get())
    joint_rows.append({
        "name": name,
        "type": type_name,
        "drive": "Y" if has_drive else "-",
        "mimic_apis": ";".join(mimic_apis) or "-",
        "ref_joint": refj or "-",
        "gearing": mult or "-",
    })

print(f"\n=== {len(joint_rows)} PhysicsJoints ===")
print(f"| {'name':22s} | {'type':22s} | {'drv':3s} | {'mimic_apis':40s} | {'ref':18s} | gearing |")
print("|" + "-" * 24 + "|" + "-" * 24 + "|" + "-" * 5 + "|" + "-" * 42 + "|" + "-" * 20 + "|" + "-" * 9 + "|")
for r in joint_rows:
    print(f"| {r['name']:22s} | {str(r['type']):22s} | {r['drive']:3s} | "
          f"{r['mimic_apis']:40s} | {r['ref_joint']:18s} | {r['gearing']} |")

simulation_app.close()
