"""USD diagnostic — extract joint list + mimic API metadata from current USD and backup,
print a side-by-side diff. Read-only.

Launches Isaac Sim AppLauncher just to get pxr accessible (it's bundled inside Isaac
Sim's extscache and needs the surrounding LD_LIBRARY_PATH).
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--current_usd", type=str,
                    default="/home/j-k14d101/jabis_sim/usd/configuration/so101_pincopen_physics.usd")
parser.add_argument("--bak_usd", type=str,
                    default="/home/j-k14d101/jabis_sim/usd/configuration.bak.20260508/so101_pincopen_physics.usd")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics  # noqa: E402


JOINT_TYPES = {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint", "PhysicsFixedJoint",
               "PhysicsSphericalJoint", "PhysicsDistanceJoint", "PhysicsJoint"}


def collect_joints(stage: Usd.Stage):
    """Return list of dicts: name, prim_path, type, mimic_ref, mimic_gearing, mimic_offset, body0, body1."""
    out = []
    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        if type_name not in JOINT_TYPES:
            continue
        rec = {
            "name": prim.GetName(),
            "path": str(prim.GetPath()),
            "type": str(type_name),
            "mimic_ref": None,
            "mimic_gearing": None,
            "mimic_offset": None,
            "applied_apis": [str(a) for a in prim.GetAppliedSchemas()],
        }

        joint = UsdPhysics.Joint(prim)
        b0 = joint.GetBody0Rel().GetTargets() if joint else []
        b1 = joint.GetBody1Rel().GetTargets() if joint else []
        rec["body0"] = str(b0[0]) if b0 else ""
        rec["body1"] = str(b1[0]) if b1 else ""

        for attr in prim.GetAttributes():
            an = attr.GetName().lower()
            if "mimic" in an:
                val = attr.Get()
                if "referencejoint" in an or "ref_joint" in an or an.endswith(":referencejoint"):
                    rec["mimic_ref"] = str(val) if val is not None else "(set, no value)"
                elif "gearing" in an:
                    rec["mimic_gearing"] = val
                elif "offset" in an:
                    rec["mimic_offset"] = val
                else:
                    rec.setdefault("other_mimic_attrs", []).append((attr.GetName(), val))
        out.append(rec)
    return out


def fmt_joint(rec):
    bits = [f"type={rec['type'].replace('Physics', '')}"]
    if any("mimic" in s.lower() for s in rec["applied_apis"]):
        bits.append("MIMIC-API")
    if rec.get("mimic_ref"):
        bits.append(f"ref={rec['mimic_ref']}")
    if rec.get("mimic_gearing") is not None:
        bits.append(f"gearing={rec['mimic_gearing']}")
    if rec.get("mimic_offset") is not None:
        bits.append(f"offset={rec['mimic_offset']}")
    other = rec.get("other_mimic_attrs", [])
    if other:
        bits.append(f"extras={other}")
    return " ".join(bits)


def main():
    print(f"\n[USD CURRENT]  {args_cli.current_usd}")
    print(f"[USD BAK]      {args_cli.bak_usd}")

    cur_stage = Usd.Stage.Open(args_cli.current_usd)
    bak_stage = Usd.Stage.Open(args_cli.bak_usd)
    cur = collect_joints(cur_stage)
    bak = collect_joints(bak_stage)

    print(f"\n[D.0] Joint counts — current={len(cur)}  bak={len(bak)}")

    print("\n[D.0a] Applied schemas summary (any prim mentioning 'mimic')")
    for label, joints in [("CURRENT", cur), ("BAK", bak)]:
        for j in joints:
            mimic_apis = [a for a in j["applied_apis"] if "imic" in a]
            if mimic_apis:
                print(f"  [{label}] {j['name']:<18}  apis={mimic_apis}")

    # diff table
    cur_by_name = {j["name"]: j for j in cur}
    bak_by_name = {j["name"]: j for j in bak}
    all_names = sorted(set(cur_by_name.keys()) | set(bak_by_name.keys()))

    print("\n[D.1 USD Mimic Diff] (joint_name | before (bak) | after (current))")
    print("-" * 100)
    for name in all_names:
        b = bak_by_name.get(name)
        c = cur_by_name.get(name)
        b_str = fmt_joint(b) if b else "(missing)"
        c_str = fmt_joint(c) if c else "(missing)"
        changed = b_str != c_str
        marker = "*" if changed else " "
        print(f"{marker} {name:<20} | {b_str:<48} | {c_str}")

    print("\n[D.2] Per-joint full attribute dump (CURRENT only)")
    for j in cur:
        print(f"  {j['name']:<18}  type={j['type']:<28}  body0={j['body0']}  body1={j['body1']}")
        if j["applied_apis"]:
            print(f"    applied_apis: {j['applied_apis']}")
        for k in ("mimic_ref", "mimic_gearing", "mimic_offset"):
            v = j.get(k)
            if v is not None:
                print(f"    {k}: {v}")
        for k, v in j.get("other_mimic_attrs", []):
            print(f"    other_mimic: {k} = {v}")


if __name__ == "__main__":
    main()
    simulation_app.close()
