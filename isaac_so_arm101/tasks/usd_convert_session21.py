"""Session 21 wrapper around IsaacLab convert_urdf for collision-shape ablations.

Adds three CLI flags missing from the stock convert_urdf.py:
  --collision-from-visuals    (default False)        UrdfConverterCfg.collision_from_visuals
  --collider-type             (default convex_hull)  convex_hull|convex_decomposition
  --convert-mimic             (default True)         convert_mimic_joints_to_normal_joints
                                                     IMPORTANT: s20 baseline used True; default cfg is False
                                                     and silently breaks the gripper. Always pass True
                                                     unless explicitly testing native mimic constraints.

Other defaults match s20 baseline (so101_pincopen_gripper.urdf → so101_pincopen.usd).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Session 21 URDF→USD wrapper.")
parser.add_argument("input", type=str)
parser.add_argument("output", type=str)
parser.add_argument("--merge-joints", action="store_true", default=True)
parser.add_argument("--fix-base", action="store_true", default=False)
parser.add_argument("--joint-stiffness", type=float, default=100.0)
parser.add_argument("--joint-damping", type=float, default=1.0)
parser.add_argument("--joint-target-type", type=str, default="position",
                    choices=["position", "velocity", "none"])
parser.add_argument("--collision-from-visuals", action="store_true", default=False)
parser.add_argument("--collider-type", type=str, default="convex_hull",
                    choices=["convex_hull", "convex_decomposition"])
parser.add_argument("--convert-mimic", action="store_true", default=True,
                    help="Convert mimic joints to normal (matches s20 baseline; default True).")
parser.add_argument("--keep-mimic", action="store_true", default=False,
                    help="Keep mimic joints (overrides --convert-mimic). Likely breaks gripper.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict


def main():
    urdf_path = os.path.abspath(args_cli.input)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid URDF path: {urdf_path}")
    dest_path = os.path.abspath(args_cli.output)

    convert_mimic = False if args_cli.keep_mimic else args_cli.convert_mimic

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=args_cli.fix_base,
        merge_fixed_joints=args_cli.merge_joints,
        force_usd_conversion=True,
        convert_mimic_joints_to_normal_joints=convert_mimic,
        collision_from_visuals=args_cli.collision_from_visuals,
        collider_type=args_cli.collider_type,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=args_cli.joint_stiffness,
                damping=args_cli.joint_damping,
            ),
            target_type=args_cli.joint_target_type,
        ),
    )

    print("=" * 80)
    print("[s21 wrapper] URDF→USD configuration:")
    print_dict(cfg.to_dict(), nesting=0)
    print("=" * 80)

    converter = UrdfConverter(cfg)
    print(f"[s21 wrapper] Generated: {converter.usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
