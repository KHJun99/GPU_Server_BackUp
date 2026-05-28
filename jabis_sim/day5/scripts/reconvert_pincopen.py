"""Re-convert SO-ARM101 PincOpen URDF -> USD with mimic joints unrolled.

Sets convert_mimic_joints_to_normal_joints=True so each former mimic joint
becomes an independent USD joint (no PhysX mimic coupling). After this, the
articulation cfg must explicitly drive the now-independent finger joints.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

URDF = "/home/j-k14d101/jabis_sim/urdf/so101/so101_pincopen_gripper.urdf"
USD_DIR = "/home/j-k14d101/jabis_sim/usd"
USD_NAME = "so101_pincopen.usd"

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

cfg = UrdfConverterCfg(
    asset_path=URDF,
    usd_dir=USD_DIR,
    usd_file_name=USD_NAME,
    force_usd_conversion=True,
    make_instanceable=True,
    fix_base=False,
    merge_fixed_joints=True,
    convert_mimic_joints_to_normal_joints=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=100.0,
            damping=1.0,
        ),
        target_type="position",
    ),
)
print("[reconvert] config:")
for k, v in cfg.__dict__.items():
    print(f"  {k}: {v}")

converter = UrdfConverter(cfg)
print(f"[reconvert] USD written: {converter.usd_path}")
print(f"[reconvert] file size: {Path(converter.usd_path).stat().st_size} B")

simulation_app.close()
