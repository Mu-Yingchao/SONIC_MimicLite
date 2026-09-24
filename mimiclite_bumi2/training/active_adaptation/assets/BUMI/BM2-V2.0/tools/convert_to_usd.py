#!/usr/bin/env python3
from pathlib import Path
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)
simulation_app = app.app

try:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    root = Path("/home/user/zcx/active-adaptation/active_adaptation/assets/BUMI/BM2-V2.0")
    source = root / "urdf/bumi_v2_0904_rl_collision.urdf"
    output_dir = root / "usd/bumi_v2_0810_rl"
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = UrdfConverterCfg(
        asset_path=str(source),
        usd_dir=str(output_dir),
        usd_file_name="bumi_v2_0810_rl.usd",
        force_usd_conversion=True,
        make_instanceable=False,
        fix_base=False,
        merge_fixed_joints=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        collider_type="convex_hull",
        self_collision=True,
        replace_cylinders_with_capsules=False,
        collision_from_visuals=False,
    )
    converter = UrdfConverter(cfg)
    print("USD_PATH", converter.usd_path, flush=True)
finally:
    simulation_app.close()
