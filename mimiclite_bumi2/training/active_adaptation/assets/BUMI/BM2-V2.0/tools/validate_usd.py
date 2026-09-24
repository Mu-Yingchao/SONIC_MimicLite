#!/usr/bin/env python3
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

try:
    from pxr import UsdPhysics
    import omni.usd
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import SimulationContext, SimulationCfg

    usd_path = "/home/user/zcx/active-adaptation/active_adaptation/assets/BUMI/BM2-V2.0/usd/bumi_v2_0810_rl/bumi_v2_0810_rl.usd"
    sim = SimulationContext(SimulationCfg(dt=0.005, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.482138)),
        actuators={},
    )
    robot = Articulation(robot_cfg)
    sim.reset()
    robot.update(sim.get_physics_dt())
    print("ARTICULATION", "joints", robot.num_joints, "bodies", robot.num_bodies, flush=True)
    print("JOINT_NAMES", robot.joint_names, flush=True)
    print("BODY_NAMES", robot.body_names, flush=True)

    stage = omni.usd.get_context().get_stage()
    colliders = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append((str(prim.GetPath()), prim.GetTypeName()))
    print("COLLIDERS", len(colliders), flush=True)
    for path, type_name in colliders:
        if path.startswith("/World/Robot"):
            print(" COLLIDER", type_name, path, flush=True)
    minimum_root_z = float("inf")
    for _ in range(300):
        sim.step(render=False)
        robot.update(sim.get_physics_dt())
        minimum_root_z = min(minimum_root_z, float(robot.data.root_pos_w[0, 2]))
    print("DROP_FINAL_ROOT_Z", float(robot.data.root_pos_w[0, 2]), flush=True)
    print("DROP_MIN_ROOT_Z", minimum_root_z, flush=True)
finally:
    simulation_app.close()
