# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

import isaaclab.sim as sim_utils

from isaaclab.assets.articulation import ArticulationCfg
from NoetixRobot.actuators import DelayedImplicitActuatorCfg
from NoetixRobot.assets import ASSET_DIR

ARMATURE_431025 =  0.0018606249999999999
ARMATURE_431536 =  0.007642512
ARMATURE_431040 =  0.0045024
ARMATURE_DM4340 = 0.032
ARMATURE_YKS4315 = 0.033048
ARMATURE_LZ00 =  0.0007
ARMATURE_LZ05 =  0.001

NATURAL_FREQ_1 = 3 * 2.0 * 3.1415926535  #  for 3Hz
NATURAL_FREQ_2 = 4 * 2.0 * 3.1415926535  #  for 4Hz
NATURAL_FREQ_3 = 5 * 2.0 * 3.1415926535  #  for 5Hz

NATURAL_FREQ_4 = 10 * 2.0 * 3.1415926535  #  for 5Hz

noetix_4308_max_torque = 27
noetix_5014_max_torque = 75
noetix_8522_max_torque = 216

noetix_4308_max_vel = 152.78 /60 * 2 * 3.1415926
noetix_5014_max_vel = 132.33 /60 * 2 * 3.1415926
noetix_8522_max_vel = 177.50 /60 * 2 * 3.1415926

BUMI_21DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=f"{ASSET_DIR}/BUMI/BM2-V2.0/urdf/bumi_v2_0904_rl_collision.urdf",
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            "l_leg_yaw_joint": 0.0,
            "l_leg_roll_joint": 0.0,
            "l_leg_pitch_joint": -0.1495,
            "l_knee_pitch_joint": 0.3215,
            "l_ankle_pitch_joint": -0.1720,
            "l_ankle_roll_joint": 0.0,
            "r_leg_yaw_joint": 0.0,
            "r_leg_roll_joint": 0.0,
            "r_leg_pitch_joint": -0.1495,
            "r_knee_pitch_joint": 0.3215,
            "r_ankle_pitch_joint": -0.1720,
            "r_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "l_arm_pitch_joint": 0.0,
            "l_arm_roll_joint": 0.3,
            "l_arm_yaw_joint": 0.0,
            "l_elbow_pitch_joint": 0.0,
            "r_arm_pitch_joint": 0.0,
            "r_arm_roll_joint": -0.3,
            "r_arm_yaw_joint": 0.0,
            "r_elbow_pitch_joint": 0.0,


            # "l_leg_yaw_joint": 0.0,
            # "l_leg_roll_joint": 0.0,
            # "l_leg_pitch_joint": -0.1495,
            # "l_knee_pitch_joint": 0.3215,
            # "l_ankle_pitch_joint": -0.1720,
            # "l_ankle_roll_joint": 0.0,
            # "r_leg_yaw_joint": 0.0,
            # "r_leg_roll_joint": 0.0,
            # "r_leg_pitch_joint": -0.1495,
            # "r_knee_pitch_joint": 0.3215,
            # "r_ankle_pitch_joint": -0.1720,
            # "r_ankle_roll_joint": 0.0,
            # "waist_yaw_joint": 0.08,
            # "l_arm_pitch_joint": -0.2,
            # "l_arm_roll_joint": 0.715,
            # "l_arm_yaw_joint": -0.03,
            # "l_elbow_pitch_joint": -2.0,
            # "r_arm_pitch_joint": -0.26,
            # "r_arm_roll_joint": -0.645,
            # "r_arm_yaw_joint": 0.36,
            # "r_elbow_pitch_joint": -2.14,
        },
        joint_vel={".*": 0.0},
    ),



    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_leg_yaw_joint",
                ".*_leg_roll_joint",
                ".*_leg_pitch_joint",
                ".*_knee_pitch_joint",
            ],
            effort_limit_sim={
                ".*_leg_yaw_joint": noetix_4308_max_torque,
                ".*_leg_roll_joint": noetix_5014_max_torque,
                ".*_leg_pitch_joint": noetix_5014_max_torque,
                ".*_knee_pitch_joint": noetix_5014_max_torque,
            },
            velocity_limit_sim={
                ".*_leg_yaw_joint": noetix_4308_max_vel,
                ".*_leg_roll_joint": noetix_5014_max_vel,
                ".*_leg_pitch_joint": noetix_5014_max_vel,
                ".*_knee_pitch_joint": noetix_5014_max_vel,
            },
            stiffness={
                ".*_leg_yaw_joint": 30,
                ".*_leg_roll_joint": 80,
                ".*_leg_pitch_joint": 80,
                ".*_knee_pitch_joint": 80,
            },
            damping={
                ".*_leg_yaw_joint": 3.0,
                ".*_leg_roll_joint": 4.0,
                ".*_leg_pitch_joint": 4.0,
                ".*_knee_pitch_joint": 4.0,
            },
            armature={
                ".*_leg_yaw_joint": 0.001918208888783688,
                ".*_leg_roll_joint": 0.0029074653328846193,
                ".*_leg_pitch_joint":  0.0029074653328846193,
                ".*_knee_pitch_joint": 0.0029074653328846193,
            },
            min_delay=0,
            max_delay=4,
        ),
        "waist": DelayedImplicitActuatorCfg(
            effort_limit_sim=noetix_5014_max_torque,
            velocity_limit_sim=noetix_5014_max_vel,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=80,
            damping=4,
            min_delay=0,
            armature=0.0029074653328846193,
            max_delay=4,
        ),
        "feet": DelayedImplicitActuatorCfg(
            # armature=0.001918208888783688,
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": noetix_4308_max_torque *1.2 ,
                ".*_ankle_roll_joint": noetix_4308_max_torque *1.2,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": noetix_4308_max_vel,
                ".*_ankle_roll_joint": noetix_4308_max_vel,
            },
            stiffness={
                ".*_ankle_pitch_joint": 8,
                ".*_ankle_roll_joint": 8,
            },
            damping={
                ".*_ankle_pitch_joint": 0.8,
                ".*_ankle_roll_joint": 0.8,
            },
            # friction={
            #     ".*_ankle_pitch_joint": 1.930293,
            #     ".*_ankle_roll_joint": 0.735826,
            # },
            armature={
                ".*_ankle_pitch_joint": 0.001918208888783688,
                ".*_ankle_roll_joint": 0.001918208888783688,
            },
            # viscous_friction={
            #     ".*_ankle_pitch_joint": 0.099857,
            #     ".*_ankle_roll_joint": 0.269741,
            # },
            min_delay=0,
            max_delay=4,
        ),
        "arms": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_arm_pitch_joint",
                ".*_arm_roll_joint",
                ".*_arm_yaw_joint",
                ".*_elbow_pitch_joint",
            ],
            effort_limit_sim=noetix_4308_max_torque,
            velocity_limit_sim=noetix_4308_max_vel,
            stiffness=8,
            damping=0.8,
            armature=0.001918208888783688,
            min_delay=0,
            max_delay=4,
        ),
    },
)
BUMI_21DOF_NAMES=[
     'l_leg_pitch_joint',
     'r_leg_pitch_joint',
     'waist_yaw_joint',
     'l_leg_roll_joint',
     'r_leg_roll_joint',
     'l_arm_pitch_joint',
     'r_arm_pitch_joint',
     'l_leg_yaw_joint',
     'r_leg_yaw_joint',
     'l_arm_roll_joint',
     'r_arm_roll_joint',
     'l_knee_pitch_joint',
     'r_knee_pitch_joint',
     'l_arm_yaw_joint',
     'r_arm_yaw_joint',
     'l_ankle_pitch_joint',
     'r_ankle_pitch_joint',
     'l_elbow_pitch_joint',
     'r_elbow_pitch_joint',
     'l_ankle_roll_joint',
     'r_ankle_roll_joint'
     ]

BUMI_21DOF_ACTION_SCALE = {}
for a in BUMI_21DOF_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            BUMI_21DOF_ACTION_SCALE[n] = 0.25# * e[n] / s[n]


BUMI_12DOF_JOINT_NAMES = [
    "l_leg_yaw_joint",
    "l_leg_roll_joint",
    "l_leg_pitch_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_leg_yaw_joint",
    "r_leg_roll_joint",
    "r_leg_pitch_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
]

BUMI_12DOF_ACTION_SCALE = {}
for actuator_name in ("legs", "feet"):
    a = BUMI_21DOF_CFG.actuators[actuator_name]
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            BUMI_12DOF_ACTION_SCALE[n] = 0.25  # * e[n] / s[n]

BUMI_13DOF_JOINT_NAMES = [
    'waist_yaw_joint',
    "l_leg_yaw_joint",
    "l_leg_roll_joint",
    "l_leg_pitch_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_leg_yaw_joint",
    "r_leg_roll_joint",
    "r_leg_pitch_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
]

BUMI_13DOF_ACTION_SCALE = {}
for actuator_name in ("legs", "feet", "waist"):
    a = BUMI_21DOF_CFG.actuators[actuator_name]
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            BUMI_13DOF_ACTION_SCALE[n] = 0.25
