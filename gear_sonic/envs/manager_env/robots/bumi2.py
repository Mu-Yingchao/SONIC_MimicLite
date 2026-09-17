# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""BUMI2（BM2-V2.0）在 Isaac Lab 与 SONIC 中使用的原生机器人配置。

本模块严格镜像 ``bumi3.py`` 的实现方式，只把关节/刚体名称、URDF/MJCF 路径和
执行器数值换成 BUMI2。参考来源是同事提供的 ``MimicLite_bumi2.tar.gz`` 里
``active_adaptation/assets/BUMI/BM2-V2.0/bumi.py``（该文件 SHA256：
``c47c3706d352f7b5c9bb533bfcb5f8631f1c7a21117b04c93e59c0804d518e91``），这是
同事已经用 MimicLite 训出好效果的那份 BUMI2 权威机器人定义，不是凭空假设的
参数。URDF 源文件为 ``urdf/bumi_v2_0904_rl_collision.urdf``（SHA256：
``b595f0060015944eb489c4a5b3d36f6d7914399bb326f55a97ae5de0afef0f13``），
MJCF 源文件为 ``mjcf/bumi_v2_0810_rl.xml``（SHA256：
``14cc43828ff038dfae7b40d77d5a56b4ee234d9be32f3319f190038cd1507005``）。

关键结论（写这份文件时已核实，供后续复查）：
1. BUMI2 与 BUMI3 的关节/刚体命名和排列顺序完全一致（21 自由度、同一组
   关节名、Isaac Lab 与 MuJoCo 下的排列规则相同），因此下面的 DOF/Body
   映射数组与 ``bumi3.py`` 里的完全相同，只是分别独立生成、独立断言，
   不直接复用 BUMI3 的变量，避免两个机器人的模块产生隐性耦合。
2. BUMI2 与 BUMI3 的执行器数值（力矩、速度、KP、KD、armature）明显不同，
   典型例子是踝关节力矩上限：BUMI2 约 32.4N·m，BUMI3 仅 9N·m，相差约
   3.6 倍——这与用户说明的"同尺寸、换了电机"一致，不能把两者的执行器
   参数混用。
3. ``BUMI2_ACTION_SCALE`` 没有采用 ``bumi3.py`` 那种
   ``0.25 * effort_limit_sim / stiffness`` 公式实时推导的方式。原因是
   实测同事真实训练并导出的 BUMI2 部署配置（``BumiV2TrackBase`` checkpoint
   39000 的 ``deploy.yaml``）里的逐关节 action_scale，和用这个公式套用
   ``bumi.py`` 里的 effort/stiffness 算出来的值对不上（例如 leg_yaw 公式
   算出 0.225，实际部署值是 0.45；leg_roll 和 leg_pitch 公式算出的比例
   相同，但实际部署值分别是 0.35 和 0.45，说明部署时使用的不是这个公式，
   具体来源尚未查证）。为了不把一个已知对不上的公式当成事实，这里改为
   直接把 deploy.yaml 里的逐关节实测值抄成字面常量，并在下面单独注明。
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


ASSET_DIR = "gear_sonic/data/assets"


BUMI2_MUJOCO_DOF_NAMES = [
    "waist_yaw_joint",
    "l_arm_pitch_joint",
    "l_arm_roll_joint",
    "l_arm_yaw_joint",
    "l_elbow_pitch_joint",
    "r_arm_pitch_joint",
    "r_arm_roll_joint",
    "r_arm_yaw_joint",
    "r_elbow_pitch_joint",
    "l_leg_pitch_joint",
    "l_leg_roll_joint",
    "l_leg_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_leg_pitch_joint",
    "r_leg_roll_joint",
    "r_leg_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
]

BUMI2_ISAACLAB_DOF_NAMES = [
    "l_leg_pitch_joint",
    "r_leg_pitch_joint",
    "waist_yaw_joint",
    "l_leg_roll_joint",
    "r_leg_roll_joint",
    "l_arm_pitch_joint",
    "r_arm_pitch_joint",
    "l_leg_yaw_joint",
    "r_leg_yaw_joint",
    "l_arm_roll_joint",
    "r_arm_roll_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_arm_yaw_joint",
    "r_arm_yaw_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_elbow_pitch_joint",
    "r_elbow_pitch_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
]

BUMI2_MUJOCO_BODY_NAMES = [
    "base_link",
    "waist_yaw_link",
    "l_arm_pitch_link",
    "l_arm_roll_link",
    "l_arm_yaw_link",
    "l_elbow_pitch_link",
    "r_arm_pitch_link",
    "r_arm_roll_link",
    "r_arm_yaw_link",
    "r_elbow_pitch_link",
    "l_leg_pitch_link",
    "l_leg_roll_link",
    "l_leg_yaw_link",
    "l_knee_pitch_link",
    "l_ankle_pitch_link",
    "l_ankle_roll_link",
    "r_leg_pitch_link",
    "r_leg_roll_link",
    "r_leg_yaw_link",
    "r_knee_pitch_link",
    "r_ankle_pitch_link",
    "r_ankle_roll_link",
]

# Isaac Lab 的刚体顺序等于根刚体加 URDF 导入后的 21 个关节子刚体顺序。
BUMI2_ISAACLAB_BODY_NAMES = [
    "base_link",
    "l_leg_pitch_link",
    "r_leg_pitch_link",
    "waist_yaw_link",
    "l_leg_roll_link",
    "r_leg_roll_link",
    "l_arm_pitch_link",
    "r_arm_pitch_link",
    "l_leg_yaw_link",
    "r_leg_yaw_link",
    "l_arm_roll_link",
    "r_arm_roll_link",
    "l_knee_pitch_link",
    "r_knee_pitch_link",
    "l_arm_yaw_link",
    "r_arm_yaw_link",
    "l_ankle_pitch_link",
    "r_ankle_pitch_link",
    "l_elbow_pitch_link",
    "r_elbow_pitch_link",
    "l_ankle_roll_link",
    "r_ankle_roll_link",
]


def _target_order_indices(source_names: list[str], target_names: list[str]) -> list[int]:
    """按名称生成从 source 排列到 target 排列所需的索引（与 bumi3.py 完全一致的实现）。"""

    assert len(source_names) == len(set(source_names)), "source 名称存在重复"
    assert len(target_names) == len(set(target_names)), "target 名称存在重复"
    assert set(source_names) == set(target_names), "source/target 名称集合不一致"
    return [source_names.index(name) for name in target_names]


BUMI2_ISAACLAB_TO_MUJOCO_DOF = _target_order_indices(
    BUMI2_ISAACLAB_DOF_NAMES, BUMI2_MUJOCO_DOF_NAMES
)
BUMI2_MUJOCO_TO_ISAACLAB_DOF = _target_order_indices(
    BUMI2_MUJOCO_DOF_NAMES, BUMI2_ISAACLAB_DOF_NAMES
)
BUMI2_ISAACLAB_TO_MUJOCO_BODY = _target_order_indices(
    BUMI2_ISAACLAB_BODY_NAMES, BUMI2_MUJOCO_BODY_NAMES
)
BUMI2_MUJOCO_TO_ISAACLAB_BODY = _target_order_indices(
    BUMI2_MUJOCO_BODY_NAMES, BUMI2_ISAACLAB_BODY_NAMES
)

# 与 BUMI3 的关节/刚体命名和排列规则完全相同，因此这四个映射数组的数值
# 也应与 bumi3.py 里的断言完全一致；这里独立断言，既核实本文件自己的
# 命名表没有写错，也顺带验证了"BUMI2/BUMI3 排列约定相同"这条结论。
assert BUMI2_ISAACLAB_TO_MUJOCO_DOF == [
    2, 5, 9, 13, 17, 6, 10, 14, 18, 0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20,
]
assert BUMI2_MUJOCO_TO_ISAACLAB_DOF == [
    9, 15, 0, 10, 16, 1, 5, 11, 17, 2, 6, 12, 18, 3, 7, 13, 19, 4, 8, 14, 20,
]

BUMI2_LOWER_JOINT_INDICES_MUJOCO = list(range(9, 21))

BUMI2_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": BUMI2_ISAACLAB_BODY_NAMES,
    "isaaclab_dof_names": BUMI2_ISAACLAB_DOF_NAMES,
    "mujoco_dof_names": BUMI2_MUJOCO_DOF_NAMES,
    "isaaclab_to_mujoco_dof": BUMI2_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": BUMI2_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": BUMI2_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": BUMI2_MUJOCO_TO_ISAACLAB_BODY,
    "lower_joint_indices_mujoco": BUMI2_LOWER_JOINT_INDICES_MUJOCO,
}


# noetix_4308（27N·m，约 15.998rad/s）驱动腿部 yaw、全部手臂/肘、踝关节
# （踝关节力矩额外乘 1.2 倍）；noetix_5014（75N·m，约 13.857rad/s）驱动
# 腿部 roll/pitch、膝、腰。数值直接取自参考 bumi.py 的
# noetix_4308_max_torque/vel、noetix_5014_max_torque/vel 计算结果，
# 不是估算值。
BUMI2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=f"{ASSET_DIR}/robot_description/urdf/bumi2/bumi.urdf",
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 与 BUMI3 相同的根部初始高度和腿部/手臂姿态，来自参考 bumi.py
        # 的 init_state，两代机型的静止站姿标定一致。
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
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_leg_yaw_joint",
                ".*_leg_roll_joint",
                ".*_leg_pitch_joint",
                ".*_knee_pitch_joint",
            ],
            effort_limit_sim={
                ".*_leg_yaw_joint": 27.0,
                ".*_leg_roll_joint": 75.0,
                ".*_leg_pitch_joint": 75.0,
                ".*_knee_pitch_joint": 75.0,
            },
            velocity_limit_sim={
                ".*_leg_yaw_joint": 15.9977,
                ".*_leg_roll_joint": 13.8567,
                ".*_leg_pitch_joint": 13.8567,
                ".*_knee_pitch_joint": 13.8567,
            },
            stiffness={
                ".*_leg_yaw_joint": 30.0,
                ".*_leg_roll_joint": 80.0,
                ".*_leg_pitch_joint": 80.0,
                ".*_knee_pitch_joint": 80.0,
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
                ".*_leg_pitch_joint": 0.0029074653328846193,
                ".*_knee_pitch_joint": 0.0029074653328846193,
            },
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=75.0,
            velocity_limit_sim=13.8567,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=80.0,
            damping=4.0,
            armature=0.0029074653328846193,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            # BUMI2 踝关节力矩额外乘 1.2（noetix_4308_max_torque * 1.2），
            # 明显强于 BUMI3 的 9N·m，是两代机型执行器差异最大的一组关节。
            effort_limit_sim=32.4,
            velocity_limit_sim=15.9977,
            stiffness={".*_ankle_pitch_joint": 8.0, ".*_ankle_roll_joint": 8.0},
            damping={".*_ankle_pitch_joint": 0.8, ".*_ankle_roll_joint": 0.8},
            armature={
                ".*_ankle_pitch_joint": 0.001918208888783688,
                ".*_ankle_roll_joint": 0.001918208888783688,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_arm_pitch_joint",
                ".*_arm_roll_joint",
                ".*_arm_yaw_joint",
                ".*_elbow_pitch_joint",
            ],
            effort_limit_sim=27.0,
            velocity_limit_sim=15.9977,
            stiffness=8.0,
            damping=0.8,
            armature=0.001918208888783688,
        ),
    },
)


# 直接抄自同事真实训练并导出的 BUMI2 部署配置（BumiV2TrackBase
# checkpoint_39000 deploy.yaml 的 action_scale 字段，按
# joint_names_simulation 顺序逐关节核对后转成按关节名索引的字典），
# 不是用 effort/stiffness 公式现算的——公式算出来的值和这份实测值对不上，
# 详见文件顶部说明。左右对称关节的值经核对完全相同。
BUMI2_ACTION_SCALE = {
    "l_leg_yaw_joint": 0.45,
    "r_leg_yaw_joint": 0.45,
    "l_leg_roll_joint": 0.35,
    "r_leg_roll_joint": 0.35,
    "l_leg_pitch_joint": 0.45,
    "r_leg_pitch_joint": 0.45,
    "l_knee_pitch_joint": 0.35,
    "r_knee_pitch_joint": 0.35,
    "waist_yaw_joint": 0.45,
    "l_ankle_pitch_joint": 0.3,
    "r_ankle_pitch_joint": 0.3,
    "l_ankle_roll_joint": 0.25,
    "r_ankle_roll_joint": 0.25,
    "l_arm_pitch_joint": 0.4,
    "r_arm_pitch_joint": 0.4,
    "l_arm_roll_joint": 0.4,
    "r_arm_roll_joint": 0.4,
    "l_arm_yaw_joint": 0.4,
    "r_arm_yaw_joint": 0.4,
    "l_elbow_pitch_joint": 0.4,
    "r_elbow_pitch_joint": 0.4,
}
assert set(BUMI2_ACTION_SCALE) == set(BUMI2_MUJOCO_DOF_NAMES), (
    "BUMI2_ACTION_SCALE 的关节集合必须和 21 自由度关节表完全一致"
)
