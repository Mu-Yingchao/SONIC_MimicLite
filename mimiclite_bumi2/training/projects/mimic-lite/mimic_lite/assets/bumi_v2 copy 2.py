"""BUMI v2 asset configuration shared by the MJLab and IsaacLab backends.

The source URDF is kept as the authoritative robot description, while each
backend consumes its native derived asset:

* MJLab: ``mjcf/bumi_v2_0810_rl.xml``
* IsaacLab: ``usd/bumi_v2_0810_rl/bumi_v2_0810_rl.usd``

This module intentionally depends only on active-adaptation's backend-neutral
asset configuration.  Importing it therefore does not require NoetixRobot or
IsaacLab when running with MJLab.
"""

from __future__ import annotations

import math
from pathlib import Path

from active_adaptation.assets.asset_cfg import (
    ActuatorCfg,
    AssetCfg,
    ContactSensorCfg,
    InitialStateCfg,
    MjlabCollisionCfg,
)
from active_adaptation.registry import Registry
from active_adaptation.utils import symmetry as symmetry_utils


_REPO_ROOT = Path(__file__).resolve().parents[4]
_BUMI_V2_DIR = _REPO_ROOT / "active_adaptation" / "assets" / "BUMI" / "BM2-V2.0"

BUMI_V2_URDF_PATH = _BUMI_V2_DIR / "urdf" / "bumi_v2_0904_rl_collision.urdf"
BUMI_V2_MJCF_PATH = _BUMI_V2_DIR / "mjcf" / "bumi_v2_0810_rl.xml"
BUMI_V2_USD_PATH = (
    _BUMI_V2_DIR / "usd" / "bumi_v2_0810_rl" / "bumi_v2_0810_rl.usd"
)

registry = Registry.instance()


# Order used by the BUMI motion dataset.  Keep this explicit: policy actions,
# motion joint positions, and simulator joints must all resolve to this order.
BUMI_V2_JOINT_NAMES = [
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


# The fixed base_imu body is present in the robot model but not in the 22-body
# motion tensor.  It is placed last so the first 22 entries retain dataset order.
BUMI_V2_BODY_NAMES = [
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
    "base_imu",
]


BUMI_V2_JOINT_SYMMETRY = symmetry_utils.mirrored(
    {
        "l_leg_pitch_joint": (1, "r_leg_pitch_joint"),
        "l_leg_roll_joint": (-1, "r_leg_roll_joint"),
        "l_leg_yaw_joint": (-1, "r_leg_yaw_joint"),
        "l_knee_pitch_joint": (1, "r_knee_pitch_joint"),
        "l_ankle_pitch_joint": (1, "r_ankle_pitch_joint"),
        "l_ankle_roll_joint": (-1, "r_ankle_roll_joint"),
        "waist_yaw_joint": (-1, "waist_yaw_joint"),
        "l_arm_pitch_joint": (1, "r_arm_pitch_joint"),
        "l_arm_roll_joint": (-1, "r_arm_roll_joint"),
        "l_arm_yaw_joint": (-1, "r_arm_yaw_joint"),
        "l_elbow_pitch_joint": (1, "r_elbow_pitch_joint"),
    }
)


BUMI_V2_SPATIAL_SYMMETRY = symmetry_utils.mirrored(
    {
        "base_link": "base_link",
        "waist_yaw_link": "waist_yaw_link",
        "base_imu": "base_imu",
        "l_leg_pitch_link": "r_leg_pitch_link",
        "l_leg_roll_link": "r_leg_roll_link",
        "l_leg_yaw_link": "r_leg_yaw_link",
        "l_knee_pitch_link": "r_knee_pitch_link",
        "l_ankle_pitch_link": "r_ankle_pitch_link",
        "l_ankle_roll_link": "r_ankle_roll_link",
        "l_arm_pitch_link": "r_arm_pitch_link",
        "l_arm_roll_link": "r_arm_roll_link",
        "l_arm_yaw_link": "r_arm_yaw_link",
        "l_elbow_pitch_link": "r_elbow_pitch_link",
    }
)


_NOETIX_4308_MAX_TORQUE = 27.0
_NOETIX_5014_MAX_TORQUE = 75.0
_NOETIX_4308_MAX_VELOCITY = 152.78 / 60.0 * 2.0 * math.pi
_NOETIX_5014_MAX_VELOCITY = 132.33 / 60.0 * 2.0 * math.pi


def _actuators() -> dict[str, ActuatorCfg]:
    """Create non-overlapping actuator groups using the BUMI control gains."""

    return {
        "leg_yaw": ActuatorCfg(
            joint_names_expr=".*_leg_yaw_joint",
            effort_limit=_NOETIX_4308_MAX_TORQUE,
            velocity_limit=_NOETIX_4308_MAX_VELOCITY,
            stiffness=30.0,
            damping=3.0,
            friction=0.0,
            armature=0.0,
        ),
        "legs": ActuatorCfg(
            joint_names_expr=[
                ".*_leg_pitch_joint",
                ".*_leg_roll_joint",
                ".*_knee_pitch_joint",
            ],
            effort_limit=_NOETIX_5014_MAX_TORQUE,
            velocity_limit=_NOETIX_5014_MAX_VELOCITY,
            stiffness=80.0,
            damping=4.0,
            friction=0.0,
            armature=0.0,
        ),
        "waist": ActuatorCfg(
            joint_names_expr="waist_yaw_joint",
            effort_limit=_NOETIX_5014_MAX_TORQUE,
            velocity_limit=_NOETIX_5014_MAX_VELOCITY,
            stiffness=80.0,
            damping=4.0,
            friction=0.0,
            armature=0.0,
        ),
        "feet": ActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=_NOETIX_4308_MAX_TORQUE * 1.2,
            velocity_limit=_NOETIX_4308_MAX_VELOCITY,
            stiffness=8.0,
            damping=0.8,
            friction=0.0,
            armature=0.01,
        ),
        "arms": ActuatorCfg(
            joint_names_expr=[
                ".*_arm_pitch_joint",
                ".*_arm_roll_joint",
                ".*_arm_yaw_joint",
                ".*_elbow_pitch_joint",
            ],
            effort_limit=_NOETIX_4308_MAX_TORQUE,
            velocity_limit=_NOETIX_4308_MAX_VELOCITY,
            stiffness=8.0,
            damping=0.8,
            friction=0.0,
            armature=0.0,
        ),
    }


def _make_bumi_v2_cfg(*, self_collisions: bool) -> AssetCfg:
    return AssetCfg(
        joint_names_simulation=BUMI_V2_JOINT_NAMES,
        body_names_simulation=BUMI_V2_BODY_NAMES,
        joint_symmetry_mapping=BUMI_V2_JOINT_SYMMETRY,
        spatial_symmetry_mapping=BUMI_V2_SPATIAL_SYMMETRY,
        usd_path=str(BUMI_V2_USD_PATH),
        mjcf_path=str(BUMI_V2_MJCF_PATH),
        init_state=InitialStateCfg(
            pos=(0.0, 0.0, 0.65),
            joint_pos={
                ".*_leg_yaw_joint": 0.0,
                ".*_leg_roll_joint": 0.0,
                ".*_leg_pitch_joint": -0.1495,
                ".*_knee_pitch_joint": 0.3215,
                ".*_ankle_pitch_joint": -0.1720,
                ".*_ankle_roll_joint": 0.0,
                "waist_yaw_joint": 0.0,
                "l_arm_pitch_joint": 0.0,
                "l_arm_roll_joint": 0.3,
                "l_arm_yaw_joint": 0.0,
                "l_elbow_pitch_joint": 0.0,
                "r_arm_pitch_joint": 0.0,
                "r_arm_roll_joint": -0.3,
                "r_arm_yaw_joint": 0.0,
                "r_elbow_pitch_joint": 0.0,
                ".*": 0.0,
            },
        ),
        actuators=_actuators(),
        self_collisions=self_collisions,
        sensors_isaaclab=[
            ContactSensorCfg(
                name="contact_forces",
                primary=".*",
                history_length=4,
                track_air_time=True,
            ),
            # Ground reaction at the two feet is handled by contact_forces;
            # excluding the foot bodies makes this sensor represent self hits.
            ContactSensorCfg(
                name="self_collision",
                primary=r"^(?!.*_ankle_roll_link$).+$",
                history_length=4,
            ),
        ],
        sensors_mjlab=[
            ContactSensorCfg(
                name="contact_forces",
                primary_contact_match_mode="subtree",
                primary_contact_match_pattern=r"^(l_ankle_roll_link|r_ankle_roll_link)$",
                primary_contact_match_entity="robot",
                secondary_contact_match_mode="body",
                secondary_contact_match_pattern="terrain",
                # The plane body is a global MuJoCo body named ``terrain``.
                # Scoping it to the terrain entity would prefix it a second
                # time (``terrain/terrain``) and make scene compilation fail.
                secondary_contact_match_entity=None,
                fields=("found", "force"),
                reduce="netforce",
                num_slots=1,
                history_length=4,
                track_air_time=True,
            ),
            ContactSensorCfg(
                name="self_collision",
                primary_contact_match_mode="subtree",
                primary_contact_match_pattern="base_link",
                primary_contact_match_entity="robot",
                secondary_contact_match_mode="subtree",
                secondary_contact_match_pattern="base_link",
                secondary_contact_match_entity="robot",
                fields=("found", "force"),
                reduce="none",
                history_length=4,
            ),
        ],
        mjlab_collisions=[
            # Generated collision geoms include *_collision_* for the body
            # All generated primitive collisions, including the seven sole
            # capsules per foot, contain `_collision` in their names.  Keep
            # the legacy foot-box expression for older exported models.
            MjlabCollisionCfg(
                geom_names_expr=(".*_collision.*", ".*_foot_box"),
                friction=(0.8, 0.02, 0.001),
                disable_other_geoms=False,
            ),
        ],
    )


# Match the imported BUMI configuration by enabling robot self-collision.  The
# alternative is useful during initial policy bring-up if self contacts dominate.
BUMI_V2_CFG = _make_bumi_v2_cfg(self_collisions=True)
BUMI_V2_NO_SELF_COLLISION_CFG = _make_bumi_v2_cfg(self_collisions=False)


registry.register("asset", "bumi-v2", BUMI_V2_CFG)
registry.register(
    "asset",
    "bumi-v2-no-self-collision",
    BUMI_V2_NO_SELF_COLLISION_CFG,
)
