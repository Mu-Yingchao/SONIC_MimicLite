from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

SOURCE = Path("active_adaptation/assets/BUMI/BM2-V2.0/mjcf/bumi_v2_0810_rl.xml")
SCENE = SOURCE.parent / ".bumi_v2_0810_rl_validation_scene.xml"


def geom_name(model, geom_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom#{geom_id}"


def contact_pairs(model, data, exclude_ground=False):
    pairs = []
    for contact in data.contact:
        a, b = geom_name(model, contact.geom1), geom_name(model, contact.geom2)
        if exclude_ground and "ground" in (a, b):
            continue
        pairs.append((a, b, float(contact.dist)))
    return pairs


# Self-collision at the exact zero-joint pose.
model = mujoco.MjModel.from_xml_path(str(SOURCE))
data = mujoco.MjData(model)
data.qpos[3] = 1.0
mujoco.mj_forward(model, data)
self_pairs = contact_pairs(model, data)
print("default_self_contacts", len(self_pairs))
for pair in self_pairs:
    print(" SELF", pair)

# Add a plane without changing the generated robot MJCF.
tree = ET.parse(SOURCE)
worldbody = tree.getroot().find("worldbody")
worldbody.insert(0, ET.Element("geom", {
    "name": "ground", "type": "plane", "size": "2 2 0.1",
    "rgba": "0.25 0.25 0.25 1", "contype": "1", "conaffinity": "1",
}))
tree.write(SCENE, encoding="utf-8", xml_declaration=True)
scene = mujoco.MjModel.from_xml_path(str(SCENE))
state = mujoco.MjData(scene)
state.qpos[3] = 1.0
mujoco.mj_forward(scene, state)

# Calculate the exact root height that puts the capsule soles on z=0.
bottoms = []
foot_geom_ids = [
    geom_id for geom_id in range(scene.ngeom)
    if "_foot" in geom_name(scene, geom_id) and "_collision" in geom_name(scene, geom_id)
]
if not foot_geom_ids:
    raise RuntimeError("No foot collision geoms found")
for geom_id in foot_geom_ids:
    rotation = state.geom_xmat[geom_id].reshape(3, 3)
    radius, half_length = scene.geom_size[geom_id, :2]
    # A capsule's local z-axis follows its fromto segment.
    z_extent = radius + abs(rotation[2, 2]) * half_length
    bottoms.append(state.geom_xpos[geom_id, 2] - z_extent)
root_ground_z = -min(bottoms)
state.qpos[:] = 0.0
state.qpos[2] = root_ground_z - 0.0005
state.qpos[3] = 1.0
mujoco.mj_forward(scene, state)
ground_pairs = contact_pairs(scene, state)
print("root_ground_z", root_ground_z)
print("ground_pose_contacts", len(ground_pairs))
for pair in ground_pairs:
    print(" GROUND_POSE", pair)

# Drop 5 cm with joint-space PD holding the neutral pose. Root is unassisted.
state = mujoco.MjData(scene)
state.qpos[2] = root_ground_z + 0.05
state.qpos[3] = 1.0
target = np.zeros(scene.nq)
target[3] = 1.0
hinge_joints = [j for j in range(scene.njnt) if scene.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
max_contacts = 0
self_during_drop = set()
steps = int(2.0 / scene.opt.timestep)
for _ in range(steps):
    mujoco.mj_forward(scene, state)
    applied = np.zeros(scene.nv)
    for joint_id in hinge_joints:
        qadr = scene.jnt_qposadr[joint_id]
        dadr = scene.jnt_dofadr[joint_id]
        applied[dadr] = state.qfrc_bias[dadr] - 80.0 * state.qpos[qadr] - 3.0 * state.qvel[dadr]
        force_range = scene.jnt_actfrcrange[joint_id]
        if force_range[0] < force_range[1]:
            applied[dadr] = np.clip(applied[dadr], force_range[0], force_range[1])
    state.qfrc_applied[:] = applied
    mujoco.mj_step(scene, state)
    max_contacts = max(max_contacts, state.ncon)
    for a, b, _ in contact_pairs(scene, state, exclude_ground=True):
        self_during_drop.add(tuple(sorted((a, b))))

finite = bool(np.isfinite(state.qpos).all() and np.isfinite(state.qvel).all())
print("drop_finite", finite)
print("drop_final_root_z", float(state.qpos[2]))
print("drop_max_contacts", max_contacts)
print("drop_self_contact_pairs", len(self_during_drop))
for pair in sorted(self_during_drop):
    print(" DROP_SELF", pair)
