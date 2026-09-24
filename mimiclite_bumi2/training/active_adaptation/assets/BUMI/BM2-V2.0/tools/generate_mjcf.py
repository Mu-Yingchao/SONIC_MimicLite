#!/usr/bin/env python3
"""Generate a floating-base MJCF with visual meshes and primitive collisions."""

from pathlib import Path
import os
import xml.etree.ElementTree as ET

import mujoco

ROOT = Path("active_adaptation/assets/BUMI/BM2-V2.0")
SOURCE = ROOT / "urdf/bumi_v2_0904_rl_collision.urdf"
OUTPUT = ROOT / "mjcf/bumi_v2_0810_rl.xml"
TEMP_URDF = Path("/tmp/bumi_v2_0810_rl_mujoco_import.urdf")
TEMP_XML = Path("/tmp/bumi_v2_0810_rl_mujoco_import.xml")

# Hand-tuned capsule half-lengths from the reviewed MJCF.  Keep these in the
# generator so regenerating the model does not overwrite manual edits.
CAPSULE_COLLISION_OVERRIDES = {
    "l_arm_roll_link_collision_cylinder": {
        "half_length": 0.05,
        "pos": "-0.00547846 0.0160568 -0.05",
    },
    "l_elbow_pitch_link_collision_cylinder": {
        "half_length": 0.065,
        "pos": "-0.00236432 -0.00280334 -0.065",
    },
    "r_arm_roll_link_collision_cylinder": {
        "half_length": 0.05,
        "pos": "-0.00545274 -0.0160147 -0.05",
    },
    "r_elbow_pitch_link_collision_cylinder": {
        "half_length": 0.065,
        "pos": "-0.00234341 -0.00286343 -0.065",
    },
    "l_leg_roll_link_collision_cylinder": {
        "half_length": 0.09,
        "pos": "0.00633064 -9.66653e-05 -0.07",
    },
    "l_knee_pitch_link_collision_cylinder": {
        "half_length": 0.07,
        "pos": "0.00949147 -0.000653414 -0.07",
    },
    "r_leg_roll_link_collision_cylinder": {
        "half_length": 0.09,
        "pos": "0.00640817 -2.70168e-05 -0.07",
    },
    "r_knee_pitch_link_collision_cylinder": {
        "half_length": 0.07,
        "pos": "0.00977739 0.000616862 -0.07",
    },
}

# Replace the remaining torso primitives as well.  Base and waist dimensions
# preserve the old primitive envelope closely; the flatter head cylinder uses
# a compact vertical capsule that stays inside the visual head mesh.
BODY_CAPSULE_OVERRIDES = {
    "base_link_collision_box": (0.056, 0.020),
    "waist_yaw_link_collision_cylinder": (0.095, 0.0207125),
    "head_collision_cylinder": (0.065, 0.010),
}

# Only the ankle-pitch auxiliary capsules remain enabled in the reviewed model.
# Shoulder, arm-yaw, hip-pitch, and hip-yaw fits are deliberately omitted.
ADDITIONAL_LIMB_CAPSULES = {
    "l_ankle_pitch_link": (
        "0.03 0.00131606489 -0.010",
        "0.053645648 -1.56479929 3.14159265",
        0.025,
        0.057,
    ),
    "r_ankle_pitch_link": (
        "0.03 -0.00131939751 -0.010",
        "-0.056251844 -1.56539963 3.14159265",
        0.025,
        0.057,
    ),
}

# Seven overlapping sole capsules per foot, following Unitree G1's contact
# layout while retaining the BUMI sole's original x/y footprint and bottom z.
FOOT_CAPSULE_ROWS = (
    ((0.090, -0.030, -0.039), (0.050, -0.030, -0.039)),
    ((-0.044, -0.020, -0.039), (0.103, -0.020, -0.039)),
    ((-0.050, -0.010, -0.039), (0.108, -0.010, -0.039)),
    ((-0.052, 0.000, -0.039), (0.110, 0.000, -0.039)),
    ((-0.050, 0.010, -0.039), (0.108, 0.010, -0.039)),
    ((-0.044, 0.020, -0.039), (0.103, 0.020, -0.039)),
    ((0.090, 0.030, -0.039), (0.050, 0.030, -0.039)),
)


def collision_geom(name: str, geom_type: str, size: str, **attrs: str) -> ET.Element:
    return ET.Element("geom", {
        "name": name,
        "type": geom_type,
        "size": size,
        "group": "3",
        "contype": "1",
        "conaffinity": "1",
        "rgba": "0.2 0.6 0.2 0.3",
        **attrs,
    })


def add_floating_import_root() -> None:
    tree = ET.parse(SOURCE)
    robot = tree.getroot()
    robot.insert(0, ET.Element("link", {"name": "floating_root"}))
    joint = ET.SubElement(robot, "joint", {"name": "floating_root_fixed", "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": "floating_root"})
    ET.SubElement(joint, "child", {"link": "base_link"})
    ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    extension = ET.SubElement(robot, "mujoco")
    ET.SubElement(extension, "compiler", {"fusestatic": "false"})
    tree.write(TEMP_URDF, encoding="utf-8", xml_declaration=True)


def urdf_color(link: ET.Element) -> str:
    color = link.find("visual/material/color")
    return color.get("rgba") if color is not None else "0.7 0.7 0.7 1"


add_floating_import_root()
spec = mujoco.MjSpec.from_file(str(TEMP_URDF))
spec.body("floating_root").add_freejoint(name="floating_base_joint")
spec.modelname = "bumi_v2_0904_rl"
spec.compile()
spec.to_file(str(TEMP_XML))

tree = ET.parse(TEMP_XML)
root = tree.getroot()
compiler = root.find("compiler")
compiler.set("meshdir", "../meshes")

# Remove the temporary wrapper so base_link itself is the free body.
worldbody = root.find("worldbody")
floating_root = worldbody.find("body[@name='floating_root']")
base = floating_root.find("body[@name='base_link']")
freejoint = floating_root.find("joint[@name='floating_base_joint']")
floating_root.remove(base)
floating_root.remove(freejoint)
base.insert(0, freejoint)
worldbody.remove(floating_root)
worldbody.insert(0, base)

asset = ET.Element("asset")
worldbody_index = list(root).index(worldbody)
root.insert(worldbody_index, asset)

# The reviewed thigh and shin capsules intentionally overlap around the knee.
# Exclude only those two body pairs while keeping all other self-collisions.
contact = ET.Element("contact")
for side in ("l", "r"):
    ET.SubElement(contact, "exclude", {
        "body1": f"{side}_leg_roll_link",
        "body2": f"{side}_knee_pitch_link",
    })
root.insert(worldbody_index + 1, contact)

optimized_urdf = ET.parse(SOURCE).getroot()
bodies = {body.get("name"): body for body in root.findall(".//body")}
mesh_count = 0
for link in optimized_urdf.findall("link"):
    visual = link.find("visual")
    mesh = visual.find("geometry/mesh") if visual is not None else None
    if mesh is None:
        continue
    link_name = link.get("name")
    body = bodies.get(link_name)
    if body is None:
        raise RuntimeError(f"MJCF body missing for visual link {link_name}")
    mesh_name = f"{link_name}_visual_mesh"
    mesh_attrs = {"name": mesh_name, "file": Path(mesh.get("filename")).name}
    if mesh.get("scale"):
        mesh_attrs["scale"] = mesh.get("scale")
    ET.SubElement(asset, "mesh", mesh_attrs)

    visual_geom_attrs = {
        "name": f"{link_name}_visual",
        "type": "mesh",
        "mesh": mesh_name,
        "group": "2",
        "contype": "0",
        "conaffinity": "0",
        "density": "0",
        "rgba": urdf_color(link),
    }
    origin = visual.find("origin")
    if origin is not None:
        if origin.get("xyz"):
            visual_geom_attrs["pos"] = origin.get("xyz")
        if origin.get("rpy"):
            visual_geom_attrs["euler"] = origin.get("rpy")
    body.append(ET.Element("geom", visual_geom_attrs))
    mesh_count += 1

# Primitive collisions use the same group convention as the repository G1 MJCF.
for geom in root.findall(".//geom"):
    if geom.get("contype") == "0":
        continue
    geom.set("group", "3")
    geom.set("contype", "1")
    geom.set("conaffinity", "1")
    geom.set("rgba", "0.2 0.6 0.2 0.3")
    body_override = BODY_CAPSULE_OVERRIDES.get(geom.get("name"))
    if body_override is not None:
        radius, half_length = body_override
        geom.set("type", "capsule")
        geom.set("size", f"{radius:.9g} {half_length:.9g}")
        geom.set("name", geom.get("name").replace("_box", "_capsule").replace(
            "_cylinder", "_capsule"
        ))
    override = CAPSULE_COLLISION_OVERRIDES.get(geom.get("name"))
    if override is not None:
        if geom.get("type") != "cylinder":
            raise RuntimeError(
                f"Expected cylinder for {geom.get('name')}, got {geom.get('type')}"
            )
        size = [float(value) for value in geom.get("size", "").split()]
        if len(size) < 2:
            raise RuntimeError(f"Missing cylinder size for {geom.get('name')}")
        radius = size[0]
        capsule_half_length = override["half_length"]
        geom.set("type", "capsule")
        geom.set("size", f"{radius:.9g} {capsule_half_length:.9g}")
        geom.set("name", geom.get("name").replace("_cylinder", "_capsule"))
        if "pos" in override:
            geom.set("pos", override["pos"])

# Add primitive coverage for limb links whose cylinder fits are intentionally
# commented out in the source URDF.
for body_name, (pos, euler, radius, half_length) in ADDITIONAL_LIMB_CAPSULES.items():
    bodies[body_name].append(collision_geom(
        f"{body_name}_collision_capsule",
        "capsule",
        f"{radius:.9g} {half_length:.9g}",
        pos=pos,
        euler=euler,
    ))

# One spherical hand per arm, attached to the distal elbow link.  The sphere
# overlaps the forearm capsule slightly so there is no collision gap.
for side in ("l", "r"):
    body_name = f"{side}_elbow_pitch_link"
    bodies[body_name].append(collision_geom(
        f"{side}_hand_collision_sphere",
        "sphere",
        "0.035",
        pos="0 0 -0.165",
    ))

# Replace each broad sole box with seven thin, overlapping sole capsules.
for side in ("l", "r"):
    body_name = f"{side}_ankle_roll_link"
    body = bodies[body_name]
    foot_box = body.find(f"geom[@name='{body_name}_foot_box']")
    if foot_box is None:
        raise RuntimeError(f"Missing source foot box on {body_name}")
    body.remove(foot_box)
    for index, (start, end) in enumerate(FOOT_CAPSULE_ROWS, start=1):
        fromto = " ".join(f"{value:.9g}" for value in (*start, *end))
        body.append(collision_geom(
            f"{side}_foot{index}_collision",
            "capsule",
            "0.01",
            fromto=fromto,
        ))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ET.indent(tree, space="  ")
tree.write(OUTPUT, encoding="utf-8", xml_declaration=True)

model = mujoco.MjModel.from_xml_path(str(OUTPUT))
print(
    f"Wrote {OUTPUT}: nbody={model.nbody}, njnt={model.njnt}, nq={model.nq}, "
    f"nv={model.nv}, ngeom={model.ngeom}, nmesh={model.nmesh}, "
    f"mass={model.body_mass.sum():.6f}, visuals={mesh_count}",
    flush=True,
)
os._exit(0)
