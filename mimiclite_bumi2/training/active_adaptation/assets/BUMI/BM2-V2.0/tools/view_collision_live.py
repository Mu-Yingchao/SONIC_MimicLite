#!/usr/bin/env python3
"""Fast live collision viewer for the hand-tuned BUMI URDF.

Visual STL files are decimated for browser performance. Saving the tuned URDF
automatically regenerates MJCF and refreshes collision primitives.
"""

from pathlib import Path
import subprocess
import sys
import time

import mujoco
import numpy as np
import trimesh
import viser

WORKSPACE = Path("/home/user/zcx/active-adaptation")
ROOT = WORKSPACE / "active_adaptation/assets/BUMI/BM2-V2.0"
URDF = ROOT / "urdf/bumi_v2_0904_rl_collision.urdf"
MJCF = ROOT / "mjcf/bumi_v2_0810_rl.xml"
GENERATOR = ROOT / "tools/generate_mjcf.py"
ROOT_Z = 0.48


def regenerate_mjcf() -> None:
    result = subprocess.run(
        [sys.executable, "-u", str(GENERATOR)], cwd=WORKSPACE,
        text=True, capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)
    print(result.stdout.strip(), flush=True)


def pose_model():
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    data.qpos[2] = ROOT_Z
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    return model, data


def quat_from_matrix(matrix):
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, matrix)
    return quaternion


regenerate_mjcf()
model, data = pose_model()
server = viser.ViserServer(host="127.0.0.1", port=8091, label="BUMI Live Collision Tuning")
server.scene.set_up_direction("+z")
server.scene.add_grid(
    "/ground", width=4.0, height=4.0, plane="xy", cell_size=0.1,
    section_size=0.5, plane_opacity=0.2, shadow_opacity=0.25,
)

# Send only low-poly visual references to the browser.
visual_handles = []
for geom_id in range(model.ngeom):
    if int(model.geom_group[geom_id]) != 2:
        continue
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    link_name = geom_name.removesuffix("_visual")
    mesh = trimesh.load_mesh(ROOT / "meshes" / f"{link_name}.STL", process=True)
    if len(mesh.faces) > 900:
        mesh = mesh.simplify_quadric_decimation(face_count=900, aggression=7)
    mesh.visual.face_colors = [178, 184, 194, 105]
    visual_handles.append(server.scene.add_mesh_trimesh(
        f"/visual_reference/{link_name}", mesh,
        position=data.geom_xpos[geom_id],
        wxyz=quat_from_matrix(data.geom_xmat[geom_id]),
        cast_shadow=False, receive_shadow=False,
    ))


def collision_mesh(model, geom_id):
    geom_type = mujoco.mjtGeom(int(model.geom_type[geom_id]))
    size = model.geom_size[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2.0 * size[:3])
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=float(size[0]), height=float(2.0 * size[1]), sections=32)
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(radius=float(size[0]), height=float(2.0 * size[1]), count=[12, 12])
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    return None


collision_handles = []
generation = 0


def refresh_collisions():
    global collision_handles, generation, model, data
    regenerate_mjcf()
    model, data = pose_model()
    for handle in collision_handles:
        handle.remove()
    collision_handles = []
    generation += 1
    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) != 3:
            continue
        mesh = collision_mesh(model, geom_id)
        if mesh is None:
            continue
        mesh.visual.face_colors = [30, 235, 80, 185]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
        collision_handles.append(server.scene.add_mesh_trimesh(
            f"/live_collisions/v{generation}/{name}", mesh,
            position=data.geom_xpos[geom_id],
            wxyz=quat_from_matrix(data.geom_xmat[geom_id]),
            cast_shadow=False, receive_shadow=False,
        ))
    print(
        f"REFRESH_OK generation={generation} collisions={len(collision_handles)} "
        f"mtime={URDF.stat().st_mtime_ns}", flush=True,
    )


server.gui.add_markdown(
    "**BUMI 实时碰撞调节**  \n"
    "灰色：降面视觉参考　绿色：当前碰撞体  \n"
    "保存 `bumi_v2_0904_rl_collision.urdf` 后约 1 秒自动刷新。"
)
show_visual = server.gui.add_checkbox("显示视觉参考", initial_value=True)
show_collision = server.gui.add_checkbox("显示碰撞体", initial_value=True)

@show_visual.on_update
def _(_event):
    for handle in visual_handles:
        handle.visible = show_visual.value

@show_collision.on_update
def _(_event):
    for handle in collision_handles:
        handle.visible = show_collision.value

@server.on_client_connect
def _(client):
    client.camera.position = (1.15, 1.15, 0.85)
    client.camera.look_at = (0.0, 0.0, 0.48)
    client.camera.up_direction = (0.0, 0.0, 1.0)

refresh_collisions()
last_mtime = URDF.stat().st_mtime_ns
print("BUMI live viewer: http://127.0.0.1:8091", flush=True)
while True:
    time.sleep(0.5)
    current_mtime = URDF.stat().st_mtime_ns
    if current_mtime == last_mtime:
        continue
    try:
        refresh_collisions()
        last_mtime = current_mtime
    except Exception as error:
        print(f"REFRESH_FAILED {error}", flush=True)
