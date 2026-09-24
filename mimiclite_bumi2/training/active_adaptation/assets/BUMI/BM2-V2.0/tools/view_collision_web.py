#!/usr/bin/env python3
"""Lightweight interactive BUMI visual/original/optimized collision viewer."""

from pathlib import Path
import time
import mujoco
import numpy as np
import trimesh
import viser

ROOT = Path("/home/user/zcx/active-adaptation/active_adaptation/assets/BUMI/BM2-V2.0")
model = mujoco.MjModel.from_xml_path(str(ROOT / "mjcf/bumi_v2_0810_rl.xml"))
data = mujoco.MjData(model)
data.qpos[2] = 0.482138
data.qpos[3] = 1.0
mujoco.mj_forward(model, data)

server = viser.ViserServer(host="127.0.0.1", port=8091, label="BUMI V2 Collision Viewer")
server.scene.set_up_direction("+z")
server.scene.add_grid(
    "/ground", width=4.0, height=4.0, cell_size=0.1, section_size=0.5,
    plane="xy", plane_opacity=0.25, shadow_opacity=0.35,
)
mesh_handles = []
optimized_handles = []
foot_sphere_handles = []

for geom_id in range(model.ngeom):
    group = int(model.geom_group[geom_id])
    if group not in (2, 3):
        continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
    geom_type = mujoco.mjtGeom(int(model.geom_type[geom_id]))
    if group == 2:
        link_name = name.removesuffix("_visual")
        mesh = trimesh.load_mesh(ROOT / "meshes" / f"{link_name}.STL", process=False)
        mesh.visual.face_colors = np.clip(model.geom_rgba[geom_id] * 255, 0, 255).astype(np.uint8)
        path = f"/original_mesh_collisions/{link_name}"
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2.0 * model.geom_size[geom_id, :3])
        mesh.visual.face_colors = [40, 220, 90, 130]
        path = f"/optimized_collisions/{name}"
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(
            radius=float(model.geom_size[geom_id, 0]),
            height=float(2.0 * model.geom_size[geom_id, 1]), sections=32,
        )
        mesh.visual.face_colors = [40, 220, 90, 130]
        path = f"/optimized_collisions/{name}"
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(
            radius=float(model.geom_size[geom_id, 0]),
            height=float(2.0 * model.geom_size[geom_id, 1]),
            count=[12, 12],
        )
        mesh.visual.face_colors = [40, 220, 90, 130]
        path = f"/optimized_collisions/{name}"
    else:
        continue
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, data.geom_xmat[geom_id])
    handle = server.scene.add_mesh_trimesh(
        path, mesh, position=data.geom_xpos[geom_id], wxyz=quat,
        cast_shadow=(group == 2), receive_shadow=(group == 2),
    )
    (mesh_handles if group == 2 else optimized_handles).append(handle)

# Original URDF: eight 5 mm support spheres under each foot.
support_points = (
    (-0.047, -0.030, -0.046), (-0.047, 0.030, -0.046),
    (0.010, -0.028, -0.046), (0.010, 0.028, -0.046),
    (0.060, -0.035, -0.046), (0.060, 0.035, -0.046),
    (0.105, -0.025, -0.046), (0.105, 0.025, -0.046),
)
for link_name in ("l_ankle_roll_link", "r_ankle_roll_link"):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    rotation = data.xmat[body_id].reshape(3, 3)
    for index, point in enumerate(support_points):
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.005)
        sphere.visual.face_colors = [235, 55, 55, 230]
        foot_sphere_handles.append(server.scene.add_mesh_trimesh(
            f"/original_foot_spheres/{link_name}_{index}", sphere,
            position=data.xpos[body_id] + rotation @ np.asarray(point),
            visible=True, cast_shadow=False, receive_shadow=False,
        ))

server.gui.add_markdown(
    "**BUMI V2 · 碰撞检查**  \n"
    "灰色 STL 就是原始网格碰撞；红色是原始脚底 16 球；绿色是优化碰撞。"
)
show_mesh = server.gui.add_checkbox("原始网格碰撞 / 视觉（灰）", initial_value=True)
show_spheres = server.gui.add_checkbox("原始脚底球（红）", initial_value=True)
show_optimized = server.gui.add_checkbox("优化碰撞（绿）", initial_value=False)

@show_mesh.on_update
def _(_event):
    for handle in mesh_handles:
        handle.visible = show_mesh.value

@show_spheres.on_update
def _(_event):
    for handle in foot_sphere_handles:
        handle.visible = show_spheres.value

@show_optimized.on_update
def _(_event):
    for handle in optimized_handles:
        handle.visible = show_optimized.value

@server.on_client_connect
def _(client):
    client.camera.position = (1.15, 1.15, 0.85)
    client.camera.look_at = (0.0, 0.0, 0.48)
    client.camera.up_direction = (0.0, 0.0, 1.0)

print("BUMI viewer ready: http://127.0.0.1:8091", flush=True)
print(
    f"original_meshes={len(mesh_handles)} foot_spheres={len(foot_sphere_handles)} "
    f"optimized={len(optimized_handles)}", flush=True,
)
while True:
    time.sleep(1.0)
