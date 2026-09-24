#!/usr/bin/env python3
"""View the original bumi_v2_0810_rl.urdf collision geometry only."""

from pathlib import Path
import time
import mujoco
import numpy as np
import trimesh
import viser

ROOT = Path("/home/user/zcx/active-adaptation/active_adaptation/assets/BUMI/BM2-V2.0")
URDF = ROOT / "urdf/bumi_v2_0810_rl.urdf"

model = mujoco.MjModel.from_xml_path(str(URDF))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

server = viser.ViserServer(host="127.0.0.1", port=8091, label="Original BUMI V2 URDF")
server.scene.set_up_direction("+z")
server.scene.add_grid(
    "/ground", width=4.0, height=4.0, plane="xy", cell_size=0.1,
    section_size=0.5, plane_opacity=0.2, shadow_opacity=0.3,
)

mesh_count = 0
sphere_count = 0
for geom_id in range(model.ngeom):
    geom_type = mujoco.mjtGeom(int(model.geom_type[geom_id]))
    body_id = int(model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        # MuJoCo folds base_link into world when importing a URDF directly.
        link_name = "base_link" if geom_id == 0 else body_name
        mesh = trimesh.load_mesh(ROOT / "meshes" / f"{link_name}.STL", process=False)
        mesh.visual.face_colors = [180, 185, 195, 255]
        mesh_count += 1
        path = f"/original_mesh_collisions/{link_name}"
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(
            subdivisions=2, radius=float(model.geom_size[geom_id, 0])
        )
        mesh.visual.face_colors = [235, 55, 55, 255]
        sphere_count += 1
        path = f"/original_foot_spheres/{body_name}_{geom_id}"
    else:
        continue

    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, data.geom_xmat[geom_id])
    server.scene.add_mesh_trimesh(
        path, mesh, position=data.geom_xpos[geom_id], wxyz=quaternion,
        cast_shadow=True, receive_shadow=True,
    )

server.gui.add_markdown(
    "**原始文件：bumi_v2_0810_rl.urdf**  \n"
    "灰色：原始 mesh 碰撞　红色：脚底原始 sphere 碰撞"
)

@server.on_client_connect
def _(client):
    client.camera.position = (1.15, 1.15, 0.85)
    client.camera.look_at = (0.0, 0.0, 0.48)
    client.camera.up_direction = (0.0, 0.0, 1.0)

print("Original URDF viewer: http://127.0.0.1:8091", flush=True)
print(f"source={URDF}", flush=True)
print(f"meshes={mesh_count} spheres={sphere_count} total={model.ngeom}", flush=True)
while True:
    time.sleep(1.0)
