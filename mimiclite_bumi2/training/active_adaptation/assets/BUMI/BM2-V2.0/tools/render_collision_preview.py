from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
import mujoco

ROOT = Path("/home/user/zcx/active-adaptation")
SOURCE = ROOT / "active_adaptation/assets/BUMI/BM2-V2.0/mjcf/bumi_v2_0810_rl.xml"
SCENE = SOURCE.parent / ".bumi_v2_0810_rl_preview_scene.xml"
OUTPUT = ROOT / "outputs/bm2_collision_validation/bumi_v2_visual_collision.png"

tree = ET.parse(SOURCE)
worldbody = tree.getroot().find("worldbody")
worldbody.insert(0, ET.Element("geom", {
    "name": "ground", "type": "plane", "size": "2 2 0.1", "group": "0",
    "rgba": "0.18 0.18 0.20 1", "contype": "1", "conaffinity": "1",
}))
tree.write(SCENE, encoding="utf-8", xml_declaration=True)

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
data.qpos[2] = 0.482138
data.qpos[3] = 1.0
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=480, width=480)

def render(azimuth, collision):
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1
    option.geomgroup[3 if collision else 2] = 1
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.52]
    camera.distance = 1.30
    camera.azimuth = azimuth
    camera.elevation = -8
    renderer.update_scene(data, camera=camera, scene_option=option)
    return renderer.render().copy()

panels = [
    (render(135, False), "Visual · 3/4"),
    (render(135, True), "Collision · 3/4"),
    (render(90, False), "Visual · Side"),
    (render(90, True), "Collision · Side"),
]
canvas = Image.new("RGB", (960, 1008), (30, 30, 34))
draw = ImageDraw.Draw(canvas)
for index, (pixels, title) in enumerate(panels):
    x = (index % 2) * 480
    y = (index // 2) * 504
    canvas.paste(Image.fromarray(pixels), (x, y + 24))
    draw.text((x + 12, y + 5), title, fill=(240, 240, 240))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUTPUT)
print(OUTPUT)
