"""Render exported CAD scenes with MuJoCo's offscreen OpenGL renderer."""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np


def render(request_path):
    request = json.loads(Path(request_path).read_text(encoding='utf-8'))
    model = mujoco.MjModel.from_xml_path(str(Path(request['scene']).resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = request['lookat_m']
    camera.distance = request['distance_m']
    camera.azimuth = request.get('azimuth_deg', 135)
    camera.elevation = request.get('elevation_deg', -16)

    width, height = request.get('size_px', (1600, 1200))
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
    output = Path(request['output']).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(output)
    return output


if __name__ == '__main__':
    print(render(sys.argv[1]))
