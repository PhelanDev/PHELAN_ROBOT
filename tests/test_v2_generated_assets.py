"""Keep scene-free Isaac interchange assets synchronized with canonical V2."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from duck_v2.robot_spec import MODEL_PATH, ROBOT_ONLY_MJCF_PATH, URDF_PATH
from duck_v2.scripts.build_robot_only_mjcf import build_robot_only_mjcf
from duck_v2.scripts import export_v2_urdf as urdf_exporter


def test_robot_only_mjcf_is_fresh_and_has_no_scene(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    generated = build_robot_only_mjcf(output=tmp_path / "robot.xml")
    assert generated.read_bytes() == ROBOT_ONLY_MJCF_PATH.read_bytes(), (
        "Regenerate duck_v2/cyberduck_v2_robot.xml after editing the canonical MJCF"
    )
    source = ET.parse(MODEL_PATH).getroot()
    robot = ET.parse(generated).getroot()
    assert robot.find("visual") is None
    assert robot.findall("worldbody/geom") == []
    assert robot.findall("worldbody/light") == []
    assert len(robot.findall("worldbody/body")) == 1
    for section in ("worldbody/body", "actuator", "sensor", "default"):
        actual = ET.canonicalize(ET.tostring(robot.find(section), encoding="unicode"), strip_text=True)
        expected = ET.canonicalize(ET.tostring(source.find(section), encoding="unicode"), strip_text=True)
        assert actual == expected


def test_robot_only_compiles_with_identical_articulation_dynamics():
    scene = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    robot = mujoco.MjModel.from_xml_path(str(ROBOT_ONLY_MJCF_PATH))
    for count in ("nbody", "nq", "nv", "nu", "nsensor"):
        assert getattr(scene, count) == getattr(robot, count)
    for field in (
        "body_mass", "body_pos", "body_quat", "body_ipos", "body_inertia",
        "jnt_axis", "jnt_range", "actuator_trnid", "actuator_gear",
        "actuator_ctrlrange", "actuator_forcerange", "sensor_type", "sensor_objid",
    ):
        np.testing.assert_array_equal(getattr(scene, field), getattr(robot, field))
    assert scene.ngeom > robot.ngeom
    for pose in (np.zeros(8), np.linspace(0.01, 0.08, 8)):
        dynamics = []
        for model in (scene, robot):
            data = mujoco.MjData(model)
            data.qpos[7:] = pose
            mujoco.mj_forward(model, data)
            inertia = data.M if hasattr(data, "M") else data.qM
            dynamics.append((inertia.copy(), data.qfrc_bias.copy()))
        for source, derived in zip(*dynamics):
            np.testing.assert_allclose(source, derived, atol=1e-12, rtol=0)


def test_urdf_is_fresh_and_uses_existing_visual_meshes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    robot = build_robot_only_mjcf(output=tmp_path / "robot.xml")
    # Export into a temporary folder without touching the shipped MJCF asset.
    monkeypatch.setattr(urdf_exporter, "build_robot_only_mjcf", lambda: robot)
    generated = urdf_exporter.export_v2_urdf(output=tmp_path / "robot.urdf")
    assert generated.read_bytes() == URDF_PATH.read_bytes(), (
        "Regenerate duck_v2/cyberduck_v2.urdf after editing the canonical MJCF"
    )
    urdf = ET.parse(generated).getroot()
    visual_meshes = urdf.findall("link/visual/geometry/mesh")
    assert visual_meshes
    for mesh in visual_meshes:
        assert (URDF_PATH.parent / mesh.get("filename")).is_file()
    assert urdf.findall("link/collision/geometry/mesh") == []
    assert urdf.find("joint[@type='floating']/child").get("link") == "trunk_base"
    assert len(urdf.findall("joint[@type='revolute']")) == 8
