import os
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from robot.robot_spec import MODEL_PATH, URDF_PATH, TOTAL_MASS_KG

def test_urdf_structure_and_actuators():
    urdf_path = str(URDF_PATH)
    assert os.path.exists(urdf_path), f"File {urdf_path} does not exist"

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    assert root.tag == "robot"
    assert root.attrib.get("name") == "cyberduck_v2"

    expected_joints = {
        "head_yaw": {"axis": [0, 0, 1], "lower": -1.57, "upper": 1.57},
        "head_pitch": {"axis": [0, 1, 0], "lower": -0.52, "upper": 0.78},
        "left_hip_roll": {"axis": [1, 0, 0], "lower": -0.44, "upper": 0.44},
        "left_hip_pitch": {"axis": [0, 1, 0], "lower": -0.87, "upper": 0.87},
        "left_knee_pitch": {"axis": [0, 1, 0], "lower": 0.0, "upper": 1.48},
        "right_hip_roll": {"axis": [1, 0, 0], "lower": -0.44, "upper": 0.44},
        "right_hip_pitch": {"axis": [0, 1, 0], "lower": -0.87, "upper": 0.87},
        "right_knee_pitch": {"axis": [0, 1, 0], "lower": 0.0, "upper": 1.48},
    }

    revolute_joints = [j for j in root.iter("joint") if j.attrib.get("type") == "revolute"]
    assert len(revolute_joints) == 8, f"Expected 8 revolute joints, found {len(revolute_joints)}"

    joints_by_name = {j.attrib["name"]: j for j in revolute_joints}
    assert set(joints_by_name.keys()) == set(expected_joints.keys())

    for name, exp in expected_joints.items():
        j = joints_by_name[name]
        limit = j.find("limit")
        assert limit is not None, f"Joint {name} missing <limit>"
        assert float(limit.attrib["effort"]) == 1.0
        assert float(limit.attrib["velocity"]) == 6.0
        assert pytest.approx(float(limit.attrib["lower"]), abs=0.01) == exp["lower"]
        assert pytest.approx(float(limit.attrib["upper"]), abs=0.01) == exp["upper"]

        dynamics = j.find("dynamics")
        assert dynamics is not None, f"Joint {name} missing <dynamics>"
        assert pytest.approx(float(dynamics.attrib.get("damping", 0.0)), abs=1e-6) == 0.25
        assert pytest.approx(float(dynamics.attrib.get("friction", 0.0)), abs=1e-6) == 0.008

        axis = j.find("axis")
        assert axis is not None, f"Joint {name} missing <axis>"
        axis_xyz = [float(x) for x in axis.attrib["xyz"].split()]
        assert axis_xyz == exp["axis"]

def test_urdf_links_and_inertials():
    urdf_path = str(URDF_PATH)
    assert os.path.exists(urdf_path), f"File {urdf_path} does not exist"

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    expected_links = [
        "world", "trunk_base", "neck", "head",
        "left_hip", "left_thigh", "left_shin", "left_foot",
        "right_hip", "right_thigh", "right_shin", "right_foot",
    ]

    links_by_name = {l.attrib["name"]: l for l in root.iter("link")}
    for exp_link in expected_links:
        assert exp_link in links_by_name, f"Missing link {exp_link}"

    total_mass = 0.0
    for name, link in links_by_name.items():
        if name == "world":
            assert link.find("inertial") is None
            continue
        inertial = link.find("inertial")
        assert inertial is not None, f"Link {name} missing <inertial>"
        mass_elem = inertial.find("mass")
        assert mass_elem is not None, f"Link {name} missing <mass>"
        m = float(mass_elem.attrib["value"])
        assert m > 0, f"Link {name} mass must be positive"
        total_mass += m

        inertia = inertial.find("inertia")
        assert inertia is not None, f"Link {name} missing <inertia>"
        ixx = float(inertia.attrib["ixx"])
        iyy = float(inertia.attrib["iyy"])
        izz = float(inertia.attrib["izz"])
        assert ixx > 0 and iyy > 0 and izz > 0, f"Link {name} principal inertias must be positive"
        # Triangle inequality for valid rigid body
        assert ixx + iyy >= izz - 1e-9, f"Link {name} violates triangle inequality (ixx+iyy < izz)"
        assert ixx + izz >= iyy - 1e-9, f"Link {name} violates triangle inequality (ixx+izz < iyy)"
        assert iyy + izz >= ixx - 1e-9, f"Link {name} violates triangle inequality (iyy+izz < ixx)"

        # Check visual and collision
        visual = link.find("visual")
        collision = link.find("collision")
        assert visual is not None, f"Link {name} missing <visual>"
        assert collision is not None, f"Link {name} missing <collision>"

    assert total_mass == pytest.approx(TOTAL_MASS_KG, abs=1e-6)

    # Verify the canonical V2 hip-knee and knee-foot link frames.
    joints_by_name = {j.attrib["name"]: j for j in root.findall("joint")}
    knee_origin = [float(x) for x in joints_by_name["left_knee_pitch"].find("origin").attrib["xyz"].split()]
    assert knee_origin == [0.0, 0.0, -0.060]
    foot_origin = [float(x) for x in joints_by_name["left_foot_fixed"].find("origin").attrib["xyz"].split()]
    assert foot_origin == [0.004, 0.0, -0.054]
    assert joints_by_name["trunk_freejoint"].get("type") == "floating"

def test_urdf_mujoco_compilation():
    urdf_path = str(URDF_PATH)
    assert os.path.exists(urdf_path), f"File {urdf_path} does not exist"

    try:
        import mujoco
    except ImportError:
        pytest.skip("MuJoCo python package not available")

    model = mujoco.MjModel.from_xml_path(urdf_path)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    assert model.njnt == 9
    assert model.nq == 15 and model.nv == 14


def test_urdf_head_link_com_synchronization():
    """Verify V2 head CoM and all body tensors match the canonical MJCF."""
    from tools.robot.verify_model import verify_urdf_physics
    root = ET.parse(URDF_PATH).getroot()
    head = root.find("link[@name='head']")
    xyz = [float(value) for value in head.find("inertial/origin").get("xyz").split()]
    np.testing.assert_allclose(xyz, [0.025233, 0.000626, 0.032953], atol=1e-9)
    assert verify_urdf_physics(str(MODEL_PATH), str(URDF_PATH))["valid"]
