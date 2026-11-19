"""Export a URDF view of the canonical V2 MJCF articulation."""

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot.robot_spec import (  # noqa: E402
    BASE_HEIGHT_M,
    EXPECTED_JOINTS,
    ROBOT_ONLY_MJCF_PATH,
    URDF_PATH,
)
from duck_v2.scripts.build_robot_only_mjcf import build_robot_only_mjcf  # noqa: E402


def _values(element, attribute, default):
    return [float(item) for item in element.get(attribute, default).split()]


def _rpy_from_quaternion(quaternion):
    w, x, y, z = quaternion
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return roll, pitch, math.atan2(siny_cosp, cosy_cosp)


def _rpy_from_mjcf(element):
    if element.get("quat"):
        return _rpy_from_quaternion(_values(element, "quat", "1 0 0 0"))
    if element.get("euler"):
        return tuple(_values(element, "euler", "0 0 0"))
    return (0.0, 0.0, 0.0)


def _format(values):
    return " ".join(f"{value:.12g}" for value in values)


def _rotation_from_z(vector):
    length = math.sqrt(sum(value * value for value in vector))
    z_axis = [value / length for value in vector]
    if abs(z_axis[2]) > 0.999999:
        x_axis = [1.0, 0.0, 0.0]
    else:
        x_axis = [z_axis[1], -z_axis[0], 0.0]
        norm = math.sqrt(sum(value * value for value in x_axis))
        x_axis = [value / norm for value in x_axis]
    y_axis = [
        z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
        z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
        z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
    ]
    matrix = [
        [x_axis[0], y_axis[0], z_axis[0]],
        [x_axis[1], y_axis[1], z_axis[1]],
        [x_axis[2], y_axis[2], z_axis[2]],
    ]
    pitch = math.asin(max(-1.0, min(1.0, -matrix[2][0])))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = math.atan2(-matrix[1][2], matrix[1][1])
        yaw = 0.0
    return roll, pitch, yaw


def _geometry(parent, geom, meshes, *, collision):
    mesh_class = geom.get("class", "")
    is_visual = mesh_class == "visual"
    if (collision and is_visual) or (not collision and not is_visual):
        return False

    container = ET.SubElement(parent, "collision" if collision else "visual")
    xyz = _values(geom, "pos", "0 0 0")
    rpy = _rpy_from_mjcf(geom)
    ET.SubElement(container, "origin", xyz=_format(xyz), rpy=_format(rpy))
    shape = ET.SubElement(container, "geometry")
    kind = geom.get("type", "sphere")

    if kind == "mesh":
        mesh = meshes[geom.get("mesh")]
        ET.SubElement(shape, "mesh", filename=f"meshes/{mesh.get('file')}")
    elif kind == "box":
        half = _values(geom, "size", "0.5 0.5 0.5")
        ET.SubElement(shape, "box", size=_format([2.0 * value for value in half]))
    elif kind == "sphere":
        radius = _values(geom, "size", "0.5")[0]
        ET.SubElement(shape, "sphere", radius=f"{radius:.12g}")
    elif kind == "cylinder":
        radius, half_length = _values(geom, "size", "0.5 0.5")[:2]
        ET.SubElement(shape, "cylinder", radius=f"{radius:.12g}", length=f"{2.0 * half_length:.12g}")
    elif kind == "capsule":
        start_end = _values(geom, "fromto", "0 0 0 0 0 0")
        start, end = start_end[:3], start_end[3:]
        delta = [end[i] - start[i] for i in range(3)]
        radius = _values(geom, "size", "0.01")[0]
        midpoint = [(start[i] + end[i]) * 0.5 for i in range(3)]
        direction_rpy = _rotation_from_z(delta)
        length = math.sqrt(sum(value * value for value in delta))
        parent.remove(container)
        for xyz_offset, origin_rpy, shape_kind in (
            (midpoint, direction_rpy, "cylinder"),
            (start, (0.0, 0.0, 0.0), "sphere"),
            (end, (0.0, 0.0, 0.0), "sphere"),
        ):
            part = ET.SubElement(parent, "collision" if collision else "visual")
            ET.SubElement(part, "origin", xyz=_format(xyz_offset), rpy=_format(origin_rpy))
            part_geometry = ET.SubElement(part, "geometry")
            if shape_kind == "cylinder":
                ET.SubElement(part_geometry, "cylinder", radius=f"{radius:.12g}", length=f"{length:.12g}")
            else:
                ET.SubElement(part_geometry, "sphere", radius=f"{radius:.12g}")
        return True
    else:
        raise ValueError(f"Unsupported MJCF geometry type: {kind}")

    material_name = geom.get("material")
    if not collision and material_name:
        ET.SubElement(container, "material", name=material_name)
    return True


def _add_link(robot, body, meshes):
    name = body.get("name")
    link = ET.SubElement(robot, "link", name=name)
    inertial = body.find("inertial")
    if inertial is None:
        raise ValueError(f"V2 body {name!r} has no explicit inertial")
    inertial_node = ET.SubElement(link, "inertial")
    ET.SubElement(
        inertial_node,
        "origin",
        xyz=_format(_values(inertial, "pos", "0 0 0")),
        rpy=_format(_rpy_from_mjcf(inertial)),
    )
    ET.SubElement(inertial_node, "mass", value=inertial.get("mass"))
    inertia = _values(inertial, "diaginertia", "0.00001 0.00001 0.00001")
    ET.SubElement(
        inertial_node,
        "inertia",
        ixx=f"{inertia[0]:.12g}",
        iyy=f"{inertia[1]:.12g}",
        izz=f"{inertia[2]:.12g}",
        ixy="0",
        ixz="0",
        iyz="0",
    )
    for geom in body.findall("geom"):
        _geometry(link, geom, meshes, collision=False)
        _geometry(link, geom, meshes, collision=True)
    return link


def _add_body(robot, parent_body, parent_link, meshes, materials):
    for body in parent_body.findall("body"):
        link = _add_link(robot, body, meshes)
        joint = body.find("joint")
        if joint is not None:
            if len(body.findall("joint")) != 1:
                raise ValueError(f"URDF V2 exporter expects one joint per body: {body.get('name')}")
            joint_name = joint.get("name")
            spec = EXPECTED_JOINTS.get(joint_name)
            if spec is None:
                raise ValueError(f"Unexpected V2 joint: {joint_name}")
            joint_type = "revolute"
        elif body.find("freejoint") is not None:
            joint_name = None
            joint_type = None
        else:
            joint_name = f"{body.get('name')}_fixed"
            joint_type = "fixed"

        if joint_type:
            joint_node = ET.SubElement(robot, "joint", name=joint_name, type=joint_type)
            ET.SubElement(joint_node, "parent", link=parent_link)
            ET.SubElement(joint_node, "child", link=body.get("name"))
            ET.SubElement(
                joint_node,
                "origin",
                xyz=_format(_values(body, "pos", "0 0 0")),
                rpy=_format(_rpy_from_mjcf(body)),
            )
            if joint_type == "revolute":
                ET.SubElement(joint_node, "axis", xyz=_format(_values(joint, "axis", "0 0 1")))
                ET.SubElement(
                    joint_node,
                    "limit",
                    lower=f"{spec['lower']:.12g}",
                    upper=f"{spec['upper']:.12g}",
                    effort="1",
                    velocity="6",
                )
                ET.SubElement(joint_node, "dynamics", damping="0.25", friction="0.008")

        _add_body(robot, body, body.get("name"), meshes, materials)


def export_v2_urdf(output: Path = URDF_PATH) -> Path:
    mjcf_path = build_robot_only_mjcf()
    root = ET.parse(mjcf_path).getroot()
    asset = root.find("asset")
    meshes = {element.get("name"): element for element in asset.findall("mesh")}
    materials = {
        element.get("name"): element
        for element in asset.findall("material")
        if element.get("rgba")
    }

    robot = ET.Element("robot", name="cyberduck_v2")
    robot.append(ET.Comment("Generated from cyberduck_v2.xml; edit the MJCF source and regenerate."))
    mujoco_settings = ET.SubElement(robot, "mujoco")
    ET.SubElement(mujoco_settings, "compiler", fusestatic="false", discardvisual="false")
    ET.SubElement(robot, "link", name="world")
    freejoint = ET.SubElement(robot, "joint", name="trunk_freejoint", type="floating")
    ET.SubElement(freejoint, "parent", link="world")
    ET.SubElement(freejoint, "child", link="trunk_base")
    ET.SubElement(freejoint, "origin", xyz=f"0 0 {BASE_HEIGHT_M:.12g}", rpy="0 0 0")
    for name, material in materials.items():
        ET.SubElement(robot, "material", name=name).append(
            ET.Element("color", rgba=material.get("rgba"))
        )

    worldbody = root.find("worldbody")
    root_body = next(iter(worldbody.findall("body")))
    _add_link(robot, root_body, meshes)
    _add_body(robot, root_body, root_body.get("name"), meshes, materials)

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(output, encoding="utf-8", xml_declaration=True)
    return output


if __name__ == "__main__":
    print(export_v2_urdf())
