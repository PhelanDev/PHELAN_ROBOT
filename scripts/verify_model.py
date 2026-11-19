#!/usr/bin/env python3
"""Validate the canonical CyberDuck V2 MJCF and its derived URDF."""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from duck_v2.robot_spec import (  # noqa: E402
    BASE_HEIGHT_M,
    EXPECTED_JOINTS,
    MAX_SPEED_RAD_S,
    MAX_TORQUE_NM,
    MJCF_MODEL_NAME,
    MODEL_PATH,
    NUM_ACTIONS,
    THIGH_LINK_M,
    SHIN_LINK_M,
    TOTAL_MASS_KG,
    URDF_PATH,
)


def _numbers(value):
    return [float(part) for part in value.split()]


def _body_rpy(body):
    if body.get("euler"):
        return _numbers(body.get("euler"))
    if body.get("quat"):
        w, x, y, z = _numbers(body.get("quat"))
        return [
            math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        ]
    return [0.0, 0.0, 0.0]


def verify_joint_consistency(mjcf_path: str, urdf_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Check the canonical V2 joint contract against MJCF and generated URDF."""
    import mujoco

    if not os.path.exists(mjcf_path):
        raise FileNotFoundError(f"V2 MJCF not found: {mjcf_path}")
    if urdf_path is not None and not os.path.exists(urdf_path):
        raise FileNotFoundError(f"V2 URDF not found: {urdf_path}")
    if ET.parse(mjcf_path).getroot().get("model") != MJCF_MODEL_NAME:
        raise ValueError(f"Expected CyberDuck V2 MJCF identity {MJCF_MODEL_NAME!r}")

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    mjcf_joints = {}
    for name, expected in EXPECTED_JOINTS.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if joint_id < 0 or actuator_id < 0:
            mjcf_joints[name] = None
            continue
        mjcf_joints[name] = {
            "axis": [float(x) for x in model.jnt_axis[joint_id]],
            "range": [float(x) for x in model.jnt_range[joint_id]],
            "max_torque": float(max(abs(model.actuator_forcerange[actuator_id]))),
            "max_velocity": MAX_SPEED_RAD_S,
            "actuator_joint_id": int(model.actuator_trnid[actuator_id, 0]),
            "joint_id": int(joint_id),
        }

    urdf_joints = {}
    urdf_root_valid = True
    if urdf_path:
        root = ET.parse(urdf_path).getroot()
        floating_joint = root.find("joint[@name='trunk_freejoint']")
        floating_origin = floating_joint.find("origin") if floating_joint is not None else None
        urdf_root_valid = bool(
            floating_joint is not None
            and floating_joint.get("type") == "floating"
            and floating_joint.find("parent") is not None
            and floating_joint.find("parent").get("link") == "world"
            and floating_joint.find("child") is not None
            and floating_joint.find("child").get("link") == "trunk_base"
            and floating_origin is not None
            and math.isclose(_numbers(floating_origin.get("xyz"))[2], BASE_HEIGHT_M, abs_tol=1e-6)
        )
        mjcf_root = ET.parse(mjcf_path).getroot()
        mjcf_joint_origins = {}
        for body in mjcf_root.iter("body"):
            for joint in body.findall("joint"):
                mjcf_joint_origins[joint.get("name")] = {
                    "xyz": _numbers(body.get("pos", "0 0 0")),
                    "rpy": _body_rpy(body),
                }
        for joint in root.findall("joint"):
            name = joint.get("name")
            if name not in EXPECTED_JOINTS:
                continue
            origin = joint.find("origin")
            axis = joint.find("axis")
            limit = joint.find("limit")
            urdf_joints[name] = {
                "axis": _numbers(axis.get("xyz")) if axis is not None else [0.0, 0.0, 1.0],
                "range": [_numbers(limit.get("lower"))[0], _numbers(limit.get("upper"))[0]],
                "effort": float(limit.get("effort")),
                "velocity": float(limit.get("velocity")),
                "origin_xyz": _numbers(origin.get("xyz", "0 0 0")) if origin is not None else [0.0] * 3,
                "origin_rpy": _numbers(origin.get("rpy", "0 0 0")) if origin is not None else [0.0] * 3,
            }

    results = {}
    for name, expected in EXPECTED_JOINTS.items():
        mj = mjcf_joints[name]
        ur = urdf_joints.get(name) if urdf_path else None
        present = mj is not None and (urdf_path is None or ur is not None)
        match = bool(
            present
            and mj["actuator_joint_id"] == mj["joint_id"]
            and all(math.isclose(mj["axis"][i], expected["axis"][i], abs_tol=1e-6) for i in range(3))
            and all(math.isclose(mj["range"][i], expected[key], abs_tol=1e-6)
                    for i, key in enumerate(("lower", "upper")))
            and math.isclose(mj["max_torque"], MAX_TORQUE_NM, abs_tol=1e-6)
        )
        if ur is not None:
            expected_origin = mjcf_joint_origins[name]
            match = match and all(math.isclose(ur["axis"][i], expected["axis"][i], abs_tol=1e-6) for i in range(3))
            match = match and all(math.isclose(ur["range"][i], expected[key], abs_tol=1e-6)
                                  for i, key in enumerate(("lower", "upper")))
            match = match and math.isclose(ur["effort"], MAX_TORQUE_NM, abs_tol=1e-6)
            match = match and math.isclose(ur["velocity"], MAX_SPEED_RAD_S, abs_tol=1e-6)
            match = match and all(math.isclose(ur["origin_xyz"][i], expected_origin["xyz"][i], abs_tol=1e-6)
                                  for i in range(3))
            match = match and all(math.isclose(ur["origin_rpy"][i], expected_origin["rpy"][i], abs_tol=1e-6)
                                  for i in range(3))
        if urdf_path:
            match = match and urdf_root_valid
        results[name] = {
            "match": match,
            "mjcf_range": mj["range"] if mj else None,
            "urdf_range": ur["range"] if ur else None,
            "axis": list(expected["axis"]),
            "effort": ur["effort"] if ur else (mj["max_torque"] if mj else None),
            "velocity": ur["velocity"] if ur else (mj["max_velocity"] if mj else None),
        }
    return results


def verify_mass_and_inertials(mjcf_path: str, target_mass: Optional[float] = None, tolerance: float = 0.01):
    """Check compiled MJCF mass, positive inertias, and rigid-body inequalities."""
    import mujoco

    if not os.path.exists(mjcf_path):
        raise FileNotFoundError(f"V2 MJCF not found: {mjcf_path}")
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    target_mass = TOTAL_MASS_KG if target_mass is None else target_mass
    links = []
    all_valid = True
    for body_id in range(1, model.nbody):
        mass = float(model.body_mass[body_id])
        if mass <= 0.0:
            continue
        inertias = [float(value) for value in model.body_inertia[body_id]]
        positive = all(value > 0.0 for value in inertias)
        triangle = all(inertias[i] + inertias[j] >= inertias[k] - 1e-10
                       for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)))
        valid = positive and triangle
        all_valid = all_valid and valid
        links.append({"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                      "mass": mass, "ixx": inertias[0], "iyy": inertias[1], "izz": inertias[2],
                      "positive_diagonals": positive, "triangle_inequality": triangle, "valid": valid})
    total_mass = float(model.body_subtreemass[1])
    return {
        "links": links,
        "total_mass": total_mass,
        "target_mass": target_mass,
        "mass_in_range": abs(total_mass - target_mass) <= target_mass * tolerance,
        "all_inertias_valid": all_valid,
    }


def verify_urdf_physics(mjcf_path: str, urdf_path: str):
    """Compare compiled link frames, mass, CoM and full body-frame inertia tensors."""
    import mujoco
    import numpy as np

    source = mujoco.MjModel.from_xml_path(mjcf_path)
    derived = mujoco.MjModel.from_xml_path(urdf_path)
    matches = {}
    for source_id in range(1, source.nbody):
        name = mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_BODY, source_id)
        derived_id = mujoco.mj_name2id(derived, mujoco.mjtObj.mjOBJ_BODY, name)
        if derived_id < 0:
            matches[name] = False
            continue
        source_rotation = np.empty(9)
        derived_rotation = np.empty(9)
        mujoco.mju_quat2Mat(source_rotation, source.body_iquat[source_id])
        mujoco.mju_quat2Mat(derived_rotation, derived.body_iquat[derived_id])
        source_rotation = source_rotation.reshape(3, 3)
        derived_rotation = derived_rotation.reshape(3, 3)
        source_tensor = source_rotation @ np.diag(source.body_inertia[source_id]) @ source_rotation.T
        derived_tensor = derived_rotation @ np.diag(derived.body_inertia[derived_id]) @ derived_rotation.T
        matches[name] = bool(
            np.allclose(source.body_pos[source_id], derived.body_pos[derived_id], atol=1e-9, rtol=0.0)
            and np.allclose(source.body_quat[source_id], derived.body_quat[derived_id], atol=1e-9, rtol=0.0)
            and math.isclose(float(source.body_mass[source_id]), float(derived.body_mass[derived_id]), abs_tol=1e-9)
            and np.allclose(source.body_ipos[source_id], derived.body_ipos[derived_id], atol=1e-9, rtol=0.0)
            and np.allclose(source_tensor, derived_tensor, atol=1e-9, rtol=0.0)
        )
    floating_match = bool(source.nq == derived.nq == 15 and source.nv == derived.nv == 14
                          and np.allclose(source.qpos0, derived.qpos0, atol=1e-9, rtol=0.0))
    return {"body_matches": matches, "floating_match": floating_match,
            "valid": all(matches.values()) and floating_match}


def verify_center_of_mass(mjcf_path: str):
    """Check the computed V2 center of mass relative to the floating trunk frame."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    trunk = model.body("trunk_base").id
    com_world = [float(value) for value in data.subtree_com[trunk]]
    base_pos = [float(value) for value in data.xpos[trunk]]
    com_rel = [com_world[i] - base_pos[i] for i in range(3)]
    centered = abs(com_rel[1]) < 0.01 and abs(com_rel[0]) < 0.02
    low = -0.06 <= com_rel[2] <= 0.0
    return {
        "com_world": com_world,
        "base_pos": base_pos,
        "com_rel": com_rel,
        "com_low_and_centered": bool(centered and low),
        "used_mujoco": True,
    }


def verify_v2_geometry(mjcf_path: str):
    """Check V2 root/link dimensions and sole footprint in compiled coordinates."""
    import mujoco
    from duck_v2.robot_spec import SOLE_HALF_SIZE_M

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    expected_bodies = {
        "trunk_base": (0.0, 0.0, BASE_HEIGHT_M),
        "left_shin": (0.0, 0.0, -THIGH_LINK_M),
        "right_shin": (0.0, 0.0, -THIGH_LINK_M),
        "left_foot": (0.004, 0.0, -SHIN_LINK_M),
        "right_foot": (0.004, 0.0, -SHIN_LINK_M),
    }
    body_matches = {}
    for name, expected_pos in expected_bodies.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        body_pos = model.body_pos[body_id]
        body_matches[name] = body_id >= 0 and all(
            math.isclose(float(body_pos[i]), expected_pos[i], abs_tol=1e-6) for i in range(3)
        )
    sole_matches = True
    sole_bottoms = []
    for name in ("left_sole_geom", "right_sole_geom"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        sole_matches = sole_matches and all(
            math.isclose(float(model.geom_size[geom_id][i]), SOLE_HALF_SIZE_M[i], abs_tol=1e-6)
            for i in range(3)
        )
        sole_bottoms.append(float(data.geom_xpos[geom_id][2] - model.geom_size[geom_id][2]))
    return {
        "body_matches": body_matches,
        "sole_matches": sole_matches,
        "sole_bottoms": sole_bottoms,
        "valid": all(body_matches.values()) and sole_matches and all(abs(z) < 1e-5 for z in sole_bottoms),
    }


def run_simulation_check(mjcf_path: str, urdf_path: Optional[str] = None, num_steps: int = 100) -> bool:
    """Step the dynamic floating-base V2 model with the training PD controller."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    target = [0.0, 0.0, 0.0, -0.15, 0.15, 0.0, -0.15, 0.15]
    for step in range(num_steps):
        if step % 10 == 0:
            error = [target[i] - data.qpos[7 + i] for i in range(NUM_ACTIONS)]
            velocity = [-data.qvel[6 + i] for i in range(NUM_ACTIONS)]
            data.ctrl[:] = [max(-MAX_TORQUE_NM, min(MAX_TORQUE_NM, 13.5 * error[i] + 0.9 * velocity[i]))
                            for i in range(NUM_ACTIONS)]
        mujoco.mj_step(model, data)
        if not all(math.isfinite(float(value)) for value in data.qpos) or not all(
            math.isfinite(float(value)) for value in data.qvel
        ):
            return False
    return True


def print_summary(joint_results, mass_info, com_info, sim_ok, num_steps, geometry_info, urdf_info):
    print("=" * 80)
    print("CyberDuck V2 Physical Model Verification")
    print("=" * 80)
    print("\n1. V2 JOINT CONTRACT")
    for name, result in joint_results.items():
        status = "PASS" if result["match"] else "FAIL"
        print(f"{name:<20} | {result['mjcf_range']} | {result['axis']} | {status}")
    print("\n2. MASS AND INERTIA")
    print(f"Total Mass: {mass_info['total_mass']:.3f} kg (V2 model target: {mass_info['target_mass']:.3f} kg)")
    print(f"Positive, physically consistent body inertia: {mass_info['all_inertias_valid']}")
    print(f"Derived URDF frames, mass, CoM and inertia parity: {urdf_info['valid']}")
    print("\n3. V2 FRAMES AND FOOTPRINT")
    print(f"Base frame height: {geometry_info['body_matches']['trunk_base']} (expected {BASE_HEIGHT_M:.3f} m)")
    print(f"Thigh/shin frames: {geometry_info['body_matches']['left_shin']}, {geometry_info['body_matches']['left_foot']}")
    print(f"Foot collider footprint: {geometry_info['sole_matches']} (50 x 26 x 8 mm)")
    print(f"Sole bottom heights: {[round(z, 6) for z in geometry_info['sole_bottoms']]} m")
    print("\n4. CENTER OF MASS AND PD SIMULATION")
    print(f"CoM relative to trunk: {[round(value, 6) for value in com_info['com_rel']]} m")
    print(f"PD simulation: {'PASS' if sim_ok else 'FAIL'} ({num_steps} physics steps, finite state)")


def main() -> int:
    parser = argparse.ArgumentParser(description="CyberDuck V2 physical model verification")
    parser.add_argument("--mjcf", default=str(MODEL_PATH), help="Canonical V2 MJCF scene")
    parser.add_argument("--urdf", default=str(URDF_PATH), help="Derived V2 URDF for joint contract checks")
    parser.add_argument("--steps", type=int, default=100, help="Number of physics steps for PD smoke test")
    args = parser.parse_args()
    try:
        joint_results = verify_joint_consistency(args.mjcf, args.urdf)
        mass_info = verify_mass_and_inertials(args.mjcf)
        com_info = verify_center_of_mass(args.mjcf)
        geometry_info = verify_v2_geometry(args.mjcf)
        urdf_info = verify_urdf_physics(args.mjcf, args.urdf)
        sim_ok = run_simulation_check(args.mjcf, num_steps=args.steps)
    except Exception as exc:
        print(f"[ERROR] V2 verification aborted: {exc}", file=sys.stderr)
        return 1

    print_summary(joint_results, mass_info, com_info, sim_ok, args.steps, geometry_info, urdf_info)
    valid = (
        len(joint_results) == NUM_ACTIONS
        and all(result["match"] for result in joint_results.values())
        and mass_info["mass_in_range"]
        and mass_info["all_inertias_valid"]
        and com_info["com_low_and_centered"]
        and geometry_info["valid"]
        and urdf_info["valid"]
        and sim_ok
    )
    if valid:
        print("\nVerification SUCCESS: canonical CyberDuck V2 checks passed.")
        return 0
    print("\nVerification FAILED: one or more CyberDuck V2 checks failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
