import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, TESTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_test_helpers import (
    EXPECTED_JOINTS,
    EXPECTED_LIDAR_RAYS,
    assert_mjcf_structure_and_joints,
    assert_mjcf_perception_suite,
    assert_mujoco_compilation_and_mass,
    assert_urdf_structure_and_joint_limits,
    assert_urdf_mujoco_compilation,
    assert_slalom_pillars_and_stairs,
    assert_sole_clearance_and_target_mass,
)

from robot.robot_spec import MODEL_PATH as MJCF_PATH, URDF_PATH, MJCF_MODEL_NAME, TOTAL_MASS_KG


def test_rpi5_mjcf_structure_and_joints():
    assert_mjcf_structure_and_joints(MJCF_PATH, MJCF_MODEL_NAME)


def test_rpi5_mjcf_perception_suite():
    assert_mjcf_perception_suite(MJCF_PATH)


def test_rpi5_mujoco_compilation_and_mass_budget():
    assert_mujoco_compilation_and_mass(
        MJCF_PATH,
        min_mass=TOTAL_MASS_KG - 1e-6,
        max_mass=TOTAL_MASS_KG + 1e-6,
        base_mass_min=0.578958 - 1e-6,
        base_mass_max=0.578958 + 1e-6,
        base_ipos_z_min=-0.015274 - 1e-6,
        base_ipos_z_max=-0.015274 + 1e-6,
        n_steps=50,
    )


def test_rpi5_urdf_structure_and_inertials():
    assert_urdf_structure_and_joint_limits(
        URDF_PATH,
        robot_name="cyberduck_v2",
        min_total_mass=TOTAL_MASS_KG - 1e-6,
        max_total_mass=TOTAL_MASS_KG + 1e-6,
        min_base_mass=0.578958 - 1e-6,
        max_base_mass=0.578958 + 1e-6,
    )


def test_rpi5_urdf_mujoco_compilation():
    assert_urdf_mujoco_compilation(URDF_PATH)


def test_rpi5_mjcf_obstacles():
    assert_slalom_pillars_and_stairs(MJCF_PATH)


def test_rpi5_sole_clearance_and_target_mass():
    from hardware.rpi5.training.env_config import _DEFAULT_CFG
    assert_sole_clearance_and_target_mass(MJCF_PATH, _DEFAULT_CFG.TARGET_MASS)
