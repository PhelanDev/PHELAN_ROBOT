"""Canonical V2 robot identity and physical specifications."""

from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(__file__).resolve().with_name("cyberduck_v2.xml")
MODEL_NAME = "cyberduck_v2.xml"
MJCF_MODEL_NAME = "cyberduck_v2_pro"
ROBOT_ONLY_MJCF_PATH = Path(__file__).resolve().with_name("cyberduck_v2_robot.xml")
URDF_PATH = Path(__file__).resolve().with_name("cyberduck_v2.urdf")
CHECKPOINT_PREFIX = "cyberduck_v2"
DEFAULT_ONNX_PATH = REPO_ROOT / "models" / "cyberduck_v2_policy.onnx"
SERVO_MODEL = "MG996R"
SERVO_STALL_TORQUE_NM = 1.0
SERVO_MAX_SPEED_RAD_S = 6.0

TOTAL_MASS_KG = 0.811
BASE_HEIGHT_M = 0.128
THIGH_LINK_M = 0.044
SHIN_LINK_M = 0.036
CONTROL_DT_S = 0.02
PHYSICS_DT_S = 0.002
CONTROL_HZ = 50
NUM_OBSERVATIONS = 42
NUM_ACTIONS = 8
KP_NM_PER_RAD = 13.5
KD_NM_S_PER_RAD = 0.9
MAX_TORQUE_NM = 1.0
MAX_SPEED_RAD_S = 6.0

JOINT_ORDER = (
    "head_yaw",
    "head_pitch",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee_pitch",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee_pitch",
)

EXPECTED_JOINTS = {
    "head_yaw": {"axis": (0.0, 0.0, 1.0), "lower": -1.57, "upper": 1.57},
    "head_pitch": {"axis": (0.0, 1.0, 0.0), "lower": -0.52, "upper": 0.78},
    "left_hip_roll": {"axis": (1.0, 0.0, 0.0), "lower": -0.44, "upper": 0.44},
    "left_hip_pitch": {"axis": (0.0, 1.0, 0.0), "lower": -0.87, "upper": 0.87},
    "left_knee_pitch": {"axis": (0.0, 1.0, 0.0), "lower": 0.0, "upper": 1.48},
    "right_hip_roll": {"axis": (1.0, 0.0, 0.0), "lower": -0.44, "upper": 0.44},
    "right_hip_pitch": {"axis": (0.0, 1.0, 0.0), "lower": -0.87, "upper": 0.87},
    "right_knee_pitch": {"axis": (0.0, 1.0, 0.0), "lower": 0.0, "upper": 1.48},
}

SOLE_HALF_SIZE_M = (0.025, 0.013, 0.004)
TOUCH_SITE_HALF_SIZE_M = (0.027, 0.019, 0.005)
