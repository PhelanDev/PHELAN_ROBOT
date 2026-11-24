"""Reinforcement Learning Environment Configuration for MicroDuck 8-DOF (RPi 5 Stack).

Thin parameterized wrapper delegating common training environment logic to sim.env.env_config.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from robot.robot_spec import NUM_OBSERVATIONS, TOTAL_MASS_KG

from sim.env.env_config import (
    MicroDuck8DoFConfig as BaseConfig,
    compute_observations,
    scale_actions,
    clip_action_rates,
    quat_rotate_inverse,
    compute_rewards,
)


@dataclass
class MicroDuck8DoFConfig(BaseConfig):
    """Configuration dataclass for MicroDuck 8-DOF on Raspberry Pi 5 Stack."""

    STACK_NAME: str = "rpi5_stack"
    TARGET_MASS: float = TOTAL_MASS_KG  # Shared V2 model; runtime profile does not change physics.
    POLICY_DTYPE: str = "float32"

    OBS_DIM: int = NUM_OBSERVATIONS
    ACTION_DIM: int = 8
    LIDAR_RAYS: int = 0
    LIDAR_MIN_RANGE: float = 0.05
    LIDAR_MAX_RANGE: float = 2.0
    OBSTACLE_THRESHOLD: float = 0.35  # meters

    NOMINAL_JOINT_POS: Dict[str, float] = field(
        default_factory=lambda: {
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": -0.15,
            "left_knee_pitch": 0.15,
            "right_hip_roll": 0.0,
            "right_hip_pitch": -0.15,
            "right_knee_pitch": 0.15,
        }
    )


# Module-level cached configuration for simulation steps
_DEFAULT_CFG = MicroDuck8DoFConfig()

DEFAULT_NOMINAL_JOINT_POS: Dict[str, float] = _DEFAULT_CFG.NOMINAL_JOINT_POS
DEFAULT_NOMINAL_QPOS: np.ndarray = _DEFAULT_CFG.nominal_qpos
