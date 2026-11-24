"""Batched Parallel Simulation Runner and FPS Benchmark for CyberDuck V2.

Executes vectorized MuJoCo simulation instances sharing a single mjModel across
multiple environments using multi-threaded stepping or batched loops.
Integrates with sim.env.env_config for observation packing, action scaling,
rate limiting, and reward computation.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple, Union

import mujoco
import numpy as np

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim.env.env_config import (
    MicroDuck8DoFConfig,
    _DEFAULT_CFG,
    DEFAULT_NOMINAL_QPOS,
    EmotionState,
    EMOTION_TARGET_HEAD_ARRAY,
    NUM_EMOTIONS,
    LocomotionMode,
    SIT_NOMINAL_QPOS,
    TARGET_SIT_HEIGHT,
    LIE_NOMINAL_QPOS,
    TARGET_LIE_HEIGHT,
    get_emotion_one_hot,
    clip_action_rates,
    compute_battery_sag_torque_limit,
    compute_observations,
    compute_rewards,
    compute_gait_phase,
    compute_slewed_posture_target,
    quat_rotate_inverse,
    quat_to_yaw,
    scale_actions,
    small_angle_rotation_matrix,
)
from sim.env.terrain_curriculum import apply_stage, apply_stair_stage, STAIR_STAGES, randomize_pillars
from sim.behaviors.active_head_scanner import ActiveHeadScanner
from sim.behaviors.self_righting import FallClassifier, FallPredictor, SelfRightingController
from sim.behaviors.nav_planner import LidarWaypointPlanner, NavPlannerConfig
from robot.robot_spec import MODEL_PATH, MJCF_MODEL_NAME, REPO_ROOT, SRC_ROOT


def resolve_v2_model_path(model_path: Union[str, Path] = MODEL_PATH) -> Path:
    """Resolve a V2 MJCF, rejecting retired robot identities before simulation."""
    path = Path(str(model_path).replace("\\", "/")).expanduser()
    if path.name.startswith("microduck"):
        raise ValueError("Retired V1 model is unsupported; use src/robot/cyberduck_v2.xml")
    if not path.is_absolute():
        if path.is_file():
            pass
        elif (REPO_ROOT / path).is_file():
            path = REPO_ROOT / path
        else:
            path = SRC_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V2 MuJoCo model not found at {path}")
    if ET.parse(path).getroot().get("model") != MJCF_MODEL_NAME:
        raise ValueError(f"Only CyberDuck V2 model identity {MJCF_MODEL_NAME!r} is supported: {path}")
    return path


def has_obstacle_contact(data: Any, obstacle_gid_set: set) -> bool:
    """Check if any active contact involves a geom in obstacle_gid_set."""
    if not obstacle_gid_set or getattr(data, "ncon", 0) == 0:
        return False
    for c_idx in range(data.ncon):
        c = data.contact[c_idx]
        if c.geom1 in obstacle_gid_set or c.geom2 in obstacle_gid_set:
            return True
    return False


LIDAR_SITE_NAMES: Tuple[str, ...] = (
    "site_lidar_m30",
    "site_lidar_m15",
    "site_lidar_0",
    "site_lidar_p15",
    "site_lidar_p30",
)


def extract_filtered_lidar_ranges(
    model: Optional[mujoco.MjModel],
    data: Any,
    obs_dim: int = 42,
    lidar_max: float = 2.0,
    site_ids: Optional[List[int]] = None,
) -> Optional[np.ndarray]:
    """Extract 5-ray LiDAR readings in meters with floor-hit filtering (`z_hit <= 0.012`)."""
    if len(data.sensordata) >= 33:
        raw_lidar = np.array(data.sensordata[28:33], dtype=np.float32)
        lidar_ranges = np.where(raw_lidar < 0.0, lidar_max, raw_lidar)
        if site_ids is None and model is not None:
            site_ids = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
                for sname in LIDAR_SITE_NAMES
            ]
        if site_ids and hasattr(data, "site_xpos") and hasattr(data, "site_xmat"):
            for k, sid in enumerate(site_ids):
                if sid != -1 and 0.0 <= raw_lidar[k] < lidar_max:
                    z_hit = float(data.site_xpos[sid, 2]) + float(raw_lidar[k]) * float(data.site_xmat[sid, 8])
                    if z_hit <= 0.012:
                        lidar_ranges[k] = lidar_max
        return lidar_ranges
    if obs_dim == 42:
        return np.full(5, lidar_max, dtype=np.float32)
    return None


def extract_filtered_head_tof_range(
    model: Optional[mujoco.MjModel],
    data: Any,
    lidar_max: float = 2.0,
    site_id: int = -1,
    sensor_adr: int = 28,
) -> float:
    """Extract single brow head_tof rangefinder distance (m) with floor-hit filter."""
    if sensor_adr < 0 or len(data.sensordata) <= sensor_adr:
        return float(lidar_max)
    raw = float(data.sensordata[sensor_adr])
    if raw < 0.0 or raw >= lidar_max:
        return float(lidar_max)
    if site_id != -1 and hasattr(data, "site_xpos") and hasattr(data, "site_xmat"):
        z_hit = float(data.site_xpos[site_id, 2]) + raw * float(data.site_xmat[site_id, 8])
        if z_hit <= 0.012:
            return float(lidar_max)
    return float(raw)


_SOLE_SAMPLE_SIGNS: np.ndarray = np.array(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def update_gait_hysteresis_state(
    active_walk: np.ndarray,
    body_vx: np.ndarray,
    accepted_dx: np.ndarray,
    curr_contact: np.ndarray,
    sole_clearance: np.ndarray,
    standstill_time: np.ndarray,
    progress_streak: np.ndarray,
    double_support_time: np.ndarray,
    swing_clear_time: np.ndarray,
    swing_clear_steps: np.ndarray,
    dt: float,
    clearance_thresh: float = 0.008,
    min_swing_time: float = 0.04,
    min_swing_steps: int = 2,
) -> np.ndarray:
    """Update walking standstill, swing-gate, and double-support hysteresis timers in place.

    Returns per-foot boolean swing_valid mask of shape (N, 2).
    """
    n_envs = active_walk.shape[0]
    dt_f = float(dt)

    inactive = ~active_walk
    if np.any(inactive):
        standstill_time[inactive] = 0.0
        progress_streak[inactive] = 0
        double_support_time[inactive] = 0.0
        swing_clear_time[inactive] = 0.0
        swing_clear_steps[inactive] = 0

    qualifying_move = active_walk & (body_vx >= 0.05) & (accepted_dx > 0.0)
    slow_standstill = active_walk & (body_vx < 0.04)
    hold_band = active_walk & (~qualifying_move) & (~slow_standstill)

    if np.any(qualifying_move):
        progress_streak[qualifying_move] += 1
        reset_st = qualifying_move & (progress_streak >= 2)
        standstill_time[reset_st] = 0.0
    if np.any(slow_standstill):
        standstill_time[slow_standstill] += dt_f
        progress_streak[slow_standstill] = 0
    if np.any(hold_band):
        progress_streak[hold_band] = 0

    c_left = curr_contact[:, 0]
    c_right = curr_contact[:, 1]
    left_cand = active_walk & (~c_left) & c_right & (sole_clearance[:, 0] >= clearance_thresh)
    right_cand = active_walk & (~c_right) & c_left & (sole_clearance[:, 1] >= clearance_thresh)

    if np.any(left_cand):
        swing_clear_steps[left_cand, 0] += 1
        swing_clear_time[left_cand, 0] += dt_f
    swing_clear_steps[~left_cand, 0] = 0
    swing_clear_time[~left_cand, 0] = 0.0

    if np.any(right_cand):
        swing_clear_steps[right_cand, 1] += 1
        swing_clear_time[right_cand, 1] += dt_f
    swing_clear_steps[~right_cand, 1] = 0
    swing_clear_time[~right_cand, 1] = 0.0

    swing_valid = np.zeros((n_envs, 2), dtype=bool)
    swing_valid[:, 0] = (
        left_cand
        & (swing_clear_steps[:, 0] >= min_swing_steps)
        & (swing_clear_time[:, 0] >= min_swing_time - 1e-7)
    )
    swing_valid[:, 1] = (
        right_cand
        & (swing_clear_steps[:, 1] >= min_swing_steps)
        & (swing_clear_time[:, 1] >= min_swing_time - 1e-7)
    )

    both_contact = active_walk & c_left & c_right
    any_swing_valid = active_walk & (swing_valid[:, 0] | swing_valid[:, 1])
    if np.any(both_contact):
        fwd_moving = np.abs(body_vx) >= 0.05
        dt_accum = np.where(fwd_moving, 0.5 * dt_f, dt_f)
        double_support_time[both_contact] += dt_accum[both_contact]
    if np.any(any_swing_valid):
        double_support_time[any_swing_valid] = 0.0

    return swing_valid


class ParallelSimRunner:
    """Vectorized / batched simulation runner for CyberDuck V2."""

    def __init__(
        self,
        model_path: Union[str, Path] = MODEL_PATH,
        num_envs: int = 16,
        cfg: Optional[MicroDuck8DoFConfig] = None,
        decimation: int = 10,
        num_threads: Optional[int] = None,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
        max_episode_steps: int = 500,
        stage: int = 0,
        terrain_mode: str = "hurdles",
        stand_still: bool = False,
        include_emotions: bool = False,
        enable_auto_recovery: bool = False,
        allow_teleport_assist: bool = False,
        walk_min_z: float = 0.095,
        friction_range: Tuple[float, float] = (1.1, 1.6),
        embed_phase_clock: Optional[bool] = None,
        randomize_pillars: bool = False,
        pillar_density: float = 1.0,
        pillar_jitter_xy: float = 0.06,
        pillar_jitter: Optional[float] = None,
        use_nav_planner: bool = False,
        nav_goal_xy: Tuple[float, float] = (2.0, 0.0),
        nav_planner_cfg: Optional[NavPlannerConfig] = None,
        max_action_delay_steps: int = 0,
        enable_limp_fall: bool = False,
        limp_kp_scale: float = 0.25,
        zero_cmd_prob: float = 0.0,
        turn_in_place_prob: float = 0.0,
        posture_ramp_s: float = 0.0,
        encode_posture_commands: bool = False,
        multiplicative_posture_reward: bool = False,
        enable_head_ema_penalty: bool = False,
        action_over_limit_threshold: Optional[float] = None,
        enable_battery_sag: bool = False,
        encoder_bias_rad: float = 0.0,
        imu_mount_max_rad: float = 0.0,
        delay_joint_vel_obs: bool = False,
    ):
        self.model_path = resolve_v2_model_path(model_path)
        if int(max_action_delay_steps) < 0:
            raise ValueError("max_action_delay_steps must be >= 0")
        if not (0.0 <= float(limp_kp_scale) <= 1.0):
            raise ValueError("limp_kp_scale must be in [0.0, 1.0]")
        if float(posture_ramp_s) < 0.0:
            raise ValueError("posture_ramp_s must be >= 0")
        if float(encoder_bias_rad) < 0.0 or float(imu_mount_max_rad) < 0.0:
            raise ValueError("encoder_bias_rad and imu_mount_max_rad must be >= 0")
        self.enable_limp_fall = bool(enable_limp_fall)
        self.limp_kp_scale = float(limp_kp_scale)
        self.posture_ramp_s = float(posture_ramp_s)
        self.encode_posture_commands = bool(encode_posture_commands)
        self.multiplicative_posture_reward = bool(multiplicative_posture_reward)
        self.enable_head_ema_penalty = bool(enable_head_ema_penalty)
        self.action_over_limit_threshold = (
            float(action_over_limit_threshold) if action_over_limit_threshold is not None else None
        )
        self.enable_battery_sag = bool(enable_battery_sag)
        self.encoder_bias_rad = float(encoder_bias_rad)
        self.imu_mount_max_rad = float(imu_mount_max_rad)
        self.delay_joint_vel_obs = bool(delay_joint_vel_obs)
        self.set_command_bucket_probs(zero_cmd_prob, turn_in_place_prob)

        self.terrain_mode = str(terrain_mode)
        self.randomize_pillars_enabled = bool(randomize_pillars)
        self.pillar_density = float(pillar_density)
        self.pillar_jitter_xy = float(pillar_jitter if pillar_jitter is not None else pillar_jitter_xy)
        self.include_emotions = bool(include_emotions)
        self.walk_min_z = float(walk_min_z)
        self.friction_range = (float(friction_range[0]), float(friction_range[1]))
        self.cfg = cfg if cfg is not None else _DEFAULT_CFG
        for field in ("TORSO_MASS_RANDOMIZATION_KG", "TORSO_COM_RANDOMIZATION_M"):
            amplitude = float(getattr(self.cfg, field))
            if not np.isfinite(amplitude) or amplitude < 0:
                raise ValueError(f"Invalid dynamics randomization amplitude: {field}")
        if embed_phase_clock is not None:
            self.embed_phase_clock = bool(embed_phase_clock)
        else:
            self.embed_phase_clock = True
        self.num_envs = int(num_envs)
        self.decimation = max(1, int(decimation))
        self.kp = float(kp) if kp is not None else float(self.cfg.KP)
        self.kd = float(kd) if kd is not None else float(self.cfg.KD)

        # Load shared MuJoCo model and instantiate per-env data
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.datas = [mujoco.MjData(self.model) for _ in range(self.num_envs)]

        # Time tracking
        self.sim_dt = float(self.model.opt.timestep)
        self.control_dt = self.sim_dt * self.decimation

        # Threading configuration
        if num_threads is None:
            # Auto choose reasonable thread count
            cpu_count = os.cpu_count() or 4
            self.num_threads = min(cpu_count, max(1, self.num_envs // 4))
        else:
            self.num_threads = max(1, int(num_threads))

        self.executor = (
            ThreadPoolExecutor(max_workers=self.num_threads)
            if self.num_threads > 1 and self.num_envs > 1
            else None
        )

        # Tracking state buffers (canonical V2 contract: 37D base, 47D with emotions, +11 critic)
        self.base_obs_dim = int(self.cfg.OBS_DIM)
        if self.include_emotions:
            self.obs_dim = self.base_obs_dim + NUM_EMOTIONS
        else:
            self.obs_dim = self.base_obs_dim
        self.privileged_obs_dim = self.obs_dim + 11
        self.action_dim = self.cfg.ACTION_DIM

        self.last_actions = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        self.last_targets = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        self.max_action_delay_steps = int(max_action_delay_steps)
        self.env_action_delay = np.zeros(self.num_envs, dtype=np.int32)
        self.action_delay_buffer = np.zeros(
            (self.num_envs, self.max_action_delay_steps + 1, self.action_dim),
            dtype=np.float32,
        )
        self.commands = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.stand_still = bool(stand_still)
        if not self.stand_still:
            # Default forward velocity command [vx=0.5, vy=0.0, wz=0.0]
            self.commands[:, 0] = 0.5

        self.max_episode_steps = int(max_episode_steps)
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.completed_ep_lens: deque = deque(maxlen=100)
        self.completed_distances_x: deque = deque(maxlen=100)
        self.completed_distances_y: deque = deque(maxlen=100)
        self.feet_air_time = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.last_feet_contact = np.ones((self.num_envs, 2), dtype=bool)
        self.double_support_time = np.zeros(self.num_envs, dtype=np.float64)
        self.prev_root_x = np.zeros(self.num_envs, dtype=np.float64)
        self.prev_world_vx = np.zeros(self.num_envs, dtype=np.float64)
        self.standstill_time = np.zeros(self.num_envs, dtype=np.float64)
        self.progress_streak = np.zeros(self.num_envs, dtype=np.int32)
        self.swing_clear_time = np.zeros((self.num_envs, 2), dtype=np.float64)
        self.swing_clear_steps = np.zeros((self.num_envs, 2), dtype=np.int32)
        self.last_joint_vel = np.zeros((self.num_envs, 8), dtype=np.float32)
        self.rng = np.random.default_rng(seed=42)
        self.current_emotions = self.rng.integers(0, NUM_EMOTIONS, size=self.num_envs, dtype=np.int32)
        self.cmd_modes = np.zeros(self.num_envs, dtype=np.int32)
        self.posture_blend = np.zeros(self.num_envs, dtype=np.float32)
        self.head_err_ema = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.env_v_in = np.full(self.num_envs, 7.4, dtype=np.float32)
        self.env_k_drop = np.zeros(self.num_envs, dtype=np.float32)
        self.env_encoder_bias = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        self.env_imu_rot = np.broadcast_to(np.eye(3, dtype=np.float32), (self.num_envs, 3, 3)).copy()
        self.obs_prev_joint_vel = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        self.tilt_exceed_count = np.zeros(self.num_envs, dtype=np.int32)

        # Obstacle avoidance and approach-velocity tracking state
        self.control_dt = float(self.model.opt.timestep * self.decimation)
        self.obstacle_geom_ids: List[int] = []
        _SLALOM_PILLAR_NAMES = [
            "pillar_l1_left", "pillar_l1_right",
            "pillar_l2_center",
            "pillar_l3_left", "pillar_l3_right",
            "pillar_back",
        ]
        _LEGACY_OBSTACLE_NAMES = [
            "obstacle_front", "obstacle_left", "obstacle_right", "obstacle_back",
        ]
        obstacle_names = _SLALOM_PILLAR_NAMES + _LEGACY_OBSTACLE_NAMES
        for name in obstacle_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid != -1:
                self.obstacle_geom_ids.append(gid)

        self.lidar_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, sname)
            for sname in ("site_lidar_m30", "site_lidar_m15", "site_lidar_0", "site_lidar_p15", "site_lidar_p30")
        ]
        self.head_tof_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "site_head_tof")
        self.head_tof_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "head_tof")
        self.head_tof_sensor_adr = (
            int(self.model.sensor_adr[self.head_tof_sensor_id]) if self.head_tof_sensor_id != -1 else -1
        )
        self.head_scanners = [
            ActiveHeadScanner(max_range_m=float(getattr(self.cfg, "LIDAR_MAX_RANGE", 2.0)))
            for _ in range(self.num_envs)
        ]

        self.prev_d_min = np.zeros(self.num_envs, dtype=np.float32)
        self.d_min = np.zeros(self.num_envs, dtype=np.float32)
        self.d_min_dot = np.zeros(self.num_envs, dtype=np.float32)

        # Canonical V2 foot bodies and sole colliders for clearance and stumble detection
        self.left_foot_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_sole_geom")
        self.right_foot_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_sole_geom")
        self.left_foot_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        self.right_foot_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")

        # Floor geom and base body for domain randomization (friction and torso mass)
        self.floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        if min(self.left_foot_gid, self.right_foot_gid, self.left_foot_bid,
               self.right_foot_bid, self.base_body_id) < 0:
            raise ValueError("V2 model is missing canonical foot or trunk geometry")
        self.default_base_mass = float(self.model.body_mass[self.base_body_id])
        self.default_base_inertia = self.model.body_inertia[self.base_body_id].copy()
        self.default_base_com = self.model.body_ipos[self.base_body_id].copy()
        if (self.cfg.RANDOMIZE_HARDWARE_VARIANTS and
                self.cfg.TORSO_MASS_RANDOMIZATION_KG >= self.default_base_mass):
            self.close()
            raise ValueError("Mass randomization must preserve positive torso mass")
        self._dynamics_data = mujoco.MjData(self.model)
        self._randomize_model_dynamics()

        # Curriculum stage
        self.curriculum_stage = int(stage)
        self.set_stage(self.curriculum_stage)

        # Determine staircase bounding box for edge-fall detection
        self.stair_min_x = 0.08
        self.stair_max_x = 1.10
        self.stair_half_w = 0.40
        stair_gids = []
        for gid in range(self.model.ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if gname and ("stair" in gname or "platform" in gname):
                stair_gids.append(gid)
        self.terrain_gids = stair_gids
        self.terrain_box_gids: List[int] = [
            gid
            for gid in range(self.model.ngeom)
            if self.model.geom_bodyid[gid] == 0
            and self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
        ]
        if (
            self.left_foot_gid != -1
            and self.right_foot_gid != -1
            and self.model.geom_type[self.left_foot_gid] == mujoco.mjtGeom.mjGEOM_BOX
            and self.model.geom_type[self.right_foot_gid] == mujoco.mjtGeom.mjGEOM_BOX
        ):
            self._sole_pts_local: Optional[np.ndarray] = np.stack(
                [
                    _SOLE_SAMPLE_SIGNS * self.model.geom_size[self.left_foot_gid],
                    _SOLE_SAMPLE_SIGNS * self.model.geom_size[self.right_foot_gid],
                ],
                axis=0,
            )
        else:
            self._sole_pts_local = None
        if stair_gids:
            self.stair_min_x = float(min(self.model.geom_pos[g, 0] - self.model.geom_size[g, 0] for g in stair_gids))
            self.stair_max_x = float(max(self.model.geom_pos[g, 0] + self.model.geom_size[g, 0] for g in stair_gids))
            self.stair_half_w = float(min(self.model.geom_size[g, 1] for g in stair_gids))

        # Self-righting recovery controllers & pre-impact limp-fall predictors
        self.enable_auto_recovery = bool(enable_auto_recovery)
        self.allow_teleport_assist = bool(allow_teleport_assist)
        if self.enable_auto_recovery:
            self.recovery_controllers: List[SelfRightingController] = [
                SelfRightingController.from_model(self.model) for _ in range(self.num_envs)
            ]
        else:
            self.recovery_controllers = []
        self.fall_predictors: List[FallPredictor] = [
            FallPredictor(limp_kp_scale=self.limp_kp_scale) for _ in range(self.num_envs)
        ]
        self.limp_active = np.zeros(self.num_envs, dtype=bool)

        # Hierarchical waypoint navigation planner (L2)
        self.use_nav_planner = bool(use_nav_planner)
        self.nav_goal_xy = tuple(nav_goal_xy)
        if self.use_nav_planner:
            self.nav_planner = LidarWaypointPlanner(nav_planner_cfg)
        else:
            self.nav_planner = None

        # Initialize environments
        self.reset()

    def set_stand_still(self, stand_still: bool) -> None:
        """Toggle stationary standing mode (zero velocity commands)."""
        self.stand_still = bool(stand_still)
        if self.stand_still:
            self.commands[:, :] = 0.0

    def set_command_bucket_probs(self, zero_cmd_prob: float, turn_in_place_prob: float) -> None:
        """Configure fractions of episodes sampled with [0,0,0] or pure turn-in-place commands."""
        zp = float(zero_cmd_prob)
        tp = float(turn_in_place_prob)
        if not (np.isfinite(zp) and np.isfinite(tp)) or zp < 0.0 or tp < 0.0 or (zp + tp) > 1.0 + 1e-6:
            raise ValueError("zero_cmd_prob and turn_in_place_prob must be >= 0 and sum to <= 1.0")
        self.zero_cmd_prob = zp
        self.turn_in_place_prob = tp

    def set_emotions(self, emotions: Union[int, EmotionState, np.ndarray, List[int]]) -> None:
        """Set emotion state across environments."""
        if isinstance(emotions, (int, EmotionState)):
            self.current_emotions[:] = int(emotions)
        else:
            self.current_emotions[:] = np.asarray(emotions, dtype=np.int32)

    def _apply_posture_command_encoding(self, env_indices: Optional[Union[List[int], np.ndarray, range]] = None) -> None:
        """Encode SIT/LIE posture mode into commands so Actor observes requested posture."""
        if not self.encode_posture_commands:
            return
        indices = range(self.num_envs) if env_indices is None else env_indices
        for idx in indices:
            mode = int(self.cmd_modes[idx])
            if mode == int(LocomotionMode.SIT):
                self.commands[idx] = [0.0, -0.5, 0.0]
            elif mode == int(LocomotionMode.LIE):
                self.commands[idx] = [0.0, -1.0, 0.0]

    def set_cmd_modes(self, modes: Union[int, List[int], np.ndarray, LocomotionMode]) -> None:
        """Set locomotion mode (WALK=0, SIT=1, LIE=2) across environments."""
        prev_modes = self.cmd_modes.copy()
        if isinstance(modes, (int, LocomotionMode)):
            self.cmd_modes[:] = int(modes)
        else:
            self.cmd_modes[:] = np.asarray(modes, dtype=np.int32)
        changed = (self.cmd_modes != prev_modes) | (self.cmd_modes == int(LocomotionMode.WALK))
        self.posture_blend[changed] = 0.0
        self._apply_posture_command_encoding()

    def set_cmd_mode(self, mode: Union[int, List[int], np.ndarray, LocomotionMode]) -> None:
        """Alias for set_cmd_modes."""
        self.set_cmd_modes(mode)

    def set_stage(self, stage: int) -> None:
        """Update terrain and obstacle curriculum stage."""
        prev_stage = getattr(self, "curriculum_stage", None)
        self.curriculum_stage = int(stage)
        if self.terrain_mode == "stairs":
            apply_stair_stage(self.model, self.curriculum_stage)
        else:
            apply_stage(self.model, self.curriculum_stage)
        if prev_stage is not None and self.curriculum_stage != prev_stage:
            self.completed_ep_lens.clear()
            self.completed_distances_x.clear()
            self.completed_distances_y.clear()

    def _extract_foot_dynamics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract foot heights, horizontal velocities, and 3D contact forces in world frame.

        Returns:
            foot_z: (num_envs, 2)
            foot_vel_xy: (num_envs, 2, 2)
            foot_contact_forces: (num_envs, 2, 3)
        """
        foot_z = np.zeros((self.num_envs, 2), dtype=np.float32)
        foot_vel_xy = np.zeros((self.num_envs, 2, 2), dtype=np.float32)
        foot_contact_forces = np.zeros((self.num_envs, 2, 3), dtype=np.float32)

        has_feet = (
            self.left_foot_bid != -1
            and self.right_foot_bid != -1
            and self.left_foot_gid != -1
            and self.right_foot_gid != -1
        )
        if not has_feet:
            return foot_z, foot_vel_xy, foot_contact_forces

        v_buf = np.zeros(6, dtype=np.float64)
        c_force = np.zeros(6, dtype=np.float64)

        for env_idx, d in enumerate(self.datas):
            # 1. Foot Heights
            foot_z[env_idx, 0] = d.xpos[self.left_foot_bid][2]
            foot_z[env_idx, 1] = d.xpos[self.right_foot_bid][2]

            # 2. Foot Velocities in World Frame (flg_local=0)
            mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, self.left_foot_bid, v_buf, 0)
            foot_vel_xy[env_idx, 0] = v_buf[3:5]

            mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, self.right_foot_bid, v_buf, 0)
            foot_vel_xy[env_idx, 1] = v_buf[3:5]

            # 3. 3D Contact Forces on feet transformed to world frame
            for c_idx in range(d.ncon):
                contact = d.contact[c_idx]
                g1 = contact.geom1
                g2 = contact.geom2

                is_left = (g1 == self.left_foot_gid) or (g2 == self.left_foot_gid)
                is_right = (g1 == self.right_foot_gid) or (g2 == self.right_foot_gid)
                if not (is_left or is_right):
                    continue

                mujoco.mj_contactForce(self.model, d, c_idx, c_force)
                r_c2w = contact.frame.reshape(3, 3).T
                f_world = r_c2w @ c_force[0:3]

                if is_left:
                    foot_contact_forces[env_idx, 0] += f_world
                if is_right:
                    foot_contact_forces[env_idx, 1] += f_world

        return foot_z, foot_vel_xy, foot_contact_forces

    def _get_lidar_ranges(self, env_idx: int) -> Optional[np.ndarray]:
        """Extract 5-ray LiDAR readings in meters with floor hit filtering."""
        lidar_max = float(getattr(self.cfg, "LIDAR_MAX_RANGE", 2.0))
        return extract_filtered_lidar_ranges(
            self.model,
            self.datas[env_idx],
            obs_dim=self.base_obs_dim,
            lidar_max=lidar_max,
            site_ids=self.lidar_site_ids,
        )

    def get_head_tof_range(self, env_idx: int) -> float:
        """Read filtered brow head_tof distance (m) and feed into ActiveHeadScanner."""
        lidar_max = float(getattr(self.cfg, "LIDAR_MAX_RANGE", 2.0))
        d = self.datas[env_idx]
        rng_m = extract_filtered_head_tof_range(
            self.model,
            d,
            lidar_max=lidar_max,
            site_id=self.head_tof_site_id,
            sensor_adr=self.head_tof_sensor_adr,
        )
        head_yaw = float(d.sensordata[12]) if len(d.sensordata) > 13 else 0.0
        head_pitch = float(d.sensordata[13]) if len(d.sensordata) > 13 else 0.0
        now_s = float(self.episode_steps[env_idx] * self.control_dt)
        self.head_scanners[env_idx].update(head_yaw, head_pitch, rng_m, now_s)
        return rng_m

    def update_nav_commands(
        self,
        env_indices: Optional[Union[List[int], np.ndarray, range]] = None,
        lidar_obs: Optional[np.ndarray] = None,
    ) -> None:
        """Update self.commands for specified environments using LidarWaypointPlanner.

        Extracts current robot pose (qpos[:2], qpos[3:7]) and active head_tof / 5-ray LiDAR ranges.
        """
        if not getattr(self, "use_nav_planner", False) or getattr(self, "nav_planner", None) is None:
            return

        if env_indices is None:
            indices = list(range(self.num_envs))
        else:
            indices = [int(i) for i in env_indices]

        goal_xy = np.asarray(self.nav_goal_xy, dtype=np.float32)[:2]

        for idx_k, env_idx in enumerate(indices):
            d = self.datas[env_idx]
            pos_xy = np.asarray(d.qpos[:2], dtype=np.float32)
            quat_wxyz = np.asarray(d.qpos[3:7], dtype=np.float32)
            now_s = float(self.episode_steps[env_idx] * self.control_dt)

            if lidar_obs is not None:
                lidar_arr = np.asarray(lidar_obs, dtype=np.float32)
                if lidar_arr.ndim == 1:
                    if lidar_arr.shape[0] == 5 or lidar_arr.shape[0] >= 42:
                        raw_ray = lidar_arr[-5:]
                    else:
                        self.get_head_tof_range(env_idx)
                        raw_ray = self.head_scanners[env_idx].get_virtual_rays(now_s)
                elif lidar_arr.ndim == 2:
                    k = idx_k if lidar_arr.shape[0] == len(indices) else env_idx
                    if lidar_arr.shape[1] >= 42:
                        raw_ray = lidar_arr[k, 37:42]
                    elif lidar_arr.shape[1] == 5:
                        raw_ray = lidar_arr[k, :5]
                    else:
                        self.get_head_tof_range(env_idx)
                        raw_ray = self.head_scanners[env_idx].get_virtual_rays(now_s)
                else:
                    raw_ray = None
            elif self.head_tof_sensor_adr >= 0:
                self.get_head_tof_range(env_idx)
                raw_ray = self.head_scanners[env_idx].get_virtual_rays(now_s)
            else:
                raw_ray = self._get_lidar_ranges(env_idx)

            cmd = self.nav_planner.compute_command_from_quat(
                pos_xy=pos_xy,
                quat_wxyz=quat_wxyz,
                goal_xy=goal_xy,
                lidar_ranges=raw_ray,
            )
            self.commands[env_idx, :3] = cmd

    def _has_active_obstacles(self) -> bool:
        """Return True if at least one obstacle geom is raised above the floor (z > 0.0)."""
        return any(float(self.model.geom_pos[gid, 2]) > 0.0 for gid in self.obstacle_geom_ids)

    def _compute_d_min(self, env_idx: Optional[int] = None) -> Optional[Union[float, np.ndarray]]:
        """Compute minimum distance to obstacle cylinders (minus individual cylinder radius)."""
        if not self.obstacle_geom_ids:
            return None
        active_gids = [gid for gid in self.obstacle_geom_ids if self.model.geom_pos[gid, 2] > 0.0]
        if not active_gids:
            return None
        obs_xy = np.array([self.model.geom_pos[gid, :2] for gid in active_gids], dtype=np.float32)
        obs_radii = np.array([self.model.geom_size[gid, 0] for gid in active_gids], dtype=np.float32)
        if env_idx is not None:
            base_xy = np.array(self.datas[env_idx].qpos[0:2], dtype=np.float32)
            diff = base_xy - obs_xy
            dist = np.linalg.norm(diff, axis=-1) - obs_radii
            return float(np.clip(np.min(dist), 0.0, None))
        else:
            base_xy = np.array([d.qpos[0:2] for d in self.datas], dtype=np.float32)
            diff = base_xy[:, None, :] - obs_xy[None, :, :]
            dist = np.linalg.norm(diff, axis=-1) - obs_radii[None, :]
            return np.clip(np.min(dist, axis=-1), 0.0, None).astype(np.float32)

    def randomize_stage_pillars(self) -> List[str]:
        """Synchronously randomize active slalom pillars on shared model outside worker threads."""
        if self.randomize_pillars_enabled and self.terrain_mode == "hurdles" and self.curriculum_stage >= 1:
            return randomize_pillars(
                self.model,
                stage=self.curriculum_stage,
                rng=self.rng,
                density_rho=self.pillar_density,
                jitter_xy=self.pillar_jitter_xy,
            )
        return []

    def _randomize_model_dynamics(self):
        """Randomize shared dynamics between worker batches and refresh MuJoCo constants."""
        if self.floor_gid != -1:
            self.model.geom_friction[self.floor_gid, 0] = float(
                self.rng.uniform(self.friction_range[0], self.friction_range[1])
            )
        if self.base_body_id != -1:
            variant_dynamics = bool(self.cfg.RANDOMIZE_HARDWARE_VARIANTS)
            mass_delta = (float(self.cfg.TORSO_MASS_RANDOMIZATION_KG) if variant_dynamics
                          else self.default_base_mass * 0.05)
            com_delta = float(self.cfg.TORSO_COM_RANDOMIZATION_M) if variant_dynamics else 0.0
            mass = self.default_base_mass + self.rng.uniform(-mass_delta, mass_delta)
            self.model.body_mass[self.base_body_id] = mass
            self.model.body_inertia[self.base_body_id] = (
                self.default_base_inertia * mass / self.default_base_mass
            )
            self.model.body_ipos[self.base_body_id] = self.default_base_com
            if com_delta:
                self.model.body_ipos[self.base_body_id] += self.rng.uniform(-com_delta, com_delta, size=3)
            mujoco.mj_setConst(self.model, self._dynamics_data)

    def randomize_dynamics(self) -> None:
        """Randomize shared model dynamics at synchronized batch/episode boundaries."""
        self._randomize_model_dynamics()

    def reset_env(self, env_idx: int) -> np.ndarray:
        """Reset a single environment with random velocity commands."""
        d = self.datas[env_idx]
        mujoco.mj_resetData(self.model, d)

        # Domain randomization on shared self.model is performed synchronously in
        # reset() / _randomize_model_dynamics() outside worker thread execution to avoid data races.
        if self.num_envs == 1:
            self._randomize_model_dynamics()

        if self.include_emotions:
            self.current_emotions[env_idx] = self.rng.integers(0, NUM_EMOTIONS)

        d.qpos[0:3] = self.model.qpos0[0:3]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        # Khởi tạo góc khớp ban đầu quanh tư thế danh định đứng cân bằng (chùng gối nhẹ)
        nominal = getattr(self.cfg, "nominal_qpos", DEFAULT_NOMINAL_QPOS)
        d.qpos[7:15] = nominal + self.rng.uniform(-0.02, 0.02, size=self.action_dim)
        d.qpos[11] = np.clip(
            d.qpos[11],
            self.cfg.JOINT_LIMITS["left_knee_pitch"][0],
            self.cfg.JOINT_LIMITS["left_knee_pitch"][1],
        )
        d.qpos[14] = np.clip(
            d.qpos[14],
            self.cfg.JOINT_LIMITS["right_knee_pitch"][0],
            self.cfg.JOINT_LIMITS["right_knee_pitch"][1],
        )
        d.qvel[:] = 0.0

        # Ngẫu nhiên hóa lệnh vận tốc mục tiêu [vx, vy, wz] (hoặc giữ đứng yên nếu stand_still)
        if self.stand_still:
            self.commands[env_idx, :] = 0.0
        else:
            bucket_mode = "walk"
            if self.zero_cmd_prob > 0.0 or self.turn_in_place_prob > 0.0:
                u = float(self.rng.random())
                if u < self.zero_cmd_prob:
                    bucket_mode = "zero"
                elif u < self.zero_cmd_prob + self.turn_in_place_prob:
                    bucket_mode = "turn"

            if bucket_mode == "zero":
                self.commands[env_idx, :] = 0.0
            elif bucket_mode == "turn":
                wz_lim = (
                    0.50
                    if self.terrain_mode == "flat"
                    else min(0.10, 0.04 + 0.02 * getattr(self, "curriculum_stage", 0))
                )
                sign = -1.0 if float(self.rng.random()) < 0.5 else 1.0
                mag = float(self.rng.uniform(0.4 * wz_lim, wz_lim))
                self.commands[env_idx, 0] = 0.0
                self.commands[env_idx, 1] = 0.0
                self.commands[env_idx, 2] = sign * mag
            elif self.terrain_mode in ("hurdles", "stairs"):
                self.commands[env_idx, 0] = self.rng.uniform(0.18, 0.25)
                self.commands[env_idx, 1] = self.rng.uniform(-0.03, 0.03)
                wz_lim = min(0.10, 0.04 + 0.02 * getattr(self, "curriculum_stage", 0))
                self.commands[env_idx, 2] = self.rng.uniform(-wz_lim, wz_lim)
            else:
                self.commands[env_idx, 0] = self.rng.uniform(0.10, 0.25)
                self.commands[env_idx, 1] = self.rng.uniform(-0.08, 0.08)
                self.commands[env_idx, 2] = self.rng.uniform(-0.50, 0.50)

        self.posture_blend[env_idx] = 0.0
        self.head_err_ema[env_idx] = 0.0
        self._apply_posture_command_encoding([env_idx])

        mujoco.mj_forward(self.model, d)

        self.last_actions[env_idx] = 0.0
        self.action_delay_buffer[env_idx] = 0.0
        self.env_action_delay[env_idx] = (
            int(self.rng.integers(0, self.max_action_delay_steps + 1))
            if self.max_action_delay_steps > 0
            else 0
        )
        self.last_targets[env_idx] = d.qpos[7:15].astype(np.float32)
        self.last_joint_vel[env_idx] = 0.0
        self.obs_prev_joint_vel[env_idx] = 0.0
        if self.enable_battery_sag:
            self.env_v_in[env_idx] = float(self.rng.uniform(6.5, 8.2))
            self.env_k_drop[env_idx] = float(self.rng.uniform(0.0, 0.20))
        if self.encoder_bias_rad > 0.0:
            self.env_encoder_bias[env_idx] = self.rng.uniform(
                -self.encoder_bias_rad, self.encoder_bias_rad, size=self.action_dim
            ).astype(np.float32)
        if self.imu_mount_max_rad > 0.0:
            ax = self.rng.normal(size=3).astype(np.float32)
            ang = float(self.rng.uniform(0.0, self.imu_mount_max_rad))
            self.env_imu_rot[env_idx] = small_angle_rotation_matrix(ax, ang)
        self.double_support_time[env_idx] = 0.0
        self.prev_root_x[env_idx] = float(d.qpos[0])
        self.prev_world_vx[env_idx] = float(d.qvel[0])
        self.standstill_time[env_idx] = 0.0
        self.progress_streak[env_idx] = 0
        self.swing_clear_time[env_idx] = 0.0
        self.swing_clear_steps[env_idx] = 0
        self.episode_steps[env_idx] = 0
        self.feet_air_time[env_idx] = 0.0
        self.last_feet_contact[env_idx] = True
        if self.enable_auto_recovery and env_idx < len(self.recovery_controllers):
            self.recovery_controllers[env_idx].reset()
        self.fall_predictors[env_idx].reset()
        self.limp_active[env_idx] = False
        self.head_scanners[env_idx].reset()
        if self.head_tof_sensor_adr >= 0:
            self.get_head_tof_range(env_idx)
        if self.use_nav_planner:
            self.update_nav_commands(env_indices=[env_idx])
        obs = self._extract_obs(env_idx)
        if not self._has_active_obstacles():
            d_init = None
        elif self.base_obs_dim == 42 and len(obs) >= 42:
            d_init = float(np.min(obs[37:42]))
        else:
            d_init = self._compute_d_min(env_idx)
        if d_init is not None:
            self.prev_d_min[env_idx] = d_init
            self.d_min[env_idx] = d_init
            self.d_min_dot[env_idx] = 0.0
        else:
            self.prev_d_min[env_idx] = 0.0
            self.d_min[env_idx] = 0.0
            self.d_min_dot[env_idx] = 0.0
        return obs

    def reset(self, env_ids: Optional[Union[List[int], np.ndarray]] = None) -> np.ndarray:
        """Reset specified environments or all if None. Returns array of observations."""
        self._randomize_model_dynamics()
        if env_ids is None:
            self.randomize_stage_pillars()
            indices = range(self.num_envs)
        else:
            indices = env_ids

        for i in indices:
            self.reset_env(i)

        if self.use_nav_planner:
            self.update_nav_commands(env_indices=indices)

        return self.get_observations()

    def _extract_obs(self, env_idx: int, corrupt_actor: bool = True) -> np.ndarray:
        """Extract the V2 observation vector from environment data."""
        d = self.datas[env_idx]

        # Base angular velocity from gyroscope sensor
        base_ang_vel = np.array(d.sensordata[3:6], dtype=np.float32)

        # Projected gravity vector from base quaternion
        w, x, y, z = d.sensordata[6:10]
        proj_grav = np.array(
            [
                2.0 * (w * y - x * z),
                -2.0 * (w * x + y * z),
                2.0 * (x * x + y * y) - 1.0,
            ],
            dtype=np.float32,
        )

        cmd = self.commands[env_idx]
        joint_pos = np.array(d.sensordata[12:20], dtype=np.float32)
        joint_vel = np.array(d.sensordata[20:28], dtype=np.float32)
        if corrupt_actor:
            if self.imu_mount_max_rad > 0.0:
                rot = self.env_imu_rot[env_idx]
                base_ang_vel = (rot @ base_ang_vel).astype(np.float32)
                proj_grav = (rot @ proj_grav).astype(np.float32)
            if self.encoder_bias_rad > 0.0:
                joint_pos = (joint_pos + self.env_encoder_bias[env_idx]).astype(np.float32)
            if self.delay_joint_vel_obs:
                joint_vel = self.obs_prev_joint_vel[env_idx].copy()
        last_act = self.last_actions[env_idx]

        # Foot contact sensors: left (10), right (11), padded to 4 dimensions
        # Binarize normal contact forces (> 0.1 N) to match hardware deployment
        left_touch = float(d.sensordata[10])
        right_touch = float(d.sensordata[11])
        foot_contact = np.array(
            [1.0 if left_touch > 0.1 else 0.0, 1.0 if right_touch > 0.1 else 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

        # Legacy 5-ray LiDAR readings only when explicitly requested (OBS_DIM == 42)
        lidar_ranges = self._get_lidar_ranges(env_idx) if self.base_obs_dim == 42 else None

        emotion_one_hot = (
            get_emotion_one_hot(self.current_emotions[env_idx])
            if self.include_emotions
            else None
        )

        if self.embed_phase_clock:
            _, clock_sin, clock_cos = compute_gait_phase(
                self.episode_steps[env_idx], dt_control=self.control_dt, period=0.8
            )
            phase_clock = np.array([clock_sin, clock_cos], dtype=np.float32)
        else:
            phase_clock = None

        return compute_observations(
            base_ang_vel=base_ang_vel,
            proj_grav=proj_grav,
            commands=cmd,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            last_actions=last_act,
            foot_contact=foot_contact,
            lidar_ranges=lidar_ranges,
            emotion_one_hot=emotion_one_hot,
            phase_clock=phase_clock,
        )

    def get_observations(self) -> np.ndarray:
        """Extract all current observations as a (num_envs, obs_dim) numpy array."""
        obs = np.empty((self.num_envs, self.obs_dim), dtype=np.float32)
        for i in range(self.num_envs):
            obs[i] = self._extract_obs(i)
        return obs

    def _get_terrain_height_at_xy(self, x: float, y: float) -> float:
        """Query top elevation of terrain / stairs at world coordinate (x, y)."""
        h = 0.0
        terrain_gids = getattr(self, "terrain_gids", None)
        if terrain_gids:
            for gid in terrain_gids:
                gx, gy, gz = self.model.geom_pos[gid]
                sx, sy, sz = self.model.geom_size[gid]
                top_z = float(gz + sz)
                if top_z > h and abs(x - gx) <= sx and abs(y - gy) <= sy:
                    h = top_z
        return float(h)

    def _compute_sole_clearance(self) -> np.ndarray:
        """Compute sole-to-local-terrain clearance (m) for both feet, shape (num_envs, 2)."""
        floor_z = (
            float(
                self.model.geom_pos[self.floor_gid, 2]
                + (
                    self.model.geom_size[self.floor_gid, 2]
                    if self.model.geom_type[self.floor_gid] == mujoco.mjtGeom.mjGEOM_BOX
                    else 0.0
                )
            )
            if self.floor_gid != -1
            else 0.0
        )

        active_boxes = [
            gid
            for gid in getattr(self, "terrain_box_gids", ())
            if float(self.model.geom_pos[gid, 2] + self.model.geom_size[gid, 2]) > floor_z
            and (int(self.model.geom_contype[gid]) != 0 or int(self.model.geom_conaffinity[gid]) != 0)
        ]

        if self._sole_pts_local is not None:
            l_gid = self.left_foot_gid
            r_gid = self.right_foot_gid
            centers = np.array(
                [[d.geom_xpos[l_gid], d.geom_xpos[r_gid]] for d in self.datas],
                dtype=np.float64,
            )
            rmats = np.array(
                [[d.geom_xmat[l_gid], d.geom_xmat[r_gid]] for d in self.datas],
                dtype=np.float64,
            ).reshape(self.num_envs, 2, 3, 3)
            pts_world = centers[:, :, None, :] + np.einsum("efij,fpj->efpi", rmats, self._sole_pts_local)

            if active_boxes:
                g_pos = self.model.geom_pos[active_boxes]
                g_size = self.model.geom_size[active_boxes]
                top_z = g_pos[:, 2] + g_size[:, 2]
                px = pts_world[..., 0:1]
                py = pts_world[..., 1:2]
                contained = (np.abs(px - g_pos[:, 0]) <= g_size[:, 0]) & (
                    np.abs(py - g_pos[:, 1]) <= g_size[:, 1]
                )
                terrain_z = np.maximum(floor_z, np.max(np.where(contained, top_z, floor_z), axis=-1))
            else:
                terrain_z = floor_z

            min_clr = np.min(pts_world[..., 2] - terrain_z, axis=-1)
            return np.where(np.isfinite(min_clr), np.maximum(0.0, min_clr), 0.0).astype(np.float32)

        # Unsupported sole geometry cannot establish clearance. Body-origin height
        # is not a safe substitute: it can unlock the swing gate on a resting foot.
        return np.zeros((self.num_envs, 2), dtype=np.float32)

    def get_privileged_obs(
        self,
        raw_obs: Optional[np.ndarray] = None,
        env_indices: Optional[Union[List[int], np.ndarray]] = None,
    ) -> np.ndarray:
        """Extract privileged observation tensor of shape (N, obs_dim + 11).

        Components:
        - raw_obs (obs_dim): standard actor observation vector
        - base_lin_vel (3): linear velocity in base body frame
        - contact_forces_z (2): vertical contact forces on left/right foot soles
        - terrain_height_at_feet (2): terrain elevation directly under left/right foot
        - nearest_obstacle_dist (1): distance to nearest obstacle surface
        - nearest_obstacle_rel_xy (2): relative 2D offset to nearest obstacle in robot body frame
        - current_stage_norm (1): curriculum stage normalized to [0, 1]
        """
        if env_indices is None:
            indices = list(range(self.num_envs))
        else:
            indices = [int(i) for i in env_indices]

        n_envs = len(indices)
        if n_envs == 0:
            return np.empty((0, self.privileged_obs_dim), dtype=np.float32)

        has_actor_corruption = (
            self.encoder_bias_rad > 0.0
            or self.imu_mount_max_rad > 0.0
            or self.delay_joint_vel_obs
        )
        if raw_obs is None or has_actor_corruption:
            actor_obs = np.array([self._extract_obs(i, corrupt_actor=False) for i in indices], dtype=np.float32)
        else:
            raw_arr = np.asarray(raw_obs, dtype=np.float32)
            if raw_arr.shape[0] == self.num_envs and len(indices) < self.num_envs:
                actor_obs = raw_arr[indices]
            else:
                actor_obs = raw_arr

        stage_norm = float(getattr(self, "curriculum_stage", 0)) / 3.0
        priv_extra = np.zeros((n_envs, 11), dtype=np.float32)

        # 1. Base linear velocity in body frame (3)
        base_lin_vel_world = np.stack([self.datas[i].qvel[0:3] for i in indices], axis=0).astype(np.float32)
        base_quats = np.stack([self.datas[i].qpos[3:7] for i in indices], axis=0).astype(np.float32)
        priv_extra[:, 0:3] = quat_rotate_inverse(base_quats, base_lin_vel_world)

        # 2. Foot contact forces Z (2)
        if len(self.datas[indices[0]].sensordata) >= 12:
            priv_extra[:, 3:5] = np.stack([self.datas[i].sensordata[10:12] for i in indices], axis=0).astype(np.float32)

        # 3. Terrain height at feet (2)
        active_terrain_gids = [
            gid for gid in (getattr(self, "terrain_gids", None) or [])
            if float(self.model.geom_pos[gid, 2] + self.model.geom_size[gid, 2]) > 0.0
        ]
        if active_terrain_gids and self.left_foot_bid != -1 and self.right_foot_bid != -1:
            g_pos = self.model.geom_pos[active_terrain_gids]
            g_size = self.model.geom_size[active_terrain_gids]
            gx = g_pos[:, 0]
            gy = g_pos[:, 1]
            sx = g_size[:, 0]
            sy = g_size[:, 1]
            top_z = g_pos[:, 2] + g_size[:, 2]

            left_pos = np.stack([self.datas[i].xpos[self.left_foot_bid, :2] for i in indices], axis=0)
            right_pos = np.stack([self.datas[i].xpos[self.right_foot_bid, :2] for i in indices], axis=0)
            feet_xy = np.stack([left_pos, right_pos], axis=1)  # (n_envs, 2, 2)

            fx = feet_xy[..., 0:1]  # (n_envs, 2, 1)
            fy = feet_xy[..., 1:2]  # (n_envs, 2, 1)

            contained = (np.abs(fx - gx) <= sx) & (np.abs(fy - gy) <= sy)  # (n_envs, 2, K)
            candidates = np.where(contained, top_z, 0.0)
            priv_extra[:, 5:7] = np.max(candidates, axis=-1).astype(np.float32)

        # 4. Nearest obstacle dist (1) & rel XY in body frame (2)
        active_obstacle_gids = [
            gid for gid in self.obstacle_geom_ids if float(self.model.geom_pos[gid, 2]) > 0.0
        ]
        if len(active_obstacle_gids) > 0:
            obs_xy = np.array([self.model.geom_pos[gid, :2] for gid in active_obstacle_gids], dtype=np.float32)
            obs_radii = np.array([self.model.geom_size[gid, 0] for gid in active_obstacle_gids], dtype=np.float32)
            base_xy = np.stack([self.datas[i].qpos[0:2] for i in indices], axis=0).astype(np.float32)
            diff = obs_xy[None, :, :] - base_xy[:, None, :]  # (n_envs, M, 2)
            dist = np.linalg.norm(diff, axis=-1) - obs_radii[None, :]  # (n_envs, M)
            min_i = np.argmin(dist, axis=-1)  # (n_envs,)
            row_idx = np.arange(n_envs)
            priv_extra[:, 7] = np.clip(dist[row_idx, min_i], 0.0, None)

            yaws = quat_to_yaw(base_quats)
            cos_y = np.cos(yaws)
            sin_y = np.sin(yaws)
            min_diff = diff[row_idx, min_i, :]  # (n_envs, 2)
            dx_w = min_diff[:, 0]
            dy_w = min_diff[:, 1]
            priv_extra[:, 8] = cos_y * dx_w + sin_y * dy_w
            priv_extra[:, 9] = -sin_y * dx_w + cos_y * dy_w
        else:
            priv_extra[:, 7] = 2.0
            priv_extra[:, 8] = 2.0
            priv_extra[:, 9] = 0.0

        # 5. Curriculum stage norm (1)
        priv_extra[:, 10] = stage_norm

        return np.nan_to_num(
            np.concatenate([actor_obs, priv_extra], axis=-1),
            nan=0.0,
            posinf=100.0,
            neginf=-100.0,
        ).astype(np.float32)

    def _step_subrange(
        self,
        indices: np.ndarray,
        target_positions: np.ndarray,
    ):
        """Worker function to simulate a subset of environments."""
        for idx in indices:
            d = self.datas[idx]
            target_pos = target_positions[idx]
            kp_eff = self.kp * (self.limp_kp_scale if (self.enable_limp_fall and self.limp_active[idx]) else 1.0)
            v_in_i = float(self.env_v_in[idx]) if self.enable_battery_sag else 7.4
            k_drop_i = float(self.env_k_drop[idx]) if self.enable_battery_sag else 0.0
            for _ in range(self.decimation):
                current_pos = d.qpos[7:15]
                current_vel = d.qvel[6:14]
                # PD controller for MG996R servo position tracking
                torques = kp_eff * (target_pos - current_pos) - self.kd * current_vel
                torque_lim = (
                    compute_battery_sag_torque_limit(
                        torques,
                        v_in=v_in_i,
                        k_drop=k_drop_i,
                        nominal_torque_limit=self.cfg.TORQUE_LIMIT,
                    )
                    if self.enable_battery_sag
                    else self.cfg.TORQUE_LIMIT
                )
                d.ctrl[:] = np.clip(torques, -torque_lim, torque_lim)
                mujoco.mj_step(self.model, d)

    def step(
        self, actions: np.ndarray, fast: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        """Step all environments by one control step with given normalized actions.

        Returns: (next_observations, rewards, dones, infos)
        """
        actions = np.asarray(actions, dtype=np.float32)
        assert actions.shape == (
            self.num_envs,
            self.action_dim,
        ), f"Expected action shape ({self.num_envs}, {self.action_dim}), got {actions.shape}"

        pre_step_finite = np.array(
            [np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)) for d in self.datas],
            dtype=bool,
        )
        vx_world_before = np.array([d.qvel[0] for d in self.datas], dtype=np.float64)

        if self.posture_ramp_s > 0.0:
            posture_active = (self.cmd_modes == int(LocomotionMode.SIT)) | (self.cmd_modes == int(LocomotionMode.LIE))
            blend_step = float(self.control_dt / self.posture_ramp_s)
            self.posture_blend = np.where(
                posture_active,
                np.clip(self.posture_blend + blend_step, 0.0, 1.0),
                0.0,
            ).astype(np.float32)

        # 1. Apply optional actuator delay queue and scale actions around nominal stance (or slewed posture)
        if self.max_action_delay_steps > 0:
            self.action_delay_buffer[:, 1:, :] = self.action_delay_buffer[:, :-1, :]
            self.action_delay_buffer[:, 0, :] = actions
            exec_actions = self.action_delay_buffer[np.arange(self.num_envs), self.env_action_delay]
        else:
            exec_actions = actions
        if self.encode_posture_commands:
            posture_mask = (self.cmd_modes == int(LocomotionMode.SIT)) | (self.cmd_modes == int(LocomotionMode.LIE))
            if np.any(posture_mask):
                alphas = (
                    self.posture_blend
                    if self.posture_ramp_s > 0.0
                    else np.ones(self.num_envs, dtype=np.float32)
                )
                slewed_centers, _ = compute_slewed_posture_target(alphas, self.cmd_modes)
                nominal_centers = np.where(
                    posture_mask[:, None],
                    slewed_centers,
                    self.cfg.nominal_qpos[None, :],
                )
                clipped_act = np.clip(np.asarray(exec_actions, dtype=np.float32), -1.0, 1.0)
                scaled_targets = np.clip(
                    nominal_centers + clipped_act * self.cfg.action_scales,
                    self.cfg.low_limits,
                    self.cfg.high_limits,
                ).astype(np.float32)
            else:
                scaled_targets = scale_actions(exec_actions, self.cfg, nominal_centered=True)
        else:
            scaled_targets = scale_actions(exec_actions, self.cfg, nominal_centered=True)

        assisted_recovery_flags = np.zeros(self.num_envs, dtype=bool)
        if self.enable_auto_recovery:
            for i in range(self.num_envs):
                ctrl = self.recovery_controllers[i]
                if ctrl.is_recovering:
                    qpos = self.datas[i].qpos[7:15]
                    root_z_val = float(self.datas[i].qpos[2])
                    w, x, y, z = self.datas[i].sensordata[6:10]
                    proj_grav_val = np.array([
                        2.0 * (w * y - x * z),
                        -2.0 * (w * x + y * z),
                        2.0 * (x * x + y * y) - 1.0,
                    ], dtype=np.float32)
                    target_q, is_done = ctrl.step(qpos, root_z_val, proj_grav_val)
                    scaled_targets[i] = target_q
                    if self.allow_teleport_assist and (
                        ctrl.is_push_completed
                        or (ctrl.state == SelfRightingController.PHASE3_PUSH_TO_STAND and ctrl.phase_steps >= 15)
                    ):
                        d = self.datas[i]
                        d.qpos[2] = max(float(d.qpos[2]), float(self.model.qpos0[2]))
                        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                        d.qvel[:] = 0.0
                        mujoco.mj_forward(self.model, d)
                        assisted_recovery_flags[i] = True

        # 2. Rate-limit target joint changes (vectorized)
        target_positions = clip_action_rates(
            scaled_targets,
            self.last_targets,
            dt=self.control_dt,
            max_velocity=self.cfg.MAX_VELOCITY,
        )
        self.last_targets[:] = target_positions

        # Snapshot pre-step joint velocity so optional 1-step joint_vel lag exposes previous tick's readings
        if self.delay_joint_vel_obs:
            for i in range(self.num_envs):
                self.obs_prev_joint_vel[i] = self.datas[i].sensordata[20:28]

        # 3. Step physics simulation (either multi-threaded or sequential)
        if self.executor is not None and self.num_threads > 1:
            chunks = np.array_split(np.arange(self.num_envs), self.num_threads)
            futures = [
                self.executor.submit(self._step_subrange, chunk, target_positions)
                for chunk in chunks
                if len(chunk) > 0
            ]
            for f in futures:
                f.result()
        else:
            self._step_subrange(np.arange(self.num_envs), target_positions)

        self.episode_steps += 1

        # 4. Extract observations, compute rewards, and check done flags
        if self.use_nav_planner:
            self.update_nav_commands()

        next_obs = np.empty((self.num_envs, self.obs_dim), dtype=np.float32)
        for i in range(self.num_envs):
            next_obs[i] = self._extract_obs(i)

        # Vectorized extraction of kinodynamic tensors
        base_lin_vel_world = np.array([d.qvel[0:3] for d in self.datas], dtype=np.float32)
        base_quat = np.array([d.qpos[3:7] for d in self.datas], dtype=np.float32)
        # Crucial: Project linear velocity from World Frame into Body Frame!
        base_lin_vel_body = quat_rotate_inverse(base_quat, base_lin_vel_world)

        cmd_lin_vel = np.zeros((self.num_envs, 3), dtype=np.float32)
        cmd_lin_vel[:, :2] = self.commands[:, :2]

        base_ang_vel_raw = np.array([d.sensordata[3:6] for d in self.datas], dtype=np.float32)
        # MuJoCo <gyro> sensor is already mounted on torso in body frame matching _extract_obs.
        base_ang_vel_body = base_ang_vel_raw
        cmd_ang_vel = self.commands[:, 2]
        if self.imu_mount_max_rad > 0.0:
            proj_grav = np.array(
                [
                    [
                        2.0 * (d.sensordata[6] * d.sensordata[8] - d.sensordata[7] * d.sensordata[9]),
                        -2.0 * (d.sensordata[6] * d.sensordata[7] + d.sensordata[8] * d.sensordata[9]),
                        2.0 * (d.sensordata[7] * d.sensordata[7] + d.sensordata[8] * d.sensordata[8]) - 1.0,
                    ]
                    for d in self.datas
                ],
                dtype=np.float32,
            )
        else:
            proj_grav = next_obs[:, 3:6]
        joint_torques = np.array([d.ctrl[:] for d in self.datas], dtype=np.float32)
        joint_vel = np.array([d.sensordata[20:28] for d in self.datas], dtype=np.float32)
        joint_pos = np.array([d.sensordata[12:20] for d in self.datas], dtype=np.float32)
        foot_contact = next_obs[:, 33:37]
        lidar_ranges = next_obs[:, 37:42] if self.base_obs_dim == 42 else None
        qpos_head = np.array([d.sensordata[12:14] for d in self.datas], dtype=np.float32)

        current_yaw = quat_to_yaw(base_quat)
        _, clock_sin, clock_cos = compute_gait_phase(
            self.episode_steps, dt_control=self.control_dt, period=0.8
        )

        # Compute obstacle distances and approach velocity
        if not self._has_active_obstacles():
            curr_d_min = None
        elif lidar_ranges is not None:
            curr_d_min = np.min(lidar_ranges, axis=-1)
        else:
            curr_d_min = self._compute_d_min()

        if curr_d_min is not None:
            self.d_min[:] = curr_d_min
            raw_dot = (curr_d_min - self.prev_d_min) / self.control_dt
            invalid_trans = (
                (self.prev_d_min <= 0.0)
                | (self.prev_d_min >= 1.99)
                | (curr_d_min >= 1.99)
            )
            self.d_min_dot[:] = np.where(invalid_trans, 0.0, np.clip(raw_dot, -1.0, 1.0))
            self.prev_d_min[:] = curr_d_min
            reward_d_min = self.d_min
            reward_d_min_dot = self.d_min_dot
        else:
            self.d_min[:] = 0.0
            self.d_min_dot[:] = 0.0
            self.prev_d_min[:] = 0.0
            reward_d_min = None
            reward_d_min_dot = None

        # Extract foot kinematics and 3D contact forces for clearance and stumble rewards
        foot_z, foot_vel_xy, foot_contact_forces = self._extract_foot_dynamics()
        root_x_f64 = np.array([d.qpos[0] for d in self.datas], dtype=np.float64)
        vx_world_after = np.array([d.qvel[0] for d in self.datas], dtype=np.float64)
        root_x = root_x_f64.astype(np.float32)
        root_y = np.array([d.qpos[1] for d in self.datas], dtype=np.float32)
        root_z = np.array([d.qpos[2] for d in self.datas], dtype=np.float32)

        # World-X displacement consistency check (against world-X velocity, before clipping)
        raw_dx = root_x_f64 - self.prev_root_x
        mean_world_vx = 0.5 * (vx_world_before + vx_world_after)
        finite_disp = (
            pre_step_finite
            & np.isfinite(self.prev_root_x)
            & np.isfinite(root_x_f64)
            & np.isfinite(vx_world_before)
            & np.isfinite(vx_world_after)
            & np.isfinite(raw_dx)
        )
        vel_diff = np.abs(raw_dx / max(self.control_dt, 1e-8) - mean_world_vx)
        vel_tol = 0.05 + 0.25 * np.abs(mean_world_vx)
        valid_disp = finite_disp & (vel_diff <= vel_tol)
        max_dx = 0.02 * (self.control_dt / 0.02)
        accepted_dx = np.where(valid_disp, np.clip(raw_dx, -max_dx, max_dx), 0.0).astype(np.float32)
        invalid_disp_mask = ~valid_disp
        invalid_displacement_count = int(np.sum(invalid_disp_mask))
        self.prev_root_x[:] = root_x_f64
        self.prev_world_vx[:] = vx_world_after

        # Feet air time tracking: left foot (0) and right foot (1)
        curr_contact = (foot_contact[:, :2] > 0.5)
        touchdown = curr_contact & (~self.last_feet_contact)
        self.feet_air_time[~curr_contact] += self.control_dt

        sole_clearance = self._compute_sole_clearance()
        active_walk = (
            (self.commands[:, 0] > 0.05)
            & (self.cmd_modes != int(LocomotionMode.SIT))
            & (self.cmd_modes != int(LocomotionMode.LIE))
        )
        swing_valid = update_gait_hysteresis_state(
            active_walk=active_walk,
            body_vx=base_lin_vel_body[:, 0],
            accepted_dx=accepted_dx,
            curr_contact=curr_contact,
            sole_clearance=sole_clearance,
            standstill_time=self.standstill_time,
            progress_streak=self.progress_streak,
            double_support_time=self.double_support_time,
            swing_clear_time=self.swing_clear_time,
            swing_clear_steps=self.swing_clear_steps,
            dt=self.control_dt,
        )

        if self.use_nav_planner:
            goal_xy = np.asarray(self.nav_goal_xy, dtype=np.float32)[:2]
            dx_g = goal_xy[0] - root_x
            dy_g = goal_xy[1] - root_y
            dist_g = np.hypot(dx_g, dy_g)
            goal_tol = max(float(getattr(getattr(self.nav_planner, "cfg", None), "goal_tol", 0.15)), 1e-6)
            target_yaw = np.where(dist_g > goal_tol, np.arctan2(dy_g, dx_g), current_yaw).astype(np.float32)
        else:
            target_yaw = np.zeros(self.num_envs, dtype=np.float32)

        if self.enable_head_ema_penalty:
            target_head = (
                EMOTION_TARGET_HEAD_ARRAY[self.current_emotions]
                if self.include_emotions
                else np.zeros((self.num_envs, 2), dtype=np.float32)
            )
            ema_alpha = min(1.0, float(self.control_dt))
            self.head_err_ema = (
                (1.0 - ema_alpha) * self.head_err_ema + ema_alpha * (qpos_head - target_head)
            ).astype(np.float32)

        # Vectorized reward computation across all environments
        reward_dict, weighted_terms = compute_rewards(
            base_lin_vel=base_lin_vel_body,
            cmd_lin_vel=cmd_lin_vel,
            base_ang_vel=base_ang_vel_body,
            cmd_ang_vel=cmd_ang_vel,
            proj_gravity=proj_grav,
            joint_torques=joint_torques,
            joint_vel=joint_vel,
            foot_contact=foot_contact,
            lidar_ranges=lidar_ranges,
            actions=actions,
            last_actions=self.last_actions,
            weights=self.cfg.REWARD_WEIGHTS,
            d_min=reward_d_min,
            d_min_dot=reward_d_min_dot,
            foot_z=foot_z,
            foot_vel_xy=foot_vel_xy,
            foot_contact_forces=foot_contact_forces,
            root_x=root_x,
            root_y=root_y,
            root_z=root_z,
            terrain_mode=self.terrain_mode,
            feet_air_time=self.feet_air_time,
            feet_touchdown=touchdown,
            current_emotion=self.current_emotions if self.include_emotions else None,
            qpos_head=qpos_head,
            joint_pos=joint_pos,
            cmd_mode=self.cmd_modes,
            double_support_time=self.double_support_time,
            last_joint_vel=self.last_joint_vel,
            current_yaw=current_yaw,
            target_yaw=target_yaw,
            clock_sin=clock_sin,
            clock_cos=clock_cos,
            delta_root_x=accepted_dx,
            standstill_time=self.standstill_time,
            swing_valid=swing_valid,
            return_components=True,
            posture_blend=(self.posture_blend if self.posture_ramp_s > 0.0 else None),
            multiplicative_posture_reward=self.multiplicative_posture_reward,
            head_err_ema=(self.head_err_ema if self.enable_head_ema_penalty else None),
            action_over_limit_threshold=self.action_over_limit_threshold,
        )
        rewards = reward_dict["total_reward"]
        self.last_joint_vel[:] = joint_vel

        # Reset air time for feet in contact and update last contact history
        self.feet_air_time[curr_contact] = 0.0
        self.last_feet_contact[:] = curr_contact

        # Vectorized termination check: fall over, excessive tilt, numerical divergence, or timeout
        grav_z = proj_grav[:, 2]
        diverged = (~pre_step_finite) | np.array([
            (not np.all(np.isfinite(d.qpos))) or (not np.all(np.isfinite(d.qvel)))
            for d in self.datas
        ], dtype=bool)

        timeouts = self.episode_steps >= self.max_episode_steps

        is_walk = (self.cmd_modes == int(LocomotionMode.WALK))
        is_sit = (self.cmd_modes == int(LocomotionMode.SIT))
        is_lie = (self.cmd_modes == int(LocomotionMode.LIE))

        is_sitting_legs = (joint_pos[:, 4] > 0.9) | (joint_pos[:, 7] > 0.9)
        walk_moving = is_walk & (self.episode_steps > 1) & (base_lin_vel_body[:, 0] >= 0.03) & (~is_sitting_legs)
        configured_min_z = getattr(self, "walk_min_z", 0.095)
        sit_collapse_z = np.where(is_walk & is_sitting_legs, 0.125, configured_min_z)
        walk_z_thresh = np.where(walk_moving, min(0.085, configured_min_z), sit_collapse_z)
        z_fall_threshold = np.where(
            is_lie,
            0.040,
            np.where(is_sit, 0.080, walk_z_thresh),
        )

        tilt_thresh = np.where(is_lie, 0.0, -0.707)
        exceeding_tilt = (grav_z > tilt_thresh)
        severe_tilt = (grav_z > -0.50)

        self.tilt_exceed_count = np.where(exceeding_tilt, self.tilt_exceed_count + 1, 0)
        tilt_fallen = exceeding_tilt & (
            severe_tilt
            | (self.episode_steps <= 1)
            | (~is_walk)
            | (self.tilt_exceed_count >= 2)
        )

        z_fallen = (root_z < z_fall_threshold)

        # Stage 0 heading yaw termination in WALK mode (when straight walking is commanded)
        yaw_fallen = np.zeros(self.num_envs, dtype=bool)
        if self.curriculum_stage == 0:
            straight_cmd = np.abs(self.commands[:, 2]) < 0.05
            yaw_err = np.arctan2(np.sin(current_yaw - target_yaw), np.cos(current_yaw - target_yaw))
            yaw_fallen = is_walk & straight_cmd & (np.abs(yaw_err) > 1.0)

        fallen = z_fallen | tilt_fallen | yaw_fallen | diverged

        edge_fallen = np.zeros(self.num_envs, dtype=bool)
        if self.terrain_mode == "stairs":
            stair_cfg = STAIR_STAGES[max(0, min(self.curriculum_stage, len(STAIR_STAGES) - 1))]
            half_w = getattr(self, "stair_half_w", stair_cfg.stair_width / 2.0)
            min_x = getattr(self, "stair_min_x", 0.08)
            max_x = getattr(self, "stair_max_x", 1.10)
            in_stair_x = (root_x >= min_x) & (root_x <= max_x)
            edge_fallen = in_stair_x & (np.abs(root_y) > (half_w + 0.04))
            fallen = fallen | edge_fallen

        corridor_fallen = np.zeros(self.num_envs, dtype=bool)
        if self.terrain_mode == "hurdles":
            in_pillar_zone = (root_x >= 0.20) & (root_x <= 1.20)
            corridor_fallen = in_pillar_zone & (np.abs(root_y) > 0.55)
            fallen = fallen | corridor_fallen

        if self.enable_auto_recovery:
            for i in range(self.num_envs):
                ctrl = self.recovery_controllers[i]
                if diverged[i]:
                    fallen[i] = True
                elif fallen[i]:
                    if ctrl.state == SelfRightingController.IDLE:
                        fall_type = FallClassifier.classify(root_z[i], proj_grav[i], self.cmd_modes[i])
                        ctrl.trigger(fall_type, self.datas[i].qpos[7:15])
                        fallen[i] = False
                    elif ctrl.is_recovering:
                        fallen[i] = False
                    elif ctrl.state == SelfRightingController.DONE:
                        fallen[i] = False
                        ctrl.reset()
                    elif ctrl.state == SelfRightingController.FAILED_LOCKED:
                        fallen[i] = True
                else:
                    if ctrl.state == SelfRightingController.DONE:
                        ctrl.reset()

        fall_penalty = float(self.cfg.REWARD_WEIGHTS.get("fall_penalty", -5.0))
        fall_penalty_vec = np.where(fallen, fall_penalty, 0.0).astype(np.float32)
        if np.any(fallen):
            rewards = (rewards + fall_penalty_vec).astype(np.float32)
        weighted_terms["fall_penalty"] = fall_penalty_vec

        dones = fallen | timeouts
        time_outs = timeouts & ~fallen

        self.last_actions[:] = actions
        # Rewards above use the previous action. The next (including terminal)
        # observation belongs to the action just executed; avoid another sensor pass.
        next_obs[:, 25:33] = actions

        delta_root_x_mean = float(np.mean(accepted_dx))
        standstill_time_mean = float(np.mean(self.standstill_time))
        terminal_distances_x = root_x.copy()

        # Pre-impact fall predictor update (updates limp gain scaling for next control step)
        if self.enable_limp_fall:
            for i in range(self.num_envs):
                self.limp_active[i] = self.fall_predictors[i].update(
                    base_ang_vel_body[i], proj_grav[i], imu_ready=True
                )

        if fast:
            if np.any(time_outs):
                timeout_indices = np.where(time_outs)[0]
                terminal_obs = next_obs[time_outs].copy()
                terminal_critic_obs = self.get_privileged_obs(raw_obs=next_obs, env_indices=timeout_indices)
            else:
                terminal_obs = np.empty((0, self.obs_dim), dtype=np.float32)
                terminal_critic_obs = np.empty((0, self.privileged_obs_dim), dtype=np.float32)

            reward_components = {k: float(np.mean(v)) for k, v in weighted_terms.items()}

            if np.any(dones):
                for i in np.where(dones)[0]:
                    self.completed_ep_lens.append(int(self.episode_steps[i]))
                    self.completed_distances_x.append(float(root_x[i]))
                    self.completed_distances_y.append(float(root_y[i]))
                    next_obs[i] = self.reset_env(i)
                if self.num_envs > 1 and np.all(dones):
                    self._randomize_model_dynamics()
            fast_info = {
                "time_outs": time_outs,
                "tracking_lin_vel": float(np.mean(reward_dict["tracking_lin_vel"])) if "tracking_lin_vel" in reward_dict else 0.0,
                "terminal_obs": terminal_obs,
                "terminal_critic_obs": terminal_critic_obs,
                "reward_components": reward_components,
                "delta_root_x_mean": delta_root_x_mean,
                "standstill_time_mean": standstill_time_mean,
                "invalid_displacement_count": invalid_displacement_count,
                "terminal_distances_x": terminal_distances_x,
            }
            return next_obs, rewards, dones, fast_info

        # Maintain per-env info dictionary list for diagnostic logging / evaluation
        infos: List[Dict[str, Any]] = [
            {k: float(v[i]) for k, v in reward_dict.items()}
            for i in range(self.num_envs)
        ]
        active_obstacle_gids = {
            gid for gid in self.obstacle_geom_ids if float(self.model.geom_pos[gid, 2]) > 0.0
        }
        for i in range(self.num_envs):
            infos[i]["timeout"] = bool(timeouts[i])
            infos[i]["time_outs"] = bool(time_outs[i])
            infos[i]["edge_fallen"] = bool(edge_fallen[i])
            infos[i]["corridor_fallen"] = bool(corridor_fallen[i])
            infos[i]["yaw_fallen"] = bool(yaw_fallen[i])
            infos[i]["tilt_fallen"] = bool(tilt_fallen[i])
            infos[i]["fallen"] = bool(fallen[i])
            infos[i]["distance_x"] = float(root_x[i])
            infos[i]["episode_distance_y"] = float(root_y[i])
            infos[i]["yaw"] = float(current_yaw[i])
            infos[i]["d_min"] = float(self.d_min[i]) if curr_d_min is not None else None
            infos[i]["obstacle_contact"] = bool(has_obstacle_contact(self.datas[i], active_obstacle_gids))
            infos[i]["delta_root_x"] = float(accepted_dx[i])
            infos[i]["standstill_time"] = float(self.standstill_time[i])
            infos[i]["invalid_displacement"] = bool(invalid_disp_mask[i])
            infos[i]["swing_valid"] = swing_valid[i].copy()
            infos[i]["step_ctrl"] = joint_torques[i].copy()
            infos[i]["step_qvel"] = np.array(self.datas[i].qvel[6:14], dtype=np.float32).copy()
            infos[i]["reward_components"] = {k: float(v[i]) for k, v in weighted_terms.items()}
            if self.enable_auto_recovery:
                infos[i]["recovering"] = bool(self.recovery_controllers[i].is_recovering)
                infos[i]["recovery_state"] = int(self.recovery_controllers[i].state)
                infos[i]["assisted_recovery"] = bool(assisted_recovery_flags[i])

        if np.any(dones):
            for i in np.where(dones)[0]:
                self.completed_ep_lens.append(int(self.episode_steps[i]))
                self.completed_distances_x.append(float(root_x[i]))
                self.completed_distances_y.append(float(root_y[i]))
                infos[i]["episode_length"] = int(self.episode_steps[i])
                infos[i]["distance_x"] = float(root_x[i])
                infos[i]["episode_distance_y"] = float(root_y[i])
                infos[i]["terminal_obs"] = next_obs[i].copy()
                infos[i]["terminal_critic_obs"] = self.get_privileged_obs(raw_obs=next_obs[i:i+1], env_indices=[i])[0]
                next_obs[i] = self.reset_env(i)
            if self.num_envs > 1 and np.all(dones):
                self._randomize_model_dynamics()

        return next_obs, rewards, dones, infos

    def step_arrays(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Fast vectorized step returning NumPy arrays directly without per-env dict allocations."""
        return self.step(actions, fast=True)

    def get_mean_episode_length(self) -> float:
        """Return average episode length from completed episodes window (or current steps if empty)."""
        if len(self.completed_ep_lens) > 0:
            return float(np.mean(self.completed_ep_lens))
        return float(self.episode_steps.mean())

    def get_mean_distance_x(self) -> float:
        """Return average forward distance along X from completed episodes window (or current positions if empty)."""
        if len(self.completed_distances_x) > 0:
            return float(np.mean(self.completed_distances_x))
        return float(np.mean([d.qpos[0] for d in self.datas]))

    def get_mean_distance_y(self) -> float:
        """Return average lateral distance along Y from completed episodes window (or current positions if empty)."""
        if len(self.completed_distances_y) > 0:
            return float(np.mean(self.completed_distances_y))
        return float(np.mean([d.qpos[1] for d in self.datas]))

    def get_success_rate(self, min_dist: float = 1.5) -> float:
        """Return proportion of completed episodes that achieved at least min_dist along X."""
        if len(self.completed_distances_x) > 0:
            return float(sum(1 for d in self.completed_distances_x if d >= min_dist) / len(self.completed_distances_x))
        if self.num_envs == 0:
            return 0.0
        cur_dists = [float(d.qpos[0]) for d in self.datas]
        return float(sum(1 for d in cur_dists if d >= min_dist) / len(cur_dists))

    def close(self):
        """Clean up thread executor resources."""
        if getattr(self, "executor", None) is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def __del__(self):
        self.close()


DuckParallelSim = ParallelSimRunner


def run_benchmark(
    model_path: Union[str, Path] = MODEL_PATH,
    num_envs: int = 16,
    steps: int = 50,
    num_threads: Optional[int] = None,
    decimation: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run parallel simulation benchmark and calculate FPS metrics."""
    runner = ParallelSimRunner(
        model_path=model_path,
        num_envs=num_envs,
        decimation=decimation,
        num_threads=num_threads,
    )

    try:
        # Warm-up step
        warmup_actions = np.zeros((num_envs, runner.action_dim), dtype=np.float32)
        runner.step(warmup_actions)

        # Generate action batches for benchmark
        rng = np.random.default_rng(seed=42)

        start_time = time.perf_counter()
        for _ in range(steps):
            step_actions = rng.uniform(-1.0, 1.0, size=(num_envs, runner.action_dim)).astype(np.float32)
            runner.step(step_actions)
        elapsed_time = time.perf_counter() - start_time

        total_steps = num_envs * steps
        fps = total_steps / elapsed_time if elapsed_time > 0 else 0.0
        # Real-time factor = total simulated environment time / elapsed real time
        real_time_factor = (total_steps * runner.control_dt) / elapsed_time if elapsed_time > 0 else 0.0

        metrics = {
            "num_envs": num_envs,
            "steps": steps,
            "total_steps": total_steps,
            "elapsed_time": elapsed_time,
            "fps": fps,
            "real_time_factor": real_time_factor,
            "sim_dt": runner.sim_dt,
            "control_dt": runner.control_dt,
            "decimation": runner.decimation,
            "threads": runner.num_threads,
        }

        if verbose:
            print("=" * 60)
            print("Parallel Simulation Benchmark: CyberDuck V2")
            print("=" * 60)
            print(f"Model: {model_path}")
            print(f"Environments: {num_envs}")
            print(f"Steps per Env: {steps}")
            print(f"Total Steps: {total_steps}")
            print(
                f"Decimation: {runner.decimation} (Physics dt: {runner.sim_dt:.4f}s, Control dt: {runner.control_dt:.4f}s)"
            )
            print(f"Threads: {runner.num_threads}")
            print(f"Elapsed Time: {elapsed_time:.4f}s")
            print(f"Steps Per Second (FPS): {fps:.2f}")
            print(f"Real-time Factor: {real_time_factor:.2f}x")
            print("=" * 60)

        return metrics
    finally:
        runner.close()


def main():
    """Command-line interface for parallel simulation runner and benchmark."""
    parser = argparse.ArgumentParser(
        description="Parallel Batched Simulation Runner & FPS Benchmark for CyberDuck V2"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(MODEL_PATH),
        help="Path to robot MuJoCo MJCF XML model",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Number of parallel environments to simulate",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of simulation steps per environment",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of worker threads (default: auto)",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=10,
        help="Control step decimation factor (substeps)",
    )

    args = parser.parse_args()

    try:
        run_benchmark(
            model_path=args.model_path,
            num_envs=args.num_envs,
            steps=args.steps,
            num_threads=args.threads,
            decimation=args.decimation,
            verbose=True,
        )
        sys.exit(0)
    except Exception as exc:
        print(f"Error during parallel simulation: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
