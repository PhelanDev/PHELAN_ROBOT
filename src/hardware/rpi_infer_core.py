"""Shared Raspberry Pi Inference Core and Actuation Runtime for CyberDuck V2.

Deduplicated common logic between Raspberry Pi 5 and Raspberry Pi Zero 2 W:
- Float32 ONNX Runtime inference engine for 42-dimensional observation vector
- 8-DOF MG996R servo angle-to-PWM mapping for PCA9685 I2C servo controller
- MPU6050 hardware I2C driver with 42Hz DLPF, Mahony AHRS fusion, and impact gating
- Mock drivers for offline testing and desktop simulation
- Rate-limiting, safety joint angle clamping, and 50 Hz real-time control loop
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.env.env_config import (
    MicroDuck8DoFConfig,
    compute_observations,
    scale_actions,
    clip_action_rates,
)
from sim.behaviors.active_head_scanner import ActiveHeadScanner
from sim.behaviors.self_righting import FallPredictor
from hardware.oled_eyes import EyeRenderer, SSD1309Display
from robot.robot_spec import DEFAULT_ONNX_PATH, MODEL_NAME, NUM_OBSERVATIONS

# Optional hardware libraries
try:
    import smbus2 as smbus  # type: ignore
except ImportError:
    try:
        import smbus  # type: ignore
    except ImportError:
        smbus = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def angle_to_pwm(
    angle_rad: float,
    min_rad: float = -1.57,
    max_rad: float = 1.57,
    min_pulse_us: int = 500,
    max_pulse_us: int = 2500,
    neutral_pulse_us: int = 1500,
) -> int:
    """Map joint angle in radians to servo PWM pulse width in microseconds (us).

    Neutral angle 0.0 rad maps to neutral_pulse_us (1500us).
    Servo sensitivity: 2000.0 us span over pi radians (-pi/2 to +pi/2, ~636.62 us/rad).
    Clamps angle safely within [min_rad, max_rad] and output pulse to [min_pulse_us, max_pulse_us].
    Prevents knee jamming on asymmetric joint limits (e.g. [0.0, 1.48] rad).
    """
    clamped_angle = float(np.clip(angle_rad, min_rad, max_rad))
    pulse = neutral_pulse_us + clamped_angle * (2000.0 / np.pi)
    return int(np.clip(np.round(pulse), min_pulse_us, max_pulse_us))


def pack_rpi_observation(
    base_ang_vel: Union[List[float], np.ndarray],
    proj_grav: Union[List[float], np.ndarray],
    commands: Union[List[float], np.ndarray],
    joint_pos: Union[List[float], np.ndarray],
    joint_vel: Union[List[float], np.ndarray],
    last_actions: Union[List[float], np.ndarray],
    foot_contact: Union[List[float], np.ndarray],
    lidar_ranges: Optional[Union[List[float], np.ndarray]] = None,
    phase_clock: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pack raw sensory signals into V2 37-dimensional actor observation vector."""
    del lidar_ranges  # Brow ToF scan feeds ActiveHeadScanner / nav planner, not actor obs.
    return compute_observations(
        base_ang_vel=np.asarray(base_ang_vel, dtype=np.float32),
        proj_grav=np.asarray(proj_grav, dtype=np.float32),
        commands=np.asarray(commands, dtype=np.float32),
        joint_pos=np.asarray(joint_pos, dtype=np.float32),
        joint_vel=np.asarray(joint_vel, dtype=np.float32),
        last_actions=np.asarray(last_actions, dtype=np.float32),
        foot_contact=np.asarray(foot_contact, dtype=np.float32),
        lidar_ranges=None,
        phase_clock=phase_clock,
    )


class MockPCA9685:
    """Mock PCA9685 driver for offline testing and desktop simulation."""

    def __init__(self, bus: int = 1, address: int = 0x40):
        self.bus = bus
        self.address = address
        self.channel_pwms = [1500] * 16
        self.outputs_enabled = True
        print(f"[MockPCA9685] Initialized mock servo controller on I2C bus {bus}, address {hex(address)}")

    def set_pwm_us(self, channel: int, pulse_us: int) -> None:
        """Record commanded pulse width for given channel."""
        if 0 <= channel < 16:
            self.outputs_enabled = True
            self.channel_pwms[channel] = pulse_us

    def disable_all_channels(self) -> None:
        """Cut servo PWM pulses across all 16 channels."""
        self.channel_pwms = [0] * 16
        self.outputs_enabled = False


class HardwarePCA9685:
    """Physical I2C PCA9685 driver via smbus2."""

    PCA9685_MODE1 = 0x00
    PCA9685_PRESCALE = 0xFE
    LED0_ON_L = 0x06
    ALL_LED_ON_L = 0xFA

    def __init__(self, bus_num: int = 1, address: int = 0x40, freq: int = 50):
        self.address = address
        self.freq = freq
        if smbus is None:
            raise RuntimeError("smbus2/smbus module not found. Run with --mock on desktop systems.")
        self.bus = smbus.SMBus(bus_num)

        # Reset PCA9685
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, 0x00)
        time.sleep(0.01)

        # Set PWM frequency (50 Hz for analog/digital hobby servos)
        prescale_val = int(25000000.0 / (4096.0 * freq) - 1.0 + 0.5)
        old_mode = self.bus.read_byte_data(self.address, self.PCA9685_MODE1)
        new_mode = (old_mode & 0x7F) | 0x10  # Sleep mode to set prescale
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, new_mode)
        self.bus.write_byte_data(self.address, self.PCA9685_PRESCALE, prescale_val)
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, old_mode)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, old_mode | 0xA1)  # Auto-increment

    def set_pwm_us(self, channel: int, pulse_us: int) -> None:
        """Convert microsecond pulse to 12-bit PCA9685 tick value (0-4095)."""
        period_us = 1000000.0 / self.freq
        ticks = int((pulse_us / period_us) * 4096.0)
        ticks = max(0, min(4095, ticks))

        reg_base = self.LED0_ON_L + 4 * channel
        data = [0, 0, ticks & 0xFF, (ticks >> 8) & 0xFF]
        self.bus.write_i2c_block_data(self.address, reg_base, data)

    def disable_all_channels(self) -> None:
        """Broadcast FULL_OFF (bit 4 in ALL_LED_OFF_H) to release all MG996R servos safely."""
        self.bus.write_i2c_block_data(self.address, self.ALL_LED_ON_L, [0, 0, 0, 0x10])


class VL53L0XReader:
    """Real I2C driver for VL53L0X single-ray ToF sensor (0x29) with explicit validity state."""

    VL53L0X_ADDR = 0x29
    REG_SYSRANGE_START = 0x00
    REG_RESULT_RANGE_STATUS = 0x14
    REG_RESULT_RANGE_MM = 0x1E

    def __init__(
        self,
        bus_num: int = 1,
        address: int = VL53L0X_ADDR,
        mock: bool = False,
        min_range_m: float = 0.03,
        max_range_m: float = 2.0,
    ):
        self.address = int(address)
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        self.mock = bool(mock) or (smbus is None)
        self.is_valid = bool(self.mock)
        self.last_range_m = 1.5 if self.mock else 0.0
        self.last_error: Optional[Exception] = None
        self.bus = None
        if not self.mock:
            try:
                self.bus = smbus.SMBus(bus_num)
            except (OSError, RuntimeError) as exc:
                self.last_error = exc
                self.is_valid = False

    def read_range_m(self) -> Tuple[float, bool]:
        """Read measured distance in meters and whether hardware reading succeeded."""
        if self.mock:
            self.is_valid = True
            self.last_range_m = 1.5
            return 1.5, True
        if self.bus is None:
            self.is_valid = False
            self.last_range_m = 0.0
            return 0.0, False
        try:
            self.bus.write_byte_data(self.address, self.REG_SYSRANGE_START, 0x01)
            raw = self.bus.read_i2c_block_data(self.address, self.REG_RESULT_RANGE_MM, 2)
            if len(raw) < 2:
                raise ValueError(f"Expected 2 VL53L0X range bytes, got {len(raw)}")
            mm = (int(raw[0]) << 8) | int(raw[1])
            dist_m = float(np.clip(mm * 0.001, self.min_range_m, self.max_range_m))
            self.is_valid = True
            self.last_range_m = dist_m
            self.last_error = None
            return dist_m, True
        except (OSError, RuntimeError, ValueError) as exc:
            self.is_valid = False
            self.last_range_m = 0.0
            self.last_error = exc
            return 0.0, False


class MPU6050FilterReader:
    """Hardware I2C driver for MPU6050 with 42Hz DLPF, Mahony AHRS sensor fusion, and impact gating."""

    MPU6050_ADDR = 0x68
    PWR_MGMT_1 = 0x6B
    CONFIG = 0x1A
    GYRO_CONFIG = 0x1B
    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43

    def __init__(
        self,
        bus_num: int = 1,
        address: int = 0x68,
        mock: bool = False,
        dt: float = 0.02,
        kp: float = 1.0,
        ki: float = 0.05,
        impact_threshold: float = 0.30,
        ready_confirm_samples: int = 10,
        allow_mock_fallback: bool = False,
    ):
        self.address = address
        self.allow_mock_fallback = bool(allow_mock_fallback)
        if smbus is None and not mock and not self.allow_mock_fallback:
            raise RuntimeError("MPU6050 hardware initialization failed: smbus module not available")
        self.mock = bool(mock) or (smbus is None)
        self.dt = float(dt)
        self.kp = float(kp)
        self.ki = float(ki)
        self.impact_threshold = float(impact_threshold)
        self.ready_confirm_samples = max(1, int(ready_confirm_samples))
        self.valid_startup_samples = 0

        # Mahony AHRS quaternion [qw, qx, qy, qz] & integral feedback error
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.integral_fb = np.zeros(3, dtype=np.float32)

        self.gyro_bias = np.zeros(3, dtype=np.float32)
        self.filt_gyro = np.zeros(3, dtype=np.float32)
        self.filt_grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.alpha_gyro = 0.70
        self.alpha_grav = 0.80
        self.is_stale = False
        self.last_read_ok = True
        self.consecutive_read_errors = 0
        self.last_read_error: Optional[Exception] = None

        if not self.mock:
            try:
                self.bus = smbus.SMBus(bus_num)
                # Wake up MPU6050
                self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0x00)
                # 42Hz hardware DLPF cutoff to filter servo motor vibration
                self.bus.write_byte_data(self.address, self.CONFIG, 0x03)
                # Gyro sensitivity +-250 deg/s
                self.bus.write_byte_data(self.address, self.GYRO_CONFIG, 0x00)
                self._calibrate_gyro()
            except (OSError, RuntimeError) as e:
                if self.allow_mock_fallback:
                    print(f"[IMU] Real I2C MPU6050 unavailable ({e}), using mock/fallback.")
                    self.mock = True
                else:
                    raise RuntimeError(f"MPU6050 hardware initialization failed: {e}") from e

    @property
    def imu_ready(self) -> bool:
        """Return True once IMU has emitted enough finite stationary 1g updates to converge."""
        return int(getattr(self, "valid_startup_samples", 0)) >= int(getattr(self, "ready_confirm_samples", 10))

    def reset_readiness(self) -> None:
        """Reset startup convergence counter to gate fall detection during initialization."""
        self.valid_startup_samples = 0

    @staticmethod
    def _decode_int16_be(hi: int, lo: int) -> int:
        if isinstance(hi, bool) or isinstance(lo, bool) or not isinstance(hi, (int, np.integer)) or not isinstance(lo, (int, np.integer)):
            raise ValueError("I2C register bytes must be integers")
        hi_i, lo_i = int(hi), int(lo)
        if not (0 <= hi_i <= 255 and 0 <= lo_i <= 255):
            raise ValueError("I2C register byte out of [0, 255] range")
        val = (hi_i << 8) | lo_i
        return val - 0x10000 if (val & 0x8000) else val

    def _calibrate_gyro(self, samples: int = 100) -> None:
        """Calibrate gyro zero-rate offset while standing still."""
        print("[IMU] Calibrating MPU6050 gyro bias...")
        acc = np.zeros(3, dtype=np.float64)
        count = 0
        for _ in range(samples):
            try:
                data = self.bus.read_i2c_block_data(self.address, self.GYRO_XOUT_H, 6)
            except OSError:
                continue
            if len(data) < 6:
                raise ValueError(f"Expected 6 gyro calibration bytes, got {len(data)}")
            gx = (self._decode_int16_be(data[0], data[1]) / 131.0) * (np.pi / 180.0)
            gy = (self._decode_int16_be(data[2], data[3]) / 131.0) * (np.pi / 180.0)
            gz = (self._decode_int16_be(data[4], data[5]) / 131.0) * (np.pi / 180.0)
            acc += [gx, gy, gz]
            count += 1
            time.sleep(0.002)
        if samples > 0 and count == 0:
            raise RuntimeError("Failed to read any valid MPU6050 gyro calibration samples")
        if count > 0:
            self.gyro_bias = (acc / count).astype(np.float32)

    def update(
        self,
        ax: float,
        ay: float,
        az: float,
        gx: float,
        gy: float,
        gz: float,
        dt: Optional[float] = None,
    ) -> None:
        """Vectorized Mahony AHRS update with dynamic impact gating."""
        if dt is None:
            dt = self.dt

        acc = np.array([ax, ay, az], dtype=np.float32)
        acc_sq = float(np.sum(acc * acc))

        corr_gx = float(gx)
        corr_gy = float(gy)
        corr_gz = float(gz)

        if acc_sq > 1e-8:
            acc_norm = float(np.sqrt(acc_sq))
            # Dynamic Foot-Strike Impact Gating: gate correction if |norm - 1.0g| > 0.30g
            if abs(acc_norm - 1.0) <= self.impact_threshold:
                confirm = int(getattr(self, "ready_confirm_samples", 10))
                cur_samples = int(getattr(self, "valid_startup_samples", 0))
                if cur_samples < confirm and np.all(np.isfinite([ax, ay, az, gx, gy, gz])):
                    self.valid_startup_samples = cur_samples + 1
                inv_norm = 1.0 / acc_norm
                ax_n = acc[0] * inv_norm
                ay_n = acc[1] * inv_norm
                az_n = acc[2] * inv_norm

                qw, qx, qy, qz = self.q
                # Estimated gravity direction in body frame
                vx = 2.0 * (qx * qz - qw * qy)
                vy = 2.0 * (qw * qx + qy * qz)
                vz = qw * qw - qx * qx - qy * qy + qz * qz

                # Error is cross product between estimated and measured direction of gravity
                ex = ay_n * vz - az_n * vy
                ey = az_n * vx - ax_n * vz
                ez = ax_n * vy - ay_n * vx

                # Accumulate integral feedback
                self.integral_fb[0] += self.ki * ex * dt
                self.integral_fb[1] += self.ki * ey * dt
                self.integral_fb[2] += self.ki * ez * dt

                # Apply proportional and integral feedback to angular rates
                corr_gx += self.kp * ex + self.integral_fb[0]
                corr_gy += self.kp * ey + self.integral_fb[1]
                corr_gz += self.kp * ez + self.integral_fb[2]
            else:
                # Gated: use integral feedback only (no accelerometer correction)
                corr_gx += self.integral_fb[0]
                corr_gy += self.integral_fb[1]
                corr_gz += self.integral_fb[2]
        else:
            corr_gx += self.integral_fb[0]
            corr_gy += self.integral_fb[1]
            corr_gz += self.integral_fb[2]

        # Integrate rate of change of quaternion: q_dot = 0.5 * q * omega
        hdt = 0.5 * dt
        corr_gx *= hdt
        corr_gy *= hdt
        corr_gz *= hdt

        qw, qx, qy, qz = self.q
        dq_w = -qx * corr_gx - qy * corr_gy - qz * corr_gz
        dq_x =  qw * corr_gx + qy * corr_gz - qz * corr_gy
        dq_y =  qw * corr_gy - qx * corr_gz + qz * corr_gx
        dq_z =  qw * corr_gz + qx * corr_gy - qy * corr_gx

        new_q = np.array([qw + dq_w, qx + dq_x, qy + dq_y, qz + dq_z], dtype=np.float32)
        q_norm = float(np.linalg.norm(new_q))
        if q_norm > 1e-8:
            self.q = new_q / q_norm
        else:
            self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.filt_grav = self.get_proj_grav()

    mahony_update = update

    def get_proj_grav(self) -> np.ndarray:
        """Compute projected gravity vector in robot base frame matching MuJoCo convention [0, 0, -1]."""
        qw, qx, qy, qz = self.q
        return np.array([
            -2.0 * (qx * qz - qw * qy),
            -2.0 * (qy * qz + qw * qx),
            -(qw * qw - qx * qx - qy * qy + qz * qz),
        ], dtype=np.float32)

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read and return filtered (base_ang_vel, proj_grav)."""
        if self.mock:
            self.is_stale = False
            self.last_read_ok = True
            self.consecutive_read_errors = 0
            self.last_read_error = None
            confirm = int(getattr(self, "ready_confirm_samples", 10))
            cur_samples = int(getattr(self, "valid_startup_samples", 0))
            if cur_samples < confirm:
                self.valid_startup_samples = cur_samples + 1
            return self.filt_gyro.copy(), self.get_proj_grav()

        try:
            data = self.bus.read_i2c_block_data(self.address, self.ACCEL_XOUT_H, 14)
        except OSError as exc:
            self.is_stale = True
            self.last_read_ok = False
            self.consecutive_read_errors = int(getattr(self, "consecutive_read_errors", 0)) + 1
            self.last_read_error = exc
            return self.filt_gyro.copy(), self.get_proj_grav()

        if len(data) < 14:
            raise ValueError(f"Expected 14 MPU6050 bytes, got {len(data)}")

        ax_raw = self._decode_int16_be(data[0], data[1])
        ay_raw = self._decode_int16_be(data[2], data[3])
        az_raw = self._decode_int16_be(data[4], data[5])
        gx_raw = self._decode_int16_be(data[8], data[9])
        gy_raw = self._decode_int16_be(data[10], data[11])
        gz_raw = self._decode_int16_be(data[12], data[13])

        raw_gx = ((gx_raw / 131.0) * (np.pi / 180.0)) - self.gyro_bias[0]
        raw_gy = ((gy_raw / 131.0) * (np.pi / 180.0)) - self.gyro_bias[1]
        raw_gz = ((gz_raw / 131.0) * (np.pi / 180.0)) - self.gyro_bias[2]

        self.filt_gyro = self.alpha_gyro * self.filt_gyro + (1.0 - self.alpha_gyro) * np.array(
            [raw_gx, raw_gy, raw_gz], dtype=np.float32
        )

        ax = float(ax_raw) / 16384.0
        ay = float(ay_raw) / 16384.0
        az = float(az_raw) / 16384.0

        self.update(ax, ay, az, self.filt_gyro[0], self.filt_gyro[1], self.filt_gyro[2], dt=self.dt)
        self.is_stale = False
        self.last_read_ok = True
        self.consecutive_read_errors = 0
        self.last_read_error = None

        return self.filt_gyro.copy(), self.get_proj_grav()


class BaseRPiInferenceEngine:
    """Base Real-Time Locomotion Engine for Raspberry Pi (RPi 5 & Pi Zero 2 W)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        i2c_bus: int = 1,
        mock: bool = False,
        freq: float = 50.0,
        engine_name: str = "BaseRPiInferenceEngine",
        legacy_observation_schema: Optional[dict] = None,
        deadman_timeout_s: float = 0.50,
        standup_model_path: Optional[str] = None,
        allow_mock_fallback: bool = False,
    ):
        self.engine_name = engine_name
        self.cfg = MicroDuck8DoFConfig(OBS_DIM=NUM_OBSERVATIONS)
        self.freq = float(freq)
        if not np.isfinite(self.freq) or self.freq <= 0:
            raise ValueError("Control frequency must be finite and positive")
        self.dt = 1.0 / self.freq
        self.observation_schema = None
        self.episode_steps = 0
        self.allow_mock_fallback = bool(allow_mock_fallback)
        if smbus is None and not mock and not self.allow_mock_fallback:
            raise RuntimeError("PCA9685 / I2C hardware initialization failed: smbus module not available")
        self.mock = bool(mock) or (smbus is None)
        self.deadman_timeout_s = max(0.0, float(deadman_timeout_s))
        self.deadman_tripped = False
        self._active_command = [0.0, 0.0, 0.0]
        self._last_cmd_timestamp: Optional[float] = None
        self.fall_predictor = FallPredictor()
        self.safety_mode = "initializing"
        self.nonfinite_action_drops = 0
        self.head_scanner = ActiveHeadScanner()
        self.eye_renderer = EyeRenderer()
        self.oled_display = SSD1309Display(mock=self.mock)

        # Servo controller setup
        if self.mock:
            self.servo_controller = MockPCA9685(bus=i2c_bus)
        else:
            try:
                self.servo_controller = HardwarePCA9685(bus_num=i2c_bus, freq=int(self.freq))
            except Exception as e:
                if self.allow_mock_fallback:
                    print(f"[Warning] Failed to initialize hardware I2C ({e}). Falling back to Mock PCA9685.")
                    self.mock = True
                    self.servo_controller = MockPCA9685(bus=i2c_bus)
                else:
                    raise RuntimeError(f"PCA9685 hardware initialization failed: {e}") from e

        # IMU sensor & filter setup (fail closed unless mock=True or explicit opt-in fallback)
        self.imu_reader = MPU6050FilterReader(
            bus_num=i2c_bus,
            mock=self.mock,
            dt=self.dt,
            allow_mock_fallback=self.allow_mock_fallback,
        )
        self.tof_reader = VL53L0XReader(bus_num=i2c_bus, mock=self.mock)
        self.last_lidar_valid = bool(self.mock)

        # ONNX Model setup (walking policy + optional hot-swappable standup recovery policy)
        self.model_path = str(model_path) if model_path is not None else (None if self.mock else str(DEFAULT_ONNX_PATH))
        self.ort_session, self.input_name, self.observation_schema = self._load_onnx_policy(
            self.model_path,
            legacy_observation_schema=legacy_observation_schema,
            require_identity=(not self.mock),
            label="walking",
        )
        self.standup_model_path = str(standup_model_path) if standup_model_path is not None else None
        if self.standup_model_path is not None:
            self.standup_ort_session, self.standup_input_name, _ = self._load_onnx_policy(
                self.standup_model_path,
                legacy_observation_schema=legacy_observation_schema,
                require_identity=(not self.mock),
                label="standup",
            )
        else:
            self.standup_ort_session = None
            self.standup_input_name = None

        # State tracking for rate limiting (initialized to nominal stand posture)
        self.last_targets = self.cfg.nominal_qpos.copy()
        self.last_raw_actions = np.zeros(self.cfg.ACTION_DIM, dtype=np.float32)
        self.joint_limits_list = [
            self.cfg.JOINT_LIMITS[name] for name in self.cfg.JOINT_NAMES
        ]
        self.prev_q: np.ndarray = self.cfg.nominal_qpos.copy()
        self._warned_foot_fallback: bool = False
        self._warned_lidar_fallback: bool = False
        self.imu_stale_steps: int = 0
        self.last_imu_stale: bool = False

    def _load_onnx_policy(
        self,
        model_path: Optional[str],
        legacy_observation_schema: Optional[dict] = None,
        require_identity: bool = True,
        label: str = "walking",
    ) -> Tuple[Optional[object], Optional[str], Optional[dict]]:
        """Load and validate an ONNX policy session against V2 identity and 42D schema contracts."""
        if model_path is not None and not Path(model_path).is_file():
            raise FileNotFoundError(
                f"CyberDuck V2 ONNX policy not found: {model_path}; export a verified V2 checkpoint first"
            )
        if not model_path or not Path(model_path).exists():
            if label == "walking":
                print(f"[{self.engine_name}] No ONNX model provided. Running mock zero-action mode.")
            return None, None, None
        if ort is None:
            raise RuntimeError("onnxruntime is required to load ONNX model.")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        session = ort.InferenceSession(model_path, opts)
        input_name = session.get_inputs()[0].name
        from sim.env.policy_contract import validate_model_identity, validate_observation_schema

        metadata = session.get_modelmeta().custom_metadata_map
        if metadata.get("model_name") or metadata.get("model_path"):
            validate_model_identity(metadata, MODEL_NAME)
        elif require_identity:
            raise ValueError("Hardware runtime requires CyberDuck V2 model identity in ONNX metadata")
        raw_schema = metadata.get("observation_schema")
        if raw_schema is None and legacy_observation_schema is None:
            raise ValueError("Legacy policy has no observation_schema; provide an explicit verified legacy schema")
        try:
            schema = validate_observation_schema(
                json.loads(raw_schema) if raw_schema is not None else legacy_observation_schema
            )
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid policy observation_schema") from exc
        if schema["obs_dim"] != NUM_OBSERVATIONS or schema["version"] != 2 or session.get_inputs()[0].shape[-1] != NUM_OBSERVATIONS:
            raise ValueError(f"RPi observation_schema requires {NUM_OBSERVATIONS} observations (version 2)")
        if not np.isclose(schema["control_dt"], self.dt, rtol=1e-7, atol=1e-10):
            raise ValueError("Policy observation_schema control_dt does not match runtime frequency")
        print(f"[{self.engine_name}] Loaded {label} ONNX policy model from {model_path}")
        return session, input_name, schema

    def set_velocity_command(
        self, vx: float, vy: float, wz: float, timestamp: Optional[float] = None
    ) -> None:
        """Update target velocity command [vx, vy, wz] and refresh Deadman watchdog timestamp."""
        self._active_command = [float(vx), float(vy), float(wz)]
        self._last_cmd_timestamp = float(time.monotonic() if timestamp is None else timestamp)
        self.deadman_tripped = False

    def get_active_command(self, now: Optional[float] = None) -> List[float]:
        """Return active velocity twist, zeroing [0, 0, 0] on Deadman timeout without going limp."""
        if self._last_cmd_timestamp is None:
            return list(self._active_command)
        t_now = float(time.monotonic() if now is None else now)
        if (t_now - self._last_cmd_timestamp) > self.deadman_timeout_s:
            self.deadman_tripped = True
            return [0.0, 0.0, 0.0]
        self.deadman_tripped = False
        return list(self._active_command)

    def update_safety_state(self, base_ang_vel: np.ndarray, proj_grav: np.ndarray) -> str:
        """Evaluate IMU readiness and analytical pre-impact fall predictor for limp/standup routing."""
        if not bool(getattr(self.imu_reader, "imu_ready", False)):
            self.safety_mode = "initializing"
            return self.safety_mode
        if self.fall_predictor.update(base_ang_vel, proj_grav, imu_ready=True):
            self.safety_mode = "standup_recovery" if self.standup_ort_session is not None else "limp_fall"
        else:
            self.safety_mode = "walk"
        return self.safety_mode

    def _compute_pwms(self, targets: np.ndarray, dispatch: bool = True) -> np.ndarray:
        """Convert 8D joint targets (rad) to PWM microseconds and optionally dispatch to servos."""
        pwms = np.zeros(self.cfg.ACTION_DIM, dtype=np.int32)
        for i, (angle, limits) in enumerate(zip(targets, self.joint_limits_list)):
            pwm = angle_to_pwm(
                angle_rad=float(angle),
                min_rad=limits[0],
                max_rad=limits[1],
                min_pulse_us=500,
                max_pulse_us=2500,
            )
            pwms[i] = pwm
            if dispatch:
                self.servo_controller.set_pwm_us(channel=i, pulse_us=pwm)
        return pwms

    def reset_policy_state(self):
        """Reset observation history and clock together at an episode boundary."""
        self.episode_steps = 0
        self.last_raw_actions = np.zeros(self.cfg.ACTION_DIM, dtype=np.float32)

    def phase_clock(self):
        if self.observation_schema is None:
            return None
        from sim.env.policy_contract import phase_clock_for_schema
        return phase_clock_for_schema(self.observation_schema, self.episode_steps)

    def read_foot_contact(self) -> List[float]:
        """Read 4-channel foot contact states [Left, Right, Padded, Padded]."""
        if not self._warned_foot_fallback:
            if self.mock:
                print("[Warning] Foot contact sensors running in mock/fallback mode. Using nominal [1.0, 1.0, 0.0, 0.0].")
            else:
                print("[Warning] Hardware foot contact ADC unavailable on I2C. Using fallback contact states.")
            self._warned_foot_fallback = True
        return [1.0, 1.0, 0.0, 0.0]

    def read_lidar_ranges(self) -> List[float]:
        """Read ToF distance sensor (projected to 5 rays for 42D policy or mocked when mock=True)."""
        if self.mock:
            if not self._warned_lidar_fallback:
                print("[Warning] LiDAR sensor array running in mock/fallback mode. Using clear 1.5m range.")
                self._warned_lidar_fallback = True
            self.last_lidar_valid = True
            return [1.5, 1.5, 1.5, 1.5, 1.5]
        dist_m, valid = self.tof_reader.read_range_m()
        self.last_lidar_valid = bool(valid)
        if not valid:
            if not self._warned_lidar_fallback:
                print(f"[Warning] Hardware VL53L0X ToF read failed ({self.tof_reader.last_error}); failing safe to 0.0m.")
                self._warned_lidar_fallback = True
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        return [dist_m] * 5

    def predict_action(self, obs: np.ndarray) -> np.ndarray:
        """Run policy model inference to produce normalized actions [-1, 1]."""
        use_standup = (
            getattr(self, "safety_mode", "walk") in ("limp_fall", "standup_recovery")
            and getattr(self, "standup_ort_session", None) is not None
        )
        session = self.standup_ort_session if use_standup else self.ort_session
        input_name = self.standup_input_name if use_standup else getattr(self, "input_name", None)
        if session is not None and input_name is not None:
            input_tensor = obs.reshape(1, self.cfg.OBS_DIM).astype(np.float32)
            outputs = session.run(None, {input_name: input_tensor})
            return np.asarray(outputs[0], dtype=np.float32).ravel()
        return np.zeros(self.cfg.ACTION_DIM, dtype=np.float32)

    def step(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one control cycle:

        1. Reject non-finite observations/actions by holding last safe targets
        2. Scale actions to joint physical limits
        3. Rate-limit target angles
        4. Map angles to servo PWMs and dispatch to PCA9685
        """
        assert obs.shape == (self.cfg.OBS_DIM,), (
            f"Expected obs shape ({self.cfg.OBS_DIM},), got {obs.shape}"
        )

        if not np.all(np.isfinite(obs)):
            self.nonfinite_action_drops += 1
            safe_targets = self.last_targets.copy()
            return safe_targets, self._compute_pwms(safe_targets, dispatch=False)

        if (
            getattr(self, "safety_mode", "walk") == "limp_fall"
            and getattr(self, "standup_ort_session", None) is None
        ):
            self.servo_controller.disable_all_channels()
            self.episode_steps += 1
            return self.last_targets.copy(), np.zeros(self.cfg.ACTION_DIM, dtype=np.int32)

        # 1. Infer raw actions and reject non-finite vectors BEFORE clipping
        predicted = np.asarray(self.predict_action(obs), dtype=np.float32).ravel()
        if predicted.shape != (self.cfg.ACTION_DIM,) or not np.all(np.isfinite(predicted)):
            self.nonfinite_action_drops += 1
            safe_targets = self.last_targets.copy()
            return safe_targets, self._compute_pwms(safe_targets, dispatch=False)

        raw_actions = np.clip(predicted, -1.0, 1.0)
        self.last_raw_actions = raw_actions.copy()

        # 2. Scale actions to joint angles
        scaled_targets = scale_actions(raw_actions, self.cfg, nominal_centered=True)

        # 3. Rate-limit targets to protect MG996R gears
        clipped_targets = clip_action_rates(
            current_targets=scaled_targets,
            last_targets=self.last_targets,
            dt=self.dt,
            max_velocity=self.cfg.MAX_VELOCITY,
        )
        if not np.all(np.isfinite(clipped_targets)):
            self.nonfinite_action_drops += 1
            safe_targets = self.last_targets.copy()
            return safe_targets, self._compute_pwms(safe_targets, dispatch=False)

        self.last_targets = clipped_targets.copy()

        # 4. Map to PWM signals and dispatch
        pwms = self._compute_pwms(clipped_targets, dispatch=True)

        self.episode_steps += 1
        return clipped_targets, pwms

    def run_loop(self, steps: Optional[int] = None) -> None:
        """Run real-time control loop at specified frequency."""
        print(f"[{self.engine_name}] Starting control loop at {self.freq:.1f} Hz (dt={self.dt*1000:.1f}ms)...")
        step_count = 0
        target_period = self.dt

        commands = [0.2, 0.0, 0.0]  # Walk forward 0.2 m/s
        prev_q = self.last_targets.copy()
        self.prev_q = prev_q.copy()

        try:
            while steps is None or step_count < steps:
                t0 = time.perf_counter()

                # Read filtered IMU angular velocity and gravity vector
                base_ang_vel, proj_grav = self.imu_reader.read()
                self.update_safety_state(base_ang_vel, proj_grav)
                imu_stale = bool(getattr(self.imu_reader, "is_stale", False))
                self.last_imu_stale = imu_stale
                # Read foot contact and brow ToF perception hooks
                foot_contact = self.read_foot_contact()
                lidar_ranges = self.read_lidar_ranges()
                lidar_valid = True if self.mock else bool(getattr(self.tof_reader, "is_valid", False))

                base_commands = (
                    self.get_active_command()
                    if self._last_cmd_timestamp is not None
                    else commands
                )
                if imu_stale or not lidar_valid:
                    if imu_stale:
                        self.imu_stale_steps += 1
                        if self.imu_stale_steps == 1 or self.imu_stale_steps % max(1, int(self.freq)) == 0:
                            print(f"[{self.engine_name} Warning] Stale IMU sample ({self.imu_reader.last_read_error}); zeroing velocity commands.")
                    loop_commands = [0.0, 0.0, 0.0]
                else:
                    loop_commands = base_commands

                # Finite-difference joint velocity: (current_q - prev_q) / dt
                current_q = self.last_targets.copy()
                effective_dt = max(self.dt, 1e-4)
                joint_vel = (current_q - prev_q) / effective_dt
                prev_q = current_q.copy()
                self.prev_q = prev_q.copy()

                if len(lidar_ranges) >= 3:
                    self.head_scanner.update(
                        head_yaw_rad=float(current_q[0]),
                        head_pitch_rad=float(current_q[1]),
                        tof_range_m=float(lidar_ranges[2]),
                        now_s=time.monotonic(),
                    )

                # Generate observation (37D actor contract)
                obs = pack_rpi_observation(
                    base_ang_vel=base_ang_vel,
                    proj_grav=proj_grav,
                    commands=loop_commands,
                    joint_pos=current_q,
                    joint_vel=joint_vel,
                    last_actions=self.last_raw_actions,
                    foot_contact=foot_contact,
                    phase_clock=self.phase_clock(),
                )

                targets, pwms = self.step(obs)
                step_count += 1

                elapsed = time.perf_counter() - t0
                sleep_time = target_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if step_count % max(1, int(self.freq)) == 0 or (steps and step_count == steps):
                    actual_hz = 1.0 / max(1e-6, (time.perf_counter() - t0))
                    print(
                        f"[Step {step_count}] Hz={actual_hz:.1f} | "
                        f"PWMs: {pwms.tolist()} | Targets: {np.round(targets, 2).tolist()}"
                    )

        except KeyboardInterrupt:
            print(f"\n[{self.engine_name}] Control loop interrupted by user.")

        print(f"[{self.engine_name}] Completed {step_count} control steps.")


def run_rpi_cli(engine_cls, description: str, default_model: Optional[str] = None):
    """Generic CLI parser and runner for Raspberry Pi deployment engines."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model_path",
        "--model",
        type=str,
        default=default_model,
        help="Path to Float32 ONNX policy model",
    )
    parser.add_argument("--i2c_bus", type=int, default=1, help="I2C bus number for PCA9685 (default: 1)")
    parser.add_argument("--mock", "--mock-pca", dest="mock", action="store_true", help="Run in mock mode without I2C hardware")
    parser.add_argument(
        "--freq",
        "--rate",
        type=float,
        default=50.0,
        help="Control loop frequency in Hz (default: 50.0)",
    )
    parser.add_argument("--steps", type=int, default=None, help="Number of steps to execute (default: continuous)")
    parser.add_argument("--legacy-observation-schema", type=Path, default=None, help="JSON schema for legacy policy, verified from training provenance")
    args = parser.parse_args()

    engine = engine_cls(
        model_path=args.model_path,
        i2c_bus=args.i2c_bus,
        mock=args.mock,
        freq=args.freq,
        legacy_observation_schema=json.loads(args.legacy_observation_schema.read_text(encoding="utf-8")) if args.legacy_observation_schema else None,
    )
    engine.run_loop(steps=args.steps)
