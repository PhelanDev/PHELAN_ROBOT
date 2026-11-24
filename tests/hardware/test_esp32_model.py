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


def test_esp32_mjcf_structure_and_joints():
    assert_mjcf_structure_and_joints(MJCF_PATH, MJCF_MODEL_NAME)


def test_esp32_mjcf_perception_suite():
    assert_mjcf_perception_suite(MJCF_PATH)


def test_esp32_mujoco_compilation_and_mass_budget():
    assert_mujoco_compilation_and_mass(
        MJCF_PATH,
        min_mass=TOTAL_MASS_KG - 1e-6,
        max_mass=TOTAL_MASS_KG + 1e-6,
        base_mass_min=0.36 - 1e-6,
        base_mass_max=0.36 + 1e-6,
        base_ipos_z_min=-0.012 - 1e-6,
        base_ipos_z_max=-0.012 + 1e-6,
        n_steps=50,
    )


def test_esp32_urdf_structure_and_inertials():
    assert_urdf_structure_and_joint_limits(
        URDF_PATH,
        robot_name="cyberduck_v2",
        min_total_mass=TOTAL_MASS_KG - 1e-6,
        max_total_mass=TOTAL_MASS_KG + 1e-6,
        min_base_mass=0.36 - 1e-6,
        max_base_mass=0.36 + 1e-6,
        base_inertial_z_max=-0.012 + 1e-6,
    )


def test_esp32_urdf_mujoco_compilation():
    assert_urdf_mujoco_compilation(URDF_PATH)


def test_esp32_mjcf_obstacles():
    assert_slalom_pillars_and_stairs(MJCF_PATH)


def test_esp32_sole_clearance_and_target_mass():
    from hardware.dual_esp32s3.training.env_config import _DEFAULT_CFG
    assert_sole_clearance_and_target_mass(MJCF_PATH, _DEFAULT_CFG.TARGET_MASS)


def test_mcu2_fallback_gravity_sign_imu_stale_and_mcu1_tof_valid_contract(tmp_path):
    """Task 2.3: Verify MCU2 fallback gravity sign parity + g_imu_stale and MCU1 real VL53L0X validity/stop via g++."""
    import shutil
    import subprocess

    header_path = REPO_ROOT / "src/hardware/dual_esp32s3/deployment/uart_packet_protocol.h"
    mcu1_src = (REPO_ROOT / "src/hardware/dual_esp32s3/deployment/mcu1_vision_audio_s3/mcu1_main.ino").read_text(encoding="utf-8")
    mcu2_src = (REPO_ROOT / "src/hardware/dual_esp32s3/deployment/mcu2_locomotion_s3/mcu2_main.ino").read_text(encoding="utf-8")
    header_src = header_path.read_text(encoding="utf-8")

    assert "sensor_valid" in header_src, "LidarPerceptionPacket must expose uint8_t sensor_valid"
    assert "project_quat_to_gravity" in mcu2_src, "mcu2_main.ino must share project_quat_to_gravity helper"
    assert "g_imu_stale" in mcu2_src, "mcu2_main.ino must track g_imu_stale"
    assert "g_lidar_valid" in mcu1_src, "mcu1_main.ino must track g_lidar_valid"
    assert "read_vl53l0x_single_mm" in mcu1_src, "mcu1_main.ino must implement VL53L0X I2C reader"

    mcu2_slice = mcu2_src[mcu2_src.index("static float s_gyro_bias[3]"):mcu2_src.index("// ============================================================================\n// Inter-MCU Packet Processing")]
    mcu1_vars = mcu1_src[mcu1_src.index("static float g_lidar_distances[5]"):mcu1_src.index("// ============================================================================\n// Peripheral Initialization Helpers")]
    mcu1_tof = mcu1_src[mcu1_src.index("void init_lidar()"):mcu1_src.index("// ============================================================================\n// Main Setup & Loop")]
    mcu1_slice = mcu1_vars + "\n" + mcu1_tof

    gpp = shutil.which("g++")
    assert gpp is not None, "g++ compiler is required for ESP32 firmware host contract verification"

    cpp_path = tmp_path / "esp32_driver_contract.cpp"
    exe_path = tmp_path / "esp32_driver_contract.exe"
    cpp_path.write_text(
        f"""#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include "{header_path.as_posix()}"

#define PI 3.14159265358979323846f
#define CONTROL_DT 0.02f
#define MPU6050_ADDR 0x68
#define PIN_I2C_SDA 4
#define PIN_I2C_SCL 5
#define LIDAR_I2C_ADDR 0x29
#define OBSTACLE_THRESHOLD 0.35f

static int g_wire_mode = 0; // 0: 14B valid, 1: IMU NACK (0B), 2: ToF 420mm, 3: ToF NACK
static uint8_t g_wire_buf[32];
static size_t g_wire_len = 0;
static size_t g_wire_pos = 0;
static uint8_t g_last_addr = 0;
static uint8_t g_last_reg = 0;

struct FakeWire {{
    void begin(int, int, uint32_t) {{}}
    void beginTransmission(uint8_t addr) {{ g_last_addr = addr; g_last_reg = 0; }}
    void write(uint8_t v) {{ static int wcount = 0; (void)wcount; g_last_reg = v; }}
    uint8_t endTransmission(bool = true) {{
        if (g_last_addr == LIDAR_I2C_ADDR && g_wire_mode == 3) return 2; // NACK
        return 0;
    }}
    uint8_t requestFrom(uint8_t addr, size_t len, bool = true) {{
        g_wire_pos = 0;
        g_wire_len = 0;
        if (addr == MPU6050_ADDR) {{
            if (g_wire_mode == 1) return 0;
            g_wire_len = len;
            memset(g_wire_buf, 0, sizeof(g_wire_buf));
            // 20 deg nose-down static accel matching q = [cos(10), 0, sin(10), 0]
            int16_t ax = (int16_t)(-sinf(20.0f * PI / 180.0f) * 16384.0f);
            int16_t az = (int16_t)(cosf(20.0f * PI / 180.0f) * 16384.0f);
            g_wire_buf[0] = (uint8_t)((ax >> 8) & 0xFF); g_wire_buf[1] = (uint8_t)(ax & 0xFF);
            g_wire_buf[4] = (uint8_t)((az >> 8) & 0xFF); g_wire_buf[5] = (uint8_t)(az & 0xFF);
            return (uint8_t)len;
        }}
        if (addr == LIDAR_I2C_ADDR) {{
            if (g_wire_mode == 3) return 0;
            g_wire_buf[0] = 0x01; g_wire_buf[1] = 0xA4; // 420 mm
            g_wire_len = 2;
            return 2;
        }}
        return 0;
    }}
    int available() {{ return (int)(g_wire_len - g_wire_pos); }}
    uint8_t read() {{ return (g_wire_pos < g_wire_len) ? g_wire_buf[g_wire_pos++] : 0; }}
}} Wire;

struct FakeSerial {{
    void println(const char*) {{}}
    void printf(const char*, ...) {{}}
}} Serial;
struct FakeI2S {{}} _dummy_i2s;
void delay(uint32_t) {{}}

namespace mcu2 {{
{mcu2_slice}
}}

namespace mcu1 {{
{mcu1_slice}
}}

int main() {{
    // 1. MCU2: s_q0 = cos(10 deg), s_q2 = sin(10 deg) => proj_grav[0] == +sin(20 deg) ~ +0.3420 in BOTH valid & fallback branches
    float rad10 = 10.0f * PI / 180.0f;
    mcu2::s_q0 = cosf(rad10);
    mcu2::s_q1 = 0.0f;
    mcu2::s_q2 = sinf(rad10);
    mcu2::s_q3 = 0.0f;

    float g_live[3] = {{0}}, pg_live[3] = {{0}};
    g_wire_mode = 0;
    mcu2::read_mpu6050(g_live, pg_live);
    assert(mcu2::g_imu_stale == false);
    assert(fabsf(pg_live[0] - 0.34202f) < 0.01f);

    float g_fb[3] = {{0}}, pg_fb[3] = {{0}};
    g_wire_mode = 1;
    mcu2::read_mpu6050(g_fb, pg_fb);
    assert(mcu2::g_imu_stale == true);
    assert(fabsf(pg_fb[0] - pg_live[0]) < 1e-4f);

    // 2. MCU1: ToF 420mm -> g_lidar_valid=true, 0.42m; NACK -> g_lidar_valid=false, g_cmd_vx=0.0f
    g_wire_mode = 2;
    mcu1::sample_lidar_sensors();
    assert(mcu1::g_lidar_valid == true);
    assert(fabsf(mcu1::g_lidar_distances[0] - 0.42f) < 1e-4f);

    g_wire_mode = 3;
    mcu1::sample_lidar_sensors();
    assert(mcu1::g_lidar_valid == false);
    assert(mcu1::g_cmd_vx == 0.0f);
    return 0;
}}
""",
        encoding="utf-8",
    )
    comp = subprocess.run([gpp, "-std=c++17", str(cpp_path), "-o", str(exe_path)], capture_output=True, text=True, check=False)
    assert comp.returncode == 0, f"g++ compile failed:\n{comp.stderr}"
    run = subprocess.run([str(exe_path)], capture_output=True, text=True, check=False)
    assert run.returncode == 0, f"C++ binary assertion failed:\n{run.stderr}"

