"""Tests for Microduck-inspired hardware runtime safety invariants."""

from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hardware.rpi_infer_core import BaseRPiInferenceEngine
from sim.env.policy_contract import observation_schema
from tools.robot.export_policy import export_onnx
from tools.train.train_ppo import ActorCritic


def test_imu_ready_startup_gate_blocks_premature_fall_detection():
    """Verify imu_ready startup convergence gates fall prediction until IMU stabilizes."""
    engine = BaseRPiInferenceEngine(mock=True, freq=50.0)
    engine.imu_reader.reset_readiness()
    assert engine.imu_reader.imu_ready is False

    tilted_grav = np.array([0.5, 0.0, -0.866], dtype=np.float32)
    tipping_w = np.array([0.0, 3.0, 0.0], dtype=np.float32)
    for _ in range(5):
        mode = engine.update_safety_state(tipping_w, tilted_grav)
        assert mode == "initializing"
        assert engine.fall_predictor.armed is False

    # Feed ready_confirm_samples (10) valid stationary 1g samples to converge
    for _ in range(engine.imu_reader.ready_confirm_samples):
        engine.imu_reader.update(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert engine.imu_reader.imu_ready is True
    assert engine.update_safety_state(np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32)) == "walk"


def test_nonfinite_obs_or_policy_output_holds_previous_targets(monkeypatch):
    """Verify NaN/Inf in observation or policy action never crashes and holds previous safe pose."""
    engine = BaseRPiInferenceEngine(mock=True, freq=50.0)
    safe_pose = np.array([0.05, -0.05, 0.0, -0.20, 0.20, 0.0, -0.20, 0.20], dtype=np.float32)
    engine.last_targets = safe_pose.copy()
    engine.last_raw_actions = np.full(8, 0.1, dtype=np.float32)

    bad_obs = np.zeros(42, dtype=np.float32)
    bad_obs[4] = np.nan
    targets1, _ = engine.step(bad_obs)
    np.testing.assert_allclose(targets1, safe_pose)
    assert engine.nonfinite_action_drops == 1

    good_obs = np.zeros(42, dtype=np.float32)
    monkeypatch.setattr(
        engine,
        "predict_action",
        lambda _o: np.array([np.nan, np.inf, -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    targets2, _ = engine.step(good_obs)
    np.testing.assert_allclose(targets2, safe_pose)
    assert engine.nonfinite_action_drops == 2


def test_deadman_timeout_zeroes_velocity_twist_without_going_limp():
    """Verify Deadman timeout zeroes velocity commands [0, 0, 0] while keeping walking controller alive."""
    engine = BaseRPiInferenceEngine(mock=True, freq=50.0, deadman_timeout_s=0.50)
    engine.set_velocity_command(0.22, 0.0, 0.10, timestamp=10.0)
    assert engine.get_active_command(now=10.2) == pytest.approx([0.22, 0.0, 0.10])
    assert engine.deadman_tripped is False

    assert engine.get_active_command(now=10.6) == [0.0, 0.0, 0.0]
    assert engine.deadman_tripped is True

    # Refreshing command clears deadman latch
    engine.set_velocity_command(0.15, 0.0, 0.0, timestamp=10.7)
    assert engine.get_active_command(now=10.8) == pytest.approx([0.15, 0.0, 0.0])
    assert engine.deadman_tripped is False


def test_limp_fall_and_standup_policy_hotswap_lifecycle(tmp_path):
    """Verify pre-impact fall predictor switches ONNX session to standup policy and recovers to walk."""
    schema = observation_schema(42, True, 0.02)

    walk_model = ActorCritic(hidden_dim=16, squash_actions=True)
    walk_model.model_name = "cyberduck_v2.xml"
    walk_model.observation_schema = schema
    with torch.no_grad():
        for p in walk_model.actor.parameters():
            p.zero_()
    walk_onnx = export_onnx(walk_model, tmp_path / "walk.onnx")

    standup_model = ActorCritic(hidden_dim=16, squash_actions=True)
    standup_model.model_name = "cyberduck_v2.xml"
    standup_model.observation_schema = schema
    with torch.no_grad():
        for p in standup_model.actor.parameters():
            p.zero_()
        standup_model.actor[-1].bias.fill_(0.8)
    standup_onnx = export_onnx(standup_model, tmp_path / "standup.onnx")

    engine = BaseRPiInferenceEngine(
        model_path=str(walk_onnx),
        standup_model_path=str(standup_onnx),
        mock=True,
        freq=50.0,
    )
    engine.imu_reader.reset_readiness()
    for _ in range(engine.imu_reader.ready_confirm_samples):
        engine.imu_reader.update(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    upright_g = np.array([0.0, 0.0, -0.98], dtype=np.float32)
    zero_w = np.zeros(3, dtype=np.float32)
    obs = np.zeros(42, dtype=np.float32)

    # Arm predictor during 10 upright control ticks
    for _ in range(10):
        mode = engine.update_safety_state(zero_w, upright_g)
    assert mode == "walk"
    assert np.allclose(engine.predict_action(obs), 0.0, atol=1e-4)

    # Tip rapidly forward -> trips FallPredictor -> switches to standup policy
    tipping_g = np.array([0.50, 0.0, -0.866], dtype=np.float32)
    tipping_w = np.array([0.0, 2.5, 0.0], dtype=np.float32)
    for _ in range(3):
        mode = engine.update_safety_state(tipping_w, tipping_g)
    assert mode in ("limp_fall", "standup_recovery")
    assert float(np.mean(engine.predict_action(obs))) > 0.5

    # Recover upright for 10 ticks -> returns to walk policy
    for _ in range(10):
        mode = engine.update_safety_state(zero_w, upright_g)
    assert mode == "walk"
    assert np.allclose(engine.predict_action(obs), 0.0, atol=1e-4)


def test_hardware_mode_refuses_to_drive_servos_when_imu_init_fails(monkeypatch):
    """Fail closed when mock=False and PCA9685 succeeds at 0x40 but MPU6050 NACKs at 0x68."""
    import hardware.rpi_infer_core as core

    class _FakeBus:
        def __init__(self, _bus_num):
            pass

        def read_byte_data(self, addr, _reg):
            if addr == 0x68:
                raise OSError("MPU6050 read NACK")
            return 0x00

        def write_byte_data(self, addr, _reg, _val):
            if addr == 0x68:
                raise OSError("MPU6050 write NACK")

        def write_i2c_block_data(self, _addr, _reg, _data):
            return None

    class _FakeSMBusMod:
        SMBus = _FakeBus

    monkeypatch.setattr(core, "smbus", _FakeSMBusMod)
    with pytest.raises(RuntimeError, match="MPU6050"):
        BaseRPiInferenceEngine(mock=False, freq=50.0, model_path=None)

    monkeypatch.setattr(core, "smbus", None)
    with pytest.raises(RuntimeError, match="I2C|PCA9685|smbus"):
        BaseRPiInferenceEngine(mock=False, freq=50.0, model_path=None)


def test_limp_fall_without_standup_policy_disables_all_servo_pwm_outputs(monkeypatch):
    """When safety_mode=='limp_fall' with no standup policy, PCA9685 outputs must be disabled (PWM=0)."""
    import hardware.rpi_infer_core as core

    engine = BaseRPiInferenceEngine(mock=True, freq=50.0, standup_model_path=None)
    engine.safety_mode = "limp_fall"
    called = False

    def _forbid_predict(_obs):
        nonlocal called
        called = True
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(engine, "predict_action", _forbid_predict)
    targets, pwms = engine.step(np.zeros(42, dtype=np.float32))
    assert called is False
    assert pwms.tolist() == [0] * 8
    assert engine.servo_controller.channel_pwms[:8] == [0] * 8
    assert engine.servo_controller.outputs_enabled is False
    np.testing.assert_allclose(targets, engine.last_targets)

    # Verify HardwarePCA9685.disable_all_channels writes FULL_OFF (0x10) to ALL_LED_ON_L (0xFA..0xFD)
    writes = []

    class _RecordingBus:
        def __init__(self, _bus_num):
            pass

        def read_byte_data(self, _addr, _reg):
            return 0x00

        def write_byte_data(self, addr, reg, val):
            writes.append((addr, reg, val))

        def write_i2c_block_data(self, addr, reg, data):
            writes.append((addr, reg, list(data)))

    monkeypatch.setattr(core, "smbus", type("_SM", (), {"SMBus": _RecordingBus}))
    hw_pca = core.HardwarePCA9685(bus_num=1, freq=50)
    hw_pca.disable_all_channels()
    assert (0x40, 0xFA, [0, 0, 0, 0x10]) in writes

