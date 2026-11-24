"""Tests for Microduck-inspired Sim2Real and pre-impact safety upgrades."""

from pathlib import Path
import sys
import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot.robot_spec import BASE_HEIGHT_M
from sim.behaviors.self_righting import FallPredictor, compute_grav_z_dot
from sim.env.env_config import (
    DEFAULT_NOMINAL_QPOS,
    LIE_NOMINAL_QPOS,
    LocomotionMode,
    MicroDuck8DoFConfig,
    SIT_NOMINAL_QPOS,
    TARGET_LIE_HEIGHT,
    TARGET_SIT_HEIGHT,
    compute_action_over_limit_penalty,
    compute_battery_sag_torque_limit,
    compute_rewards,
    compute_slewed_posture_target,
    small_angle_rotation_matrix,
)
from sim.env.run_parallel_sim import ParallelSimRunner
from tools.train.train_ppo import compute_gae, compute_scheduled_action_rate_weight


def test_analytical_grav_z_dot_and_fall_predictor_pre_impact_trip():
    """Verify analytical g_z_dot sign, imu_ready gate, arming, and 3-tick pre-impact trip."""
    assert compute_grav_z_dot([0.0, 2.5, 0.0], [0.5, 0.0, -0.866]) == pytest.approx(1.25)

    predictor = FallPredictor(
        lookahead_s=0.30,
        armed_grav_z=-0.90,
        fall_grav_z=-0.50,
        arm_confirm_steps=10,
        trip_confirm_steps=3,
        limp_kp_scale=0.25,
    )
    upright_g = np.array([0.0, 0.0, -0.98], dtype=np.float32)
    zero_w = np.zeros(3, dtype=np.float32)

    # 1. Does not arm while imu_ready=False
    for _ in range(15):
        assert predictor.update(zero_w, upright_g, imu_ready=False) is False
    assert predictor.armed is False

    # 2. Arms only after arm_confirm_steps upright updates
    for step in range(9):
        assert predictor.update(zero_w, upright_g, imu_ready=True) is False
        assert predictor.armed is False
    assert predictor.update(zero_w, upright_g, imu_ready=True) is False
    assert predictor.armed is True

    # 3. Tips forward: g_z=-0.866 (> -0.90), g_z_dot=+1.25 -> g_z_pred=-0.491 > -0.50
    tipping_g = np.array([0.50, 0.0, -0.866], dtype=np.float32)
    tipping_w = np.array([0.0, 2.5, 0.0], dtype=np.float32)
    assert predictor.update(tipping_w, tipping_g, imu_ready=True) is False
    assert predictor.update(tipping_w, tipping_g, imu_ready=True) is False
    assert predictor.update(tipping_w, tipping_g, imu_ready=True) is True
    assert predictor.limp_active is True


def test_parallel_sim_runner_limp_fall_reduces_servo_torque_before_impact():
    """Verify ParallelSimRunner drops servo Kp and actuator torque when FallPredictor trips."""
    actions = np.full((1, 8), 0.5, dtype=np.float32)

    sim_normal = ParallelSimRunner(num_envs=1, decimation=1, terrain_mode="flat", enable_limp_fall=False)
    sim_limp = ParallelSimRunner(
        num_envs=1, decimation=1, terrain_mode="flat", enable_limp_fall=True, limp_kp_scale=0.25
    )
    try:
        assert len(sim_limp.fall_predictors) == 1
        assert sim_limp.limp_active.shape == (1,)

        # Arm predictor with upright pose, then apply pre-impact pitch + angular velocity
        sim_limp.fall_predictors[0].armed = True
        for sim in (sim_normal, sim_limp):
            d = sim.datas[0]
            # Pitch nose-down ~30 deg (q=[cos(15 deg), 0, sin(15 deg), 0] -> g_x=+0.5, g_z=-0.866)
            d.qpos[3:7] = [np.cos(np.pi / 12), 0.0, np.sin(np.pi / 12), 0.0]
            d.qvel[4] = 3.0  # pitch rate +3 rad/s -> g_z_dot = +1.5, g_z_pred = -0.416 > -0.50
            mujoco.mj_forward(sim.model, d)

        # Feed 3 confirmation steps so FallPredictor trips before hitting ground
        for _ in range(3):
            for sim in (sim_normal, sim_limp):
                d = sim.datas[0]
                d.qpos[2] = 0.128
                d.qpos[3:7] = [np.cos(np.pi / 12), 0.0, np.sin(np.pi / 12), 0.0]
                d.qvel[4] = 3.0
                mujoco.mj_forward(sim.model, d)
            sim_normal.step(actions)
            sim_limp.step(actions)

        assert sim_limp.limp_active[0]
        assert not sim_normal.limp_active[0]

        # On the next control step, limp runner must apply scaled KP torques (< normal runner)
        for sim in (sim_normal, sim_limp):
            d = sim.datas[0]
            d.qpos[2] = 0.128
            d.qpos[3:7] = [np.cos(np.pi / 12), 0.0, np.sin(np.pi / 12), 0.0]
            d.qvel[4] = 3.0
            d.qpos[7:15] = sim.cfg.nominal_qpos
            d.qvel[6:14] = 0.0
            sim.last_targets[0] = sim.cfg.nominal_qpos
            mujoco.mj_forward(sim.model, d)

        _, _, _, info_norm = sim_normal.step(actions)
        _, _, _, info_limp = sim_limp.step(actions)
        norm_ctrl = np.linalg.norm(info_norm[0]["step_ctrl"])
        limp_ctrl = np.linalg.norm(info_limp[0]["step_ctrl"])
        assert limp_ctrl < 0.5 * norm_ctrl
    finally:
        sim_normal.close()
        sim_limp.close()


def test_command_sampling_buckets_zero_and_turn_in_place():
    """Verify command sampling buckets produce exact zero-cmd, pure spot-turn, and forward walk."""
    for bad_zero, bad_turn in [(-0.1, 0.0), (0.0, -0.1), (0.6, 0.5)]:
        with pytest.raises(ValueError):
            ParallelSimRunner(
                num_envs=1,
                terrain_mode="flat",
                zero_cmd_prob=bad_zero,
                turn_in_place_prob=bad_turn,
            )

    runner = ParallelSimRunner(
        num_envs=64,
        terrain_mode="flat",
        zero_cmd_prob=0.25,
        turn_in_place_prob=0.25,
    )
    try:
        cmds = runner.commands
        is_zero = np.all(cmds == 0.0, axis=-1)
        is_turn = (cmds[:, 0] == 0.0) & (cmds[:, 1] == 0.0) & (np.abs(cmds[:, 2]) >= 0.20)
        is_walk = cmds[:, 0] >= 0.10

        assert np.all(is_zero | is_turn | is_walk)
        assert 6 <= int(np.sum(is_zero)) <= 28
        assert 6 <= int(np.sum(is_turn)) <= 28
        assert 16 <= int(np.sum(is_walk)) <= 48

        # Mutating via setter works too
        runner.set_command_bucket_probs(1.0, 0.0)
        runner.reset()
        np.testing.assert_array_equal(runner.commands, 0.0)
    finally:
        runner.close()


def test_slewed_posture_target_and_multiplicative_goal_composite():
    """Verify posture target slewing, multiplicative goal gate, and encoded posture commands."""
    q0, z0 = compute_slewed_posture_target(0.0, LocomotionMode.SIT)
    np.testing.assert_allclose(q0, DEFAULT_NOMINAL_QPOS)
    assert z0 == pytest.approx(BASE_HEIGHT_M)

    q1, z1 = compute_slewed_posture_target(1.0, LocomotionMode.SIT)
    np.testing.assert_allclose(q1, SIT_NOMINAL_QPOS)
    assert z1 == pytest.approx(TARGET_SIT_HEIGHT)

    qb, zb = compute_slewed_posture_target(
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([LocomotionMode.SIT, LocomotionMode.LIE], dtype=np.int32),
    )
    assert qb.shape == (2, 8)
    np.testing.assert_allclose(qb[1], LIE_NOMINAL_QPOS)
    np.testing.assert_allclose(zb, [BASE_HEIGHT_M, TARGET_LIE_HEIGHT])

    # Multiplicative posture reward blocks farming height reward while lying sideways
    common = dict(
        base_lin_vel=np.zeros(3, dtype=np.float32),
        cmd_lin_vel=np.zeros(3, dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        joint_pos=SIT_NOMINAL_QPOS,
        root_z=TARGET_SIT_HEIGHT,
        cmd_mode=LocomotionMode.SIT,
        posture_blend=1.0,
        multiplicative_posture_reward=True,
    )
    r_upright = compute_rewards(proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32), **common)
    r_tilted = compute_rewards(proj_gravity=np.array([1.0, 0.0, 0.0], dtype=np.float32), **common)
    assert r_upright["sitting_pose"] == pytest.approx(1.0, abs=1e-5)
    assert r_tilted["sitting_pose"] < 0.05

    # Runner encodes SIT vs LIE commands cleanly and advances posture_blend smoothly
    runner = ParallelSimRunner(
        num_envs=2,
        decimation=1,
        terrain_mode="flat",
        posture_ramp_s=1.0,
        encode_posture_commands=True,
        multiplicative_posture_reward=True,
    )
    try:
        runner.set_cmd_modes([LocomotionMode.SIT, LocomotionMode.LIE])
        np.testing.assert_allclose(runner.commands[0], [0.0, -0.5, 0.0])
        np.testing.assert_allclose(runner.commands[1], [0.0, -1.0, 0.0])
        obs = runner.get_observations()
        assert not np.allclose(obs[0, 6:9], obs[1, 6:9])
        assert np.all(runner.posture_blend == 0.0)
        runner.step(np.zeros((2, 8), dtype=np.float32))
        assert np.all((runner.posture_blend > 0.0) & (runner.posture_blend < 1.0))
    finally:
        runner.close()


def test_head_ema_dc_bias_penalty_and_action_over_limit_penalty():
    """Verify EMA head DC-bias L1 penalty, over-limit penalty, and runner reward sum integrity."""
    cfg = MicroDuck8DoFConfig()
    assert cfg.REWARD_WEIGHTS["head_ema_penalty"] == -0.5
    assert cfg.REWARD_WEIGHTS["action_over_limit_penalty"] == -0.2

    assert compute_action_over_limit_penalty(np.full(8, 0.95, dtype=np.float32)) == 0.0
    over_val = compute_action_over_limit_penalty(np.full(8, 1.0, dtype=np.float32))
    assert over_val > 0.0

    base_args = dict(
        base_lin_vel=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        actions=np.ones(8, dtype=np.float32),
        last_actions=np.ones(8, dtype=np.float32),
    )
    r_no_extra = compute_rewards(**base_args)
    r_ema = compute_rewards(head_err_ema=np.array([0.0, 0.3], dtype=np.float32), **base_args)
    assert r_ema["head_ema_penalty"] == pytest.approx(0.3, abs=1e-5)
    assert r_no_extra["total_reward"] - r_ema["total_reward"] == pytest.approx(0.15, abs=1e-5)

    r_over = compute_rewards(action_over_limit_threshold=0.95, **base_args)
    assert r_over["action_over_limit_penalty"] == pytest.approx(over_val, abs=1e-6)
    assert r_no_extra["total_reward"] - r_over["total_reward"] == pytest.approx(0.2 * over_val, abs=1e-6)

    runner = ParallelSimRunner(
        num_envs=1,
        decimation=1,
        terrain_mode="flat",
        enable_head_ema_penalty=True,
        action_over_limit_threshold=0.95,
    )
    try:
        _, rewards, _, infos = runner.step(np.ones((1, 8), dtype=np.float32))
        assert sum(infos[0]["reward_components"].values()) == pytest.approx(float(rewards[0]), abs=1e-5)
    finally:
        runner.close()


def test_battery_voltage_sag_reduces_torque_limit_under_high_load():
    """Verify battery voltage sag lowers effective torque ceiling and floors at V_min / V_nom."""
    no_load = compute_battery_sag_torque_limit(np.zeros(8), v_in=7.4, k_drop=0.2)
    assert no_load == pytest.approx(1.0)

    heavy_load = compute_battery_sag_torque_limit(np.full(8, 1.0), v_in=7.4, k_drop=0.2)
    # 7.4 - 0.2 * 8.0 = 5.8 V -> 5.8 / 7.4
    assert heavy_load == pytest.approx(5.8 / 7.4, abs=1e-5)

    extreme_load = compute_battery_sag_torque_limit(np.full(8, 5.0), v_in=6.5, k_drop=0.5, v_min=5.0)
    assert extreme_load == pytest.approx(5.0 / 7.4, abs=1e-5)

    R = small_angle_rotation_matrix(np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.1)
    assert R.shape == (3, 3)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)

    runner = ParallelSimRunner(num_envs=2, decimation=1, terrain_mode="flat", enable_battery_sag=True)
    try:
        runner.env_v_in[:] = [6.6, 6.6]
        runner.env_k_drop[:] = [0.2, 0.2]
        # Command large step so PD torques hit the sagged torque limit
        _, _, _, infos = runner.step(np.ones((2, 8), dtype=np.float32))
        assert np.max(np.abs(infos[0]["step_ctrl"])) < 1.0 - 1e-3
    finally:
        runner.close()


def test_actor_only_sensor_corruption_and_clean_asymmetric_critic():
    """Verify Actor sees encoder bias, IMU mount tilt, and 1-step lagged joint_vel while Critic sees clean state."""
    runner = ParallelSimRunner(
        num_envs=2,
        decimation=5,
        terrain_mode="flat",
        encoder_bias_rad=0.015,
        imu_mount_max_rad=0.10,
        delay_joint_vel_obs=True,
    )
    try:
        assert np.any(runner.env_encoder_bias != 0.0)
        assert not np.allclose(runner.env_imu_rot[0], np.eye(3))

        # Step 1: physics moves joints so current sensordata[20:28] != 0, while Actor still receives prev_joint_vel == 0
        obs1, _, dones1, _ = runner.step(np.full((2, 8), 0.6, dtype=np.float32))
        assert not dones1.any()
        priv1 = runner.get_privileged_obs(raw_obs=obs1)

        true_qpos = np.array(runner.datas[0].sensordata[12:20], dtype=np.float32)
        true_qvel = np.array(runner.datas[0].sensordata[20:28], dtype=np.float32)
        assert np.linalg.norm(true_qvel) > 1e-3

        # Actor has bias + lagged velocity (0.0); Critic has uncorrupted qpos + current qvel
        np.testing.assert_allclose(obs1[0, 9:17], true_qpos + runner.env_encoder_bias[0], atol=1e-6)
        np.testing.assert_array_equal(obs1[0, 17:25], np.zeros(8, dtype=np.float32))
        np.testing.assert_allclose(priv1[0, 9:17], true_qpos, atol=1e-6)
        np.testing.assert_allclose(priv1[0, 17:25], true_qvel, atol=1e-6)

        # Actor grav matches rotated true grav
        w, x, y, z = runner.datas[0].sensordata[6:10]
        true_grav = np.array([2.0 * (w * y - x * z), -2.0 * (w * x + y * z), 2.0 * (x * x + y * y) - 1.0], dtype=np.float32)
        np.testing.assert_allclose(obs1[0, 3:6], runner.env_imu_rot[0] @ true_grav, atol=1e-6)
        np.testing.assert_allclose(priv1[0, 3:6], true_grav, atol=1e-6)

        # Step 2: Actor receives step 1's true_qvel (1-step lag)
        obs2, _, dones2, _ = runner.step(np.full((2, 8), -0.6, dtype=np.float32))
        assert not dones2.any()
        np.testing.assert_allclose(obs2[0, 17:25], true_qvel, atol=1e-6)
    finally:
        runner.close()


def test_nan_safe_critic_and_gae_plus_scheduled_action_rate_weight():
    """Verify GAE & privileged obs sanitize NaN/Inf and smoothness weight ramps properly."""
    import torch

    rewards = torch.tensor([[1.0, float("nan")], [0.5, float("inf")]], dtype=torch.float32)
    dones = torch.zeros((2, 2), dtype=torch.float32)
    values = torch.tensor([[0.2, float("-inf")], [0.3, 0.4]], dtype=torch.float32)
    next_val = torch.tensor([0.1, float("nan")], dtype=torch.float32)
    adv, ret = compute_gae(rewards, dones, values, next_val)
    assert torch.isfinite(adv).all()
    assert torch.isfinite(ret).all()

    runner = ParallelSimRunner(num_envs=1, decimation=1, terrain_mode="flat")
    try:
        runner.datas[0].qvel[0] = float("nan")
        priv = runner.get_privileged_obs()
        assert np.all(np.isfinite(priv))
    finally:
        runner.close()

    assert compute_scheduled_action_rate_weight(0.0) == pytest.approx(-0.02)
    assert compute_scheduled_action_rate_weight(0.20) == pytest.approx(-0.06)
    assert compute_scheduled_action_rate_weight(0.40) == pytest.approx(-0.10)
    assert compute_scheduled_action_rate_weight(1.0) == pytest.approx(-0.10)


def test_train_ppo_forwards_all_sim2real_flags_to_runner(monkeypatch, tmp_path):
    """Verify train_ppo passes all 7 Sim2Real parameters through to ParallelSimRunner."""
    import tools.train.train_ppo as train_ppo_mod

    captured_kwargs = {}
    orig_runner = train_ppo_mod.ParallelSimRunner

    class _SpyRunner(orig_runner):
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(train_ppo_mod, "ParallelSimRunner", _SpyRunner)

    train_ppo_mod.train_ppo(
        save_path=tmp_path / "sim2real_ppo.pt",
        dry_run=True,
        num_envs=2,
        rollout_steps=2,
        max_iterations=1,
        verbose=False,
        enable_limp_fall=True,
        limp_kp_scale=0.33,
        enable_head_ema_penalty=True,
        action_over_limit_threshold=0.92,
        enable_battery_sag=True,
        encoder_bias_rad=0.015,
        imu_mount_max_rad=0.08,
        delay_joint_vel_obs=True,
    )

    assert captured_kwargs["enable_limp_fall"] is True
    assert captured_kwargs["limp_kp_scale"] == pytest.approx(0.33)
    assert captured_kwargs["enable_head_ema_penalty"] is True
    assert captured_kwargs["action_over_limit_threshold"] == pytest.approx(0.92)
    assert captured_kwargs["enable_battery_sag"] is True
    assert captured_kwargs["encoder_bias_rad"] == pytest.approx(0.015)
    assert captured_kwargs["imu_mount_max_rad"] == pytest.approx(0.08)
    assert captured_kwargs["delay_joint_vel_obs"] is True

