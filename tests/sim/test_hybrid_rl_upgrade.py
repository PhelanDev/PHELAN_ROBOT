"""Tests for the CyberDuck V2 3-in-1 Modular Hybrid RL Algorithm Upgrade."""

from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
import pytest
import torch

from sim.behaviors.advanced_gaits import BipedalGaitGenerator


def test_symmetry_involution_and_kinematic_consistency():
    from sim.env.symmetry import (
        ACT_MIRROR_PERM_8,
        ACT_MIRROR_SIGN_8,
        CRITIC_MIRROR_PERM_48,
        OBS_MIRROR_PERM_37,
        augment_symmetry_batch,
        compute_actor_symmetry_loss,
        mirror_actions_8,
        mirror_critic_obs,
        mirror_obs,
    )

    assert len(ACT_MIRROR_PERM_8) == 8
    assert len(ACT_MIRROR_SIGN_8) == 8
    assert len(OBS_MIRROR_PERM_37) == 37
    assert len(CRITIC_MIRROR_PERM_48) == 48

    rng = np.random.default_rng(123)
    # NumPy 1D and 2D involution
    for shape_obs, shape_act, shape_crit in [((37,), (8,), (48,)), ((6, 37), (6, 8), (6, 48))]:
        obs_np = rng.standard_normal(shape_obs, dtype=np.float32)
        act_np = rng.standard_normal(shape_act, dtype=np.float32)
        crit_np = rng.standard_normal(shape_crit, dtype=np.float32)

        np.testing.assert_allclose(mirror_obs(mirror_obs(obs_np)), obs_np, atol=1e-6)
        np.testing.assert_allclose(mirror_actions_8(mirror_actions_8(act_np)), act_np, atol=1e-6)
        np.testing.assert_allclose(mirror_critic_obs(mirror_critic_obs(crit_np)), crit_np, atol=1e-6)

        obs_t = torch.from_numpy(obs_np.copy())
        act_t = torch.from_numpy(act_np.copy())
        crit_t = torch.from_numpy(crit_np.copy())

        torch.testing.assert_close(mirror_obs(mirror_obs(obs_t)), obs_t)
        torch.testing.assert_close(mirror_actions_8(mirror_actions_8(act_t)), act_t)
        torch.testing.assert_close(mirror_critic_obs(mirror_critic_obs(crit_t)), crit_t)

    # Kinematic consistency with BipedalGaitGenerator.compute_walk(t, freq=1.25)
    # Period T = 0.8s -> half period = 0.4s swaps left/right and flips yaw/roll
    gen = BipedalGaitGenerator(default_freq=1.25)
    for t in (0.0, 0.1, 0.25, 0.37):
        q_t = gen.compute_walk(t, freq=1.25)
        q_half = gen.compute_walk(t + 0.4, freq=1.25)
        np.testing.assert_allclose(mirror_actions_8(q_t), q_half, atol=1e-6)

    # Verify symmetry loss scalar & gradient flow
    actor = torch.nn.Sequential(
        torch.nn.Linear(37, 32),
        torch.nn.ELU(),
        torch.nn.Linear(32, 8),
    )
    obs_batch = torch.randn(10, 37)
    sym_loss = compute_actor_symmetry_loss(actor, obs_batch)
    assert sym_loss.ndim == 0
    assert torch.isfinite(sym_loss) and float(sym_loss.item()) > 0.0
    sym_loss.backward()
    for p in actor.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()

    # Verify augment_symmetry_batch
    o_aug, a_aug, r_aug = augment_symmetry_batch(
        np.ones((4, 37), dtype=np.float32),
        np.ones((4, 8), dtype=np.float32),
        np.arange(4, dtype=np.float32),
    )
    assert o_aug.shape == (8, 37)
    assert a_aug.shape == (8, 8)
    assert r_aug.shape == (8,)
    np.testing.assert_allclose(r_aug[:4], r_aug[4:])


def test_phase_reference_gait_imitation_symmetry_and_reward():
    from sim.env.env_config import compute_reference_gait_targets, compute_rewards
    from sim.env.symmetry import mirror_actions_8

    # Scalar & batch sagittal symmetry + gait_sync alignment (clock_sin > 0 => left leg lifted)
    for s, c in [(0.8, 0.6), (-0.5, 0.866), (0.0, 1.0)]:
        q_ref = compute_reference_gait_targets(s, c)
        q_mir = compute_reference_gait_targets(-s, -c)
        assert q_ref.shape == (8,)
        np.testing.assert_allclose(mirror_actions_8(q_ref), q_mir, atol=1e-6)
        if s > 0:
            assert q_ref[4] > q_ref[7]  # left_knee_pitch > right_knee_pitch

    s_batch = np.array([0.8, -0.8], dtype=np.float32)
    c_batch = np.array([0.6, -0.6], dtype=np.float32)
    q_batch = compute_reference_gait_targets(s_batch, c_batch)
    assert q_batch.shape == (2, 8)
    np.testing.assert_allclose(mirror_actions_8(q_batch[0]), q_batch[1], atol=1e-6)

    # Reward calculation with gait_imitation (scalar vs batch parity)
    q_match = q_batch[0].copy()
    q_bad = q_match.copy()
    q_bad[2:8] += 0.5
    common = dict(
        base_lin_vel=np.array([0.15, 0.0, 0.0], dtype=np.float32),
        cmd_lin_vel=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        cmd_ang_vel=0.0,
        proj_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        joint_torques=np.zeros(8, dtype=np.float32),
        joint_vel=np.zeros(8, dtype=np.float32),
        foot_contact=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        clock_sin=0.8,
        clock_cos=0.6,
    )
    r_match, terms_match = compute_rewards(**common, joint_pos=q_match, return_components=True)
    r_bad = compute_rewards(**common, joint_pos=q_bad)
    assert r_match["gait_imitation"] == pytest.approx(0.6, abs=1e-5)
    assert r_match["gait_imitation"] > r_bad["gait_imitation"]
    assert r_match["total_reward"] == pytest.approx(sum(terms_match.values()))

    batch_args = {
        k: (np.stack([v, v]) if isinstance(v, np.ndarray) else np.array([v, v], dtype=np.float32) if k.startswith("clock_") else v)
        for k, v in common.items()
    }
    r_batch = compute_rewards(**batch_args, joint_pos=np.stack([q_match, q_bad]))
    np.testing.assert_allclose(r_batch["gait_imitation"], [r_match["gait_imitation"], r_bad["gait_imitation"]], atol=1e-5)


def test_audit_f07_multi_env_reset_randomizes_dynamics():
    from sim.env.run_parallel_sim import ParallelSimRunner

    runner = ParallelSimRunner(
        model_path="src/robot/cyberduck_v2.xml",
        num_envs=2,
        num_threads=1,
        max_episode_steps=1,
    )
    try:
        # Direct single-env reset on multi-env runner preserves contract test isolation
        before = runner.model.body_mass.copy()
        runner.reset_env(1)
        np.testing.assert_array_equal(before, runner.model.body_mass)

        # Synchronous step boundary after multi-env episode resets resamples dynamics
        masses = []
        frictions = []
        zero_act = np.zeros((2, 8), dtype=np.float32)
        for _ in range(6):
            runner.step(zero_act, fast=True)
            masses.append(float(runner.model.body_mass[runner.base_body_id]))
            frictions.append(float(runner.model.geom_friction[runner.floor_gid, 0]))
        assert len(set(round(m, 8) for m in masses)) > 1
        assert len(set(round(f, 8) for f in frictions)) > 1
    finally:
        runner.close()


def test_actuator_delay_queue_delays_targets_but_keeps_obs_last_action():
    from sim.env.env_config import scale_actions
    from sim.env.run_parallel_sim import ParallelSimRunner

    runner = ParallelSimRunner(
        model_path="src/robot/cyberduck_v2.xml",
        num_envs=2,
        num_threads=1,
        max_action_delay_steps=2,
    )
    try:
        runner.reset()
        runner.env_action_delay[:] = [0, 2]
        a0 = np.full((2, 8), 0.25, dtype=np.float32)
        a1 = np.zeros((2, 8), dtype=np.float32)

        obs0, _, _, _ = runner.step(a0, fast=True)
        np.testing.assert_allclose(obs0[:, 25:33], a0, atol=1e-6)
        expected_target_a0 = scale_actions(a0[0], runner.cfg, nominal_centered=True)
        expected_target_zero = scale_actions(a1[0], runner.cfg, nominal_centered=True)
        # Env 0 has delay=0 (receives a0 immediately); Env 1 has delay=2 (receives 0 initially)
        assert not np.allclose(runner.last_targets[0], expected_target_zero, atol=1e-4)
        np.testing.assert_allclose(runner.last_targets[1], expected_target_zero, atol=1e-4)

        obs1, _, _, _ = runner.step(a1, fast=True)
        np.testing.assert_allclose(obs1[:, 25:33], a1, atol=1e-6)
        # Env 1 still has 0 from t=-1
        np.testing.assert_allclose(runner.last_targets[1], expected_target_zero, atol=1e-4)

        # At step 2, env 1 receives a0 from step 0 (2 steps delayed)
        runner.step(a1, fast=True)
        assert np.linalg.norm(runner.last_targets[1] - expected_target_zero) > 1e-3
        assert np.all(np.abs(runner.last_targets[1] - expected_target_a0) <= np.abs(expected_target_a0 - expected_target_zero) + 1e-4)
    finally:
        runner.close()


def test_audit_f08_benchmark_captures_terminal_fall_tilt_and_torque(monkeypatch):
    from sim.env.run_parallel_sim import ParallelSimRunner
    from tools.train.evaluate_models_sim import run_benchmark_scenario

    runner = ParallelSimRunner(model_path="src/robot/cyberduck_v2.xml", num_envs=2, num_threads=1)
    try:
        theta = np.deg2rad(90.0)

        def force_terminal_fall(indices, targets):
            for idx in indices:
                d = runner.datas[idx]
                d.qpos[3:7] = [np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0]
                d.ctrl[:] = 0.5
                mujoco.mj_forward(runner.model, d)

        monkeypatch.setattr(runner, "_step_subrange", force_terminal_fall)

        dummy_actor = torch.nn.Sequential(torch.nn.Linear(37, 8))
        with torch.no_grad():
            for p in dummy_actor.parameters():
                p.zero_()
        dummy_actor.obs_dim = 37

        metrics = run_benchmark_scenario(
            runner,
            dummy_actor,
            cmd_vx=0.2,
            max_steps=1,
            scenario_name="Terminal Fall F08",
            use_nav_planner=False,
        )
        assert metrics["success_rate"] == 0.0
        assert metrics["mean_tilt_deg"] > 80.0
        assert metrics["mean_torque_sq"] == pytest.approx(2.0, abs=1e-4)
    finally:
        runner.close()


def test_bc_symmetry_augmentation_and_dagger_teacher_collection():
    from sim.env.run_parallel_sim import ParallelSimRunner
    from sim.env.symmetry import compute_actor_symmetry_loss
    from tools.train.pretrain_bc import collect_dagger_demonstrations, train_behavior_cloning
    from tools.train.train_ppo import ActorCritic

    torch.manual_seed(0)
    teacher = ActorCritic(obs_dim=37, action_dim=8, hidden_dim=32, squash_actions=True).eval()
    student = ActorCritic(obs_dim=37, action_dim=8, hidden_dim=32, squash_actions=True).eval()

    runner = ParallelSimRunner(model_path="src/robot/cyberduck_v2.xml", num_envs=2, num_threads=1, stage=0)
    try:
        obs_arr, act_arr, ret_arr = collect_dagger_demonstrations(
            runner,
            teacher_actor=teacher,
            student_actor=student,
            num_steps=12,
            beta=0.5,
        )
    finally:
        runner.close()

    assert obs_arr.ndim == 2 and obs_arr.shape[1] == 37
    assert act_arr.shape == (len(obs_arr), 8)
    assert ret_arr.shape == (len(obs_arr),)
    with torch.no_grad():
        expected_teacher_acts = teacher.forward_actor(torch.from_numpy(obs_arr)).numpy()
    np.testing.assert_allclose(act_arr, expected_teacher_acts, atol=1e-5)

    model, losses = train_behavior_cloning(
        obs_arr,
        act_arr,
        returns=ret_arr,
        obs_dim=37,
        action_dim=8,
        hidden_dim=64,
        epochs=8,
        batch_size=16,
        augment_symmetry=True,
        symmetry_coef=0.5,
    )
    assert losses[-1] < losses[0]
    # Symmetrized obs_rms cancels antisymmetric components (e.g., y-gravity index 4, roll joints 11/14)
    assert abs(float(model._bc_obs_rms.mean[4])) < 1e-5
    with torch.no_grad():
        norm_obs = torch.from_numpy(model._bc_obs_rms.normalize(obs_arr))
        sym_err = float(compute_actor_symmetry_loss(model.forward_actor, norm_obs).item())
    assert np.isfinite(sym_err)
    assert sym_err < 0.05


def test_ppo_symmetry_loss_and_action_delay(tmp_path: Path):
    from tools.train.train_ppo import train_ppo

    save_path = tmp_path / "sym_ppo.pt"
    result = train_ppo(
        model_path="src/robot/cyberduck_v2.xml",
        num_envs=2,
        rollout_steps=8,
        max_iterations=1,
        num_epochs=2,
        mini_batch_size=8,
        symmetry_coef=0.5,
        max_action_delay=1,
        save_path=save_path,
        device="cpu",
        verbose=False,
    )
    assert result["success"] is True
    assert "symmetry_loss" in result["metrics"]
    assert np.isfinite(result["metrics"]["symmetry_loss"])
    assert save_path.exists()
    ckpt = torch.load(str(save_path), map_location="cpu", weights_only=False)
    assert ckpt["model_name"] == "cyberduck_v2.xml"
    assert ckpt["observation_schema"]["obs_dim"] == 37
    assert ckpt["observation_schema"]["version"] == 2


def test_fast_sac_elu_nstep_symmetry_warmstart_and_esp32_export(tmp_path: Path):
    import onnxruntime as ort
    from sim.env.env_config import DEFAULT_OBS_BOUNDS
    from tools.robot.export_policy import (
        export_esp32_headers,
        export_onnx,
        load_actor_from_checkpoint,
        policy_actor_forward,
        prepare_target_exports,
    )
    from tools.train.pretrain_bc import save_bc_checkpoint, train_behavior_cloning
    from tools.train.train_sac import SquashedGaussianActor, save_sac_checkpoint, train_sac

    rng = np.random.default_rng(9)
    obs_init = rng.standard_normal((32, 37), dtype=np.float32)
    act_init = rng.uniform(-0.5, 0.5, size=(32, 8)).astype(np.float32)
    bc_model, _ = train_behavior_cloning(obs_init, act_init, obs_dim=37, action_dim=8, hidden_dim=64, epochs=2, batch_size=16)
    bc_ckpt = tmp_path / "bc_seed.pt"
    save_bc_checkpoint(bc_ckpt, bc_model, obs_dim=37, action_dim=8, hidden_dim=64)

    sac_ckpt = tmp_path / "cyberduck_v2_sac_elu.pt"
    out_path = train_sac(
        model_path="src/robot/cyberduck_v2.xml",
        num_envs=2,
        total_steps=20,
        warmup_steps=8,
        batch_size=4,
        hidden_dim=64,
        save_path=sac_ckpt,
        activation="elu",
        n_step=3,
        symmetry_coef=0.5,
        augment_symmetry=True,
        use_obs_rms=True,
        load_checkpoint=bc_ckpt,
        max_action_delay=1,
    )
    assert Path(out_path).exists()
    ckpt = torch.load(str(out_path), map_location="cpu", weights_only=False)
    assert ckpt["policy_arch"] == "sac_elu_squashed_gaussian"
    assert ckpt["activation"] == "elu"
    assert ckpt["squash_actions"] is True
    assert "obs_rms" in ckpt

    # 1. Verify ONNX export parity
    loaded_actor = load_actor_from_checkpoint(out_path).eval()
    onnx_path = export_onnx(out_path, tmp_path / "sac_elu.onnx")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    probe = np.linspace(-0.5, 0.5, 37, dtype=np.float32)[None, :]
    with torch.no_grad():
        expected = policy_actor_forward(loaded_actor, torch.from_numpy(probe)).numpy()
    np.testing.assert_allclose(sess.run(None, {"obs": probe})[0], expected, atol=1e-5)

    # 2. Verify ESP32 preflight & INT8 C++ header compilation & numerical parity
    prepared = prepare_target_exports({"esp32": out_path})
    assert "esp32" in prepared
    weights_h, config_h = export_esp32_headers(out_path, tmp_path / "weights.h", tmp_path / "config.h")
    obs_i8 = np.zeros(37, dtype=np.int8)
    obs_i8[0] = 60
    obs_i8[3] = 20
    obs_i8[5] = -120
    cpp_src = tmp_path / "verify_elu_sac.cpp"
    cpp_src.write_text(
        '#include <cstdio>\n#include "' + config_h.name + '"\n#include "' + weights_h.name + '"\n'
        'int main(){ int8_t obs[37]={' + ",".join(map(str, obs_i8)) + '};'
        'int8_t action[8]; evaluate_policy_int8(obs, action);'
        'for(int i=0;i<8;i++) printf("%d ", (int)action[i]); }\n',
        encoding="utf-8",
    )
    exe_path = tmp_path / "verify_elu_sac.exe"
    subprocess.run(["g++", "-std=c++17", str(cpp_src), "-o", str(exe_path)], check=True, capture_output=True, text=True)
    res = subprocess.run([str(exe_path)], check=True, capture_output=True, text=True)
    c_act = np.fromstring(res.stdout, sep=" ")
    dequant_raw = obs_i8.astype(np.float32) * DEFAULT_OBS_BOUNDS / 127.0
    with torch.no_grad():
        py_act = policy_actor_forward(loaded_actor, torch.from_numpy(dequant_raw)).numpy()
    np.testing.assert_allclose(c_act, np.round(py_act * 127.0), atol=3)

    # 3. Ensure ReLU SAC remains explicitly rejected for ESP32 INT8
    relu_actor = SquashedGaussianActor(37, 8, hidden_dim=64, activation="relu")
    relu_ckpt = tmp_path / "sac_relu.pt"
    save_sac_checkpoint(relu_ckpt, relu_actor, torch.zeros(1), 37, 8, hidden_dim=64, model_path="src/robot/cyberduck_v2.xml", observation_schema=ckpt["observation_schema"])
    with pytest.raises(ValueError, match="SAC|ELU|unsupported"):
        export_esp32_headers(relu_ckpt, tmp_path / "bad_w.h", tmp_path / "bad_c.h")


def test_replay_buffer_nstep_accumulation_and_episode_boundary():
    from tools.train.train_sac import ReplayBuffer

    buf = ReplayBuffer(obs_dim=2, act_dim=1, capacity=16, n_step=3, gamma=0.5)
    for t in range(4):
        obs = np.array([[float(t), 0.0]], dtype=np.float32)
        act = np.array([[float(t)]], dtype=np.float32)
        rew = np.array([1.0], dtype=np.float32)
        nxt = np.array([[float(t + 1), 0.0]], dtype=np.float32)
        # Step t=3 hits a time_out truncation (episode_ends=True, true_done=0.0)
        buf.add_batch(obs, act, rew, nxt, np.array([0.0]), episode_ends=np.array([t == 3]))

    # Steps 0..2 form 1st n-step transition (R = 1 + 0.5 + 0.25 = 1.75); steps 1..3 form 2nd
    assert buf.size == 2
    assert buf.rew[0, 0] == pytest.approx(1.75)
    assert buf.obs[0, 0] == pytest.approx(0.0)
    assert buf.next_obs[0, 0] == pytest.approx(3.0)
    assert buf.rew[1, 0] == pytest.approx(1.75)
    assert buf.next_obs[1, 0] == pytest.approx(4.0)
    # Remainder in queue cleared on timeout so next episode is never contaminated
    assert len(buf._env_queues[0]) == 0
