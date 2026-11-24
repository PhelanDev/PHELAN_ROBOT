from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from robot.robot_spec import MODEL_NAME
from sim.env.run_parallel_sim import ParallelSimRunner
from tools.train.train_ppo import train_ppo


def test_multi_env_single_done_does_not_mutate_shared_dynamics_mid_episode():
    runner = ParallelSimRunner(num_envs=4, friction_range=(0.5, 1.1))
    runner.rng = np.random.default_rng(42)
    try:
        runner.reset()
        base_bid = runner.base_body_id
        floor_gid = runner.floor_gid
        assert base_bid != -1 and floor_gid != -1

        mass_before = float(runner.model.body_mass[base_bid])
        inertia_before = runner.model.body_inertia[base_bid].copy()
        friction_before = float(runner.model.geom_friction[floor_gid, 0])

        # Invert ONLY env 0 so it terminates immediately while envs 1..3 remain active.
        runner.datas[0].qpos[3:7] = [0.0, 1.0, 0.0, 0.0]
        mujoco.mj_forward(runner.model, runner.datas[0])

        actions = np.zeros((4, runner.action_dim), dtype=np.float32)
        _, _, dones, _ = runner.step_arrays(actions)
        assert dones.tolist() == [True, False, False, False]

        assert float(runner.model.body_mass[base_bid]) == pytest.approx(mass_before, abs=1e-9)
        np.testing.assert_allclose(runner.model.body_inertia[base_bid], inertia_before, atol=1e-9)
        assert float(runner.model.geom_friction[floor_gid, 0]) == pytest.approx(friction_before, abs=1e-9)

        # Synchronized boundary randomization still mutates the shared model explicitly.
        runner.randomize_dynamics()
        assert (
            float(runner.model.body_mass[base_bid]) != pytest.approx(mass_before, abs=1e-9)
            or float(runner.model.geom_friction[floor_gid, 0]) != pytest.approx(friction_before, abs=1e-9)
        )
    finally:
        runner.close()


def test_fresh_train_ppo_backs_up_pre_schema_checkpoint_instead_of_crashing(tmp_path):
    legacy_ckpt = tmp_path / "cyberduck_v2_ppo.pt"
    torch.save({"model_state_dict": {}}, str(legacy_ckpt))

    result = train_ppo(
        save_path=legacy_ckpt,
        dry_run=True,
        num_envs=2,
        rollout_steps=2,
        max_iterations=1,
        verbose=False,
    )
    assert isinstance(result, dict)
    backup_path = tmp_path / "cyberduck_v2_ppo.legacy_backup.pt"
    assert backup_path.exists()
    saved = torch.load(str(legacy_ckpt), map_location="cpu", weights_only=False)
    assert saved["model_name"] == MODEL_NAME

    # Mismatched model_name must still fail closed even during a fresh run.
    wrong_robot_ckpt = tmp_path / "wrong_robot_ppo.pt"
    torch.save({"model_name": "other_robot.xml", "model_state_dict": {}}, str(wrong_robot_ckpt))
    with pytest.raises(ValueError):
        train_ppo(
            save_path=wrong_robot_ckpt,
            dry_run=True,
            num_envs=2,
            rollout_steps=2,
            max_iterations=1,
            verbose=False,
        )
