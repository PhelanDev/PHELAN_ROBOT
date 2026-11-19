"""V2 identity and dimensions remain consistent across training consumers."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from robot.robot_spec import JOINT_ORDER, MODEL_NAME, MODEL_PATH, TOTAL_MASS_KG
from tools.train.evaluate_models_sim import load_actor
from tools.robot.interactive_viewer import load_viewer_policy, policy_step, resolve_viewer_model_path
from tools.train.pretrain_bc import collect_teacher_demonstrations, save_bc_checkpoint
from tools.train.train_ppo import ActorCritic, train_ppo
from tools.train.train_sac import train_sac
from sim.env.env_config import MicroDuck8DoFConfig
from sim.env.run_parallel_sim import ParallelSimRunner, resolve_v2_model_path


def test_default_runner_is_v2_with_canonical_observations():
    cfg = MicroDuck8DoFConfig()
    assert cfg.STACK_NAME == "cyberduck_v2"
    assert cfg.TARGET_MASS == pytest.approx(TOTAL_MASS_KG)
    assert cfg.JOINT_NAMES == list(JOINT_ORDER)
    assert (cfg.OBS_DIM, cfg.ACTION_DIM) == (42, 8)
    runner = ParallelSimRunner(num_envs=1, num_threads=1)
    try:
        assert runner.model_path == MODEL_PATH.resolve()
        assert runner.embed_phase_clock is True
        obs, rewards, _, _ = runner.step(np.zeros((1, 8), dtype=np.float32))
        assert obs.shape == (1, 42)
        assert np.isfinite(obs).all() and np.isfinite(rewards).all()
    finally:
        runner.close()
    assert Path(resolve_viewer_model_path()) == MODEL_PATH.resolve()


@pytest.mark.parametrize("entrypoint", ["runner", "ppo", "sac", "viewer"])
def test_retired_v1_model_rejected_before_checkpoint_write(tmp_path, entrypoint):
    retired = "models/mjcf/microduck_8dof.xml"
    output = tmp_path / "out" / "policy.pt"
    with pytest.raises(ValueError, match="V1|V2"):
        if entrypoint == "runner":
            ParallelSimRunner(model_path=retired, num_envs=1)
        elif entrypoint == "ppo":
            train_ppo(model_path=retired, save_path=output, dry_run=True, verbose=False)
        elif entrypoint == "sac":
            train_sac(model_path=retired, save_path=output, total_steps=0)
        else:
            resolve_viewer_model_path(retired)
    assert not output.parent.exists()


def test_unknown_mjcf_identity_cannot_be_renamed_to_v2(tmp_path):
    renamed = tmp_path / MODEL_NAME
    renamed.write_text('<mujoco model="microduck_8dof"/>', encoding="utf-8")
    with pytest.raises(ValueError, match="V2"):
        resolve_v2_model_path(renamed)


@pytest.mark.parametrize("metadata", [{}, {"model_name": "microduck_8dof.xml"}])
def test_viewer_and_evaluator_require_declared_v2_checkpoint_identity(tmp_path, metadata):
    checkpoint = tmp_path / "cyberduck_v2_best.pt"
    torch.save(metadata, checkpoint)
    with pytest.raises(ValueError, match="identity"):
        resolve_viewer_model_path(policy_path=checkpoint)
    with pytest.raises(ValueError, match="identity"):
        load_viewer_policy(checkpoint)
    with pytest.raises(ValueError, match="identity"):
        load_actor(checkpoint)


def test_viewer_accepts_relocated_v2_identity_and_rejects_observation_adaptation(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    torch.save({"model_path": r"Z:\former-machine\robot\cyberduck_v2.xml", "model_name": MODEL_NAME}, checkpoint)
    assert Path(resolve_viewer_model_path(policy_path=checkpoint)) == MODEL_PATH.resolve()
    with pytest.raises(ValueError, match="42"):
        policy_step(nn.Linear(42, 8), np.zeros(37, dtype=np.float32))
    with pytest.raises(ValueError, match="mismatch"):
        policy_step(nn.Linear(37, 8), np.zeros(42, dtype=np.float32))


def test_bc_writes_v2_metadata_and_teacher37_fails(tmp_path):
    checkpoint = tmp_path / "bc.pt"
    save_bc_checkpoint(checkpoint, ActorCritic(hidden_dim=16), 42, 8, hidden_dim=16)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["model_name"] == MODEL_NAME
    assert Path(payload["model_path"]) == MODEL_PATH
    assert payload["observation_schema"]["obs_dim"] == 42
    assert payload["observation_schema"]["embed_phase_clock"] is True
    runner = ParallelSimRunner(num_envs=1, num_threads=1)
    try:
        with pytest.raises(ValueError, match="42-observation"):
            collect_teacher_demonstrations(runner, nn.Linear(37, 8), num_steps=1)
    finally:
        runner.close()


def test_trainers_save_canonical_v2_identity_for_copied_mjcf(tmp_path):
    copied_model = tmp_path / "calibrated_robot.xml"
    copied_model.write_text(MODEL_PATH.read_text(encoding="utf-8").replace(
        'meshdir="meshes"', f'meshdir="{(MODEL_PATH.parent / "meshes").as_posix()}"'), encoding="utf-8")
    ppo_checkpoint = tmp_path / "ppo.pt"
    result = train_ppo(model_path=copied_model, num_envs=1, rollout_steps=2, max_iterations=1,
                       save_path=ppo_checkpoint, dry_run=True, verbose=False, device="cpu")
    assert result["success"] is True
    sac_checkpoint = train_sac(model_path=copied_model, num_envs=1, total_steps=0,
                               hidden_dim=16, save_path=tmp_path / "sac.pt")
    for checkpoint in (ppo_checkpoint, sac_checkpoint):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["model_name"] == MODEL_NAME
        assert Path(payload["model_path"]) == MODEL_PATH
        assert (payload["obs_dim"], payload["action_dim"]) == (42, 8)
        assert Path(resolve_viewer_model_path(policy_path=checkpoint)) == MODEL_PATH.resolve()
        assert load_actor(checkpoint).obs_dim == 42
