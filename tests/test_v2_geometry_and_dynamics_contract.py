"""Probe geometry consumers and variant dynamics against real MuJoCo models."""

from contextlib import closing
import numpy as np
import pytest

from duck_v2.robot_spec import BASE_HEIGHT_M
from training.env_config import MicroDuck8DoFConfig
from training.run_parallel_sim import ParallelSimRunner
from training.self_righting import SelfRightingController


def test_reset_follows_loaded_geometry_instead_of_old_height_literal():
    with closing(ParallelSimRunner(num_envs=1)) as runner:
        runner.model.qpos0[2] = 0.193
        runner.reset_env(0)
        assert runner.datas[0].qpos[2] == pytest.approx(0.193)
    assert SelfRightingController.DEFAULT_STAND_MIN_Z == pytest.approx(0.85 * BASE_HEIGHT_M)


def test_variant_randomization_updates_mass_inertia_com_and_derived_constants():
    cfg = MicroDuck8DoFConfig(RANDOMIZE_HARDWARE_VARIANTS=True,
                            TORSO_MASS_RANDOMIZATION_KG=0.08, TORSO_COM_RANDOMIZATION_M=0.006)
    with closing(ParallelSimRunner(num_envs=2, cfg=cfg)) as runner:
        bid = runner.base_body_id
        baseline = (runner.default_base_mass, runner.default_base_inertia.copy(),
                    runner.default_base_com.copy())
        masses, centers = [], []
        for _ in range(20):
            runner._randomize_model_dynamics()
            mass = runner.model.body_mass[bid]
            masses.append(mass)
            centers.append(runner.model.body_ipos[bid].copy())
            assert abs(mass - baseline[0]) <= 0.08
            np.testing.assert_allclose(runner.model.body_inertia[bid], baseline[1] * mass / baseline[0])
            assert np.max(np.abs(runner.model.body_ipos[bid] - baseline[2])) <= 0.006
            assert runner.model.body_subtreemass[bid] == pytest.approx(runner.model.body_mass.sum())
        assert np.ptp(masses) > 0.10
        assert np.max(np.ptp(centers, axis=0)) > 0.006
        before = runner.model.body_mass.copy()
        runner.reset_env(1)
        np.testing.assert_array_equal(before, runner.model.body_mass)


@pytest.mark.parametrize('value', [-1.0, np.nan, np.inf])
@pytest.mark.parametrize('field', ['TORSO_MASS_RANDOMIZATION_KG', 'TORSO_COM_RANDOMIZATION_M'])
def test_variant_randomization_rejects_invalid_amplitudes(field, value):
    cfg = MicroDuck8DoFConfig(**{field: value})
    with pytest.raises(ValueError, match='randomization'):
        ParallelSimRunner(num_envs=1, cfg=cfg)


def test_zero_variant_amplitudes_preserve_nominal_inertial_values():
    cfg = MicroDuck8DoFConfig(RANDOMIZE_HARDWARE_VARIANTS=True,
                            TORSO_MASS_RANDOMIZATION_KG=0, TORSO_COM_RANDOMIZATION_M=0)
    with closing(ParallelSimRunner(num_envs=1, cfg=cfg)) as runner:
        for _ in range(3):
            runner._randomize_model_dynamics()
            bid = runner.base_body_id
            assert runner.model.body_mass[bid] == runner.default_base_mass
            np.testing.assert_array_equal(runner.model.body_ipos[bid], runner.default_base_com)
            np.testing.assert_allclose(runner.model.body_inertia[bid], runner.default_base_inertia)


def test_randomized_torso_mass_cannot_cross_zero():
    cfg = MicroDuck8DoFConfig(RANDOMIZE_HARDWARE_VARIANTS=True, TORSO_MASS_RANDOMIZATION_KG=1)
    with pytest.raises(ValueError, match='positive torso mass'):
        ParallelSimRunner(num_envs=2, cfg=cfg)
