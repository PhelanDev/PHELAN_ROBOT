"""Fail-closed numerical and evidence boundaries in the CAD pipeline."""

import json

import numpy as np
import pytest

pytest.importorskip('build123d')
trimesh = pytest.importorskip('trimesh')

from build123d import Box, Pos
from cad.collision import interference
from cad.geometry import Component
from cad.kinematics import frame_transforms
from cad.mass_properties import mass_properties
from cad.print_plates import pack_plates


@pytest.mark.parametrize('tolerance', [-1, np.nan, np.inf, -np.inf])
def test_interference_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match='tolerance'):
        interference([], tolerance_mm3=tolerance)


@pytest.mark.parametrize('limit', [0, -1, 1.5, True])
def test_interference_rejects_invalid_early_stop_limit(limit):
    with pytest.raises(ValueError, match='limit'):
        interference([], limit=limit)


def test_interference_threshold_is_strict_and_partial_report_cannot_pass():
    a = Component('a', 'a', Box(10, 10, 10), 'PETG')
    b = Component('b', 'b', Pos(5, 0, 0) * Box(10, 10, 10), 'PETG')
    report = interference([a, b], limit=1)
    volume = report['interferences'][0]['volume_mm3']
    assert not report['complete'] and not report['passed']
    assert interference([a, b], tolerance_mm3=volume)['passed']
    assert not interference([a, b], tolerance_mm3=np.nextafter(volume, -np.inf))['passed']


@pytest.mark.parametrize('pose', [{'head_yaw': np.nan}, {'head_pitch': np.inf},
                                {'left_knee_pich': 0.2}, {'head_yaw': 'wrong'}])
def test_kinematics_rejects_invalid_or_misspelled_pose(pose):
    with pytest.raises(ValueError):
        frame_transforms(pose)


@pytest.mark.parametrize('mass', [0, -0.1, np.nan, np.inf])
def test_mass_integration_rejects_invalid_catalog_mass(mass):
    with pytest.raises(ValueError, match='mass'):
        mass_properties([Component('part', 'trunk', Box(10, 10, 10), 'PETG', mass_kg=mass)])


def test_mass_integration_rejects_empty_assembly():
    with pytest.raises(ValueError, match='empty'):
        mass_properties([])


@pytest.mark.parametrize('bed,gap', [(np.nan, 5), (256, -1), (0, 5), (10, 5)])
def test_print_plates_reject_invalid_bed_configuration(tmp_path, bed, gap):
    with pytest.raises(ValueError, match='bed|gap'):
        pack_plates([], tmp_path, bed=bed, gap=gap)


def test_duplicate_instance_names_do_not_silently_replace_3mf_geometry(tmp_path):
    cube = trimesh.creation.box((10, 10, 10))
    with pytest.raises(ValueError, match='unique'):
        pack_plates([('copy', cube, 'PETG'), ('copy', cube, 'PETG')], tmp_path)


def test_release_requires_dfam_and_complete_hardware_even_if_nominal_and_samples_pass():
    from cad.release import assess_release
    clean_pose = {'pose': 'straight', 'geometry': {'passed': True, 'complete': True},
                  'wires': [{'passed': True}], 'mass': {'within_plan_budget': True}}
    assemblies = [{'variant': v, 'sweep_requested': True, 'sweep_completed': True,
                   'passed': True, 'poses': [clean_pose]} for v in ('rpi5', 'pizero2w', 'dual_esp32s3')]
    result = assess_release([{'passed': True}], assemblies)
    assert not result['release_ready']
    assert 'dfam_wall_thickness' in result['unmet_gates']
    assert 'complete_fastener_mating' in result['unmet_gates']


def test_source_receipt_changes_with_inputs_but_ignores_generated_outputs(tmp_path):
    from cad.provenance import source_receipt
    source = tmp_path / 'cad'
    source.mkdir(parents=True)
    (source / 'params.py').write_text('length = 68\n', encoding='utf-8')
    before = source_receipt(tmp_path)
    assert before == source_receipt(tmp_path)
    (source / '__pycache__').mkdir()
    (source / '__pycache__/params.pyc').write_bytes(b'compiled')
    generated = tmp_path / 'artifacts/cad'
    generated.mkdir(parents=True)
    (generated / 'verification.json').write_text('{}', encoding='utf-8')
    assert before == source_receipt(tmp_path)
    (source / 'params.py').write_text('length = 70\n', encoding='utf-8')
    assert before['sha256'] != source_receipt(tmp_path)['sha256']
    assert json.loads(json.dumps(before)) == before


def test_assess_release_wires_live_dfam_report():
    from cad.release import assess_release

    assert assess_release([], [])['gates']['dfam_wall_thickness'] is False
    clean_report = {'parts': {'trunk_front': {'observed_below_requirement': False}}}
    thin_report = {'parts': {'trunk_front': {'observed_below_requirement': True}}}
    assert assess_release([], [], dfam_report=clean_report)['gates']['dfam_wall_thickness'] is True
    assert assess_release([], [], dfam_report=thin_report)['gates']['dfam_wall_thickness'] is False
