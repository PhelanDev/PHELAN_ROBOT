"""Analytical wall probes, including limits of a finite sampling method."""

import pytest

pytest.importorskip('build123d')

from build123d import Box, Pos
from duck_v2.cad_src.dfam import sample_wall_thickness


@pytest.mark.parametrize('wall', [1.2, 2.2, 3.0])
def test_box_wall_probe_measures_known_thickness_and_reports_thin_region(wall):
    shape = Box(30, 40, wall)
    report = sample_wall_thickness(shape, 2.5)
    assert report['sampled_min_mm'] == pytest.approx(wall)
    assert report['observed_below_requirement'] == (wall < 2.5)
    assert report['samples_tested'] > 0
    assert not report['certified_global_minimum']


def test_hollow_part_probe_crosses_material_and_not_the_air_cavity():
    shell = Box(30, 40, 20) - Pos(0, 0, 2.2) * Box(25.6, 35.6, 20)
    report = sample_wall_thickness(shell, 1.6)
    assert report['sampled_min_mm'] == pytest.approx(2.2, abs=1e-6)
    assert not report['observed_below_requirement']
    assert not report['certified_global_minimum']


def test_small_sample_budget_is_explicitly_incomplete():
    report = sample_wall_thickness(Box(10, 20, 30), 2.5, max_samples=1)
    assert report['samples_tested'] == 1
    assert report['faces_sampled'] < report['face_count']
    assert not report['certified_global_minimum']
