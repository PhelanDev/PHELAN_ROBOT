"""Mechanical unit tests; the assembly release gate runs separately and may fail."""

from collections import Counter
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pytest

pytest.importorskip('build123d')
trimesh = pytest.importorskip('trimesh')

from build123d import Box, CenterOf, Pos, Vector, import_step
import numpy as np

from cad.assembly import build_assembly
from cad.catalogue import printable_parts
from cad.collision import interference
from cad.export_pipeline import clean_tessellation
from cad.geometry import Component
from cad.mass_properties import mass_properties
from cad.params import VARIANTS
from cad.print_plates import pack_plates
from cad.purchased.catalogue import purchased_parts
from cad.purchased.electronics import BOARDS, board_model
from cad.skeleton.interfaces import bearing_plate, insert_boss
from cad.validation import sample_poses
from cad.wiring.routes import measure_harnesses


@pytest.mark.parametrize('name', printable_parts())
def test_each_printable_is_a_single_connected_valid_solid(name):
    shape = printable_parts()[name][0]
    assert shape.is_valid
    assert len(shape.solids()) == 1
    assert shape.volume > 0


@pytest.mark.parametrize('name', purchased_parts())
def test_purchased_reference_step_roundtrip_has_matching_volume(name):
    from pathlib import Path
    shape, _ = purchased_parts()[name]
    path = Path(__file__).resolve().parents[2] / 'src/cad/purchased/step' / f'{name}.step'
    imported = import_step(path)
    assert imported.is_valid
    assert imported.volume == pytest.approx(shape.volume, rel=1e-8)
    np.testing.assert_allclose(tuple(imported.bounding_box().size), tuple(shape.bounding_box().size), atol=1e-6)


@pytest.mark.parametrize('key', ['raspberry_pi_4b', 'raspberry_pi_5', 'raspberry_pi_zero_2w',
                                 'pi_camera_3', 'oled_242'])
def test_pcb_holes_remain_native_sizes(key):
    board, shape = BOARDS[key], board_model(key)
    for x, y in board.holes:
        assert not shape.is_inside(Vector(x + board.hole / 2 - 0.02, y, 0.8))
        assert shape.is_inside(Vector(x + board.hole / 2 + 0.02, y, 0.8))


def test_heat_insert_and_bearing_seat_diameters_are_physical_cuts():
    boss = insert_boss(0, 0)
    assert not boss.is_inside(Vector(2.025 - 0.01, 0, 6))
    assert boss.is_inside(Vector(2.025 + 0.01, 0, 6))
    for double in (False, True):
        seat = bearing_plate(double)
        assert not seat.is_inside(Vector(5.075 - 0.01, 0, 2))
        assert seat.is_inside(Vector(5.075 + 0.01, 0, 2))
        depth = 8.5 if double else 4.5
        assert not seat.is_inside(Vector(5.9 - 0.01, 0, depth - 0.5))
        assert seat.is_inside(Vector(5.9 + 0.01, 0, depth - 0.5))
        if double:
            assert not seat.is_inside(Vector(5.9 - 0.01, 0, 0.5))
            assert seat.is_inside(Vector(5.9 + 0.01, 0, 0.5))


def test_interference_detects_known_volume_and_does_not_exempt_neighbours():
    a = Component('a', 'left_thigh', Box(10, 10, 10), 'PETG')
    b = Component('b', 'left_shin', Pos(5, 0, 0) * Box(10, 10, 10), 'PETG')
    report = interference([a, b])
    assert not report['passed'] and report['complete']
    assert report['interferences'][0]['volume_mm3'] == pytest.approx(500)
    c = Component('c', 'right_shin', Pos(10, 0, 0) * Box(10, 10, 10), 'PETG')
    assert interference([a, c])['passed']


@pytest.mark.parametrize('variant', VARIANTS)
def test_assemblies_include_eight_servos_eleven_f623_one_neck_bearing(variant):
    assembly = build_assembly(variant)
    names = [part.name for part in assembly]
    assert len(names) == len(set(names))
    assert sum(n.startswith('servo_') for n in names) == 8
    assert sum(n.startswith('f623_') for n in names) == 11
    assert names.count('neck_6803') == 1
    assert not any(n.startswith(('cassette_', 'idler_cap_')) for n in names)
    assert sum(n.startswith('tof_sensor_') for n in names) == 1
    assert 'oled_display' in names
    assert 'tof_mux' not in names
    assert ('head_mcu1' in names) == (variant == 'dual_esp32s3')
    assert ('pca9685' in names) == (variant != 'dual_esp32s3')
    assert all(p.shape.is_valid for p in assembly)
    for route in measure_harnesses(variant):
        assert route['passed']


def test_sweeps_use_shared_postures_and_explicit_inward_hip_corner():
    poses = dict(sample_poses(9))
    assert len(poses) == 5 + 8 * 9
    assert {'stand', 'sit', 'lie', 'both_hips_inward'} <= poses.keys()
    with pytest.raises(ValueError):
        list(sample_poses(1))


def test_mass_properties_integrate_si_units_and_parallel_axis():
    parts = [Component('a', 'trunk', Box(10, 10, 10), 'PETG', mass_kg=1),
             Component('b', 'trunk', Pos(20, 0, 0) * Box(10, 10, 10), 'PETG', mass_kg=1)]
    props = mass_properties(parts)
    assert props['mass_kg'] == 2
    np.testing.assert_allclose(props['com_m'], [0.01, 0, 0], atol=1e-12)
    # Each 1 kg cube contributes m*(a²+b²)/12; both CoMs are 10 mm from the pair CoM.
    own = 2 * (0.01**2 + 0.01**2) / 12
    np.testing.assert_allclose(np.diag(props['inertia_kg_m2']), [own, own + 2*0.01**2, own + 2*0.01**2])


def test_3mf_roundtrip_and_plate_layout_preserve_quantities_units_and_bed(tmp_path):
    mesh = trimesh.creation.box((100, 100, 3))
    items = [(f'copy_{i}', mesh, 'PETG') for i in range(5)]
    plates = pack_plates(items, tmp_path)
    assert len(plates) == 2
    assert sum(len(p['instances']) for p in plates) == 5
    for plate in plates:
        path = tmp_path / plate['file']
        with ZipFile(path) as zipped:
            model = ET.fromstring(zipped.read('3D/3dmodel.model'))
        assert model.get('unit') == 'millimeter'
        loaded = trimesh.load_scene(path)
        assert len(loaded.graph.nodes_geometry) == len(plate['instances'])
        assert loaded.bounds[0].min() >= -1e-6
        assert max(loaded.bounds[1][:2]) <= 256


def test_stl_cleanup_removes_only_zero_area_facets_and_preserves_volume():
    mesh = trimesh.creation.box()
    vertices = np.vstack((mesh.vertices, mesh.vertices[0]))
    faces = np.vstack((mesh.faces, [0, len(vertices) - 1, len(vertices) - 1]))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    cleaned, removed, relative_error = clean_tessellation(mesh, 'test_cube', 1.0)
    assert removed == 1
    assert cleaned.is_watertight and cleaned.is_volume
    assert len(cleaned.split(only_watertight=False)) == 1
    assert cleaned.volume == pytest.approx(1.0)
    assert relative_error == pytest.approx(0.0)


def test_stl_cleanup_does_not_repair_a_real_open_boundary():
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    with pytest.raises(ValueError, match='Non-watertight'):
        clean_tessellation(mesh, 'open_cube')
