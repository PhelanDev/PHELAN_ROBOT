"""Assembly contract checks independent of the manufacturing release gate."""

import pytest

pytest.importorskip('build123d')

import numpy as np
from build123d import Axis, Plane, Pos

from cad.assembly import build_assembly
from cad.catalogue import printable_parts
from cad.collision import interference
from cad.geometry import Component
from cad.params import MG996R, VARIANTS, VARIANT_SUFFIX
from cad.purchased.hardware import flanged_bearing
from cad.skeleton.interfaces import clevis


@pytest.mark.parametrize('stem', ['thigh', 'shin', 'foot'])
def test_right_load_bearing_parts_are_reflections_of_left(stem):
    parts = printable_parts()
    left = parts[f'{stem}_L'][0].mirror(Plane.XZ)
    right = parts[f'{stem}_R'][0]
    assert (left - right).volume == pytest.approx(0, abs=1e-5)
    assert (right - left).volume == pytest.approx(0, abs=1e-5)


@pytest.mark.parametrize('stem', ['thigh', 'shin'])
def test_nominal_assembled_legs_have_symmetric_bounding_boxes(stem):
    parts = {p.name: p for p in build_assembly(wiring=False)}
    left = parts[f'{stem}_L'].shape.mirror(Plane.XZ)
    right = parts[f'{stem}_R'].shape
    for bound in ('min', 'max'):
        np.testing.assert_allclose(tuple(getattr(left.bounding_box(), bound)),
                                   tuple(getattr(right.bounding_box(), bound)), atol=1e-6)


@pytest.mark.parametrize('double', [False, True])
def test_bearings_fit_clevis_including_both_outer_flanges(double):
    # All dimensions are in the common horn-face frame; no adjacent-link exemption.
    from cad.skeleton.interfaces import bearing_locations
    parts = [Component('clevis', 'moving', clevis(double), 'PETG')]
    for i, location in enumerate(bearing_locations(double)):
        parts.append(Component(f'bearing_{i}', 'moving', location * flanged_bearing(), 'STEEL'))
    report = interference(parts)
    assert report['passed'], report['interferences']


@pytest.mark.parametrize('double', [False, True])
@pytest.mark.parametrize('angle', [-90, -50, -30, 0, 30, 45, 50, 85, 90])
def test_clevis_coupon_bridge_clears_interface_envelopes(double, angle):
    from cad.purchased.servo import mg996r
    from cad.skeleton.interfaces import idler_cap, servo_mount
    base = Pos(0, 0, -MG996R.horn_face)
    parts = [Component('clevis', 'moving', clevis(double).rotate(Axis.Z, angle), 'PETG'),
             Component('servo', 'fixed', base * mg996r(), 'SERVO'),
             Component('cassette', 'fixed', base * servo_mount(), 'PETG'),
             Component('cup', 'fixed', base * idler_cap(), 'PETG')]
    report = interference(parts)
    assert report['passed'], report['interferences']


@pytest.mark.parametrize('variant', VARIANTS)
def test_electronics_tray_boards_clamps_screws_and_cooler_fit_as_a_module(variant):
    from cad.skeleton.electronics import TRAY_LAYOUTS
    wanted = {f'trunk_back_sled_{VARIANT_SUFFIX[variant]}', 'active_cooler'}
    wanted.update(key for key, _ in TRAY_LAYOUTS[variant])
    parts = [p for p in build_assembly(variant, wiring=False)
             if p.name in wanted or p.name.startswith(('clamp_', 'clamp_screw_', 'clamp_insert_'))]
    expected = 1 + len(TRAY_LAYOUTS[variant]) + (
        1 if variant == 'rpi5' else 0) + 6 * len(TRAY_LAYOUTS[variant])
    assert len(parts) == expected
    assert any(p.name == f'trunk_back_sled_{VARIANT_SUFFIX[variant]}' for p in parts)
    report = interference(parts)
    assert report['passed'], report['interferences']


@pytest.mark.parametrize('variant', VARIANTS)
def test_backpack_housing_lid_and_tray_are_geometrically_compatible(variant):
    from cad.skeleton.electronics import TRAY_LAYOUTS
    suffix = VARIANT_SUFFIX[variant]
    wanted = {f'trunk_back_sled_{suffix}', f'trunk_back_cover_{suffix}', 'active_cooler'}
    wanted.update(key for key, _ in TRAY_LAYOUTS[variant])
    parts = [p for p in build_assembly(variant, wiring=False)
             if p.name in wanted or p.name.startswith(('clamp_', 'sled_screw_', 'sled_insert_',
                                                       'hatch_screw_', 'hatch_insert_'))]
    expected = 2 + len(TRAY_LAYOUTS[variant]) + (
        1 if variant == 'rpi5' else 0) + 8 + 6 * len(TRAY_LAYOUTS[variant])
    assert len(parts) == expected
    assert any(p.name == f'trunk_back_cover_{suffix}' for p in parts)
    assert any(p.name == f'trunk_back_sled_{suffix}' for p in parts)
    report = interference(parts)
    assert report['passed'], report['interferences']


def test_battery_pack_foam_and_sliding_hatch_do_not_interpenetrate():
    wanted = {'trunk_bottom', 'battery_hatch', 'battery_foam_spacer',
              'battery', 'battery_connector', 'main_power'}
    parts = [p for p in build_assembly() if p.name in wanted]
    assert len(parts) == len(wanted)
    report = interference(parts)
    assert report['passed'], report['interferences']
