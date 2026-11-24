"""Assembly contract checks independent of the manufacturing release gate."""

import pytest

pytest.importorskip('build123d')

import numpy as np
from build123d import Axis, Plane, Pos, Rot, Vector

from cad.assembly import build_assembly
from cad.catalogue import printable_parts
from cad.collision import interference
from cad.dfam import required_wall, sample_wall_thickness
from cad.geometry import Component, box, cylinder
from cad.layout import CAMERA_LOCATION, OLED_LOCATION, STATIONS, STATION_BY_JOINT
from cad.params import MG996R, VARIANTS, VARIANT_SUFFIX
from cad.purchased.catalogue import purchased_parts
from cad.purchased.electronics import BOARDS, board_model
from cad.purchased.hardware import flanged_bearing
from cad.skeleton.interfaces import clevis


@pytest.mark.parametrize('stem', ['thigh', 'thigh_cover', 'thigh_idler_cap', 'shin', 'shin_idler_cap', 'foot'])
def test_right_load_bearing_parts_are_reflections_of_left(stem):
    parts = printable_parts()
    left = parts[f'{stem}_L'][0].mirror(Plane.XZ)
    right = parts[f'{stem}_R'][0]
    assert (left - right).volume == pytest.approx(0, abs=1e-5)
    assert (right - left).volume == pytest.approx(0, abs=1e-5)


@pytest.mark.parametrize('stem', ['thigh', 'thigh_cover', 'thigh_idler_cap', 'shin', 'shin_idler_cap'])
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
              'battery', 'battery_connector', 'main_power',
              'belly_insert_0', 'belly_insert_1', 'belly_screw_0', 'belly_screw_1'}
    parts = [p for p in build_assembly() if p.name in wanted]
    assert len(parts) == len(wanted)
    report = interference(parts)
    assert report['passed'], report['interferences']


def test_universal_belly_bay_clears_stacked_103450_dual_18650_xt30_channel_and_m3_bosses():
    tb = printable_parts()['trunk_bottom'][0]
    bh = printable_parts()['battery_hatch'][0]
    # Open XT30 wire tunnel from cavity to rear bay and open underside access.
    assert not tb.is_inside(Vector(-41.25, 0, -36.5))
    assert not tb.is_inside(Vector(-49.5, 0, -49.0))
    # Physical Ø4.05 mm M3 heat-set insert pockets inside solid PETG rear bosses at X=-47.5, Y=±18.5.
    for sign in (-1, 1):
        assert not tb.is_inside(Vector(-47.5 + 2.00, sign * 18.5, -47.0))
        assert tb.is_inside(Vector(-47.5 + 2.10, sign * 18.5, -47.0))
    belly_fasteners = [
        p for p in build_assembly(wiring=False)
        if p.name in {'belly_insert_0', 'belly_insert_1', 'belly_screw_0', 'belly_screw_1'}
    ]
    assert len(belly_fasteners) == 4
    for name, bshape in (
        ('stacked_2x103450', Pos(0, 0, -38.5) * box((55.0, 35.0, 21.0))),
        ('dual_18650_holder', Pos(0, 0, -38.5) * box((77.0, 41.0, 21.0))),
    ):
        report = interference([
            Component('trunk_bottom', 'trunk_base', tb, 'PETG'),
            Component('battery_hatch', 'trunk_base', bh, 'PETG'),
            Component(name, 'trunk_base', bshape, 'BATTERY'),
            *belly_fasteners,
        ])
        assert report['passed'], (name, report['interferences'])


def test_microduck_leg_outer_haunch_flat_inner_plate_and_sleek_shin_bounds():
    parts = printable_parts()
    hc = parts['hip_clevis_L'][0]
    th = parts['thigh_L'][0]
    tc = parts['thigh_cover_L'][0]
    sh = parts['shin_L'][0]
    assert hc.is_valid and len(hc.solids()) == 1
    assert tc.is_valid and len(tc.solids()) == 1
    # Detachable Microduck outer haunch cover (Y in [32.0, 40.6]) spans hip through knee with solid bottom lip.
    assert tc.is_inside(Vector(0.0, 34.0, -74.0))
    assert tc.is_inside(Vector(12.0, 36.5, -60.0))
    assert tc.is_inside(Vector(10.0, 38.0, -35.0))
    assert tc.bounding_box().max.Y <= 41.5
    # Structural thigh chassis has flat inner rigidity plate (Y in [-27.2, -24.2]) & twin Ø4.05 mm M3 insert pockets.
    assert not th.is_inside(Vector(3.0, -25.7, -26.0))
    assert th.is_inside(Vector(14.0, -25.7, -34.0))
    for bx, bz in ((-7.0, -37.5), (17.5, -23.5)):
        assert not th.is_inside(Vector(bx + 2.00, 28.0, bz))
        assert th.is_inside(Vector(bx + 2.15, 28.0, bz))
    # Physical MG996R +Y drop-in insertion clearance (open side mouth before cover installation).
    pp = purchased_parts()
    st_kp = STATION_BY_JOINT['left_knee_pitch']
    st_hp = STATION_BY_JOINT['left_hip_pitch']
    mg_base = st_kp.servo_base * pp['mg996r'][0]
    for dy in (0, 2, 5, 10, 15, 20, 30, 45, 60):
        assert ((Pos(0, dy, 0) * mg_base) & th).volume <= 1e-4, f'MG996R drop-in blocked at dy={dy}'
    # Physical Ø5.6 mm straight-in Allen-wrench clearance for all 12 screws (4x knee ear, 4x knee horn, 4x hip horn).
    for hx, hy in MG996R.ear_holes:
        pt = (st_kp.servo_base * Pos(hx, hy, MG996R.ear_z + MG996R.ear_thickness + 3.0)).position
        shaft = Pos(pt.X, pt.Y + 0.5, pt.Z) * Rot(-90, 0, 0) * cylinder(5.6, 60.0)
        assert (shaft & th).volume <= 1e-4, f'knee ear wrench blocked at {(hx, hy)}'
    for hx, hy in ((7, 0), (-7, 0), (0, 7), (0, -7)):
        for jname, loc in (('hip_horn', st_hp.moving_location), ('knee_horn', st_kp.location)):
            pt = (loc * Pos(hx, hy, 6.0)).position
            shaft = Pos(pt.X, pt.Y + 0.5, pt.Z) * Rot(-90, 0, 0) * cylinder(5.6, 60.0)
            assert (shaft & th).volume <= 1e-4, f'{jname} wrench blocked at {(hx, hy)}'
    # Sleek shin tucked behind outer thigh cover without protruding hub caps.
    assert sh.bounding_box().max.Y <= 32.0
    assert sh.bounding_box().min.Y >= -35.0
    # Detachable inboard idler nut covers (single solid, uniform DfAM wall, straight-in 5.5 mm socket access).
    for cap_name, expected_mat in (('thigh_idler_cap_L', 'PLA'), ('shin_idler_cap_L', 'PETG')):
        cap_shape, cap_mat, _, _ = parts[cap_name]
        assert cap_mat == expected_mat
        assert cap_shape.is_valid and len(cap_shape.solids()) == 1
        wall_rep = sample_wall_thickness(cap_shape, required_wall(cap_mat), max_samples=256)
        assert wall_rep['observed_below_requirement'] is False, (cap_name, wall_rep)
    hip_drv = Pos(0, -31.75, 0) * Rot(90, 0, 0) * cylinder(10.5, 45.0)
    knee_drv = Pos(0, -31.75, 0) * Rot(90, 0, 0) * cylinder(10.5, 45.0)
    assert (hip_drv & th).volume <= 1e-4
    assert (knee_drv & sh).volume <= 1e-4
    # Narrow stance overall width <= 214.0 mm, exact Z = 0.000 ground datum, and all fasteners/caps present.
    straight = build_assembly(wiring=False)
    by_name = {p.name: p for p in straight}
    names = set(by_name)
    assert {f'thigh_cover_insert_{s}_{i}' for s in ('L', 'R') for i in (0, 1)} <= names
    assert {f'thigh_cover_screw_{s}_{i}' for s in ('L', 'R') for i in (0, 1)} <= names
    assert {'thigh_idler_cap_L', 'thigh_idler_cap_R', 'shin_idler_cap_L', 'shin_idler_cap_R'} <= names
    assert by_name['idler_screw_left_hip_pitch'].sku == 'm3x18'
    assert by_name['idler_screw_left_knee_pitch'].sku == 'm3x14'
    assert by_name['thigh_idler_cap_L'].shape.bounding_box().min.Y < by_name['idler_screw_left_hip_pitch'].shape.bounding_box().min.Y
    assert by_name['shin_idler_cap_L'].shape.bounding_box().min.Y < by_name['idler_screw_left_knee_pitch'].shape.bounding_box().min.Y
    assert max(p.shape.bounding_box().max.Y for p in straight) <= 107.0
    assert abs(min(p.shape.bounding_box().min.Z for p in straight)) < 1e-4
    nominal_report = interference(build_assembly(pose=None, wiring=True))
    assert nominal_report['passed'], nominal_report['interferences']


@pytest.mark.slow
def test_microduck_leg_multi_pose_sweep_with_wiring():
    for pose in (
        {'left_hip_roll': -0.22, 'right_hip_roll': 0.22},
        {'left_hip_roll': -0.44},
        {'left_hip_roll': 0.44},
        {'left_hip_pitch': -0.87},
        {'left_hip_pitch': 0.87},
        {'left_knee_pitch': 1.48},
    ):
        report = interference(build_assembly(pose=pose, wiring=True))
        assert report['passed'], (pose, report['interferences'])


def test_all_eight_servos_have_unobstructed_wire_exit_channels():
    parts = printable_parts()
    host_by_joint = {
        'head_yaw': parts['trunk_front'][0],
        'head_pitch': parts['neck_link'][0],
        'left_hip_roll': parts['trunk_front'][0],
        'right_hip_roll': parts['trunk_front'][0],
        'left_hip_pitch': parts['hip_clevis_L'][0],
        'right_hip_pitch': parts['hip_clevis_R'][0],
        'left_knee_pitch': parts['thigh_L'][0],
        'right_knee_pitch': parts['thigh_R'][0],
    }
    assert set(host_by_joint) == {st.joint for st in STATIONS}
    end = MG996R.body_x_min + MG996R.body[0]
    for st in STATIONS:
        stub = st.servo_base * box((12.0, 8.0, 6.0), (end + 6.0, 0.0, 3.0))
        assert (stub & host_by_joint[st.joint]).volume <= 1e-4, f'Wire exit blocked for {st.joint}'
    wired = build_assembly('rpi5', wiring=True)
    by_name = {c.name: c for c in wired}
    expected_wires = {
        f'{prefix}_{side}'
        for prefix in ('knee_pitch_wire', 'hip_pitch_wire', 'leg_harness')
        for side in ('left', 'right')
    }
    assert expected_wires <= set(by_name)
    for wname in expected_wires:
        shape = by_name[wname].shape
        assert shape.is_valid and len(shape.solids()) == 1


def test_verified_electronics_envelopes_and_universal_cam_a_clear_head_and_backpack():
    assert BOARDS['oled_242'].size == (43.5, 72.0, 1.6)
    assert BOARDS['oled_242'].pitch == (39.0, 68.0)
    assert BOARDS['oled_242'].hole == pytest.approx(2.8)
    assert len(BOARDS['oled_242'].rear_features) == 1
    assert BOARDS['pi_camera_3'].features[0][0] == (10.8, 10.8, 10.8)
    assert BOARDS['raspberry_pi_4b'].features[-1][0] == (51, 5, 8.5)

    hf = printable_parts()['head_front'][0]
    for label, obj in (
        ('pi_camera_3', CAMERA_LOCATION * board_model('pi_camera_3')),
        ('mini_usb_25x25', CAMERA_LOCATION * box((25.0, 25.0, 14.0), (0, 0, 7.0))),
        ('mini_usb_26x26', CAMERA_LOCATION * box((26.0, 26.0, 14.0), (0, 0, 7.0))),
        ('oled_242', OLED_LOCATION * board_model('oled_242')),
    ):
        assert (hf & obj).volume <= 1e-4, label

    for dy, dz in ((0, 8), (0, -8), (8, 0), (-8, 0)):
        hits = hf.find_intersection_points(Axis(Vector(38.0, -9.0 + dy, 59.5 + dz), Vector(1, 0, 0)))
        xs = sorted(pt.X for pt, _ in hits if pt.X > 45.0)
        assert len(xs) >= 2 and xs[-1] - xs[0] >= 1.8, (dy, dz, xs)

    cover_check = [
        p for p in build_assembly('rpi5', wiring=False)
        if p.name in {'trunk_back_cover_rpi5', 'main_compute', 'active_cooler'}
    ]
    assert interference(cover_check)['passed']


@pytest.mark.slow
def test_head_pitch_multi_pose_sweep_with_wiring():
    for pose in ({'head_pitch': 0.785}, {'head_pitch': -0.523}):
        report = interference(build_assembly('rpi5', pose=pose, wiring=True))
        assert report['passed'], (pose, report['interferences'])


def test_wave15_dfam_wall_fixes_hold_without_thin_wafers():
    parts = printable_parts()
    for name in ('thigh_cover_L', 'thigh_cover_R', 'sole_tread_L', 'sole_tread_R'):
        shape, mat, _, _ = parts[name]
        rep = sample_wall_thickness(shape, required_wall(mat), max_samples=512)
        assert rep['observed_below_requirement'] is False, (name, rep)

    back_rep = sample_wall_thickness(parts['trunk_back_cover_rpi5'][0], 1.8, max_samples=512)
    assert back_rep['sampled_min_mm'] >= 1.2, back_rep

    neck_shape = parts['neck_link'][0]
    assert not neck_shape.is_inside(Vector(0.0, 0.0, 15.1))
    neck_rep = sample_wall_thickness(neck_shape, 2.5, max_samples=512)
    assert neck_rep['sampled_min_mm'] >= 0.65, neck_rep

    hc = parts['hip_clevis_L'][0]
    for x_p in (-10.0, 0.0, 10.0):
        hits = hc.find_intersection_points(Axis(Vector(x_p, 48.0, -35.0), Vector(0, 0, 1)))
        zs = sorted(pt.Z for pt, _ in hits if pt.Z > -32.0)
        assert len(zs) >= 2 and zs[-1] - zs[0] >= 2.3, (x_p, zs)

    foot = parts['foot_L'][0]
    assert foot.is_inside(Vector(-18.0, 15.0, 2.4))
    foot_rep = sample_wall_thickness(foot, 1.8, max_samples=512)
    assert foot_rep['sampled_min_mm'] >= 0.75, foot_rep

