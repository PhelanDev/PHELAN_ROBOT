"""Create draft variant assemblies; geometric validation is a separate gate."""

from build123d import Pos, Rot

from .catalogue import printable_parts
from .armor.backpacks import module_frames
from .geometry import Component
from .kinematics import frame_transforms, location_from_matrix
from .layout import (STATIONS, UBEC_LOCATION, CAPACITOR_LOCATION, IMU_LOCATION,
                     LOGIC_BUCK_LOCATION, SWITCH_LOCATION, OLED_LOCATION,
                     CAMERA_LOCATION, TOF_LOCATION, MIC_LOCATION, HEAD_FORWARD,
                     THIGH_COVER_BOSSES)
from .params import MG996R, VARIANTS
from .placements import print_placements
from .purchased.catalogue import purchased_parts
from .purchased.electronics import BOARDS
from .skeleton.electronics import COOLER_POSITION, TRAY_LAYOUTS, TRAY_MOUNTS, clamp_positions
from .skeleton.interfaces import IDLER_BEARING_TOP, bearing_locations
from .wiring.routes import harnesses


def build_assembly(variant='rpi5', pose=None, *, armor=True, wiring=True):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown hardware variant: {variant}')
    parts, purchased, assembly = printable_parts(), purchased_parts(), []

    def reference(name, key, link, location, material='STEEL'):
        shape, mass = purchased[key]
        assembly.append(Component(name, link, location * shape, material, mass_kg=mass, sku=key))

    for name, key, link, location in print_placements(variant):
        shape, material, _, _ = parts[key]
        if armor or material in ('PETG', 'TPU', 'EVA'):
            assembly.append(Component(name, link, location * shape, material,
                                      'printed' if material != 'EVA' else 'reference', key))

    def station(spec):
        joint, fixed_link, moving_link = spec.joint, spec.fixed, spec.moving
        location, base = spec.location, spec.servo_base
        bearings, moving_location = spec.bearings, spec.moving_location
        reference(f'servo_{joint}', 'mg996r', fixed_link, base, 'SERVO')
        reference(f'horn_{joint}', 'horn_25t_20mm', moving_link,
                  moving_location * Pos(0, 0, -4.5), 'ALUMINUM')
        for x, y in MG996R.ear_holes:
            reference(f'ear_screw_{joint}_{x}_{y}', 'm3x12', fixed_link,
                      base * Pos(x, y, MG996R.ear_z + MG996R.ear_thickness + 3) * Rot(0, 180, 0))
        for i, (x, y) in enumerate(((7, 0), (-7, 0), (0, 7), (0, -7))):
            reference(f'horn_screw_{joint}_{i}', 'm3x8', moving_link,
                      moving_location * Pos(x, y, 6) * Rot(0, 180, 0))
        reference(f'shaft_screw_{joint}', 'm3x6', fixed_link,
                  location * Pos(0, 0, 3.0) * Rot(0, 180, 0))
        if bearings:
            for i, bearing_location in enumerate(bearing_locations(bearings == 2)):
                reference(f'f623_{joint}_{i}', 'f623zz', moving_link,
                          moving_location * bearing_location)
            reference(f'idler_screw_{joint}', 'm3x18' if bearings == 2 else 'm3x14',
                      fixed_link, location * Pos(0, 0, -47.95) * Rot(180, 0, 0))
            bearing_bottom = IDLER_BEARING_TOP - 4 * bearings
            reference(f'idler_washer_{joint}', 'm3_washer', moving_link,
                      moving_location * Pos(0, 0, bearing_bottom - 0.5))
            reference(f'idler_nyloc_{joint}', 'm3_nyloc', moving_link,
                      moving_location * Pos(0, 0, bearing_bottom - 4.5))

    for spec in STATIONS:
        station(spec)
    reference('neck_6803', '6803_2rs', 'trunk_base', Pos(6, 0, 34.5))
    reference('battery', 'battery_2s_2000mah', 'trunk_base', Pos(0, 0, -39.5), 'BATTERY')
    for idx, sign in enumerate((-1, 1)):
        reference(f'belly_insert_{idx}', 'm3_insert', 'trunk_base', Pos(-47.5, sign * 18.5, -49.8))
        reference(f'belly_screw_{idx}', 'm3x8', 'trunk_base', Pos(-47.5, sign * 18.5, -55.6))
    for side, sign, prefix in (('L', 1, 'left'), ('R', -1, 'right')):
        for idx, (bx, bz) in enumerate(THIGH_COVER_BOSSES):
            reference(f'thigh_cover_insert_{side}_{idx}', 'm3_insert', f'{prefix}_thigh',
                      Pos(bx, sign * 31.8, bz) * Rot(90 * sign, 0, 0))
            reference(f'thigh_cover_screw_{side}_{idx}', 'm3x8', f'{prefix}_thigh',
                      Pos(bx, sign * 38.6, bz) * Rot(90 * sign, 0, 0))
    reference('ubec', 'ubec_6v_15a', 'trunk_base', UBEC_LOCATION, 'ELECTRONICS')
    reference('capacitor', 'capacitor_4700uf', 'trunk_base', CAPACITOR_LOCATION, 'ELECTRONICS')
    reference('imu', 'mpu6050', 'trunk_base', IMU_LOCATION, 'PCB')
    housing_location, tray_location, _ = module_frames(variant)
    for index, (x, y) in enumerate(TRAY_MOUNTS):
        reference(f'sled_screw_{index}', 'm3x8', 'trunk_base',
                  tray_location * Pos(x, y, 5.8) * Rot(180, 0, 0))
        reference(f'sled_insert_{index}', 'm3_insert', 'trunk_base', tray_location * Pos(x, y, -5))
    for key, pos in TRAY_LAYOUTS[variant]:
        reference(key, key, 'trunk_base', tray_location * Pos(*pos), 'PCB')
        for index, clamp_position in enumerate(clamp_positions(key, pos)):
            clamp_location = tray_location * Pos(*clamp_position)
            reference(f'clamp_washer_{key}_{index}', 'm3_wide_washer', 'trunk_base',
                      clamp_location, 'STEEL')
            reference(f'clamp_screw_{key}_{index}', 'm3x6', 'trunk_base',
                      clamp_location * Pos(0, 0, 3.8) * Rot(0, 180, 0))
            reference(f'clamp_insert_{key}_{index}', 'm3_insert', 'trunk_base',
                      clamp_location * Pos(0, 0, -5))
    camera = 'ov2640' if variant == 'dual_esp32s3' else 'pi_camera_3'
    reference('oled_display', 'oled_242', 'head', OLED_LOCATION, 'PCB')
    reference('camera', camera, 'head', CAMERA_LOCATION, 'PCB')
    if variant == 'dual_esp32s3':
        reference('head_mcu1', 'esp32_s3_devkit', 'head', Pos(-22 + HEAD_FORWARD, 0, 34) * Rot(0, 0, 90), 'PCB')
    if variant == 'rpi5':
        reference('active_cooler', 'pi5_active_cooler', 'trunk_base',
                  tray_location * Pos(*COOLER_POSITION), 'ALUMINUM')
    power = {'rpi5': 'buck_5v_5a', 'pizero2w': 'mini360', 'dual_esp32s3': 'buck_5v_3a'}[variant]
    reference('logic_buck', power, 'trunk_base', LOGIC_BUCK_LOCATION, 'PCB')
    if variant == 'dual_esp32s3':
        reference('ldo_1', 'ams1117', 'head', Pos(12 + HEAD_FORWARD, 28, 28), 'PCB')
        reference('ldo_2', 'ams1117', 'trunk_base', Pos(32, 24, -10), 'PCB')
    reference('tof_sensor_0', 'vl53l0x', 'head', TOF_LOCATION, 'PCB')
    reference('microphone', 'inmp441', 'head', MIC_LOCATION, 'PCB')
    reference('main_switch', 'main_switch', 'trunk_base', SWITCH_LOCATION, 'ELECTRONICS')
    reference('battery_connector', 'xt30_connector', 'trunk_base', Pos(-53, 0, -34), 'ELECTRONICS')
    transforms = frame_transforms(pose)
    placed = [part.placed(location_from_matrix(transforms[part.link])) for part in assembly]
    if wiring:
        for route in harnesses(variant):
            wire_len_mm = route.installed_length_mm if route.installed_length_mm is not None else route.path(transforms).length
            placed.append(Component(route.name, 'harness', route.solid(transforms), 'WIRE',
                                    mass_kg=wire_len_mm * 0.00002))
    return placed
