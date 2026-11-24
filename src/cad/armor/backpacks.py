"""Variant housings with integrated twin turbine nozzles, vents and I/O windows."""

from build123d import Box, Plane, Pos, Rot
import numpy as np

from ..geometry import box, cylinder, ring
from ..params import ARMOR_WALL, M3_CLEAR, M3_INSERT, M3_INSERT_DEPTH
from ..skeleton.electronics import TRAY_MOUNTS
from .panels import cover, shell


DEPTH = {'rpi5': 38, 'pizero2w': 29, 'dual_esp32s3': 29}
WIDTH = {'rpi5': 120, 'pizero2w': 110, 'dual_esp32s3': 110}
POST_HEIGHT = 6.0
LID_GAP = 0.30


def _chamfer_cutter(normal, origin):
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return Plane(origin=tuple(origin), z_dir=tuple(n)) * Pos(0, 0, 120) * Box(260, 260, 240)


def lid_mounts(variant):
    return tuple((x, y) for x in (-47.5, 47.5)
                 for y in (-WIDTH[variant] / 2 + 7, WIDTH[variant] / 2 - 7))


def module_frames(variant):
    housing = Pos(-60 - DEPTH[variant] / 2, 0, -12) * Rot(0, -90, 0)
    floor = -DEPTH[variant] / 2 + ARMOR_WALL
    sled = housing * Pos(0, 0, floor + POST_HEIGHT)
    hatch = housing * Pos(0, 0, DEPTH[variant] / 2 + LID_GAP + 1.1)
    return housing, sled, hatch


def backpack(variant):
    depth = DEPTH[variant]
    width = WIDTH[variant]
    wall = 1.82
    collar = 7.8  # Extends forward to X_trunk=-52.2 mm, mating torso cleanly with 0.2 mm clearance.
    # Sealed octagonal dorsal pack with closed front bulkhead wall (zero exposed PCB on overhanging wings).
    outer = Box(104, width, depth)
    inner = Box(104 - 2 * wall, width - 2 * wall, depth - 2 * wall)
    cham = 9.5 if depth > 35 else 6.0
    bevels = (
        ((0.707, 0.0, 0.707), (52.0 - cham / 2, 0.0, depth / 2 - cham / 2)),
        ((-0.707, 0.0, 0.707), (-52.0 + cham / 2, 0.0, depth / 2 - cham / 2)),
        ((0.0, 0.707, 0.707), (0.0, width / 2 - (cham - 2.0) / 2, depth / 2 - (cham - 2.0) / 2)),
        ((0.0, -0.707, 0.707), (0.0, -width / 2 + (cham - 2.0) / 2, depth / 2 - (cham - 2.0) / 2)),
    )
    for n, orig in bevels:
        outer -= _chamfer_cutter(n, orig)
        ni = np.asarray(n) / np.linalg.norm(n)
        inner -= _chamfer_cutter(n, np.asarray(orig) - wall * ni)
    # Corner octagons keep sharp sleek dorsal profile while clearing internal sled corners cleanly.
    for sx in (-1, 1):
        for sy in (-1, 1):
            c_orig = (sx * (52.0 - 2.6), sy * (width / 2 - 2.6), 0.0)
            cn = (sx * 0.707, sy * 0.707, 0.0)
            outer -= _chamfer_cutter(cn, c_orig)
            inner -= _chamfer_cutter(cn, np.asarray(c_orig) - wall * np.asarray(cn))
    housing = (outer - inner).clean()

    # Contoured dual-tier flush torso-mating collar: cavity stops cleanly at z_loc=-depth/2, preserving solid front bulkhead.
    col_cz = -depth / 2 - (collar - 1.0) / 2
    col_len = collar + 1.0
    col_outer = Pos(17.75, 0, col_cz) * Box(56.5, 97.0, col_len)
    # Match trunk_front rear shoulder bevels (0.907*(Z-4) + 0.42*|Y| <= 38.8) so collar sits flush with zero protruding ears.
    for sy in (-1, 1):
        col_outer -= _chamfer_cutter((0.907, sy * 0.42, 0.0), (36.3, sy * 48.5, col_cz))
    col_inner = Pos(17.75, 0, -depth / 2 - collar / 2 - 1.0) * Box(56.5 - 2 * wall, 97.0 - 2 * wall, collar + 2.0)
    for sy in (-1, 1):
        ni = np.array([0.907, sy * 0.42, 0.0])
        ni = ni / np.linalg.norm(ni)
        orig_in = np.array([36.3, sy * 48.5, col_cz]) - wall * ni
        col_inner -= _chamfer_cutter(ni, orig_in)

    # Lower spine collar wrapping torso_lower rear (Z=-48..-22.5 => x_loc=-36..-10.5, Y=±27).
    low_outer = Pos(-23.25, 0, col_cz) * Box(27.5, 54.0, col_len)
    low_inner = Pos(-23.25, 0, -depth / 2 - collar / 2 - 1.0) * Box(27.5 - 2 * wall, 54.0 - 2 * wall, collar + 2.0)
    spine_bridge = Pos(-10.5, 0, -depth / 2 - collar / 2 - 1.0) * Box(12.0, 54.0 - 2 * wall, collar + 2.0)
    housing = (housing + col_outer + low_outer - col_inner - low_inner - spine_bridge).clean()

    floor = -depth / 2 + ARMOR_WALL
    for x, y in TRAY_MOUNTS:
        housing += cylinder(8.4, POST_HEIGHT + (ARMOR_WALL - wall), (x, y, -depth / 2 + wall))
        housing += box((8, 4.0, POST_HEIGHT),
                       (48 if x > 0 else -48, y, floor + POST_HEIGHT / 2))
    # Focused rear heatsink cooling louvres without exposing side/top/bottom PCBs.
    for x_slit in (-31, -24, -17, -10, -3, 4, 11, 18):
        housing -= box((4.2, 50.0, 10.0), (x_slit, -16 if variant == 'rpi5' else 0, depth / 2))
    # Minimal wire portals: narrow center-spine head_harness channel behind neck + flush collar-only leg exits backed by solid wall.
    housing -= box((28.0, 13.0, collar + 5.8), (39.5, 0, -depth / 2 - collar / 2 + 1.9))
    for sy in (-1, 1):
        housing -= box((16.0, 12.0, collar + 0.4), (-3.5, sy * 48.0, -depth / 2 - (collar + 0.4) / 2))
    # Compact, low-profile Pi 4B prioritized I/O bays (also fits Pi 5) plus pizero2w & esp32s3 slots.
    if variant == 'rpi5':
        housing -= box((10.0, 44.0, 13.0), (50.0, -16.0, 5.0))
        housing -= box((44.0, 10.0, 7.0), (-10.0, -58.0, -1.0))
    elif variant == 'pizero2w':
        housing -= box((43, 12, 7.0), (0, -51, 0))
        housing -= box((12, 22, 6.0), (-49, 0, 0))
    else:
        housing -= box((12, 28, 8.0), (49, 0, 0))
    for x, y in TRAY_MOUNTS:
        housing -= cylinder(M3_INSERT, M3_INSERT_DEPTH,
                            (x, y, floor + POST_HEIGHT - M3_INSERT_DEPTH))
    return housing.clean()


def backpack_hatch(variant):
    part = cover((101.5, WIDTH[variant] - 0.5, 2.2), holes=False)
    for x, y in lid_mounts(variant):
        part -= cylinder(M3_CLEAR, 5, (x, y, -2.5))
    for x in (-32, -16, 0, 16, 32):
        part -= box((9, 40, 5), (x, 0, 0))
    for y in (-28, 28):
        part -= cylinder(16.6, 5, (10, y, -2.5))
    return part
