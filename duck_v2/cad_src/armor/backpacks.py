"""Variant housings with integrated twin turbine nozzles, vents and I/O windows."""

from build123d import Pos, Rot

from ..geometry import box, cylinder, ring
from ..params import ARMOR_WALL, M3_CLEAR, M3_INSERT, M3_INSERT_DEPTH
from ..skeleton.electronics import TRAY_MOUNTS
from .panels import cover, nozzle, shell


DEPTH = {'rpi5': 38, 'pizero2w': 29, 'dual_esp32s3': 29}
WIDTH = {'rpi5': 120, 'pizero2w': 110, 'dual_esp32s3': 110}
POST_HEIGHT = 6.0
LID_GAP = 0.30


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
    housing = shell((104, WIDTH[variant], depth), wall=1.9, opening='bottom')
    floor = -depth / 2 + ARMOR_WALL
    for x, y in TRAY_MOUNTS:
        housing += cylinder(9.0, POST_HEIGHT, (x, y, floor))
        housing += box((10, 9, POST_HEIGHT),
                       (48 if x > 0 else -48, y, floor + POST_HEIGHT / 2))
    # Twin turbine exhaust rings molded into the rear face (compact 8 mm length).
    for y in (-28, 28):
        housing += ring(20.0, 16.2, 8, (10, y, depth / 2 - 0.5))
    # Sleek horizontal rear louvre slats + side/top/bottom cooling grilles keeping continuous rim.
    slot_h = 22 if depth > 35 else 8
    slot_z = 2 if depth > 35 else 2
    for x in (-34, -17, 0, 17, 34):
        housing -= box((14, WIDTH[variant] + 4, slot_h), (x, 0, slot_z))
    for y in (-32, 0, 32):
        housing -= box((110, 26, slot_h), (0, y, slot_z))
    for x_slit in (-34, -22, -10):
        housing -= box((10, 84, 10), (x_slit, 0, depth / 2))
    housing -= box((26, 32, 10), (14, 0, depth / 2))
    housing -= box((22, 84, 10), (35, 0, depth / 2))
    if variant == 'rpi5':
        housing -= box((32, 36, 21), (45, 9, 3))
        housing -= box((26, 10, 8), (-15, -52, -3))
    elif variant == 'pizero2w':
        housing -= box((43, 12, 8), (0, -51, 0))
        housing -= box((12, 22, 6), (-49, 0, 0))
    else:
        housing -= box((12, 28, 10), (49, 0, 0))
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
