"""Removable PCB trays with original-hole pegs and external M3 clamps."""

from build123d import Axis, fillet

from ..geometry import box, cylinder
from ..params import M3_CLEAR, M3_INSERT, M3_INSERT_DEPTH
from ..purchased.electronics import BOARDS
from .interfaces import insert_boss


TRAY_LAYOUTS = {
    'rpi5': [('raspberry_pi_4b', (0, -16, 6)), ('pca9685', (0, 36.7, 6))],
    'pizero2w': [('raspberry_pi_zero_2w', (0, -10, 6)), ('pca9685', (0, 28, 6))],
    'dual_esp32s3': [('esp32_s3_devkit', (0, 0, 6))],
}
TRAY_MOUNTS = tuple((x, y) for x in (-44, 44) for y in (-46, 46))
COOLER_POSITION = (-12.0, -16.0, 8.1)


def clamp_positions(key, position):
    cx, cy, z = position
    board = BOARDS[key]
    offsets = {'raspberry_pi_4b': (-19, 19), 'raspberry_pi_5': (-19, 19),
               'raspberry_pi_zero_2w': (-28.5, 28.5)}
    return [(cx + dx, cy - board.size[1] / 2 - 4.8, z + board.size[2])
            for dx in offsets.get(key, (-16, 16))]


def electronics_sled(variant):
    width = 112 if variant == 'rpi5' else 104
    tray = box((98, width, 2.2), (0, 0, 1.1))
    tray = fillet(tray.edges().filter_by(Axis.Z), 9.0)
    # Clean central weight-saving windows keeping >= 6 mm structural perimeter & pad bridges.
    if variant == 'dual_esp32s3':
        for sy in (-1, 1):
            tray -= box((74, 28, 6), (0, sy * 32, 1.1))
        tray -= box((44, 14, 6), (0, 0, 1.1))
    elif variant == 'rpi5':
        tray -= box((72, 41, 6), (0, -16, 1.1))
        tray -= box((68, 21, 6), (0, 36.7, 1.1))
        tray -= box((78, 11, 6), (0, 13.5, 1.1))
    else:
        tray -= box((42, 14, 6), (0, -10, 1.1))
        tray -= box((42, 10, 6), (0, 28, 1.1))
        tray -= box((50, 7, 6), (0, 3.5, 1.1))
    for x, y in TRAY_MOUNTS:
        tray -= cylinder(M3_CLEAR, 5, (x, y, -1))
    for key, (cx, cy, z) in TRAY_LAYOUTS[variant]:
        board = BOARDS[key]
        length, bwidth, _ = board.size
        for hx, hy in board.holes:
            x, y = cx + hx, cy + hy
            tray += cylinder(5.4, z - 2.2, (x, y, 2.2))
            tray += cylinder(min(board.hole - 0.3, 2.4), 1.4, (x, y, z))
        if not board.holes:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    tray += cylinder(5.6, z - 2.2,
                                     (cx + sx * (length / 2 - 7), cy + sy * (bwidth / 2 - 3), 2.2))
            for sy in (-1, 1):
                tray += box((length - 4, 2.6, 3.6), (cx, cy + sy * (bwidth / 2 + 1.65), 4.0))
        for x, y, top in clamp_positions(key, (cx, cy, z)):
            tray += insert_boss(x, y, 2.2, top - 2.2)
    # Blind insert bores cleared cleanly after boss union.
    for key, position in TRAY_LAYOUTS[variant]:
        for x, y, top in clamp_positions(key, position):
            tray -= cylinder(M3_INSERT, M3_INSERT_DEPTH, (x, y, top - M3_INSERT_DEPTH))
    return tray.clean()


def head_mcu_mount():
    tray = box((76, 35, 2.8), (0, 0, 1.4))
    for sy in (-1, 1):
        tray += box((65, 2.8, 4), (0, sy * 16, 4))
    for x in (-29, 29):
        tray -= cylinder(M3_CLEAR, 6, (x, 0, -1))
    return tray.clean()
