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
    tray = box((98, width, 2.6), (0, 0, 1.3))
    tray = fillet(tray.edges().filter_by(Axis.Z), 9.0)
    # Clean central weight-saving windows keeping >= 6 mm structural perimeter & pad bridges.
    if variant == 'dual_esp32s3':
        for sy in (-1, 1):
            tray -= box((74, 29, 6), (0, sy * 32, 1.3))
        tray -= box((48, 16, 6), (0, 0, 1.3))
    elif variant == 'rpi5':
        tray -= box((74, 42, 6), (0, -16, 1.3))
        tray -= box((70, 22, 6), (0, 36.7, 1.3))
        tray -= box((78, 11.5, 6), (0, 13.5, 1.3))
    else:
        tray -= box((46, 16, 6), (0, -10, 1.3))
        tray -= box((44, 12, 6), (0, 28, 1.3))
        tray -= box((22, 6, 6), (0, 3.5, 1.3))
    for x, y in TRAY_MOUNTS:
        tray -= cylinder(M3_CLEAR, 6, (x, y, -1))
    for key, (cx, cy, z) in TRAY_LAYOUTS[variant]:
        board = BOARDS[key]
        length, bwidth, _ = board.size
        for hx, hy in board.holes:
            x, y = cx + hx, cy + hy
            tray += cylinder(5.4, z - 2.6, (x, y, 2.6))
        if not board.holes:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    tray += cylinder(5.6, z - 2.6,
                                     (cx + sx * (length / 2 - 7), cy + sy * (bwidth / 2 - 3), 2.6))
            for sy in (-1, 1):
                tray += box((length - 4, 2.65, 3.6), (cx, cy + sy * (bwidth / 2 + 1.675), 4.0))
        for x, y, top in clamp_positions(key, (cx, cy, z)):
            tray += insert_boss(x, y, 2.6, top - 2.6)
    # Through-bored M3 insert sleeves prevent thin blind floor membranes beneath bosses.
    for key, position in TRAY_LAYOUTS[variant]:
        for x, y, top in clamp_positions(key, position):
            tray -= cylinder(M3_INSERT, top + 2.0, (x, y, -1.0))
    return tray.clean()


def head_mcu_mount():
    tray = box((76, 35, 2.8), (0, 0, 1.4))
    for sy in (-1, 1):
        tray += box((65, 2.8, 4), (0, sy * 16, 4))
    for x in (-29, 29):
        tray -= cylinder(M3_CLEAR, 6, (x, 0, -1))
    return tray.clean()
