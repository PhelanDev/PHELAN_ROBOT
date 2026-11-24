"""Nominal mechanical references. Module vendors and connector styles vary."""

from dataclasses import dataclass
from functools import cache

from ..geometry import box, cylinder


@dataclass(frozen=True)
class Board:
    size: tuple
    mass: float
    pitch: tuple | None = None
    hole: float = 0.0
    features: tuple = ()
    rear_features: tuple = ()

    @property
    def holes(self):
        return () if self.pitch is None else tuple(
            (self.pitch[0] * sx / 2, self.pitch[1] * sy / 2)
            for sx in (-1, 1) for sy in (-1, 1)
        )


BOARDS = {
    'raspberry_pi_4b': Board((85, 56, 1.6), 0.046, (58, 49), 2.75, (
        ((18, 17, 13.5), (35.5, 17.5)),   # RJ45 Ethernet (+Y GPIO side)
        ((18, 14, 15.5), (35.5, 1.0)),    # USB 3.0 stack
        ((18, 14, 15.5), (35.5, -17.0)),  # USB 2.0 stack
        ((9, 7.5, 3.2), (-31.3, -24.25)), # USB-C 5V/3A power (flush edge)
        ((7, 7.5, 3.2), (-16.5, -24.25)), # micro-HDMI0 (flush edge)
        ((7, 7.5, 3.2), (-3.0, -24.25)),  # micro-HDMI1 (flush edge)
        ((7, 14.5, 6.0), (11.0, -20.75)), # 3.5 mm audio jack
        ((51, 5, 8.5), (-4, 23)),         # 40-pin GPIO (official 8.5 mm pin height)
    )),
    'raspberry_pi_5': Board((85, 56, 1.6), 0.046, (58, 49), 2.75, (
        ((18, 17, 15.5), (37, -17)), ((18, 14, 15.5), (37, 1)),
        ((18, 14, 15.5), (37, 17)), ((9, 7.5, 3.2), (-28, -26)),
        ((7, 7.5, 3.2), (-11, -26)), ((7, 7.5, 3.2), (0, -26)),
        ((51, 5, 8), (-4, 23)),
    )),
    'oled_242': Board((43.5, 72.0, 1.6), 0.012, (39.0, 68.0), 2.8, (
        ((37.0, 60.5, 3.4), (0, 0)),      # 2.42" SSD1309 glass panel facing +Z_local
    ), rear_features=(
        ((4.0, 20.0, 10.0), (-19.0, 0)),  # 7-pin 2.54 mm SPI/I2C rear header (top edge, clears head_pitch)
    )),
    'raspberry_pi_zero_2w': Board((65, 30, 1.6), 0.012, (58, 23), 2.75, (
        ((11, 8, 3.5), (-18, -12)), ((8, 6, 3), (3, -13)),
        ((8, 6, 3), (17, -13)), ((8, 14, 2.5), (-31, 0)),
    )),
    'esp32_s3_devkit': Board((70, 29, 1.6), 0.012, features=(
        ((60, 2.5, 9), (-2, -11)), ((60, 2.5, 9), (-2, 11)),
        ((8, 7, 3.2), (32, -5)), ((8, 7, 3.2), (32, 5)),
    )),
    'pca9685': Board((62.5, 25.4, 1.6), 0.010, (56, 19), 2.5, (
        ((49, 7.6, 10), (2, 7)), ((9, 10, 11), (-23, -3)),
    )),
    'mpu6050': Board((20, 16, 1.6), 0.002, (15, 11), 2.0),
    'pi_camera_3': Board((25, 24, 1.6), 0.004, (21, 12.5), 2.2, (
        ((10.8, 10.8, 10.8), (0, 1)),     # Wide 12.4 mm / Std 11.5 mm autofocus block
    )),
    'ov2640': Board((24, 24, 1.6), 0.003, features=(((8, 8, 7), (0, 0)),)),
    'vl53l0x': Board((13.5, 10.5, 1.6), 0.0015, features=(((5, 3, 1.5), (0, 0)),)),
    'inmp441': Board((14, 12, 1.6), 0.0015),
    'tca9548a': Board((30.6, 17.6, 1.6), 0.003),
    'buck_5v_5a': Board((40, 22, 1.6), 0.018, features=(((35, 18, 8), (0, 0)),)),
    'buck_5v_3a': Board((35, 17, 1.6), 0.010, features=(((31, 14, 6), (0, 0)),)),
    'mini360': Board((17, 11, 1.6), 0.003, features=(((12, 8, 3), (0, 0)),)),
    'ams1117': Board((13, 10, 1.6), 0.002),
}


@cache
def board_model(key):
    board = BOARDS[key]
    length, width, thickness = board.size
    shape = box(board.size, (0, 0, thickness / 2))
    for x, y in board.holes:
        shape -= cylinder(board.hole, thickness + 2, (x, y, -1))
    for (length, width, height), (x, y) in board.features:
        shape += box((length, width, height), (x, y, thickness + height / 2))
    for (length, width, height), (x, y) in board.rear_features:
        shape += box((length, width, height), (x, y, -height / 2))
    return shape.clean()


def active_cooler():
    return box((36.0, 36.0, 12.0), (0, 0, 6.0))
