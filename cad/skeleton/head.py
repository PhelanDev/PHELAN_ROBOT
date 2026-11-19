"""Neck bearings, open harness channel and head sensor fixture."""

from math import cos, radians, sin

from build123d import Axis, Pos

from ..geometry import box, cylinder, ring
from ..params import M3_CLEAR, NECK_SEAT, NECK_SHOULDER, TOF_ANGLES
from .interfaces import servo_mount


def neck_yoke():
    bearing = ring(34, NECK_SHOULDER, 8)
    bearing -= cylinder(NECK_SEAT, 5, (0, 0, 3))
    bearing += box((4, 62, 4), (-13, 0, 10))
    for y in (-29, 29):
        bearing += box((24, 4, 25), (-3, y, 18.5))
    bearing -= box((12, 20, 50), (3, 0, 25))
    return bearing.clean()


def head_frame():
    frame = box((62, 92, 3), (-8, 0, -13))
    frame -= box((40, 52, 10), (-5, 0, -13))
    frame += box((3, 92, 31), (-37.5, 0, 1))
    for sy in (-1, 1):
        frame += box((62, 3, 24), (-8, sy * 44.5, 0))
    return frame.clean()


def tof_arc_bracket():
    arc = ring(48, 42, 14).rotate(Axis.X, 90)
    arc = arc & box((16, 60, 35), (22, -4, 0))
    # Individual sensor landing pads rotate with each optical axis.
    for angle in TOF_ANGLES:
        pad = box((3, 15, 12), (24, 0, 0)).rotate(Axis.Z, angle)
        arc += pad
    return arc.clean()
