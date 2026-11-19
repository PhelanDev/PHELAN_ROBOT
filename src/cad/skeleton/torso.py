"""Shared belly battery drawer, rigid IMU deck and power bracket."""

from build123d import Pos

from ..geometry import box, cylinder, ring
from ..params import M3_CLEAR, M3_INSERT
from .interfaces import insert_boss


def torso_lower():
    """Lightweight 2.6 mm wall belly drawer around the locked 82 x 48 x 18 mm cavity."""
    # Cavity: X in [-41, 41], Y in [-24, 24], Z in [-45.5, -27.5] (18 mm tall).
    # Outer:  X in [-43.6, 43.6], Y in [-26.6, 26.6], Z in [-48.5, -24.9] (2.6 mm walls).
    outer = box((87.2, 53.2, 23.6), (0, 0, -36.7))
    outer += box((16, 18.0, 21.0), (-50.0, 0, -36.5))
    cavity = box((82, 48, 22), (0, 0, -38.5))
    part = outer - cavity
    # Roof & side weight-saving windows keeping 2.6 mm PETG structural frame.
    part -= box((68, 34, 8), (0, 0, -25.5))
    part -= box((64, 62, 12), (0, 0, -36.5))
    # Bottom recess for sliding hatch (hatch Z in [-48.4, -45.8], gap 0.2 mm).
    part -= box((84.6, 50.6, 3.4), (0, 0, -47.3))
    # Front latch tongue recess (tongue Z in [-46.6, -43.0]).
    part -= box((4.4, 13.0, 4.2), (42.8, 0, -44.7))
    # Rear XT30 connector bay and top wire exit (main_power passes up at X=-53, Y=0).
    part -= box((18, 13.6, 16), (-50.5, 0, -36.5))
    part -= cylinder(8, 14, (-53, 0, -29.5))
    return part.clean()


def battery_hatch():
    """2.6 mm PETG sliding belly hatch with lightening window and 2.6 mm front latch."""
    plate = box((84, 50, 2.6), (0, 0, -47.1))
    plate += box((3.0, 12, 3.6), (42.6, 0, -44.8))
    plate -= box((56, 26, 6), (2, 0, -47.1))
    for y in (-19, 19):
        plate -= cylinder(M3_CLEAR, 8, (-38, y, -50))
    return plate.clean()


def torso_upper():
    part = box((100, 110, 2.8), (0, 0, -20.8))
    part -= box((64, 74, 8), (-3, 0, -20))
    for x in (-43, 43):
        for y in (-26, 26):
            part += insert_boss(x, y, -23, 40)
    part += box((25, 21, 2.8), (35, 0, -18))
    for y in (-5.5, 5.5):
        part += cylinder(1.8, 2, (35 - 7.5, y, -16.6))
    for y in (-42, 42):
        part += box((84, 2.8, 8), (0, y, -15.4))
    return part.clean()


def power_bracket():
    base = box((63, 26, 2.8), (0, 0, 1.4))
    base += ring(24.6, 19.0, 9, (19, 0, 2.8))
    for x in (-25, -5):
        base -= box((3.5, 20, 8), (x, 0, 2))
    return base.clean()


def wire_comb():
    comb = box((28, 8, 6), (0, 0, 3))
    for x in (-10, -5, 0, 5, 10):
        comb -= box((3, 10, 4), (x, 0, 5))
    return comb
