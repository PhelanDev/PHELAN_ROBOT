"""Thigh cassette, passive shin beam, three-toe foot and non-slip sole."""

from build123d import Axis, Pos, Sphere, fillet

from ..geometry import box, cylinder, ring
from ..params import FOOT_SIZE, M3_CLEAR, THIGH_MM, SHIN_MM
from ..armor.panels import oval_armor
from .interfaces import clevis, servo_mount


def thigh():
    """68 mm hip-knee spacing; knee servo is carried by this member."""
    top = clevis(True).rotate(Axis.X, -90).translate((0, 24, 0))
    beam = box((3.2, 60, THIGH_MM), (-17, -5, -THIGH_MM / 2))
    beam += box((20, 3.2, THIGH_MM), (-17, -5, -THIGH_MM / 2))
    knee_mount = servo_mount().translate((0, 0, -47.6)).rotate(
        Axis.X, -90).translate((0, 24, -THIGH_MM))
    core = (top + beam + knee_mount).clean()
    shell = oval_armor((40, 36, THIGH_MM), wall=2.4).translate((-8, 0, -THIGH_MM / 2))
    return (core + shell).clean()


def shin():
    top = clevis(False).rotate(Axis.X, -90).translate((0, 24, 0))
    beam = box((3.2, 54, SHIN_MM), (-17, -3, -SHIN_MM / 2))
    beam += box((16, 3.2, SHIN_MM), (-17, -3, -SHIN_MM / 2))
    flange = box((24, 25, 5), (-11, 0, -SHIN_MM + 2.5))
    for x in (-18, -4):
        for y in (-8, 8):
            flange -= cylinder(M3_CLEAR, 9, (x, y, -SHIN_MM - 2))
    core = (top + beam + flange).clean()
    shell = oval_armor((32, 26, SHIN_MM), wall=2.4).translate((-12, 0, -SHIN_MM / 2))
    return (core + shell).clean()


def foot_frame():
    plate = box((74, 40, 5), (4, 0, -3.5))
    plate = fillet(plate.edges().filter_by(Axis.Z), 8.0)
    # A single rounded boot toe keeps the ground contact continuous and avoids
    # the three-lobed silhouette of the earlier duck-foot draft.
    plate += Sphere(1).scale((12, 18, 3.2)).translate((29, 0, -0.2))
    plate += Sphere(1).scale((8, 18, 4)).translate((-25, 0, 0))
    plate -= box((28, 20, 3), (8, 0, -5))
    for x in (-18, -4):
        for y in (-8, 8):
            plate -= cylinder(M3_CLEAR, 9, (x, y, -8))
    for x in (-25, 31):
        for y in (-15, 15):
            plate -= cylinder(M3_CLEAR, 9, (x, y, -8))
    return plate


def sole():
    sole = box((82, 46, 5), (4, 0, -8.25))
    # Tread grooves leave 2.2 mm of continuous material above each channel.
    for x in range(-31, 43, 8):
        sole -= box((2, 48, 2.8), (x, 0, -9.35))
    sole -= box((18, 16, 1), (6, 0, -6.8))
    for x in (-25, 31):
        for y in (-15, 15):
            sole -= cylinder(5.8, 1.6, (x, y, -9.5))
            sole -= cylinder(M3_CLEAR, 5, (x, y, -10.5))
    return sole


def cable_clip():
    return (ring(10, 6, 5).rotate(Axis.X, 90)
            + box((14, 3, 3), (0, -2.5, -4)))
