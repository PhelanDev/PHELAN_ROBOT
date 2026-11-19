"""MG996R with spline-centred XY datum and a bottom face at Z=0."""

from functools import cache

from build123d import Axis

from ..geometry import box, cylinder, ring, slot
from ..params import HORN_DISK, HORN_HUB, HORN_HUB_HEIGHT, HORN_OD, HORN_PCD, MG996R


def wire_clearance():
    width, height = MG996R.wire_exit
    end = MG996R.body_x_min + MG996R.body[0]
    return slot(height, width - height, 4, (0, height / 2, 0)).rotate(
        Axis.X, 90
    ).rotate(Axis.Z, 90).translate((end - 1, 0, 0))


@cache
def mg996r():
    s = MG996R
    body = box(s.body, (s.body_center_x, 0, s.body[2] / 2))
    ears = box((s.ear_length, s.body[1], s.ear_thickness),
               (s.body_center_x, 0, s.ear_z + s.ear_thickness / 2))
    for x, y in s.ear_holes:
        ears -= cylinder(s.ear_hole_diameter, s.ear_thickness + 2, (x, y, s.ear_z - 1))
    collar = cylinder(s.collar_diameter, 2.2, (0, 0, s.body[2]))
    spline = ring(s.shaft_diameter, 3.0, s.shaft_top - s.body[2] - 2.2,
                  (0, 0, s.body[2] + 2.2))
    return (body + ears + collar + spline - wire_clearance()).clean()


@cache
def horn_25t():
    """Smooth spline interface envelope, Ø20 disc and four M3 tap holes."""
    hub = ring(HORN_HUB, MG996R.shaft_diameter, HORN_HUB_HEIGHT)
    disc = ring(HORN_OD, 3.2, HORN_DISK, (0, 0, HORN_HUB_HEIGHT))
    radius = HORN_PCD / 2
    for x, y in ((radius, 0), (-radius, 0), (0, radius), (0, -radius)):
        disc -= cylinder(3.0, HORN_DISK + 2, (x, y, HORN_HUB_HEIGHT - 1))
    return (hub + disc).clean()
