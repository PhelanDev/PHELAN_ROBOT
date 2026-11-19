"""Faceted shells with genuine cavities and restrained edge chamfers."""

from math import radians, sin

from build123d import Axis, Pos, Sphere, fillet

from ..geometry import box, cylinder, ring
from ..params import ARMOR_WALL, M3_CLEAR, TOF_ANGLES


def shell(size, wall=ARMOR_WALL, opening='bottom'):
    """Open rounded shell with a matched inner corner radius."""
    x, y, z = size
    outer = box(size)
    corner = min(12.0, x / 4, y / 4)
    outer = fillet(outer.edges().filter_by(Axis.Z), corner)
    offset = -wall if opening == 'bottom' else wall
    inner = box((x - 2 * wall, y - 2 * wall, z), (0, 0, offset))
    inner_corner = corner - wall
    if inner_corner > 0.1:
        inner = fillet(inner.edges().filter_by(Axis.Z), inner_corner)
    return (outer - inner).clean()


def ellipsoid_shell(radii, wall=ARMOR_WALL):
    outer = Sphere(1).scale(radii)
    inner_radii = tuple(radius - wall for radius in radii)
    inner = Sphere(1).scale(inner_radii)
    return (outer - inner).clean()


def chest_shell():
    outer = box((72, 88, 30))
    outer = fillet(outer.edges(), 9.0)
    cavity = box((68, 84, 34), (0, 0, -4.5))
    cavity = fillet(cavity.edges(), 7.0)
    return (outer - cavity).clean()


def helmet_shell():
    outer = box((70, 82, 62))
    outer = fillet(outer.edges(), 11.0)
    cavity = box((74, 76, 56), (-3.0, 0, 0))
    cavity = fillet(cavity.edges(), 8.0)
    shell = outer - cavity
    eye_opening = box((18, 66, 22), (32, 0, 5))
    return (shell - eye_opening).clean()


def oval_armor(size, wall=ARMOR_WALL):
    return ellipsoid_shell(tuple(dimension / 2 for dimension in size), wall)


def cover(size, holes=True):
    panel = box(size)
    thickness_axis = min(range(3), key=lambda index: size[index])
    axes = (Axis.X, Axis.Y, Axis.Z)
    corner = min(size[index] for index in range(3) if index != thickness_axis) / 5
    panel = fillet(panel.edges().filter_by(axes[thickness_axis]), corner)
    if holes:
        x, y, z = size
        for hx in (-x / 2 + 5, x / 2 - 5):
            if y < z:
                cut = cylinder(M3_CLEAR, y + 2, (0, 0, -y / 2 - 1)).rotate(
                    Axis.X, 90).translate((hx, 0, 0))
            else:
                cut = cylinder(M3_CLEAR, z + 2, (hx, 0, -z / 2 - 1))
            panel -= cut
    return panel


def visor():
    panel = ellipsoid_shell((4.2, 46, 12.5), wall=2.2)
    panel -= cylinder(11, 14, (0, 0, -7)).rotate(Axis.Y, 90)
    for angle in TOF_ANGLES:
        y = 65 * sin(radians(angle))
        cut = cylinder(7, 14, (0, 0, -7)).rotate(Axis.Y, 90).translate((0, y, 0))
        panel -= cut
    return panel.clean()


def reactor(diffuser=False):
    return cylinder(16, 2) if diffuser else ring(23, 17, 3.2)


def nozzle():
    part = ring(21, 16, 25)
    part += box((16, 2.2, 4), (0, 0, 22))
    part += box((2.2, 16, 4), (0, 0, 22))
    return part.clean()
