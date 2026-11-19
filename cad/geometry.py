"""Common B-rep operations and assembly part metadata."""

from dataclasses import dataclass, replace

from build123d import Align, Box, Cylinder, Part, Pos


def box(size, center=(0, 0, 0)):
    return Pos(*center) * Box(*size)


def cylinder(diameter, height, base=(0, 0, 0)):
    return Pos(*base) * Cylinder(diameter / 2, height, align=(Align.CENTER, Align.CENTER, Align.MIN))


def ring(outer, inner, height, base=(0, 0, 0)):
    x, y, z = base
    return cylinder(outer, height, base) - cylinder(inner, height + 2, (x, y, z - 1))


def slot(diameter, traverse, height, base):
    """Capsule cut along local X, where traverse is end-centre spacing."""
    x, y, z = base
    dx = traverse / 2
    return (
        box((traverse, diameter, height), (x, y, z + height / 2))
        + cylinder(diameter, height, (x - dx, y, z))
        + cylinder(diameter, height, (x + dx, y, z))
    )


@dataclass(frozen=True)
class Component:
    name: str
    link: str
    shape: Part
    material: str
    category: str = 'reference'
    print_key: str | None = None
    mass_kg: float | None = None
    orientation: tuple = (0, 0, 0)
    sku: str | None = None

    def placed(self, location, *, link=None, name=None):
        return replace(self, shape=location * self.shape, link=link or self.link,
                       name=name or self.name)
