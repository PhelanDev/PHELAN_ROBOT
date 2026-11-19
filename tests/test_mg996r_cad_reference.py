"""Geometry probes for the purchased servo datum and horn interface."""

import pytest

pytest.importorskip('build123d')

from build123d import Vector

from duck_v2.cad_src.geometry import cylinder
from duck_v2.cad_src.purchased.servo import horn_25t, mg996r, wire_clearance


def test_mg996r_casing_ears_and_shaft_datum():
    servo = mg996r()
    bounds = servo.bounding_box()
    assert tuple(bounds.size) == pytest.approx((54.0, 19.7, 42.9))
    assert bounds.min.Z == pytest.approx(0.0)
    assert len(servo.solids()) == 1 and servo.is_valid
    bottom_section = servo.intersect(cylinder(200, 1, (0, 0, 10)))
    assert tuple(bottom_section[0].bounding_box().size) == pytest.approx((40.7, 19.7, 1))
    for x in (-14.25, 34.35):
        for y in (-5, 5):
            assert not servo.is_inside(Vector(x, y, 28))
            assert servo.is_inside(Vector(x, y + 2.3, 28))
    assert servo.is_inside(Vector(2, 0, 42.8))
    assert servo.is_inside(Vector(-2, 0, 42.8))
    assert not servo.is_inside(Vector(3, 0, 42.8))


def test_wire_exit_and_horn_are_cut_from_the_solids():
    cut = wire_clearance().bounding_box()
    assert (cut.size.Y, cut.size.Z) == pytest.approx((8, 6))
    assert not mg996r().is_inside(Vector(30, 0, 3))
    assert mg996r().is_inside(Vector(30, 4.1, 3))
    horn = horn_25t()
    assert tuple(horn.bounding_box().size) == pytest.approx((20, 20, 4.5))
    for x, y in ((7, 0), (-7, 0), (0, 7), (0, -7), (0, 0)):
        assert not horn.is_inside(Vector(x, y, 3.5))
