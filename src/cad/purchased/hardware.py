"""B-rep envelopes for M3 fasteners and the two bearing families."""

from functools import cache

from build123d import RegularPolygon, extrude

from ..geometry import cylinder, ring
from ..params import F623ZZ, NECK_BEARING


@cache
def flanged_bearing():
    bore, od, depth, flange, lip = F623ZZ
    return ring(od, bore, depth) + ring(flange, bore, lip, (0, 0, depth - lip))


@cache
def neck_bearing():
    bore, od, depth = NECK_BEARING
    return ring(od, bore, depth)


@cache
def screw(length):
    head = cylinder(5.5, 3.0) - extrude(RegularPolygon(2.5 / 3**0.5, 6), amount=1.8)
    return head + cylinder(3.0, length, (0, 0, 3.0))


@cache
def insert():
    return ring(4.05, 3, 5)


@cache
def washer():
    return ring(7.0, 3.2, 0.5)


@cache
def nyloc():
    return extrude(RegularPolygon(5.5 / 3**0.5, 6), amount=4.0) - cylinder(3, 6, (0, 0, -1))
