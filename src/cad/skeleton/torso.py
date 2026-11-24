"""Shared belly battery drawer, rigid IMU deck and power bracket."""

from build123d import Pos

from ..geometry import box, cylinder, ring
from ..params import M3_CLEAR, M3_INSERT
from .interfaces import insert_boss


def torso_lower():
    """2.6+ mm PETG universal belly battery bay (82 x 48.9 x 22 mm) with open XT30 access & M3 bosses."""
    # Cavity: X in [-41.0, 41.0] (82.0 mm), Y in [-24.45, 24.45] (48.9 mm), Z in [-49.8, -27.5] (22.3 mm).
    # Outer:  X in [-43.6, 44.4] (88.0 mm), Y in [-27.0, 27.0] (54.0 mm), Z in [-52.8, -24.9] (27.9 mm).
    outer = box((88.0, 54.0, 27.9), (0.4, 0, -38.85))
    # Rear XT30 housing pod kept above backpack lower collar floor (Z >= -47.0 vs collar Z=-47.18..-49.0).
    outer += box((17.0, 19.2, 21.0), (-50.5, 0, -36.5))
    # Twin solid PETG rear M3 heat-set insert bosses at X=-47.5, Y=±18.5 (Z in [-49.8, -41.0]).
    for sign in (-1, 1):
        outer += box((9.4, 10.0, 8.8), (-47.5, sign * 18.5, -45.4))
    # Open bottom battery cavity (82.0 x 48.9 x 22.0 mm, open through Z=-52.8).
    cavity = box((82.0, 48.9, 27.0), (0, 0, -41.0))
    part = outer - cavity
    # Roof weight-saving window keeping solid perimeter walls (>= 2.6 mm PETG) hiding battery.
    part -= box((76.6, 43.4, 8.0), (0, 0, -25.5))
    # Side cooling slots preserving >=5 mm top/bottom rails and vertical ribs.
    for cx_win, lx_win in ((-24.0, 18.0), (0.0, 24.0), (24.0, 18.0)):
        part -= box((lx_win, 60.0, 12.5), (cx_win, 0, -38.5))
    # Flush full-width bottom cutout up to X=41.0 while keeping solid 2.7+ mm front latch wall & rear bosses.
    part -= box((86.0, 60.0, 4.0), (-2.0, 0, -51.8))
    # Front blind-recessed latch slot (X in [38.9, 41.7], Z in [-50.1, -46.5], leaving 2.7 mm front & 2.7 mm bottom rim).
    part -= box((2.8, 13.8, 3.6), (40.3, 0, -48.3))
    # Open bottom-accessible XT30 connector bay & direct wire pass-through into battery cavity (X=-59.5..-39.5).
    part -= box((20.0, 13.6, 26.0), (-49.5, 0, -41.5))
    # Top wire exit hole for main_power up behind logic_buck.
    part -= cylinder(6.8, 14.0, (-53.0, 0, -29.5))
    # Real Ø4.05 mm x 6.2 mm M3 heat-set insert pockets in rear bosses (X=-47.5, Y=±18.5, Z in [-50.0, -43.8]).
    for sign in (-1, 1):
        part -= cylinder(M3_INSERT, 6.2, (-47.5, sign * 18.5, -50.0))
    return part.clean()


def battery_hatch():
    """2.6 mm PETG bottom belly hatch with front tilt-in tongue and twin rear M3 screw tabs."""
    # Main bottom lid: X in [-42.0, 40.9], Y in [-26.8, 26.8], Z in [-52.6, -50.0].
    plate = box((82.9, 53.6, 2.6), (-0.55, 0, -51.3))
    # Front tilt-in latch tongue (2.6 mm thick >= 2.5 PETG, X in [38.9, 41.5], clears 77 mm battery holder).
    plate += box((2.6, 13.0, 2.8), (40.2, 0, -48.6))
    # Twin rear mounting ears aligning with M3 insert bosses at X=-47.5, Y=±18.5 (>=3.1 mm rear margin).
    for sign in (-1, 1):
        plate += box((10.8, 10.0, 2.6), (-46.9, sign * 18.5, -51.3))
    # Lightening window leaving generous perimeter rim and grip access.
    plate -= box((72.0, 41.0, 6.0), (-0.5, 0, -51.3))
    # Twin M3 clearance screw holes matching trunk_bottom rear insert bosses.
    for sign in (-1, 1):
        plate -= cylinder(M3_CLEAR, 8.0, (-47.5, sign * 18.5, -54.0))
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
