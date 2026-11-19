"""Reusable servo cassette, double support and M3 clamp geometries."""

from build123d import Axis, Pos, Rot

from ..geometry import box, cylinder, ring, slot
from ..params import (F623_FLANGE_SEAT, F623_SEAT, FRAME_WALL, HORN_PCD,
                      M3_CLEAR, M3_INSERT, M3_INSERT_DEPTH, MG996R)


# All axial stations use Z=0 at the mating face of the aluminium horn.
IDLER_SEAT_TOP = -55.0
IDLER_BEARING_TOP = IDLER_SEAT_TOP - 0.25
IDLER_SPACER = 1.5
FLANGE_RECESS_DEPTH = 1.3  # 1 mm flange plus the 0.25 mm axial seating setback.


def servo_mount():
    """Open-bottom MG996R cassette. Local datum is the servo bottom face."""
    s = MG996R
    x = s.body_center_x
    plate = box((61, 29, 8), (x, 0, 23))
    plate -= box((s.pocket[0], s.pocket[1], 42), (x, 0, 20))
    for hx, hy in s.ear_holes:
        plate -= slot(M3_CLEAR, 1.4, 12, (hx, hy, 17))
    return plate.clean()


def horn_plate(thickness=3):
    plate = ring(26, 9, thickness)
    for x, y in ((7, 0), (-7, 0), (0, 7), (0, -7)):
        plate -= cylinder(M3_CLEAR, thickness + 2, (x, y, -1))
    return plate


def bearing_plate(double=False):
    """Stepped, flanged seat; bearings insert from opposite faces when doubled."""
    depth = 8.5 if double else 4.5
    plate = box((27, 28, depth), (0, 0, depth / 2))
    plate -= cylinder(F623_SEAT, depth + 2, (0, 0, -1))
    if double:
        plate -= cylinder(F623_FLANGE_SEAT, FLANGE_RECESS_DEPTH)
    plate -= cylinder(F623_FLANGE_SEAT, FLANGE_RECESS_DEPTH,
                      (0, 0, depth - FLANGE_RECESS_DEPTH))
    return plate


def bearing_locations(double=False):
    """Back-to-back double bearings, with flanges on the two outer faces."""
    upper_base = IDLER_BEARING_TOP - 4
    locations = [Pos(0, 0, upper_base)]
    if double:
        locations.append(Pos(0, 0, upper_base) * Rot(180, 0, 0))
    return locations


def idler_cap():
    """Bottom cup registers on the casing, not on undocumented case screws."""
    s = MG996R
    bottom = IDLER_BEARING_TOP + IDLER_SPACER + s.horn_face
    top = 2.8
    cup = box((46.3, 25.3, top - bottom), (s.body_center_x, 0, (bottom + top) / 2))
    cup -= box((41.3, 20.3, 12), (s.body_center_x, 0, bottom + 2.8 + 6))
    cup -= cylinder(M3_CLEAR, 12, (0, 0, bottom - 1))
    cup -= box((9, 8, 5), (30, 0, 1.5))
    return cup.clean()


def clevis(double=False):
    """Drive and counter-bearing plates joined clear of the servo casing."""
    depth = 8.5 if double else 4.5
    drive = horn_plate()
    bottom = IDLER_SEAT_TOP - depth
    rear = Pos(0, 0, bottom) * bearing_plate(double)
    bridge = box((3.2, 28, 3 - bottom), (-28, 0, (3 + bottom) / 2))
    drive += box((21.6, 20, 3), (-18.8, 0, 1.5))
    rear += box((21.6, 20, depth), (-18.8, 0, bottom + depth / 2))
    part = drive + rear + bridge
    # Recut after joining the bridge: adding a web must not refill a mating hole.
    for x, y in ((7, 0), (-7, 0), (0, 7), (0, -7)):
        part -= cylinder(M3_CLEAR, 5, (x, y, -1))
    part -= cylinder(F623_SEAT, depth + 2, (0, 0, bottom - 1))
    recesses = [IDLER_SEAT_TOP - FLANGE_RECESS_DEPTH]
    if double:
        recesses.append(bottom)
    for z in recesses:
        part -= cylinder(F623_FLANGE_SEAT, FLANGE_RECESS_DEPTH, (0, 0, z))
    return part.clean()


def insert_boss(x, y, z=0, height=9):
    return cylinder(9.0, height, (x, y, z)) - cylinder(
        M3_INSERT, M3_INSERT_DEPTH, (x, y, z + height - M3_INSERT_DEPTH))


def pcb_clamp():
    # Origin sits on the boss/PCB top plane; the +Y tongue bears on the PCB edge.
    base = box((8, 10, 2.8), (0, 2, 1.4))
    return base - cylinder(M3_CLEAR, 5, (0, 0, -1))
