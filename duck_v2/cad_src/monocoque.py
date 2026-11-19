"""Connected monocoque structural shells with integrated joints and clean clearances."""

from math import cos, radians, sin
from functools import cache

from build123d import Axis, Circle, Cylinder, Ellipse, Plane, Pos, Rectangle, Rot, Sphere, fillet, loft, revolve

from .armor.panels import ellipsoid_shell, reactor
from .geometry import box, cylinder, ring, slot
from .layout import (CAMERA_LOCATION, CAPACITOR_LOCATION, STATION_BY_JOINT,
                     TOF_Z, UBEC_LOCATION, HEAD_FORWARD, IMU_LOCATION,
                     LOGIC_BUCK_LOCATION, SWITCH_LOCATION)
from .params import (F623_FLANGE_SEAT, F623_SEAT, M3_CLEAR, M3_INSERT,
                     M3_INSERT_DEPTH, MG996R, SHIN_MM, THIGH_MM, TOF_ANGLES)
from .skeleton.interfaces import (FLANGE_RECESS_DEPTH, IDLER_SEAT_TOP,
                                  bearing_plate, horn_plate, idler_cap, servo_mount)
from .skeleton.torso import battery_hatch, torso_lower


def rounded_box(size, radius, center=(0, 0, 0), axis=Axis.Z):
    """Fillet the 4 edges parallel to `axis` (pure analytic planes + cylinders, watertight STL)."""
    return Pos(*center) * fillet(box(size).edges().filter_by(axis), radius)


def checked(name, part):
    part = part.clean()
    if not part.is_valid or len(part.solids()) != 1 or part.volume <= 0:
        raise ValueError(f'{name} must be one connected valid positive-volume solid')
    return part


def slim_servo_mount():
    """Lightweight 3.6 mm servo flange with 2.6 mm perimeter walls."""
    s = MG996R
    x = s.body_center_x
    plate = box((58.0, 26.2, 3.6), (x, 0, 25.2))
    plate -= box((s.pocket[0], s.pocket[1], 42), (x, 0, 20))
    for hx, hy in s.ear_holes:
        plate -= slot(M3_CLEAR, 1.4, 12, (hx, hy, 18))
    return plate.clean()


def slim_idler_cap():
    """Lightweight 2.6 mm wall bottom idler cup registering on the MG996R casing."""
    s = MG996R
    bottom = IDLER_SEAT_TOP - 0.25 + 1.5 + s.horn_face
    top = 2.6
    cup = box((45.9, 24.9, top - bottom), (s.body_center_x, 0, (bottom + top) / 2))
    cup -= box((41.3, 20.3, 12), (s.body_center_x, 0, bottom + 2.6 + 6))
    cup -= cylinder(M3_CLEAR, 12, (0, 0, bottom - 1))
    cup -= box((9, 8, 5), (30, 0, 1.5))
    return cup.clean()


def servo_clearance(station):
    """Exact MG996R body + ear + collar + ear-screw clearance volume."""
    s = MG996R
    cutter = box(s.pocket, (s.body_center_x, 0, s.pocket[2] / 2 - 0.4))
    cutter += box((s.ear_pocket_length, s.body[1] + 0.8, 3.2),
                  (s.body_center_x, 0, s.ear_z + 1.5))
    cutter += cylinder(s.collar_diameter + 0.8, 12, (0, 0, s.body[2] - 0.5))
    for x, y in s.ear_holes:
        cutter += slot(M3_CLEAR, 1.4, 16, (x, y, 16.0))
        cutter += cylinder(6.4, 8.0, (x, y, s.ear_z + s.ear_thickness - 0.3))
    if station.bearings:
        cutter += cylinder(M3_CLEAR, 16, (0, 0, -14.0))
        cutter += cylinder(6.2, 4.0, (0, 0, -3.5))
    return station.servo_base * cutter


def recut_station(part, station):
    return (part - servo_clearance(station)).clean()


def recut_horn_and_bearing(part, moving_loc, double=False):
    """Re-open precision horn M3 bores, Ø9 central access bore, and F623ZZ seats."""
    depth = 8.5 if double else 4.5
    bottom = IDLER_SEAT_TOP - depth
    part -= moving_loc * cylinder(9.2, 8.0, (0, 0, -1.0))
    for x, y in ((7, 0), (-7, 0), (0, 7), (0, -7)):
        part -= moving_loc * cylinder(M3_CLEAR, 8.0, (x, y, -1.0))
        part -= moving_loc * cylinder(6.2, 4.5, (x, y, 3.0))
    part -= moving_loc * cylinder(20.8, 5.2, (0, 0, -5.2))
    part -= moving_loc * cylinder(F623_SEAT, depth + 4, (0, 0, bottom - 2))
    recesses = [IDLER_SEAT_TOP - FLANGE_RECESS_DEPTH]
    if double:
        recesses.append(bottom)
    for z in recesses:
        part -= moving_loc * cylinder(F623_FLANGE_SEAT, FLANGE_RECESS_DEPTH + 0.1, (0, 0, z))
    part -= moving_loc * cylinder(10.0, 12.0, (0, 0, bottom - 12.0))
    return part.clean()


def chest_loft_shell(wall=1.85):
    """Sculpted CyberDuck V2 wide aero ellipsoid chest shell (encloses electronics & shoulder deck)."""
    shell = ellipsoid_shell((69.0, 68.2, 35.0), wall).translate((0, 0, 2))
    # Open rear face cleanly at X = -52 for backpack service access & cable routing.
    shell -= box((40, 150, 100), (-72, 0, 2))
    # Open belly cleanly below Z = -22.5 to meet trunk_bottom battery bay.
    shell -= box((160, 150, 40), (0, 0, -42.5))
    return shell.clean()


@cache
def trunk_front():
    """Lightweight monocoque wide aero chest enclosing all trunk electronics & hip_roll shoulders."""
    part = chest_loft_shell(wall=1.85)
    # Mid-wall ellipsoid clipper ensures internal ribs/shelves fuse into wall without poking outside.
    inner_clip = Sphere(1).scale((68.0, 67.2, 34.0)).translate((0, 0, 2))
    # Recessed circular socket on front chest nose for cyan reactor bezel.
    part -= Pos(66.5, 0, 6.0) * Rot(0, 90, 0) * cylinder(23.6, 16, (0, 0, -8))
    # Slim perimeter base ring at Z in [-23.4, -21.2] (above trunk_bottom Z=-24.9).
    deck = box((82, 68, 2.2), (2, 0, -22.3)) - box((66, 54, 6), (-2, 0, -22.3))
    part += deck & inner_clip
    # Integrated slim servo mounts and idler cups for head_yaw and both hip_roll stations.
    for name in ('head_yaw', 'left_hip_roll', 'right_hip_roll'):
        st = STATION_BY_JOINT[name]
        part += st.servo_base * slim_servo_mount()
        if st.bearings:
            part += st.servo_base * slim_idler_cap()
    # Two lightened longitudinal ribs tying head_yaw mount from bottom deck straight to chest roof.
    for y in (-12.9, 12.9):
        rib = (box((44, 2.0, 59.0), (14, y, 6.5))
               - box((30, 6, 22), (14, y, -7.0))
               - box((28, 6, 18), (14, y, 21.0)))
        part += rib & inner_clip
    # Slim structural webs tying hip_roll mounts & idler cups to chest shoulders (X in [-27, +14]).
    for sign in (-1, 1):
        web = (box((30, 2.0, 34), (-10, sign * 38.5, -6))
               - box((18, 6, 18), (-10, sign * 38.5, -4))
               + box((4.0, 14, 12), (3.0, sign * 42, -16))
               + box((4.0, 14, 12), (-22.0, sign * 42, -16)))
        part += web
    # Neck 6803-2RS bearing outer-race seat at (6, 0, 34.5..39.5), blended cleanly into Z=37 roof.
    part += Pos(6, 0, 32.0) * ring(30.5, 23.5, 7.5)
    part -= Pos(6, 0, 34.3) * cylinder(26.15, 5.6)
    part -= Pos(6, 0, 18.0) * cylinder(27.0, 14.0)
    part -= Pos(6, 0, 18.0) * cylinder(23.5, 23.0)
    # Rigid IMU support tab at (48, 0, -14), tied to front nose wall.
    part += box((24, 16, 2.0), (50, 0, -15.7)) & inner_clip
    # Slim lightened power shelves for UBEC (+Y) and capacitor (-Y) fused cleanly into chest flanks.
    ubec_shelf = box((36, 48, 2.0), (-10, 46, 1.5)) - box((22, 22, 6), (-10, 42, 1.5))
    cap_shelf = box((26, 48, 2.0), (-10, -46, 3.3)) - box((14, 22, 6), (-10, -42, 3.3))
    part += (ubec_shelf + cap_shelf) & inner_clip

    # Carve exact clearances for all 3 trunk servos, power components, and moving hip sweeps.
    for name in ('head_yaw', 'left_hip_roll', 'right_hip_roll'):
        part = recut_station(part, STATION_BY_JOINT[name])
    for sign in (-1, 1):
        # Sculpted wheel-well arches under the shoulder hoods clearing hip_roll ±25.2° sweep.
        part -= box((34, 46, 52), (34, sign * 54, -40))
        part -= box((36, 46, 52), (-46, sign * 54, -40))
        part -= box((100, 26, 52), (-4, sign * 75, -36))
        part -= box((92, 28, 26), (-4, sign * 58, -28))
        # Sleek symmetrical rear-flank aero cooling vents (-6 g, keeps shoulder hoods crisp).
        part -= rounded_box((28, 20, 14), 5.0, (-30, sign * 58, 8), Axis.Y)
    # Clear trunk_bottom / battery volume strictly below Z = -24.0.
    part -= box((120, 55, 30), (-6, 0, -39.2))
    # Clear power & electronics envelopes including dual_esp32s3 ldo_2.
    part -= UBEC_LOCATION * box((50.8, 25.8, 15.0))
    part -= CAPACITOR_LOCATION * cylinder(19.2, 36.0, (0, 0, -0.5))
    part -= IMU_LOCATION * box((21.0, 17.0, 4.0), (0, 0, 1.6))
    part -= LOGIC_BUCK_LOCATION * box((41.2, 23.2, 11.6), (0, 0, 5.5))
    part -= SWITCH_LOCATION * box((19.8, 13.8, 23.0))
    part -= Pos(48, 28, -14) * box((16, 13, 6), (0, 0, 1.5))
    # Clear rear harnesses strictly behind X = -47.
    part -= box((18, 24, 42), (-54, 0, -10))
    part -= box((18, 26, 30), (-54, 0, 22))
    for sign in (-1, 1):
        part -= box((18, 22, 28), (-51, sign * 30, -10))
    part -= box((40, 160, 140), (-72.0, 0, 0))
    return checked('trunk_front', part)


@cache
def trunk_bottom():
    return checked('trunk_bottom', torso_lower())


@cache
def neck_link():
    """Lightweight monocoque neck coupling head_yaw horn + 6803 journal to head_pitch (Z=50)."""
    st_hp = STATION_BY_JOINT['head_pitch']
    part = horn_plate(3.0)
    # Four diagonal pillars at 45 deg join horn_plate (Z<=3) to 6803 journal (Z>=6.1),
    # completely bypassing the four M3 horn_screw heads at (±7, 0) and (0, ±7).
    for sx in (-1, 1):
        for sy in (-1, 1):
            part += cylinder(3.2, 5.2, (sx * 4.6, sy * 4.6, 2.0))
    part += ring(16.6, 9.2, 7.5, (0, 0, 6.1))
    part += ring(26.0, 9.2, 3.2, (0, 0, 12.6))
    part += st_hp.servo_base * slim_servo_mount()
    part += st_hp.servo_base * slim_idler_cap()
    # Two slim 3.0 mm structural side webs + sleek front throat fairing hiding raw webs.
    part += box((18, 3.0, 26.0), (8, 5.2, 26.5))
    part += box((16, 3.0, 27.0), (6, -18.5, 27.0))
    part += box((18, 27.0, 3.0), (8, -6.5, 14.1))
    part += rounded_box((3.0, 26.7, 23.0), 1.4, (16.5, -6.65, 25.0))
    part = recut_station(part, st_hp)
    part -= cylinder(9.2, 16.0, (0, 0, -1.0))
    for x, y in ((7, 0), (-7, 0), (0, 7), (0, -7)):
        part -= cylinder(M3_CLEAR, 8.0, (x, y, -1.0))
        part -= cylinder(6.2, 3.1, (x, y, 3.0))
    part -= ring(36.0, 16.8, 6.2, (0, 0, 6.0))
    part -= box((24, 24, 70), (-42, 0, 35))
    return checked('neck_link', part)


@cache
def hip_clevis():
    """Lightweight left hip clevis linking hip_roll (X axis) to hip_pitch servo (Y axis)."""
    st_hr = STATION_BY_JOINT['left_hip_roll']
    st_hp = STATION_BY_JOINT['left_hip_pitch']
    ml_hr = st_hr.moving_location

    drive = horn_plate(3.0) + box((14, 22, 3.0), (0, -11, 1.5))
    rear = Pos(0, 0, IDLER_SEAT_TOP - 8.5) * (bearing_plate(True) + box((20, 22, 8.5), (0, -11, 4.25)))
    part = ml_hr * (drive + rear)
    # Outer side bridge at Y in [+21.0, +24.0] clears hip_roll servo across full ±25.2° roll!
    bridge = box((66.5, 3.0, 21.5), (-4.25, 22.5, 1.75))
    bridge -= box((36.0, 8.0, 10.0), (-4.25, 22.5, 1.75))
    part += bridge
    part += st_hp.servo_base * slim_servo_mount()
    part += st_hp.servo_base * slim_idler_cap()
    # Stiff upper-front spar strictly outside thigh_L sweep radius (X in [+25, +31], Z in [+8.5, +13.0]).
    part += box((6.0, 53.0, 4.5), (28.0, 48.0, 10.75))

    part = recut_horn_and_bearing(part, ml_hr, double=True)
    part = recut_station(part, st_hp)
    part -= box((56, 36, 36), (-2, 0, 0))
    return checked('hip_clevis', part)


@cache
def thigh_shell():
    """Lightweight monocoque left thigh linking hip_pitch clevis to vertical knee_pitch servo."""
    st_hp = STATION_BY_JOINT['left_hip_pitch']
    st_kp = STATION_BY_JOINT['left_knee_pitch']
    ml_hp = st_hp.moving_location

    part = ml_hp * horn_plate(3.0)
    part += ml_hp * Pos(0, 0, IDLER_SEAT_TOP - 8.5) * bearing_plate(True)
    part += st_kp.servo_base * slim_servo_mount()
    part += st_kp.servo_base * slim_idler_cap()

    # Slim outer & inner cheek webs positioned slightly rearward (X in [-14, +2]) below Z=-8.5
    # so the thigh joins horn/bearing rims cleanly without intersecting M3 bores and clears hip_pitch.
    part += box((16, 3.0, 21.0), (-6, 42.5, -19.0))
    part += box((16, 3.2, 31.0), (-6, -20.9, -24.0))
    # Lightweight 3.2 mm upper cross-yoke at X in [-14, +2], Z=-28.0 joining cheeks to knee_pitch.
    yoke = box((16, 66.5, 3.2), (-6, 10.75, -28.0))
    yoke -= box((8, 26.0, 8.0), (-6, 22.0, -28.0))
    part += yoke
    part += box((12, 16, 4.0), (-4, 3.4, -29.0))
    # Sculpted curved wraparound front-outer thigh armor (strictly Z <= -39 to clear hip_pitch).
    front_shell = rounded_box((11.0, 40, 20), 3.0, (12.5, 0, -49.0))
    front_shell -= box((11.0, 35.6, 24), (10.3, 0, -49.0))
    part += front_shell

    part = recut_horn_and_bearing(part, ml_hp, double=True)
    part = recut_station(part, st_kp)
    # Relief bevel on top-front flange tip (strictly outside servo body X<=10.0) to clear hip_pitch at -0.87 rad.
    part -= box((6, 22, 6), (13.2, 4, -29.5))
    part -= box((72, 50, 32), (10, 12, 0))
    part -= box((48, 20, 44), (0, 36, -68))
    part -= box((48, 20, 44), (0, -36, -68))
    return checked('thigh', part)


@cache
def shin_shell():
    """Lightweight monocoque left shin with integrated knee shield and 4-bolt ankle flange."""
    st_kp = STATION_BY_JOINT['left_knee_pitch']
    ml_kp = st_kp.moving_location

    part = ml_kp * horn_plate(3.0)
    part += ml_kp * Pos(0, 0, IDLER_SEAT_TOP - 4.5) * bearing_plate(False)
    part += box((18, 3.0, 27), (0, 29.5, -13.5))
    part += box((18, 3.6, 27), (0, -29.3, -13.5))
    # Slim 3.2 mm lower knee yoke clearing knee_pitch servo across 0..85° (-4 g/pair).
    part += box((16, 62.0, 3.2), (0, -0.1, -26.0))
    # Hollow graphite shin column (2.6 mm walls) with clean side lightening slots.
    column = rounded_box((20, 24, 32), 2.5, (2, 0, -42.0))
    column -= box((14.8, 18.8, 28), (2, 0, -42.0))
    column -= box((10, 30, 16), (2, 0, -43.0))
    part += column
    # Sculpted front shin & knee guard.
    guard = rounded_box((5.5, 24, 24), 2.0, (12.5, 0, -37.0))
    guard -= box((4.0, 19.6, 26), (10.5, 0, -37.0))
    part += guard
    # Flat ankle flange at Z in [-59.0, -55.4] mating flush on top of foot_L.
    part += box((26, 26, 3.6), (2, 0, -57.2))
    for x in (-7, 11):
        for y in (-8, 8):
            part -= cylinder(M3_CLEAR, 10, (x, y, -61))

    part = recut_horn_and_bearing(part, ml_kp, double=False)
    part -= box((40, 54, 48), (0, -1, 2))
    return checked('shin', part)


@cache
def integrated_foot():
    """Lightweight rounded mecha boot foot (Z in [-5.0, 0.0], hollowed underside)."""
    plate = box((74, 40, 5.0), (4, 0, -2.5))
    plate = fillet(plate.edges().filter_by(Axis.Z), 9.0)
    # Weight-saving underside pocket leaving 2.2 mm top deck (>= 1.8 mm PLA).
    plate -= box((58, 28, 2.8), (4, 0, -3.6))
    # Hollow front boot toe dome strictly forward of shin ankle flange.
    toe = rounded_box((20, 30, 4.2), 4.0, (26, 0, 1.6))
    toe -= box((15.6, 25.6, 3.2), (25, 0, 0.1))
    plate += toe
    for x in (-11, 7):
        for y in (-8, 8):
            plate -= cylinder(M3_CLEAR, 10, (x, y, -7))
    for x in (-24, 30):
        for y in (-14, 14):
            plate -= cylinder(M3_CLEAR, 10, (x, y, -7))
    return checked('foot', plate)


@cache
def sole_tread():
    """5.0 mm TPU non-slip sole at exact Z=0.000 ground datum with >= 2.2 mm solid web."""
    part = box((82, 46, 5.0), (4, 0, -7.5))
    part = fillet(part.edges().filter_by(Axis.Z), 9.0)
    # Upper weight-saving pocket stays clear of M3 screw counterbores (leaves >= 2.4 mm TPU base).
    part -= box((46, 24, 2.4), (3, 0, -6.2))
    for x in (-12, 0, 12):
        part -= box((3.4, 38, 1.2), (x, 0, -9.4))
    for x in (-24, 30):
        for y in (-14, 14):
            part -= cylinder(5.8, 1.8, (x, y, -10.0))
            part -= cylinder(M3_CLEAR, 7.0, (x, y, -11.0))
    return checked('sole_tread', part)


@cache
def head_front():
    """Lightweight pebble mecha helmet with integrated head_pitch clevis, crest & rosettes."""
    shell = ellipsoid_shell((48, 52, 36), 1.85).translate((-3, 0, 23))
    shell -= box((34, 80, 26), (38, 0, 24))
    shell -= box((30, 120, 96), (-57, 0, 22))
    shell -= box((58, 54, 36), (-4, 0, -14))
    shell += rounded_box((42, 6, 5.5), 2.0, (-4, 0, 58.0), Axis.Y)
    for sign in (-1, 1):
        shell += Pos(-6, sign * 50, 22) * Rot(-90 * sign, 0, 0) * cylinder(17, 3.6)
    shell += rounded_box((9, 30, 5.5), 2.2, (33, 0, -2), Axis.Z)
    part = shell.translate((HEAD_FORWARD, 0, 0))

    st_hp = STATION_BY_JOINT['head_pitch']
    ml_hp = st_hp.moving_location
    part += ml_hp * horn_plate(3.0)
    part += ml_hp * Pos(0, 0, IDLER_SEAT_TOP - 4.5) * bearing_plate(False)
    part += box((18, 3.0, 23), (6, 31.5, 11.5))
    part += box((18, 3.8, 23), (6, -27.2, 11.5))
    part += box((18, 22, 3.0), (6, 40, 21))
    part += box((18, 22, 3.0), (6, -38, 21))

    part = recut_horn_and_bearing(part, ml_hp, double=False)
    # Clear neck_link & head_pitch servo across the full -30° .. +45° pitch sweep.
    part -= box((64, 58, 38), (6, 0, -6))
    part -= CAMERA_LOCATION * box((26, 25, 20), (0, 0, 8))
    part -= Pos(32 + HEAD_FORWARD, 0, -6) * box((16, 14, 6), (0, 0, 1.5))
    part -= Pos(6 + HEAD_FORWARD, 0, 39) * box((33, 20, 6), (0, 0, 1.5))
    part -= Pos(-22 + HEAD_FORWARD, 0, 34) * box((32, 73, 14), (0, 0, 5))
    part -= Pos(12 + HEAD_FORWARD, 28, 28) * box((15, 12, 5), (0, 0, 1.5))
    for angle in TOF_ANGLES:
        a = radians(angle)
        loc = Pos(-26 + HEAD_FORWARD + 64 * cos(a), 64 * sin(a), TOF_Z) * Rot(0, 0, angle) * Rot(0, 90, 0)
        part -= loc * box((16, 13, 8), (0, 0, 2))
    # Rear opening cleanly clears head_rear_cover (X <= -37.2 + HEAD_FORWARD).
    part -= box((40, 120, 100), (-57.2 + HEAD_FORWARD, 0, 22))
    return checked('head_front', part)


@cache
def head_rear_cover():
    """Flush contoured rear head cap cleanly enclosing rear helmet opening & dual_esp32s3 PCB."""
    plate = box((2.0, 58, 30), (-39.6, 0, 30))
    plate = fillet(plate.edges().filter_by(Axis.X), 11.0)
    # Compact PCB shelf strictly clear of curved inner helmet sidewalls & head_pitch sweep.
    plate += box((10, 34, 2.0), (-33.8, 0, 28.2))
    # Rear cable pass-through slot for head_harness at (Y=4, Z=19..33).
    plate -= box((14, 18, 16), (-39.6, 4, 24))
    for y in (-22, 22):
        plate -= Pos(-42, y, 30) * Rot(0, 90, 0) * cylinder(M3_CLEAR, 8)
    return checked('head_rear_cover', plate.translate((HEAD_FORWARD, 0, 0)))


@cache
def visor_lens():
    """Flush curved cyan optical visor with uniform 2.4 mm radial thickness and sensor ports."""
    profile = Plane.XZ * Pos(70.8, 24.5) * Rectangle(2.4, 23.0)
    part = Pos(-26, 0, 0) * Rot(0, 0, -33) * revolve(profile, Axis.Z, 66.0)
    # Clean bottom U-notch for camera lens at Z=12 (avoids thin crescent below the bore).
    part -= Pos(30, 0, 12) * Rot(0, 90, 0) * cylinder(11, 30)
    part -= box((30, 11, 10), (42, 0, 11))
    for angle in TOF_ANGLES:
        a = radians(angle)
        optical = Pos(-26 + 64 * cos(a), 64 * sin(a), TOF_Z) * Rot(0, 0, angle) * Rot(0, 90, 0)
        part -= optical * cylinder(7, 30)
    return checked('visor_lens', part.translate((HEAD_FORWARD, 0, 0)))
