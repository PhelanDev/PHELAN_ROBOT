"""Connected Cyber-Athletic monocoque structural shells with integrated joints and clean clearances."""

from functools import cache
from math import cos, radians, sin, sqrt

from build123d import (Align, Axis, Box, Circle, Cone, Cylinder, Plane, Pos,
                       Rectangle, RectangleRounded, Rot, extrude, fillet, loft, make_hull, revolve)
import numpy as np

from .geometry import box, cylinder, ring, slot
from .layout import (CAMERA_LOCATION, CAPACITOR_LOCATION, HEAD_FORWARD, IMU_LOCATION,
                     LOGIC_BUCK_LOCATION, MIC_LOCATION, OLED_LOCATION,
                     STATION_BY_JOINT, SWITCH_LOCATION, THIGH_COVER_BOSSES,
                     TOF_LOCATION, UBEC_LOCATION)
from .params import (F623_FLANGE_SEAT, F623_SEAT, M3_CLEAR, M3_INSERT, MG996R)
from .skeleton.interfaces import IDLER_SEAT_TOP, FLANGE_RECESS_DEPTH, bearing_plate, horn_plate
from .skeleton.torso import battery_hatch, torso_lower


def rounded_box(size, radius, center=(0, 0, 0), axis=Axis.Z):
    """Fillet the 4 edges parallel to `axis` (analytic planes + cylinders, watertight STL)."""
    return Pos(*center) * fillet(box(size).edges().filter_by(axis), radius)


def plane_cut(normal, distance, size=320.0):
    """Half-space cutter removing region n . p > distance for unit normal n."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    pl = Plane(origin=tuple(n * distance), z_dir=tuple(n))
    return pl * Pos(0, 0, size / 2) * Box(size, size, size)


def faceted_solid(half_extents, planes, center=(0, 0, 0)):
    """Closed convex polyhedron bounded by +/- half_extents and planar bevels."""
    hx, hy, hz = half_extents
    sol = Box(2 * hx, 2 * hy, 2 * hz)
    for normal, dist in planes:
        sol -= plane_cut(normal, dist)
    return Pos(*center) * sol.clean()


def faceted_shell(half_extents, planes, wall=1.9, center=(0, 0, 0)):
    """Hollow faceted shell with uniform exact wall thickness along every facet normal."""
    outer = faceted_solid(half_extents, planes, center)
    inner = faceted_solid(tuple(e - wall for e in half_extents),
                          [(n, d - wall) for n, d in planes], center)
    return (outer - inner).clean()


def checked(name, part):
    part = part.clean()
    if not part.is_valid or len(part.solids()) != 1 or part.volume <= 0:
        raise ValueError(f'{name} must be one connected valid positive-volume solid')
    return part


# Compute exact cheek bevel distance so Vector(40, 53, -6) sits centered in the 1.9 mm chest wall.
_CHEST_CHEEK_N = np.array([0.58, 0.78, -0.23])
_CHEST_CHEEK_N = _CHEST_CHEEK_N / np.linalg.norm(_CHEST_CHEEK_N)
_CHEST_CHEEK_D = float(np.dot(_CHEST_CHEEK_N, np.array([40.0, 53.0, -10.0]))) + 0.95

CHEST_PLANES = (
    ((0.42, 0.0, 0.907), 35.0),
    ((0.88, 0.0, -0.475), 61.0),
    ((0.0, 0.42, 0.907), 39.5),
    ((0.0, -0.42, 0.907), 39.5),
    ((0.0, 0.88, -0.475), 54.0),
    ((0.0, -0.88, -0.475), 54.0),
    (tuple(_CHEST_CHEEK_N), _CHEST_CHEEK_D),
    ((_CHEST_CHEEK_N[0], -_CHEST_CHEEK_N[1], _CHEST_CHEEK_N[2]), _CHEST_CHEEK_D),
    ((-0.45, 0.89, 0.0), 67.5),
    ((-0.45, -0.89, 0.0), 67.5),
    ((0.48, 0.55, 0.68), 53.5),
    ((0.48, -0.55, 0.68), 53.5),
)

def _rounded_stadium(width, height, radius, depth, center=(0, 0, 0)):
    """Extrude a smooth RectangleRounded profile in the YZ plane along +X centered at `center`."""
    cx, cy, cz = center
    pl = Plane(origin=(cx - depth / 2, cy, cz), x_dir=(0, 1, 0), z_dir=(1, 0, 0))
    r = min(radius, width / 2 - 0.1, height / 2 - 0.1)
    return extrude(pl * RectangleRounded(width, height, r), depth)


def _loft_stations(stations):
    """Smooth multi-station YZ RectangleRounded loft along X: (x, width, zmin, zmax, radius)."""
    faces = []
    for x, w, zmin, zmax, r in stations:
        h = zmax - zmin
        cz = (zmin + zmax) / 2.0
        pl = Plane(origin=(x, 0.0, cz), x_dir=(0, 1, 0), z_dir=(1, 0, 0))
        faces.append(pl * RectangleRounded(w, h, min(r, w / 2 - 0.5, h / 2 - 0.5)))
    return loft(faces, ruled=False).clean()


def _xz_hull_face(y_val, circles_xzr):
    """Smooth tangent convex hull of (x, z, r) circles in XY mapped to XZ at Y=y_val."""
    edges = []
    for cx, cz, r in circles_xzr:
        edges.extend((Pos(cx, cz) * Circle(r)).edges())
    return Pos(0.0, y_val, 0.0) * Rot(90, 0, 0) * make_hull(edges)


def _yz_hull_face(x_val, circles_yzr):
    """Smooth tangent convex hull of (y, z, r) circles in XY mapped to YZ at X=x_val."""
    edges = []
    for cy, cz, r in circles_yzr:
        edges.extend((Pos(cy, cz) * Circle(r)).edges())
    pl = Plane(origin=(x_val, 0.0, 0.0), x_dir=(0, 1, 0), z_dir=(1, 0, 0))
    return pl * make_hull(edges)


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


def horn_hub_cap(wall=2.65, skirt_depth=2.4):
    """Sculpted chamfered joint dome enclosing horn_plate, 4 M3 bolts, shaft screw & rear collar."""
    cap = cylinder(29.0, 6.6, (0, 0, 0))
    cap += Pos(0, 0, 6.6) * Cone(14.5, 11.0, wall + 0.5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    if skirt_depth > 0:
        cap += ring(28.0, 21.6, skirt_depth, (0, 0, -skirt_depth))
    # Hollow pocket clearing M3 socket heads (z in [3.0, 6.0], r <= 10.2 mm).
    cap -= cylinder(22.6, 3.8, (0, 0, 3.0))
    # Preserve horn mating pocket, hub bore & M3 through-bores inside z <= 3.1.
    cap -= cylinder(20.8, 5.2, (0, 0, -5.2))
    cap -= cylinder(9.2, 4.2, (0, 0, -1.0))
    for x, y in ((7, 0), (-7, 0), (0, 7), (0, -7)):
        cap -= cylinder(M3_CLEAR, 4.2, (x, y, -1.0))
    return cap.clean()


def idler_hub_cap(bottom, tip_offset=10.2, wall=2.65):
    """Sculpted tapered idler dome enclosing protruding M3 bolt, washer & nyloc nut."""
    total_h = tip_offset + wall
    dome = Pos(0, 0, bottom - total_h) * Cone(10.5, 13.5, total_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    dome -= cylinder(12.8, tip_offset + 0.4, (0, 0, bottom - tip_offset))
    return dome.clean()


def _idler_nut_cap(y_start, r_in, wall, total_h, bevel=2.0):
    """Uniform-wall press-fit cylindrical cap with concentric 45-deg beveled dome along -Y."""
    r_out = r_in + wall
    h_cyl = total_h - bevel
    r_top = r_out - bevel
    outer = cylinder(2 * r_out, h_cyl, (0, 0, 0))
    outer += Pos(0, 0, h_cyl) * Cone(r_out, r_top, bevel, align=(Align.CENTER, Align.CENTER, Align.MIN))
    k = sqrt(2.0) - 1.0
    z_in_cyl = h_cyl - k * wall
    z_in_top = total_h - wall
    r_in_top = r_top - k * wall
    inner = cylinder(2 * r_in, z_in_cyl + 0.1, (0, 0, -0.1))
    inner += Pos(0, 0, z_in_cyl) * Cone(r_in, r_in_top, z_in_top - z_in_cyl,
                                        align=(Align.CENTER, Align.CENTER, Align.MIN))
    return (Pos(0, y_start, 0) * Rot(90, 0, 0) * (outer - inner)).clean()


def chest_loft_shell(wall=1.82):
    """Cyber-Athletic V-tapered faceted chest cowl with muscular shoulders & sternum apron."""
    shell = faceted_shell((62.5, 66.0, 32.0), CHEST_PLANES, wall=wall, center=(0, 0, 4))
    shell -= box((40, 160, 100), (-72, 0, 4))
    # Cut lateral/rear skirts at Z=-22.5 while keeping front V-tapered abdomen apron to Z=-39.0.
    shell -= box((109, 160, 40), (-10.5, 0, -42.5))
    for sign in (-1, 1):
        shell -= box((160, 56, 40), (0, sign * 51.5, -42.5))
    shell -= box((160, 160, 40), (0, 0, -59.0))
    return shell.clean()


@cache
def trunk_front():
    """Cyber-Athletic faceted monocoque chest integrating head_yaw, hips, collar & power bays."""
    part = chest_loft_shell(wall=1.82)
    inner_clip = faceted_solid((61.5, 65.0, 31.0),
                               [(n, d - 0.90) for n, d in CHEST_PLANES],
                               center=(0, 0, 4))

    # Slim forward deck flange at Z in [-23.2, -21.4] tying longitudinal ribs cleanly to front walls.
    deck = box((44, 48, 1.82), (14, 0, -22.3)) - box((34, 38, 6), (12, 0, -22.3))
    part += deck & inner_clip

    # Integrated slim servo mounts and idler cups for head_yaw and pulled-in hip_roll (Y=±42).
    for name in ('head_yaw', 'left_hip_roll', 'right_hip_roll'):
        st = STATION_BY_JOINT[name]
        part += st.servo_base * slim_servo_mount()
        if st.bearings:
            part += st.servo_base * slim_idler_cap()

    # Two lightened longitudinal truss ribs tying head_yaw mount from deck straight to roof.
    for y in (-12.9, 12.9):
        rib = (box((64, 1.82, 59.0), (12, y, 6.5))
               - box((38, 6, 30), (16, y, -9.0))
               - box((30, 6, 34), (34, y, 6.0)))
        part += rib & inner_clip

    # Slim structural webs tying hip_roll mounts & idler cups (Y=±42) into chest shoulders.
    for sign in (-1, 1):
        web = (box((26, 1.82, 30), (-10, sign * 31.5, -6))
               - box((20, 6, 22), (-10, sign * 31.5, -4))
               + box((2.4, 10, 8), (3.0, sign * 35, -16))
               + box((2.4, 10, 8), (-22.0, sign * 35, -16)))
        part += web
        # Integrated 1.82 mm PLA hip_roll flank shield & rear baffle enclosing servo body up to chest skirt.
        flank = (box((43.6, 1.82, 31.0), (-1.1, sign * 53.5, -28.5))
                 + box((1.82, 13.0, 31.0), (-22.6, sign * 47.9, -28.5))
                 + box((43.6, 5.5, 1.82), (-1.1, sign * 51.6, -13.9)))
        part += flank

    # Concave-flared trapezius neck collar rooted inside chest roof (Z=29.5..39.5) around 6803-2RS.
    collar = loft([
        Plane.XY.offset(29.5) * Pos(6, 0) * Circle(21.5),
        Plane.XY.offset(34.2) * Pos(6, 0) * Circle(19.6),
        Plane.XY.offset(37.8) * Pos(6, 0) * Circle(17.5),
        Plane.XY.offset(39.5) * Pos(6, 0) * Circle(16.6),
    ], ruled=False)
    part += collar
    part -= Pos(6, 0, 34.3) * cylinder(26.15, 5.6)
    part -= Pos(6, 0, 18.0) * cylinder(27.0, 14.0)
    part -= Pos(6, 0, 18.0) * cylinder(23.5, 23.0)

    # Internal IMU shelf anchored directly into front chest prow at IMU_LOCATION = (50, 0, 0) rot 90.
    imu_shelf = box((18, 18, 1.82), (55, 0, -1.66))
    part += imu_shelf & inner_clip

    # Lightened power shelves for UBEC (+Y) and capacitor (-Y) fused cleanly into chest flanks.
    ubec_shelf = box((22, 40, 1.82), (-10, 46, 1.5)) - box((16, 32, 6), (-10, 42, 1.5))
    cap_shelf = box((10, 48, 1.82), (-8.4, -49, 3.3)) - box((6, 30, 6), (-8.4, -42, 3.3))
    part += (ubec_shelf + cap_shelf) & inner_clip

    # Carve exact clearances for all 3 trunk servos, hip sweeps (Y=±42), and electronics.
    for name in ('head_yaw', 'left_hip_roll', 'right_hip_roll'):
        part = recut_station(part, STATION_BY_JOINT[name])
    for sign in (-1, 1):
        # Under-pauldron wheel-well arches clearing hip_roll horn/idler/clevis across ±25.2°.
        part -= box((36, 46, 54), (39.0, sign * 47, -40))
        part -= box((38, 46, 54), (-46, sign * 47, -40))
        part -= box((100, 26, 54), (-4, sign * 68, -36))
        part -= box((92, 20, 28), (-4, sign * 65.2, -28))
    # Open central belly drawer bay strictly up to X=+44.5 so front ab apron stays intact.
    part -= box((110.5, 54.6, 30), (-10.75, 0, -39.2))
    part -= UBEC_LOCATION * box((50.8, 25.8, 15.0))
    part -= CAPACITOR_LOCATION * cylinder(19.2, 36.0, (0, 0, -0.5))
    part -= IMU_LOCATION * box((21.0, 17.0, 4.0), (0, 0, 1.6))
    part -= LOGIC_BUCK_LOCATION * box((41.2, 23.2, 11.6), (0, 0, 5.5))
    part -= SWITCH_LOCATION * box((19.8, 13.8, 23.0))
    part -= Pos(32, 24, -10) * box((16, 13, 6), (0, 0, 1.5))
    part -= box((26, 30, 46), (-54, 0, -10))
    part -= box((26, 38, 36), (-54, 0, 22))
    for sign in (-1, 1):
        part -= box((20, 38, 30), (-51, sign * 37, -10))
    # Open 10x10 mm vertical wire pass-through above left & right hip_roll servo cable exits.
    for sign in (-1, 1):
        part -= box((10.0, 10.0, 14.0), (-18.6, sign * 42.0, -3.0))
    part -= box((40, 160, 140), (-72.0, 0, 0))
    return checked('trunk_front', part)


@cache
def trunk_bottom():
    return checked('trunk_bottom', torso_lower())


@cache
def neck_link():
    """Concentric sculpted waisted dark-graphite neck coupling 6803 collar to head_pitch (Z=44)."""
    st_hp = STATION_BY_JOINT['head_pitch']
    part = horn_plate(3.0)
    for sx in (-1, 1):
        for sy in (-1, 1):
            part += cylinder(3.2, 5.2, (sx * 4.6, sy * 4.6, 2.0))
    part += ring(16.6, 9.2, 7.5, (0, 0, 6.1))
    # Concentric sculpted hourglass column (0.7 mm rotary reveal above white collar Z_neck=11.5).
    column = loft([
        Plane.XY.offset(12.2) * Circle(15.6),
        Plane.XY.offset(14.5) * Circle(16.2),
        Plane.XY.offset(22.5) * Circle(13.6),
        Plane.XY.offset(30.5) * Circle(14.8),
        Plane.XY.offset(35.0) * Circle(15.6),
    ], ruled=False)
    column -= cylinder(19.6, 14.0, (0, 0, 15.2))
    part += column
    # Tapered upper yoke hidden under chin bridging column to head_pitch servo mount & idler cap.
    yoke = loft([
        Plane.XY.offset(29.5) * Pos(0.0, -2.0) * RectangleRounded(16.0, 22.0, 7.5),
        Plane.XY.offset(34.5) * Pos(2.0, -6.5) * RectangleRounded(18.0, 29.0, 6.0),
        Plane.XY.offset(36.0) * Pos(2.0, -6.5) * RectangleRounded(18.0, 29.0, 6.0),
    ], ruled=False)
    part += yoke
    part += st_hp.servo_base * slim_servo_mount()
    part += st_hp.servo_base * slim_idler_cap()
    part = recut_station(part, st_hp)
    part -= cylinder(9.2, 18.0, (0, 0, -1.0))
    for x, y in ((7, 0), (-7, 0), (0, 7), (0, -7)):
        part -= cylinder(M3_CLEAR, 8.0, (x, y, -1.0))
        part -= cylinder(6.2, 3.1, (x, y, 3.0))
    part -= ring(36.0, 16.8, 6.2, (0, 0, 6.0))
    part -= box((24, 24, 70), (-42, 0, 35))
    return checked('neck_link', part)


@cache
def hip_clevis():
    """Compact dark-graphite PETG vertical drop-clevis suspending hip_pitch under hip_roll."""
    st_hr = STATION_BY_JOINT['left_hip_roll']
    st_hp = STATION_BY_JOINT['left_hip_pitch']
    ml_hr = st_hr.moving_location

    drive = horn_plate(3.0)
    rear = Pos(0, 0, IDLER_SEAT_TOP - 8.5) * cylinder(19.0, 8.5)
    part = ml_hr * (drive + rear)

    # Sleek chamfered dark-graphite PETG >= 2.65 mm hip_pitch pod centered at (-10, 29, -38).
    pod_planes = (
        (( 0.42,  0.0,   0.907), 22.8),
        ((-0.42,  0.0,   0.907), 22.8),
        (( 0.0,   0.45,  0.893), 21.4),
        (( 0.0,  -0.45,  0.893), 19.8),
        (( 0.65,  0.76,  0.0),   32.5),
        ((-0.65,  0.76,  0.0),   32.5),
    )
    pod_outer = faceted_solid((30.0, 21.8, 13.6), pod_planes, center=(-10.0, 29.0, -38.0))
    pod = faceted_shell((30.0, 21.8, 13.6), pod_planes, wall=2.65, center=(-10.0, 29.0, -38.0))
    # Open underside for servo insertion & cooling, open outer horn face & inboard idler clearance.
    pod -= box((46.0, 36.0, 12.0), (-10.0, 29.5, -52.5))
    pod -= box((44.0, 12.0, 22.0), (-10.0, 50.8, -38.0))
    pod -= Pos(0, 55.0, -38.0) * Rot(90, 0, 0) * cylinder(22.5, 9.0)
    pod -= box((64.0, 14.0, 32.0), (-10.0, 0.4, -38.0))
    part += pod

    part += (st_hp.servo_base * slim_servo_mount()) & (pod_outer + box((62, 10, 28), (-10, 32.6, -38)))
    part += st_hp.servo_base * slim_idler_cap()

    # Non-coplanar front drop-strap (X in [25.6, 28.8]) sweeping from hip_roll horn to pod roof.
    front_strap = extrude(_yz_hull_face(25.6, ((0.0, 0.0, 8.0), (15.5, -25.8, 7.2))), 3.2)
    front_tie = box((18.0, 17.0, 2.8), (19.6, 17.5, -25.8))
    part += front_strap + front_tie

    # Non-coplanar rear drop-arm (X in [-36.8, -29.6]) inside rear hub [-37.5, -29.0] into pod rear wall.
    rear_strap = extrude(_yz_hull_face(-36.8, ((0.0, 0.0, 7.6), (17.5, -34.0, 9.5))), 7.2)
    part += rear_strap

    part = recut_horn_and_bearing(part, ml_hr, double=True)
    part = recut_station(part, st_hp)
    part -= box((48, 32, 38), (0, 0, 0))
    part -= Pos(0, 5.0, -38.0) * Rot(90, 0, 0) * cylinder(24.0, 20.0)
    part -= box((24, 60, 40), (32.0, 29.0, -47.2))

    part += ml_hr * horn_hub_cap(wall=2.65, skirt_depth=2.0)
    part += ml_hr * idler_hub_cap(IDLER_SEAT_TOP - 8.5, tip_offset=8.2, wall=2.65)
    # Open-bottom 9.5 mm wire exit slot through rear pod & rear_strap for hip_pitch cable & 3-pin Dupont plug.
    part -= box((16.0, 9.5, 21.0), (-37.0, 11.25, -43.7))
    return checked('hip_clevis', part)


@cache
def thigh_shell():
    """Structural Microduck thigh chassis with open +Y drop-in knee bay, flush shroud & M3 insert bosses."""
    st_hp = STATION_BY_JOINT['left_hip_pitch']
    st_kp = STATION_BY_JOINT['left_knee_pitch']
    ml_hp = st_hp.moving_location

    part = ml_hp * horn_plate(3.0)
    part += ml_hp * Pos(0, 0, IDLER_SEAT_TOP - 8.5) * cylinder(16.6, 8.5)
    part += st_kp.servo_base * slim_servo_mount()
    part += st_kp.servo_base * slim_idler_cap()

    # Smooth flush rounded shroud completely enveloping slim_servo_mount (59.4 x 27.8 vs 58.0 x 26.2 -> zero ridge/dots).
    s = MG996R
    cx = s.body_center_x
    shroud_out = st_kp.servo_base * extrude(
        Plane.XY.offset(-7.6) * Pos(cx, 0.0) * RectangleRounded(59.4, 27.8, 5.5), 42.6)
    shroud_in_body = st_kp.servo_base * extrude(
        Plane.XY.offset(-9.0) * Pos(cx, 0.0) * RectangleRounded(51.8, 21.6, 3.0), 36.2)
    shroud_in_ears = st_kp.servo_base * extrude(
        Plane.XY.offset(26.8) * Pos(cx, 0.0) * RectangleRounded(55.4, 23.6, 1.8), 14.0)
    k_shroud = (shroud_out - shroud_in_body - shroud_in_ears).clean()
    k_shroud -= st_kp.servo_base * box((32.0, 10.0, 18.0), (cx, -14.0, 12.0))
    part += k_shroud

    # Lightweight flat Microduck inner rigidity plate (Y in [-27.2, -24.2]) with pill relief cutout.
    inner_face = _xz_hull_face(-27.2, ((0.0, 0.0, 10.5), (15.5, -34.0, 12.0), (0.0, -60.0, 12.0)))
    inner_plate = extrude(inner_face, -2.4)
    inner_plate -= Pos(3.0, -26.0, -26.0) * Rot(90, 0, 0) * extrude(RectangleRounded(11.5, 20.0, 4.5), 8.0, both=True)
    part += inner_plate

    # Lightened outer tie-web at Y in [29.4, 32.0] connecting hip horn_plate (0,0) to both M3 cover bosses.
    tie_face = _xz_hull_face(29.4, ((0.0, 0.0, 12.6), (-7.0, -37.5, 4.6), (17.5, -23.5, 4.6)))
    tie_web = extrude(tie_face, -2.6)
    tie_web -= Pos(2.5, 30.7, -21.0) * Rot(90, 0, 0) * extrude(RectangleRounded(8.0, 9.0, 3.0), 8.0, both=True)
    part += tie_web

    # Twin Ø8.4 mm PETG/PLA M3 insert boss pillars + slim structural cross-bridges into inner plate & nose.
    for bx, bz in THIGH_COVER_BOSSES:
        part += Pos(bx, 22.5, bz) * Rot(-90, 0, 0) * cylinder(8.4, 9.5)
    part += box((5.0, 50.0, 2.4), (-7.0, -1.0, -37.5))
    part += st_kp.servo_base * extrude(
        Plane.XY.offset(24.0) * Pos(36.0, -9.5) * RectangleRounded(7.5, 11.5, 2.5), 27.6)

    part = recut_horn_and_bearing(part, ml_hp, double=True)
    part = recut_station(part, st_kp)
    part -= box((80, 46.5, 31), (-8, 5.25, 0))
    # Open unobstructed +Y drop-in corridor for 54.0 x 19.7 mm MG996R body & ears from ear seat out to +Y.
    part -= st_kp.servo_base * extrude(
        Plane.XY.offset(26.8) * Pos(cx, 0.0) * RectangleRounded(55.4, 21.4, 1.2), 80.0)
    part -= Pos(0, 27.4, -60.0) * Rot(-90, 0, 0) * cylinder(28.0, 8.0)
    part -= box((60, 8.0, 36), (-8.0, 31.4, -76.0))
    part -= Pos(0, -33.0, -60.0) * Rot(-90, 0, 0) * cylinder(28.0, 5.5)
    part -= box((60, 8.0, 44), (-8.0, -31.5, -60.0))

    for bx, bz in THIGH_COVER_BOSSES:
        part -= Pos(bx, 25.6, bz) * Rot(-90, 0, 0) * cylinder(M3_INSERT, 6.8)

    # Wire exit portal & upper-rear turn relief at knee_pitch cable exit into inner thigh routing pocket.
    part -= st_kp.servo_base * box((15.0, 19.5, 8.5), (36.0, -5.25, 3.25))

    return checked('thigh', part)


@cache
def thigh_cover():
    """Detachable smooth one-piece Microduck outer haunch cover bolted via 2x M3 after servo assembly."""
    wall = 2.1
    haunch_planes = (
        (( 0.0,   0.999, -0.045), 40.6),
        (( 0.40,  0.916,  0.0),   47.5),
        ((-0.40,  0.916,  0.0),   44.5),
        (( 0.0,   0.85,   0.527), 41.5),
        (( 0.0,   0.85,  -0.527), 69.0),
    )
    outer_slab = extrude(_xz_hull_face(32.0, ((0.0, 0.0, 15.5), (22.5, -35.0, 14.5), (0.0, -60.0, 15.5))), -9.0)
    inner_cut  = extrude(_xz_hull_face(31.0, ((0.0, 0.0, 13.0), (19.5, -35.0, 11.8), (0.0, -60.0, 13.0))), -9.0)
    for n, d in haunch_planes:
        outer_slab -= plane_cut(n, d)
        inner_cut  -= plane_cut(n, d - wall)
    part = (outer_slab - inner_cut).clean()

    for bx, bz in THIGH_COVER_BOSSES:
        part += Pos(bx, 32.2, bz) * Rot(-90, 0, 0) * cylinder(9.9, 6.4)
    part = (part & outer_slab).clean()

    # Clear hip horn plate & 4x M3x8 hip screws, knee horn screws (preserving solid bottom lip), & inner tie-web.
    part -= Pos(0, 31.8, 0.0) * Rot(-90, 0, 0) * cylinder(27.2, 3.6)
    part -= Pos(0, 35.0, 0.0) * Rot(-90, 0, 0) * cylinder(23.0, 3.3)
    part -= Pos(0, 32.0, -60.0) * Rot(-90, 0, 0) * cylinder(23.6, 2.8)
    part -= Pos(0, 26.0, -30.0) * box((100, 12.4, 130))

    # Coaxial Ø5.4 mm Microduck hex-key pivot dimples at hip (0,0) & knee (0,-60).
    part -= Pos(0, 30.0,   0.0) * Rot(-90, 0, 0) * cylinder(5.4, 15.0)
    part -= Pos(0, 30.0, -60.0) * Rot(-90, 0, 0) * cylinder(5.4, 15.0)

    # Twin recessed M3 clearance bores + counterbores matching THIGH_COVER_BOSSES.
    for bx, bz in THIGH_COVER_BOSSES:
        part -= Pos(bx, 30.0, bz) * Rot(-90, 0, 0) * cylinder(M3_CLEAR, 14.0)
        part -= Pos(bx, 35.6, bz) * Rot(-90, 0, 0) * cylinder(6.2, 8.0)

    return checked('thigh_cover', part)


@cache
def thigh_idler_cap():
    """Detachable PLA friction-fit cap over Ø16.6 mm hip idler spigot (flush Ø21.0 mm, uniform 2.075 mm wall)."""
    return checked('thigh_idler_cap', _idler_nut_cap(-27.5, 8.425, 2.075, 11.90, bevel=2.0))


@cache
def shin_shell():
    """Microduck-inspired sleek rounded-pill carbon-graphite shin tucked cleanly inside thigh fairing."""
    st_kp = STATION_BY_JOINT['left_knee_pitch']
    ml_kp = st_kp.moving_location

    part = ml_kp * horn_plate(3.0)
    part += ml_kp * Pos(0, 0, IDLER_SEAT_TOP - 4.5) * cylinder(24.0, 4.5)

    # Flush rounded outer knee arm (Y in [28.0, 31.2]) & slim inner knee arm (Y in [-31.7, -28.0]).
    part += extrude(_xz_hull_face(28.0,  ((0.0, 0.0, 12.0), (0.5, -28.0, 11.0))), -3.2)
    part += extrude(_xz_hull_face(-31.7, ((0.0, 0.0, 12.0), (0.5, -28.0, 11.0))), -3.7)

    # Sculpted rounded knee yoke bridge clearing tilted knee_pitch servo (Z_knee >= -23.0).
    yoke = loft([
        Plane.XY.offset(-32.0) * Pos(1.0, 0.0) * RectangleRounded(21.0, 48.0, 9.5),
        Plane.XY.offset(-27.0) * Pos(0.5, 0.0) * RectangleRounded(20.0, 62.4, 9.0),
    ], ruled=False)
    part += yoke

    # Sleek tapered rounded-pill lower shin bone (hollow >= 2.65 mm PETG wall) with rear Achilles slot.
    strut_out = loft([
        Plane.XY.offset(-51.5) * Pos(2.0, 0.0) * RectangleRounded(25.0, 25.0, 10.0),
        Plane.XY.offset(-40.5) * Pos(1.5, 0.0) * RectangleRounded(24.0, 29.0, 10.5),
        Plane.XY.offset(-29.5) * Pos(1.0, 0.0) * RectangleRounded(22.0, 42.0, 10.0),
    ], ruled=False)
    strut_in = loft([
        Plane.XY.offset(-50.5) * Pos(2.0, 0.0) * RectangleRounded(18.5, 18.5, 7.0),
        Plane.XY.offset(-40.5) * Pos(1.5, 0.0) * RectangleRounded(17.5, 22.5, 7.5),
        Plane.XY.offset(-28.5) * Pos(1.0, 0.0) * RectangleRounded(15.5, 34.0, 7.0),
    ], ruled=False)
    strut = (strut_out - strut_in).clean()
    strut -= Pos(-10.0, 0.0, -40.5) * box((14.0, 14.0, 16.0))
    part += strut

    # Rounded 4-bolt ankle flange mating flush onto rounded duck boot (Z in [-54.0, -50.2]).
    part += extrude(Plane.XY.offset(-54.0) * Pos(2.0, 0.0) * RectangleRounded(25.0, 25.0, 7.5), 3.8)
    for x in (-7, 11):
        for y in (-8, 8):
            part -= cylinder(M3_CLEAR, 10, (x, y, -56))

    part = recut_horn_and_bearing(part, ml_kp, double=False)
    part -= box((56, 54, 48), (0, -1, 0.0))
    part += Pos(0, -31.7, 0) * Rot(90, 0, 0) * ring(17.2, 12.0, 2.8)
    return checked('shin', part)


@cache
def shin_idler_cap():
    """Detachable PETG friction-fit cap over Ø17.2 mm knee idler collar (Ø22.7 mm, uniform 2.625 mm wall)."""
    return checked('shin_idler_cap', _idler_nut_cap(-32.0, 8.725, 2.625, 7.95, bevel=2.0))


@cache
def integrated_foot():
    """Smooth rounded 36 mm narrow Microduck duck-boot foot (Z in [-5.0, +5.5], lightened underside)."""
    boot_out = loft([
        Plane.XY.offset(-5.0) * Pos(4.0, 0.0) * RectangleRounded(71.0, 36.0, 13.5),
        Plane.XY.offset(-0.5) * Pos(4.0, 0.0) * RectangleRounded(70.0, 35.0, 13.0),
        Plane.XY.offset( 3.2) * Pos(7.0, 0.0) * RectangleRounded(59.0, 29.0, 11.5),
    ], ruled=False)
    boot_out -= extrude(Plane.XY.offset(-5.5) * Pos(4.0, 0.0) * RectangleRounded(62.0, 27.0, 10.0), 3.5)
    boot_out -= extrude(Plane.XY.offset(-2.5) * Pos(18.0, 0.0) * RectangleRounded(30.0, 21.0, 8.0), 3.7)
    boot_out -= box((31.0, 26.8, 8.0), (-2.0, 0.0, 4.0))
    for sign in (-1, 1):
        cheek = loft([
            Plane.XY.offset(-1.0) * Pos(-2.0, sign * 15.0) * RectangleRounded(27.0, 4.6, 2.0),
            Plane.XY.offset( 5.5) * Pos(-2.0, sign * 14.8) * RectangleRounded(21.0, 3.8, 1.7),
        ], ruled=False)
        boot_out += cheek
    for x in (-11, 7):
        for y in (-8, 8):
            boot_out -= cylinder(M3_CLEAR, 14, (x, y, -7))
    for x in (-23, 29):
        for y in (-12, 12):
            boot_out -= cylinder(M3_CLEAR, 14, (x, y, -7))
    return checked('foot', boot_out)


@cache
def sole_tread():
    """Smooth rounded 5.0 mm TPU duck-boot sole (78 x 40.4 mm) at exact Z=0.000 ground datum."""
    part = loft([
        Plane.XY.offset(-10.0) * Pos(4.0, 0.0) * RectangleRounded(76.5, 39.2, 14.5),
        Plane.XY.offset( -8.5) * Pos(4.0, 0.0) * RectangleRounded(78.0, 40.4, 15.0),
        Plane.XY.offset( -5.0) * Pos(4.0, 0.0) * RectangleRounded(78.0, 40.4, 15.0),
    ], ruled=False)
    part -= extrude(Plane.XY.offset(-7.15) * Pos(4.0, 0.0) * RectangleRounded(57.0, 24.0, 9.5), 3.0)
    for x in (-23, 29):
        for y in (-12, 12):
            part += cylinder(10.8, 2.15, (x, y, -7.15))
    for x in (-12, 0, 12):
        part -= box((3.4, 32.0, 0.6), (x, 0, -9.7))
    for x in (-23, 29):
        for y in (-12, 12):
            part -= cylinder(5.8, 1.8, (x, y, -10.0))
            part -= cylinder(M3_CLEAR, 7.0, (x, y, -11.0))
    return checked('sole_tread', part)


@cache
def head_front():
    """Refined smoothly lofted aero visor helmet with streamlined teardrop sensor wave & pill bezel."""
    w = 2.0
    main_out = _loft_stations((
        (-22.0, 74.0,  8.0, 50.5, 11.5),
        ( -9.0, 86.0, -9.5, 53.5, 14.5),
        ( 14.0, 90.5, -9.5, 54.0, 17.5),
        ( 39.0, 91.5, -1.5, 54.0, 14.5),
        ( 51.0, 89.5, -0.5, 53.0, 12.5),
        ( 55.5, 83.5,  2.0, 50.5, 11.5),
    ))
    main_in = _loft_stations((
        (-20.0, 74.0 - 2 * w,  8.0 + w, 50.5 - w,  9.5),
        ( -9.0, 86.0 - 2 * w, -9.5 + w, 53.5 - w, 12.5),
        ( 14.0, 90.5 - 2 * w, -9.5 + w, 54.0 - w, 15.5),
        ( 39.0, 91.5 - 2 * w, -1.5 + w, 54.0 - w, 12.5),
        ( 51.0, 89.5 - 2 * w, -0.5 + w, 53.0 - w, 10.5),
        ( 53.5, 83.5 - 2 * w,  2.0 + w, 50.5 - w,  9.5),
    ))
    # Smooth streamlined teardrop dorsal wave enclosing Camera (-9.0) & ToF (+11.5) cleanly.
    wave_out = _loft_stations((
        (-10.0, 24.0, 44.0, 51.0,  3.2),
        (  8.0, 43.0, 45.0, 63.0, 10.5),
        ( 28.0, 56.5, 45.0, 75.5, 14.0),
        ( 46.0, 57.5, 45.0, 76.0, 14.0),
        ( 54.8, 52.0, 46.5, 73.0, 12.5),
    ))
    wave_in = _loft_stations((
        ( -6.0, 24.0 - 2 * w, 44.0, 51.0 - w,  2.0),
        (  8.0, 43.0 - 2 * w, 45.0, 63.0 - w,  8.5),
        ( 28.0, 56.5 - 2 * w, 45.0, 75.5 - w, 12.0),
        ( 46.0, 57.5 - 2 * w, 45.0, 76.0 - w, 12.0),
        ( 52.0, 52.0 - 2 * w, 46.5, 73.0 - w, 10.5),
    ))

    outer = (main_out + wave_out).clean()
    inner = (main_in + wave_in).clean()
    part = (outer - inner).clean()

    # Stepped silky-smooth pill visor viewport hiding visor_lens edges cleanly (exposes 100% of 55x27.5 OLED).
    part -= _rounded_stadium(62.0, 31.0, 9.2, 18.0, center=(61.85, 0.0, 25.0))
    part -= _rounded_stadium(70.8, 36.4, 10.8, 8.0, center=(48.85, 0.0, 25.0))
    # Subtle 0.7 mm recessed centered camera-island pill bezel framing Camera (-9.0) + ToF (+11.5).
    part -= _rounded_stadium(38.0, 16.4, 7.8, 10.0, center=(59.1, 0.0, 59.5))
    part -= Pos(42.0, -9.0, 59.5) * Rot(0, 90, 0) * cylinder(13.2, 24.0)
    part -= _rounded_stadium(8.4, 10.4, 2.2, 24.0, center=(54.0, 11.5, 59.5))
    # Rounded rear cover bay & lower neck entrance.
    part -= _rounded_stadium(58.5, 22.0, 6.5, 34.0, center=(-33.5, 0.0, 27.0))
    part -= box((58.0, 54.0, 36.0), (18.0, 0.0, -14.0))

    st_hp = STATION_BY_JOINT['head_pitch']
    ml_hp = st_hp.moving_location
    part += ml_hp * horn_plate(3.0)
    part += ml_hp * Pos(0, 0, IDLER_SEAT_TOP - 4.5) * bearing_plate(False)
    part += box((16, 2.4, 22), (6, 31.5, 11.0))
    part += box((16, 2.8, 22), (6, -27.2, 11.0))
    part += box((14, 16, 2.2), (6, 36, 21))
    part += box((14, 16, 2.2), (6, -34, 21))

    part = (part & outer).clean()
    part = recut_horn_and_bearing(part, ml_hp, double=False)
    # Clear neck_link & head_pitch servo across full -30° .. +45° pitch sweep.
    part -= box((64, 58, 38), (6, 0, -6))
    # Clearances for 2.42" OLED, forehead Camera & ToF, mic, ESP32 head MCU & rear cable bay.
    part -= OLED_LOCATION * box((44.6, 76.0, 6.4), (0, 0, 2.4))
    part -= CAMERA_LOCATION * box((26.0, 26.0, 14.2), (0, 0, 6.9))
    part -= TOF_LOCATION * box((14.5, 11.5, 8.0), (0, 0, 2.5))
    part -= MIC_LOCATION * box((16, 14, 6), (0, 0, 1.5))
    part -= Pos(-22 + HEAD_FORWARD, 0, 34) * box((30.6, 72.4, 12.0), (0, 0, 5.2))
    part -= Pos(12 + HEAD_FORWARD, 28, 28) * box((15, 12, 5), (0, 0, 1.5))
    # Rear opening cleanly clears head_rear_cover (Z<=38.0) while preserving smooth occipital cheeks.
    part -= _rounded_stadium(58.5, 22.0, 6.5, 36.0, center=(-34.5, 0.0, 27.0))
    part -= box((32, 58, 28), (-26, 0, 2))
    part -= box((18, 24, 14), (22 + HEAD_FORWARD, 0, -4))
    return checked('head_front', part)


@cache
def head_rear_cover():
    """Chamfered hexagonal rear skull cap flush below roof crown cleanly enclosing dual_esp32s3."""
    cap_planes = (
        ((0.0, 0.707, 0.707), 27.5),
        ((0.0, -0.707, 0.707), 27.5),
        ((0.0, 0.707, -0.707), 27.5),
        ((0.0, -0.707, -0.707), 27.5),
    )
    plate = faceted_solid((1.0, 29.0, 10.75), cap_planes, center=(-39.6, 0, 27.25))
    plate += box((10, 34, 2.0), (-33.8, 0, 28.2))
    # Rear cable pass-through notch cleanly clears 6.5 mm head_harness across -30°..+45° pitch.
    plate -= box((16, 18, 18), (-39.6, 5.0, 23.0))
    for y in (-22, 22):
        plate -= Pos(-42, y, 27.25) * Rot(0, 90, 0) * cylinder(M3_CLEAR, 8)
    return checked('head_rear_cover', plate.translate((HEAD_FORWARD, 0, 0)))


@cache
def visor_lens():
    """Smooth stadium-pill smoked-cyan polycarbonate shield recessed flush inside visor bezel."""
    return checked('visor_lens', _rounded_stadium(70.0, 35.6, 10.5, 2.1, center=(51.5, 0.0, 25.0)))
