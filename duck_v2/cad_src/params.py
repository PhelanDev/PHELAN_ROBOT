"""Locked mechanical interfaces from the CyberDuck V2 MG996R plan."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ServoSpec:
    body: tuple = (40.7, 19.7, 37.0)
    body_x_min: float = -10.35
    pocket: tuple = (41.5, 20.6, 39.2)
    ear_length: float = 54.0
    ear_pocket_length: float = 55.0
    ear_z: float = 27.0
    ear_thickness: float = 2.3
    ear_x: tuple = (-14.25, 34.35)
    ear_y: tuple = (-5.0, 5.0)
    ear_hole_diameter: float = 4.3
    ear_slot_travel: float = 0.7
    underside_clearance: float = 28.5
    shaft_top: float = 42.9
    shaft_diameter: float = 5.8
    collar_diameter: float = 12.0
    horn_face: float = 47.6
    wire_exit: tuple = (8.0, 6.0)
    wire_radius: float = 3.0
    mass_kg: float = 0.055

    @property
    def body_center_x(self):
        return self.body_x_min + self.body[0] / 2

    @property
    def ear_holes(self):
        return tuple((x, y) for x in self.ear_x for y in self.ear_y)


MG996R = ServoSpec()
HORN_OD, HORN_HUB, HORN_PCD = 20.0, 9.0, 14.0
HORN_DISK, HORN_HUB_HEIGHT = 2.0, 2.5
M3_CLEAR, M3_HEAD, M3_HEAD_DEPTH = 3.4, 5.8, 3.2
M3_INSERT, M3_INSERT_DEPTH, M3_WALL = 4.05, 6.6, 1.8
F623ZZ = (3.0, 10.0, 4.0, 11.5, 1.0)
F623_SEAT, F623_FLANGE_SEAT = 10.15, 11.8
NECK_BEARING = (17.0, 26.0, 5.0)
NECK_SEAT, NECK_SHOULDER = 26.15, 23.5
BATTERY_POCKET = (82.0, 48.0, 18.0)
BATTERY_ENVELOPE = (75.0, 48.0, 16.0)
FRAME_WALL, ARMOR_WALL = 2.8, 2.2
SLIDE_XY, SLIDE_Z = 0.25, 0.30
THIGH_MM, SHIN_MM = 68.0, 59.0
# Place the agreed 5 mm TPU sole datum on the ground in the zero-pose CAD model.
BASE_HEIGHT_MM = 177.0
FOOT_SIZE = (82.0, 46.0, 12.0)
VARIANTS = ('rpi5', 'pizero2w', 'dual_esp32s3')
VARIANT_SUFFIX = dict(zip(VARIANTS, ('rpi5', 'pizero2w', 'esp32s3')))
TOF_ANGLES = (-30, -15, 0, 15, 30)
