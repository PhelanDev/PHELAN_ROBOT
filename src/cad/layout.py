"""Shared joint and hardware datums for the monocoque assembly (millimetres)."""

from dataclasses import dataclass

from build123d import Pos, Rot

from .kinematics import FRAMES
from .params import MG996R, THIGH_MM


@dataclass(frozen=True)
class Station:
    joint: str
    fixed: str
    moving: str
    origin: tuple
    rotation: tuple
    bearings: int

    @property
    def location(self):
        rx, ry, rz = self.rotation
        return Pos(*self.origin) * Rot(rx, 0, 0) * Rot(0, ry, 0) * Rot(0, 0, rz)

    @property
    def servo_base(self):
        return self.location * Pos(0, 0, -MG996R.horn_face)

    @property
    def moving_location(self):
        offset = FRAMES[self.moving][1]
        return Pos(*(-value for value in offset)) * self.location


STATIONS = (
    Station('head_yaw', 'trunk_base', 'neck', (6, 0, 28), (0, 0, 0), 0),
    Station('head_pitch', 'neck', 'head', (0, 30, 44), (-90, 0, 0), 1),
    *(Station(f'{side}_hip_roll', 'trunk_base', f'{side}_hip',
              (26, sign * 42, -40), (0, 90, 180), 2)
      for side, sign in (('left', 1), ('right', -1))),
    *(Station(f'{side}_hip_pitch', f'{side}_hip', f'{side}_thigh',
              (0, sign * 55, -38), (-90 * sign, 0, 180), 2)
      for side, sign in (('left', 1), ('right', -1))),
    *(Station(f'{side}_knee_pitch', f'{side}_thigh', f'{side}_shin',
              (0, sign * 28, -THIGH_MM), (-90 * sign, 0, -48 * sign), 1)
      for side, sign in (('left', 1), ('right', -1))),
)
STATION_BY_JOINT = {station.joint: station for station in STATIONS}
THIGH_COVER_BOSSES = ((-7.0, -37.5), (17.5, -23.5))

# Power and sensor modules sit in dedicated non-overlapping bays clear of all servos.
UBEC_LOCATION = Pos(-10, 30, 10)
CAPACITOR_LOCATION = Pos(-10, -30, 14) * Rot(0, 90, 0)
IMU_LOCATION = Pos(50, 0, 0) * Rot(0, 0, 90)
LOGIC_BUCK_LOCATION = Pos(-26, 0, -14)
SWITCH_LOCATION = Pos(18, -24, -12)
HEAD_FORWARD = 22.0
OLED_LOCATION = Pos(22.5 + HEAD_FORWARD, 0.0, 25.0) * Rot(0, 90, 0)
CAMERA_LOCATION = Pos(16.0 + HEAD_FORWARD, -9.0, 59.5) * Rot(0, 90, 0)
TOF_LOCATION = Pos(22.0 + HEAD_FORWARD, 11.5, 59.5) * Rot(0, 90, 0)
MIC_LOCATION = Pos(12.0 + HEAD_FORWARD, -24.0, 34.0)
TOF_Z = 59.5
