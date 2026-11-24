"""Pose-aware service loops with explicit anchors in adjacent link frames."""

from dataclasses import dataclass

import numpy as np
from build123d import Circle, Plane, Spline, sweep

from ..kinematics import frame_transforms
from ..layout import HEAD_FORWARD


@dataclass(frozen=True)
class Harness:
    name: str
    anchors: tuple
    diameter: float
    maximum_length: float | None = None
    tail_allowance: float = 20.0
    start_tangent: tuple | None = None
    installed_length_mm: float | None = None

    def path(self, transforms):
        points = []
        for link, xyz in self.anchors:
            p = transforms[link] @ np.array((*xyz, 1))
            points.append(tuple(p[:3]))
        if self.start_tangent is not None:
            start = transforms[self.anchors[0][0]][:3, :3] @ np.asarray(self.start_tangent)
            end = np.asarray(points[-1]) - np.asarray(points[-2])
            return Spline(*points, tangents=(tuple(start), tuple(end)))
        return Spline(*points)

    def solid(self, transforms):
        path = self.path(transforms)
        plane = Plane(origin=path @ 0, z_dir=path % 0)
        return sweep(plane * Circle(self.diameter / 2), path)


def harnesses(variant):
    diameter = 4.5 if variant == 'dual_esp32s3' else 6.5
    routes = [
        # Rear S-loop through the open neck cable channel (all neck X <= -30).
        Harness('head_harness', (
            ('trunk_base', (-55, 0, 14)),
            ('trunk_base', (-51, 0, 40)),
            ('neck', (-44, 0, 20)),
            ('neck', (-46, 0, 38)),
            ('neck', (-46.5, 0, 50)),
            ('head', (-52 + HEAD_FORWARD, 4, 24)),
            ('head', (-39 + HEAD_FORWARD, 4, 24)),
        ), diameter, 240.0, installed_length_mm=240.0),
        # Vertical power trunk from XT30 exit up behind the logic buck.
        Harness('main_power', (
            ('trunk_base', (-53, 0, -30.45)),
            ('trunk_base', (-53, 0, -24)),
            ('trunk_base', (-53, 0, -19)),
            ('trunk_base', (-53, 0, -14)),
            ('trunk_base', (-51, 0, -8)),
        ), 5.0, start_tangent=(0, 0, 1), installed_length_mm=65.0),
    ]
    for side, sign in (('left', 1), ('right', -1)):
        # Knee servo wire exits knee_pitch portal, curves hidden inside thigh under cross-bridge to rear exit.
        routes.append(Harness(f'knee_pitch_wire_{side}', (
            (f'{side}_thigh', (21.8, -sign * 16.6, -35.8)),
            (f'{side}_thigh', (18.0, -sign * 15.8, -27.5)),
            (f'{side}_thigh', ( 1.0, -sign * 13.5, -34.5)),
            (f'{side}_thigh', (-3.0, -sign * 12.2, -43.5)),
            (f'{side}_thigh', (-13.0, -sign * 10.2, -45.5)),
            (f'{side}_thigh', (-22.0, -sign *  8.5, -45.5)),
        ), 2.6, 95.0, installed_length_mm=95.0))
        # Hip pitch servo wire exits straight back through open-bottom rear slot in hip_clevis to Y-junction.
        routes.append(Harness(f'hip_pitch_wire_{side}', (
            (f'{side}_hip', (-31.6, sign * 10.4, -38.0)),
            (f'{side}_hip', (-41.5, sign * 11.2, -37.5)),
            (f'{side}_hip', (-47.0, sign * 14.2, -33.5)),
            (f'{side}_hip', (-49.0, sign * 15.8, -22.0)),
        ), 2.6, 80.0, installed_length_mm=80.0))
        # Braided leg service loop running behind hip_clevis pod & trunk rear (X <= -54) into backpack.
        routes.append(Harness(f'leg_harness_{side}', (
            (f'{side}_thigh', (-25.5, -sign * 7.5, -45.5)),
            (f'{side}_thigh', (-46.0, -sign * 4.0, -30.0)),
            (f'{side}_hip',   (-54.0,  sign * 24.0, -42.0)),
            (f'{side}_hip',   (-55.0,  sign * 28.0,  10.0)),
            ('trunk_base',    (-57.0,  sign * 34.0,  -8.0)),
            ('trunk_base',    (-57.0,  sign * 20.0,   4.0)),
        ), 4.5, 265.0, installed_length_mm=265.0))
    return routes


def measure_harnesses(variant, pose=None):
    transforms = frame_transforms(pose)
    measurements = []
    for route in harnesses(variant):
        length = route.path(transforms).length + route.tail_allowance
        installed = route.installed_length_mm if route.installed_length_mm is not None else route.maximum_length
        passed = (
            (route.maximum_length is None or length <= route.maximum_length)
            and (installed is None or length <= installed)
        )
        measurements.append({'name': route.name, 'length_mm': length,
                             'tail_allowance_mm': route.tail_allowance,
                             'installed_mm': installed,
                             'maximum_mm': route.maximum_length,
                             'passed': passed})
    return measurements
