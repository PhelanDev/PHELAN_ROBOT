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
    diameter = 4.5 if variant == 'dual_esp32s3' else 6.0
    routes = [
        # Rear S-loop through the open neck cable channel (all neck X <= -30).
        Harness('head_harness', (
            ('trunk_base', (-55, 0, 14)),
            ('trunk_base', (-53, 0, 28)),
            ('trunk_base', (-50, 0, 43)),
            ('neck', (-46, 0, 22)),
            ('neck', (-46, 0, 40)),
            ('neck', (-44, 0, 58)),
            ('head', (-52 + HEAD_FORWARD, 4, 24)),
            ('head', (-34 + HEAD_FORWARD, 4, 24)),
        ), diameter, 240.0),
        # Vertical power trunk from XT30 exit up behind the logic buck.
        Harness('main_power', (
            ('trunk_base', (-53, 0, -30.45)),
            ('trunk_base', (-53, 0, -24)),
            ('trunk_base', (-53, 0, -19)),
            ('trunk_base', (-53, 0, -14)),
            ('trunk_base', (-51, 0, -8)),
        ), 5.0, start_tangent=(0, 0, 1)),
    ]
    for side, sign in (('left', 1), ('right', -1)):
        # Service loop running strictly behind hip_roll idler (X <= -48) and above Z = -10.
        routes.append(Harness(f'leg_harness_{side}', (
            (f'{side}_thigh', (-20, -sign * 4, -56)),
            (f'{side}_thigh', (-28, -sign * 4, -30)),
            (f'{side}_thigh', (-34, -sign * 4, -10)),
            (f'{side}_hip', (-49, sign * 24, 24)),
            ('trunk_base', (-55, sign * 28, -10)),
            ('trunk_base', (-56, sign * 20, 4)),
        ), 4.5, 260.0))
    return routes


def measure_harnesses(variant, pose=None):
    transforms = frame_transforms(pose)
    measurements = []
    for route in harnesses(variant):
        length = route.path(transforms).length + route.tail_allowance
        measurements.append({'name': route.name, 'length_mm': length,
                             'tail_allowance_mm': route.tail_allowance,
                             'maximum_mm': route.maximum_length,
                             'passed': route.maximum_length is None or length <= route.maximum_length})
    return measurements
