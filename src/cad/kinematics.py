"""One millimetre-based frame tree for CAD, sweeps and derived assets."""

from math import cos, isfinite, sin

import numpy as np

from .params import BASE_HEIGHT_MM, SHIN_MM, THIGH_MM


FRAMES = {
    'trunk_base': (None, (0, 0, BASE_HEIGHT_MM), None, None),
    'neck': ('trunk_base', (6, 0, 28), 'head_yaw', 'z'),
    'head': ('neck', (0, 0, 44), 'head_pitch', 'y'),
    'left_hip': ('trunk_base', (0, 42, -40), 'left_hip_roll', 'x'),
    'left_thigh': ('left_hip', (0, 23, -38), 'left_hip_pitch', 'y'),
    'left_shin': ('left_thigh', (0, 0, -THIGH_MM), 'left_knee_pitch', 'y'),
    'left_foot': ('left_shin', (4, 0, -SHIN_MM), None, None),
    'right_hip': ('trunk_base', (0, -42, -40), 'right_hip_roll', 'x'),
    'right_thigh': ('right_hip', (0, -23, -38), 'right_hip_pitch', 'y'),
    'right_shin': ('right_thigh', (0, 0, -THIGH_MM), 'right_knee_pitch', 'y'),
    'right_foot': ('right_shin', (4, 0, -SHIN_MM), None, None),
}


def frame_transforms(pose=None):
    pose = {} if pose is None else pose
    joints = {entry[2] for entry in FRAMES.values() if entry[2]}
    if not hasattr(pose, 'items') or set(pose) - joints:
        raise ValueError('Pose must map known joint names to finite angles')
    try:
        pose = {name: float(value) for name, value in pose.items()}
    except (ValueError, TypeError) as exc:
        raise ValueError('Pose angles must be numeric') from exc
    if not all(isfinite(value) for value in pose.values()):
        raise ValueError('Pose angles must be finite')
    result = {}
    for link, (parent, offset, joint, axis) in FRAMES.items():
        transform = np.eye(4)
        transform[:3, 3] = offset
        if joint:
            index = 'xyz'.index(axis)
            a, b = (index + 1) % 3, (index + 2) % 3
            angle = float(pose.get(joint, 0))
            transform[a, a] = transform[b, b] = cos(angle)
            transform[b, a], transform[a, b] = sin(angle), -sin(angle)
        result[link] = transform if parent is None else result[parent] @ transform
    return result


def location_from_matrix(matrix):
    from build123d import Location, Plane
    return Location(Plane(origin=matrix[:3, 3].tolist(),
                          x_dir=matrix[:3, 0].tolist(), z_dir=matrix[:3, 2].tolist()))
