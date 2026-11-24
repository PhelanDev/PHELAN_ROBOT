"""Integrate CAD volumes and tensors without rescaling to a target robot mass."""

import numpy as np
from build123d import CenterOf


DENSITY_KG_MM3 = {'PETG': 1.27e-6, 'PLA': 1.24e-6, 'BRONZE': 1.24e-6,
                  'CLEAR': 1.27e-6, 'TPU': 1.20e-6, 'EVA': 0.1e-6,
                  'WIRE': 2.0e-6}
FDM_RIGID_MATERIALS = ('PLA', 'PETG', 'BRONZE', 'CLEAR')
HARNESS_LINK_MAP = {
    'head_harness': 'trunk_base',
    'main_power': 'trunk_base',
    'leg_harness_left': 'left_hip',
    'hip_pitch_wire_left': 'left_hip',
    'knee_pitch_wire_left': 'left_thigh',
    'leg_harness_right': 'right_hip',
    'hip_pitch_wire_right': 'right_hip',
    'knee_pitch_wire_right': 'right_thigh',
}


def part_mass(part, *, polymer_fill_ratio=1.0):
    ratio = float(polymer_fill_ratio)
    if not np.isfinite(ratio) or ratio <= 0 or ratio > 1.0:
        raise ValueError(f'polymer_fill_ratio must be in (0, 1], got {polymer_fill_ratio}')
    if part.mass_kg is not None:
        mass = part.mass_kg
    else:
        try:
            mass = part.shape.volume * DENSITY_KG_MM3[part.material]
        except KeyError as exc:
            raise ValueError(f'Missing mass/density for {part.name}') from exc
    if part.category == 'printed' and part.material in FDM_RIGID_MATERIALS:
        mass *= ratio
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError(f'Catalog mass must be finite and positive: {part.name}')
    return float(mass)


def _integrate_parts_local(parts, frame_matrix, *, polymer_fill_ratio=1.0):
    masses = np.array([part_mass(p, polymer_fill_ratio=polymer_fill_ratio) for p in parts], dtype=float)
    centers = np.array([tuple(p.shape.center(CenterOf.MASS)) for p in parts], dtype=float)
    total = float(masses.sum())
    center_w = np.average(centers, weights=masses, axis=0)
    inertia_w = np.zeros((3, 3), dtype=float)
    for part, mass, com in zip(parts, masses, centers):
        tensor = np.array(part.shape.matrix_of_inertia, dtype=float) * (mass / part.shape.volume)
        delta = com - center_w
        inertia_w += tensor + mass * (np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
    rot, origin = frame_matrix[:3, :3], frame_matrix[:3, 3]
    com_local_m = rot.T @ ((center_w - origin) * 1e-3)
    inertia_local = rot.T @ (inertia_w * 1e-6) @ rot
    diag = [max(float(inertia_local[i, i]), 1e-7) for i in range(3)]
    return {'mass_kg': total, 'com_m': com_local_m.tolist(), 'diaginertia_kg_m2': diag}


def link_inertial_properties(variant='rpi5', *, polymer_fill_ratio=0.60):
    from .assembly import build_assembly
    from .kinematics import FRAMES, frame_transforms

    transforms = frame_transforms(None)
    buckets = {link: [] for link in FRAMES}
    for part in build_assembly(variant, pose=None, wiring=True):
        link = HARNESS_LINK_MAP.get(part.name, 'trunk_base') if part.link == 'harness' else part.link
        buckets[link if link in buckets else 'trunk_base'].append(part)

    links = {link: _integrate_parts_local(items, transforms[link], polymer_fill_ratio=polymer_fill_ratio)
             for link, items in buckets.items()}
    for stem in ('hip', 'thigh', 'shin', 'foot'):
        left, right = links[f'left_{stem}'], links[f'right_{stem}']
        m = 0.5 * (left['mass_kg'] + right['mass_kg'])
        cx = 0.5 * (left['com_m'][0] + right['com_m'][0])
        cy = 0.5 * (abs(left['com_m'][1]) + abs(right['com_m'][1]))
        cz = 0.5 * (left['com_m'][2] + right['com_m'][2])
        diag = [0.5 * (left['diaginertia_kg_m2'][i] + right['diaginertia_kg_m2'][i]) for i in range(3)]
        left.update(mass_kg=m, com_m=[cx, cy, cz], diaginertia_kg_m2=list(diag))
        right.update(mass_kg=m, com_m=[cx, -cy, cz], diaginertia_kg_m2=list(diag))
    return links


def mass_properties(parts):
    parts = tuple(parts)
    if not parts:
        raise ValueError('Cannot integrate an empty assembly')
    if any(not np.isfinite(p.shape.volume) or p.shape.volume <= 0 for p in parts):
        raise ValueError('All mass-bearing shapes must have finite positive volume')
    masses = np.array([part_mass(p) for p in parts])
    centers = np.array([tuple(p.shape.center(CenterOf.MASS)) for p in parts])
    total = float(masses.sum())
    if total <= 0:
        raise ValueError('Assembly mass must be positive')
    center = np.average(centers, weights=masses, axis=0)
    inertia = np.zeros((3, 3))
    for part, mass, com in zip(parts, masses, centers):
        tensor = np.array(part.shape.matrix_of_inertia) * (mass / part.shape.volume)
        delta = com - center
        inertia += tensor + mass * (np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
    return {'mass_kg': total, 'com_m': (center * 1e-3).tolist(),
            'inertia_kg_m2': (inertia * 1e-6).tolist(),
            'method': 'B-rep volume integration; nominal catalog masses; solid printed polymer',
            'measured_hardware': False, 'infill_compensated': False}
