"""Integrate CAD volumes and tensors without rescaling to a target robot mass."""

import numpy as np
from build123d import CenterOf


DENSITY_KG_MM3 = {'PETG': 1.27e-6, 'PLA': 1.24e-6, 'BRONZE': 1.24e-6,
                  'CLEAR': 1.27e-6, 'TPU': 1.20e-6, 'EVA': 0.1e-6,
                  'WIRE': 2.0e-6}


def part_mass(part):
    if part.mass_kg is not None:
        mass = part.mass_kg
    else:
        try:
            mass = part.shape.volume * DENSITY_KG_MM3[part.material]
        except KeyError as exc:
            raise ValueError(f'Missing mass/density for {part.name}') from exc
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError(f'Catalog mass must be finite and positive: {part.name}')
    return float(mass)


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
