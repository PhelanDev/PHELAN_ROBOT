"""Surface-normal wall probes: detect thin regions without claiming global certification."""

import argparse
from itertools import product
import json
from math import isfinite
from numbers import Integral
from pathlib import Path

from build123d import Axis

from .catalogue import printable_parts
from .provenance import source_receipt


WALL_THRESHOLDS = {'PETG': 2.5, 'TPU': 2.2, 'PLA': 1.8, 'BRONZE': 1.8, 'CLEAR': 1.8}
WALL_SAMPLE_BUDGET = 512


def required_wall(material):
    if material not in WALL_THRESHOLDS:
        raise ValueError(f'No wall requirement for {material}')
    return WALL_THRESHOLDS[material]


def sample_wall_thickness(shape, required_mm, *, max_samples=WALL_SAMPLE_BUDGET):
    if not isfinite(required_mm) or required_mm <= 0:
        raise ValueError('Required wall thickness must be finite and positive')
    if isinstance(max_samples, bool) or not isinstance(max_samples, Integral) or max_samples < 1:
        raise ValueError('max_samples must be a positive integer')
    if not shape.is_valid or len(shape.solids()) != 1 or shape.volume <= 0:
        raise ValueError('Wall sampling requires one valid positive-volume solid')
    faces = list(shape.faces())
    samples, missing, tested_faces = [], 0, set()
    epsilon = 1e-4
    attempted = 0
    for u, v in product((0.2, 0.5, 0.8), repeat=2):
        for index, face in enumerate(faces):
            if attempted >= max_samples:
                break
            attempted += 1
            point = face.position_at(u, v)
            if not face.is_inside(point):
                continue
            direction = -face.normal_at(point).normalized()
            if not shape.is_inside(point + direction * epsilon):
                direction = -direction
                if not shape.is_inside(point + direction * epsilon):
                    missing += 1
                    continue
            hits = shape.find_intersection_points(Axis(point, direction))
            distances = [(p - point).dot(direction) for p, _ in hits]
            positive = [d for d in distances if d > epsilon and isfinite(d)]
            if not positive:
                missing += 1
                continue
            samples.append(min(positive))
            tested_faces.add(index)
        if attempted >= max_samples:
            break
    minimum = min(samples) if samples else None
    return {'method': 'B-rep face UV samples and inward normal line intersections',
            'required_mm': required_mm, 'sampled_min_mm': minimum,
            'samples_tested': len(samples), 'samples_attempted': attempted,
            'faces_sampled': len(tested_faces), 'face_count': len(faces),
            'undetermined_samples': missing,
            'observed_below_requirement': minimum is not None and minimum < required_mm - 1e-5,
            'certified_global_minimum': False}


def inspect_printable_walls(max_samples=WALL_SAMPLE_BUDGET):
    rows = {}
    for name, (shape, material, design_wall, _) in printable_parts().items():
        if material == 'EVA':
            continue
        required = required_wall(material)
        rows[name] = sample_wall_thickness(shape, required, max_samples=max_samples)
        rows[name].update(material=material, declared_design_wall_mm=design_wall)
    return {'source_receipt': source_receipt(), 'parts': rows,
            'release_ready': False, 'global_wall_thickness_verified': False,
            'note': 'A sampled minimum is an upper bound on the global minimum. '
                    'Thin samples require review; feature-specific exemptions are not assumed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-samples', type=int, default=WALL_SAMPLE_BUDGET)
    parser.add_argument('--output', type=Path, default=Path('artifacts/cad/wall_thickness_sampled.json'))
    args = parser.parse_args()
    report = inspect_printable_walls(args.max_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'parts': len(report['parts']), 'global_wall_thickness_verified': False,
                      'observed_thin_parts': [name for name, row in report['parts'].items()
                                               if row['observed_below_requirement']]}))
