"""Fail-closed checks for the staged CAD-to-manufacturing pipeline."""

import argparse
import json
from math import isfinite
from pathlib import Path

import numpy as np

from robot.robot_spec import EXPECTED_JOINTS
from .assembly import build_assembly
from .catalogue import printable_parts
from .collision import interference
from .mass_properties import mass_properties
from .params import VARIANTS
from .provenance import source_receipt
from .release import assess_release
from .wiring.routes import measure_harnesses


JOINT_SWEEP_SAMPLES = 9
MAX_MASS_KG = 1.50


def within_mass_budget(mass_kg):
    return isfinite(mass_kg) and 0 < mass_kg <= MAX_MASS_KG


def sample_poses(samples=JOINT_SWEEP_SAMPLES):
    if samples < 2:
        raise ValueError('A sweep requires at least the two joint endpoints')
    yield 'straight', {}
    from sim.env.env_config import DEFAULT_NOMINAL_QPOS, LIE_NOMINAL_QPOS, SIT_NOMINAL_QPOS
    for name, values in (('stand', DEFAULT_NOMINAL_QPOS), ('sit', SIT_NOMINAL_QPOS),
                         ('lie', LIE_NOMINAL_QPOS)):
        yield name, dict(zip(EXPECTED_JOINTS, map(float, values)))
    for name, spec in EXPECTED_JOINTS.items():
        for value in np.linspace(spec['lower'], spec['upper'], samples):
            yield f'{name}_{value:+.6f}', {name: float(value)}
    yield 'both_hips_inward', {'left_hip_roll': -0.22, 'right_hip_roll': 0.22}


def validate_parts():
    rows = []
    for name, (shape, material, design_wall, variant) in printable_parts().items():
        rows.append({'name': name, 'material': material, 'variant': variant,
                     'valid_brep': shape.is_valid, 'solid_count': len(shape.solids()),
                     'volume_mm3': shape.volume, 'design_wall_mm': design_wall,
                     'passed': bool(shape.is_valid and len(shape.solids()) == 1 and shape.volume > 0)})
    return rows


def validate_assembly(variant, *, samples=JOINT_SWEEP_SAMPLES, sweep=True, limit=None):
    reports = []
    poses = list(sample_poses(samples)) if sweep else [('straight', {})]
    for name, pose in poses:
        assembly = build_assembly(variant, pose)
        report = interference(assembly, limit=limit)
        wires = measure_harnesses(variant, pose)
        mass = mass_properties(assembly)
        mass['budget_max_kg'] = MAX_MASS_KG
        mass['within_plan_budget'] = within_mass_budget(mass['mass_kg'])
        reports.append({'pose': name, 'joint_angles_rad': pose, 'geometry': report,
                        'wires': wires, 'mass': mass})
        if not report['passed']:
            # Phase order is a release gate: don't claim a sweep if nominal fit failed.
            break
    return {'variant': variant, 'sweep_requested': sweep,
            'sweep_completed': bool(sweep and len(reports) == len(poses)),
            'passed': all(r['geometry']['passed'] and r['mass']['within_plan_budget']
                          and all(w['passed'] for w in r['wires']) for r in reports),
            'poses': reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=(*VARIANTS, 'all'), default='all')
    parser.add_argument('--nominal-only', action='store_true')
    parser.add_argument('--samples', type=int, default=JOINT_SWEEP_SAMPLES)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path, default=Path('artifacts/cad/verification.json'))
    args = parser.parse_args()
    variants = VARIANTS if args.variant == 'all' else (args.variant,)
    from .dfam import inspect_printable_walls

    parts = validate_parts()
    dfam_report = inspect_printable_walls()
    result = {'units': 'mm', 'source_receipt': source_receipt(), 'parts': parts, 'dfam': dfam_report,
              'assemblies': [validate_assembly(v, samples=args.samples,
                              sweep=not args.nominal_only, limit=args.limit) for v in variants]}
    result['passed'] = all(p['passed'] for p in parts) and all(a['passed'] for a in result['assemblies'])
    result.update(assess_release(parts, result['assemblies'], dfam_report=dfam_report))
    result['continuous_sweep_certified'] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'release_ready': result['release_ready'],
                      'report': str(args.output)}))
    return 0 if result['release_ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
