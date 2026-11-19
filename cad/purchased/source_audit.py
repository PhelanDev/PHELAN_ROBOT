"""Inspect pinned references without silently substituting incompatible catalog parts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from build123d import import_step

from ..provenance import source_receipt
from .fetch_sources import COMMIT, MODELS


def audit_sources(root=None):
    root = Path(root or Path(__file__).with_name('sources'))
    manifest = json.loads((root / 'sources.json').read_text(encoding='utf-8'))
    if manifest['upstream_commit'] != COMMIT or set(manifest['models']) != set(MODELS):
        raise ValueError('Unexpected source catalog revision or model set')
    results = {}
    for name, data in manifest['models'].items():
        path = root / (name + '.step')
        if hashlib.sha256(path.read_bytes()).hexdigest() != data['sha256']:
            raise ValueError(f'Source STEP hash mismatch: {name}')
        shape = import_step(path)
        bounds = tuple(shape.bounding_box().size)
        expected = None
        if name == 'bearing_flanged_f623_simple':
            expected = (11.5, 11.5, 4.0)
        elif name.startswith('iso4762_socket_head_cap_screw_m3x'):
            expected = (5.5, 5.5, int(name.rsplit('x', 1)[1]) + 3.0)
        compatible = (bool(np.allclose(bounds, expected, rtol=0, atol=1e-5))
                      if expected is not None else None)
        results[name] = {
            'sha256_verified': True, 'valid_brep': bool(shape.is_valid),
            'solid_count': len(shape.solids()), 'bbox_mm_in_source_axes': bounds,
            'checked_envelope_mm': expected, 'envelope_compatible': compatible,
            'assembly_datum_verified': False, 'used_in_assembly': False,
        }
    return {'upstream_commit': COMMIT, 'models': results,
            'source_receipt': source_receipt(), 'all_references_approved': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/cad/source_reference_review.json'))
    args = parser.parse_args()
    report = audit_sources()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'inspected': len(report['models']), 'approved': False,
                      'incompatible_envelopes': [name for name, data in report['models'].items()
                                                 if data['envelope_compatible'] is False]}))
