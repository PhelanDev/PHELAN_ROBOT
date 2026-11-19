"""Export and inspect catalog fit references without touching simulation assets."""

import argparse
import hashlib
import json
from pathlib import Path

from build123d import export_step

from .purchased.catalogue import purchased_parts


DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'purchased' / 'step'


def export_references(output=DEFAULT_OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    references = {}
    for name, (shape, mass) in purchased_parts().items():
        if not shape.is_valid or shape.volume <= 0:
            raise ValueError(f'Invalid catalog reference: {name}')
        path = output / f'{name}.step'
        if not export_step(shape, path):
            raise RuntimeError(f'STEP export failed: {path}')
        bounds = shape.bounding_box()
        references[name] = {
            'path': path.name, 'units': 'mm', 'bbox_mm': list(bounds.size),
            'volume_mm3': shape.volume, 'nominal_mass_kg': mass,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'geometry_kind': 'parametric_fit_reference',
            'vendor_certified': False,
        }
    manifest = output / 'references.json'
    manifest.write_text(json.dumps(references, indent=2) + '\n', encoding='utf-8')
    return references


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps({'references': len(export_references(args.output))}))
