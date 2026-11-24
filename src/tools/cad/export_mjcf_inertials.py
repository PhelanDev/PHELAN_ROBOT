"""CLI tool exporting CAD link inertials with FDM infill scaling for MuJoCo & URDF."""

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cad.mass_properties import link_inertial_properties


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='rpi5', choices=('rpi5', 'pizero2w', 'dual_esp32s3'))
    parser.add_argument('--fill-ratio', type=float, default=0.60)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    links = link_inertial_properties(variant=args.variant, polymer_fill_ratio=args.fill_ratio)
    total_mass = round(sum(round(v['mass_kg'], 6) for v in links.values()), 6)
    payload = {
        'variant': args.variant,
        'polymer_fill_ratio': args.fill_ratio,
        'total_mass_kg': total_mass,
        'links': links,
    }
    text = json.dumps(payload, indent=2) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(f'total_mass_kg={total_mass:.6f}')
    for name, info in links.items():
        pos = ' '.join(f'{c:.6f}' for c in info['com_m'])
        diag = ' '.join(f'{d:.8f}' for d in info['diaginertia_kg_m2'])
        print(f'  {name:12s}: <inertial pos="{pos}" mass="{info["mass_kg"]:.6f}" diaginertia="{diag}" />')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
