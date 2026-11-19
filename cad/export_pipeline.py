"""Export reviewable CAD drafts. Production output requires a passing release gate."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from build123d import Compound, export_step, export_stl

from .assembly import build_assembly
from .artifacts import artifact_inventory
from .catalogue import printable_parts
from .mass_properties import mass_properties
from .params import VARIANTS
from .print_plates import pack_plates
from .provenance import source_receipt


FOLDERS = {'PETG': '0_Shared_PETG_Khung', 'PLA': '1_Shared_PLA_VoGiap',
           'BRONZE': '1_Shared_PLA_VoGiap', 'CLEAR': '1_Shared_PLA_VoGiap',
           'TPU': '2_Shared_TPU_DeChan'}
VARIANT_FOLDERS = dict(zip(VARIANTS, ('3_Variant_Pi5', '4_Variant_PiZero2W',
                                    '5_Variant_DualESP32S3')))


def oriented_mesh(mesh):
    """Put the shortest AABB dimension up; record this as a heuristic, not DfAM proof."""
    mesh = mesh.copy()
    axis = int(np.argmin(mesh.extents))
    if axis != 2:
        rotation_axis = (0, 1, 0) if axis == 0 else (1, 0, 0)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, rotation_axis))
    mesh.apply_translation(-mesh.bounds[0])
    return mesh


def clean_tessellation(mesh, name, expected_volume=None):
    """Drop zero-area STL facets only; never close holes or join components."""
    before_volume = float(mesh.volume)
    original_faces = len(mesh.faces)
    mesh.update_faces(mesh.nondegenerate_faces())
    removed_faces = original_faces - len(mesh.faces)
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight or not mesh.is_volume:
        raise ValueError(f'Non-watertight STL after degenerate-facet cleanup: {name}')
    if abs(float(mesh.volume) - before_volume) > max(1e-8, abs(before_volume) * 1e-10):
        raise ValueError(f'Degenerate-facet cleanup changed STL volume: {name}')
    relative_volume_error = (None if expected_volume is None else
                             abs(float(mesh.volume) - expected_volume) / expected_volume)
    return mesh, removed_faces, relative_volume_error


def export_drafts(output, variants=VARIANTS):
    output = Path(output)
    step, stl = output / 'step', output / 'print_oriented'
    step.mkdir(parents=True, exist_ok=True)
    stl.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps({
        'status': 'DRAFT_EXPORT_INCOMPLETE', 'release_ready': False,
    }) + '\n', encoding='utf-8')
    receipt = source_receipt()
    catalogue, meshes, part_rows, generated = printable_parts(), {}, [], []
    for name, (shape, material, wall, variant) in catalogue.items():
        if material == 'EVA':
            continue
        if not shape.is_valid or len(shape.solids()) != 1:
            raise ValueError(f'Part must be one valid connected solid: {name}')
        export_step(shape, step / f'{name}.step')
        generated.append(step / f'{name}.step')
        folder = FOLDERS[material] if variant == 'shared' else VARIANT_FOLDERS[variant]
        target = stl / folder / f'{name}.stl'
        target.parent.mkdir(parents=True, exist_ok=True)
        export_stl(shape, target, tolerance=0.08, angular_tolerance=0.15)
        original = trimesh.load_mesh(target)
        original, removed_facets, volume_relative_error = clean_tessellation(
            original, name, shape.volume)
        mesh = oriented_mesh(original)
        mesh.export(target)
        mesh = trimesh.load_mesh(target)
        generated.append(target)
        if not mesh.is_watertight or not mesh.is_volume:
            raise ValueError(f'Non-watertight or non-positive STL: {name}')
        meshes[name] = (mesh, material)
        mesh_components = len(mesh.split(only_watertight=False))
        part_rows.append({'name': name, 'stl': target.relative_to(output).as_posix(),
                          'units': 'mm', 'watertight': bool(mesh.is_watertight),
                          'volume_mm3': mesh.volume, 'degenerate_facets_removed': removed_facets,
                          'stl_brep_volume_relative_error': volume_relative_error,
                          'closed_mesh_component_count': mesh_components,
                          'single_component_mesh': mesh_components == 1,
                          'minimum_wall_verified': False,
                          'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    results = {'status': 'DRAFT_NOT_FOR_FABRICATION', 'source_receipt': receipt,
               'parts': part_rows, 'variants': {}, 'release_ready': False,
               'mesh_note': 'Zero-area facets only are removed. Closed mesh components are counted, never fused; multiple shells remain a review finding. STL/B-rep volume difference is recorded as a diagnostic, not used to relax a plan tolerance or claim fabrication readiness.'}
    for variant in variants:
        parts = build_assembly(variant)
        for part in parts:
            part.shape.label = part.name
        export_step(Compound(children=[p.shape for p in parts]), step / f'assembly_{variant}.step')
        generated.append(step / f'assembly_{variant}.step')
        quantities = Counter(p.print_key for p in parts if p.category == 'printed')
        items = [(f'{name}_{i+1}', meshes[name][0], meshes[name][1])
                 for name, count in quantities.items() for i in range(count)]
        plates = pack_plates(items, output / 'plates_3mf' / variant)
        generated.extend(output / 'plates_3mf' / variant / row['file'] for row in plates)
        results['variants'][variant] = {
            'print_quantities': quantities, 'plates': plates,
            'purchased_quantities': Counter(p.sku for p in parts if p.sku),
            'mass': mass_properties(parts), 'release_ready': False,
        }
    bom = ['# Danh mục linh kiện bản nháp', '',
           '**Chưa nghiệm thu lắp ráp; không dùng danh mục này để đặt mua bộ hoàn chỉnh.**', '',
           'Số lượng dưới đây đếm từ instance trong CAD nháp, không phải BOM sản xuất đã chốt.', '']
    for variant, data in results['variants'].items():
        bom.extend([f'## {variant}', '', '| Mã linh kiện | Số lượng |', '| --- | ---: |'])
        for name, count in sorted({**data['print_quantities'], **data['purchased_quantities']}.items()):
            bom.append(f'| `{name}` | {count} |')
        bom.append('')
    (output / 'bom_and_fasteners.md').write_text('\n'.join(bom), encoding='utf-8')
    generated.append(output / 'bom_and_fasteners.md')
    media = output / 'media'
    if media.is_dir():
        generated.extend(path for path in media.rglob('*') if path.is_file())
    results['artifacts'] = artifact_inventory(output, generated)
    if source_receipt() != receipt:
        raise RuntimeError('CAD sources changed during export; incomplete manifest retained')
    (output / 'manifest.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', action='store_true', help='Export explicitly unapproved review assets')
    parser.add_argument('--output', type=Path, default=Path('artifacts/cad/draft'))
    args = parser.parse_args()
    if not args.draft:
        parser.error('Manufacturing release is blocked until assembly, sweep and DfAM checks pass; use --draft for review')
    result = export_drafts(args.output)
    print(json.dumps({'status': result['status'], 'parts': len(result['parts']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
