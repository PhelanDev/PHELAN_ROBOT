"""Pack individual copies on 256 mm print beds using their true bounds."""

from collections import defaultdict
from math import isfinite
from pathlib import Path

import trimesh


def pack_plates(items, output, bed=256.0, gap=5.0):
    """items contains (name, mesh, material). All lengths are millimetres."""
    if not isfinite(bed) or not isfinite(gap) or bed <= 0 or gap < 0 or 2 * gap >= bed:
        raise ValueError('Print bed and gap must be finite with bed > 2 * gap >= 0')
    items = tuple(items)
    names = [name for name, _, _ in items]
    if len(names) != len(set(names)):
        raise ValueError('Print instance names must be unique')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups, reports = defaultdict(list), []
    for name, mesh, material in items:
        groups[material].append((name, mesh))
    for material, group in sorted(groups.items()):
        scene, placements = trimesh.Scene(), []
        plate, x, y, row_height = 1, gap, gap, 0.0

        def flush():
            if not placements:
                return
            filename = f'{material}_{plate:02d}.3mf'
            scene.export(output / filename)
            reports.append({'file': filename, 'bed_mm': [bed, bed],
                            'material': material, 'instances': list(placements)})

        for name, original in sorted(group, key=lambda item: -item[1].extents[1]):
            mesh = original.copy()
            mesh.apply_translation(-mesh.bounds[0])
            width, depth, _ = mesh.extents
            if max(width, depth) > bed - 2 * gap:
                raise ValueError(f'Part exceeds {bed} mm bed: {name}')
            if x + width > bed - gap:
                x, y, row_height = gap, y + row_height + gap, 0.0
            if y + depth > bed - gap:
                flush()
                plate += 1
                scene, placements = trimesh.Scene(), []
                x, y, row_height = gap, gap, 0.0
            mesh.apply_translation((x, y, 0))
            scene.add_geometry(mesh, node_name=name, geom_name=name)
            placements.append({'name': name, 'min_mm': mesh.bounds[0].tolist(),
                               'max_mm': mesh.bounds[1].tolist()})
            x += width + gap
            row_height = max(row_height, depth)
        flush()
    return reports
