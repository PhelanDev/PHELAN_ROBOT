"""Render review-only CAD assemblies in a MuJoCo studio scene."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import numpy as np
from build123d import export_stl

from .assembly import build_assembly
from .params import VARIANTS
from .provenance import source_receipt


SRC_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'artifacts/cad/draft/media'
COLORS = {
    'PLA': (0.34, 0.39, 0.43, 1),
    'PETG': (0.18, 0.23, 0.27, 1),
    'TPU': (0.075, 0.09, 0.105, 1),
    'BRONZE': (0.27, 0.16, 0.09, 1),
    'CLEAR': (0.02, 0.25, 0.31, 1),
    'SERVO': (0.065, 0.075, 0.085, 1),
    'ALUMINUM': (0.47, 0.51, 0.54, 1),
    'STEEL': (0.31, 0.36, 0.39, 1),
    'PCB': (0.08, 0.27, 0.19, 1),
    'ELECTRONICS': (0.12, 0.23, 0.36, 1),
    'BATTERY': (0.24, 0.09, 0.085, 1),
    'WIRE': (0.48, 0.20, 0.09, 1),
}

CUTAWAY_HIDE_PREFIXES = ('trunk_front', 'head_front', 'trunk_back_cover_')


def _scene_xml(parts, mesh_dir, *, exploded=False):
    mesh_dir = Path(mesh_dir).resolve()
    root = ET.Element('mujoco', model='cyberduck_v2_cad_review')
    ET.SubElement(root, 'compiler', angle='radian', meshdir=str(mesh_dir),
                  autolimits='true')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1600', offheight='1200')
    ET.SubElement(visual, 'quality', shadowsize='4096')
    ET.SubElement(visual, 'map', znear='0.004', zfar='4')
    ET.SubElement(visual, 'headlight', ambient='0.42 0.45 0.50',
                  diffuse='0.48 0.50 0.54', specular='0.12 0.13 0.15')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'texture', name='studio_sky', type='skybox',
                  builtin='gradient', rgb1='0.025 0.035 0.05',
                  rgb2='0.10 0.13 0.16', width='512', height='3072')
    ET.SubElement(asset, 'material', name='studio_floor',
                  rgba='0.075 0.09 0.105 1', specular='0.16', shininess='0.24')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', name='key', pos='0.5 -0.8 1.8',
                  dir='-0.2 0.3 -1', diffuse='0.82 0.86 0.92',
                  specular='0.34 0.36 0.40', directional='true')
    ET.SubElement(world, 'light', name='fill', pos='-0.8 0.7 1.1',
                  dir='0.4 -0.3 -1', diffuse='0.28 0.36 0.44',
                  specular='0.08 0.10 0.12', directional='true')
    ET.SubElement(world, 'light', name='rim', pos='-0.4 -0.4 1.6',
                  dir='0.2 0.4 -1', diffuse='0.28 0.34 0.42',
                  specular='0.15 0.18 0.22', directional='true')
    ET.SubElement(world, 'geom', name='studio_floor', type='plane',
                  size='2 2 0.1', material='studio_floor', contype='0',
                  conaffinity='0')

    model_body = ET.SubElement(world, 'body', name='cad_model')
    points = []
    for index, part in enumerate(parts):
        if not exploded and getattr(part, 'render_offset_m', None) is not None:
            offset = np.asarray(part.render_offset_m, dtype=float)
        else:
            offset = np.zeros(3)
        bounds = part.shape.bounding_box()
        lower = np.asarray(tuple(bounds.min)) * 0.001
        upper = np.asarray(tuple(bounds.max)) * 0.001
        center = (lower + upper) / 2
        points.extend((lower + offset, upper + offset))
        mesh_name = f'part_{index:03d}'
        mesh_path = mesh_dir / f'{mesh_name}.stl'
        export_stl(part.shape, mesh_path, tolerance=0.08, angular_tolerance=0.15)
        color = COLORS.get(part.material, COLORS['PETG'])
        rgba = ' '.join(f'{value:.4g}' for value in color)
        material_name = f'material_{index:03d}'
        ET.SubElement(asset, 'material', name=material_name, rgba=rgba,
                      specular='0.24', shininess='0.28')
        ET.SubElement(asset, 'mesh', name=mesh_name, file=mesh_path.name,
                      scale='0.001 0.001 0.001')
        ET.SubElement(model_body, 'geom', name=f'geom_{index:03d}', type='mesh',
                      mesh=mesh_name, material=material_name,
                      pos=' '.join(f'{value:.7g}' for value in offset),
                      contype='0', conaffinity='0', group='0')

    if not points:
        raise ValueError('Cannot render an empty CAD assembly')
    array = np.asarray(points)
    minimum, maximum = array.min(axis=0), array.max(axis=0)
    lookat = (minimum + maximum) / 2
    radius = max(float(np.linalg.norm(maximum - minimum)) / 2, 0.1)
    return ET.tostring(root, encoding='unicode'), lookat.tolist(), max(radius * 2.25, 0.3)


def _cutaway(parts):
    return [part for part in parts
            if not part.name.startswith(CUTAWAY_HIDE_PREFIXES)]


def _exterior(parts):
    """Render recognizable outer armor; reserve internal prints for cutaway."""
    fixed_shells = {'trunk_front', 'head_front', 'head_rear_cover', 'visor_lens',
                    'reactor_bezel', 'reactor_diffuser'}
    return [part for part in parts
            if part.category == 'printed'
            and (part.name in fixed_shells
                 or part.name.startswith(('trunk_back_cover_', 'thigh_', 'shin_',
                                          'foot_', 'sole_tread_')))]


def _exploded(parts):
    exploded = []
    for part in parts:
        if part.category == 'reference' and (part.sku or '').startswith('m3'):
            exploded.append(part)
            continue
        center = np.asarray(tuple(part.shape.center()), dtype=float) * 0.001
        radial = center - np.array((0.0, 0.0, 0.17))
        length = np.linalg.norm(radial)
        offset = radial / length * 0.018 if length > 1e-6 else np.zeros(3)
        exploded.append(part.placed(__import__('build123d').Pos(*(offset * 1000))))
    return exploded


def _render_scene(parts, output, renderer_python, scratch, *, exploded=False,
                  azimuth=180):
    mesh_dir = Path(scratch) / f'meshes_{output.stem}'
    mesh_dir.mkdir(parents=True)
    xml, lookat, distance = _scene_xml(parts, mesh_dir, exploded=exploded)
    scene_path = Path(scratch) / f'{output.stem}.xml'
    scene_path.write_text(xml, encoding='utf-8')
    request_path = Path(scratch) / f'{output.stem}.json'
    request_path.write_text(json.dumps({
        'scene': str(scene_path), 'output': str(output),
        'lookat_m': lookat, 'distance_m': distance,
        'size_px': [1600, 1200], 'azimuth_deg': azimuth, 'elevation_deg': -18,
    }), encoding='utf-8')
    helper = SRC_ROOT / 'cad/render_mujoco.py'
    subprocess.run([renderer_python, str(helper), str(request_path)],
                   cwd=ROOT, check=True)


def render_drafts(output=DEFAULT_OUTPUT, renderer_python=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    renderer_python = (renderer_python or os.environ.get('CYBERDUCK_MUJOCO_PYTHON')
                       or shutil.which('python'))
    if not renderer_python:
        raise RuntimeError('Set CYBERDUCK_MUJOCO_PYTHON to a Python with MuJoCo and Pillow installed')
    probe = subprocess.run([renderer_python, '-c',
                            'import mujoco, PIL; print(mujoco.__version__)'],
                           cwd=ROOT, capture_output=True, text=True)
    if probe.returncode:
        raise RuntimeError('MuJoCo render Python is unavailable: ' + probe.stderr.strip())

    receipt = source_receipt()
    renders = []
    with TemporaryDirectory(prefix='cyberduck-mujoco-') as scratch:
        for variant in VARIANTS:
            parts = build_assembly(variant)
            filename = f'{variant}_assembly.png'
            # Three-quarter view shows outer armor consistently across board
            # variants; internal prints and hardware stay in cutaway/exploded.
            azimuth = 145
            _render_scene(_exterior(parts), output / filename, renderer_python, scratch,
                          azimuth=azimuth)
            path = output / filename
            renders.append({'file': filename, 'variant': variant, 'view': 'assembly',
                            'bytes': path.stat().st_size,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            if variant == 'rpi5':
                for label, selected, is_exploded in (
                    ('cutaway', _cutaway(parts), False),
                    ('exploded', _exploded(parts), True),
                ):
                    filename = f'{variant}_{label}.png'
                    _render_scene(selected, output / filename, renderer_python,
                                  scratch, exploded=is_exploded)
                    path = output / filename
                    renders.append({'file': filename, 'variant': variant, 'view': label,
                                    'bytes': path.stat().st_size,
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if source_receipt() != receipt:
        raise RuntimeError('CAD sources changed during rendering; render manifest not updated')
    manifest = {'status': 'DRAFT_NOT_FOR_FABRICATION', 'release_ready': False,
                'source_receipt': receipt, 'renderer': 'MuJoCo offscreen OpenGL studio (not path-traced PBR)',
                'palette': 'titanium, graphite, cyan',
                'assembly_render_scope': 'outer armor selection; internal prints and hardware in cutaway/exploded',
                'renders': renders}
    (output / 'render_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    result = render_drafts()
    print(json.dumps({'status': result['status'], 'renders': len(result['renders']),
                      'output': str(DEFAULT_OUTPUT)}))
