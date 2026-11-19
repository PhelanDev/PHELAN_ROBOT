"""Content receipts for CAD inputs, independent of generated output timestamps."""

import hashlib
from importlib.metadata import version
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def source_receipt(root=REPO_ROOT):
    root = Path(root).resolve()
    sources = root / 'duck_v2/cad_src'
    inputs = list(sources.rglob('*.py'))
    inputs += list(sources.glob('requirements*.txt'))
    source_models = sources / 'purchased/sources'
    inputs += [p for p in source_models.glob('*') if p.suffix in ('.step', '.json', '.txt')]
    for name in ('duck_v2/robot_spec.py', 'training/env_config.py'):
        path = root / name
        if path.is_file():
            inputs.append(path)
    if not inputs:
        raise ValueError('No CAD source inputs found')
    entries = []
    for path in sorted(set(inputs), key=lambda item: item.relative_to(root).as_posix()):
        entries.append({'path': path.relative_to(root).as_posix(),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    environment = {name: version(name) for name in ('build123d', 'trimesh', 'numpy')}
    digest = hashlib.sha256()
    for entry in entries:
        digest.update((entry['path'] + '\0' + entry['sha256'] + '\n').encode('utf-8'))
    for name, value in sorted(environment.items()):
        digest.update((name + '\0' + value + '\n').encode('utf-8'))
    return {'schema': 1, 'sha256': digest.hexdigest(), 'inputs': entries,
            'environment': environment}
