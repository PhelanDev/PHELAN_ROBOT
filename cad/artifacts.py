"""Verify generated draft files against hashes and the current CAD source receipt."""

import argparse
import hashlib
import json
from pathlib import Path

from .provenance import source_receipt


def artifact_inventory(output, files):
    root = Path(output).resolve()
    rows = []
    seen = set()
    for entry in files:
        path = Path(entry).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Artifact must be a file within the export directory: {entry}')
        relative = path.relative_to(root).as_posix()
        if relative in seen:
            raise ValueError(f'Duplicate artifact path: {relative}')
        seen.add(relative)
        rows.append({'path': relative, 'bytes': path.stat().st_size,
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return sorted(rows, key=lambda row: row['path'])


def verify_artifacts(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    current = data.get('source_receipt') == source_receipt()
    issues, seen = [], set()
    rows = data.get('artifacts', [])
    for row in rows:
        relative = Path(row['path'])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or row['path'] in seen:
            issues.append({'path': row['path'], 'reason': 'unsafe or duplicate path'})
            continue
        seen.add(row['path'])
        if not path.is_file():
            issues.append({'path': row['path'], 'reason': 'missing'})
        elif path.stat().st_size != row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            issues.append({'path': row['path'], 'reason': 'content changed'})
    passed = bool(rows) and current and not issues
    return {'passed': passed, 'source_current': current, 'artifacts_checked': len(rows),
            'issues': issues, 'declared_release_ready': data.get('release_ready') is True,
            'release_ready': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path, nargs='?', default=Path('artifacts/cad/draft/manifest.json'))
    args = parser.parse_args()
    report = verify_artifacts(args.manifest)
    print(json.dumps(report))
    raise SystemExit(0 if report['passed'] else 1)
