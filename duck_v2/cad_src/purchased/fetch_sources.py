"""Fetch pinned step.parts source models, verifying Git LFS content hashes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen


COMMIT = 'c6113328a5695b976a010a203a90fe86191769bf'
RAW = f'https://raw.githubusercontent.com/earthtojake/step.parts/{COMMIT}/'
MEDIA = f'https://media.githubusercontent.com/media/earthtojake/step.parts/{COMMIT}/'
MODELS = (
    'raspberry_pi_5', 'esp32_s3_wroom_1', 'bearing_flanged_f623_simple',
    'raspberry_pi_zero_mounting_holes_cutter_reference',
    *(f'iso4762_socket_head_cap_screw_m3x{n}' for n in (6, 8, 12, 18, 20)),
)
NOTES = {
    'esp32_s3_wroom_1': 'RF module only; NOT the 70 x 29 mm DevKit carrier board.',
    'bearing_flanged_f623_simple': 'Simplified open F623 envelope; NOT a certified F623ZZ model.',
    'raspberry_pi_zero_mounting_holes_cutter_reference':
        'Mounting-hole cutter pattern only; NOT a Pi Zero 2 W board model.',
}


def read_url(url):
    request = Request(url, headers={'User-Agent': 'CyberDuck-CAD-Reference/1.0'})
    with urlopen(request, timeout=60) as response:
        return response.read()


def fetch_sources(output=None):
    output = Path(output or Path(__file__).with_name('sources'))
    output.mkdir(parents=True, exist_ok=True)
    metadata_bytes = read_url(RAW + 'catalog/parts.json')
    metadata = {row['id']: row for row in json.loads(metadata_bytes)}
    license_bytes = read_url(RAW + 'LICENSE')
    (output / 'LICENSE-step.parts.txt').write_bytes(license_bytes)
    manifest = {'upstream_commit': COMMIT, 'catalog_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
                'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                'assembly_uses_these_sources': False, 'models': {}}
    for name in MODELS:
        relative = f'catalog/step/{name}.step'
        pointer = read_url(RAW + relative).decode('ascii')
        match = re.fullmatch(r'version https://git-lfs.github.com/spec/v1\n'
                             r'oid sha256:([a-f0-9]{64})\nsize (\d+)\n?', pointer)
        if not match:
            raise ValueError(f'Expected pinned Git LFS pointer for {name}')
        digest, size = match.group(1), int(match.group(2))
        target = output / f'{name}.step'
        data = target.read_bytes() if target.is_file() else b''
        if hashlib.sha256(data).hexdigest() != digest:
            data = read_url(MEDIA + relative)
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'LFS size/hash mismatch for {name}')
        if not data.lstrip().startswith(b'ISO-10303-21;'):
            raise ValueError(f'Not an ISO STEP exchange file: {name}')
        target.write_bytes(data)
        manifest['models'][name] = {
            'file': target.name, 'sha256': digest, 'bytes': size,
            'download': MEDIA + relative, 'catalog': metadata[name],
            'fit_reference_compared': False, 'vendor_certified': False,
            'note': NOTES.get(name, 'Source reference pending datum/fit comparison.'),
        }
    (output / 'sources.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = fetch_sources(args.output)
    print(json.dumps({'downloaded': len(manifest['models']), 'upstream_commit': COMMIT}))
