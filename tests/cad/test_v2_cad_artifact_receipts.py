"""Source and output tampering must invalidate the generated-file receipt."""

import json

import pytest

pytest.importorskip('build123d')

from cad.artifacts import artifact_inventory, verify_artifacts
from cad.provenance import source_receipt


@pytest.mark.parametrize('declared_ready', [False, True])
def test_artifact_roundtrip_detects_content_changes_and_stale_sources(tmp_path, declared_ready):
    path = tmp_path / 'part.step'
    path.write_text('example file bytes', encoding='ascii')
    data = {'source_receipt': source_receipt(), 'release_ready': declared_ready,
            'artifacts': artifact_inventory(tmp_path, [path])}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(data), encoding='utf-8')
    report = verify_artifacts(manifest)
    assert report['passed'] and not report['release_ready']
    path.write_text('changed file bytes', encoding='ascii')
    assert not verify_artifacts(manifest)['passed']
    data['artifacts'] = artifact_inventory(tmp_path, [path])
    data['source_receipt']['sha256'] = '0' * 64
    manifest.write_text(json.dumps(data), encoding='utf-8')
    report = verify_artifacts(manifest)
    assert not report['passed'] and not report['source_current']


def test_artifact_inventory_rejects_paths_outside_export_directory(tmp_path):
    output = tmp_path / 'export'
    output.mkdir()
    outside = tmp_path / 'outside.step'
    outside.write_bytes(b'outside')
    with pytest.raises(ValueError, match='within'):
        artifact_inventory(output, [outside])


def test_manifest_cannot_read_a_parent_path(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'source_receipt': source_receipt(), 'release_ready': True,
                                  'artifacts': [{'path': '../outside.step', 'bytes': 0,
                                                 'sha256': '0' * 64}]}), encoding='utf-8')
    report = verify_artifacts(manifest)
    assert not report['passed'] and not report['release_ready']
    assert report['issues'][0]['reason'] == 'unsafe or duplicate path'
