"""Local source-file verification; these tests never download or execute remote code."""

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip('build123d')

from build123d import import_step
from cad.purchased.fetch_sources import COMMIT, MODELS
from cad.purchased.source_audit import audit_sources


SOURCES = Path(__file__).resolve().parents[2] / 'cad/purchased/sources'


@pytest.mark.parametrize('name', MODELS)
def test_pinned_source_has_matching_lfs_hash_and_valid_step(name):
    metadata = json.loads((SOURCES / 'sources.json').read_text(encoding='utf-8'))
    assert metadata['upstream_commit'] == COMMIT
    entry = metadata['models'][name]
    path = SOURCES / entry['file']
    assert path.stat().st_size == entry['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    shape = import_step(path)
    assert shape.is_valid and shape.volume > 0


def test_catalog_f623_axial_mismatch_is_reported_and_never_auto_approved():
    report = audit_sources(SOURCES)
    bearing = report['models']['bearing_flanged_f623_simple']
    assert bearing['bbox_mm_in_source_axes'][2] == pytest.approx(5)
    assert bearing['checked_envelope_mm'][2] == 4
    assert bearing['envelope_compatible'] is False
    assert not report['all_references_approved']
    assert all(not data['used_in_assembly'] for data in report['models'].values())
