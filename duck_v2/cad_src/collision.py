"""Broad-phase AABBs followed by real OpenCASCADE solid intersections."""

from itertools import combinations
from math import isfinite
from numbers import Integral


def interference(parts, *, tolerance_mm3=1e-6, limit=None):
    """Never exempt adjacent links or hardware: report every positive overlap."""
    if not isfinite(tolerance_mm3) or tolerance_mm3 < 0:
        raise ValueError('Overlap tolerance must be finite and nonnegative')
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1):
        raise ValueError('Interference limit must be a positive integer')
    parts = tuple(parts)
    bounds = [part.shape.bounding_box() for part in parts]
    failures, tested = [], 0
    for a, b in combinations(range(len(parts)), 2):
        ba, bb = bounds[a], bounds[b]
        if any(min(tuple(ba.max)[i], tuple(bb.max)[i]) -
               max(tuple(ba.min)[i], tuple(bb.min)[i]) <= 1e-7 for i in range(3)):
            continue
        tested += 1
        solids = parts[a].shape.intersect(parts[b].shape)
        volume = sum(abs(s.volume) for s in solids) if solids is not None else 0.0
        if volume > tolerance_mm3:
            failures.append({'a': parts[a].name, 'b': parts[b].name,
                             'volume_mm3': volume})
            if limit is not None and len(failures) >= limit:
                return {'passed': False, 'complete': False,
                        'tested_pairs': tested, 'interferences': failures}
    return {'passed': not failures, 'complete': True,
            'tested_pairs': tested, 'interferences': failures}
