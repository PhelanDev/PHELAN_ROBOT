"""Physical assembly contracts for the sculpt/monocoque revision."""

from collections import Counter

import numpy as np
import pytest

pytest.importorskip('build123d')
from build123d import Vector

from cad.assembly import build_assembly
from cad.catalogue import printable_parts
from cad.monocoque import chest_loft_shell
from cad.params import VARIANTS


def test_chest_front_profile_covers_integrated_shoulder_mounts():
    shell = chest_loft_shell()
    assert shell.is_valid
    assert len(shell.solids()) == 1
    # This wall point overlaps the shoulder deck envelope while the interior
    # remains hollow; the former narrow loft stopped well inside this line.
    assert shell.is_inside(Vector(40, 53, -6))
    tf = printable_parts()['trunk_front'][0]
    assert tf.bounding_box().max.X == pytest.approx(62.5, abs=1e-4)
    assert tf.is_inside(Vector(61.5, 0.0, 6.0))


@pytest.mark.parametrize('variant', VARIANTS)
def test_monocoque_print_inventory_matches_the_plan(variant):
    assembly = build_assembly(variant, wiring=False)
    printed = [part for part in assembly if part.category == 'printed']
    quantities = Counter(part.print_key for part in printed)
    # The retired separate face piece is intentionally absent from this design.
    assert 21 <= len(printed) <= 28
    assert all(count == 1 for count in quantities.values())
    assert 'beak' not in quantities
    assert 'reactor_bezel' not in quantities and 'reactor_diffuser' not in quantities
    assert not any(part.name.startswith(('cassette_', 'idler_cap_', 'clamp_'))
                   for part in printed)
    assert sum(part.sku == 'mg996r' for part in assembly) == 8
    assert sum(part.sku == 'f623zz' for part in assembly) == 11
    assert sum(part.sku == '6803_2rs' for part in assembly) == 1


def test_catalogue_exports_only_used_parts_and_no_more_than_32_stls():
    catalogue = printable_parts()
    used = {part.print_key for variant in VARIANTS
            for part in build_assembly(variant, wiring=False)
            if part.category == 'printed'}
    stls = {name for name, (_, material, _, _) in catalogue.items() if material != 'EVA'}
    assert used == stls
    assert 25 <= len(stls) <= 32
    assert 'beak' not in catalogue
    assert 'reactor_bezel' not in catalogue and 'reactor_diffuser' not in catalogue
    assert all(material != 'BRONZE' for _, material, _, _ in catalogue.values())


def test_assembly_hero_render_excludes_hardware_only_views_keep_it():
    from types import SimpleNamespace
    from cad.render_drafts import _cutaway, _exterior

    shell = SimpleNamespace(category='printed', name='trunk_front')
    chassis = SimpleNamespace(category='printed', name='hip_clevis_L')
    hardware = SimpleNamespace(category='reference', name='servo')
    assert _exterior([shell, chassis, hardware]) == [shell]
    assert _cutaway([shell, chassis, hardware]) == [chassis, hardware]


@pytest.mark.parametrize('variant', VARIANTS)
def test_both_soles_have_five_mm_height_and_flat_ground_datum(variant):
    parts = {part.name: part for part in build_assembly(variant, wiring=False)}
    for side in ('L', 'R'):
        bounds = parts[f'sole_tread_{side}'].shape.bounding_box()
        assert bounds.size.Z == pytest.approx(5.0, abs=1e-6)
        assert bounds.min.Z == pytest.approx(0.0, abs=1e-6)


def test_every_monocoque_part_is_a_real_connected_solid():
    for name, (shape, _, _, _) in printable_parts().items():
        assert shape.is_valid, name
        assert len(shape.solids()) == 1, name
        assert np.isfinite(shape.volume) and shape.volume > 0, name
