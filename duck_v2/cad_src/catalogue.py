"""Connected printable shells; coupon factories are not assembly inventory."""

from functools import cache
from build123d import Plane

from . import monocoque as mono
from .armor.backpacks import backpack
from .armor.panels import reactor
from .geometry import box
from .params import VARIANTS, VARIANT_SUFFIX
from .skeleton.electronics import electronics_sled
from .skeleton.torso import battery_hatch


@cache
def printable_parts():
    """name -> (connected B-rep, material, declared design wall, variant)."""
    parts = {}

    def add(name, shape, material='PETG', wall=2.8, variant='shared'):
        shape = mono.checked(name, shape)
        shape.label = name
        parts[name] = (shape, material, wall, variant)

    add('trunk_front', mono.trunk_front(), 'PLA', 2.0)
    for name, factory in (('trunk_bottom', mono.trunk_bottom),
                          ('battery_hatch', battery_hatch), ('neck_link', mono.neck_link)):
        add(name, factory())
    add('head_front', mono.head_front(), 'PLA', 2.0)
    add('head_rear_cover', mono.head_rear_cover(), 'PLA', 2.0)
    # Keep the reactor bezel in the slate palette; no gold/yellow duck styling.
    add('reactor_bezel', reactor(), 'PLA', 2.2)
    add('reactor_diffuser', reactor(True), 'CLEAR', 2)
    add('visor_lens', mono.visor_lens(), 'CLEAR', 2.4)
    add('battery_foam_spacer', box((70, 32, 2)), 'EVA', 2)
    for side in ('L', 'R'):
        for name, factory, material, wall in (
            ('hip_clevis', mono.hip_clevis, 'PETG', 2.8),
            ('thigh', mono.thigh_shell, 'PLA', 2.2),
            ('shin', mono.shin_shell, 'PETG', 2.6),
            ('foot', mono.integrated_foot, 'PLA', 2.2),
            ('sole_tread', mono.sole_tread, 'TPU', 2.4),
        ):
            shape = factory()
            if side == 'R':
                shape = shape.mirror(Plane.XZ)
            add(f'{name}_{side}', shape, material, wall)
    for variant in VARIANTS:
        suffix = VARIANT_SUFFIX[variant]
        add(f'trunk_back_sled_{suffix}', electronics_sled(variant), variant=variant)
        add(f'trunk_back_cover_{suffix}', backpack(variant), 'PLA', 2.2, variant)
    return parts
