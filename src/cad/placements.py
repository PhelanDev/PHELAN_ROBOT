"""Named monocoque instances in the common kinematic link frames (mm)."""

from build123d import Pos, Rot
from .params import VARIANT_SUFFIX
from .armor.backpacks import module_frames


def print_placements(variant):
    placements = []

    def put(key, link='trunk_base', pos=(0, 0, 0), rot=(0, 0, 0), location=None):
        frame = location if location is not None else Pos(*pos) * Rot(*rot)
        placements.append((key, key, link, frame))

    for name in ('trunk_front', 'trunk_bottom', 'battery_hatch'):
        put(name)
    put('battery_foam_spacer', pos=(0, 0, -48.5))
    put('neck_link', 'neck')
    for name in ('head_front', 'head_rear_cover', 'visor_lens'):
        put(name, 'head')
    suffix = VARIANT_SUFFIX[variant]
    housing, sled, _ = module_frames(variant)
    put(f'trunk_back_sled_{suffix}', location=sled)
    put(f'trunk_back_cover_{suffix}', location=housing)
    for side, prefix in (('L', 'left'), ('R', 'right')):
        for stem, link in (('hip_clevis', 'hip'), ('thigh', 'thigh'),
                           ('thigh_cover', 'thigh'), ('thigh_idler_cap', 'thigh'),
                           ('shin', 'shin'), ('shin_idler_cap', 'shin'),
                           ('foot', 'foot'), ('sole_tread', 'foot')):
            put(f'{stem}_{side}', f'{prefix}_{link}')
    return placements
