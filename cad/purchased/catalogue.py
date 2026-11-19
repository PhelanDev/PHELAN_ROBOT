"""Reference solids with explicit mass assumptions and source provenance."""

from functools import cache

from .electronics import BOARDS, active_cooler, board_model
from .hardware import flanged_bearing, insert, neck_bearing, nyloc, screw, washer
from .servo import horn_25t, mg996r
from ..geometry import box, cylinder, ring


@cache
def purchased_parts():
    parts = {name: (board_model(name), board.mass) for name, board in BOARDS.items()}
    parts.update({
        'mg996r': (mg996r(), 0.055), 'horn_25t_20mm': (horn_25t(), 0.003),
        'f623zz': (flanged_bearing(), 0.0018), '6803_2rs': (neck_bearing(), 0.008),
        'm3_insert': (insert(), 0.00035), 'm3_nyloc': (nyloc(), 0.0004),
        'm3_washer': (washer(), 0.00015),
        'm3_wide_washer': (ring(9.0, 3.2, 0.8), 0.0004),
        'pi5_active_cooler': (active_cooler(), 0.025),
        'battery_2s_2000mah': (box((75, 48, 16)), 0.105),
        'ubec_6v_15a': (box((50, 25, 14)), 0.025),
        'capacitor_4700uf': (cylinder(18, 35), 0.009),
        'main_switch': (box((19, 13, 22)), 0.005),
        'xt30_connector': (box((14, 12, 7)), 0.0025),
    })
    for length in (6, 8, 12, 18, 20):
        parts[f'm3x{length}'] = (screw(length), 0.0005 + length * 0.000045)
    return parts
