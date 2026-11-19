"""Explicit acceptance gates; sampled collision success is not fabrication approval."""

from .params import VARIANTS


def assess_release(parts, assemblies):
    variants_complete = (len(assemblies) == len(VARIANTS)
                         and {a['variant'] for a in assemblies} == set(VARIANTS))
    poses = [p for a in assemblies for p in a['poses']]
    gates = {
        'all_three_variants': variants_complete,
        'parts_brep': bool(parts) and all(p['passed'] for p in parts),
        'nominal_fit': variants_complete and all(
            a['poses'] and a['poses'][0]['geometry']['passed']
            and a['poses'][0]['geometry']['complete'] for a in assemblies),
        'sampled_motion': variants_complete and all(
            a['sweep_requested'] and a['sweep_completed']
            and all(p['geometry']['passed'] and p['geometry']['complete'] for p in a['poses'])
            for a in assemblies),
        'wire_lengths': bool(poses) and all(p['wires'] and all(w['passed'] for w in p['wires'])
                                          for p in poses),
        'mass_budget': bool(poses) and all(p['mass']['within_plan_budget'] for p in poses),
        # These require separate evidence which this staged implementation cannot supply yet.
        'verified_reference_sources': False,
        'complete_fastener_mating': False,
        'neck_service_loop_clearance': False,
        'dfam_wall_thickness': False,
        'print_orientation_support': False,
        'cad_simulation_parity': False,
        'servo_torque_envelope': False,
    }
    return {'release_ready': all(gates.values()), 'gates': gates,
            'unmet_gates': [name for name, passed in gates.items() if not passed]}
