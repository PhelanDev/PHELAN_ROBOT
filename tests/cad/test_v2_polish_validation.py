"""Acceptance boundaries introduced by the monocoque polish plan."""

import numpy as np
import pytest

pytest.importorskip('build123d')


def test_mass_budget_accepts_exact_upper_bound_and_rejects_next_float():
    from cad.validation import within_mass_budget
    assert within_mass_budget(1.50)
    assert not within_mass_budget(np.nextafter(1.50, np.inf))
    assert within_mass_budget(1.17)  # The target lower mass is not a safety minimum.
    for invalid in (0, -1, np.nan, np.inf):
        assert not within_mass_budget(invalid)


@pytest.mark.parametrize('material, expected', [('PETG', 2.5), ('TPU', 2.2),
                                              ('PLA', 1.8), ('BRONZE', 1.8),
                                              ('CLEAR', 1.8)])
def test_wall_requirements_follow_the_polish_plan(material, expected):
    from cad.dfam import required_wall
    assert required_wall(material) == expected
    with pytest.raises(ValueError):
        required_wall('unknown')


def test_all_servo_station_axes_match_the_kinematic_joint_origins():
    from cad.kinematics import FRAMES
    from cad.layout import STATIONS
    for station in STATIONS:
        parent, offset, joint, axis = FRAMES[station.moving]
        assert parent == station.fixed and joint == station.joint
        index = 'xyz'.index(axis)
        # Sliding along the axle is allowed; transverse offsets are not.
        for dimension in set(range(3)) - {index}:
            assert station.origin[dimension] == pytest.approx(offset[dimension])


def test_default_sweep_contains_nine_samples_for_every_joint():
    from cad.validation import sample_poses
    assert len(list(sample_poses())) == 5 + 8 * 9


def test_neck_harness_is_routed_behind_the_yaw_axis():
    from cad.wiring.routes import harnesses
    route = next(route for route in harnesses('rpi5') if route.name == 'head_harness')
    neck_points = [xyz for link, xyz in route.anchors if link == 'neck']
    assert len(neck_points) >= 3
    assert all(xyz[0] <= -30 for xyz in neck_points)
    assert route.maximum_length is not None
