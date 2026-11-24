"""TDD tests for ActiveHeadScanner virtual 5-ray sweeping perception and planner integration."""

import math
import numpy as np
import pytest


def test_active_head_scanner_projection_floor_filter_ttl_and_fov():
    from sim.behaviors.active_head_scanner import RAY_ANGLES_RAD, ActiveHeadScanner

    assert RAY_ANGLES_RAD.dtype == np.float32
    np.testing.assert_allclose(
        RAY_ANGLES_RAD,
        np.array([-math.pi / 6, -math.pi / 12, 0.0, math.pi / 12, math.pi / 6], dtype=np.float32),
        atol=1e-6,
    )

    scanner = ActiveHeadScanner(ttl_s=2.0, max_range_m=2.0, floor_pitch_cutoff_rad=0.35)
    np.testing.assert_allclose(scanner.get_virtual_rays(0.0), np.full(5, 2.0, dtype=np.float32))

    # 1. Valid tilted reading projects horizontally with cos(head_pitch)
    scanner.update(head_yaw_rad=0.0, head_pitch_rad=0.25, tof_range_m=1.0, now_s=1.0)
    rays = scanner.get_virtual_rays(1.1)
    assert rays.dtype == np.float32
    assert rays.shape == (5,)
    assert rays[2] == pytest.approx(math.cos(0.25), abs=1e-5)

    # 2. Floor pitch rejection: head_pitch_rad > 0.35 must ignore the sample completely
    scanner.update(head_yaw_rad=0.0, head_pitch_rad=0.45, tof_range_m=0.20, now_s=1.2)
    assert scanner.get_virtual_rays(1.2)[2] == pytest.approx(math.cos(0.25), abs=1e-5)

    # 3. Out-of-FOV yaw > deg2rad(37.5) outside outer bins must be ignored
    scanner.update(head_yaw_rad=1.2, head_pitch_rad=0.0, tof_range_m=0.30, now_s=1.3)
    np.testing.assert_allclose(scanner.get_virtual_rays(1.3)[[0, 1, 3, 4]], 2.0)

    # 4. Invalid/out-of-range readings (<0, NaN, >=max_range_m) normalize to max_range_m
    scanner.update(head_yaw_rad=math.pi / 6, head_pitch_rad=0.0, tof_range_m=-1.0, now_s=1.4)
    scanner.update(head_yaw_rad=-math.pi / 6, head_pitch_rad=0.0, tof_range_m=float("nan"), now_s=1.4)
    assert scanner.get_virtual_rays(1.4)[4] == pytest.approx(2.0)
    assert scanner.get_virtual_rays(1.4)[0] == pytest.approx(2.0)

    # 5. TTL expiration > 2.0 s reverts stale bin back to max_range_m
    assert scanner.get_virtual_rays(3.01)[2] == pytest.approx(2.0)


def test_active_head_scanner_modes_and_waypoint_planner_handoff():
    from sim.behaviors.active_head_scanner import ActiveHeadScanner
    from sim.behaviors.nav_planner import LidarWaypointPlanner, NavPlannerConfig

    scanner = ActiveHeadScanner()
    yaw, pitch = scanner.desired_head_angles("SCAN_NAV", now_s=0.25)
    assert yaw == pytest.approx(0.52 * math.sin(2.0 * math.pi * 0.75 * 0.25), abs=1e-5)
    assert pitch == pytest.approx(0.0)

    iyaw, ipitch = scanner.desired_head_angles("INTERACT", now_s=0.5, target_norm_xy=(3.0, -3.0))
    assert -1.57 <= iyaw <= 1.57
    assert -0.52 <= ipitch <= 0.78

    # Obstacles in left/center bins steer planner right cleanly
    scanner.update(head_yaw_rad=0.0, head_pitch_rad=0.0, tof_range_m=0.30, now_s=1.0)
    scanner.update(head_yaw_rad=-math.pi / 12, head_pitch_rad=0.0, tof_range_m=0.28, now_s=1.05)
    planner = LidarWaypointPlanner(NavPlannerConfig())
    cmd = planner.compute_command(
        pos_xy=np.zeros(2, dtype=np.float32),
        yaw=0.0,
        goal_xy=np.array([2.0, 0.0], dtype=np.float32),
        lidar_ranges=scanner.get_virtual_rays(1.1),
    )
    assert cmd.shape == (3,)
    assert np.all(np.isfinite(cmd))
    assert abs(float(cmd[2])) > 0.0
