import numpy as np
import pytest

from sim.behaviors.nav_planner import LidarWaypointPlanner, NavPlannerConfig, RAY_ANGLES


def test_nav_planner_steers_toward_goal_and_avoids_blocked_front():
    planner = LidarWaypointPlanner(NavPlannerConfig())
    # Clear front -> move forward toward goal at (2.0, 0.0)
    cmd_clear = planner.compute_command(
        pos_xy=np.array([0.0, 0.0]),
        yaw=0.0,
        goal_xy=np.array([2.0, 0.0]),
        lidar_ranges=np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32),
    )
    assert cmd_clear[0] > 0.15 and abs(cmd_clear[2]) < 0.05

    # Blocked center & right, wide left (+15/+30 deg = indices 3,4) -> yaw left (wz > 0) and slow vx
    cmd_left = planner.compute_command(
        pos_xy=np.array([0.0, 0.0]),
        yaw=0.0,
        goal_xy=np.array([2.0, 0.0]),
        lidar_ranges=np.array([0.25, 0.25, 0.25, 1.80, 1.90], dtype=np.float32),
    )
    assert cmd_left[2] > 0.15
    assert cmd_left[0] < cmd_clear[0]


def test_nav_planner_steers_right_when_left_blocked():
    planner = LidarWaypointPlanner(NavPlannerConfig())
    cmd_clear = planner.compute_command(
        pos_xy=np.array([0.0, 0.0]),
        yaw=0.0,
        goal_xy=np.array([2.0, 0.0]),
        lidar_ranges=np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32),
    )
    # Blocked center & left (indices 2, 3, 4), wide right (indices 0, 1: -30, -15 deg)
    cmd_right = planner.compute_command(
        pos_xy=np.array([0.0, 0.0]),
        yaw=0.0,
        goal_xy=np.array([2.0, 0.0]),
        lidar_ranges=np.array([1.80, 1.90, 0.25, 0.25, 0.25], dtype=np.float32),
    )
    assert cmd_right[2] < -0.15  # wz < 0 steers right
    assert cmd_right[0] < cmd_clear[0]


def test_nav_planner_stops_at_goal():
    planner = LidarWaypointPlanner(NavPlannerConfig(goal_tol=0.15))
    pos = np.array([1.95, 0.02])
    goal = np.array([2.0, 0.0])
    assert planner.is_goal_reached(pos, goal)
    cmd = planner.compute_command(pos, yaw=0.0, goal_xy=goal)
    assert np.allclose(cmd, 0.0)

    # Away from goal
    assert not planner.is_goal_reached(np.array([0.0, 0.0]), goal)


def test_nav_planner_handles_none_short_lidar_and_quat_wraparound():
    planner = LidarWaypointPlanner()
    pos = np.array([0.0, 0.0])
    goal = np.array([0.0, 2.0])  # +90 deg left

    # 1. lidar_ranges=None steers toward bearing (+pi/2 -> wz > 0)
    cmd_none = planner.compute_command(pos, yaw=0.0, goal_xy=goal, lidar_ranges=None)
    assert cmd_none.shape == (3,)
    assert cmd_none.dtype == np.float32
    assert cmd_none[2] > 0.2

    # 2. Fewer than 5 rays does not crash
    cmd_short = planner.compute_command(pos, yaw=0.0, goal_xy=goal, lidar_ranges=np.array([1.0, 1.0]))
    assert cmd_short.shape == (3,)
    assert cmd_short.dtype == np.float32
    assert cmd_short[2] > 0.2

    # 3. Negative / zero distance handling
    cmd_neg = planner.compute_command(
        pos,
        yaw=0.0,
        goal_xy=np.array([2.0, 0.0]),
        lidar_ranges=np.array([-1.0, 0.0, 2.0, 2.0, 2.0], dtype=np.float32),
    )
    assert not np.isnan(cmd_neg).any()
    assert not np.isinf(cmd_neg).any()

    # 4. compute_command_from_quat with identity quat [1, 0, 0, 0]
    cmd_quat = planner.compute_command_from_quat(
        pos_xy=pos,
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        goal_xy=np.array([2.0, 0.0]),
    )
    assert cmd_quat[0] > 0.15
    assert abs(cmd_quat[2]) < 0.05

    # 5. compute_command_from_quat with 90-deg yaw quat (cos(pi/4), 0, 0, sin(pi/4))
    # Facing +Y, goal at (2, 0) is to the robot's right (-90 deg -> wz < 0)
    q_90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], dtype=np.float32)
    cmd_turn_right = planner.compute_command_from_quat(
        pos_xy=pos,
        quat_wxyz=q_90,
        goal_xy=np.array([2.0, 0.0]),
    )
    assert cmd_turn_right[2] < -0.15


def test_parallel_sim_runner_nav_planner_integration():
    import mujoco
    from sim.env.run_parallel_sim import ParallelSimRunner

    runner = ParallelSimRunner(
        model_path="robot/cyberduck_v2.xml",
        num_envs=2,
        num_threads=1,
        max_episode_steps=20,
        stage=1,
        use_nav_planner=True,
        nav_goal_xy=(2.0, 0.0),
    )
    try:
        assert runner.use_nav_planner is True
        assert isinstance(runner.nav_planner, LidarWaypointPlanner)
        assert np.allclose(runner.nav_goal_xy, [2.0, 0.0])

        obs = runner.reset()
        # Env 0 starts at (0, 0) facing +X, goal at (2, 0) -> positive forward velocity cmd
        assert runner.commands[0, 0] > 0.1
        # Observation channel 6:9 matches runner.commands
        assert np.allclose(obs[0, 6:9], runner.commands[0, :3], atol=1e-5)

        # Place robot close to pillar_l1_left (located around x=0.55, y=0.10)
        pillar_gid = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_GEOM, "pillar_l1_left")
        p_pos = runner.model.geom_pos[pillar_gid]
        runner.datas[0].qpos[0] = p_pos[0] - 0.25  # 25cm in front of pillar
        runner.datas[0].qpos[1] = p_pos[1]
        runner.datas[0].qpos[2] = 0.128
        runner.datas[0].qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        runner.datas[0].qvel[:] = 0.0
        mujoco.mj_forward(runner.model, runner.datas[0])

        runner.update_nav_commands(env_indices=[0])
        # Front ray is blocked by pillar -> robot steers away (wz != 0)
        assert abs(runner.commands[0, 2]) > 0.05

        # Place robot at goal position -> commands zeroed
        runner.datas[0].qpos[0] = 2.0
        runner.datas[0].qpos[1] = 0.0
        runner.datas[0].qpos[2] = 0.128
        runner.datas[0].qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        runner.datas[0].qvel[:] = 0.0
        mujoco.mj_forward(runner.model, runner.datas[0])

        runner.update_nav_commands(env_indices=[0])
        assert np.allclose(runner.commands[0, :3], 0.0, atol=1e-5)

        # Take one step and verify next_obs contains updated nav commands
        actions = np.zeros((runner.num_envs, runner.action_dim), dtype=np.float32)
        next_obs, rewards, dones, infos = runner.step(actions)
        assert np.allclose(next_obs[0, 6:9], runner.commands[0, :3], atol=1e-5)
    finally:
        runner.close()


@pytest.mark.parametrize("goal_xy,yaw", [((0.0, 2.0), np.pi / 2), ((0.0, -2.0), -np.pi / 2), ((2.0, 0.0), 0.0)])
def test_nav_planner_stage0_bearing_heading_and_yaw_termination(goal_xy, yaw):
    """F5: Stage 0 navigation must align target_yaw to waypoint bearing rather than world-X."""
    import mujoco
    from sim.env.run_parallel_sim import ParallelSimRunner

    runner = ParallelSimRunner(
        model_path="robot/cyberduck_v2.xml",
        num_envs=1,
        num_threads=1,
        max_episode_steps=50,
        stage=0,
        terrain_mode="flat",
        use_nav_planner=True,
        nav_goal_xy=goal_xy,
    )
    try:
        runner.reset()
        d = runner.datas[0]
        d.qpos[0:3] = [0.0, 0.0, 0.128]
        d.qpos[3:7] = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
        d.qvel[:] = 0.0
        mujoco.mj_forward(runner.model, d)
        runner.update_nav_commands(env_indices=[0])

        # Facing goal -> straight forward command (|wz| < 0.05) and no heading penalty or yaw_fallen
        assert abs(runner.commands[0, 2]) < 0.05
        assert runner.commands[0, 0] > 0.1
        _, _, dones, infos = runner.step(np.zeros((1, 8), dtype=np.float32))
        assert not dones[0], f"Aligned navigation goal={goal_xy} yaw={yaw:.2f} triggered false fall"
        assert infos[0]["heading_penalty"] == pytest.approx(0.0, abs=1e-3)

        # Slightly off bearing within straight-command deadband (|wz| < 0.05) -> negative heading_penalty
        slight_off_yaw = yaw + 0.03
        d.qpos[3:7] = [np.cos(slight_off_yaw / 2.0), 0.0, 0.0, np.sin(slight_off_yaw / 2.0)]
        d.qvel[:] = 0.0
        mujoco.mj_forward(runner.model, d)
        _, _, dones_slight, infos_slight = runner.step(np.zeros((1, 8), dtype=np.float32))
        assert not dones_slight[0]
        assert infos_slight[0]["heading_penalty"] < -1e-4

        # Standing at goal within goal_tol at arbitrary yaw must NOT falsely yaw-fall or penalize heading
        d.qpos[0:2] = [goal_xy[0], goal_xy[1]]
        off_yaw = yaw + np.pi / 2.0
        d.qpos[3:7] = [np.cos(off_yaw / 2.0), 0.0, 0.0, np.sin(off_yaw / 2.0)]
        d.qvel[:] = 0.0
        mujoco.mj_forward(runner.model, d)
        _, _, dones_at_goal, infos_at_goal = runner.step(np.zeros((1, 8), dtype=np.float32))
        assert not dones_at_goal[0], "Standing at goal within goal_tol must not falsely yaw-fall"
        assert infos_at_goal[0]["heading_penalty"] == pytest.approx(0.0, abs=1e-5)

        # Large bearing error (> 1.0 rad) when commanding straight walk triggers bearing-relative yaw_fallen
        runner.reset()
        d = runner.datas[0]
        d.qpos[0:3] = [0.0, 0.0, 0.128]
        far_off_yaw = yaw + 1.2
        d.qpos[3:7] = [np.cos(far_off_yaw / 2.0), 0.0, 0.0, np.sin(far_off_yaw / 2.0)]
        d.qvel[:] = 0.0
        mujoco.mj_forward(runner.model, d)
        runner.commands[0] = [0.2, 0.0, 0.0]
        runner.update_nav_commands = lambda *a, **kw: None
        _, _, dones_far, infos_far = runner.step(np.zeros((1, 8), dtype=np.float32))
        assert dones_far[0], "Large bearing error > 1.0 rad under straight command must terminate"
        assert infos_far[0]["heading_penalty"] < -1.0
    finally:
        runner.close()

