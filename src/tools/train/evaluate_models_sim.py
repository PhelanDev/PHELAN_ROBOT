"""Simulation benchmark and evaluation for CyberDuck V2.

Evaluates:
- cyberduck_v2_best.pt on CyberDuck V2 MJCF
Across multiple locomotion scenarios:
1. Nominal forward walking (vx = 0.5 m/s)
2. Chibi forward walking (vx = 0.25 m/s)
3. Stationary standing balance (vx = 0.0 m/s)
4. Hurdles / obstacle terrain (Stage 1)
"""

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot.robot_spec import CHECKPOINT_PREFIX, MODEL_NAME, MODEL_PATH, NUM_ACTIONS, NUM_OBSERVATIONS, REPO_ROOT
from sim.env.env_config import _DEFAULT_CFG, quat_to_yaw
from sim.env.policy_contract import validate_model_identity
from sim.env.run_parallel_sim import ParallelSimRunner, has_obstacle_contact, resolve_v2_model_path
from sim.behaviors.nav_planner import LidarWaypointPlanner
from tools.robot.export_policy import load_actor_from_checkpoint, PolicyOnnxWrapper

__all__ = [
    "compute_path_unstability",
    "has_obstacle_contact",
    "run_benchmark_scenario",
    "get_xml_supported_stage",
    "load_actor",
]


def compute_path_unstability(yaw_history: np.ndarray, dist_x: float) -> Tuple[float, float]:
    """Compute Total Variation of yaw angles (raw rad and distance-normalized rad/m).

    Handles wraparound across [-pi, pi) smoothly.
    Returns:
        (raw_yaw_tv_rad, norm_yaw_unstability_rad_per_m)
    """
    yaw_arr = np.asarray(yaw_history, dtype=np.float32)
    if yaw_arr.size < 2:
        return 0.0, 0.0
    diffs = (np.diff(yaw_arr) + np.pi) % (2.0 * np.pi) - np.pi
    raw = float(np.sum(np.abs(diffs)))
    norm = float(raw / max(abs(float(dist_x)), 0.1))
    return raw, norm


def get_xml_supported_stage(xml_path: Path, requested_stage: int) -> int:
    """Return requested_stage if XML contains obstacle/stair geometry, else fallback to 0."""
    xml_path = resolve_v2_model_path(xml_path)
    if requested_stage <= 0:
        return 0
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        has_stairs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "stair_step_1") != -1
        has_obstacles = (
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pillar_l1_left") != -1
            or mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_front") != -1
        )
        return requested_stage if (has_stairs or has_obstacles) else 0
    except Exception:
        return 0


def load_actor(checkpoint_path: Path) -> nn.Module:
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(cp, dict):
        raise ValueError("V2 checkpoint model identity metadata is required")
    normalized_cp = dict(cp)
    if normalized_cp.get("model_path"):
        normalized_cp["model_path"] = str(normalized_cp["model_path"]).replace("\\", "/")
    validate_model_identity(normalized_cp, MODEL_NAME)
    raw_actor = load_actor_from_checkpoint(cp)
    actor = PolicyOnnxWrapper(raw_actor).eval()
    actor.obs_dim = raw_actor[0].in_features
    actor.hidden_dim = raw_actor[0].out_features
    actor.action_dim = raw_actor[4].out_features
    if actor.obs_dim != NUM_OBSERVATIONS or actor.action_dim != NUM_ACTIONS:
        raise ValueError(f"V2 evaluation requires {NUM_OBSERVATIONS} observations and {NUM_ACTIONS} actions")
    actor.critic_obs_dim = cp.get("critic_obs_dim") if isinstance(cp, dict) else None
    actor.stage = int(cp.get("stage", 0)) if isinstance(cp, dict) else 0
    return actor


def run_benchmark_scenario(
    runner: ParallelSimRunner,
    actor: nn.Module,
    cmd_vx: float = 0.5,
    max_steps: int = 500,
    scenario_name: str = "Forward Walk",
    use_nav_planner: Optional[bool] = None,
    goal_x: float = 2.0,
) -> Dict[str, Any]:
    orig_use_nav = getattr(runner, "use_nav_planner", False)
    orig_goal_xy = getattr(runner, "nav_goal_xy", (2.0, 0.0))
    orig_nav_planner = getattr(runner, "nav_planner", None)

    effective_nav = orig_use_nav if use_nav_planner is None else bool(use_nav_planner)
    if runner.obs_dim != NUM_OBSERVATIONS or getattr(actor, "obs_dim", NUM_OBSERVATIONS) != NUM_OBSERVATIONS:
        raise ValueError(f"V2 evaluation requires {NUM_OBSERVATIONS}-dimensional observations")
    runner.use_nav_planner = effective_nav
    runner.nav_goal_xy = (float(goal_x), 0.0)
    if effective_nav and getattr(runner, "nav_planner", None) is None:
        runner.nav_planner = LidarWaypointPlanner()

    try:
        obs = runner.reset()
        if effective_nav:
            runner.update_nav_commands()
            obs = runner.get_observations()
        else:
            runner.commands[:, 0] = cmd_vx
            runner.commands[:, 1] = 0.0
            runner.commands[:, 2] = 0.0
            obs = runner.get_observations()

        num_envs = runner.num_envs
        alive_steps = np.zeros(num_envs, dtype=np.int32)
        accum_reward = np.zeros(num_envs, dtype=np.float32)
        max_dist_x = np.zeros(num_envs, dtype=np.float32)
        final_dist_x = np.zeros(num_envs, dtype=np.float32)
        max_tilt_deg = np.zeros(num_envs, dtype=np.float32)
        accum_torque_sq = np.zeros(num_envs, dtype=np.float32)
        accum_work = np.zeros(num_envs, dtype=np.float32)
        survived = np.ones(num_envs, dtype=bool)

        # Initial positions
        start_x = np.array([d.qpos[0] for d in runner.datas], dtype=np.float32)

        # Track obstacle avoidance & yaw trajectory metrics
        yaw_histories: List[List[float]] = [
            [float(quat_to_yaw(runner.datas[i].qpos[3:7]))] for i in range(num_envs)
        ]
        env_collided = np.zeros(num_envs, dtype=bool)
        min_clearance = np.full(num_envs, 2.0, dtype=np.float32)

        for step in range(max_steps):
            with torch.no_grad():
                inp = torch.from_numpy(obs).float()
                if inp.shape[1] != NUM_OBSERVATIONS:
                    raise ValueError(f"V2 evaluation observation mismatch: {inp.shape[1]} != {NUM_OBSERVATIONS}")
                actions = actor(inp).numpy()

            next_obs, rewards, dones, infos = runner.step(actions)

            # Track metrics for currently alive envs
            for i in range(num_envs):
                if survived[i]:
                    alive_steps[i] += 1
                    accum_reward[i] += float(rewards[i])
                    curr_x = float(infos[i].get("distance_x", 0.0))
                    final_dist_x[i] = curr_x
                    if curr_x > max_dist_x[i]:
                        max_dist_x[i] = curr_x

                    # Tilt angle from projected gravity (grav_z)
                    grav_z = float(next_obs[i, 5])
                    tilt_rad = math.acos(max(-1.0, min(1.0, -grav_z)))
                    tilt_deg = math.degrees(tilt_rad)
                    if tilt_deg > max_tilt_deg[i]:
                        max_tilt_deg[i] = tilt_deg

                    # Effort / Torque
                    ctrl = runner.datas[i].ctrl[:]
                    qvel = runner.datas[i].qvel[6:14]
                    accum_torque_sq[i] += float(np.sum(ctrl ** 2))
                    accum_work[i] += float(np.sum(np.abs(ctrl * qvel))) * runner.control_dt

                    # Yaw trajectory & obstacle tracking
                    pre_yaw = float(infos[i].get("yaw", quat_to_yaw(runner.datas[i].qpos[3:7])))
                    yaw_histories[i].append(pre_yaw)

                    env_collided[i] |= bool(infos[i].get("obstacle_contact", False))

                    d_val = infos[i].get("d_min")
                    if d_val is None and next_obs.shape[1] >= 42:
                        lidar_slice = (
                            infos[i].get("terminal_obs", next_obs[i])[37:42]
                            if dones[i]
                            else next_obs[i, 37:42]
                        )
                        d_val = float(np.min(lidar_slice))
                    if d_val is not None:
                        min_clearance[i] = min(min_clearance[i], float(d_val))

                    if dones[i]:
                        survived[i] = False

            obs = next_obs
            if not np.any(survived):
                break

        success_rate = float(np.mean(alive_steps >= max_steps))
        mean_eplen = float(np.mean(alive_steps))
        mean_reward = float(np.mean(accum_reward))
        mean_dist_x = float(np.mean(final_dist_x))
        mean_max_dist_x = float(np.mean(max_dist_x))
        mean_speed_x = float(np.mean(final_dist_x / np.maximum(alive_steps * runner.control_dt, 1e-3)))
        mean_tilt = float(np.mean(max_tilt_deg))
        mean_torque_sq = float(np.mean(accum_torque_sq / np.maximum(alive_steps, 1)))
        mean_power_watt = float(np.mean(accum_work / np.maximum(alive_steps * runner.control_dt, 1e-3)))

        unstab_list = [
            compute_path_unstability(np.array(yaw_histories[i], dtype=np.float32), final_dist_x[i])[1]
            for i in range(num_envs)
        ]
        mean_yaw_unstability = float(np.mean(unstab_list)) if unstab_list else 0.0
        collision_rate_per_100 = float(np.mean(env_collided) * 100.0)
        mean_min_clearance = float(np.mean(min_clearance))

        return {
            "scenario": scenario_name,
            "cmd_vx": cmd_vx,
            "max_steps": max_steps,
            "num_envs": num_envs,
            "success_rate": success_rate,
            "mean_eplen": mean_eplen,
            "mean_reward": mean_reward,
            "mean_dist_x": mean_dist_x,
            "mean_max_dist_x": mean_max_dist_x,
            "mean_speed_x": mean_speed_x,
            "mean_tilt_deg": mean_tilt,
            "mean_torque_sq": mean_torque_sq,
            "mean_power_watt": mean_power_watt,
            "yaw_unstability": mean_yaw_unstability,
            "collision_rate_per_100": collision_rate_per_100,
            "min_clearance_m": mean_min_clearance,
        }
    finally:
        runner.use_nav_planner = orig_use_nav
        runner.nav_goal_xy = orig_goal_xy
        runner.nav_planner = orig_nav_planner


def main():
    default_checkpoint = REPO_ROOT / "models" / "checkpoints" / f"{CHECKPOINT_PREFIX}_best.pt"
    parser = argparse.ArgumentParser(description="CyberDuck V2 simulation benchmark")
    parser.add_argument("--use-nav-planner", action="store_true", help="Include Stage 1 Hurdles + L2 NavPlanner scenario")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments (default: 16)")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per scenario (default: 500)")
    parser.add_argument(
        "--checkpoint", "--v2-checkpoint",
        dest="checkpoint",
        type=str,
        default=str(default_checkpoint),
        help=f"Path to CyberDuck V2 checkpoint (default: {default_checkpoint})",
    )
    args = parser.parse_args()

    model_path = resolve_v2_model_path(MODEL_PATH)
    actor = load_actor(Path(args.checkpoint))

    print("================================================================================")
    print("CYBERDUCK V2 SIMULATION EVALUATION")
    print("================================================================================")

    scenarios = [
        {"name": "Standard Forward Walk (0.50 m/s)", "cmd_vx": 0.50, "steps": args.steps, "stage": 0, "use_nav": False},
        {"name": "Chibi Walking Pace (0.25 m/s)", "cmd_vx": 0.25, "steps": args.steps, "stage": 0, "use_nav": False},
        {"name": "Stationary Balance Stand (0.0 m/s)", "cmd_vx": 0.00, "steps": min(300, args.steps), "stage": 0, "use_nav": False},
        {"name": "Stage 1 Hurdles Navigation", "cmd_vx": 0.40, "steps": args.steps, "stage": 1, "use_nav": False},
    ]
    if args.use_nav_planner:
        scenarios.append(
            {"name": "Stage 1 Hurdles + L2 NavPlanner", "cmd_vx": 0.40, "steps": args.steps, "stage": 1, "use_nav": True}
        )

    results = []

    for sc in scenarios:
        print(f"\n>>> Running V2 scenario: {sc['name']} ...")
        stage = get_xml_supported_stage(model_path, sc.get("stage", 0))
        runner = ParallelSimRunner(
            model_path=model_path,
            num_envs=args.num_envs,
            max_episode_steps=sc["steps"],
            stage=stage,
            use_nav_planner=sc.get("use_nav", False),
        )
        try:
            result = run_benchmark_scenario(
                runner,
                actor,
                cmd_vx=sc["cmd_vx"],
                max_steps=sc["steps"],
                scenario_name=sc["name"],
                use_nav_planner=sc.get("use_nav", False),
            )
            result["model"] = "CyberDuck V2"
            results.append(result)
        finally:
            runner.close()
        print(
            f"  EpLen: {result['mean_eplen']:.1f}/{sc['steps']} | DistX: {result['mean_dist_x']:+.3f}m | "
            f"Reward: {result['mean_reward']:+.2f} | Tilt: {result['mean_tilt_deg']:.1f}° | "
            f"Unstab: {result['yaw_unstability']:.2f} | Coll/100: {result['collision_rate_per_100']:.1f} | "
            f"MinClr: {result['min_clearance_m']:.2f}m"
        )

    header = f"{'Model':<16} | {'Scenario':<30} | {'Surv%':<6} | {'EpLen':<8} | {'DistX (m)':<10} | {'Reward':<9} | {'Tilt°':<6} | {'Unstab':<8} | {'Coll/100':<9} | {'MinClr':<8} | {'Pwr (W)':<7}"
    print("\n" + "=" * len(header))
    print("DETAILED BENCHMARK SUMMARY TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{'CyberDuck V2':<16} | {r['scenario'][:30]:<30} | {r['success_rate']*100:5.1f}% | {r['mean_eplen']:8.1f} | "
            f"{r['mean_dist_x']:+10.3f} | {r['mean_reward']:+9.2f} | {r['mean_tilt_deg']:5.1f}° | "
            f"{r['yaw_unstability']:8.2f} | {r['collision_rate_per_100']:8.1f}% | {r['min_clearance_m']:7.2f}m | {r['mean_power_watt']:6.2f}W"
        )
    print("=" * len(header))


if __name__ == "__main__":
    main()
