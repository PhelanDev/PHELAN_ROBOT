#!/usr/bin/env python3
"""Closed-loop Behavioral Cloning warmup for Actor and Critic initialization."""

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC_ROOT = str(Path(__file__).resolve().parents[2])
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from tools.robot.export_policy import load_actor_from_checkpoint, policy_actor_forward
from tools.train.train_ppo import ActorCritic, RunningMeanStd
from robot.robot_spec import CHECKPOINT_PREFIX, CONTROL_DT_S, MODEL_NAME, MODEL_PATH, NUM_ACTIONS, NUM_OBSERVATIONS, REPO_ROOT
from sim.behaviors.advanced_gaits import BipedalGaitGenerator
from sim.env.env_config import unscale_actions
from sim.env.policy_contract import observation_schema, validate_model_identity
from sim.env.run_parallel_sim import ParallelSimRunner
from sim.env.symmetry import (
    augment_symmetry_batch,
    compute_actor_symmetry_loss,
    mirror_critic_obs,
    mirror_obs,
)


def _get_actor_in_dim(actor: nn.Module) -> int:
    for attr in ("obs_dim", "in_features"):
        if hasattr(actor, attr):
            return int(getattr(actor, attr))
    if hasattr(actor, "actor"):
        return _get_actor_in_dim(actor.actor)
    if hasattr(actor, "__getitem__"):
        return _get_actor_in_dim(actor[0])
    return NUM_OBSERVATIONS


def _collect_rollouts(
    runner: ParallelSimRunner,
    policy_fn: Callable[[np.ndarray, int], Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]],
    num_steps: int,
    gamma: float = 0.99,
    return_privileged: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    obs = runner.reset()
    traj_obs, traj_priv, traj_act, traj_rew, traj_done = [], [], [], [], []
    for s in range(num_steps):
        if return_privileged:
            traj_priv.append(runner.get_privileged_obs(raw_obs=obs).copy())
        out = policy_fn(obs, s)
        exec_act, label_act = out if isinstance(out, tuple) else (out, out)
        next_obs, rews, dones, _ = runner.step(exec_act, fast=True)
        traj_obs.append(obs.copy())
        traj_act.append(label_act.copy())
        traj_rew.append(rews.copy())
        traj_done.append(dones.copy())
        obs = next_obs

    obs_seq, act_seq = np.array(traj_obs, dtype=np.float32), np.array(traj_act, dtype=np.float32)
    rew_seq, done_seq = np.array(traj_rew, dtype=np.float32), np.array(traj_done, dtype=bool)

    t_steps, n_envs = rew_seq.shape
    returns, running_ret = np.zeros_like(rew_seq), np.zeros(n_envs, dtype=np.float32)
    for t in reversed(range(t_steps)):
        running_ret = rew_seq[t] + gamma * running_ret * (1.0 - done_seq[t].astype(np.float32))
        returns[t] = running_ret

    valid = np.ones((t_steps, n_envs), dtype=bool)
    for e in range(n_envs):
        fallen = np.where(done_seq[:, e])[0]
        if len(fallen) > 0:
            valid[fallen[0]:, e] = False
    if not np.any(valid):
        # shortcut: fallback to non-done steps if all envs fell early, upgrade when robust recovery online
        valid = ~done_seq if np.any(~done_seq) else np.ones((t_steps, n_envs), dtype=bool)

    if return_privileged:
        priv_seq = np.array(traj_priv, dtype=np.float32)
        return obs_seq[valid], priv_seq[valid], act_seq[valid], returns[valid]
    return obs_seq[valid], act_seq[valid], returns[valid]


def collect_gait_demonstrations(
    runner: ParallelSimRunner,
    num_steps: int = 200,
    freq: float = 1.25,
    gamma: float = 0.99,
    return_privileged: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Collect closed-loop demonstrations using BipedalGaitGenerator."""
    if runner.obs_dim != NUM_OBSERVATIONS or runner.action_dim != NUM_ACTIONS:
        raise ValueError("V2 BC demonstrations require 42 observations and 8 actions")
    gait, dt = BipedalGaitGenerator(default_freq=freq), float(runner.control_dt)

    def policy_fn(_obs: np.ndarray, step: int) -> np.ndarray:
        q = gait.compute_walk(step * dt, freq=freq)
        raw_act = unscale_actions(q, runner.cfg, nominal_centered=True)
        return np.tile(raw_act, (runner.num_envs, 1))

    if return_privileged:
        return _collect_rollouts(runner, policy_fn, num_steps, gamma, return_privileged=True)
    return _collect_rollouts(runner, policy_fn, num_steps, gamma)


def _eval_actor_batch(actor: nn.Module, obs: np.ndarray) -> np.ndarray:
    """Evaluate a 42-obs -> 8-act PyTorch policy deterministically."""
    if obs.shape[1] != NUM_OBSERVATIONS:
        raise ValueError(f"V2 teacher observation mismatch: {obs.shape[1]} != {NUM_OBSERVATIONS}")
    norm_obs = obs
    bc_rms = getattr(actor, "_bc_obs_rms", None)
    if bc_rms is not None:
        norm_obs = bc_rms.normalize(obs)
    with torch.no_grad():
        t_obs = torch.as_tensor(norm_obs, dtype=torch.float32)
        out = actor.forward_actor(t_obs) if hasattr(actor, "forward_actor") else policy_actor_forward(actor, t_obs)
        if isinstance(out, tuple):
            out = out[0]
        act = out.detach().cpu().numpy()
    if act.shape[-1] != NUM_ACTIONS:
        raise ValueError(f"V2 teacher action mismatch: {act.shape[-1]} != {NUM_ACTIONS}")
    return np.clip(act, -1.0, 1.0).astype(np.float32)


def collect_teacher_demonstrations(
    runner: ParallelSimRunner,
    teacher_actor: nn.Module,
    num_steps: int = 200,
    gamma: float = 0.99,
    return_privileged: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Collect closed-loop demonstrations from a pretrained teacher policy."""
    teacher_actor.eval()
    in_dim = _get_actor_in_dim(teacher_actor)
    if runner.obs_dim != NUM_OBSERVATIONS or runner.action_dim != NUM_ACTIONS or in_dim != NUM_OBSERVATIONS:
        raise ValueError(f"V2 teacher demonstrations require a {NUM_OBSERVATIONS}-observation, {NUM_ACTIONS}-action policy")

    def policy_fn(obs: np.ndarray, _step: int) -> np.ndarray:
        return _eval_actor_batch(teacher_actor, obs)

    if return_privileged:
        return _collect_rollouts(runner, policy_fn, num_steps, gamma, return_privileged=True)
    return _collect_rollouts(runner, policy_fn, num_steps, gamma)


def collect_dagger_demonstrations(
    runner: ParallelSimRunner,
    teacher_actor: nn.Module,
    student_actor: Optional[nn.Module] = None,
    num_steps: int = 200,
    beta: float = 0.5,
    gamma: float = 0.99,
    return_privileged: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Collect DAgger trajectories (blended execution, teacher action labels)."""
    teacher_actor.eval()
    t_in_dim = _get_actor_in_dim(teacher_actor)
    if runner.obs_dim != NUM_OBSERVATIONS or runner.action_dim != NUM_ACTIONS or t_in_dim != NUM_OBSERVATIONS:
        raise ValueError(f"V2 teacher demonstrations require a {NUM_OBSERVATIONS}-observation, {NUM_ACTIONS}-action policy")
    if student_actor is not None:
        student_actor.eval()
        if _get_actor_in_dim(student_actor) != NUM_OBSERVATIONS:
            raise ValueError(f"V2 student demonstrations require a {NUM_OBSERVATIONS}-observation, {NUM_ACTIONS}-action policy")
    mix_beta = float(np.clip(beta, 0.0, 1.0))

    def policy_fn(obs: np.ndarray, _step: int) -> Tuple[np.ndarray, np.ndarray]:
        teacher_act = _eval_actor_batch(teacher_actor, obs)
        if student_actor is None or mix_beta >= 1.0:
            return teacher_act, teacher_act
        student_act = _eval_actor_batch(student_actor, obs)
        exec_act = np.clip(mix_beta * teacher_act + (1.0 - mix_beta) * student_act, -1.0, 1.0).astype(np.float32)
        return exec_act, teacher_act

    if return_privileged:
        return _collect_rollouts(runner, policy_fn, num_steps, gamma, return_privileged=True)
    return _collect_rollouts(runner, policy_fn, num_steps, gamma)


def _mirror_obs_by_dim(arr: np.ndarray) -> np.ndarray:
    dim = arr.shape[-1]
    if dim in (37, 42):
        return mirror_obs(arr)
    if dim in (48, 53):
        return mirror_critic_obs(arr)
    raise ValueError(f"Unsupported symmetry mirror dimension {dim}")


def train_behavior_cloning(
    obs: np.ndarray, actions: np.ndarray, returns: Optional[np.ndarray] = None,
    obs_dim: int = NUM_OBSERVATIONS, action_dim: int = NUM_ACTIONS, hidden_dim: int = 256,
    epochs: int = 30, batch_size: int = 256, lr: float = 1e-3, device: str = "cpu",
    obs_rms: Optional[RunningMeanStd] = None,
    augment_symmetry: bool = True,
    symmetry_coef: float = 0.25,
    init_model: Optional[ActorCritic] = None,
    critic_obs: Optional[np.ndarray] = None,
    critic_obs_dim: Optional[int] = None,
    critic_obs_rms: Optional[RunningMeanStd] = None,
    update_rms_slice: Optional[slice] = None,
) -> Tuple[ActorCritic, List[float]]:
    """Train Actor (MSE) and Critic (SmoothL1) to avoid cold critic shock at PPO iteration 1."""
    if obs_dim != NUM_OBSERVATIONS or action_dim != NUM_ACTIONS:
        raise ValueError(f"V2 behavior cloning requires {NUM_OBSERVATIONS} observations and {NUM_ACTIONS} actions")
    obs_raw = np.asarray(obs, dtype=np.float32)
    act_raw = np.asarray(actions, dtype=np.float32)
    if obs_raw.ndim != 2 or obs_raw.shape[1] != NUM_OBSERVATIONS:
        raise ValueError(f"V2 behavior cloning input must have shape [samples, {NUM_OBSERVATIONS}]")
    if act_raw.ndim != 2 or act_raw.shape[1] != NUM_ACTIONS:
        raise ValueError(f"V2 behavior cloning actions must have shape [samples, {NUM_ACTIONS}]")

    priv_raw = np.asarray(critic_obs, dtype=np.float32) if critic_obs is not None else None
    eff_critic_dim = (
        int(priv_raw.shape[1]) if priv_raw is not None
        else (
            int(critic_obs_dim) if critic_obs_dim is not None
            else (int(init_model.critic_obs_dim) if init_model is not None else int(obs_dim))
        )
    )
    if priv_raw is not None and priv_raw.shape != (len(obs_raw), eff_critic_dim):
        raise ValueError(f"Expected critic_obs shape ({len(obs_raw)}, {eff_critic_dim}), got {priv_raw.shape}")

    if obs_rms is None:
        obs_rms = RunningMeanStd(shape=obs_dim, clip=10.0, min_std=0.10)
    rms_obs_new = obs_raw[update_rms_slice] if update_rms_slice is not None else obs_raw
    if len(rms_obs_new) > 0:
        if augment_symmetry:
            obs_rms.update(np.concatenate([rms_obs_new, _mirror_obs_by_dim(rms_obs_new)], axis=0))
        else:
            obs_rms.update(rms_obs_new)

    if priv_raw is not None or critic_obs_rms is not None or eff_critic_dim != obs_dim:
        if critic_obs_rms is None:
            critic_obs_rms = RunningMeanStd(shape=eff_critic_dim, clip=10.0, min_std=0.10)
        if priv_raw is not None:
            rms_priv_new = priv_raw[update_rms_slice] if update_rms_slice is not None else priv_raw
            if len(rms_priv_new) > 0:
                if augment_symmetry:
                    critic_obs_rms.update(np.concatenate([rms_priv_new, _mirror_obs_by_dim(rms_priv_new)], axis=0))
                else:
                    critic_obs_rms.update(rms_priv_new)

    if augment_symmetry:
        obs_train, actions_train, returns_train = augment_symmetry_batch(obs_raw, act_raw, returns)
        priv_train = (
            np.concatenate([priv_raw, _mirror_obs_by_dim(priv_raw)], axis=0)
            if priv_raw is not None
            else None
        )
    else:
        obs_train, actions_train, returns_train = obs_raw, act_raw, returns
        priv_train = priv_raw

    model = (
        init_model.to(device)
        if init_model is not None
        else ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            critic_obs_dim=eff_critic_dim,
        ).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    norm_obs = obs_rms.normalize(obs_train)
    model._bc_obs_rms = obs_rms
    if critic_obs_rms is not None:
        model._bc_critic_obs_rms = critic_obs_rms

    obs_t = torch.as_tensor(norm_obs, dtype=torch.float32, device=device)
    if priv_train is not None and critic_obs_rms is not None:
        norm_priv = critic_obs_rms.normalize(priv_train)
        priv_t = torch.as_tensor(norm_priv, dtype=torch.float32, device=device)
    elif int(model.critic_obs_dim) == obs_dim:
        priv_t = obs_t
    else:
        priv_t = None
    act_t = torch.as_tensor(actions_train, dtype=torch.float32, device=device)
    ret_t = torch.as_tensor(returns_train, dtype=torch.float32, device=device) if returns_train is not None else None

    dataset_size, epoch_losses = len(obs_train), []
    sym_w = float(symmetry_coef)
    for _ in range(epochs):
        perm, total_loss, batches = torch.randperm(dataset_size, device=device), 0.0, 0
        for i in range(0, dataset_size, batch_size):
            idx = perm[i : i + batch_size]
            b_obs, b_act = obs_t[idx], act_t[idx]
            loss = F.mse_loss(model.forward_actor(b_obs), b_act)
            if sym_w > 0.0 and obs_dim == NUM_OBSERVATIONS and action_dim == NUM_ACTIONS:
                loss = loss + sym_w * compute_actor_symmetry_loss(model.forward_actor, b_obs)
            if ret_t is not None and priv_t is not None:
                loss = loss + 0.5 * F.smooth_l1_loss(model.forward_critic(priv_t[idx]), ret_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        epoch_losses.append(total_loss / max(1, batches))
    return model, epoch_losses


def save_bc_checkpoint(
    path: Union[str, Path],
    model: ActorCritic,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    obs_rms: Optional[RunningMeanStd] = None,
    critic_obs_dim: Optional[int] = None,
    critic_obs_rms: Optional[RunningMeanStd] = None,
) -> Path:
    """Save ActorCritic weights into a PPO-compatible warm start checkpoint."""
    if obs_dim != NUM_OBSERVATIONS or action_dim != NUM_ACTIONS:
        raise ValueError(f"V2 BC checkpoints require {NUM_OBSERVATIONS} observations and {NUM_ACTIONS} actions")
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eff_rms = obs_rms if obs_rms is not None else getattr(model, "_bc_obs_rms", None)
    eff_c_dim = int(critic_obs_dim) if critic_obs_dim is not None else int(getattr(model, "critic_obs_dim", obs_dim))
    eff_c_rms = critic_obs_rms if critic_obs_rms is not None else getattr(model, "_bc_critic_obs_rms", None)
    payload = {
        "actor_state_dict": model.actor.state_dict(),
        "critic_state_dict": model.critic.state_dict(),
        "obs_dim": int(obs_dim),
        "critic_obs_dim": eff_c_dim,
        "action_dim": int(action_dim),
        "hidden_dim": int(hidden_dim),
        "squash_actions": bool(model.squash_actions),
        "stage": 0,
        "iterations": 0,
        "model_path": str(MODEL_PATH),
        "model_name": MODEL_NAME,
        "observation_schema": observation_schema(NUM_OBSERVATIONS, True, CONTROL_DT_S),
    }
    if eff_rms is not None:
        payload["obs_rms"] = eff_rms.state_dict()
    if eff_c_rms is not None:
        payload["critic_obs_rms"] = eff_c_rms.state_dict()
    torch.save(payload, str(out_path))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Closed-Loop BC Pretraining & Critic Warmup")
    parser.add_argument("--mode", choices=["gait", "teacher"], default="gait")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dagger-rounds", type=int, default=1)
    parser.add_argument("--no-symmetry", action="store_true")
    parser.add_argument("--output", default=str(REPO_ROOT / "models" / "checkpoints" / f"{CHECKPOINT_PREFIX}_bc_init.pt"))
    args = parser.parse_args()

    use_sym = not args.no_symmetry
    sym_coef = 0.25 if use_sym else 0.0
    runner = ParallelSimRunner(model_path=args.model_path, num_envs=args.num_envs, num_threads=1, stage=0)
    try:
        if args.mode == "gait":
            obs, priv, acts, rets = collect_gait_demonstrations(runner, num_steps=args.steps, return_privileged=True)
            model, losses = train_behavior_cloning(
                obs, acts, returns=rets, obs_dim=runner.obs_dim, action_dim=runner.action_dim,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                augment_symmetry=use_sym, symmetry_coef=sym_coef,
                critic_obs=priv, critic_obs_dim=runner.privileged_obs_dim,
            )
        else:
            if not args.teacher_checkpoint:
                raise ValueError("--teacher-checkpoint is required when --mode=teacher")
            teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(teacher_checkpoint, dict):
                raise ValueError("V2 teacher checkpoint identity metadata is required")
            validate_model_identity(teacher_checkpoint, MODEL_NAME)
            teacher = load_actor_from_checkpoint(args.teacher_checkpoint)
            obs, priv, acts, rets = collect_teacher_demonstrations(runner, teacher, num_steps=args.steps, return_privileged=True)
            model, losses = train_behavior_cloning(
                obs, acts, returns=rets, obs_dim=runner.obs_dim, action_dim=runner.action_dim,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                augment_symmetry=use_sym, symmetry_coef=sym_coef,
                critic_obs=priv, critic_obs_dim=runner.privileged_obs_dim,
            )
            rounds = max(1, int(args.dagger_rounds))
            for r in range(1, rounds):
                beta = max(0.2, 1.0 - 0.5 * (r / max(1, rounds - 1)))
                d_obs, d_priv, d_acts, d_rets = collect_dagger_demonstrations(
                    runner, teacher, student_actor=model, num_steps=args.steps, beta=beta, return_privileged=True
                )
                prev_len = len(obs)
                obs = np.concatenate([obs, d_obs], axis=0)
                priv = np.concatenate([priv, d_priv], axis=0)
                acts = np.concatenate([acts, d_acts], axis=0)
                rets = np.concatenate([rets, d_rets], axis=0)
                model, losses = train_behavior_cloning(
                    obs, acts, returns=rets, obs_dim=runner.obs_dim, action_dim=runner.action_dim,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    obs_rms=model._bc_obs_rms, augment_symmetry=use_sym, symmetry_coef=sym_coef,
                    init_model=model, critic_obs=priv, critic_obs_dim=runner.privileged_obs_dim,
                    critic_obs_rms=getattr(model, "_bc_critic_obs_rms", None),
                    update_rms_slice=slice(prev_len, len(obs)),
                )
    finally:
        runner.close()

    out_path = save_bc_checkpoint(args.output, model, runner.obs_dim, runner.action_dim)
    print(f"Pretrained BC model saved to {out_path} (final loss: {losses[-1]:.4f})")


if __name__ == "__main__":
    main()
