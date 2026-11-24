"""Off-Policy FastSAC Training Pipeline for CyberDuck V2 (ELU/ReLU + N-Step + Symmetry)."""

import argparse
from collections import deque
import math
import sys
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

_SRC_ROOT = str(Path(__file__).resolve().parents[2])
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from robot.robot_spec import CHECKPOINT_PREFIX, MODEL_NAME, MODEL_PATH, REPO_ROOT
from sim.env.policy_contract import observation_schema, validate_model_identity
from sim.env.run_parallel_sim import ParallelSimRunner, resolve_v2_model_path
from sim.env.symmetry import compute_actor_symmetry_loss, mirror_actions_8, mirror_obs_42
from tools.train.train_ppo import RunningMeanStd

DEFAULT_SAVE_PATH = REPO_ROOT / "models" / "checkpoints" / f"{CHECKPOINT_PREFIX}_sac.pt"


def _resolve_activation(activation: str) -> type:
    if activation not in ("relu", "elu"):
        raise ValueError(f"Unsupported SAC activation: {activation!r}; expected 'relu' or 'elu'")
    return nn.ELU if activation == "elu" else nn.ReLU


class SquashedGaussianActor(nn.Module):
    """Squashed Gaussian actor supporting both ReLU and ESP32-INT8-compatible ELU."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, activation: str = "relu"):
        super().__init__()
        act_cls = _resolve_activation(activation)
        self.activation = activation
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), act_cls(),
            nn.Linear(hidden_dim, hidden_dim), act_cls(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mean_head(self.net(obs)))

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(obs)
        mean = self.mean_head(feat)
        log_std = torch.clamp(self.log_std_head(feat), -20.0, 2.0)
        normal = Normal(mean, log_std.exp())
        u = normal.rsample()
        action = torch.tanh(u)
        log_prob = (normal.log_prob(u) - 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(dim=-1, keepdim=True)
        return action, log_prob


class TwinQCritic(nn.Module):
    """Twin-Q critic network for SAC."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, activation: str = "relu"):
        super().__init__()
        act_cls = _resolve_activation(activation)
        self.activation = activation
        mlp = lambda: nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), act_cls(),
            nn.Linear(hidden_dim, hidden_dim), act_cls(),
            nn.Linear(hidden_dim, 1),
        )
        self.q1, self.q2 = mlp(), mlp()

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([obs, act], dim=-1)
        return self.q1(xu), self.q2(xu)


class ReplayBuffer:
    """Ring replay buffer with optional N-step discounted return accumulation."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 100_000, n_step: int = 1, gamma: float = 0.99):
        self.capacity, self.ptr, self.size = capacity, 0, 0
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self._env_queues: Optional[List[Deque[Tuple[np.ndarray, np.ndarray, float]]]] = None

    def _push_single(self, obs_i: np.ndarray, act_i: np.ndarray, rew_i: float, next_obs_i: np.ndarray, done_i: float) -> None:
        idx = self.ptr
        self.obs[idx], self.act[idx] = obs_i, act_i
        self.rew[idx, 0] = float(rew_i)
        self.next_obs[idx] = next_obs_i
        self.done[idx, 0] = float(done_i)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _emit_nstep(self, queue: Deque[Tuple[np.ndarray, np.ndarray, float]], next_obs_i: np.ndarray, done_i: float) -> None:
        ret = 0.0
        discount = 1.0
        for _, _, r in queue:
            ret += discount * r
            discount *= self.gamma
        obs_0, act_0, _ = queue.popleft()
        self._push_single(obs_0, act_0, ret, next_obs_i, done_i)

    def add_batch(self, obs, act, rew, next_obs, done, episode_ends: Optional[np.ndarray] = None):
        n = len(obs)
        if self.n_step <= 1:
            idx = (self.ptr + np.arange(n)) % self.capacity
            self.obs[idx], self.act[idx] = obs, act
            self.rew[idx] = np.asarray(rew, dtype=np.float32).reshape(-1, 1)
            self.next_obs[idx] = next_obs
            self.done[idx] = np.asarray(done, dtype=np.float32).reshape(-1, 1)
            self.ptr = (self.ptr + n) % self.capacity
            self.size = min(self.size + n, self.capacity)
            return

        if self._env_queues is None or len(self._env_queues) != n:
            self._env_queues = [deque() for _ in range(n)]
        rew_arr = np.asarray(rew, dtype=np.float32).ravel()
        done_arr = np.asarray(done, dtype=np.float32).ravel()
        end_arr = np.asarray(episode_ends, dtype=bool).ravel() if episode_ends is not None else (done_arr > 0.5)
        for e in range(n):
            q = self._env_queues[e]
            q.append((np.array(obs[e], dtype=np.float32, copy=True), np.array(act[e], dtype=np.float32, copy=True), float(rew_arr[e])))
            if len(q) == self.n_step:
                self._emit_nstep(q, next_obs[e], float(done_arr[e]))
            if end_arr[e]:
                if done_arr[e] > 0.5:
                    while len(q) > 0:
                        self._emit_nstep(q, next_obs[e], 1.0)
                else:
                    q.clear()

    def sample(self, batch_size: int, device: str = "cpu") -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda arr: torch.as_tensor(arr[idx], dtype=torch.float32, device=device)
        return to_t(self.obs), to_t(self.act), to_t(self.rew).view(-1, 1), to_t(self.next_obs), to_t(self.done).view(-1, 1)


def sac_update_step(
    actor: SquashedGaussianActor, critic: TwinQCritic, target_critic: TwinQCritic,
    log_alpha: torch.Tensor, batch: Tuple[torch.Tensor, ...],
    optimizers: Dict[str, torch.optim.Optimizer], target_entropy: float,
    gamma: float = 0.99, tau: float = 0.005,
    symmetry_coef: float = 0.0, augment_symmetry: bool = False, n_step: int = 1,
) -> Dict[str, float]:
    obs, act, rew, next_obs, done = batch
    rew, done = rew.view(-1, 1), done.view(-1, 1)
    can_mirror = (obs.shape[-1] == 42 and act.shape[-1] == 8)
    if augment_symmetry and can_mirror:
        obs = torch.cat([obs, mirror_obs_42(obs)], dim=0)
        act = torch.cat([act, mirror_actions_8(act)], dim=0)
        rew = torch.cat([rew, rew], dim=0)
        next_obs = torch.cat([next_obs, mirror_obs_42(next_obs)], dim=0)
        done = torch.cat([done, done], dim=0)

    discount = float(gamma) ** max(1, int(n_step))
    with torch.no_grad():
        next_act, next_log_prob = actor.sample(next_obs)
        t_q1, t_q2 = target_critic(next_obs, next_act)
        alpha = log_alpha.exp()
        target_q = rew + (1.0 - done) * discount * (torch.min(t_q1, t_q2) - alpha * next_log_prob)

    q1, q2 = critic(obs, act)
    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
    optimizers["critic"].zero_grad()
    critic_loss.backward()
    optimizers["critic"].step()

    sym_loss_val = 0.0
    critic.requires_grad_(False)
    try:
        pi_act, log_prob = actor.sample(obs)
        q1_pi, q2_pi = critic(obs, pi_act)
        actor_loss = (alpha.detach() * log_prob - torch.min(q1_pi, q2_pi)).mean()
        if float(symmetry_coef) > 0.0 and can_mirror:
            sym_loss = compute_actor_symmetry_loss(actor, obs)
            sym_loss_val = float(sym_loss.item())
            actor_loss = actor_loss + float(symmetry_coef) * sym_loss
        optimizers["actor"].zero_grad()
        actor_loss.backward()
        optimizers["actor"].step()
    finally:
        critic.requires_grad_(True)

    alpha_loss = -(log_alpha * (log_prob.detach() + target_entropy)).mean()
    optimizers["alpha"].zero_grad()
    alpha_loss.backward()
    optimizers["alpha"].step()
    log_alpha.data.clamp_(-10.0, 2.0)

    with torch.no_grad():
        for p, pt in zip(critic.parameters(), target_critic.parameters()):
            pt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    return {
        "critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item()),
        "alpha_loss": float(alpha_loss.item()), "alpha": float(alpha.item()),
        "symmetry_loss": sym_loss_val,
    }


def _load_sac_warmstart(
    actor: SquashedGaussianActor,
    load_checkpoint: Union[str, Path],
    obs_dim: int,
    act_dim: int,
    hidden_dim: int,
    device: str,
) -> Optional[RunningMeanStd]:
    ckpt_path = Path(load_checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAC warmstart checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("V2 checkpoint model identity metadata is required")
    validate_model_identity(ckpt, MODEL_NAME)

    loaded_rms: Optional[RunningMeanStd] = None
    if "obs_rms" in ckpt and ckpt["obs_rms"] is not None:
        loaded_rms = RunningMeanStd(shape=obs_dim, clip=10.0)
        loaded_rms.load_state_dict(ckpt["obs_rms"])

    a_sd = ckpt.get("actor_state_dict")
    m_sd = ckpt.get("model_state_dict")
    with torch.no_grad():
        if isinstance(a_sd, dict) and "net.0.weight" in a_sd:
            if a_sd["net.0.weight"].shape != (hidden_dim, obs_dim) or a_sd["mean_head.weight"].shape != (act_dim, hidden_dim):
                raise ValueError("SAC warmstart checkpoint dimensions disagree")
            actor.load_state_dict(a_sd, strict=False)
        else:
            src = a_sd if isinstance(a_sd, dict) and "0.weight" in a_sd else (
                {k.replace("actor.", ""): v for k, v in m_sd.items() if k.startswith("actor.")}
                if isinstance(m_sd, dict) else None
            )
            if not src or "0.weight" not in src or "2.weight" not in src or "4.weight" not in src:
                raise ValueError("Incompatible actor weights for SAC warmstart")
            if src["0.weight"].shape != (hidden_dim, obs_dim) or src["4.weight"].shape != (act_dim, hidden_dim):
                raise ValueError("Warmstart checkpoint actor dimensions disagree")
            actor.net[0].weight.copy_(src["0.weight"])
            actor.net[0].bias.copy_(src["0.bias"])
            actor.net[2].weight.copy_(src["2.weight"])
            actor.net[2].bias.copy_(src["2.bias"])
            actor.mean_head.weight.copy_(src["4.weight"])
            actor.mean_head.bias.copy_(src["4.bias"])
    return loaded_rms


def save_sac_checkpoint(
    path: Union[str, Path], actor: SquashedGaussianActor, log_alpha: torch.Tensor,
    obs_dim: int, act_dim: int, hidden_dim: int = 256,
    model_path: Optional[Union[str, Path]] = None,
    observation_schema: Optional[dict] = None,
    obs_rms: Optional[Any] = None,
    activation: Optional[str] = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    act_name = activation or getattr(actor, "activation", "relu")
    _resolve_activation(act_name)
    policy_arch = "sac_elu_squashed_gaussian" if act_name == "elu" else "sac_squashed_gaussian"
    payload = {
        "actor_state_dict": actor.state_dict(), "sac_alpha": float(log_alpha.exp().item()),
        "obs_dim": int(obs_dim), "action_dim": int(act_dim), "hidden_dim": int(hidden_dim),
        "policy_arch": policy_arch, "activation": act_name, "algorithm": "sac", "squash_actions": True,
    }
    if obs_rms is not None:
        payload["obs_rms"] = obs_rms.state_dict() if hasattr(obs_rms, "state_dict") else dict(obs_rms)
    if model_path is not None:
        source_model_path = resolve_v2_model_path(model_path)
        payload["model_path"] = str(MODEL_PATH)
        payload["model_name"] = MODEL_NAME
        payload["source_model_path"] = str(source_model_path)
    if observation_schema is not None:
        payload["observation_schema"] = dict(observation_schema)
    torch.save(payload, p)
    return p


def train_sac(
    model_path: Union[str, Path] = MODEL_PATH,
    num_envs: int = 16, total_steps: int = 100_000, warmup_steps: int = 1_000,
    batch_size: int = 256, hidden_dim: int = 256,
    save_path: Union[str, Path] = DEFAULT_SAVE_PATH,
    lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
    buffer_capacity: int = 100_000, device: str = "cpu",
    target_entropy: Optional[float] = None,
    activation: str = "relu",
    n_step: int = 3,
    symmetry_coef: float = 0.25,
    augment_symmetry: bool = True,
    use_obs_rms: bool = False,
    load_checkpoint: Optional[Union[str, Path]] = None,
    max_action_delay: int = 0,
) -> Path:
    _resolve_activation(activation)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model_path = resolve_v2_model_path(model_path)
    runner = ParallelSimRunner(model_path=model_path, num_envs=num_envs, max_action_delay_steps=max_action_delay)
    try:
        obs_dim, act_dim = runner.obs_dim, runner.action_dim
        t_entropy = -float(act_dim) if target_entropy is None else float(target_entropy)
        actor = SquashedGaussianActor(obs_dim, act_dim, hidden_dim, activation=activation).to(device)
        critic = TwinQCritic(obs_dim, act_dim, hidden_dim, activation=activation).to(device)
        target_critic = TwinQCritic(obs_dim, act_dim, hidden_dim, activation=activation).to(device)
        target_critic.load_state_dict(critic.state_dict())
        log_alpha = torch.zeros(1, requires_grad=True, device=device)

        obs_rms: Optional[RunningMeanStd] = RunningMeanStd(shape=obs_dim, clip=10.0) if use_obs_rms else None
        if load_checkpoint is not None:
            loaded_rms = _load_sac_warmstart(actor, load_checkpoint, obs_dim, act_dim, hidden_dim, device)
            if loaded_rms is not None:
                obs_rms = loaded_rms

        optimizers = {
            "actor": torch.optim.Adam(actor.parameters(), lr=lr),
            "critic": torch.optim.Adam(critic.parameters(), lr=lr),
            "alpha": torch.optim.Adam([log_alpha], lr=lr),
        }
        buf = ReplayBuffer(obs_dim, act_dim, capacity=buffer_capacity, n_step=n_step, gamma=gamma)
        obs, collected = runner.reset(), 0

        def prep_obs(raw_obs: np.ndarray, update_stats: bool = True) -> np.ndarray:
            if obs_rms is None:
                return raw_obs
            if update_stats:
                if augment_symmetry and obs_dim == 42:
                    obs_rms.update(np.concatenate([raw_obs, mirror_obs_42(raw_obs)], axis=0))
                else:
                    obs_rms.update(raw_obs)
            return obs_rms.normalize(raw_obs)

        norm_obs = prep_obs(obs, update_stats=True)
        while collected < total_steps:
            if collected < warmup_steps:
                acts = np.random.uniform(-1.0, 1.0, size=(num_envs, act_dim)).astype(np.float32)
            else:
                with torch.no_grad():
                    acts = actor.sample(torch.as_tensor(norm_obs, dtype=torch.float32, device=device))[0].cpu().numpy()
            next_obs, rews, dones, fast_info = runner.step_arrays(acts)
            time_outs = np.asarray(fast_info.get("time_outs", np.zeros_like(dones, dtype=bool)), dtype=bool)
            true_dones = np.logical_and(dones, ~time_outs)
            buf_next_obs = next_obs.copy()
            if np.any(time_outs) and len(fast_info.get("terminal_obs", [])) == int(np.sum(time_outs)):
                buf_next_obs[time_outs] = fast_info["terminal_obs"]
            norm_buf_next = prep_obs(buf_next_obs, update_stats=False)
            buf.add_batch(norm_obs, acts, rews, norm_buf_next, true_dones, episode_ends=dones)
            obs, collected = next_obs, collected + num_envs
            norm_obs = prep_obs(obs, update_stats=True)
            if collected >= warmup_steps and buf.size >= batch_size:
                sac_update_step(
                    actor, critic, target_critic, log_alpha,
                    buf.sample(batch_size, device=device), optimizers, t_entropy,
                    gamma=gamma, tau=tau, symmetry_coef=symmetry_coef,
                    augment_symmetry=augment_symmetry, n_step=n_step,
                )
        schema = observation_schema(obs_dim, runner.embed_phase_clock, runner.control_dt)
        return save_sac_checkpoint(
            save_path, actor, log_alpha, obs_dim, act_dim, hidden_dim,
            model_path=model_path, observation_schema=schema, obs_rms=obs_rms, activation=activation,
        )
    finally:
        runner.close()


def main():
    parser = argparse.ArgumentParser(description="FastSAC training for CyberDuck V2")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--save-path", default=str(DEFAULT_SAVE_PATH))
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--activation", choices=["relu", "elu"], default="elu")
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--symmetry-coef", type=float, default=0.25)
    parser.add_argument("--no-symmetry", action="store_true")
    parser.add_argument("--use-obs-rms", action="store_true")
    parser.add_argument("--load-checkpoint", type=str, default=None)
    parser.add_argument("--max-action-delay", type=int, default=0)
    args = parser.parse_args()
    train_sac(
        model_path=args.model_path, num_envs=args.num_envs,
        total_steps=args.total_steps, warmup_steps=args.warmup_steps,
        batch_size=args.batch_size, hidden_dim=args.hidden_dim,
        save_path=args.save_path, target_entropy=args.target_entropy,
        device=args.device, activation=args.activation, n_step=args.n_step,
        symmetry_coef=0.0 if args.no_symmetry else args.symmetry_coef,
        augment_symmetry=not args.no_symmetry, use_obs_rms=args.use_obs_rms,
        load_checkpoint=args.load_checkpoint, max_action_delay=args.max_action_delay,
    )


if __name__ == "__main__":
    main()
