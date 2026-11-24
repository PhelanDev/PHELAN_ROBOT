"""Sagittal mirror symmetry transformations and losses for CyberDuck V2 (42D obs / 53D critic / 8D act)."""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

ArrayOrTensor = Union[np.ndarray, torch.Tensor]

ACT_MIRROR_PERM_8: Tuple[int, ...] = (0, 1, 5, 6, 7, 2, 3, 4)
ACT_MIRROR_SIGN_8: Tuple[float, ...] = (-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0)

OBS_MIRROR_PERM_42: Tuple[int, ...] = (
    0, 1, 2,                                        # base_ang_vel [wx, wy, wz]
    3, 4, 5,                                        # proj_grav [gx, gy, gz]
    6, 7, 8,                                        # commands [vx, vy, wz]
    *tuple(9 + i for i in ACT_MIRROR_PERM_8),       # joint_pos (8)
    *tuple(17 + i for i in ACT_MIRROR_PERM_8),      # joint_vel (8)
    *tuple(25 + i for i in ACT_MIRROR_PERM_8),      # last_actions (8)
    34, 33,                                         # foot_contact [left, right]
    35, 36,                                         # phase_clock [sin, cos]
    41, 40, 39, 38, 37,                             # lidar_ranges [m30, m15, 0, p15, p30]
)
OBS_MIRROR_SIGN_42: Tuple[float, ...] = (
    -1.0, 1.0, -1.0,                                # base_ang_vel
    1.0, -1.0, 1.0,                                 # proj_grav
    1.0, -1.0, -1.0,                                # commands
    *ACT_MIRROR_SIGN_8,                             # joint_pos
    *ACT_MIRROR_SIGN_8,                             # joint_vel
    *ACT_MIRROR_SIGN_8,                             # last_actions
    1.0, 1.0,                                       # foot_contact
    -1.0, -1.0,                                     # phase_clock shifted by pi
    1.0, 1.0, 1.0, 1.0, 1.0,                        # lidar_ranges
)

CRITIC_MIRROR_PERM_53: Tuple[int, ...] = (
    *OBS_MIRROR_PERM_42,
    42, 43, 44,                                     # base_lin_vel [vx, vy, vz]
    46, 45,                                         # contact_forces_z [left, right]
    48, 47,                                         # terrain_height_at_feet [left, right]
    49,                                             # nearest_obstacle_dist
    50, 51,                                         # nearest_obstacle_rel_xy [dx, dy]
    52,                                             # stage_norm
)
CRITIC_MIRROR_SIGN_53: Tuple[float, ...] = (
    *OBS_MIRROR_SIGN_42,
    1.0, -1.0, 1.0,                                 # base_lin_vel
    1.0, 1.0,                                       # contact_forces_z
    1.0, 1.0,                                       # terrain_height_at_feet
    1.0,                                            # nearest_obstacle_dist
    1.0, -1.0,                                      # nearest_obstacle_rel_xy
    1.0,                                            # stage_norm
)

_NP_SPECS: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
    8: (np.array(ACT_MIRROR_PERM_8, dtype=np.int64), np.array(ACT_MIRROR_SIGN_8, dtype=np.float32)),
    42: (np.array(OBS_MIRROR_PERM_42, dtype=np.int64), np.array(OBS_MIRROR_SIGN_42, dtype=np.float32)),
    53: (np.array(CRITIC_MIRROR_PERM_53, dtype=np.int64), np.array(CRITIC_MIRROR_SIGN_53, dtype=np.float32)),
}


def _apply_mirror(x: ArrayOrTensor, expected_dim: int) -> ArrayOrTensor:
    """Apply sagittal reflection permutation and sign flip along the last dimension."""
    perm_np, sign_np = _NP_SPECS[expected_dim]
    if isinstance(x, torch.Tensor):
        if x.shape[-1] != expected_dim:
            raise ValueError(f"Expected last dimension {expected_dim}, got {x.shape[-1]}")
        idx = torch.as_tensor(perm_np, dtype=torch.long, device=x.device)
        sign = torch.as_tensor(sign_np, dtype=x.dtype, device=x.device)
        return x.index_select(-1, idx) * sign
    arr = np.asarray(x)
    if arr.shape[-1] != expected_dim:
        raise ValueError(f"Expected last dimension {expected_dim}, got {arr.shape[-1]}")
    return (arr[..., perm_np] * sign_np.astype(arr.dtype, copy=False)).astype(arr.dtype, copy=False)


def mirror_actions_8(actions: ArrayOrTensor) -> ArrayOrTensor:
    """Reflect 8D CyberDuck V2 joint actions/positions across the sagittal (XZ) plane."""
    return _apply_mirror(actions, 8)


def mirror_obs_42(obs: ArrayOrTensor) -> ArrayOrTensor:
    """Reflect 42D CyberDuck V2 actor observations across the sagittal (XZ) plane."""
    return _apply_mirror(obs, 42)


def mirror_critic_obs_53(critic_obs: ArrayOrTensor) -> ArrayOrTensor:
    """Reflect 53D CyberDuck V2 privileged critic observations across the sagittal plane."""
    return _apply_mirror(critic_obs, 53)


def compute_actor_symmetry_loss(
    actor_fn: Callable[[torch.Tensor], torch.Tensor],
    obs_batch: torch.Tensor,
) -> torch.Tensor:
    """Compute sagittal mirror consistency loss MSE(pi(s), M_a(pi(M_s(s))))."""
    act_orig = actor_fn(obs_batch)
    act_mirrored_obs = actor_fn(mirror_obs_42(obs_batch))
    return F.mse_loss(act_orig, mirror_actions_8(act_mirrored_obs))


def augment_symmetry_batch(
    obs: np.ndarray,
    actions: np.ndarray,
    returns: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Double a batch of (obs, actions[, returns]) by appending sagittal mirror reflections."""
    obs_arr = np.asarray(obs, dtype=np.float32)
    act_arr = np.asarray(actions, dtype=np.float32)
    aug_obs = np.concatenate([obs_arr, mirror_obs_42(obs_arr)], axis=0)
    aug_act = np.concatenate([act_arr, mirror_actions_8(act_arr)], axis=0)
    if returns is None:
        return aug_obs, aug_act, None
    ret_arr = np.asarray(returns, dtype=np.float32)
    aug_ret = np.concatenate([ret_arr, ret_arr], axis=0)
    return aug_obs, aug_act, aug_ret
