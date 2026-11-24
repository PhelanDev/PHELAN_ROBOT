"""Active single-ToF head scanner building a 5-ray virtual obstacle sweep."""

import math
from typing import Optional, Tuple

import numpy as np

RAY_ANGLES_RAD: np.ndarray = np.array(
    [-math.pi / 6.0, -math.pi / 12.0, 0.0, math.pi / 12.0, math.pi / 6.0],
    dtype=np.float32,
)
_MAX_BIN_DELTA_RAD = math.radians(37.5)


class ActiveHeadScanner:
    """Projects a 1D brow ToF reading into a 5-sector horizontal memory buffer."""

    def __init__(
        self,
        ttl_s: float = 2.0,
        max_range_m: float = 2.0,
        floor_pitch_cutoff_rad: float = 0.35,
    ) -> None:
        self.ttl_s = float(ttl_s)
        self.max_range_m = float(max_range_m)
        self.floor_pitch_cutoff_rad = float(floor_pitch_cutoff_rad)
        self._ranges = np.full(5, self.max_range_m, dtype=np.float32)
        self._timestamps: list[Optional[float]] = [None] * 5

    def reset(self) -> None:
        self._ranges.fill(self.max_range_m)
        self._timestamps = [None] * 5

    def update(self, head_yaw_rad: float, head_pitch_rad: float, tof_range_m: float, now_s: float) -> None:
        yaw = float(head_yaw_rad)
        pitch = float(head_pitch_rad)
        if not (math.isfinite(yaw) and math.isfinite(pitch)) or pitch > self.floor_pitch_cutoff_rad:
            return
        diffs = np.abs(RAY_ANGLES_RAD - yaw)
        idx = int(np.argmin(diffs))
        if float(diffs[idx]) > _MAX_BIN_DELTA_RAD:
            return

        rng = float(tof_range_m)
        if not math.isfinite(rng) or rng < 0.0 or rng >= self.max_range_m:
            projected = self.max_range_m
        else:
            projected = min(self.max_range_m, max(0.0, rng * math.cos(pitch)))
        self._ranges[idx] = projected
        self._timestamps[idx] = float(now_s)

    def get_virtual_rays(self, now_s: float) -> np.ndarray:
        t = float(now_s)
        out = np.full(5, self.max_range_m, dtype=np.float32)
        for i, ts in enumerate(self._timestamps):
            if ts is not None and (t - ts) <= self.ttl_s:
                out[i] = self._ranges[i]
        return out

    def desired_head_angles(
        self,
        mode: str,
        now_s: float,
        target_norm_xy: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        if mode == "SCAN_NAV":
            return (float(0.52 * math.sin(2.0 * math.pi * 0.75 * float(now_s))), 0.0)
        if mode == "INTERACT":
            tx, ty = target_norm_xy if target_norm_xy is not None else (0.0, 0.0)
            return (float(np.clip(tx, -1.57, 1.57)), float(np.clip(ty, -0.52, 0.78)))
        return (0.0, 0.0)
