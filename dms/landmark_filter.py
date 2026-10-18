"""Speed-adaptive, frame-rate independent EMA over face landmarks (One Euro filter family).

Mirrors src/vision/landmarkFilter.ts: alpha comes from the real time between
frames, and each point's cutoff grows with its own smoothed speed (normalized
by the inter-ocular distance), so a still face loses its jitter while head
turns and iris saccades keep up with no perceptible lag.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
MAX_GAP_S = 0.5
FALLBACK_DT_S = 1.0 / 30.0


def ema_alpha(cutoff_hz, dt_s: float):
    """EMA weight of a new sample for a first-order low-pass at ``cutoff_hz`` after ``dt_s`` seconds."""
    return 1.0 - np.exp(-2.0 * math.pi * np.asarray(cutoff_hz, dtype=np.float64) * dt_s)


class AdaptiveLandmarkFilter:
    def __init__(
        self,
        min_cutoff_hz: float = 1.7,
        speed_beta: float = 4.0,
        derivative_cutoff_hz: float = 1.0,
        snap_distance: float = 1.0,
        jitter_speed: float = 0.35,
    ) -> None:
        if min_cutoff_hz <= 0 or derivative_cutoff_hz <= 0 or speed_beta < 0:
            raise ValueError("cutoffs must be positive and speed_beta non-negative")
        self.min_cutoff_hz = min_cutoff_hz
        self.speed_beta = speed_beta
        self.derivative_cutoff_hz = derivative_cutoff_hz
        self.snap_distance = snap_distance
        # Speeds below this (inter-ocular distances / s) are jitter, not motion.
        self.jitter_speed = jitter_speed
        self._value: Optional[np.ndarray] = None
        self._speed: Optional[np.ndarray] = None
        self._last_t: Optional[float] = None

    def reset(self) -> None:
        self._value = None
        self._speed = None
        self._last_t = None

    def update(self, points: np.ndarray, timestamp_s: float) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        raw_dt = None if self._last_t is None else timestamp_s - self._last_t
        self._last_t = timestamp_s
        previous = self._value
        if previous is None or self._speed is None or previous.shape != points.shape or raw_dt is None:
            return self._snap(points)
        if raw_dt > MAX_GAP_S:
            return self._snap(points)
        dt = raw_dt if raw_dt > 0 else FALLBACK_DT_S

        scale = _inter_ocular(points)
        moves = np.linalg.norm(points[:, :2] - previous[:, :2], axis=1)
        if float(np.median(moves)) > self.snap_distance * scale:
            return self._snap(points)

        raw_speed = moves / dt / scale
        self._speed = self._speed + ema_alpha(self.derivative_cutoff_hz, dt) * (raw_speed - self._speed)
        alpha = ema_alpha(self.min_cutoff_hz + self.speed_beta * np.maximum(0.0, self._speed - self.jitter_speed), dt)[
            :, None
        ]
        self._value = previous + alpha * (points - previous)
        return self._value.astype(np.float32)

    def _snap(self, points: np.ndarray) -> np.ndarray:
        self._value = points.copy()
        self._speed = np.zeros(len(points), dtype=np.float64)
        return points.astype(np.float32)


def _inter_ocular(points: np.ndarray) -> float:
    if len(points) > RIGHT_EYE_OUTER:
        span = float(np.linalg.norm(points[LEFT_EYE_OUTER, :2] - points[RIGHT_EYE_OUTER, :2]))
        if span > 1.0:
            return span
    return max(1.0, float(np.ptp(points[:, 0])) * 0.45)
