"""Mouth Aspect Ratio (MAR) and yawn tracking. Mirrors src/domain/mouthState.ts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from .config import DMSConfig
from .utils import point_distance

# MediaPipe Face Mesh inner-lip landmarks: three vertical pairs and the mouth corners.
MOUTH_VERTICAL_PAIRS = ((81, 178), (13, 14), (311, 402))
MOUTH_CORNERS = (78, 308)
YAWN_WINDOW_S = 60.0


def compute_mar(landmarks: np.ndarray) -> float:
    left, right = MOUTH_CORNERS
    horizontal = point_distance(landmarks[left][:2], landmarks[right][:2])
    if horizontal == 0:
        return 0.0
    vertical = sum(point_distance(landmarks[a][:2], landmarks[b][:2]) for a, b in MOUTH_VERTICAL_PAIRS)
    return vertical / (len(MOUTH_VERTICAL_PAIRS) * horizontal)


@dataclass
class MouthState:
    mar: Optional[float]
    mouth_open: bool
    yawning: bool
    open_duration: float
    yawn_count: float


class MouthStateTracker:
    """A yawn is a mouth opening above mar_threshold held for yawn_min_time_s; talking is shorter."""

    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self._open_start: Optional[float] = None
        self._counted = False
        self._yawn_times: Deque[float] = deque(maxlen=120)

    def reset(self) -> None:
        self._open_start = None
        self._counted = False
        self._yawn_times.clear()

    def update(self, mar: Optional[float], timestamp: float) -> MouthState:
        if mar is None or mar < self.config.mar_threshold:
            self._open_start = None
            self._counted = False
            return MouthState(
                mar=mar,
                mouth_open=False,
                yawning=False,
                open_duration=0.0,
                yawn_count=self._compute_yawn_count(timestamp),
            )

        if self._open_start is None:
            self._open_start = timestamp
        open_duration = timestamp - self._open_start
        yawning = open_duration >= self.config.yawn_min_time_s
        if yawning and not self._counted:
            self._counted = True
            self._yawn_times.append(timestamp)
        return MouthState(
            mar=mar,
            mouth_open=True,
            yawning=yawning,
            open_duration=open_duration,
            yawn_count=self._compute_yawn_count(timestamp),
        )

    def _compute_yawn_count(self, timestamp: float) -> float:
        while self._yawn_times and (timestamp - self._yawn_times[0]) > YAWN_WINDOW_S:
            self._yawn_times.popleft()
        return float(len(self._yawn_times))
