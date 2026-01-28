from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np

from .config import DMSConfig
from .utils import point_distance

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


def compute_ear(landmarks: np.ndarray, indices: Tuple[int, int, int, int, int, int]) -> float:
    p1, p2, p3, p4, p5, p6 = [landmarks[i][:2] for i in indices]
    vertical_1 = point_distance(p2, p6)
    vertical_2 = point_distance(p3, p5)
    horizontal = point_distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


@dataclass
class EyeState:
    ear: Optional[float]
    blink_rate: float
    eyes_closed: bool
    drowsy: bool
    microsleep: bool
    closed_duration: float


class EyeStateTracker:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self._blink_frames = 0
        self._closed_start: Optional[float] = None
        self._blink_times: Deque[float] = deque(maxlen=120)

    def update(self, ear: Optional[float], timestamp: float) -> EyeState:
        if ear is None:
            return EyeState(
                ear=None,
                blink_rate=self._compute_blink_rate(timestamp),
                eyes_closed=False,
                drowsy=False,
                microsleep=False,
                closed_duration=0.0,
            )

        if ear < self.config.ear_threshold:
            self._blink_frames += 1
            if self._closed_start is None:
                self._closed_start = timestamp
        else:
            if self._closed_start is not None:
                if self.config.blink_min_frames <= self._blink_frames <= self.config.blink_max_frames:
                    self._blink_times.append(timestamp)
            self._blink_frames = 0
            self._closed_start = None

        closed_duration = 0.0
        if self._closed_start is not None:
            closed_duration = timestamp - self._closed_start

        drowsy = closed_duration >= self.config.drowsy_time_s
        microsleep = closed_duration >= self.config.microsleep_time_s

        return EyeState(
            ear=ear,
            blink_rate=self._compute_blink_rate(timestamp),
            eyes_closed=self._closed_start is not None,
            drowsy=drowsy,
            microsleep=microsleep,
            closed_duration=closed_duration,
        )

    def _compute_blink_rate(self, timestamp: float) -> float:
        while self._blink_times and (timestamp - self._blink_times[0]) > 60.0:
            self._blink_times.popleft()
        return float(len(self._blink_times))
