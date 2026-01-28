from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ExponentialSmoother:
    def __init__(self, alpha: float, initial: Optional[float] = None) -> None:
        self.alpha = alpha
        self.value = initial

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1.0 - self.alpha) * new_value
        return self.value


class FPSCounter:
    def __init__(self, window_size: int = 30) -> None:
        self.times: Deque[float] = deque(maxlen=window_size)

    def update(self) -> float:
        now = time.time()
        self.times.append(now)
        if len(self.times) < 2:
            return 0.0
        duration = self.times[-1] - self.times[0]
        if duration == 0:
            return 0.0
        return (len(self.times) - 1) / duration


def point_distance(p1, p2) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
