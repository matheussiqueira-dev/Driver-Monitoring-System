"""Time-weighted share of a boolean signal over a sliding window.

Used for PERCLOS (eyes closed) and Euro NCAP VATS (gaze off-road). Mirrors
src/domain/rolling.ts operation by operation, so both runtimes produce the
same floating point results (checked by the shared scoring scenarios).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class _Sample:
    t: float
    active: bool
    # Contribution of the interval that ends at this sample.
    span: float
    active_span: float


class RollingDutyCycle:
    def __init__(self, window_s: float, max_gap_s: float) -> None:
        self.window_s = window_s
        self.max_gap_s = max_gap_s
        self._samples: Deque[_Sample] = deque()
        self._total = 0.0
        self._active = 0.0

    def reset(self) -> None:
        self._samples.clear()
        self._total = 0.0
        self._active = 0.0

    def add(self, timestamp: float, active: bool) -> None:
        span = 0.0
        active_span = 0.0
        if self._samples:
            previous = self._samples[-1]
            # A long gap (face lost, dropped frames) only counts up to max_gap_s.
            span = min(max(timestamp - previous.t, 0.0), self.max_gap_s)
            active_span = span if previous.active else 0.0
        self._samples.append(_Sample(timestamp, active, span, active_span))
        self._total += span
        self._active += active_span

        while len(self._samples) > 1 and self._samples[0].t < timestamp - self.window_s:
            self._samples.popleft()
            first = self._samples[0]
            self._total = max(self._total - first.span, 0.0)
            self._active = max(self._active - first.active_span, 0.0)
            first.span = 0.0
            first.active_span = 0.0

    @property
    def covered_s(self) -> float:
        return self._total

    @property
    def active_s(self) -> float:
        return self._active

    def ratio(self, min_coverage_s: float) -> Optional[float]:
        """Active share of the window, or None until enough time has been observed."""
        if self._total <= 0.0 or self._total < min_coverage_s:
            return None
        return self._active / self._total
