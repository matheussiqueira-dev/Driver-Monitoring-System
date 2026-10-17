"""Session aggregation. Mirrors src/domain/session.ts SessionStats.

An event counts once when it starts (not on every frame); the score trend is
sampled every ``sample_interval_s``; time is accumulated per risk level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .attention import AttentionState
from .config import DMSConfig

DEFAULT_SAMPLE_INTERVAL_S = 0.7
DEFAULT_HISTORY_SIZE = 24
MAX_ONSETS = 500


def level_for_label(label: str) -> str:
    return "ok" if label == "Atento" else "warn" if label == "Alerta" else "danger"


@dataclass(frozen=True)
class EventOnset:
    type: str
    at_s: float


@dataclass
class SessionSummary:
    duration_s: float
    average_score: Optional[float]
    min_score: Optional[float]
    event_count: int
    events_by_type: Dict[str, int]
    time_by_level_s: Dict[str, float] = field(default_factory=dict)


class SessionStats:
    def __init__(
        self, sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S, history_size: int = DEFAULT_HISTORY_SIZE
    ) -> None:
        self.sample_interval_s = sample_interval_s
        self.history_size = history_size
        self.reset()

    def reset(self) -> None:
        self._start: Optional[float] = None
        self._last: Optional[float] = None
        self._last_level: Optional[str] = None
        self._active: set[str] = set()
        self._onsets: List[EventOnset] = []
        self._trend: List[float] = []
        self._last_trend = float("-inf")
        self._score_sum = 0.0
        self._score_count = 0
        self._min_score: Optional[float] = None
        self._level_time: Dict[str, float] = {"ok": 0.0, "warn": 0.0, "danger": 0.0}

    def add(self, state: AttentionState, timestamp: float) -> List[str]:
        """Returns the events that started on this sample."""
        if self._start is None:
            self._start = timestamp
        if self._last is not None and self._last_level is not None and timestamp > self._last:
            self._level_time[self._last_level] += timestamp - self._last
        self._last = timestamp
        self._last_level = level_for_label(state.label)

        started = [event for event in state.events if event not in self._active]
        for event in started:
            self._onsets.append(EventOnset(event, timestamp - self._start))
        if len(self._onsets) > MAX_ONSETS:
            del self._onsets[: len(self._onsets) - MAX_ONSETS]
        self._active = set(state.events)

        if timestamp - self._last_trend >= self.sample_interval_s:
            self._trend.append(state.score)
            if len(self._trend) > self.history_size:
                self._trend.pop(0)
            self._last_trend = timestamp
            self._score_sum += state.score
            self._score_count += 1
            self._min_score = state.score if self._min_score is None else min(self._min_score, state.score)
        return started

    @property
    def history(self) -> Sequence[float]:
        return tuple(self._trend)

    @property
    def onsets(self) -> Sequence[EventOnset]:
        return tuple(self._onsets)

    def summary(self) -> SessionSummary:
        events_by_type: Dict[str, int] = {}
        for onset in self._onsets:
            events_by_type[onset.type] = events_by_type.get(onset.type, 0) + 1
        duration = 0.0 if self._start is None or self._last is None else self._last - self._start
        return SessionSummary(
            duration_s=duration,
            average_score=None if self._score_count == 0 else self._score_sum / self._score_count,
            min_score=self._min_score,
            event_count=len(self._onsets),
            events_by_type=events_by_type,
            time_by_level_s=dict(self._level_time),
        )


def build_session_report(
    stats: SessionStats,
    config: DMSConfig,
    *,
    mode: str,
    generated_at: str,
    sensitivity: Optional[float] = None,
) -> Dict[str, object]:
    """Same JSON shape as src/domain/report.ts buildSessionReport (schemaVersion 1)."""
    summary = stats.summary()

    def r(value: float, digits: int = 3) -> float:
        return round(value, digits)

    return {
        "schemaVersion": 1,
        "generatedAt": generated_at,
        "source": {"mode": mode, "sensitivity": sensitivity},
        "thresholds": {
            "earThreshold": config.ear_threshold,
            "perclosThreshold": config.perclos_threshold,
            "marThreshold": config.mar_threshold,
            "yawThreshold": r(config.yaw_threshold),
            "pitchThreshold": r(config.pitch_threshold),
            "vatsOffroadLimitS": config.vats_offroad_limit_s,
            "vatsWindowS": config.vats_window_s,
        },
        "summary": {
            "durationS": r(summary.duration_s),
            "averageScore": None if summary.average_score is None else r(summary.average_score, 1),
            "minScore": None if summary.min_score is None else r(summary.min_score, 1),
            "eventCount": summary.event_count,
            "eventsByType": summary.events_by_type,
            "timeByLevelS": {level: r(seconds) for level, seconds in summary.time_by_level_s.items()},
        },
        "events": [{"type": onset.type, "atS": r(onset.at_s)} for onset in stats.onsets],
        "scoreHistory": [r(score, 1) for score in stats.history],
    }
