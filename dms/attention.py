from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .config import DMSConfig
from .ear import EyeState
from .head_pose import HeadPose
from .utils import ExponentialSmoother, clamp


@dataclass
class AttentionState:
    score: float
    label: str
    events: List[str]
    raw_score: float


class AttentionScorer:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self._smoother = ExponentialSmoother(config.score_smoothing, config.score_base)
        self._offroad_start: Optional[float] = None
        self._last_face_time: Optional[float] = None

    def update(
        self,
        eye_state: EyeState,
        head_pose: Optional[HeadPose],
        phone_present: bool,
        phone_near_face: bool,
        face_present: bool,
        timestamp: float,
    ) -> AttentionState:
        events: List[str] = []
        penalty = 0.0

        if face_present:
            self._last_face_time = timestamp
        else:
            if self._last_face_time is None or (timestamp - self._last_face_time) > 1.0:
                events.append("Sem rosto")
                penalty += 25.0

        if eye_state.microsleep:
            penalty += self.config.penalty_microsleep
            events.append("Microssono")
        elif eye_state.drowsy:
            penalty += self.config.penalty_drowsy
            events.append("Sonolencia")

        offroad = False
        if head_pose is not None:
            if abs(head_pose.yaw) > self.config.yaw_threshold or abs(head_pose.pitch) > self.config.pitch_threshold:
                offroad = True
        if offroad:
            if self._offroad_start is None:
                self._offroad_start = timestamp
            elif (timestamp - self._offroad_start) >= self.config.offroad_time_s:
                penalty += self.config.penalty_offroad
                events.append("Olhar fora")
        else:
            self._offroad_start = None

        if phone_present or phone_near_face:
            penalty += self.config.penalty_phone
            events.append("Celular")

        bonus = self.config.bonus_stable if penalty == 0 else 0.0
        raw_score = self.config.score_base - penalty + bonus
        raw_score = clamp(raw_score, self.config.score_floor, self.config.score_ceiling)
        score = self._smoother.update(raw_score)
        score = clamp(score, self.config.score_floor, self.config.score_ceiling)

        if score >= 70:
            label = "Atento"
        elif score >= 40:
            label = "Alerta"
        else:
            label = "Distraido"

        return AttentionState(score=score, label=label, events=events, raw_score=raw_score)
