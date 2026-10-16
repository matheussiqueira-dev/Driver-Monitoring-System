from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from .config import DMSConfig
from .ear import EyeState
from .rolling import RollingDutyCycle
from .utils import ExponentialSmoother, clamp

if TYPE_CHECKING:
    from .head_pose import HeadPose
    from .mouth import MouthState


@dataclass
class AttentionState:
    score: float
    label: str
    events: List[str]
    raw_score: float
    # Seconds of off-road gaze within the last vats_window_s (Euro NCAP VATS).
    offroad_window_s: float = 0.0


class AttentionScorer:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self._smoother = ExponentialSmoother(config.score_smoothing, config.score_base)
        self._offroad_start: Optional[float] = None
        self._last_face_time: Optional[float] = None
        self._offroad_window = RollingDutyCycle(config.vats_window_s, config.rolling_max_gap_s)

    def update(
        self,
        eye_state: EyeState,
        head_pose: Optional["HeadPose"],
        phone_present: bool,
        phone_near_face: bool,
        face_present: bool,
        timestamp: float,
        mouth_state: Optional["MouthState"] = None,
    ) -> AttentionState:
        events: List[str] = []
        penalty = 0.0

        if face_present:
            self._last_face_time = timestamp
        else:
            if self._last_face_time is None or (timestamp - self._last_face_time) > self.config.no_face_grace_s:
                events.append("Sem rosto")
                penalty += self.config.penalty_no_face

        if eye_state.microsleep:
            penalty += self.config.penalty_microsleep
            events.append("Microssono")
        elif eye_state.drowsy:
            penalty += self.config.penalty_drowsy
            events.append("Sonolencia")

        if eye_state.perclos is not None and eye_state.perclos >= self.config.perclos_threshold:
            penalty += self.config.penalty_fatigue
            events.append("Fadiga")

        if mouth_state is not None and mouth_state.yawning:
            penalty += self.config.penalty_yawn
            events.append("Bocejo")

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

        if head_pose is not None:
            self._offroad_window.add(timestamp, offroad)
        if self._offroad_window.active_s >= self.config.vats_offroad_limit_s:
            penalty += self.config.penalty_vats
            events.append("Distracao acumulada")

        if phone_near_face:
            penalty += self.config.penalty_phone
            events.append("Celular")
        elif phone_present:
            penalty += self.config.penalty_phone_suspected
            events.append("Celular suspeito")

        bonus = self.config.bonus_stable if penalty == 0 else 0.0
        raw_score = self.config.score_base - penalty + bonus
        raw_score = clamp(raw_score, self.config.score_floor, self.config.score_ceiling)
        score = self._smoother.update(raw_score)
        score = clamp(score, self.config.score_floor, self.config.score_ceiling)

        if score >= self.config.attentive_min_score:
            label = "Atento"
        elif score >= self.config.alert_min_score:
            label = "Alerta"
        else:
            label = "Distraido"

        return AttentionState(
            score=score,
            label=label,
            events=events,
            raw_score=raw_score,
            offroad_window_s=self._offroad_window.active_s,
        )
