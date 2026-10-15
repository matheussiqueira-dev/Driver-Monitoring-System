"""Deterministic input sequences used to check scoring parity across runtimes.

Each frame describes what the vision layer observed at a timestamp. Replaying a
scenario through EyeStateTracker + AttentionScorer gives the reference outputs
stored in shared/scoring-scenarios.json.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .attention import AttentionScorer
from .config import DMSConfig
from .ear import EyeStateTracker
from .head_pose import HeadPose

Frame = Dict[str, Any]

STEP_S = 0.1


def _frames(
    count: int,
    *,
    start: float = 0.0,
    ear: Optional[float] = 0.3,
    yaw: float = 0.0,
    pitch: float = 0.0,
    face: bool = True,
    phone: bool = False,
    phone_near_face: bool = False,
) -> List[Frame]:
    frames: List[Frame] = []
    for index in range(count):
        frames.append(
            {
                "t": round(start + index * STEP_S, 4),
                "face_present": face,
                "ear": ear if face else None,
                "yaw": yaw if face else None,
                "pitch": pitch if face else None,
                "phone_present": phone,
                "phone_near_face": phone_near_face,
            }
        )
    return frames


def _sequence(*parts: List[Frame]) -> List[Frame]:
    """Concatenate parts, re-timing them so timestamps keep increasing by STEP_S."""
    frames: List[Frame] = []
    for part in parts:
        for frame in part:
            frames.append({**frame, "t": round(len(frames) * STEP_S, 4)})
    return frames


SCENARIOS: Dict[str, List[Frame]] = {
    "attentive_driver": _frames(15),
    "normal_blink": _sequence(_frames(5), _frames(3, ear=0.15), _frames(5)),
    "drowsy_then_microsleep": _sequence(_frames(3), _frames(25, ear=0.12)),
    "brief_glance_away": _sequence(_frames(3), _frames(10, yaw=26.0), _frames(5)),
    "sustained_offroad": _sequence(_frames(3), _frames(25, yaw=26.0)),
    "looking_down": _sequence(_frames(3), _frames(25, pitch=19.0)),
    "phone_near_face": _sequence(_frames(3), _frames(10, phone=True, phone_near_face=True)),
    "phone_far_from_face": _sequence(_frames(3), _frames(10, phone=True)),
    "face_lost": _sequence(_frames(5), _frames(20, face=False)),
    "no_face_from_start": _frames(10, face=False),
}


def replay_scenario(frames: List[Frame], config: Optional[DMSConfig] = None) -> List[Dict[str, Any]]:
    config = config or DMSConfig()
    eye_tracker = EyeStateTracker(config)
    scorer = AttentionScorer(config)
    outputs: List[Dict[str, Any]] = []

    for frame in frames:
        eye_state = eye_tracker.update(frame["ear"], frame["t"])
        head_pose = None
        if frame["face_present"]:
            head_pose = HeadPose(yaw=frame["yaw"], pitch=frame["pitch"], roll=0.0)
        state = scorer.update(
            eye_state=eye_state,
            head_pose=head_pose,
            phone_present=frame["phone_present"],
            phone_near_face=frame["phone_near_face"],
            face_present=frame["face_present"],
            timestamp=frame["t"],
        )
        outputs.append(
            {
                "score": round(state.score, 6),
                "raw_score": round(state.raw_score, 6),
                "label": state.label,
                "events": state.events,
                "eyes_closed": eye_state.eyes_closed,
                "drowsy": eye_state.drowsy,
                "microsleep": eye_state.microsleep,
                "blink_rate": eye_state.blink_rate,
            }
        )
    return outputs
