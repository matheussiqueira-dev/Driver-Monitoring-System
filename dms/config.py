from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .errors import ConfigError

SHARED_SCORING_CONFIG = Path(__file__).resolve().parents[1] / "shared" / "scoring-config.json"


def load_shared_scoring(path: Path = SHARED_SCORING_CONFIG) -> Dict[str, float]:
    """Scoring rules shared with the web app; the JSON is the single source of truth."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Nao foi possivel ler {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} deve conter um objeto JSON")
    invalid = [key for key, value in raw.items() if isinstance(value, bool) or not isinstance(value, (int, float))]
    if invalid:
        raise ConfigError(f"{path}: valores nao numericos em {', '.join(sorted(invalid))}")
    return raw


_SCORING = load_shared_scoring()


@dataclass
class DMSConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    mirror: bool = True

    max_faces: int = 1
    refine_landmarks: bool = True
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    landmark_smoothing: float = 0.7
    landmark_smoothing_alpha: float = 0.45
    # Face Mesh landmark filter (dms/landmark_filter.py): dt-aware, speed-adaptive EMA.
    landmark_min_cutoff_hz: float = 1.7
    landmark_speed_beta: float = 4.0
    landmark_derivative_cutoff_hz: float = 1.0
    face_landmarker_model: str = "models/face_landmarker.task"
    face_landmarker_model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )
    hand_landmarker_model: str = "models/hand_landmarker.task"
    hand_landmarker_model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    hand_detection_confidence: float = 0.5
    hand_presence_confidence: float = 0.5
    hand_tracking_confidence: float = 0.5
    download_models: bool = True

    ear_threshold: float = _SCORING["ear_threshold"]
    blink_min_frames: int = _SCORING["blink_min_frames"]
    blink_max_frames: int = _SCORING["blink_max_frames"]
    drowsy_time_s: float = _SCORING["drowsy_time_s"]
    microsleep_time_s: float = _SCORING["microsleep_time_s"]

    yaw_threshold: float = _SCORING["yaw_threshold"]
    pitch_threshold: float = _SCORING["pitch_threshold"]
    roll_threshold: float = 20.0
    offroad_time_s: float = _SCORING["offroad_time_s"]
    no_face_grace_s: float = _SCORING["no_face_grace_s"]

    # Fatigue and distraction over time windows (PERCLOS, yawn, Euro NCAP VATS).
    perclos_window_s: float = _SCORING["perclos_window_s"]
    perclos_min_coverage_s: float = _SCORING["perclos_min_coverage_s"]
    perclos_threshold: float = _SCORING["perclos_threshold"]
    mar_threshold: float = _SCORING["mar_threshold"]
    yawn_min_time_s: float = _SCORING["yawn_min_time_s"]
    vats_window_s: float = _SCORING["vats_window_s"]
    vats_offroad_limit_s: float = _SCORING["vats_offroad_limit_s"]
    rolling_max_gap_s: float = _SCORING["rolling_max_gap_s"]

    yolo_weights: str = "yolov8n.pt"
    yolo_conf: float = 0.3
    yolo_iou: float = 0.45
    yolo_device: str = "cpu"
    phone_class_name: str = "cell phone"
    hand_class_name: str = "hand"
    phone_min_area_ratio: float = 0.0005
    # YOLO and hand detection are the heaviest stages; run them every N frames and reuse the result.
    detector_interval_frames: int = 3

    score_base: float = _SCORING["score_base"]
    score_smoothing: float = _SCORING["score_smoothing"]
    score_floor: float = _SCORING["score_floor"]
    score_ceiling: float = _SCORING["score_ceiling"]
    attentive_min_score: float = _SCORING["attentive_min_score"]
    alert_min_score: float = _SCORING["alert_min_score"]

    penalty_no_face: float = _SCORING["penalty_no_face"]
    penalty_phone: float = _SCORING["penalty_phone"]
    penalty_phone_suspected: float = _SCORING["penalty_phone_suspected"]
    penalty_offroad: float = _SCORING["penalty_offroad"]
    penalty_drowsy: float = _SCORING["penalty_drowsy"]
    penalty_microsleep: float = _SCORING["penalty_microsleep"]
    penalty_fatigue: float = _SCORING["penalty_fatigue"]
    penalty_yawn: float = _SCORING["penalty_yawn"]
    penalty_vats: float = _SCORING["penalty_vats"]
    bonus_stable: float = _SCORING["bonus_stable"]

    show_mesh: bool = True
    show_face_overlay: bool = True
    show_debug: bool = True
    bar_position: str = "right"

    overlay_smoothing_alpha: float = 0.35
    overlay_hold_frames: int = 8
    overlay_fade_frames: int = 12
    max_overlay_jump_px: float = 80.0
    min_face_scale: float = 0.5
    max_face_scale: float = 2.5
    overlay_reference_face_width: float = 360.0
    overlay_roll_weight: float = 0.7
    headpose_roll_weight: float = 0.3

    def __post_init__(self) -> None:
        problems = []
        if self.frame_width <= 0 or self.frame_height <= 0:
            problems.append("frame_width/frame_height devem ser positivos")
        if not 0.0 <= self.score_smoothing < 1.0:
            problems.append("score_smoothing deve estar em [0, 1)")
        if self.score_floor >= self.score_ceiling:
            problems.append("score_floor deve ser menor que score_ceiling")
        if self.drowsy_time_s > self.microsleep_time_s:
            problems.append("drowsy_time_s nao pode exceder microsleep_time_s")
        if not 0.0 < self.perclos_threshold <= 1.0:
            problems.append("perclos_threshold deve estar em (0, 1]")
        if self.perclos_min_coverage_s > self.perclos_window_s:
            problems.append("perclos_min_coverage_s nao pode exceder perclos_window_s")
        if self.vats_offroad_limit_s > self.vats_window_s:
            problems.append("vats_offroad_limit_s nao pode exceder vats_window_s")
        if min(self.perclos_window_s, self.vats_window_s, self.rolling_max_gap_s, self.yawn_min_time_s) <= 0:
            problems.append("janelas e tempos de PERCLOS/VATS/bocejo devem ser positivos")
        if self.blink_min_frames > self.blink_max_frames:
            problems.append("blink_min_frames nao pode exceder blink_max_frames")
        if self.landmark_min_cutoff_hz <= 0 or self.landmark_derivative_cutoff_hz <= 0 or self.landmark_speed_beta < 0:
            problems.append("cortes do filtro de landmarks devem ser positivos e o beta nao negativo")
        if self.detector_interval_frames < 1:
            problems.append("detector_interval_frames deve ser >= 1")
        if not 0.0 < self.yolo_conf <= 1.0:
            problems.append("yolo_conf deve estar em (0, 1]")
        if problems:
            raise ConfigError("Configuracao invalida: " + "; ".join(problems))
