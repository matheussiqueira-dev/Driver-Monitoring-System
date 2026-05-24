from __future__ import annotations

from dataclasses import dataclass


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

    ear_threshold: float = 0.21
    blink_min_frames: int = 2
    blink_max_frames: int = 6
    drowsy_time_s: float = 1.2
    microsleep_time_s: float = 2.0

    yaw_threshold: float = 20.0
    pitch_threshold: float = 15.0
    roll_threshold: float = 20.0
    offroad_time_s: float = 1.5

    yolo_weights: str = "yolov8n.pt"
    yolo_conf: float = 0.3
    yolo_iou: float = 0.45
    yolo_device: str = "cpu"
    phone_class_name: str = "cell phone"
    hand_class_name: str = "hand"
    phone_min_area_ratio: float = 0.0005

    score_base: float = 100.0
    score_smoothing: float = 0.8
    score_floor: float = 0.0
    score_ceiling: float = 100.0

    penalty_phone: float = 40.0
    penalty_phone_suspected: float = 12.0
    penalty_offroad: float = 20.0
    penalty_drowsy: float = 30.0
    penalty_microsleep: float = 50.0
    bonus_stable: float = 5.0

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
