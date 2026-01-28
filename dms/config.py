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
    penalty_offroad: float = 20.0
    penalty_drowsy: float = 30.0
    penalty_microsleep: float = 50.0
    bonus_stable: float = 5.0

    show_mesh: bool = True
    show_debug: bool = True
    bar_position: str = "right"
