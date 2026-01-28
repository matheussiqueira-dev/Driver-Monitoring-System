from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import DMSConfig
from .mediapipe_utils import ensure_model_asset


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    conf: float
    label: str
    class_id: int


@dataclass
class YoloDetections:
    phones: List[Detection]
    hands: List[Detection]
    all: List[Detection]


class YoloPhoneDetector:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.model = None
        self.phone_class_id: Optional[int] = None
        self.hand_class_id: Optional[int] = None

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Ultralytics is required. Install with `pip install ultralytics`." ) from exc

        self.model = YOLO(config.yolo_weights)
        names = self.model.names
        self.phone_class_id = self._find_class_id(names, config.phone_class_name)
        self.hand_class_id = self._find_class_id(names, config.hand_class_name)

    @staticmethod
    def _find_class_id(names, class_name: str) -> Optional[int]:
        if isinstance(names, dict):
            for idx, name in names.items():
                if name == class_name:
                    return int(idx)
        else:
            for idx, name in enumerate(names):
                if name == class_name:
                    return int(idx)
        return None

    def detect(self, frame: np.ndarray) -> YoloDetections:
        results = self.model.predict(
            source=frame,
            conf=self.config.yolo_conf,
            iou=self.config.yolo_iou,
            device=self.config.yolo_device,
            verbose=False,
        )
        h, w = frame.shape[:2]
        detections: List[Detection] = []
        phones: List[Detection] = []
        hands: List[Detection] = []

        if not results:
            return YoloDetections(phones=phones, hands=hands, all=detections)

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
            area_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)
            if area_ratio < self.config.phone_min_area_ratio:
                continue
            label = self.model.names.get(class_id, str(class_id)) if isinstance(self.model.names, dict) else self.model.names[class_id]
            detection = Detection(bbox=(x1, y1, x2, y2), conf=conf, label=label, class_id=class_id)
            detections.append(detection)
            if self.phone_class_id is not None and class_id == self.phone_class_id:
                phones.append(detection)
            if self.hand_class_id is not None and class_id == self.hand_class_id:
                hands.append(detection)

        return YoloDetections(phones=phones, hands=hands, all=detections)


class HandDetector:
    def __init__(self, config: DMSConfig) -> None:
        import mediapipe as mp

        self._use_tasks = not hasattr(mp, "solutions")
        self._hands = None
        self._landmarker = None
        if not self._use_tasks:
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=config.hand_detection_confidence,
                min_tracking_confidence=config.hand_tracking_confidence,
            )
        else:
            from mediapipe import tasks

            model_path = config.hand_landmarker_model
            if config.download_models:
                model_path = ensure_model_asset(model_path, config.hand_landmarker_model_url)
            options = tasks.vision.HandLandmarkerOptions(
                base_options=tasks.BaseOptions(model_asset_path=model_path),
                running_mode=tasks.vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=config.hand_detection_confidence,
                min_hand_presence_confidence=config.hand_presence_confidence,
                min_tracking_confidence=config.hand_tracking_confidence,
            )
            self._landmarker = tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> List[Tuple[int, int, int, int]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        boxes = []
        if not self._use_tasks:
            results = self._hands.process(rgb)
            if not results.multi_hand_landmarks:
                return []
            for hand_landmarks in results.multi_hand_landmarks:
                coords = np.array([(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark], dtype=np.float32)
                x1, y1 = coords.min(axis=0)
                x2, y2 = coords.max(axis=0)
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
            return boxes

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not results.hand_landmarks:
            return []
        for hand_landmarks in results.hand_landmarks:
            coords = np.array([(lm.x * w, lm.y * h) for lm in hand_landmarks], dtype=np.float32)
            x1, y1 = coords.min(axis=0)
            x2, y2 = coords.max(axis=0)
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes
