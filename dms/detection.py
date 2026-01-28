from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import DMSConfig


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
    def __init__(self) -> None:
        import mediapipe as mp

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return []
        h, w = frame.shape[:2]
        boxes = []
        for hand_landmarks in results.multi_hand_landmarks:
            coords = np.array([(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark], dtype=np.float32)
            x1, y1 = coords.min(axis=0)
            x2, y2 = coords.max(axis=0)
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes
