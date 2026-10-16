from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe import tasks

from .config import DMSConfig
from .mediapipe_utils import ensure_model_asset


@dataclass
class FaceResult:
    landmarks: Optional[np.ndarray]
    bbox: Optional[Tuple[int, int, int, int]]
    face_present: bool


class FaceMeshDetector:
    """MediaPipe Tasks Face Landmarker in VIDEO mode, with EMA-smoothed landmarks in pixels.

    mediapipe 1.x removed the legacy ``mp.solutions`` API, so Tasks is the only backend.
    """

    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        model_path = config.face_landmarker_model
        if config.download_models:
            model_path = ensure_model_asset(model_path, config.face_landmarker_model_url)
        options = tasks.vision.FaceLandmarkerOptions(
            base_options=tasks.BaseOptions(model_asset_path=model_path),
            running_mode=tasks.vision.RunningMode.VIDEO,
            num_faces=config.max_faces,
            min_face_detection_confidence=config.detection_confidence,
            min_face_presence_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = tasks.vision.FaceLandmarker.create_from_options(options)
        self._smoothed: Optional[np.ndarray] = None
        self._last_landmarks: Optional[np.ndarray] = None

    def process(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> FaceResult:
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not results.face_landmarks:
            self._smoothed = None
            self._last_landmarks = None
            return FaceResult(landmarks=None, bbox=None, face_present=False)

        h, w = frame.shape[:2]
        coords = np.array(
            [(lm.x * w, lm.y * h, lm.z * w) for lm in results.face_landmarks[0]],
            dtype=np.float32,
        )
        self._last_landmarks = coords.copy()

        if self._smoothed is None:
            self._smoothed = coords
        else:
            alpha = self.config.landmark_smoothing_alpha
            self._smoothed = self._smoothed + alpha * (coords - self._smoothed)

        x_min, y_min = np.min(self._smoothed[:, :2], axis=0).astype(int)
        x_max, y_max = np.max(self._smoothed[:, :2], axis=0).astype(int)
        bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
        return FaceResult(landmarks=self._smoothed.copy(), bbox=bbox, face_present=True)

    def draw(self, frame: np.ndarray) -> None:
        if self._last_landmarks is None:
            return
        for x, y, _ in self._last_landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (80, 255, 80), -1)

    def close(self) -> None:
        self._landmarker.close()
