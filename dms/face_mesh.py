from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .config import DMSConfig
from .mediapipe_utils import ensure_model_asset


@dataclass
class FaceResult:
    landmarks: Optional[np.ndarray]
    bbox: Optional[Tuple[int, int, int, int]]
    face_present: bool


class FaceMeshDetector:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self._use_tasks = not hasattr(mp, "solutions")
        self._mesh = None
        self._landmarker = None
        self._last_landmarks: Optional[np.ndarray] = None
        if not self._use_tasks:
            self._mp_face_mesh = mp.solutions.face_mesh
            self._mp_drawing = mp.solutions.drawing_utils
            self._mp_styles = mp.solutions.drawing_styles
            self._mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=config.max_faces,
                refine_landmarks=config.refine_landmarks,
                min_detection_confidence=config.detection_confidence,
                min_tracking_confidence=config.tracking_confidence,
            )
        else:
            from mediapipe import tasks

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
        self._last_results = None

    def process(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> FaceResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not self._use_tasks:
            results = self._mesh.process(rgb)
            self._last_results = results

            if not results.multi_face_landmarks:
                self._smoothed = None
                return FaceResult(landmarks=None, bbox=None, face_present=False)

            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            coords = np.array(
                [(lm.x * w, lm.y * h, lm.z * w) for lm in face_landmarks.landmark],
                dtype=np.float32,
            )
        else:
            if timestamp_ms is None:
                timestamp_ms = int(time.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            self._last_results = results
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
            alpha = self.config.landmark_smoothing
            self._smoothed = alpha * self._smoothed + (1.0 - alpha) * coords

        x_min = int(np.min(self._smoothed[:, 0]))
        y_min = int(np.min(self._smoothed[:, 1]))
        x_max = int(np.max(self._smoothed[:, 0]))
        y_max = int(np.max(self._smoothed[:, 1]))
        bbox = (x_min, y_min, x_max, y_max)

        return FaceResult(landmarks=self._smoothed.copy(), bbox=bbox, face_present=True)

    def draw(self, frame: np.ndarray) -> None:
        if not self._last_results:
            return
        if not self._use_tasks:
            if not self._last_results.multi_face_landmarks:
                return
            for face_landmarks in self._last_results.multi_face_landmarks:
                self._mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self._mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._mp_styles.get_default_face_mesh_tesselation_style(),
                )
                self._mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self._mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._mp_styles.get_default_face_mesh_contours_style(),
                )
        else:
            if self._last_landmarks is None:
                return
            for (x, y, _) in self._last_landmarks:
                cv2.circle(frame, (int(x), int(y)), 1, (80, 255, 80), -1)
