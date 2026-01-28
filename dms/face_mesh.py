from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .config import DMSConfig


@dataclass
class FaceResult:
    landmarks: Optional[np.ndarray]
    bbox: Optional[Tuple[int, int, int, int]]
    face_present: bool


class FaceMeshDetector:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
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
        self._smoothed: Optional[np.ndarray] = None
        self._last_results = None

    def process(self, frame: np.ndarray) -> FaceResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        if not self._last_results or not self._last_results.multi_face_landmarks:
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
