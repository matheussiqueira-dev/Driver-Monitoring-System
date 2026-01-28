from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import DMSConfig


@dataclass
class HeadPose:
    yaw: float
    pitch: float
    roll: float


class HeadPoseEstimator:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (-28.9, -28.9, -24.1),
                (28.9, -28.9, -24.1),
            ],
            dtype=np.float32,
        )
        self.landmark_indices = [1, 152, 33, 263, 61, 291]

    def estimate(self, landmarks: np.ndarray, frame_size: Tuple[int, int]) -> Optional[HeadPose]:
        if landmarks is None:
            return None

        image_points = np.array([landmarks[idx][:2] for idx in self.landmark_indices], dtype=np.float32)
        h, w = frame_size
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
        pitch, yaw, roll = angles
        return HeadPose(yaw=yaw, pitch=pitch, roll=roll)

    @staticmethod
    def _rotation_matrix_to_euler_angles(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0
        return np.degrees(x), np.degrees(y), np.degrees(z)
