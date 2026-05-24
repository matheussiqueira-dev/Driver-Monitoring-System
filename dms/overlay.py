from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .config import DMSConfig
from .utils import clamp


Point2D = Tuple[float, float]
BBox = Tuple[int, int, int, int]

NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
LEFT_CHEEK = 234
RIGHT_CHEEK = 454


@dataclass
class OverlayTransform:
    center: Point2D
    scale: float
    roll: float
    face_width_px: float
    face_height_px: float
    bbox: BBox
    opacity: float
    tracking_state: str
    anchors: Dict[str, Point2D]


class LandmarkSmoother:
    """EMA smoother where alpha is the weight of the new sample."""

    def __init__(self, alpha: float) -> None:
        self.alpha = clamp(alpha, 0.0, 1.0)
        self.value: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.value = None

    def update(self, landmarks: np.ndarray) -> np.ndarray:
        landmarks = landmarks.astype(np.float32, copy=False)
        if self.value is None or self.value.shape != landmarks.shape:
            self.value = landmarks.copy()
        else:
            self.value = self.value + self.alpha * (landmarks - self.value)
        return self.value.copy()


def normalized_landmarks_to_pixels(landmarks, frame_shape) -> np.ndarray:
    """Convert MediaPipe normalized landmarks to pixel coordinates.

    If an ndarray already appears to be in pixels, a float32 copy is returned.
    This lets the function safely handle both raw MediaPipe landmarks and the
    pixel landmarks emitted by FaceMeshDetector.
    """

    frame_h, frame_w = frame_shape[:2]
    if isinstance(landmarks, np.ndarray):
        coords = landmarks.astype(np.float32, copy=True)
    else:
        coords = np.array([(lm.x, lm.y, getattr(lm, "z", 0.0)) for lm in landmarks], dtype=np.float32)

    if coords.size == 0:
        return coords.reshape(0, 3)

    max_x = float(np.nanmax(np.abs(coords[:, 0])))
    max_y = float(np.nanmax(np.abs(coords[:, 1])))
    if max_x <= 1.5 and max_y <= 1.5:
        coords[:, 0] *= frame_w
        coords[:, 1] *= frame_h
        if coords.shape[1] > 2:
            coords[:, 2] *= frame_w
    return coords


def compute_landmark_bbox(landmarks: np.ndarray) -> BBox:
    x_min = int(np.min(landmarks[:, 0]))
    y_min = int(np.min(landmarks[:, 1]))
    x_max = int(np.max(landmarks[:, 0]))
    y_max = int(np.max(landmarks[:, 1]))
    return x_min, y_min, x_max, y_max


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[:2] + b[:2]) * 0.5


def point_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def compute_face_anchor(landmarks: np.ndarray) -> Point2D:
    """Compute a stable overlay center from facial landmarks, not bbox center."""

    nose = landmarks[NOSE_TIP][:2]
    chin = landmarks[CHIN][:2]
    left_eye = midpoint(landmarks[LEFT_EYE_OUTER], landmarks[LEFT_EYE_INNER])
    right_eye = midpoint(landmarks[RIGHT_EYE_OUTER], landmarks[RIGHT_EYE_INNER])
    eye_mid = midpoint(left_eye, right_eye)
    face_mid = (eye_mid + chin) * 0.5
    anchor = 0.50 * nose + 0.30 * face_mid + 0.20 * eye_mid
    return float(anchor[0]), float(anchor[1])


def compute_face_scale(landmarks: np.ndarray, reference_face_width: float) -> Tuple[float, float, float]:
    """Return (scale ratio, face width px, face height px)."""

    eye_span = point_distance(landmarks[LEFT_EYE_OUTER], landmarks[RIGHT_EYE_OUTER])
    cheek_width = point_distance(landmarks[LEFT_CHEEK], landmarks[RIGHT_CHEEK])
    face_height = point_distance(landmarks[FOREHEAD], landmarks[CHIN])
    x1, y1, x2, y2 = compute_landmark_bbox(landmarks)
    bbox_width = max(1.0, float(x2 - x1))
    bbox_height = max(1.0, float(y2 - y1))

    face_width_px = float(np.median([eye_span * 2.25, cheek_width, bbox_width]))
    face_height_px = float(np.median([face_height, bbox_height, face_width_px * 1.22]))
    scale = face_width_px / max(1.0, reference_face_width)
    return scale, face_width_px, face_height_px


def compute_face_roll(landmarks: np.ndarray) -> float:
    """Compute image-space roll from the outer eye line.

    In image coordinates, y grows downward. Positive roll means the right eye
    corner is lower on screen than the left eye corner, which is a clockwise
    visual tilt.
    """

    left = landmarks[LEFT_EYE_OUTER]
    right = landmarks[RIGHT_EYE_OUTER]
    dx = float(right[0] - left[0])
    dy = float(right[1] - left[1])
    return float(np.degrees(np.arctan2(dy, dx)))


def blend_angles(current: float, target: float, alpha: float) -> float:
    delta = ((target - current + 180.0) % 360.0) - 180.0
    return current + alpha * delta


class FaceOverlayTracker:
    """Tracks a face-attached 2D transform for virtual overlays."""

    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self._landmarks = LandmarkSmoother(config.landmark_smoothing_alpha)
        self._transform: Optional[OverlayTransform] = None
        self._missed_frames = 0

    def reset(self) -> None:
        self._landmarks.reset()
        self._transform = None
        self._missed_frames = 0

    def update(self, landmarks, frame_shape, head_pose=None) -> Optional[OverlayTransform]:
        if landmarks is None:
            return self._update_missing()

        pixel_landmarks = normalized_landmarks_to_pixels(landmarks, frame_shape)
        if len(pixel_landmarks) <= RIGHT_CHEEK:
            return self._update_missing()

        smoothed_landmarks = self._landmarks.update(pixel_landmarks)
        raw_transform = self._compute_raw_transform(smoothed_landmarks, head_pose)
        self._transform = self._smooth_transform(raw_transform)
        self._missed_frames = 0
        return self._transform

    def _compute_raw_transform(self, landmarks: np.ndarray, head_pose=None) -> OverlayTransform:
        center = compute_face_anchor(landmarks)
        raw_scale, face_width_px, face_height_px = compute_face_scale(landmarks, self.config.overlay_reference_face_width)
        scale = clamp(raw_scale, self.config.min_face_scale, self.config.max_face_scale)
        scale_ratio = scale / max(raw_scale, 1e-6)
        face_width_px *= scale_ratio
        face_height_px *= scale_ratio

        eye_roll = compute_face_roll(landmarks)
        roll = eye_roll
        if head_pose is not None:
            roll = (
                self.config.overlay_roll_weight * eye_roll
                + self.config.headpose_roll_weight * float(head_pose.roll)
            )

        anchors = {
            "center": center,
            "nose": tuple(landmarks[NOSE_TIP][:2].astype(float)),
            "chin": tuple(landmarks[CHIN][:2].astype(float)),
            "left_eye": tuple(landmarks[LEFT_EYE_OUTER][:2].astype(float)),
            "right_eye": tuple(landmarks[RIGHT_EYE_OUTER][:2].astype(float)),
        }
        return OverlayTransform(
            center=center,
            scale=scale,
            roll=roll,
            face_width_px=face_width_px,
            face_height_px=face_height_px,
            bbox=compute_landmark_bbox(landmarks),
            opacity=1.0,
            tracking_state="tracking",
            anchors=anchors,
        )

    def _smooth_transform(self, raw: OverlayTransform) -> OverlayTransform:
        if self._transform is None:
            return raw

        alpha = clamp(self.config.overlay_smoothing_alpha, 0.0, 1.0)
        prev = self._transform
        dx = raw.center[0] - prev.center[0]
        dy = raw.center[1] - prev.center[1]
        distance = (dx * dx + dy * dy) ** 0.5
        if distance > self.config.max_overlay_jump_px:
            ratio = self.config.max_overlay_jump_px / max(distance, 1e-6)
            raw_center = (prev.center[0] + dx * ratio, prev.center[1] + dy * ratio)
        else:
            raw_center = raw.center

        center = (
            prev.center[0] + alpha * (raw_center[0] - prev.center[0]),
            prev.center[1] + alpha * (raw_center[1] - prev.center[1]),
        )
        scale = prev.scale + alpha * (raw.scale - prev.scale)
        face_width_px = prev.face_width_px + alpha * (raw.face_width_px - prev.face_width_px)
        face_height_px = prev.face_height_px + alpha * (raw.face_height_px - prev.face_height_px)
        roll = blend_angles(prev.roll, raw.roll, alpha)

        anchors = {}
        for key, value in raw.anchors.items():
            old_value = prev.anchors.get(key, value)
            anchors[key] = (
                old_value[0] + alpha * (value[0] - old_value[0]),
                old_value[1] + alpha * (value[1] - old_value[1]),
            )

        return OverlayTransform(
            center=center,
            scale=scale,
            roll=roll,
            face_width_px=face_width_px,
            face_height_px=face_height_px,
            bbox=raw.bbox,
            opacity=1.0,
            tracking_state="tracking",
            anchors=anchors,
        )

    def _update_missing(self) -> Optional[OverlayTransform]:
        self._missed_frames += 1
        if self._transform is None:
            return None

        hold = self.config.overlay_hold_frames
        fade = self.config.overlay_fade_frames
        if self._missed_frames <= hold:
            opacity = 1.0
            state = "holding"
        elif self._missed_frames <= hold + fade:
            fade_progress = (self._missed_frames - hold) / max(1, fade)
            opacity = clamp(1.0 - fade_progress, 0.0, 1.0)
            state = "fading"
        else:
            self.reset()
            return None

        return OverlayTransform(
            center=self._transform.center,
            scale=self._transform.scale,
            roll=self._transform.roll,
            face_width_px=self._transform.face_width_px,
            face_height_px=self._transform.face_height_px,
            bbox=self._transform.bbox,
            opacity=opacity,
            tracking_state=state,
            anchors=self._transform.anchors,
        )
