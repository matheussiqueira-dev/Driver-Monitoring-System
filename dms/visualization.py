from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .attention import AttentionState
from .ear import EyeState
from .head_pose import HeadPose
from .overlay import OverlayTransform


def score_color(score: float) -> Tuple[int, int, int]:
    if score >= 70:
        return (60, 200, 60)
    if score >= 40:
        return (0, 215, 255)
    return (0, 0, 255)


def draw_phone_boxes(frame: np.ndarray, detections: List[Tuple[int, int, int, int]], label: str) -> None:
    for (x1, y1, x2, y2) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)


def draw_face_box(frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], label: str = "face") -> None:
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 40), 2)
    cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 40), 1)


def _rotated_points(center: Tuple[float, float], size: Tuple[float, float], angle_deg: float) -> np.ndarray:
    cx, cy = center
    half_w, half_h = size[0] * 0.5, size[1] * 0.5
    points = np.array(
        [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ],
        dtype=np.float32,
    )
    angle = np.deg2rad(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return points @ rotation.T + np.array([cx, cy], dtype=np.float32)


def draw_tracked_face_overlay(
    frame: np.ndarray,
    transform: Optional[OverlayTransform],
    show_debug: bool = False,
) -> None:
    """Draw face-attached virtual frame using tracked center, scale and roll."""

    if transform is None or transform.opacity <= 0.0:
        return

    overlay = frame.copy()
    alpha = float(np.clip(transform.opacity, 0.0, 1.0))
    color = (65, 230, 120) if transform.tracking_state == "tracking" else (0, 200, 255)
    width = max(80.0, transform.face_width_px * 1.12)
    height = max(100.0, transform.face_height_px * 1.08)
    corners = _rotated_points(transform.center, (width, height), transform.roll).astype(np.int32)

    cv2.polylines(overlay, [corners], True, color, 2, cv2.LINE_AA)

    for start, end in ((0, 1), (1, 2), (2, 3), (3, 0)):
        p1 = corners[start]
        p2 = corners[end]
        tick = p1 + ((p2 - p1) * 0.22).astype(np.int32)
        cv2.line(overlay, tuple(p1), tuple(tick), color, 4, cv2.LINE_AA)

    center = tuple(int(v) for v in transform.center)
    cv2.circle(overlay, center, 4, (80, 255, 80), -1, cv2.LINE_AA)

    left_eye = tuple(int(v) for v in transform.anchors["left_eye"])
    right_eye = tuple(int(v) for v in transform.anchors["right_eye"])
    nose = tuple(int(v) for v in transform.anchors["nose"])
    cv2.line(overlay, left_eye, right_eye, (255, 210, 80), 2, cv2.LINE_AA)
    cv2.circle(overlay, nose, 3, (80, 255, 255), -1, cv2.LINE_AA)

    if show_debug:
        x1, y1, x2, y2 = transform.bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 160, 40), 1)
        label = f"{transform.tracking_state} scale:{transform.scale:.2f} roll:{transform.roll:.1f}"
        cv2.putText(
            overlay,
            label,
            (max(8, x1), max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def draw_detection_boxes(frame: np.ndarray, detections) -> None:
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
        text = f"{det.label} {det.conf:.2f}"
        cv2.putText(frame, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 1)


def draw_metrics(
    frame: np.ndarray,
    eye_state: EyeState,
    head_pose: Optional[HeadPose],
    attention_state: AttentionState,
    fps: float,
) -> None:
    x, y = 20, 30
    color = score_color(attention_state.score)
    panel_h = 150 if head_pose is not None else 118
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (520, panel_h), (12, 18, 24), -1)
    cv2.addWeighted(overlay, 0.48, frame, 0.52, 0, frame)
    cv2.putText(
        frame,
        f"Score: {attention_state.score:5.1f} ({attention_state.label})",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    y += 26
    if eye_state.ear is not None:
        cv2.putText(frame, f"EAR: {eye_state.ear:.3f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
        y += 22
        cv2.putText(frame, f"Blink rate: {eye_state.blink_rate:.1f}/min", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
        y += 22
    if head_pose is not None:
        cv2.putText(
            frame,
            f"Yaw: {head_pose.yaw:5.1f} Pitch: {head_pose.pitch:5.1f} Roll: {head_pose.roll:5.1f}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 230, 230),
            1,
        )
        y += 22
    cv2.putText(frame, f"FPS: {fps:4.1f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
    y += 22
    if attention_state.events:
        cv2.putText(
            frame,
            "Alertas: " + ", ".join(attention_state.events),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )


def draw_attention_bar(frame: np.ndarray, score: float) -> None:
    h, w = frame.shape[:2]
    bar_w = 26
    x1 = w - bar_w - 20
    y1 = 20
    y2 = h - 20
    cv2.rectangle(frame, (x1, y1), (x1 + bar_w, y2), (50, 50, 50), 2)
    filled_height = int((y2 - y1) * (score / 100.0))
    y_fill = y2 - filled_height
    color = score_color(score)
    cv2.rectangle(frame, (x1 + 2, y_fill), (x1 + bar_w - 2, y2 - 2), color, -1)
