from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .attention import AttentionState
from .ear import EyeState
from .head_pose import HeadPose


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
        cv2.putText(frame, "Alertas: " + ", ".join(attention_state.events), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)


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
