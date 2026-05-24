from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .attention import AttentionState
from .ear import EyeState
from .head_pose import HeadPose

Color = Tuple[int, int, int]
FONT = cv2.FONT_HERSHEY_SIMPLEX
PROJECT_CREDIT = "Desenvolvido por Matheus Siqueira"
PROJECT_URL = "www.matheussiqueira.dev"


@dataclass(frozen=True)
class OverlayTheme:
    background: Color = (13, 7, 5)
    surface_soft: Color = (41, 31, 17)
    border: Color = (238, 211, 34)
    primary: Color = (238, 211, 34)
    text: Color = (252, 250, 248)
    muted: Color = (184, 163, 148)
    success: Color = (153, 211, 52)
    warning: Color = (36, 191, 251)
    danger: Color = (133, 113, 251)
    shadow: Color = (0, 0, 0)


THEME = OverlayTheme()


def score_color(score: float) -> Color:
    if score >= 70:
        return THEME.success
    if score >= 40:
        return THEME.warning
    return THEME.danger


def _blend_rect(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Color, alpha: float) -> None:
    h, w = frame.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x1 >= x2 or y1 >= y2:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _draw_panel(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, alpha: float = 0.72) -> None:
    _blend_rect(frame, x1 + 4, y1 + 4, x2 + 4, y2 + 4, THEME.shadow, 0.25)
    _blend_rect(frame, x1, y1, x2, y2, THEME.background, alpha)
    cv2.rectangle(frame, (x1, y1), (x2, y2), THEME.border, 1)
    cv2.line(frame, (x1 + 12, y1), (min(x2 - 12, x1 + 96), y1), THEME.primary, 2)
    cv2.line(frame, (x1, y1 + 12), (x1, min(y2 - 12, y1 + 58)), THEME.primary, 2)


def _fit_text_scale(text: str, max_width: int, scale: float, thickness: int) -> float:
    while scale > 0.42:
        width = cv2.getTextSize(text, FONT, scale, thickness)[0][0]
        if width <= max_width:
            return scale
        scale -= 0.04
    return 0.42


def _truncate_text(text: str, max_width: int, scale: float, thickness: int) -> str:
    if cv2.getTextSize(text, FONT, scale, thickness)[0][0] <= max_width:
        return text
    suffix = "..."
    available = max_width - cv2.getTextSize(suffix, FONT, scale, thickness)[0][0]
    truncated = text
    while truncated and cv2.getTextSize(truncated, FONT, scale, thickness)[0][0] > available:
        truncated = truncated[:-1]
    return f"{truncated.rstrip()}{suffix}" if truncated else suffix


def _put_text(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float,
    color: Color,
    thickness: int = 1,
    max_width: Optional[int] = None,
) -> None:
    if max_width is not None:
        scale = _fit_text_scale(text, max_width, scale, thickness)
        text = _truncate_text(text, max_width, scale, thickness)
    cv2.putText(frame, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def _draw_corner_box(frame: np.ndarray, bbox: Tuple[int, int, int, int], color: Color) -> None:
    x1, y1, x2, y2 = bbox
    line = max(12, min(32, int(min(max(1, x2 - x1), max(1, y2 - y1)) * 0.22)))
    cv2.line(frame, (x1, y1), (x1 + line, y1), color, 2)
    cv2.line(frame, (x1, y1), (x1, y1 + line), color, 2)
    cv2.line(frame, (x2, y1), (x2 - line, y1), color, 2)
    cv2.line(frame, (x2, y1), (x2, y1 + line), color, 2)
    cv2.line(frame, (x1, y2), (x1 + line, y2), color, 2)
    cv2.line(frame, (x1, y2), (x1, y2 - line), color, 2)
    cv2.line(frame, (x2, y2), (x2 - line, y2), color, 2)
    cv2.line(frame, (x2, y2), (x2, y2 - line), color, 2)


def draw_phone_boxes(frame: np.ndarray, detections: List[Tuple[int, int, int, int]], label: str) -> None:
    for (x1, y1, x2, y2) in detections:
        _draw_corner_box(frame, (x1, y1, x2, y2), THEME.warning)
        _put_text(frame, label.upper(), (x1, max(20, y1 - 8)), 0.58, THEME.warning, 2)


def draw_detection_boxes(frame: np.ndarray, detections) -> None:
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = THEME.primary if "phone" in det.label.lower() else THEME.muted
        _draw_corner_box(frame, (x1, y1, x2, y2), color)
        text = f"{det.label} {det.conf:.2f}"
        _put_text(frame, text, (x1, max(20, y1 - 8)), 0.5, color, 1)


def draw_metrics(
    frame: np.ndarray,
    eye_state: EyeState,
    head_pose: Optional[HeadPose],
    attention_state: AttentionState,
    fps: float,
) -> None:
    h, w = frame.shape[:2]
    panel_x1, panel_y1 = 18, 18
    panel_w = max(220, min(max(330, int(w * 0.34)), w - 118))
    row_count = 4
    if eye_state.ear is not None:
        row_count += 2
    if head_pose is not None:
        row_count += 1
    if attention_state.events:
        row_count += 1
    panel_h = 78 + row_count * 25
    panel_x2 = panel_x1 + panel_w
    panel_y2 = min(h - 64, panel_y1 + panel_h)
    _draw_panel(frame, panel_x1, panel_y1, panel_x2, panel_y2)

    x, y = panel_x1 + 18, panel_y1 + 30
    color = score_color(attention_state.score)
    content_w = panel_w - 36
    _put_text(frame, "DRIVER MONITORING SYSTEM", (x, y), 0.58, THEME.text, 2, content_w)
    y += 24
    _put_text(frame, "Attention telemetry", (x, y), 0.48, THEME.muted, 1, content_w)
    y += 30

    rows = [
        ("Score", f"{attention_state.score:5.1f} / 100", color),
        ("Estado", attention_state.label, color),
    ]
    if eye_state.ear is not None:
        rows.append(("EAR", f"{eye_state.ear:.3f}", THEME.text))
        rows.append(("Piscadas", f"{eye_state.blink_rate:.1f}/min", THEME.text))
    if head_pose is not None:
        rows.append(
            (
                "Pose",
                f"Y {head_pose.yaw:4.1f}  P {head_pose.pitch:4.1f}  R {head_pose.roll:4.1f}",
                THEME.text,
            )
        )
    rows.append(("FPS", f"{fps:4.1f}", THEME.text))

    if attention_state.events:
        rows.append(("Alertas", ", ".join(attention_state.events), THEME.warning))

    label_w = min(112, int(content_w * 0.36))
    for label, value, value_color in rows:
        if y + 8 >= panel_y2:
            break
        _put_text(frame, label.upper(), (x, y), 0.45, THEME.muted, 1, label_w)
        _put_text(frame, value, (x + label_w, y), 0.54, value_color, 1, content_w - label_w)
        y += 25


def draw_attention_bar(frame: np.ndarray, score: float) -> None:
    h, w = frame.shape[:2]
    bar_w = 28
    x1 = max(18, w - bar_w - 30)
    y1 = 56
    y2 = max(y1 + 80, h - 78)
    _draw_panel(frame, x1 - 12, y1 - 34, x1 + bar_w + 12, y2 + 14, alpha=0.62)
    _put_text(frame, "SCORE", (x1 - 8, y1 - 13), 0.4, THEME.muted, 1, bar_w + 16)
    cv2.rectangle(frame, (x1, y1), (x1 + bar_w, y2), THEME.surface_soft, 1)
    filled_height = int((y2 - y1) * (score / 100.0))
    y_fill = y2 - filled_height
    color = score_color(score)
    _blend_rect(frame, x1 + 3, y_fill, x1 + bar_w - 3, y2 - 3, color, 0.88)
    for ratio in (0.25, 0.5, 0.75):
        tick_y = int(y2 - (y2 - y1) * ratio)
        cv2.line(frame, (x1 - 4, tick_y), (x1 + bar_w + 4, tick_y), THEME.border, 1)


def draw_project_credits(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    compact = w < 560
    footer_w = min(w - 24, 680)
    footer_h = 48 if compact else 36
    x1 = max(12, (w - footer_w) // 2)
    x2 = min(w - 12, x1 + footer_w)
    y1 = max(12, h - footer_h - 12)
    y2 = h - 12

    _blend_rect(frame, x1, y1, x2, y2, THEME.background, 0.68)
    cv2.rectangle(frame, (x1, y1), (x2, y2), THEME.border, 1)

    if compact:
        _put_text(frame, PROJECT_CREDIT, (x1 + 12, y1 + 19), 0.45, THEME.text, 1, footer_w - 24)
        _put_text(frame, PROJECT_URL, (x1 + 12, y1 + 39), 0.43, THEME.primary, 1, footer_w - 24)
        return

    text = f"{PROJECT_CREDIT} | {PROJECT_URL}"
    scale = _fit_text_scale(text, footer_w - 28, 0.48, 1)
    text_w, text_h = cv2.getTextSize(text, FONT, scale, 1)[0]
    text_x = x1 + max(12, (footer_w - text_w) // 2)
    text_y = y1 + (footer_h + text_h) // 2 - 3
    _put_text(frame, text, (text_x, text_y), scale, THEME.text, 1, footer_w - 28)
