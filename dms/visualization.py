"""OpenCV overlay with the same monochrome + neon palette as the web canvas (src/vision/overlayRenderer.ts)."""

from __future__ import annotations

import colorsys
import math
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .attention import AttentionState
from .ear import EyeState
from .gaze import IRIS_POINTS, GazeAngles, iris_center
from .head_pose import HeadPose
from .mouth import MouthState
from .overlay import OverlayTransform

Color = Tuple[int, int, int]
Connection = Tuple[int, int]

# BGR. Neon teal rgb(0, 255, 204) and acid pink rgb(255, 46, 99).
NEON: Color = (204, 255, 0)
ACID: Color = (99, 46, 255)
WHITE: Color = (243, 240, 236)
PLATE: Color = (10, 8, 6)
MESH_ALPHA = 0.08
FRAME_ALPHA = 0.42
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.42
NOSE_TIP = 1
ARROW_LENGTH = 0.85
ARROW_SEGMENTS = 14


def mix_color(a: Color, b: Color, t: float) -> Color:
    """Shortest-arc hue interpolation (as overlayRenderer.ts): neon teal to acid pink via violet, not gray."""
    k = min(1.0, max(0.0, t))
    ha, la, sa = colorsys.rgb_to_hls(a[2] / 255, a[1] / 255, a[0] / 255)
    hb, lb, sb = colorsys.rgb_to_hls(b[2] / 255, b[1] / 255, b[0] / 255)
    dh = ((hb - ha + 1.5) % 1.0) - 0.5
    r, g, bl = colorsys.hls_to_rgb((ha + dh * k) % 1.0, la + (lb - la) * k, sa + (sb - sa) * k)
    return int(round(bl * 255)), int(round(g * 255)), int(round(r * 255))


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    t = min(1.0, max(0.0, (value - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def accent_color(score: float, attentive_min: float = 70.0, alert_min: float = 40.0) -> Color:
    """Neon while attentive, interpolated to acid as the (already smoothed) score falls to the alert floor."""
    return mix_color(NEON, ACID, _smoothstep(attentive_min, alert_min, score))


def score_color(score: float) -> Color:
    return accent_color(score)


@lru_cache(maxsize=1)
def _face_connections() -> Tuple[Tuple[Connection, ...], ...]:
    """(mesh, eyes, lips, irises) from MediaPipe; empty when mediapipe is unavailable."""
    try:
        from mediapipe.tasks.python.vision import FaceLandmarksConnections as C
    except ImportError:  # pragma: no cover - mediapipe is a runtime dependency
        return ((), (), (), ())

    def pairs(*groups) -> Tuple[Connection, ...]:
        return tuple((c.start, c.end) for group in groups for c in group)

    return (
        pairs(C.FACE_LANDMARKS_TESSELATION),
        pairs(C.FACE_LANDMARKS_LEFT_EYE, C.FACE_LANDMARKS_RIGHT_EYE),
        pairs(C.FACE_LANDMARKS_LIPS),
        pairs(C.FACE_LANDMARKS_LEFT_IRIS, C.FACE_LANDMARKS_RIGHT_IRIS),
    )


def _blend(frame: np.ndarray, layer: np.ndarray, alpha: float) -> None:
    cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0.0, frame)


def _lines(
    image: np.ndarray, points: np.ndarray, connections: Sequence[Connection], color: Color, thickness: int
) -> None:
    count = len(points)
    segments = [
        np.array([points[a, :2], points[b, :2]], dtype=np.int32) for a, b in connections if a < count and b < count
    ]
    if segments:
        cv2.polylines(image, segments, False, color, thickness, cv2.LINE_AA)


def _glow(frame: np.ndarray, draw, blur: int = 9) -> None:
    """Canvas-like shadowBlur: draw on a black layer, blur it, add it, then add the crisp strokes."""
    layer = np.zeros_like(frame)
    draw(layer)
    halo = cv2.GaussianBlur(layer, (0, 0), blur / 3.0)
    cv2.add(frame, halo, dst=frame)
    mask = layer.any(axis=2)
    frame[mask] = layer[mask]


def draw_face_mesh(frame: np.ndarray, landmarks: Optional[np.ndarray], accent: Color = NEON) -> None:
    """Faint graphite tessellation plus neon eye (EAR), lip (MAR) and iris contours."""
    if landmarks is None or len(landmarks) == 0:
        return
    mesh, eyes, lips, irises = _face_connections()
    if mesh:
        layer = frame.copy()
        _lines(layer, landmarks, mesh, WHITE, 1)
        _blend(frame, layer, MESH_ALPHA)

    def zones(image: np.ndarray) -> None:
        _lines(image, landmarks, eyes, accent, 1)
        _lines(image, landmarks, lips, accent, 1)
        _lines(image, landmarks, irises, accent, 1)
        if len(landmarks) > max(IRIS_POINTS[1]):
            for indices in IRIS_POINTS:
                center, radius = iris_center(landmarks, indices)
                cv2.circle(image, tuple(int(v) for v in center), max(1, int(radius * 0.22)), accent, -1, cv2.LINE_AA)

    _glow(frame, zones)


def draw_gaze_arrow(
    frame: np.ndarray,
    origin: Tuple[float, float],
    gaze: GazeAngles,
    face_width: float,
    deviation: Optional[float],
    accent: Color = NEON,
) -> None:
    """Gaze vector out of the nose with a linear alpha gradient; opacity and color follow the road deviation."""
    yaw, pitch = math.radians(gaze.yaw), math.radians(gaze.pitch)
    length = max(24.0, face_width * ARROW_LENGTH)
    # Positive yaw turns toward image left (dms/head_pose.py convention).
    tip = (origin[0] - math.sin(yaw) * math.cos(pitch) * length, origin[1] + math.sin(pitch) * length)
    level = deviation or 0.0
    opacity = 0.3 + 0.7 * _smoothstep(0.0, 1.0, level)
    color = mix_color(accent, ACID, _smoothstep(0.65, 1.0, level))
    shaft = math.hypot(tip[0] - origin[0], tip[1] - origin[1])

    def arrow(image: np.ndarray) -> None:
        cv2.circle(image, (int(origin[0]), int(origin[1])), 3, color, -1, cv2.LINE_AA)
        if shaft <= 6:
            return
        for index in range(ARROW_SEGMENTS):
            t0, t1 = index / ARROW_SEGMENTS, (index + 1) / ARROW_SEGMENTS
            p0 = (int(origin[0] + (tip[0] - origin[0]) * t0), int(origin[1] + (tip[1] - origin[1]) * t0))
            p1 = (int(origin[0] + (tip[0] - origin[0]) * t1), int(origin[1] + (tip[1] - origin[1]) * t1))
            cv2.line(image, p0, p1, tuple(int(channel * t1) for channel in color), 2, cv2.LINE_AA)
        tail = (int(origin[0] + (tip[0] - origin[0]) * 0.8), int(origin[1] + (tip[1] - origin[1]) * 0.8))
        cv2.arrowedLine(image, tail, (int(tip[0]), int(tip[1])), color, 2, cv2.LINE_AA, tipLength=0.6)

    layer = frame.copy()
    _glow(layer, arrow)
    _blend(frame, layer, opacity)


def _label(frame: np.ndarray, text: str, origin: Tuple[int, int], color: Color = WHITE) -> None:
    (width, height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
    x, y = origin
    plate = frame.copy()
    cv2.rectangle(plate, (x, y - height - 6), (x + width + 10, y + baseline), PLATE, -1)
    _blend(frame, plate, 0.72)
    cv2.putText(frame, text, (x + 5, y - 2), FONT, FONT_SCALE, color, 1, cv2.LINE_AA)


def _brackets(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Color = WHITE) -> None:
    arm = int(max(6, min(18, (x2 - x1) * 0.18, (y2 - y1) * 0.18)))
    for cx, cy, sx, sy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x2, y2, -1, -1), (x1, y2, 1, -1)):
        cv2.line(image, (cx, cy), (cx + sx * arm, cy), color, 1, cv2.LINE_AA)
        cv2.line(image, (cx, cy), (cx, cy + sy * arm), color, 1, cv2.LINE_AA)


def draw_phone_boxes(frame: np.ndarray, detections: List[Tuple[int, int, int, int]], label: str) -> None:
    for x1, y1, x2, y2 in detections:
        layer = frame.copy()
        cv2.rectangle(layer, (x1, y1), (x2, y2), WHITE, 1, cv2.LINE_AA)
        _blend(frame, layer, FRAME_ALPHA)
        _label(frame, label.upper(), (x1, max(20, y1 - 6)))


def draw_face_box(frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], label: str = "face") -> None:
    if bbox is None:
        return
    layer = frame.copy()
    _brackets(layer, *bbox)
    _blend(frame, layer, FRAME_ALPHA)
    _label(frame, label.upper(), (bbox[0], max(20, bbox[1] - 8)))


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
    accent: Color = NEON,
) -> None:
    """Draw a face-attached frame (corner brackets) using tracked center, scale and roll."""

    if transform is None or transform.opacity <= 0.0:
        return

    overlay = frame.copy()
    alpha = float(np.clip(transform.opacity, 0.0, 1.0)) * FRAME_ALPHA
    width = max(80.0, transform.face_width_px * 1.12)
    height = max(100.0, transform.face_height_px * 1.08)
    corners = _rotated_points(transform.center, (width, height), transform.roll).astype(np.int32)

    for index in range(4):
        corner = corners[index]
        for neighbor in (corners[(index + 1) % 4], corners[(index + 3) % 4]):
            tick = corner + ((neighbor - corner) * 0.16).astype(np.int32)
            cv2.line(overlay, tuple(int(v) for v in corner), tuple(int(v) for v in tick), WHITE, 1, cv2.LINE_AA)

    if show_debug:
        x1, y1, x2, y2 = transform.bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), WHITE, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)

    if show_debug:
        x1, y1 = transform.bbox[:2]
        label = f"{transform.tracking_state.upper()} . SCALE {transform.scale:.2f} . ROLL {transform.roll:.1f}"
        _label(frame, label, (max(8, x1), max(22, y1 - 8)), accent if transform.tracking_state == "tracking" else WHITE)


def draw_detection_boxes(frame: np.ndarray, detections) -> None:
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        is_phone = det.label == "cell phone"
        layer = frame.copy()
        cv2.rectangle(layer, (x1, y1), (x2, y2), ACID if is_phone else WHITE, 1, cv2.LINE_AA)
        _blend(frame, layer, 0.9 if is_phone else FRAME_ALPHA)
        _label(frame, f"{det.label.upper()} {det.conf:.2f}", (x1, max(20, y1 - 6)), ACID if is_phone else WHITE)


def draw_metrics(
    frame: np.ndarray,
    eye_state: EyeState,
    head_pose: Optional[HeadPose],
    attention_state: AttentionState,
    fps: float,
    mouth_state: Optional[MouthState] = None,
    gaze: Optional[GazeAngles] = None,
) -> None:
    x, y = 22, 34
    color = accent_color(attention_state.score)
    rows = 5 + (head_pose is not None) + (gaze is not None)
    panel_h = 32 + rows * 22
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (520, panel_h), PLATE, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.line(frame, (10, 10), (10, panel_h), color, 2, cv2.LINE_AA)

    def row(text: str, tint: Color = WHITE, scale: float = 0.5) -> None:
        nonlocal y
        cv2.putText(frame, text, (x, y), FONT, scale, tint, 1, cv2.LINE_AA)
        y += 22

    row(f"SCORE {attention_state.score:5.1f}  {attention_state.label.upper()}", color, 0.62)
    if eye_state.ear is not None:
        row(f"EAR {eye_state.ear:.3f}   BLINK {eye_state.blink_rate:.1f}/MIN")
    if head_pose is not None:
        row(f"HEAD  YAW {head_pose.yaw:5.1f}  PITCH {head_pose.pitch:5.1f}  ROLL {head_pose.roll:5.1f}")
    if gaze is not None:
        row(f"GAZE  YAW {gaze.yaw:5.1f}  PITCH {gaze.pitch:5.1f}")
    perclos = "--" if eye_state.perclos is None else f"{eye_state.perclos * 100:.0f}%"
    mar = "--" if mouth_state is None or mouth_state.mar is None else f"{mouth_state.mar:.2f}"
    row(f"PERCLOS {perclos}   MAR {mar}   VATS {attention_state.offroad_window_s:.1f}S")
    row(f"FPS {fps:4.1f}")
    if attention_state.events:
        row("ALERTAS  " + "  .  ".join(event.upper() for event in attention_state.events), ACID)


def draw_attention_bar(frame: np.ndarray, score: float) -> None:
    h, w = frame.shape[:2]
    bar_w = 26
    x1 = w - bar_w - 20
    y1 = 20
    y2 = h - 20
    track = frame.copy()
    cv2.rectangle(track, (x1, y1), (x1 + bar_w, y2), WHITE, 1, cv2.LINE_AA)
    _blend(frame, track, 0.25)
    filled_height = int((y2 - y1) * (score / 100.0))
    y_fill = y2 - filled_height
    color = accent_color(score)
    _glow(frame, lambda image: cv2.rectangle(image, (x1 + 3, y_fill), (x1 + bar_w - 3, y2 - 3), color, -1), blur=7)
