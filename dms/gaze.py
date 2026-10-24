"""Iris-based gaze: eye-in-head rotation from the refined iris landmarks, rotated by the head pose.

Mirrors src/vision/irisGaze.ts + src/domain/gaze.ts. The web build has no
rotation matrix, so it adds the angles; here the eye direction is rotated by
the solvePnP head rotation, which stays correct with head roll and large turns.

Sign convention follows dms/head_pose.py: positive yaw turns toward image
left, positive pitch looks down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .head_pose import HeadPose

# Face Landmarker refined iris points (center + 4 extremes) and the matching eye corners.
IRIS_POINTS: Tuple[Tuple[int, ...], Tuple[int, ...]] = ((468, 469, 470, 471, 472), (473, 474, 475, 476, 477))
EYE_CORNERS: Tuple[Tuple[int, int], Tuple[int, int]] = ((33, 133), (362, 263))
# Eyeball radius over palpebral fissure width (~12 mm / ~29 mm in adults).
EYEBALL_RADIUS_RATIO = 0.42
REFINED_LANDMARK_COUNT = 478


@dataclass(frozen=True)
class GazeAngles:
    yaw: float
    pitch: float


def iris_center(landmarks: np.ndarray, indices: Sequence[int]) -> Tuple[np.ndarray, float]:
    """Mean of the 5 iris points (steadier than the center point alone) and the iris radius."""
    points = landmarks[list(indices), :2].astype(np.float64)
    center = points.mean(axis=0)
    radius = float(np.mean(np.linalg.norm(points[1:] - center, axis=1)))
    return center, radius


def eye_rotation(iris: np.ndarray, corner_a: np.ndarray, corner_b: np.ndarray) -> Optional[GazeAngles]:
    """Eye-in-head angles of a spherical eye.

    The iris offset from the canthi midpoint, along and across the eye axis (so head roll does not leak into
    yaw), over the eyeball radius is sin(angle). None when the iris lands outside the eyeball (bad landmarks).
    """
    left, right = (corner_a, corner_b) if corner_a[0] <= corner_b[0] else (corner_b, corner_a)
    axis = np.asarray(right[:2], dtype=np.float64) - np.asarray(left[:2], dtype=np.float64)
    width = float(np.linalg.norm(axis))
    if width < 1.0:
        return None
    ux, uy = axis / width
    offset = np.asarray(iris[:2], dtype=np.float64) - (np.asarray(left[:2]) + np.asarray(right[:2])) / 2.0
    radius = width * EYEBALL_RADIUS_RATIO
    horizontal = (offset[0] * ux + offset[1] * uy) / radius
    vertical = (-offset[0] * uy + offset[1] * ux) / radius
    if abs(horizontal) > 1.0 or abs(vertical) > 1.0:
        return None
    # Iris to image right is a turn toward image right: negative yaw in the head pose convention.
    return GazeAngles(yaw=-math.degrees(math.asin(horizontal)), pitch=math.degrees(math.asin(vertical)))


def estimate_eye_gaze(landmarks: Optional[np.ndarray]) -> Optional[GazeAngles]:
    """Mean eye-in-head rotation over the eyes with usable iris landmarks; None without refined landmarks."""
    if landmarks is None or len(landmarks) < REFINED_LANDMARK_COUNT:
        return None
    rotations: List[GazeAngles] = []
    for iris_indices, (a, b) in zip(IRIS_POINTS, EYE_CORNERS, strict=True):
        center, _ = iris_center(landmarks, iris_indices)
        rotation = eye_rotation(center, landmarks[a], landmarks[b])
        if rotation is not None:
            rotations.append(rotation)
    if not rotations:
        return None
    return GazeAngles(
        yaw=sum(r.yaw for r in rotations) / len(rotations),
        pitch=sum(r.pitch for r in rotations) / len(rotations),
    )


def _direction(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Unit vector out of the face (toward the camera when yaw = pitch = 0) in camera axes."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    return np.array([-math.cos(pitch) * math.sin(yaw), math.sin(pitch), -math.cos(pitch) * math.cos(yaw)])


def _angles(vector: np.ndarray) -> GazeAngles:
    x, y, z = (float(v) for v in vector / max(float(np.linalg.norm(vector)), 1e-9))
    return GazeAngles(yaw=math.degrees(math.atan2(-x, -z)), pitch=math.degrees(math.asin(max(-1.0, min(1.0, y)))))


def combine_gaze(
    head_pose: Optional[HeadPose],
    eye_gaze: Optional[GazeAngles],
    ear: Optional[float],
    ear_threshold: float,
) -> Optional[GazeAngles]:
    """Gaze direction: head pose plus eye rotation; closed eyes (unreliable iris) fall back to the head pose."""
    if head_pose is None:
        return None
    if eye_gaze is None or ear is None or ear < ear_threshold:
        return GazeAngles(yaw=float(head_pose.yaw), pitch=float(head_pose.pitch))
    if head_pose.rotation is None:
        return GazeAngles(yaw=float(head_pose.yaw + eye_gaze.yaw), pitch=float(head_pose.pitch + eye_gaze.pitch))
    return _angles(head_pose.rotation @ _direction(eye_gaze.yaw, eye_gaze.pitch))


def gaze_deviation(gaze: Optional[GazeAngles], yaw_threshold: float, pitch_threshold: float) -> Optional[float]:
    """Deviation from the road normalized by the off-road thresholds (1 = at the threshold)."""
    if gaze is None:
        return None
    return max(abs(gaze.yaw) / yaw_threshold, abs(gaze.pitch) / pitch_threshold)


CALIBRATION_TAU_S = 8.0
CALIBRATION_GATE_DEG = 15.0
CALIBRATION_MAX_STEP_S = 0.5


class EyeGazeCalibration:
    """Per-driver neutral eye position (slow EMA while the head faces forward); mirrors src/domain/gaze.ts.

    The iris center does not sit exactly on the canthi line, so a driver looking at the road can read
    ~10 deg of eye pitch. Samples count only with open eyes, the head within the gate and the eyes near
    the current baseline, so glances and sustained off-road gaze do not pull it.
    """

    def __init__(self) -> None:
        self.baseline: Optional[GazeAngles] = None
        self._last_t: Optional[float] = None

    def reset(self) -> None:
        self.baseline = None
        self._last_t = None

    def apply(
        self, eye_gaze: Optional[GazeAngles], head_pose: Optional[HeadPose], trusted: bool, timestamp_s: float
    ) -> Optional[GazeAngles]:
        dt = 0.0 if self._last_t is None else min(CALIBRATION_MAX_STEP_S, max(0.0, timestamp_s - self._last_t))
        self._last_t = timestamp_s
        if eye_gaze is None:
            return None
        facing_forward = (
            head_pose is not None
            and abs(head_pose.yaw) < CALIBRATION_GATE_DEG
            and abs(head_pose.pitch) < CALIBRATION_GATE_DEG
        )
        if trusted and facing_forward:
            base = self.baseline
            if base is None:
                self.baseline = eye_gaze
            elif (
                abs(eye_gaze.yaw - base.yaw) < CALIBRATION_GATE_DEG
                and abs(eye_gaze.pitch - base.pitch) < CALIBRATION_GATE_DEG
            ):
                k = 1.0 - math.exp(-dt / CALIBRATION_TAU_S)
                self.baseline = GazeAngles(
                    yaw=base.yaw + k * (eye_gaze.yaw - base.yaw), pitch=base.pitch + k * (eye_gaze.pitch - base.pitch)
                )
        if self.baseline is None:
            return eye_gaze
        return GazeAngles(yaw=eye_gaze.yaw - self.baseline.yaw, pitch=eye_gaze.pitch - self.baseline.pitch)
