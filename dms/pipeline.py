"""Per-frame DMS pipeline with injected detectors.

The pipeline only orchestrates: every model or heuristic is a collaborator
behind a small Protocol, so the composition root (main.py) decides which
implementations run and tests can replace any of them with fakes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .attention import AttentionScorer, AttentionState
from .detection import YoloDetections
from .ear import LEFT_EYE_IDX, RIGHT_EYE_IDX, EyeState, EyeStateTracker, compute_ear
from .face_mesh import FaceResult
from .head_pose import HeadPose
from .overlay import OverlayTransform
from .spatial import is_phone_near_face

BBox = Tuple[int, int, int, int]


class FaceSource(Protocol):
    def process(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> FaceResult: ...


class HeadPoseSource(Protocol):
    def estimate(self, landmarks: np.ndarray, frame_size: Tuple[int, int]) -> Optional[HeadPose]: ...


class OverlaySource(Protocol):
    def update(
        self, landmarks: Optional[np.ndarray], frame_shape: Sequence[int], head_pose: Optional[HeadPose] = None
    ) -> Optional[OverlayTransform]: ...


class PhoneDetector(Protocol):
    def detect(self, frame: np.ndarray) -> YoloDetections: ...


class HandSource(Protocol):
    def detect(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> List[BBox]: ...


@dataclass
class FrameResult:
    face: FaceResult
    eye_state: EyeState
    head_pose: Optional[HeadPose]
    overlay: Optional[OverlayTransform]
    attention: AttentionState
    detections: Optional[YoloDetections] = None
    hand_boxes: List[BBox] = field(default_factory=list)
    phone_present: bool = False
    phone_near_face: bool = False


class DMSPipeline:
    def __init__(
        self,
        *,
        face_source: FaceSource,
        eye_tracker: EyeStateTracker,
        head_pose_source: HeadPoseSource,
        overlay_source: OverlaySource,
        scorer: AttentionScorer,
        phone_detector: Optional[PhoneDetector] = None,
        hand_source: Optional[HandSource] = None,
        detector_interval: int = 1,
    ) -> None:
        if detector_interval < 1:
            raise ValueError("detector_interval must be >= 1")
        self.face_source = face_source
        self.eye_tracker = eye_tracker
        self.head_pose_source = head_pose_source
        self.overlay_source = overlay_source
        self.scorer = scorer
        self.phone_detector = phone_detector
        self.hand_source = hand_source
        self.detector_interval = detector_interval
        self._frame_index = 0
        self._last_detections: Optional[YoloDetections] = None
        self._last_hand_boxes: List[BBox] = []

    def _run_detectors(self, frame: np.ndarray, timestamp_ms: int) -> Tuple[Optional[YoloDetections], List[BBox]]:
        """Phone/hand detectors run every ``detector_interval`` frames; in between the last result is reused."""
        if self._frame_index % self.detector_interval == 0:
            if self.phone_detector is not None:
                self._last_detections = self.phone_detector.detect(frame)
            if self.hand_source is not None:
                self._last_hand_boxes = self.hand_source.detect(frame, timestamp_ms)
        self._frame_index += 1
        return self._last_detections, self._last_hand_boxes

    def process(self, frame: np.ndarray, timestamp_s: float) -> FrameResult:
        timestamp_ms = int(timestamp_s * 1000)
        face = self.face_source.process(frame, timestamp_ms)

        ear: Optional[float] = None
        head_pose: Optional[HeadPose] = None
        if face.face_present and face.landmarks is not None:
            ear = (compute_ear(face.landmarks, LEFT_EYE_IDX) + compute_ear(face.landmarks, RIGHT_EYE_IDX)) / 2.0
            head_pose = self.head_pose_source.estimate(face.landmarks, frame.shape[:2])
        overlay = self.overlay_source.update(face.landmarks, frame.shape, head_pose)
        eye_state = self.eye_tracker.update(ear, timestamp_s)

        detections, hand_boxes = self._run_detectors(frame, timestamp_ms)
        phone_present = False
        phone_near_face = False
        if detections is not None:
            if detections.phones:
                phone_present = True
                pitch = head_pose.pitch if head_pose else None
                phone_near_face = any(
                    is_phone_near_face(det.bbox, face.bbox, frame.shape, pitch) for det in detections.phones
                )

        attention = self.scorer.update(
            eye_state=eye_state,
            head_pose=head_pose,
            phone_present=phone_present,
            phone_near_face=phone_near_face,
            face_present=face.face_present,
            timestamp=timestamp_s,
        )
        return FrameResult(
            face=face,
            eye_state=eye_state,
            head_pose=head_pose,
            overlay=overlay,
            attention=attention,
            detections=detections,
            hand_boxes=list(hand_boxes),
            phone_present=phone_present,
            phone_near_face=phone_near_face,
        )
