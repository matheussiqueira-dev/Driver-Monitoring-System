from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from typing import List, Optional

import numpy as np

from dms.attention import AttentionScorer
from dms.config import DMSConfig
from dms.detection import Detection, YoloDetections
from dms.ear import LEFT_EYE_IDX, RIGHT_EYE_IDX, EyeStateTracker
from dms.errors import ConfigError, SourceUnavailableError
from dms.face_mesh import FaceResult
from dms.head_pose import HeadPose
from dms.pipeline import DMSPipeline
from main import FrameClock, build_arg_parser, build_config, main

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
FACE_BBOX = (250, 100, 390, 280)


def open_eye_landmarks() -> np.ndarray:
    """478 landmarks where both eyes have EAR = 0.3."""
    landmarks = np.zeros((478, 3), dtype=np.float32)
    for offset, indices in ((300.0, LEFT_EYE_IDX), (200.0, RIGHT_EYE_IDX)):
        p1, p2, p3, p4, p5, p6 = indices
        landmarks[p1] = (offset, 150, 0)
        landmarks[p4] = (offset + 10, 150, 0)
        landmarks[p2] = (offset + 3, 148.5, 0)
        landmarks[p6] = (offset + 3, 151.5, 0)
        landmarks[p3] = (offset + 7, 148.5, 0)
        landmarks[p5] = (offset + 7, 151.5, 0)
    return landmarks


class FakeFace:
    def __init__(self, present: bool) -> None:
        self.present = present

    def process(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> FaceResult:
        if not self.present:
            return FaceResult(landmarks=None, bbox=None, face_present=False)
        return FaceResult(landmarks=open_eye_landmarks(), bbox=FACE_BBOX, face_present=True)


class FakeHeadPose:
    def __init__(self, pose: HeadPose) -> None:
        self.pose = pose

    def estimate(self, landmarks: np.ndarray, frame_size) -> Optional[HeadPose]:
        return self.pose


class FakeOverlay:
    def update(self, landmarks, frame_shape, head_pose=None):
        return None


class FakePhone:
    def __init__(self, bbox) -> None:
        self.bbox = bbox

    def detect(self, frame: np.ndarray) -> YoloDetections:
        phone = Detection(bbox=self.bbox, conf=0.9, label="cell phone", class_id=67)
        return YoloDetections(phones=[phone], hands=[], all=[phone])


class FakeHands:
    def detect(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> List:
        return [(10, 10, 50, 50)]


STRAIGHT_AHEAD = HeadPose(0.0, 0.0, 0.0)


def make_pipeline(*, face: bool = True, pose: HeadPose = STRAIGHT_AHEAD, phone=None, hands=None) -> DMSPipeline:
    config = DMSConfig()
    return DMSPipeline(
        face_source=FakeFace(face),
        eye_tracker=EyeStateTracker(config),
        head_pose_source=FakeHeadPose(pose),
        overlay_source=FakeOverlay(),
        scorer=AttentionScorer(config),
        phone_detector=phone,
        hand_source=hands,
    )


class PipelineTests(unittest.TestCase):
    def test_attentive_driver_without_optional_detectors(self) -> None:
        result = make_pipeline().process(FRAME, 1.0)

        self.assertAlmostEqual(result.eye_state.ear, 0.3, places=5)
        self.assertEqual(result.attention.events, [])
        self.assertIsNone(result.detections)
        self.assertEqual(result.hand_boxes, [])

    def test_phone_next_to_face_gets_full_penalty(self) -> None:
        pipeline = make_pipeline(phone=FakePhone((380, 150, 440, 260)), hands=FakeHands())
        result = pipeline.process(FRAME, 1.0)

        self.assertTrue(result.phone_near_face)
        self.assertIn("Celular", result.attention.events)
        self.assertEqual(result.hand_boxes, [(10, 10, 50, 50)])

    def test_phone_without_face_is_only_suspected(self) -> None:
        result = make_pipeline(face=False, phone=FakePhone((380, 150, 440, 260))).process(FRAME, 1.0)

        self.assertTrue(result.phone_present)
        self.assertFalse(result.phone_near_face)
        self.assertIsNone(result.head_pose)
        self.assertEqual(result.attention.events, ["Sem rosto", "Celular suspeito"])

    def test_sustained_head_turn_uses_timestamps(self) -> None:
        pipeline = make_pipeline(pose=HeadPose(30.0, 0.0, 0.0))
        first = pipeline.process(FRAME, 1.0)
        later = pipeline.process(FRAME, 3.0)

        self.assertNotIn("Olhar fora", first.attention.events)
        self.assertIn("Olhar fora", later.attention.events)


class CountingPhone(FakePhone):
    def __init__(self, bbox) -> None:
        super().__init__(bbox)
        self.calls = 0

    def detect(self, frame: np.ndarray) -> YoloDetections:
        self.calls += 1
        return super().detect(frame)


class DetectorIntervalTests(unittest.TestCase):
    def test_detectors_run_every_n_frames_and_reuse_the_last_result(self) -> None:
        phone = CountingPhone((380, 150, 440, 260))
        config = DMSConfig()
        pipeline = DMSPipeline(
            face_source=FakeFace(True),
            eye_tracker=EyeStateTracker(config),
            head_pose_source=FakeHeadPose(STRAIGHT_AHEAD),
            overlay_source=FakeOverlay(),
            scorer=AttentionScorer(config),
            phone_detector=phone,
            detector_interval=3,
        )
        results = [pipeline.process(FRAME, 1.0 + i / 30) for i in range(7)]

        self.assertEqual(phone.calls, 3)  # frames 0, 3 and 6
        self.assertTrue(all(result.phone_near_face for result in results))

    def test_interval_must_be_positive(self) -> None:
        with self.assertRaises(ConfigError):
            DMSConfig(detector_interval_frames=0)
        with self.assertRaises(ValueError):
            DMSPipeline(
                face_source=FakeFace(True),
                eye_tracker=EyeStateTracker(DMSConfig()),
                head_pose_source=FakeHeadPose(STRAIGHT_AHEAD),
                overlay_source=FakeOverlay(),
                scorer=AttentionScorer(DMSConfig()),
                detector_interval=0,
            )


class CliTests(unittest.TestCase):
    def test_config_overrides_are_validated(self) -> None:
        args = build_arg_parser().parse_args(["--width", "640", "--no-mirror"])
        config = build_config(args)
        self.assertEqual(config.frame_width, 640)
        self.assertFalse(config.mirror)

        with self.assertRaises(ConfigError):
            build_config(build_arg_parser().parse_args(["--width", "-5"]))

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            DMSConfig(drowsy_time_s=3.0, microsleep_time_s=2.0)

    def test_unopenable_source_exits_with_source_error_code(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(["--source", "nao_existe.mp4", "--headless", "--no-yolo", "--no-hands"])

        self.assertEqual(code, SourceUnavailableError.exit_code)
        self.assertIn("Nao foi possivel abrir", stderr.getvalue())


class FakeCapture:
    def __init__(self, fps: float, positions_ms: List[float]) -> None:
        self.fps = fps
        self.positions_ms = positions_ms

    def get(self, prop: int) -> float:
        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return self.positions_ms.pop(0) if self.positions_ms else 0.0


class FrameClockTests(unittest.TestCase):
    def test_uses_video_position_when_available(self) -> None:
        clock = FrameClock(FakeCapture(30.0, [0.0, 33.3, 66.7, 100.0]), is_file=True)
        stamps = [clock.next() for _ in range(4)]

        self.assertAlmostEqual(stamps[-1], 0.1, places=3)
        self.assertEqual(stamps, sorted(set(stamps)))

    def test_falls_back_to_frame_index_when_position_is_missing(self) -> None:
        clock = FrameClock(FakeCapture(25.0, []), is_file=True)
        stamps = [clock.next() for _ in range(51)]

        # 50 frames at 25 fps = 2 s of video, enough for duration-based rules to fire.
        self.assertAlmostEqual(stamps[-1], 2.0, places=3)

    def test_camera_timestamps_strictly_increase(self) -> None:
        clock = FrameClock(FakeCapture(0.0, []), is_file=False)
        stamps = [clock.next() for _ in range(5)]

        self.assertTrue(all(b > a for a, b in zip(stamps, stamps[1:], strict=False)))


if __name__ == "__main__":
    unittest.main()
