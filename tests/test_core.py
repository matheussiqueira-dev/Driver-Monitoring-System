from __future__ import annotations

import unittest

import cv2
import numpy as np

from dms.attention import AttentionState
from dms.config import DMSConfig
from dms.ear import EyeState, EyeStateTracker
from dms.head_pose import HeadPose
from dms.utils import clamp
from dms.visualization import draw_attention_bar, draw_metrics
from main import build_arg_parser, parse_source


class CoreBehaviorTests(unittest.TestCase):
    def test_parse_source_accepts_camera_index_or_path(self) -> None:
        self.assertEqual(parse_source("0"), 0)
        self.assertEqual(parse_source("12"), 12)
        self.assertEqual(parse_source("assets/demo.mp4"), "assets/demo.mp4")

    def test_parse_source_keeps_non_numeric_values_as_paths(self) -> None:
        self.assertEqual(parse_source(""), "")
        self.assertEqual(parse_source("-1"), "-1")
        self.assertEqual(parse_source("0.5"), "0.5")

    def test_headless_and_max_frames_arguments(self) -> None:
        args = build_arg_parser().parse_args(["--headless", "--max-frames", "30"])

        self.assertTrue(args.headless)
        self.assertEqual(args.max_frames, 30)

        defaults = build_arg_parser().parse_args([])
        self.assertFalse(defaults.headless)
        self.assertIsNone(defaults.max_frames)

    def test_clamp_limits_values(self) -> None:
        self.assertEqual(clamp(-1, 0, 100), 0)
        self.assertEqual(clamp(42, 0, 100), 42)
        self.assertEqual(clamp(101, 0, 100), 100)

    def test_clamp_keeps_exact_boundaries(self) -> None:
        self.assertEqual(clamp(0, 0, 100), 0)
        self.assertEqual(clamp(100, 0, 100), 100)

    def test_eye_tracker_without_face_reports_open_eyes(self) -> None:
        tracker = EyeStateTracker(DMSConfig())

        state = tracker.update(None, 5.0)

        self.assertIsNone(state.ear)
        self.assertFalse(state.eyes_closed)
        self.assertFalse(state.drowsy)
        self.assertFalse(state.microsleep)
        self.assertEqual(state.closed_duration, 0.0)

    def test_eye_tracker_flags_drowsy_and_microsleep(self) -> None:
        config = DMSConfig(drowsy_time_s=0.5, microsleep_time_s=1.0)
        tracker = EyeStateTracker(config)

        normal = tracker.update(0.3, 9.5)
        closing = tracker.update(0.1, 10.0)
        drowsy = tracker.update(0.1, 10.6)
        microsleep = tracker.update(0.1, 11.1)

        self.assertFalse(normal.eyes_closed)
        self.assertTrue(closing.eyes_closed)
        self.assertFalse(closing.drowsy)
        self.assertTrue(drowsy.drowsy)
        self.assertFalse(drowsy.microsleep)
        self.assertTrue(microsleep.microsleep)


class VisualizationTests(unittest.TestCase):
    def test_overlay_draws_metrics_and_attention_bar(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:] = (18, 16, 14)
        before = frame.copy()

        eye_state = EyeState(
            ear=0.24,
            blink_rate=10.0,
            eyes_closed=False,
            drowsy=False,
            microsleep=False,
            closed_duration=0.0,
        )
        head_pose = HeadPose(yaw=2.0, pitch=1.0, roll=0.0)
        attention_state = AttentionState(score=84.0, label="Atento", events=[], raw_score=88.0)

        draw_attention_bar(frame, attention_state.score)
        draw_metrics(frame, eye_state, head_pose, attention_state, fps=24.0)

        panel_diff = cv2.absdiff(frame[10:150, 10:520], before[10:150, 10:520])
        bar_diff = cv2.absdiff(frame[20:340, 594:620], before[20:340, 594:620])

        self.assertGreater(int(np.count_nonzero(panel_diff)), 10000)
        self.assertGreater(int(np.count_nonzero(bar_diff)), 1000)


if __name__ == "__main__":
    unittest.main()
