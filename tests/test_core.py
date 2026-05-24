from __future__ import annotations

import unittest

import cv2
import numpy as np

from dms.attention import AttentionState
from dms.ear import EyeState, EyeStateTracker
from dms.config import DMSConfig
from dms.head_pose import HeadPose
from dms.utils import clamp
from dms.visualization import draw_attention_bar, draw_metrics, draw_project_credits
from main import is_phone_near_face, parse_source


class CoreBehaviorTests(unittest.TestCase):
    def test_parse_source_accepts_camera_index_or_path(self) -> None:
        self.assertEqual(parse_source("0"), 0)
        self.assertEqual(parse_source("12"), 12)
        self.assertEqual(parse_source("assets/demo.mp4"), "assets/demo.mp4")

    def test_clamp_limits_values(self) -> None:
        self.assertEqual(clamp(-1, 0, 100), 0)
        self.assertEqual(clamp(42, 0, 100), 42)
        self.assertEqual(clamp(101, 0, 100), 100)

    def test_phone_near_face_uses_distance_position_and_pitch(self) -> None:
        face = (100, 100, 200, 220)
        frame_shape = (480, 640, 3)

        self.assertTrue(is_phone_near_face((205, 140, 255, 210), face, frame_shape, pitch=None))
        self.assertTrue(is_phone_near_face((310, 330, 370, 410), face, frame_shape, pitch=None))
        self.assertTrue(is_phone_near_face((410, 100, 470, 170), face, frame_shape, pitch=15.0))
        self.assertFalse(is_phone_near_face((410, 100, 470, 170), face, frame_shape, pitch=0.0))
        self.assertFalse(is_phone_near_face((410, 100, 470, 170), None, frame_shape, pitch=20.0))

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
    def test_overlay_draws_metrics_bar_and_credits(self) -> None:
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
        draw_project_credits(frame)

        diff = cv2.absdiff(frame, before)
        footer_diff = cv2.absdiff(frame[300:350, 20:620], before[300:350, 20:620])

        self.assertGreater(int(np.count_nonzero(diff)), 10000)
        self.assertGreater(int(np.count_nonzero(footer_diff)), 1000)


if __name__ == "__main__":
    unittest.main()
