from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from dms.config import DMSConfig
from dms.overlay import (
    CHIN,
    FOREHEAD,
    LEFT_CHEEK,
    LEFT_EYE_OUTER,
    NOSE_TIP,
    RIGHT_CHEEK,
    RIGHT_EYE_OUTER,
    FaceOverlayTracker,
    LandmarkSmoother,
    compute_face_anchor,
    compute_face_roll,
    compute_face_scale,
    normalized_landmarks_to_pixels,
)


def synthetic_landmarks() -> np.ndarray:
    landmarks = np.zeros((468, 3), dtype=np.float32)
    landmarks[:, 0] = 100.0
    landmarks[:, 1] = 110.0
    landmarks[NOSE_TIP] = (100.0, 110.0, 0.0)
    landmarks[CHIN] = (100.0, 180.0, 0.0)
    landmarks[FOREHEAD] = (100.0, 40.0, 0.0)
    landmarks[LEFT_EYE_OUTER] = (60.0, 80.0, 0.0)
    landmarks[RIGHT_EYE_OUTER] = (140.0, 80.0, 0.0)
    landmarks[LEFT_CHEEK] = (40.0, 120.0, 0.0)
    landmarks[RIGHT_CHEEK] = (160.0, 120.0, 0.0)
    return landmarks


class OverlayGeometryTests(unittest.TestCase):
    def test_normalized_landmarks_are_converted_to_pixels(self) -> None:
        landmarks = np.array([[0.5, 0.25, 0.1]], dtype=np.float32)

        pixels = normalized_landmarks_to_pixels(landmarks, (720, 1280, 3))

        self.assertAlmostEqual(float(pixels[0, 0]), 640.0)
        self.assertAlmostEqual(float(pixels[0, 1]), 180.0)
        self.assertAlmostEqual(float(pixels[0, 2]), 128.0, places=4)

    def test_pixel_landmarks_are_not_scaled_again(self) -> None:
        landmarks = np.array([[640.0, 180.0, 20.0]], dtype=np.float32)

        pixels = normalized_landmarks_to_pixels(landmarks, (720, 1280, 3))

        self.assertEqual(float(pixels[0, 0]), 640.0)
        self.assertEqual(float(pixels[0, 1]), 180.0)

    def test_face_anchor_uses_stable_landmarks(self) -> None:
        anchor = compute_face_anchor(synthetic_landmarks())

        self.assertAlmostEqual(anchor[0], 100.0)
        self.assertGreater(anchor[1], 105.0)
        self.assertLess(anchor[1], 135.0)

    def test_face_scale_uses_face_width_ratio(self) -> None:
        scale, face_width, face_height = compute_face_scale(synthetic_landmarks(), reference_face_width=120.0)

        self.assertAlmostEqual(scale, 1.0)
        self.assertAlmostEqual(face_width, 120.0)
        self.assertGreater(face_height, 120.0)

    def test_face_roll_uses_eye_axis(self) -> None:
        landmarks = synthetic_landmarks()
        landmarks[RIGHT_EYE_OUTER] = (140.0, 120.0, 0.0)

        roll = compute_face_roll(landmarks)

        self.assertAlmostEqual(roll, 26.565, places=3)

    def test_landmark_smoother_uses_new_sample_alpha(self) -> None:
        smoother = LandmarkSmoother(alpha=0.5)
        smoother.update(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))

        value = smoother.update(np.array([[10.0, 4.0, 0.0]], dtype=np.float32))

        self.assertAlmostEqual(float(value[0, 0]), 5.0)
        self.assertAlmostEqual(float(value[0, 1]), 2.0)

    def test_tracker_holds_and_fades_when_face_disappears(self) -> None:
        config = DMSConfig(overlay_hold_frames=1, overlay_fade_frames=2, overlay_reference_face_width=120.0)
        tracker = FaceOverlayTracker(config)
        valid = tracker.update(synthetic_landmarks(), (240, 320, 3), head_pose=SimpleNamespace(roll=0.0))

        held = tracker.update(None, (240, 320, 3))
        fading = tracker.update(None, (240, 320, 3))
        gone = tracker.update(None, (240, 320, 3))
        expired = tracker.update(None, (240, 320, 3))

        self.assertIsNotNone(valid)
        self.assertEqual(held.tracking_state, "holding")
        self.assertEqual(held.opacity, 1.0)
        self.assertEqual(fading.tracking_state, "fading")
        self.assertGreater(fading.opacity, gone.opacity)
        self.assertIsNone(expired)


if __name__ == "__main__":
    unittest.main()
