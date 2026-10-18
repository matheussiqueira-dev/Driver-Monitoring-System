from __future__ import annotations

import unittest

import cv2
import numpy as np

from dms.config import DMSConfig
from dms.gaze import (
    EYE_CORNERS,
    EYEBALL_RADIUS_RATIO,
    IRIS_POINTS,
    EyeGazeCalibration,
    GazeAngles,
    combine_gaze,
    estimate_eye_gaze,
    eye_rotation,
    gaze_deviation,
)
from dms.head_pose import MODEL_TO_CAMERA_AXES, HeadPose, HeadPoseEstimator
from dms.landmark_filter import AdaptiveLandmarkFilter, ema_alpha

FRAME_W, FRAME_H = 1280, 720


def project_model(estimator: HeadPoseEstimator, rotation_vector) -> np.ndarray:
    """Synthetic 478 landmarks: the solvePnP model rotated in front of a pinhole camera."""
    rotation, _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64))
    points = (rotation @ MODEL_TO_CAMERA_AXES @ estimator.model_points.T.astype(np.float64)).T + (0.0, 0.0, 600.0)
    landmarks = np.zeros((478, 3), dtype=np.float32)
    for (x, y, z), index in zip(points, estimator.landmark_indices, strict=True):
        landmarks[index, :2] = (FRAME_W * x / z + FRAME_W / 2, FRAME_W * y / z + FRAME_H / 2)
    return landmarks


def eyes_with_iris(shift_x: float = 0.0, shift_y: float = 0.0) -> np.ndarray:
    """Two level 40 px wide eyes with the iris shifted by a fraction of the eyeball radius."""
    landmarks = np.zeros((478, 3), dtype=np.float32)
    for (outer, inner), iris, left in zip(EYE_CORNERS, IRIS_POINTS, (200.0, 300.0), strict=True):
        landmarks[outer] = (left, 150.0, 0.0)
        landmarks[inner] = (left + 40.0, 150.0, 0.0)
        radius = 40.0 * EYEBALL_RADIUS_RATIO
        cx, cy = left + 20.0 + shift_x * radius, 150.0 + shift_y * radius
        landmarks[iris[0]] = (cx, cy, 0.0)
        for index, (dx, dy) in zip(iris[1:], ((6, 0), (0, -6), (-6, 0), (0, 6)), strict=True):
            landmarks[index] = (cx + dx, cy + dy, 0.0)
    return landmarks


class HeadPoseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.estimator = HeadPoseEstimator(DMSConfig())

    def test_frontal_face_has_zero_pitch(self) -> None:
        # Regression: without undoing the model axis flip, pitch decomposed to +-180 deg (always off-road).
        pose = self.estimator.estimate(project_model(self.estimator, (0, 0, 0)), (FRAME_H, FRAME_W))

        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose.pitch, 0.0, places=3)
        self.assertAlmostEqual(pose.yaw, 0.0, places=3)

    def test_recovers_yaw_and_downward_pitch(self) -> None:
        turned = self.estimator.estimate(project_model(self.estimator, (0, np.radians(20), 0)), (FRAME_H, FRAME_W))
        down = self.estimator.estimate(project_model(self.estimator, (np.radians(15), 0, 0)), (FRAME_H, FRAME_W))

        self.assertAlmostEqual(abs(turned.yaw), 20.0, places=2)
        self.assertAlmostEqual(down.pitch, 15.0, places=2)
        self.assertIsNotNone(down.rotation)


class IrisGazeTests(unittest.TestCase):
    def test_centered_iris_is_zero_rotation(self) -> None:
        gaze = estimate_eye_gaze(eyes_with_iris())

        self.assertAlmostEqual(gaze.yaw, 0.0, places=5)
        self.assertAlmostEqual(gaze.pitch, 0.0, places=5)

    def test_iris_offset_follows_spherical_eye_model(self) -> None:
        gaze = estimate_eye_gaze(eyes_with_iris(shift_x=0.5, shift_y=0.25))

        # Iris toward image right is a turn toward image right: negative yaw (head pose convention).
        self.assertAlmostEqual(gaze.yaw, -30.0, places=3)
        self.assertAlmostEqual(gaze.pitch, float(np.degrees(np.arcsin(0.25))), places=3)

    def test_roll_does_not_leak_into_yaw(self) -> None:
        corner_a = np.array([0.0, 0.0])
        corner_b = np.array([40.0 * np.cos(0.4), 40.0 * np.sin(0.4)])
        iris = (corner_a + corner_b) / 2.0

        rotation = eye_rotation(iris, corner_a, corner_b)

        self.assertAlmostEqual(rotation.yaw, 0.0, places=5)
        self.assertAlmostEqual(rotation.pitch, 0.0, places=5)

    def test_missing_or_implausible_iris_gives_none(self) -> None:
        self.assertIsNone(estimate_eye_gaze(np.zeros((468, 3), dtype=np.float32)))
        self.assertIsNone(estimate_eye_gaze(eyes_with_iris(shift_x=3.0)))


class CombineGazeTests(unittest.TestCase):
    def test_closed_eyes_fall_back_to_head_pose(self) -> None:
        head = HeadPose(yaw=5.0, pitch=3.0, roll=0.0)

        gaze = combine_gaze(head, GazeAngles(yaw=30.0, pitch=0.0), ear=0.1, ear_threshold=0.21)

        self.assertEqual(gaze, GazeAngles(yaw=5.0, pitch=3.0))

    def test_rotation_composes_eye_with_head(self) -> None:
        estimator = HeadPoseEstimator(DMSConfig())
        head = estimator.estimate(project_model(estimator, (0, np.radians(20), 0)), (FRAME_H, FRAME_W))

        straight = combine_gaze(head, GazeAngles(yaw=0.0, pitch=0.0), ear=0.3, ear_threshold=0.21)
        compensated = combine_gaze(head, GazeAngles(yaw=-head.yaw, pitch=0.0), ear=0.3, ear_threshold=0.21)

        self.assertAlmostEqual(straight.yaw, head.yaw, places=3)
        self.assertAlmostEqual(compensated.yaw, 0.0, places=3)

    def test_deviation_is_relative_to_thresholds(self) -> None:
        self.assertAlmostEqual(gaze_deviation(GazeAngles(yaw=10.0, pitch=15.0), 20.0, 15.0), 1.0)
        self.assertIsNone(gaze_deviation(None, 20.0, 15.0))


class LandmarkFilterTests(unittest.TestCase):
    def test_alpha_matches_first_order_low_pass(self) -> None:
        self.assertAlmostEqual(float(ema_alpha(1.0, 1.0 / (2 * np.pi))), 1.0 - np.exp(-1.0))

    def test_reduces_static_jitter(self) -> None:
        rng = np.random.default_rng(7)
        base = eyes_with_iris().astype(np.float64)
        smoother = AdaptiveLandmarkFilter()
        outputs = []
        for frame in range(300):
            noisy = base + rng.normal(0.0, 0.5, base.shape)
            outputs.append(smoother.update(noisy, frame / 30.0))
        settled = np.array(outputs[30:])

        self.assertLess(float(np.std(settled[:, 33, 0])), 0.5 * 0.5)

    def test_is_frame_rate_independent(self) -> None:
        base = eyes_with_iris().astype(np.float64)
        moved = base + (6.0, 0.0, 0.0)

        def after(fps: int) -> float:
            smoother = AdaptiveLandmarkFilter(speed_beta=0.0)
            smoother.update(base, 0.0)
            value = None
            for frame in range(1, int(0.2 * fps) + 1):
                value = smoother.update(moved, frame / fps)
            return float(value[33, 0])

        self.assertAlmostEqual(after(30), after(60), places=3)

    def test_large_jump_snaps(self) -> None:
        base = eyes_with_iris().astype(np.float64)
        smoother = AdaptiveLandmarkFilter()
        smoother.update(base, 0.0)

        jumped = smoother.update(base + (400.0, 0.0, 0.0), 1.0 / 30.0)

        np.testing.assert_allclose(jumped, base + (400.0, 0.0, 0.0), atol=1e-3)


class EyeGazeCalibrationTests(unittest.TestCase):
    AHEAD = HeadPose(yaw=0.0, pitch=0.0, roll=0.0)

    def test_removes_neutral_iris_offset(self) -> None:
        calibration = EyeGazeCalibration()
        rest = GazeAngles(yaw=-1.5, pitch=-12.0)
        for step in range(50):
            calibration.apply(rest, self.AHEAD, True, step * 0.1)

        out = calibration.apply(rest, self.AHEAD, True, 5.0)

        self.assertAlmostEqual(out.yaw, 0.0, places=6)
        self.assertAlmostEqual(out.pitch, 0.0, places=6)

    def test_sustained_off_road_gaze_does_not_move_the_baseline(self) -> None:
        calibration = EyeGazeCalibration()
        calibration.apply(GazeAngles(yaw=0.0, pitch=-10.0), self.AHEAD, True, 0.0)
        for step in range(1, 200):
            calibration.apply(GazeAngles(yaw=0.0, pitch=18.0), self.AHEAD, True, step * 0.1)

        self.assertAlmostEqual(calibration.baseline.pitch, -10.0)


if __name__ == "__main__":
    unittest.main()
