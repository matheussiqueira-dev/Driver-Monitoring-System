from __future__ import annotations

import unittest

import numpy as np

from dms.config import DMSConfig
from dms.mouth import MOUTH_CORNERS, MOUTH_VERTICAL_PAIRS, MouthStateTracker, compute_mar
from dms.rolling import RollingDutyCycle


class RollingDutyCycleTests(unittest.TestCase):
    def test_time_weighted_share(self) -> None:
        window = RollingDutyCycle(window_s=10.0, max_gap_s=1.0)
        for i in range(11):
            window.add(i * 0.5, active=i < 4)  # active during [0, 2) of [0, 5]

        self.assertAlmostEqual(window.active_s, 2.0)
        self.assertAlmostEqual(window.covered_s, 5.0)
        self.assertAlmostEqual(window.ratio(min_coverage_s=1.0), 0.4)

    def test_ratio_needs_minimum_coverage(self) -> None:
        window = RollingDutyCycle(window_s=10.0, max_gap_s=1.0)
        window.add(0.0, True)
        window.add(0.5, True)

        self.assertIsNone(window.ratio(min_coverage_s=1.0))
        self.assertEqual(window.ratio(min_coverage_s=0.5), 1.0)

    def test_old_samples_leave_the_window(self) -> None:
        window = RollingDutyCycle(window_s=2.0, max_gap_s=1.0)
        for i in range(5):
            window.add(i * 0.5, active=True)  # active [0, 2]
        for i in range(5, 15):
            window.add(i * 0.5, active=False)  # inactive [2.5, 7]

        self.assertAlmostEqual(window.active_s, 0.0)
        self.assertLessEqual(window.covered_s, 2.0)

    def test_long_gaps_are_capped(self) -> None:
        window = RollingDutyCycle(window_s=60.0, max_gap_s=0.5)
        window.add(0.0, True)
        window.add(10.0, True)  # face lost for 10 s counts as 0.5 s

        self.assertAlmostEqual(window.covered_s, 0.5)


def mouth_landmarks(opening: float) -> np.ndarray:
    landmarks = np.zeros((478, 3), dtype=np.float32)
    left, right = MOUTH_CORNERS
    landmarks[left] = (100, 200, 0)
    landmarks[right] = (160, 200, 0)
    for offset, (top, bottom) in zip((115, 130, 145), MOUTH_VERTICAL_PAIRS, strict=True):
        landmarks[top] = (offset, 200 - opening / 2, 0)
        landmarks[bottom] = (offset, 200 + opening / 2, 0)
    return landmarks


class MouthTests(unittest.TestCase):
    def test_mar_is_vertical_over_horizontal_opening(self) -> None:
        self.assertAlmostEqual(compute_mar(mouth_landmarks(0.0)), 0.0)
        self.assertAlmostEqual(compute_mar(mouth_landmarks(45.0)), 0.75)

    def test_yawn_needs_sustained_opening_and_counts_once(self) -> None:
        tracker = MouthStateTracker(DMSConfig())
        states = [tracker.update(0.8, i * 0.1) for i in range(30)]

        self.assertFalse(states[10].yawning)
        self.assertTrue(states[15].yawning)  # 1.5 s after opening
        self.assertEqual(states[-1].yawn_count, 1.0)

        tracker.update(0.1, 3.1)
        self.assertEqual(tracker.update(None, 3.2).yawn_count, 1.0)


if __name__ == "__main__":
    unittest.main()
