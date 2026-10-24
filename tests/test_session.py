from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dms.attention import AttentionState
from dms.config import DMSConfig
from dms.session import SessionStats, build_session_report
from main import write_report


def state(score: float, events: list[str] | None = None) -> AttentionState:
    label = "Atento" if score >= 70 else "Alerta" if score >= 40 else "Distraido"
    return AttentionState(score=score, label=label, events=events or [], raw_score=score)


# Keys of src/domain/report.ts SessionReport; both runtimes must emit the same shape.
REPORT_KEYS = {"schemaVersion", "generatedAt", "source", "thresholds", "summary", "events", "scoreHistory"}
SUMMARY_KEYS = {"durationS", "averageScore", "minScore", "eventCount", "eventsByType", "timeByLevelS"}
THRESHOLD_KEYS = {
    "earThreshold",
    "perclosThreshold",
    "marThreshold",
    "yawThreshold",
    "pitchThreshold",
    "vatsOffroadLimitS",
    "vatsWindowS",
}


class SessionStatsTests(unittest.TestCase):
    def test_counts_an_event_once_per_occurrence(self) -> None:
        stats = SessionStats(sample_interval_s=0.5)
        stats.add(state(90), 0.0)
        self.assertEqual(stats.add(state(70, ["Olhar fora"]), 0.1), ["Olhar fora"])
        self.assertEqual(stats.add(state(65, ["Olhar fora"]), 0.2), [])
        stats.add(state(90), 0.3)
        stats.add(state(30, ["Olhar fora", "Celular"]), 0.4)

        summary = stats.summary()
        self.assertEqual(summary.event_count, 3)
        self.assertEqual(summary.events_by_type, {"Olhar fora": 2, "Celular": 1})
        self.assertAlmostEqual(summary.duration_s, 0.4)
        self.assertAlmostEqual(summary.time_by_level_s["warn"], 0.1)


class ReportTests(unittest.TestCase):
    def test_report_has_the_web_schema(self) -> None:
        stats = SessionStats()
        stats.add(state(90), 0.0)
        stats.add(state(30, ["Microssono"]), 1.23456)
        report = build_session_report(stats, DMSConfig(), mode="python-video", generated_at="2026-10-17T12:00:00.000Z")

        self.assertEqual(set(report), REPORT_KEYS)
        self.assertEqual(report["schemaVersion"], 1)
        self.assertEqual(set(report["summary"]), SUMMARY_KEYS)  # type: ignore[arg-type]
        self.assertEqual(set(report["thresholds"]), THRESHOLD_KEYS)  # type: ignore[arg-type]
        self.assertEqual(report["events"], [{"type": "Microssono", "atS": 1.235}])
        self.assertEqual(report["source"], {"mode": "python-video", "sensitivity": None})

    def test_write_report_creates_the_file(self) -> None:
        stats = SessionStats()
        stats.add(state(90), 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sub" / "sessao.json"
            write_report(str(target), stats, DMSConfig(), is_file=False)
            data = json.loads(target.read_text(encoding="utf-8"))

        self.assertEqual(data["source"]["mode"], "python-camera")
        self.assertTrue(data["generatedAt"].endswith("Z"))


if __name__ == "__main__":
    unittest.main()
