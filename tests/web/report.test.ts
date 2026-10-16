import { describe, expect, it } from "vitest";
import { DEFAULT_SCORING_CONFIG } from "../../src/domain/config";
import { buildSessionReport, eventsToCsv, REPORT_SCHEMA_VERSION } from "../../src/domain/report";
import type { SessionSummary } from "../../src/domain/session";

const summary: SessionSummary = {
  durationS: 12.34567,
  averageScore: 81.234,
  minScore: 42.987,
  eventCount: 2,
  eventsByType: { Bocejo: 1, "Olhar fora": 1 },
  timeByLevelS: { ok: 10.0001, warn: 2.3, danger: 0 },
};

function report() {
  return buildSessionReport({
    generatedAt: new Date("2026-10-16T12:00:00Z"),
    mode: "review",
    sensitivity: 60,
    config: DEFAULT_SCORING_CONFIG,
    summary,
    onsets: [
      { type: "Olhar fora", atS: 3.14159 },
      { type: "Bocejo", atS: 9.5 },
    ],
    scoreHistory: [100, 87.654],
  });
}

describe("session report", () => {
  it("builds a versioned, rounded snapshot", () => {
    const value = report();
    expect(value.schemaVersion).toBe(REPORT_SCHEMA_VERSION);
    expect(value.generatedAt).toBe("2026-10-16T12:00:00.000Z");
    expect(value.summary.durationS).toBe(12.346);
    expect(value.summary.averageScore).toBe(81.2);
    expect(value.events[0]).toEqual({ type: "Olhar fora", atS: 3.142 });
    expect(value.scoreHistory).toEqual([100, 87.7]);
    expect(value.thresholds.perclosThreshold).toBe(DEFAULT_SCORING_CONFIG.perclosThreshold);
    expect(JSON.parse(JSON.stringify(value))).toEqual(value);
  });

  it("exports one CSV row per event", () => {
    expect(eventsToCsv(report())).toBe("offset_s,event,mode\n3.142,Olhar fora,review\n9.5,Bocejo,review\n");
  });

  it("neutralises spreadsheet formulas in CSV cells", () => {
    const value = { ...report(), source: { mode: "=HYPERLINK(1)", sensitivity: 60 } };
    expect(eventsToCsv(value).split("\n")[1]).toBe(`3.142,Olhar fora,"'=HYPERLINK(1)"`);
  });
});
