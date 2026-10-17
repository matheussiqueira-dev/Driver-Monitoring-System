import { describe, expect, it } from "vitest";
import type { AttentionEvent, AttentionState } from "../../src/domain/attention";
import { DEFAULT_SCORING_CONFIG } from "../../src/domain/config";
import { applySensitivity, SENSITIVITY_MAX, SENSITIVITY_MIN, SENSITIVITY_NEUTRAL } from "../../src/domain/sensitivity";
import { SessionStats } from "../../src/domain/session";

function state(score: number, events: AttentionEvent[] = []): AttentionState {
  const level = score >= 70 ? "ok" : score >= 40 ? "warn" : "danger";
  const label = level === "ok" ? "Atento" : level === "warn" ? "Alerta" : "Distraido";
  return { score, rawScore: score, label, level, events, offroadWindowS: 0 };
}

describe("SessionStats", () => {
  it("counts an event once per occurrence, not once per frame", () => {
    const stats = new SessionStats(0.5);
    stats.add(state(90), 0);
    expect(stats.add(state(70, ["Olhar fora"]), 0.1)).toEqual(["Olhar fora"]);
    expect(stats.add(state(65, ["Olhar fora"]), 0.2)).toEqual([]);
    stats.add(state(90), 0.3);
    stats.add(state(70, ["Olhar fora", "Celular"]), 0.4);

    const summary = stats.summary();
    expect(summary.eventCount).toBe(3);
    expect(summary.eventsByType).toEqual({ "Olhar fora": 2, Celular: 1 });
    expect(stats.onsets[0]).toEqual({ type: "Olhar fora", atS: 0.1 });
  });

  it("samples the trend at a fixed interval and caps history", () => {
    const stats = new SessionStats(1, 3);
    for (let t = 0; t <= 10; t += 0.5) {
      stats.add(state(50 + t), t);
    }
    expect(stats.history).toEqual([58, 59, 60]);
  });

  it("accumulates time per risk level and reports duration", () => {
    const stats = new SessionStats(0.1);
    stats.add(state(90), 0);
    stats.add(state(50), 2);
    stats.add(state(20), 3);
    stats.add(state(20), 4);

    const summary = stats.summary();
    expect(summary.durationS).toBe(4);
    expect(summary.timeByLevelS).toEqual({ ok: 2, warn: 1, danger: 1 });
    expect(summary.minScore).toBe(20);
    expect(summary.averageScore).toBe(45);
  });

  it("returns empty aggregates before any sample and after reset", () => {
    const stats = new SessionStats();
    expect(stats.summary()).toMatchObject({ durationS: 0, averageScore: null, minScore: null, eventCount: 0 });
    stats.add(state(30, ["Sem rosto"]), 1);
    stats.reset();
    expect(stats.summary().eventCount).toBe(0);
    expect(stats.history).toEqual([]);
  });
});

describe("applySensitivity", () => {
  it("keeps defaults at neutral sensitivity", () => {
    const tuned = applySensitivity(DEFAULT_SCORING_CONFIG, SENSITIVITY_NEUTRAL);
    expect(tuned.yawThreshold).toBe(DEFAULT_SCORING_CONFIG.yawThreshold);
    expect(tuned.offroadTimeS).toBe(DEFAULT_SCORING_CONFIG.offroadTimeS);
  });

  it("tightens thresholds when sensitivity increases", () => {
    const strict = applySensitivity(DEFAULT_SCORING_CONFIG, SENSITIVITY_MAX);
    const lenient = applySensitivity(DEFAULT_SCORING_CONFIG, SENSITIVITY_MIN);
    expect(strict.yawThreshold).toBeLessThan(DEFAULT_SCORING_CONFIG.yawThreshold);
    expect(lenient.pitchThreshold).toBeGreaterThan(DEFAULT_SCORING_CONFIG.pitchThreshold);
    expect(strict.penaltyPhone).toBe(DEFAULT_SCORING_CONFIG.penaltyPhone);
  });

  it("clamps values outside the slider range", () => {
    expect(applySensitivity(DEFAULT_SCORING_CONFIG, 1000)).toEqual(
      applySensitivity(DEFAULT_SCORING_CONFIG, SENSITIVITY_MAX),
    );
  });
});
