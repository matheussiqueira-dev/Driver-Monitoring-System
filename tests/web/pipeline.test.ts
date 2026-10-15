import { describe, expect, it } from "vitest";
import { MonitoringPipeline } from "../../src/app/pipeline";
import { demoSignals, type DemoProfile } from "../../src/app/signals";
import { DEFAULT_SCORING_CONFIG } from "../../src/domain/config";

const STEP_S = 1 / 30;

function runProfile(profile: DemoProfile, seconds: number) {
  const pipeline = new MonitoringPipeline(DEFAULT_SCORING_CONFIG);
  const results = [];
  for (let t = 0; t < seconds; t += STEP_S) {
    results.push(pipeline.process(demoSignals(profile, t), t));
  }
  return { pipeline, results };
}

describe("demo signals through the real pipeline", () => {
  it("are deterministic for the same time", () => {
    expect(demoSignals("review", 12.3)).toEqual(demoSignals("review", 12.3));
    expect(demoSignals("demo", 3)).toEqual(demoSignals("demo", 3 + 24));
  });

  it("demo profile: blinks never count as drowsiness, glance and phone do register", () => {
    const { pipeline, results } = runProfile("demo", 24);
    expect(results.some((result) => result.eyeState.drowsy)).toBe(false);
    expect(results.at(-1)?.eyeState.blinkRate).toBeGreaterThan(0);

    const types = new Set(pipeline.session.onsets.map((onset) => onset.type));
    expect(types).toEqual(new Set(["Olhar fora", "Celular suspeito", "Celular"]));
  });

  it("review profile: drowsiness escalates to microsleep and the face loss is reported", () => {
    const { pipeline } = runProfile("review", 24);
    const types = new Set(pipeline.session.onsets.map((onset) => onset.type));
    expect(types).toContain("Sonolencia");
    expect(types).toContain("Microssono");
    expect(types).toContain("Sem rosto");
    expect(pipeline.session.summary().minScore).toBeLessThan(40);
  });

  it("reconfigure keeps the session aggregates", () => {
    const { pipeline } = runProfile("demo", 10);
    const before = pipeline.session.summary().eventCount;
    pipeline.reconfigure(DEFAULT_SCORING_CONFIG);
    expect(pipeline.session.summary().eventCount).toBe(before);
    pipeline.reset();
    expect(pipeline.session.summary().eventCount).toBe(0);
  });
});
