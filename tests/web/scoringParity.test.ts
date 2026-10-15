import { describe, expect, it } from "vitest";
import scenariosDocument from "../../shared/scoring-scenarios.json";
import { AttentionScorer } from "../../src/domain/attention";
import { DEFAULT_SCORING_CONFIG } from "../../src/domain/config";
import { EyeStateTracker } from "../../src/domain/eyeState";

interface ScenarioFrame {
  t: number;
  face_present: boolean;
  ear: number | null;
  yaw: number | null;
  pitch: number | null;
  phone_present: boolean;
  phone_near_face: boolean;
}

interface ScenarioExpected {
  score: number;
  raw_score: number;
  label: string;
  events: string[];
  eyes_closed: boolean;
  drowsy: boolean;
  microsleep: boolean;
  blink_rate: number;
}

interface Scenario {
  name: string;
  frames: ScenarioFrame[];
  expected: ScenarioExpected[];
}

const scenarios = (scenariosDocument as { scenarios: Scenario[] }).scenarios;

describe("scoring parity with the Python reference engine", () => {
  it("covers every committed scenario", () => {
    expect(scenarios.length).toBeGreaterThanOrEqual(10);
  });

  it.each(scenarios.map((scenario) => [scenario.name, scenario] as const))("%s", (_name, scenario) => {
    const eyeTracker = new EyeStateTracker(DEFAULT_SCORING_CONFIG);
    const scorer = new AttentionScorer(DEFAULT_SCORING_CONFIG);

    scenario.frames.forEach((frame, index) => {
      const expected = scenario.expected[index];
      if (!expected) {
        throw new Error(`Missing expected output for frame ${index}`);
      }
      const eyeState = eyeTracker.update(frame.ear, frame.t);
      const headPose =
        frame.face_present && frame.yaw !== null && frame.pitch !== null
          ? { yaw: frame.yaw, pitch: frame.pitch, roll: 0 }
          : null;
      const state = scorer.update({
        eyeState,
        headPose,
        phonePresent: frame.phone_present,
        phoneNearFace: frame.phone_near_face,
        facePresent: frame.face_present,
        timestampS: frame.t,
      });

      const context = `${scenario.name} frame ${index} (t=${frame.t})`;
      expect(state.score, context).toBeCloseTo(expected.score, 5);
      expect(state.rawScore, context).toBeCloseTo(expected.raw_score, 5);
      expect(state.label, context).toBe(expected.label);
      expect([...state.events], context).toEqual(expected.events);
      expect(eyeState.eyesClosed, context).toBe(expected.eyes_closed);
      expect(eyeState.drowsy, context).toBe(expected.drowsy);
      expect(eyeState.microsleep, context).toBe(expected.microsleep);
      expect(eyeState.blinkRate, context).toBe(expected.blink_rate);
    });
  });
});
