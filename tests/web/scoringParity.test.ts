import { describe, expect, it } from "vitest";
import scenariosDocument from "../../shared/scoring-scenarios.json";
import { AttentionScorer } from "../../src/domain/attention";
import { DEFAULT_SCORING_CONFIG } from "../../src/domain/config";
import { EyeStateTracker } from "../../src/domain/eyeState";
import { MouthStateTracker } from "../../src/domain/mouthState";
import { SessionStats } from "../../src/domain/session";

interface ScenarioFrame {
  t: number;
  face_present: boolean;
  ear: number | null;
  mar: number | null;
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
  perclos: number | null;
  yawning: boolean;
  yawn_count: number;
  offroad_window_s: number;
}

interface ScenarioSession {
  duration_s: number;
  average_score: number | null;
  min_score: number | null;
  event_count: number;
  events_by_type: Record<string, number>;
  time_by_level_s: Record<"ok" | "warn" | "danger", number>;
}

interface Scenario {
  name: string;
  frames: ScenarioFrame[];
  expected: ScenarioExpected[];
  session: ScenarioSession;
}

function expectCloseOrNull(actual: number | null, expected: number | null, context: string): void {
  if (expected === null) {
    expect(actual, context).toBeNull();
  } else {
    expect(actual, context).toBeCloseTo(expected, 5);
  }
}

const scenarios = (scenariosDocument as { scenarios: Scenario[] }).scenarios;

describe("scoring parity with the Python reference engine", () => {
  it("covers every committed scenario", () => {
    expect(scenarios.length).toBeGreaterThanOrEqual(14);
  });

  it.each(scenarios.map((scenario) => [scenario.name, scenario] as const))("%s", (_name, scenario) => {
    const eyeTracker = new EyeStateTracker(DEFAULT_SCORING_CONFIG);
    const scorer = new AttentionScorer(DEFAULT_SCORING_CONFIG);
    const mouthTracker = new MouthStateTracker(DEFAULT_SCORING_CONFIG);
    const session = new SessionStats();

    scenario.frames.forEach((frame, index) => {
      const expected = scenario.expected[index];
      if (!expected) {
        throw new Error(`Missing expected output for frame ${index}`);
      }
      const eyeState = eyeTracker.update(frame.ear, frame.t);
      const mouthState = mouthTracker.update(frame.mar, frame.t);
      const headPose =
        frame.face_present && frame.yaw !== null && frame.pitch !== null
          ? { yaw: frame.yaw, pitch: frame.pitch, roll: 0 }
          : null;
      const state = scorer.update({
        eyeState,
        mouthState,
        headPose,
        phonePresent: frame.phone_present,
        phoneNearFace: frame.phone_near_face,
        facePresent: frame.face_present,
        timestampS: frame.t,
      });

      session.add(state, frame.t);
      const context = `${scenario.name} frame ${index} (t=${frame.t})`;
      expect(state.score, context).toBeCloseTo(expected.score, 5);
      expect(state.rawScore, context).toBeCloseTo(expected.raw_score, 5);
      expect(state.label, context).toBe(expected.label);
      expect([...state.events], context).toEqual(expected.events);
      expect(eyeState.eyesClosed, context).toBe(expected.eyes_closed);
      expect(eyeState.drowsy, context).toBe(expected.drowsy);
      expect(eyeState.microsleep, context).toBe(expected.microsleep);
      expect(eyeState.blinkRate, context).toBe(expected.blink_rate);
      if (expected.perclos === null) {
        expect(eyeState.perclos, context).toBeNull();
      } else {
        expect(eyeState.perclos, context).toBeCloseTo(expected.perclos, 5);
      }
      expect(mouthState.yawning, context).toBe(expected.yawning);
      expect(mouthState.yawnCount, context).toBe(expected.yawn_count);
      expect(state.offroadWindowS, context).toBeCloseTo(expected.offroad_window_s, 5);
    });

    const summary = session.summary();
    const context = `${scenario.name} session`;
    expect(summary.durationS, context).toBeCloseTo(scenario.session.duration_s, 5);
    expectCloseOrNull(summary.averageScore, scenario.session.average_score, context);
    expectCloseOrNull(summary.minScore, scenario.session.min_score, context);
    expect(summary.eventCount, context).toBe(scenario.session.event_count);
    expect(summary.eventsByType, context).toEqual(scenario.session.events_by_type);
    for (const level of ["ok", "warn", "danger"] as const) {
      expect(summary.timeByLevelS[level], `${context} ${level}`).toBeCloseTo(
        scenario.session.time_by_level_s[level],
        5,
      );
    }
  });
});
