import type { HeadPose } from "../domain/attention";

/** What the vision layer observed in one frame; input to the scoring pipeline. */
export interface FrameSignals {
  readonly facePresent: boolean;
  readonly ear: number | null;
  readonly headPose: HeadPose | null;
  readonly phonePresent: boolean;
  readonly phoneNearFace: boolean;
}

export type DemoProfile = "demo" | "review";

interface Segment {
  readonly until: number;
  readonly kind: "attentive" | "glance" | "phoneFar" | "phoneNear" | "eyesClosed" | "lookDown" | "faceLost";
}

const CYCLE_S = 24;

/** Scripted driver behaviour; "review" replays a higher-risk trip. */
const TIMELINES: Record<DemoProfile, readonly Segment[]> = {
  demo: [
    { until: 6, kind: "attentive" },
    { until: 9, kind: "glance" },
    { until: 14, kind: "attentive" },
    { until: 15, kind: "phoneFar" },
    { until: 17.5, kind: "phoneNear" },
    { until: 24, kind: "attentive" },
  ],
  review: [
    { until: 4, kind: "attentive" },
    { until: 7.5, kind: "eyesClosed" },
    { until: 10, kind: "attentive" },
    { until: 14, kind: "lookDown" },
    { until: 16, kind: "faceLost" },
    { until: 24, kind: "attentive" },
  ],
};

const OPEN_EAR = 0.29;
const CLOSED_EAR = 0.12;
const BLINK_PERIOD_S = 4.2;
const BLINK_DURATION_S = 0.12;

function segmentAt(profile: DemoProfile, cycleTimeS: number): Segment["kind"] {
  const timeline = TIMELINES[profile];
  return (timeline.find((segment) => cycleTimeS < segment.until) ?? timeline[timeline.length - 1])?.kind ?? "attentive";
}

/**
 * Deterministic synthetic signals for the demo. They go through the same
 * EyeStateTracker + AttentionScorer as the camera, so the demo exercises the
 * real rules (blink vs drowsiness, sustained glance, phone proximity).
 */
export function demoSignals(profile: DemoProfile, timeS: number): FrameSignals {
  const cycleTime = ((timeS % CYCLE_S) + CYCLE_S) % CYCLE_S;
  const kind = segmentAt(profile, cycleTime);
  // Everything derives from the cycle time, so the demo loops seamlessly.
  const drift = {
    yaw: Math.sin(cycleTime / 0.98) * 4,
    pitch: 3 + Math.cos(cycleTime / 1.18) * 3,
    roll: Math.sin(cycleTime / 1.32) * 2,
  };
  const blinking = cycleTime % BLINK_PERIOD_S < BLINK_DURATION_S;
  const openEar = OPEN_EAR + Math.sin(cycleTime / 0.72) * 0.012;

  switch (kind) {
    case "faceLost":
      return { facePresent: false, ear: null, headPose: null, phonePresent: false, phoneNearFace: false };
    case "eyesClosed": {
      // Microsleep: eyes shut, then the head starts to drop.
      const segmentStart = 4;
      const nod = Math.min(1, Math.max(0, (cycleTime - segmentStart) / 1.2));
      const headPose = { ...drift, pitch: drift.pitch + nod * 20 };
      return { facePresent: true, ear: CLOSED_EAR, headPose, phonePresent: false, phoneNearFace: false };
    }
    case "glance":
      return {
        facePresent: true,
        ear: openEar,
        headPose: { ...drift, yaw: 27 + drift.yaw * 0.3 },
        phonePresent: false,
        phoneNearFace: false,
      };
    case "lookDown":
      return {
        facePresent: true,
        ear: openEar,
        headPose: { ...drift, pitch: 22 },
        phonePresent: true,
        phoneNearFace: true,
      };
    case "phoneFar":
      return { facePresent: true, ear: openEar, headPose: drift, phonePresent: true, phoneNearFace: false };
    case "phoneNear":
      return {
        facePresent: true,
        ear: openEar,
        headPose: { ...drift, pitch: 12 },
        phonePresent: true,
        phoneNearFace: true,
      };
    case "attentive":
      return {
        facePresent: true,
        ear: blinking ? CLOSED_EAR : openEar,
        headPose: drift,
        phonePresent: false,
        phoneNearFace: false,
      };
  }
}
