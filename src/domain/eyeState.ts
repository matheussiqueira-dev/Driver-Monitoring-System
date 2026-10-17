import type { ScoringConfig } from "./config";
import { distance, type Point } from "./geometry";
import { RollingDutyCycle } from "./rolling";

/** MediaPipe Face Mesh indices: p1..p6 of the EAR formula, per eye. */
export const LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380] as const;
export const RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144] as const;

type EyeIndices = readonly [number, number, number, number, number, number];

export interface EyeState {
  readonly ear: number | null;
  readonly blinkRate: number;
  readonly eyesClosed: boolean;
  readonly drowsy: boolean;
  readonly microsleep: boolean;
  readonly closedDuration: number;
  /** Share of the last perclosWindowS with eyes closed; null until enough coverage. */
  readonly perclos: number | null;
}

/** Eye Aspect Ratio (Soukupova & Cech, 2016). Returns null when landmarks are missing. */
export function computeEar(landmarks: readonly Point[], indices: EyeIndices): number | null {
  const [p1, p2, p3, p4, p5, p6] = indices.map((index) => landmarks[index]);
  if (!p1 || !p2 || !p3 || !p4 || !p5 || !p6) {
    return null;
  }
  const horizontal = distance(p1, p4);
  if (horizontal === 0) {
    return 0;
  }
  return (distance(p2, p6) + distance(p3, p5)) / (2 * horizontal);
}

export function computeMeanEar(landmarks: readonly Point[]): number | null {
  const left = computeEar(landmarks, LEFT_EYE_INDICES);
  const right = computeEar(landmarks, RIGHT_EYE_INDICES);
  if (left === null || right === null) {
    return null;
  }
  return (left + right) / 2;
}

const BLINK_WINDOW_S = 60;
const MAX_BLINKS_TRACKED = 120;

/**
 * Tracks eye closure over time. Mirrors dms/ear.py EyeStateTracker: a short
 * closure is a blink; only sustained closure becomes drowsiness/microsleep.
 */
export class EyeStateTracker {
  private blinkFrames = 0;
  private closedStart: number | null = null;
  private blinkTimes: number[] = [];
  private readonly closure: RollingDutyCycle;

  constructor(private config: ScoringConfig) {
    this.closure = new RollingDutyCycle(config.perclosWindowS, config.rollingMaxGapS);
  }

  /** Change thresholds without losing an eye closure in progress (the PERCLOS window keeps its size). */
  setConfig(config: ScoringConfig): void {
    this.config = config;
  }

  reset(): void {
    this.blinkFrames = 0;
    this.closedStart = null;
    this.blinkTimes = [];
    this.closure.reset();
  }

  /** @param timestampS monotonic time in seconds */
  update(ear: number | null, timestampS: number): EyeState {
    if (ear === null) {
      return {
        ear: null,
        blinkRate: this.computeBlinkRate(timestampS),
        eyesClosed: false,
        drowsy: false,
        microsleep: false,
        closedDuration: 0,
        perclos: this.closure.ratio(this.config.perclosMinCoverageS),
      };
    }

    this.closure.add(timestampS, ear < this.config.earThreshold);
    if (ear < this.config.earThreshold) {
      this.blinkFrames += 1;
      this.closedStart ??= timestampS;
    } else {
      if (this.closedStart !== null) {
        const { blinkMinFrames, blinkMaxFrames } = this.config;
        if (this.blinkFrames >= blinkMinFrames && this.blinkFrames <= blinkMaxFrames) {
          this.blinkTimes.push(timestampS);
          if (this.blinkTimes.length > MAX_BLINKS_TRACKED) {
            this.blinkTimes.shift();
          }
        }
      }
      this.blinkFrames = 0;
      this.closedStart = null;
    }

    const closedDuration = this.closedStart === null ? 0 : timestampS - this.closedStart;
    return {
      ear,
      blinkRate: this.computeBlinkRate(timestampS),
      eyesClosed: this.closedStart !== null,
      drowsy: closedDuration >= this.config.drowsyTimeS,
      microsleep: closedDuration >= this.config.microsleepTimeS,
      closedDuration,
      perclos: this.closure.ratio(this.config.perclosMinCoverageS),
    };
  }

  private computeBlinkRate(timestampS: number): number {
    while (this.blinkTimes.length > 0 && timestampS - (this.blinkTimes[0] ?? timestampS) > BLINK_WINDOW_S) {
      this.blinkTimes.shift();
    }
    return this.blinkTimes.length;
  }
}
