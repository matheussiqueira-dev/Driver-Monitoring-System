import type { ScoringConfig } from "./config";
import { distance, type Point } from "./geometry";

/** MediaPipe Face Mesh inner-lip landmarks: three vertical pairs and the mouth corners. */
export const MOUTH_VERTICAL_PAIRS = [
  [81, 178],
  [13, 14],
  [311, 402],
] as const;
export const MOUTH_CORNERS = [78, 308] as const;

const YAWN_WINDOW_S = 60;
const MAX_YAWNS_TRACKED = 120;

export interface MouthState {
  readonly mar: number | null;
  readonly mouthOpen: boolean;
  readonly yawning: boolean;
  readonly openDuration: number;
  readonly yawnCount: number;
}

/** Mouth Aspect Ratio: mean inner-lip opening over mouth width. Null when landmarks are missing. */
export function computeMar(landmarks: readonly Point[]): number | null {
  const left = landmarks[MOUTH_CORNERS[0]];
  const right = landmarks[MOUTH_CORNERS[1]];
  if (!left || !right) {
    return null;
  }
  const horizontal = distance(left, right);
  if (horizontal === 0) {
    return 0;
  }
  let vertical = 0;
  for (const [top, bottom] of MOUTH_VERTICAL_PAIRS) {
    const a = landmarks[top];
    const b = landmarks[bottom];
    if (!a || !b) {
      return null;
    }
    vertical += distance(a, b);
  }
  return vertical / (MOUTH_VERTICAL_PAIRS.length * horizontal);
}

/**
 * A yawn is a mouth opening above marThreshold held for yawnMinTimeS; talking
 * is shorter. Mirrors dms/mouth.py MouthStateTracker.
 */
export class MouthStateTracker {
  private openStart: number | null = null;
  private counted = false;
  private yawnTimes: number[] = [];

  constructor(private config: ScoringConfig) {}

  setConfig(config: ScoringConfig): void {
    this.config = config;
  }

  reset(): void {
    this.openStart = null;
    this.counted = false;
    this.yawnTimes = [];
  }

  /** @param timestampS monotonic time in seconds */
  update(mar: number | null, timestampS: number): MouthState {
    if (mar === null || mar < this.config.marThreshold) {
      this.openStart = null;
      this.counted = false;
      return { mar, mouthOpen: false, yawning: false, openDuration: 0, yawnCount: this.computeYawnCount(timestampS) };
    }

    this.openStart ??= timestampS;
    const openDuration = timestampS - this.openStart;
    const yawning = openDuration >= this.config.yawnMinTimeS;
    if (yawning && !this.counted) {
      this.counted = true;
      this.yawnTimes.push(timestampS);
      if (this.yawnTimes.length > MAX_YAWNS_TRACKED) {
        this.yawnTimes.shift();
      }
    }
    return { mar, mouthOpen: true, yawning, openDuration, yawnCount: this.computeYawnCount(timestampS) };
  }

  private computeYawnCount(timestampS: number): number {
    while (this.yawnTimes.length > 0 && timestampS - (this.yawnTimes[0] ?? timestampS) > YAWN_WINDOW_S) {
      this.yawnTimes.shift();
    }
    return this.yawnTimes.length;
  }
}
