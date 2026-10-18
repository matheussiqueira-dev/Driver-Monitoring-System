import type { HeadPose } from "./attention";

/** Eye-in-head rotation in degrees, same sign convention as the web head pose (+yaw right, +pitch down). */
export interface EyeGaze {
  readonly yaw: number;
  readonly pitch: number;
}

/** Where the driver is looking, in degrees: head pose plus eye-in-head rotation. */
export interface GazeDirection {
  readonly yaw: number;
  readonly pitch: number;
}

/**
 * Combines head pose with the iris-based eye rotation. Eyes below the EAR
 * threshold are not trusted (iris landmarks drift when the lids close), so the
 * gaze falls back to the head pose. Small-angle composition: the web head pose
 * is itself a 2D estimate, see dms/gaze.py for the solvePnP rotation version.
 */
export function combineGaze(
  headPose: HeadPose | null,
  eyeGaze: EyeGaze | null,
  ear: number | null,
  earThreshold: number,
): GazeDirection | null {
  if (!headPose) {
    return null;
  }
  if (!eyeGaze || ear === null || ear < earThreshold) {
    return { yaw: headPose.yaw, pitch: headPose.pitch };
  }
  return { yaw: headPose.yaw + eyeGaze.yaw, pitch: headPose.pitch + eyeGaze.pitch };
}

/** Deviation from the road normalized by the off-road thresholds (1 = at the threshold). */
export function gazeDeviation(gaze: GazeDirection | null, yawThreshold: number, pitchThreshold: number): number | null {
  if (!gaze) {
    return null;
  }
  return Math.max(Math.abs(gaze.yaw) / yawThreshold, Math.abs(gaze.pitch) / pitchThreshold);
}

/** Time constant of the neutral-gaze baseline (slow: real glances last 1-2 s). */
const CALIBRATION_TAU_S = 8;
/** Head pose and eye offset limits (degrees) for a sample to count as "looking ahead". */
const CALIBRATION_GATE_DEG = 15;
const CALIBRATION_MAX_STEP_S = 0.5;

/**
 * Per-driver neutral eye position. The iris center does not sit exactly on the
 * canthi line (anatomy, camera height, landmark bias), so a driver looking at
 * the road can read ~10 deg of eye pitch. This keeps a slow EMA of the eye
 * rotation while the head faces forward and the eyes stay near the baseline,
 * and returns the rotation relative to it. Mirrors dms/gaze.py.
 */
export class EyeGazeCalibration {
  private baseline: EyeGaze | null = null;
  private lastS: number | null = null;

  reset(): void {
    this.baseline = null;
    this.lastS = null;
  }

  /** @param trusted false while the eyes are closed (iris landmarks unreliable) */
  apply(eyeGaze: EyeGaze | null, headPose: HeadPose | null, trusted: boolean, timestampS: number): EyeGaze | null {
    const dtS = this.lastS === null ? 0 : Math.min(CALIBRATION_MAX_STEP_S, Math.max(0, timestampS - this.lastS));
    this.lastS = timestampS;
    if (!eyeGaze) {
      return null;
    }
    const facingForward =
      headPose !== null &&
      Math.abs(headPose.yaw) < CALIBRATION_GATE_DEG &&
      Math.abs(headPose.pitch) < CALIBRATION_GATE_DEG;
    if (trusted && facingForward) {
      const base = this.baseline;
      if (!base) {
        this.baseline = eyeGaze;
      } else if (
        Math.abs(eyeGaze.yaw - base.yaw) < CALIBRATION_GATE_DEG &&
        Math.abs(eyeGaze.pitch - base.pitch) < CALIBRATION_GATE_DEG
      ) {
        const k = 1 - Math.exp(-dtS / CALIBRATION_TAU_S);
        this.baseline = {
          yaw: base.yaw + k * (eyeGaze.yaw - base.yaw),
          pitch: base.pitch + k * (eyeGaze.pitch - base.pitch),
        };
      }
    }
    const base = this.baseline;
    return base ? { yaw: eyeGaze.yaw - base.yaw, pitch: eyeGaze.pitch - base.pitch } : eyeGaze;
  }
}
