import type { ScoringConfig } from "./config";
import { clamp } from "./geometry";

export const SENSITIVITY_MIN = 35;
export const SENSITIVITY_MAX = 85;
export const SENSITIVITY_NEUTRAL = 60;

/**
 * Applies the operator sensitivity slider to the scoring rules.
 * Higher sensitivity narrows the head-pose tolerance and shortens how long the
 * driver may look away before it counts; the neutral value keeps the defaults.
 */
export function applySensitivity(config: ScoringConfig, sensitivity: number): ScoringConfig {
  const value = clamp(sensitivity, SENSITIVITY_MIN, SENSITIVITY_MAX);
  const factor = 1 + (SENSITIVITY_NEUTRAL - value) / 100;
  return Object.freeze({
    ...config,
    yawThreshold: config.yawThreshold * factor,
    pitchThreshold: config.pitchThreshold * factor,
    offroadTimeS: config.offroadTimeS * factor,
  });
}
