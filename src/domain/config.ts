import rawScoringConfig from "../../shared/scoring-config.json";
import { AppError } from "../shared/errors";

/**
 * Scoring rules shared with the Python engine (shared/scoring-config.json).
 * Times are in seconds, angles in degrees.
 */
export interface ScoringConfig {
  readonly earThreshold: number;
  readonly blinkMinFrames: number;
  readonly blinkMaxFrames: number;
  readonly drowsyTimeS: number;
  readonly microsleepTimeS: number;
  readonly yawThreshold: number;
  readonly pitchThreshold: number;
  readonly offroadTimeS: number;
  readonly noFaceGraceS: number;
  readonly scoreBase: number;
  readonly scoreSmoothing: number;
  readonly scoreFloor: number;
  readonly scoreCeiling: number;
  readonly attentiveMinScore: number;
  readonly alertMinScore: number;
  readonly penaltyNoFace: number;
  readonly penaltyPhone: number;
  readonly penaltyPhoneSuspected: number;
  readonly penaltyOffroad: number;
  readonly penaltyDrowsy: number;
  readonly penaltyMicrosleep: number;
  readonly bonusStable: number;
}

const KEY_MAP = {
  ear_threshold: "earThreshold",
  blink_min_frames: "blinkMinFrames",
  blink_max_frames: "blinkMaxFrames",
  drowsy_time_s: "drowsyTimeS",
  microsleep_time_s: "microsleepTimeS",
  yaw_threshold: "yawThreshold",
  pitch_threshold: "pitchThreshold",
  offroad_time_s: "offroadTimeS",
  no_face_grace_s: "noFaceGraceS",
  score_base: "scoreBase",
  score_smoothing: "scoreSmoothing",
  score_floor: "scoreFloor",
  score_ceiling: "scoreCeiling",
  attentive_min_score: "attentiveMinScore",
  alert_min_score: "alertMinScore",
  penalty_no_face: "penaltyNoFace",
  penalty_phone: "penaltyPhone",
  penalty_phone_suspected: "penaltyPhoneSuspected",
  penalty_offroad: "penaltyOffroad",
  penalty_drowsy: "penaltyDrowsy",
  penalty_microsleep: "penaltyMicrosleep",
  bonus_stable: "bonusStable",
} as const satisfies Record<string, keyof ScoringConfig>;

/** Validate untyped config (snake_case, as stored in JSON) and map it to ScoringConfig. */
export function parseScoringConfig(raw: unknown): ScoringConfig {
  if (typeof raw !== "object" || raw === null) {
    throw new AppError("CONFIG_INVALID", "Scoring config must be an object");
  }
  const source = raw as Record<string, unknown>;
  const result: Partial<Record<keyof ScoringConfig, number>> = {};

  for (const [jsonKey, configKey] of Object.entries(KEY_MAP)) {
    const value = source[jsonKey];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new AppError("CONFIG_INVALID", `Scoring config "${jsonKey}" must be a finite number`);
    }
    result[configKey] = value;
  }

  const config = result as ScoringConfig;
  if (config.scoreSmoothing < 0 || config.scoreSmoothing >= 1) {
    throw new AppError("CONFIG_INVALID", "score_smoothing must be in [0, 1)");
  }
  if (config.scoreFloor >= config.scoreCeiling) {
    throw new AppError("CONFIG_INVALID", "score_floor must be lower than score_ceiling");
  }
  if (config.drowsyTimeS > config.microsleepTimeS) {
    throw new AppError("CONFIG_INVALID", "drowsy_time_s must not exceed microsleep_time_s");
  }
  return Object.freeze(config);
}

export const DEFAULT_SCORING_CONFIG: ScoringConfig = parseScoringConfig(rawScoringConfig);
