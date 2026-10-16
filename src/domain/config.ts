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
  readonly perclosWindowS: number;
  readonly perclosMinCoverageS: number;
  readonly perclosThreshold: number;
  readonly marThreshold: number;
  readonly yawnMinTimeS: number;
  readonly vatsWindowS: number;
  readonly vatsOffroadLimitS: number;
  readonly rollingMaxGapS: number;
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
  readonly penaltyFatigue: number;
  readonly penaltyYawn: number;
  readonly penaltyVats: number;
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
  perclos_window_s: "perclosWindowS",
  perclos_min_coverage_s: "perclosMinCoverageS",
  perclos_threshold: "perclosThreshold",
  mar_threshold: "marThreshold",
  yawn_min_time_s: "yawnMinTimeS",
  vats_window_s: "vatsWindowS",
  vats_offroad_limit_s: "vatsOffroadLimitS",
  rolling_max_gap_s: "rollingMaxGapS",
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
  penalty_fatigue: "penaltyFatigue",
  penalty_yawn: "penaltyYawn",
  penalty_vats: "penaltyVats",
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
  if (config.perclosThreshold <= 0 || config.perclosThreshold > 1) {
    throw new AppError("CONFIG_INVALID", "perclos_threshold must be in (0, 1]");
  }
  if (config.perclosMinCoverageS > config.perclosWindowS) {
    throw new AppError("CONFIG_INVALID", "perclos_min_coverage_s must not exceed perclos_window_s");
  }
  if (config.vatsOffroadLimitS > config.vatsWindowS) {
    throw new AppError("CONFIG_INVALID", "vats_offroad_limit_s must not exceed vats_window_s");
  }
  if (Math.min(config.perclosWindowS, config.vatsWindowS, config.rollingMaxGapS, config.yawnMinTimeS) <= 0) {
    throw new AppError("CONFIG_INVALID", "PERCLOS/VATS/yawn windows and times must be positive");
  }
  return Object.freeze(config);
}

export const DEFAULT_SCORING_CONFIG: ScoringConfig = parseScoringConfig(rawScoringConfig);
