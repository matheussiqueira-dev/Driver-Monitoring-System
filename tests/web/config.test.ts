import { describe, expect, it } from "vitest";
import rawConfig from "../../shared/scoring-config.json";
import { DEFAULT_SCORING_CONFIG, parseScoringConfig } from "../../src/domain/config";
import { AppError } from "../../src/shared/errors";

describe("parseScoringConfig", () => {
  it("maps the shared snake_case config to camelCase", () => {
    expect(DEFAULT_SCORING_CONFIG.earThreshold).toBe(rawConfig.ear_threshold);
    expect(DEFAULT_SCORING_CONFIG.penaltyMicrosleep).toBe(rawConfig.penalty_microsleep);
    expect(Object.isFrozen(DEFAULT_SCORING_CONFIG)).toBe(true);
  });

  it.each([null, 42, "config"])("rejects a non-object value (%s)", (value) => {
    expect(() => parseScoringConfig(value)).toThrow(AppError);
  });

  it("rejects missing or non-numeric fields", () => {
    expect(() => parseScoringConfig({ ...rawConfig, ear_threshold: "0.2" })).toThrow(/ear_threshold/);
    const withoutPenalty: Record<string, unknown> = { ...rawConfig };
    delete withoutPenalty.penalty_phone;
    expect(() => parseScoringConfig(withoutPenalty)).toThrow(/penalty_phone/);
    expect(() => parseScoringConfig({ ...rawConfig, yaw_threshold: Number.NaN })).toThrow(/yaw_threshold/);
  });

  it("rejects inconsistent ranges", () => {
    expect(() => parseScoringConfig({ ...rawConfig, score_smoothing: 1 })).toThrow(/score_smoothing/);
    expect(() => parseScoringConfig({ ...rawConfig, score_floor: 100 })).toThrow(/score_floor/);
    expect(() => parseScoringConfig({ ...rawConfig, drowsy_time_s: 3 })).toThrow(/drowsy_time_s/);
  });

  it("returns AppError with a stable code", () => {
    try {
      parseScoringConfig(null);
    } catch (error) {
      expect(error).toBeInstanceOf(AppError);
      expect((error as AppError).code).toBe("CONFIG_INVALID");
    }
  });
});
