import type { ScoringConfig } from "./config";
import type { EyeState } from "./eyeState";
import { gazeDeviation, type GazeDirection } from "./gaze";
import { clamp } from "./geometry";
import type { MouthState } from "./mouthState";
import { RollingDutyCycle } from "./rolling";

export type AttentionLabel = "Atento" | "Alerta" | "Distraido";
export type RiskLevel = "ok" | "warn" | "danger";

export type AttentionEvent =
  | "Sem rosto"
  | "Microssono"
  | "Sonolencia"
  | "Fadiga"
  | "Bocejo"
  | "Olhar fora"
  | "Distracao acumulada"
  | "Celular"
  | "Celular suspeito";

export interface HeadPose {
  readonly yaw: number;
  readonly pitch: number;
  readonly roll: number;
}

export interface AttentionInput {
  readonly eyeState: EyeState;
  /** Null when the mouth is not tracked (no yawn penalty). */
  readonly mouthState: MouthState | null;
  readonly headPose: HeadPose | null;
  /**
   * Head pose + iris eye rotation. When given, the off-road rules use it
   * instead of the head pose alone; null/absent falls back to the head pose.
   */
  readonly gaze?: GazeDirection | null;
  readonly phonePresent: boolean;
  readonly phoneNearFace: boolean;
  readonly facePresent: boolean;
  /** monotonic time in seconds */
  readonly timestampS: number;
}

export interface AttentionState {
  readonly score: number;
  readonly rawScore: number;
  readonly label: AttentionLabel;
  readonly level: RiskLevel;
  readonly events: readonly AttentionEvent[];
  /** Seconds of off-road gaze within the last vatsWindowS (Euro NCAP VATS). */
  readonly offroadWindowS: number;
  /** Gaze deviation over the off-road thresholds (1 = at the limit); null without a pose. */
  readonly gazeDeviation: number | null;
}

export function labelForScore(score: number, config: ScoringConfig): AttentionLabel {
  if (score >= config.attentiveMinScore) {
    return "Atento";
  }
  if (score >= config.alertMinScore) {
    return "Alerta";
  }
  return "Distraido";
}

export function levelForLabel(label: AttentionLabel): RiskLevel {
  return label === "Atento" ? "ok" : label === "Alerta" ? "warn" : "danger";
}

/**
 * Fuses eye state, head pose and phone signals into a smoothed 0-100 score.
 * Mirrors dms/attention.py AttentionScorer; parity is enforced by
 * tests/web/scoringParity.test.ts against shared/scoring-scenarios.json.
 */
export class AttentionScorer {
  private smoothed: number;
  private offroadStart: number | null = null;
  private lastFaceTime: number | null = null;
  private readonly offroadWindow: RollingDutyCycle;

  constructor(private config: ScoringConfig) {
    this.smoothed = config.scoreBase;
    this.offroadWindow = new RollingDutyCycle(config.vatsWindowS, config.rollingMaxGapS);
  }

  /** Change rules without resetting the smoothed score or timers in progress. */
  setConfig(config: ScoringConfig): void {
    this.config = config;
  }

  reset(): void {
    this.smoothed = this.config.scoreBase;
    this.offroadStart = null;
    this.lastFaceTime = null;
    this.offroadWindow.reset();
  }

  update(input: AttentionInput): AttentionState {
    const { config } = this;
    const events: AttentionEvent[] = [];
    let penalty = 0;

    if (input.facePresent) {
      this.lastFaceTime = input.timestampS;
    } else if (this.lastFaceTime === null || input.timestampS - this.lastFaceTime > config.noFaceGraceS) {
      events.push("Sem rosto");
      penalty += config.penaltyNoFace;
    }

    if (input.eyeState.microsleep) {
      penalty += config.penaltyMicrosleep;
      events.push("Microssono");
    } else if (input.eyeState.drowsy) {
      penalty += config.penaltyDrowsy;
      events.push("Sonolencia");
    }

    const { perclos } = input.eyeState;
    if (perclos !== null && perclos >= config.perclosThreshold) {
      penalty += config.penaltyFatigue;
      events.push("Fadiga");
    }

    if (input.mouthState?.yawning) {
      penalty += config.penaltyYawn;
      events.push("Bocejo");
    }

    const pose: GazeDirection | null = input.gaze ?? input.headPose;
    const offroad =
      pose !== null && (Math.abs(pose.yaw) > config.yawThreshold || Math.abs(pose.pitch) > config.pitchThreshold);
    if (offroad) {
      if (this.offroadStart === null) {
        this.offroadStart = input.timestampS;
      } else if (input.timestampS - this.offroadStart >= config.offroadTimeS) {
        penalty += config.penaltyOffroad;
        events.push("Olhar fora");
      }
    } else {
      this.offroadStart = null;
    }

    if (pose !== null) {
      this.offroadWindow.add(input.timestampS, offroad);
    }
    if (this.offroadWindow.activeS >= config.vatsOffroadLimitS) {
      penalty += config.penaltyVats;
      events.push("Distracao acumulada");
    }

    if (input.phoneNearFace) {
      penalty += config.penaltyPhone;
      events.push("Celular");
    } else if (input.phonePresent) {
      penalty += config.penaltyPhoneSuspected;
      events.push("Celular suspeito");
    }

    const bonus = penalty === 0 ? config.bonusStable : 0;
    const rawScore = clamp(config.scoreBase - penalty + bonus, config.scoreFloor, config.scoreCeiling);
    this.smoothed = config.scoreSmoothing * this.smoothed + (1 - config.scoreSmoothing) * rawScore;
    const score = clamp(this.smoothed, config.scoreFloor, config.scoreCeiling);
    const label = labelForScore(score, config);

    return {
      score,
      rawScore,
      label,
      level: levelForLabel(label),
      events,
      offroadWindowS: this.offroadWindow.activeS,
      gazeDeviation: gazeDeviation(pose, config.yawThreshold, config.pitchThreshold),
    };
  }
}
