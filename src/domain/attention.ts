import type { ScoringConfig } from "./config";
import type { EyeState } from "./eyeState";
import { clamp } from "./geometry";

export type AttentionLabel = "Atento" | "Alerta" | "Distraido";
export type RiskLevel = "ok" | "warn" | "danger";

export type AttentionEvent = "Sem rosto" | "Microssono" | "Sonolencia" | "Olhar fora" | "Celular" | "Celular suspeito";

export interface HeadPose {
  readonly yaw: number;
  readonly pitch: number;
  readonly roll: number;
}

export interface AttentionInput {
  readonly eyeState: EyeState;
  readonly headPose: HeadPose | null;
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

  constructor(private readonly config: ScoringConfig) {
    this.smoothed = config.scoreBase;
  }

  reset(): void {
    this.smoothed = this.config.scoreBase;
    this.offroadStart = null;
    this.lastFaceTime = null;
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

    const pose = input.headPose;
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

    return { score, rawScore, label, level: levelForLabel(label), events };
  }
}
