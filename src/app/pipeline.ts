import { AttentionScorer, type AttentionEvent, type AttentionState } from "../domain/attention";
import type { ScoringConfig } from "../domain/config";
import { EyeStateTracker, type EyeState } from "../domain/eyeState";
import { MouthStateTracker, type MouthState } from "../domain/mouthState";
import { SessionStats } from "../domain/session";
import type { FrameSignals } from "./signals";

export interface PipelineResult {
  readonly signals: FrameSignals;
  readonly eyeState: EyeState;
  readonly mouthState: MouthState;
  readonly attention: AttentionState;
  readonly startedEvents: readonly AttentionEvent[];
}

/**
 * Runs frame signals through eye tracking, attention scoring and session
 * aggregation. Independent of DOM, camera and MediaPipe, so it is unit-testable
 * and identical for demo and camera sources.
 */
export class MonitoringPipeline {
  private readonly eyeTracker: EyeStateTracker;
  private readonly mouthTracker: MouthStateTracker;
  private readonly scorer: AttentionScorer;
  readonly session = new SessionStats();

  constructor(config: ScoringConfig) {
    this.eyeTracker = new EyeStateTracker(config);
    this.mouthTracker = new MouthStateTracker(config);
    this.scorer = new AttentionScorer(config);
  }

  /** Swap rules (e.g. sensitivity change) keeping the score, timers and session aggregates. */
  reconfigure(config: ScoringConfig): void {
    this.eyeTracker.setConfig(config);
    this.mouthTracker.setConfig(config);
    this.scorer.setConfig(config);
  }

  reset(): void {
    this.eyeTracker.reset();
    this.mouthTracker.reset();
    this.scorer.reset();
    this.session.reset();
  }

  process(signals: FrameSignals, timestampS: number): PipelineResult {
    const eyeState = this.eyeTracker.update(signals.facePresent ? signals.ear : null, timestampS);
    const mouthState = this.mouthTracker.update(signals.facePresent ? signals.mar : null, timestampS);
    const attention = this.scorer.update({
      eyeState,
      mouthState,
      headPose: signals.facePresent ? signals.headPose : null,
      phonePresent: signals.phonePresent,
      phoneNearFace: signals.phoneNearFace,
      facePresent: signals.facePresent,
      timestampS,
    });
    const startedEvents = this.session.add(attention, timestampS);
    return { signals, eyeState, mouthState, attention, startedEvents };
  }
}
