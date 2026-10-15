import type { AttentionEvent, AttentionState, RiskLevel } from "./attention";

export interface EventOnset {
  readonly type: AttentionEvent;
  readonly atS: number;
}

export interface SessionSummary {
  readonly durationS: number;
  readonly averageScore: number | null;
  readonly minScore: number | null;
  readonly eventCount: number;
  readonly eventsByType: Readonly<Partial<Record<AttentionEvent, number>>>;
  readonly timeByLevelS: Readonly<Record<RiskLevel, number>>;
}

const DEFAULT_SAMPLE_INTERVAL_S = 0.7;
const DEFAULT_HISTORY_SIZE = 24;
const MAX_ONSETS = 500;

/**
 * Aggregates a monitoring session: score trend, event onsets (an event counts
 * once when it starts, not on every frame) and time spent per risk level.
 */
export class SessionStats {
  private startS: number | null = null;
  private lastS: number | null = null;
  private lastLevel: RiskLevel | null = null;
  private activeEvents = new Set<AttentionEvent>();
  private readonly onsetList: EventOnset[] = [];
  private readonly trend: number[] = [];
  private lastTrendS = -Infinity;
  private scoreSum = 0;
  private scoreCount = 0;
  private minScoreValue: number | null = null;
  private readonly levelTime: Record<RiskLevel, number> = { ok: 0, warn: 0, danger: 0 };

  constructor(
    private readonly sampleIntervalS = DEFAULT_SAMPLE_INTERVAL_S,
    private readonly historySize = DEFAULT_HISTORY_SIZE,
  ) {}

  reset(): void {
    this.startS = null;
    this.lastS = null;
    this.lastLevel = null;
    this.activeEvents = new Set();
    this.onsetList.length = 0;
    this.trend.length = 0;
    this.lastTrendS = -Infinity;
    this.scoreSum = 0;
    this.scoreCount = 0;
    this.minScoreValue = null;
    this.levelTime.ok = 0;
    this.levelTime.warn = 0;
    this.levelTime.danger = 0;
  }

  /** @returns events that started on this sample */
  add(state: AttentionState, timestampS: number): readonly AttentionEvent[] {
    this.startS ??= timestampS;
    if (this.lastS !== null && this.lastLevel !== null && timestampS > this.lastS) {
      this.levelTime[this.lastLevel] += timestampS - this.lastS;
    }
    this.lastS = timestampS;
    this.lastLevel = state.level;

    const started = state.events.filter((event) => !this.activeEvents.has(event));
    for (const type of started) {
      this.onsetList.push({ type, atS: timestampS - this.startS });
    }
    if (this.onsetList.length > MAX_ONSETS) {
      this.onsetList.splice(0, this.onsetList.length - MAX_ONSETS);
    }
    this.activeEvents = new Set(state.events);

    if (timestampS - this.lastTrendS >= this.sampleIntervalS) {
      this.trend.push(state.score);
      if (this.trend.length > this.historySize) {
        this.trend.shift();
      }
      this.lastTrendS = timestampS;
      this.scoreSum += state.score;
      this.scoreCount += 1;
      this.minScoreValue = this.minScoreValue === null ? state.score : Math.min(this.minScoreValue, state.score);
    }
    return started;
  }

  get history(): readonly number[] {
    return this.trend;
  }

  get onsets(): readonly EventOnset[] {
    return this.onsetList;
  }

  summary(): SessionSummary {
    const eventsByType: Partial<Record<AttentionEvent, number>> = {};
    for (const onset of this.onsetList) {
      eventsByType[onset.type] = (eventsByType[onset.type] ?? 0) + 1;
    }
    return {
      durationS: this.startS === null || this.lastS === null ? 0 : this.lastS - this.startS,
      averageScore: this.scoreCount === 0 ? null : this.scoreSum / this.scoreCount,
      minScore: this.minScoreValue,
      eventCount: this.onsetList.length,
      eventsByType,
      timeByLevelS: { ...this.levelTime },
    };
  }
}
