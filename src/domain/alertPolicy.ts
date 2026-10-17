import type { AttentionEvent } from "./attention";

export type AlertTone = "urgent" | "warning";

/** Euro NCAP: microsleep and sleep warnings must be more urgent than distraction warnings. */
const URGENT: ReadonlySet<AttentionEvent> = new Set(["Microssono", "Sonolencia"]);
const WARNING: ReadonlySet<AttentionEvent> = new Set(["Olhar fora", "Distracao acumulada", "Celular"]);
const DEFAULT_COOLDOWN_MS = 2500;

export function toneFor(events: readonly AttentionEvent[]): AlertTone | null {
  if (events.some((event) => URGENT.has(event))) {
    return "urgent";
  }
  return events.some((event) => WARNING.has(event)) ? "warning" : null;
}

/**
 * Decides when an audible alert plays: on event onsets only, never more than
 * once per cooldown, except that an urgent alert may interrupt a warning cooldown.
 * Fatigue, yawn and "suspected phone" stay visual-only.
 */
export class AlertPolicy {
  private lastMs = -Infinity;
  private lastTone: AlertTone | null = null;

  constructor(private readonly cooldownMs = DEFAULT_COOLDOWN_MS) {}

  reset(): void {
    this.lastMs = -Infinity;
    this.lastTone = null;
  }

  /** @param startedEvents events that started on this frame */
  next(startedEvents: readonly AttentionEvent[], nowMs: number): AlertTone | null {
    const tone = toneFor(startedEvents);
    if (tone === null) {
      return null;
    }
    const coolingDown = nowMs - this.lastMs < this.cooldownMs;
    const escalates = tone === "urgent" && this.lastTone === "warning";
    if (coolingDown && !escalates) {
      return null;
    }
    this.lastMs = nowMs;
    this.lastTone = tone;
    return tone;
  }
}
