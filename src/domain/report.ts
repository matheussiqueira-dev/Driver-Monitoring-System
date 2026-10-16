import type { ScoringConfig } from "./config";
import type { EventOnset, SessionSummary } from "./session";

export const REPORT_SCHEMA_VERSION = 1;

export interface SessionReportInput {
  readonly generatedAt: Date;
  readonly mode: string;
  readonly sensitivity: number;
  readonly config: ScoringConfig;
  readonly summary: SessionSummary;
  readonly onsets: readonly EventOnset[];
  readonly scoreHistory: readonly number[];
}

export interface SessionReport {
  readonly schemaVersion: typeof REPORT_SCHEMA_VERSION;
  readonly generatedAt: string;
  readonly source: { readonly mode: string; readonly sensitivity: number };
  readonly thresholds: {
    readonly earThreshold: number;
    readonly perclosThreshold: number;
    readonly marThreshold: number;
    readonly yawThreshold: number;
    readonly pitchThreshold: number;
    readonly vatsOffroadLimitS: number;
    readonly vatsWindowS: number;
  };
  readonly summary: SessionSummary;
  readonly events: readonly EventOnset[];
  readonly scoreHistory: readonly number[];
}

const round = (value: number, digits = 3): number => Number(value.toFixed(digits));

/** Portable snapshot of a monitoring session; stays on the device unless the user shares the file. */
export function buildSessionReport(input: SessionReportInput): SessionReport {
  const { config, summary } = input;
  return {
    schemaVersion: REPORT_SCHEMA_VERSION,
    generatedAt: input.generatedAt.toISOString(),
    source: { mode: input.mode, sensitivity: input.sensitivity },
    thresholds: {
      earThreshold: config.earThreshold,
      perclosThreshold: config.perclosThreshold,
      marThreshold: config.marThreshold,
      yawThreshold: round(config.yawThreshold),
      pitchThreshold: round(config.pitchThreshold),
      vatsOffroadLimitS: config.vatsOffroadLimitS,
      vatsWindowS: config.vatsWindowS,
    },
    summary: {
      ...summary,
      durationS: round(summary.durationS),
      averageScore: summary.averageScore === null ? null : round(summary.averageScore, 1),
      minScore: summary.minScore === null ? null : round(summary.minScore, 1),
      timeByLevelS: {
        ok: round(summary.timeByLevelS.ok),
        warn: round(summary.timeByLevelS.warn),
        danger: round(summary.timeByLevelS.danger),
      },
    },
    events: input.onsets.map((onset) => ({ type: onset.type, atS: round(onset.atS) })),
    scoreHistory: input.scoreHistory.map((score) => round(score, 1)),
  };
}

function csvCell(value: string | number): string {
  const text = String(value);
  // Quote and neutralise leading formula characters so spreadsheets never evaluate a cell.
  const safe = /^[=+\-@]/.test(text) ? `'${text}` : text;
  return /[",\n]/.test(safe) || safe !== text ? `"${safe.replaceAll('"', '""')}"` : safe;
}

/** One row per event onset, for spreadsheets. */
export function eventsToCsv(report: SessionReport): string {
  const rows = [
    ["offset_s", "event", "mode"],
    ...report.events.map((event) => [event.atS, event.type, report.source.mode]),
  ];
  return `${rows.map((row) => row.map(csvCell).join(",")).join("\n")}\n`;
}
