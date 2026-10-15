import type { AttentionEvent, RiskLevel } from "../domain/attention";
import type { EventOnset, SessionSummary } from "../domain/session";
import type { PipelineResult } from "../app/pipeline";
import { requireElement, setText } from "./dom";

const RISK_COLORS: Record<RiskLevel, string> = { ok: "#56f08d", warn: "#f7c75b", danger: "#ff5e67" };
const RISK_TEXT: Record<RiskLevel, string> = { ok: "baixo risco", warn: "risco moderado", danger: "risco alto" };
const DASHBOARD_INTERVAL_MS = 250;
const RECENT_EVENTS = 4;

const EVENT_LEVEL: Record<AttentionEvent, RiskLevel> = {
  "Sem rosto": "warn",
  Microssono: "danger",
  Sonolencia: "danger",
  "Olhar fora": "warn",
  Celular: "danger",
  "Celular suspeito": "warn",
};

export type SourceMode = "demo" | "camera" | "review";

export interface FrameView {
  readonly result: PipelineResult;
  readonly mode: SourceMode;
  readonly fps: number;
  readonly statusMessage: string | null;
  readonly history: readonly number[];
  readonly onsets: readonly EventOnset[];
  readonly summary: SessionSummary;
  readonly debugLines: readonly string[] | null;
}

function formatClock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function insightFor(summary: SessionSummary, mode: SourceMode): string {
  if (summary.eventCount === 0) {
    return mode === "camera"
      ? "Sessao estavel ate aqui. Os sinais vem dos landmarks reais da camera, processados no navegador."
      : "Sessao estavel ate aqui. A demo passa pelas mesmas regras de score usadas com a camera.";
  }
  const [topEvent, topCount] = Object.entries(summary.eventsByType).sort((a, b) => b[1] - a[1])[0] ?? [];
  const riskShare = summary.durationS > 0 ? (summary.timeByLevelS.danger / summary.durationS) * 100 : 0;
  const tail = riskShare >= 1 ? ` ${riskShare.toFixed(0)}% do tempo em risco alto.` : "";
  return `${summary.eventCount} evento(s) em ${formatClock(summary.durationS)}; o mais frequente foi "${topEvent}" (${topCount}x).${tail}`;
}

/** Owns every DOM write of the monitor and dashboard sections. */
export class MonitorView {
  private readonly el = {
    scoreValue: requireElement("#scoreValue", HTMLElement),
    scoreLabel: requireElement("#scoreLabel", HTMLElement),
    earValue: requireElement("#earValue", HTMLElement),
    poseValue: requireElement("#poseValue", HTMLElement),
    fpsValue: requireElement("#fpsValue", HTMLElement),
    attentionFill: requireElement("#attentionFill", HTMLElement),
    alertStrip: requireElement("#alertStrip", HTMLElement),
    scoreChart: requireElement("#scoreChart", HTMLElement),
    eventList: requireElement("#eventList", HTMLUListElement),
    insightText: requireElement("#insightText", HTMLElement),
    riskState: requireElement("#riskState", HTMLElement),
    kpiScore: requireElement("#kpiScore", HTMLElement),
    kpiScoreNote: requireElement("#kpiScoreNote", HTMLElement),
    kpiEvents: requireElement("#kpiEvents", HTMLElement),
    kpiEventsNote: requireElement("#kpiEventsNote", HTMLElement),
    kpiEar: requireElement("#kpiEar", HTMLElement),
    kpiEarNote: requireElement("#kpiEarNote", HTMLElement),
    kpiFps: requireElement("#kpiFps", HTMLElement),
    trackingDebug: requireElement("#trackingDebug", HTMLPreElement),
  };
  private lastDashboardMs = -Infinity;
  private lastAlertText = "";
  private lastEventsKey = "";

  render(frame: FrameView, nowMs: number): void {
    const { attention, eyeState, signals } = frame.result;
    const pose = signals.facePresent ? signals.headPose : null;

    setText(this.el.scoreValue, attention.score.toFixed(1));
    setText(this.el.scoreLabel, `(${attention.label})`);
    setText(this.el.earValue, eyeState.ear === null ? "--" : eyeState.ear.toFixed(3));
    setText(
      this.el.poseValue,
      pose ? `${pose.yaw.toFixed(1)} / ${pose.pitch.toFixed(1)} / ${pose.roll.toFixed(1)}` : "--",
    );
    setText(this.el.fpsValue, frame.fps.toFixed(1));
    this.el.attentionFill.style.height = `${Math.max(8, attention.score)}%`;
    this.el.attentionFill.style.background = RISK_COLORS[attention.level];
    this.renderAlert(frame);

    const showDebug = frame.debugLines !== null;
    this.el.trackingDebug.hidden = !showDebug;
    if (showDebug) {
      setText(this.el.trackingDebug, frame.debugLines.join("\n"));
    }

    if (nowMs - this.lastDashboardMs >= DASHBOARD_INTERVAL_MS) {
      this.lastDashboardMs = nowMs;
      this.renderDashboard(frame);
    }
  }

  showStatus(message: string, level: RiskLevel): void {
    this.lastAlertText = message;
    setText(this.el.alertStrip, message);
    this.el.alertStrip.classList.toggle("is-warning", level === "warn");
    this.el.alertStrip.classList.toggle("is-danger", level === "danger");
  }

  private renderAlert(frame: FrameView): void {
    const { attention, signals } = frame.result;
    let text: string;
    if (frame.statusMessage) {
      text = frame.statusMessage;
    } else if (!signals.facePresent) {
      text = "Rosto nao detectado no frame atual";
    } else if (attention.events.length > 0) {
      text = `Alerta: ${attention.events.join(", ")}`;
    } else {
      text = "Condicao estavel: motorista atento";
    }
    // role="status" announces changes; only write when the message changes.
    if (text !== this.lastAlertText) {
      this.showStatus(text, frame.statusMessage ? "warn" : attention.level);
    }
  }

  private renderDashboard(frame: FrameView): void {
    const { summary, history } = frame;
    const { attention, eyeState } = frame.result;

    this.renderChart(history);
    setText(this.el.kpiScore, summary.averageScore === null ? "--" : summary.averageScore.toFixed(1));
    setText(
      this.el.kpiScoreNote,
      summary.minScore === null ? "aguardando amostras" : `minimo ${summary.minScore.toFixed(1)} na sessao`,
    );
    setText(this.el.kpiEvents, String(summary.eventCount));
    setText(
      this.el.kpiEventsNote,
      summary.eventCount === 0 ? "nenhum evento" : `em ${formatClock(summary.durationS)} de sessao`,
    );
    setText(this.el.kpiEar, eyeState.ear === null ? "--" : eyeState.ear.toFixed(3));
    setText(
      this.el.kpiEarNote,
      eyeState.ear === null
        ? "sem leitura"
        : eyeState.eyesClosed
          ? `olhos fechados ha ${eyeState.closedDuration.toFixed(1)}s`
          : `${eyeState.blinkRate} piscadas/min`,
    );
    setText(this.el.kpiFps, frame.fps.toFixed(1));
    setText(this.el.riskState, RISK_TEXT[attention.level]);
    setText(this.el.insightText, insightFor(summary, frame.mode));
    this.renderEvents(frame);
  }

  private renderChart(history: readonly number[]): void {
    const bars = this.el.scoreChart.children;
    while (bars.length < history.length) {
      const bar = document.createElement("span");
      bar.className = "score-bar";
      this.el.scoreChart.appendChild(bar);
    }
    while (bars.length > history.length) {
      bars[bars.length - 1]?.remove();
    }
    history.forEach((value, index) => {
      const bar = bars[index];
      if (bar instanceof HTMLElement) {
        bar.style.height = `${Math.max(6, value)}%`;
        bar.style.background = RISK_COLORS[value >= 70 ? "ok" : value >= 40 ? "warn" : "danger"];
      }
    });
    const last = history.at(-1);
    this.el.scoreChart.setAttribute(
      "aria-label",
      last === undefined
        ? "Grafico de tendencia do score: sem dados"
        : `Grafico de tendencia do score: ultimo valor ${last.toFixed(0)}`,
    );
  }

  private renderEvents(frame: FrameView): void {
    const active = frame.result.attention.events;
    const recent = frame.onsets.slice(-RECENT_EVENTS).reverse();
    const key = `${active.join("|")}#${recent.map((onset) => `${onset.type}@${onset.atS.toFixed(1)}`).join("|")}`;
    if (key === this.lastEventsKey) {
      return;
    }
    this.lastEventsKey = key;

    const items: Array<[RiskLevel, string]> = [];
    if (active.length === 0) {
      items.push(["ok", "Nenhum evento ativo"]);
    } else {
      for (const event of active) {
        items.push([EVENT_LEVEL[event], `Ativo: ${event}`]);
      }
    }
    for (const onset of recent) {
      items.push([EVENT_LEVEL[onset.type], `${formatClock(onset.atS)} ${onset.type}`]);
    }

    this.el.eventList.replaceChildren(
      ...items.map(([level, text]) => {
        const item = document.createElement("li");
        const dot = document.createElement("span");
        dot.className = `event-dot ${level}`;
        item.append(dot, document.createTextNode(text));
        return item;
      }),
    );
  }
}
