import type { AlertTone } from "../domain/alertPolicy";

const STORAGE_KEY = "dms.audioAlerts";
const VOLUME = 0.18;

/** Tone patterns inside the 200-8000 Hz band recommended by EU ADDW (Reg. 2023/2590). */
const PATTERNS: Record<AlertTone, readonly { readonly hz: number; readonly startS: number; readonly durS: number }[]> =
  {
    warning: [{ hz: 740, startS: 0, durS: 0.16 }],
    urgent: [
      { hz: 1050, startS: 0, durS: 0.14 },
      { hz: 1400, startS: 0.18, durS: 0.14 },
      { hz: 1050, startS: 0.36, durS: 0.22 },
    ],
  };

/**
 * Optional audible alerts with Web Audio. Off by default; the AudioContext is
 * created on the user's toggle (autoplay policy). The preference is remembered
 * per browser.
 */
export class AudioAlert {
  private context: AudioContext | null = null;
  private enabledValue = false;

  get enabled(): boolean {
    return this.enabledValue;
  }

  /** Previous choice, or false when storage is unavailable. */
  static savedPreference(): boolean {
    try {
      return localStorage.getItem(STORAGE_KEY) === "on";
    } catch {
      return false;
    }
  }

  /** Must be called from a user gesture so the browser lets the context start. */
  setEnabled(enabled: boolean): void {
    this.enabledValue = enabled;
    try {
      localStorage.setItem(STORAGE_KEY, enabled ? "on" : "off");
    } catch {
      // Private mode or blocked storage: the toggle still works for this visit.
    }
    if (enabled) {
      this.context ??= new AudioContext();
      void this.context.resume();
    } else {
      void this.context?.suspend();
    }
  }

  play(tone: AlertTone): void {
    const context = this.context;
    if (!this.enabledValue || !context || context.state !== "running") {
      return;
    }
    const start = context.currentTime + 0.01;
    for (const { hz, startS, durS } of PATTERNS[tone]) {
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.type = "sine";
      oscillator.frequency.value = hz;
      // Short attack/release envelope avoids clicks.
      const t0 = start + startS;
      gain.gain.setValueAtTime(0, t0);
      gain.gain.linearRampToValueAtTime(VOLUME, t0 + 0.015);
      gain.gain.setValueAtTime(VOLUME, t0 + durS - 0.03);
      gain.gain.linearRampToValueAtTime(0, t0 + durS);
      oscillator.connect(gain).connect(context.destination);
      oscillator.start(t0);
      oscillator.stop(t0 + durS + 0.01);
    }
  }
}
