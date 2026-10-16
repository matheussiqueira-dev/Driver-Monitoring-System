interface Sample {
  readonly t: number;
  readonly active: boolean;
  /** Contribution of the interval that ends at this sample. */
  span: number;
  activeSpan: number;
}

/**
 * Time-weighted share of a boolean signal over a sliding window, used for
 * PERCLOS (eyes closed) and Euro NCAP VATS (gaze off-road). Mirrors
 * dms/rolling.py operation by operation so both runtimes give identical results.
 */
export class RollingDutyCycle {
  private samples: Sample[] = [];
  private head = 0;
  private total = 0;
  private activeTotal = 0;

  constructor(
    private readonly windowS: number,
    private readonly maxGapS: number,
  ) {}

  reset(): void {
    this.samples = [];
    this.head = 0;
    this.total = 0;
    this.activeTotal = 0;
  }

  add(timestampS: number, active: boolean): void {
    let span = 0;
    let activeSpan = 0;
    const previous = this.samples[this.samples.length - 1];
    if (previous) {
      // A long gap (face lost, dropped frames) only counts up to maxGapS.
      span = Math.min(Math.max(timestampS - previous.t, 0), this.maxGapS);
      activeSpan = previous.active ? span : 0;
    }
    this.samples.push({ t: timestampS, active, span, activeSpan });
    this.total += span;
    this.activeTotal += activeSpan;

    while (
      this.samples.length - this.head > 1 &&
      (this.samples[this.head]?.t ?? timestampS) < timestampS - this.windowS
    ) {
      this.head += 1;
      const first = this.samples[this.head];
      if (first) {
        this.total = Math.max(this.total - first.span, 0);
        this.activeTotal = Math.max(this.activeTotal - first.activeSpan, 0);
        first.span = 0;
        first.activeSpan = 0;
      }
    }
    // Compact occasionally instead of shifting on every frame.
    if (this.head > 1024) {
      this.samples = this.samples.slice(this.head);
      this.head = 0;
    }
  }

  get coveredS(): number {
    return this.total;
  }

  get activeS(): number {
    return this.activeTotal;
  }

  /** Active share of the window, or null until enough time has been observed. */
  ratio(minCoverageS: number): number | null {
    if (this.total <= 0 || this.total < minCoverageS) {
      return null;
    }
    return this.activeTotal / this.total;
  }
}
