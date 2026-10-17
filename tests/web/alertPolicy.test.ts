import { describe, expect, it } from "vitest";
import { AlertPolicy, toneFor } from "../../src/domain/alertPolicy";

describe("toneFor", () => {
  it("makes drowsiness more urgent than distraction", () => {
    expect(toneFor(["Olhar fora", "Microssono"])).toBe("urgent");
    expect(toneFor(["Sonolencia"])).toBe("urgent");
    expect(toneFor(["Distracao acumulada"])).toBe("warning");
    expect(toneFor(["Celular"])).toBe("warning");
  });

  it("keeps fatigue, yawn and suspected phone visual-only", () => {
    expect(toneFor(["Fadiga", "Bocejo", "Celular suspeito", "Sem rosto"])).toBeNull();
    expect(toneFor([])).toBeNull();
  });
});

describe("AlertPolicy", () => {
  it("rate-limits repeated alerts", () => {
    const policy = new AlertPolicy(2500);
    expect(policy.next(["Olhar fora"], 0)).toBe("warning");
    expect(policy.next(["Celular"], 1000)).toBeNull();
    expect(policy.next(["Celular"], 2600)).toBe("warning");
  });

  it("lets an urgent alert interrupt a warning cooldown, but not another urgent one", () => {
    const policy = new AlertPolicy(2500);
    policy.next(["Olhar fora"], 0);
    expect(policy.next(["Microssono"], 500)).toBe("urgent");
    expect(policy.next(["Sonolencia"], 900)).toBeNull();
  });

  it("starts fresh after reset", () => {
    const policy = new AlertPolicy(2500);
    policy.next(["Olhar fora"], 0);
    policy.reset();
    expect(policy.next(["Olhar fora"], 100)).toBe("warning");
  });
});
