import { fileURLToPath } from "node:url";
import { expect, test, type ConsoleMessage, type Page } from "@playwright/test";

/** Console errors, ignoring MediaPipe's native INFO lines that it prints through console.error. */
function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (message: ConsoleMessage) => {
    if (message.type() === "error" && !/^(INFO|W\d{4}|WARNING):?/.test(message.text())) {
      errors.push(message.text());
    }
  });
  page.on("pageerror", (error) => errors.push(error.message));
  return errors;
}

test("serves the production security headers", async ({ page }) => {
  const response = await page.goto("/");
  const headers = response?.headers() ?? {};

  expect(headers["content-security-policy"]).toContain("default-src 'self'");
  expect(headers["x-frame-options"]).toBe("DENY");
  await expect(page).toHaveTitle(/Driver Monitoring System/);
});

test("demo runs the scoring pipeline and the dashboard updates", async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto("/");

  await expect(page.locator("#scoreValue")).not.toHaveText("");
  await expect(page.locator("#kpiScore")).not.toHaveText("--");
  await expect(page.locator("#kpiFps")).not.toHaveText("--");
  expect(errors).toEqual([]);
});

test("review trip raises drowsiness, yawn and fatigue events", async ({ page }) => {
  test.slow();
  const errors = collectErrors(page);
  await page.goto("/");
  await page.getByRole("button", { name: "Review" }).click();
  await expect(page.getByRole("button", { name: "Review" })).toHaveAttribute("aria-pressed", "true");

  const events = page.locator("#eventList");
  await expect(events).toContainText("Sonolencia", { timeout: 15_000 });
  await expect(events).toContainText("Bocejo", { timeout: 30_000 });
  await expect(page.locator("#kpiPerclos")).toHaveText(/%$/);
  expect(errors).toEqual([]);
});

test("audible alert plays the urgent pattern when drowsiness starts", async ({ page }) => {
  test.slow();
  await page.addInitScript(() => {
    const counter = { oscillators: 0, frequencies: [] as number[] };
    (window as unknown as { __audio: typeof counter }).__audio = counter;
    const original = Object.getOwnPropertyDescriptor(BaseAudioContext.prototype, "createOscillator")?.value as (
      this: BaseAudioContext,
    ) => OscillatorNode;
    BaseAudioContext.prototype.createOscillator = function (this: BaseAudioContext) {
      const oscillator = original.call(this);
      counter.oscillators += 1;
      queueMicrotask(() => counter.frequencies.push(oscillator.frequency.value));
      return oscillator;
    };
  });
  await page.goto("/");
  await page.getByLabel("Som").check();
  await page.getByRole("button", { name: "Review" }).click();
  await expect(page.locator("#eventList")).toContainText("Sonolencia", { timeout: 15_000 });

  const audio = await page.evaluate(
    () => (window as unknown as { __audio: { oscillators: number; frequencies: number[] } }).__audio,
  );
  expect(audio.oscillators).toBeGreaterThanOrEqual(3);
  expect(audio.frequencies).toContain(1400);
  expect(audio.frequencies.every((hz) => hz >= 200 && hz <= 8000)).toBe(true);
});

test("exports the session as JSON and CSV", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#kpiScore")).not.toHaveText("--");

  const [json] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Exportar JSON" }).click(),
  ]);
  expect(json.suggestedFilename()).toMatch(/^dms-sessao-.*\.json$/);
  const jsonPath = await json.path();
  const report = JSON.parse(await (await import("node:fs/promises")).readFile(jsonPath, "utf8")) as {
    schemaVersion: number;
    source: { mode: string };
  };
  expect(report.schemaVersion).toBe(1);
  expect(report.source.mode).toBe("demo");

  const [csv] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Exportar CSV" }).click(),
  ]);
  expect(csv.suggestedFilename()).toMatch(/\.csv$/);
});

test("camera mode loads Face Landmarker and the phone detector under the production CSP", async ({ page }) => {
  test.slow();
  const errors = collectErrors(page);
  await page.goto("/");
  await page.getByRole("button", { name: "Camera" }).click();

  // The fake camera shows no face; what matters is that video, WASM and model start.
  const debug = page.locator("#trackingDebug");
  await expect(debug).toBeVisible({ timeout: 30_000 });
  await expect(debug).toContainText(/faceMesh: ready \((GPU|CPU)\)/, { timeout: 45_000 });
  // Phone detection loads in the background and shares the same WASM fileset.
  await expect(debug).toContainText(/phoneDetector: ready/, { timeout: 45_000 });
  await expect(page.getByRole("button", { name: "Camera" })).toHaveAttribute("aria-pressed", "true");
  expect(errors).toEqual([]);
});

test.describe("with a photo as the camera", () => {
  // The app's service worker would answer the fixture request itself, bypassing page.route.
  test.use({ serviceWorkers: "block" });

  test("camera mode flags a phone held next to the face", async ({ page }) => {
    test.slow();
    const errors = collectErrors(page);
    const fixture = fileURLToPath(new URL("./fixtures/driver-phone.jpg", import.meta.url));
    // Same-origin path, so the production CSP (img-src 'self') allows it.
    await page.route("**/__e2e__/driver-phone.jpg", (route) =>
      route.fulfill({ path: fixture, contentType: "image/jpeg" }),
    );
    // Replace the camera with a canvas stream that keeps painting the fixture photo.
    await page.addInitScript(() => {
      navigator.mediaDevices.getUserMedia = async () => {
        const image = new Image();
        image.src = "/__e2e__/driver-phone.jpg";
        await image.decode();
        const canvas = document.createElement("canvas");
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        const context = canvas.getContext("2d");
        const paint = (): void => {
          context?.drawImage(image, 0, 0);
          requestAnimationFrame(paint);
        };
        paint();
        return canvas.captureStream(30);
      };
    });

    await page.goto("/");
    await page.getByRole("button", { name: "Camera" }).click();

    const debug = page.locator("#trackingDebug");
    await expect(debug).toContainText(/phoneDetector: ready \([1-9]\)/, { timeout: 60_000 });
    await expect(page.locator("#eventList")).toContainText(/Celular(?! suspeito)/, { timeout: 15_000 });
    expect(errors).toEqual([]);
  });
});
