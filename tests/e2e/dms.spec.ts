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

test("camera mode loads Face Landmarker under the production CSP", async ({ page }) => {
  test.slow();
  const errors = collectErrors(page);
  await page.goto("/");
  await page.getByRole("button", { name: "Camera" }).click();

  // The fake camera shows no face; what matters is that video, WASM and model start.
  const debug = page.locator("#trackingDebug");
  await expect(debug).toBeVisible({ timeout: 30_000 });
  await expect(debug).toContainText(/faceMesh: ready \((GPU|CPU)\)/, { timeout: 45_000 });
  await expect(page.getByRole("button", { name: "Camera" })).toHaveAttribute("aria-pressed", "true");
  expect(errors).toEqual([]);
});
