import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

/** WCAG 2.2 A/AA rules checked by axe-core on the production build. */
const WCAG_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];

test("home page has no WCAG A/AA violations", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#kpiScore")).not.toHaveText("--");

  const results = await new AxeBuilder({ page }).withTags(WCAG_TAGS).analyze();
  const summary = results.violations.map((violation) => ({
    id: violation.id,
    impact: violation.impact,
    targets: violation.nodes.slice(0, 3).map((node) => node.target.join(" ")),
  }));
  expect(summary).toEqual([]);
});

test("controls are reachable and operable by keyboard", async ({ page }) => {
  await page.goto("/");
  const review = page.getByRole("button", { name: "Review" });
  await review.focus();
  await page.keyboard.press("Enter");
  await expect(review).toHaveAttribute("aria-pressed", "true");

  const sound = page.getByLabel("Som");
  await sound.focus();
  await page.keyboard.press("Space");
  await expect(sound).toBeChecked();
});
