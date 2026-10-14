import { test, expect } from "@playwright/test";
import path from "node:path";
import AxeBuilder from "@axe-core/playwright";

test("live backend: benchmark lifecycle, history and all populated workspaces", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Explore benchmark demo" }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Explore benchmark demo" }).click();
  await expect(
    page.getByRole("heading", { name: "Model overview", exact: true }),
  ).toBeVisible({ timeout: 120000 });
  await expect(page.getByText(/SYNTHETIC BENCHMARK \/ INV-/)).toBeVisible();
  for (const name of [
    "Failure Atlas",
    "Experiments",
    "Search Coverage",
    "Overview",
  ]) {
    await page
      .getByRole("navigation", { name: "Workspaces" })
      .getByRole("button", { name, exact: true })
      .click();
    await expect(page.locator(".pageIntro h1")).toBeVisible();
  }
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Model overview", exact: true }),
  ).toBeVisible();
  const saved = await page.getByLabel("Previous investigations").inputValue();
  expect(saved).toMatch(/^INV-/);
  const previousIds = await page
    .getByRole("combobox", { name: "Previous investigations" })
    .locator("option")
    .evaluateAll((options) =>
      options.map((o) => (o as HTMLOptionElement).value).filter(Boolean),
    );
  const previousId = previousIds.find((id) => id !== saved)!;
  expect(previousId).toBeTruthy();
  await page
    .getByRole("combobox", { name: "Previous investigations" })
    .selectOption(previousId);
  await expect(page.locator(".pageIntro .eyebrow")).toContainText(previousId);

  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Regression", exact: true })
    .click();
  await page.getByRole("button", { name: "Compare demo versions" }).click();
  await expect(page.getByText("SYNTHETIC BENCHMARK COMPARISON")).toBeVisible({
    timeout: 120000,
  });
  expect(errors).toEqual([]);
});

test("live backend: real model upload, new investigation and uploaded regression", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page
    .getByRole("button", { name: "Investigate a model", exact: true })
    .click();
  await page
    .getByLabel("MODEL ARTIFACT")
    .setInputFiles(path.resolve(".test-artifacts/model-v1.joblib"));
  await page
    .getByLabel("EVALUATION DATASET", { exact: true })
    .setInputFiles(path.resolve(".test-artifacts/evaluation.csv"));
  await page
    .getByRole("button", { name: "Investigate uploaded model" })
    .click();
  await expect(
    page.getByRole("heading", { name: "Model overview", exact: true }),
  ).toBeVisible({ timeout: 120000 });
  await expect(page.getByText(/UPLOADED MODEL \/ INV-/)).toBeVisible();
  await page
    .getByRole("button", { name: "New investigation", exact: true })
    .first()
    .click();
  await expect(
    page.getByRole("region", { name: "New investigation" }),
  ).toBeVisible();
  const firstRun = await page.locator('.pageIntro .eyebrow').innerText();
  const firstId = firstRun.match(/INV-[A-Z0-9]+/)![0];
  await page.getByRole('button', {name:'Investigate uploaded model'}).click();
  await expect(page.getByRole('region', {name:'New investigation'})).toHaveCount(0, {timeout:120000});
  await expect(page.locator('.pageIntro .eyebrow')).not.toContainText(firstId);


  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Regression", exact: true })
    .click();
  await page
    .getByLabel("MODEL VERSION ARTIFACT")
    .setInputFiles(path.resolve(".test-artifacts/model-v2.joblib"));
  await page
    .getByRole("button", { name: "Register model", exact: true })
    .click();
  await expect(page.getByLabel("MODEL V2", { exact: true }).locator('option').filter({hasText:'model-v2.joblib'})).toHaveCount(1);
  await expect(page.getByLabel("MODEL VERSION ARTIFACT")).toHaveValue("");
  await page.getByLabel("MODEL VERSION ARTIFACT").setInputFiles(path.resolve(".test-artifacts/model-v2.joblib"));
  await expect(page.getByRole("button", {name:"Register model",exact:true})).toBeEnabled();
  await page.getByRole("button", {name:"Register model",exact:true}).click();
  await expect(page.getByLabel("MODEL V2", {exact:true}).locator('option').filter({hasText:'model-v2.joblib'})).toHaveCount(2);
  const modelLabels = await page.getByLabel("MODEL V2", {exact:true}).locator("option").allTextContents();
  expect(new Set(modelLabels).size).toBe(modelLabels.length);
  const ids = await page
    .getByLabel("MODEL V1", { exact: true })
    .locator("option")
    .evaluateAll((opts) =>
      opts.map((o) => (o as HTMLOptionElement).value).filter(Boolean),
    );
  await page.getByLabel("MODEL V1", { exact: true }).selectOption(ids[0]);
  await page.getByLabel("MODEL V2", { exact: true }).selectOption(ids[0]);
  await expect(page.getByText("Choose different model assets for V1 and V2 to compare versions.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Compare models", exact: true })).toBeDisabled();
  await page.getByLabel("MODEL V2", { exact: true }).selectOption(ids.at(-1)!);
  const datasets = await page
    .getByLabel("DATASET", { exact: true })
    .locator("option")
    .evaluateAll((opts) =>
      opts.map((o) => (o as HTMLOptionElement).value).filter(Boolean),
    );
  await page.getByLabel("DATASET", { exact: true }).selectOption(datasets[0]);
  await page
    .getByRole("button", { name: "Compare models", exact: true })
    .click();
  await expect(page.getByLabel("MODEL V1", { exact: true })).toBeDisabled();
  await expect(page.getByLabel("REGRESSION METRIC", { exact: true })).toBeDisabled();
  await expect(page.getByText("UPLOADED MODEL COMPARISON")).toBeVisible({
    timeout: 120000,
  });
  await expect(
    page.getByRole("heading", { name: "Failure movement" }),
  ).toBeVisible();
  for (const theme of ["light"]) {

    expect((await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"]).analyze()).violations).toEqual([]);
    await page.evaluate(() => { if (document.activeElement instanceof HTMLElement) document.activeElement.blur(); window.scrollTo(0, 0); });
    await page.screenshot({ path: path.resolve(`.test-artifacts/uploaded-regression-${theme}.png`), fullPage: true });
  }
  await page.reload();
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Regression", exact: true })
    .click();
  await expect(
    page.getByLabel("MODEL V1", { exact: true }).locator("option"),
  ).toHaveCount(ids.length + 1);
  expect(errors).toEqual([]);
});
