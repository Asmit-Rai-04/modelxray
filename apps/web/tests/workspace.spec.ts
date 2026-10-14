import { test, expect, Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import fs from "node:fs";
import path from "node:path";

const result = JSON.parse(
  fs.readFileSync(path.resolve(".test-artifacts/investigation.json"), "utf8"),
);
const comparison = JSON.parse(
  fs.readFileSync(path.resolve(".test-artifacts/comparison.json"), "utf8"),
);
const base = "http://localhost:8001";
async function mockApi(page: Page, loaded = false) {
  await page.route(`${base}/**`, async (route) => {
    const url = new URL(route.request().url());
    let data: unknown = { status: "ok" };
    if (url.pathname === "/api/v1/investigations")
      data = {
        items: loaded
          ? [
              {
                investigation_id: result.investigation_id,
                mode: result.mode,
                model: result.model,
                created_at: "Backend test fixture",
                status: "COMPLETED",
              },
            ]
          : [],
      };
    else if (url.pathname.startsWith("/api/v1/investigations/")) data = result;
    else if (
      url.pathname === "/api/v1/jobs/demo/investigate" ||
      url.pathname === "/api/v1/jobs/investigate/uploaded"
    )
      data = {
        job_id: "TEST-JOB",
        kind: "DEMO_INVESTIGATION",
        status: "PENDING",
      };
    else if (url.pathname === "/api/v1/jobs/TEST-JOB")
      data = {
        job_id: "TEST-JOB",
        kind: "DEMO_INVESTIGATION",
        status: "COMPLETED",
        investigation_id: result.investigation_id,
        progress: 1,
      };
    else if (
      url.pathname === "/api/v1/demo/compare" ||
      url.pathname === "/api/v1/regression/uploaded"
    )
      data = comparison;
    else if (url.pathname.startsWith("/api/v1/assets/"))
      data = {
        asset_id: url.pathname.endsWith("model")
          ? "MODEL-TEST"
          : "DATASET-TEST",
        filename: "test artifact",
      };
    await route.fulfill({ json: data });
  });
  if (loaded)
    await page.addInitScript(
      ({ base, id }) => localStorage.setItem(`${base}:last-run`, id),
      { base, id: result.investigation_id },
    );
}
async function go(page: Page, loaded = false) {
  await mockApi(page, loaded);
  await page.goto("/");
  await expect(
    page
      .getByRole("button", { name: "New investigation", exact: true })
      .first(),
  ).toBeEnabled();
}
async function navigate(page: Page, label: string) {
  await page
    .getByRole("navigation", { name: "Workspaces" })
    .getByRole("button", { name: label, exact: true })
    .click();
}

test("primary action opens labeled upload setup; demo is explicit", async ({
  page,
}) => {
  await go(page);
  await page
    .getByRole("button", { name: "Investigate a model", exact: true })
    .click();
  await expect(
    page.getByRole("region", { name: "New investigation" }),
  ).toBeVisible();
  await expect(page.getByLabel("TARGET COLUMN", { exact: true })).toHaveValue(
    "target",
  );
  await expect(
    page.getByRole("button", { name: "Investigate uploaded model" }),
  ).toBeDisabled();
  await expect(page.getByLabel("MODEL ARTIFACT")).toBeFocused();
});

test("landing explains the real protocol and keeps the ledger compact", async ({
  page,
}) => {
  await go(page);
  await expect(page.locator(".ledgerDisclosure")).not.toHaveAttribute("open");
  const guide = page.getByRole("group", {
    name: "Explore the investigation protocol",
  });
  await guide.getByRole("button", { name: "Validate", exact: true }).click();
  await expect(
    guide.getByRole("button", { name: "Validate", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(
    page.getByRole("heading", { name: "Ask the evidence to hold up." }),
  ).toBeVisible();
  await expect(page.locator("[data-node=validate]")).toHaveClass(/protocolSelected/);
  await expect(page.locator(".protocolHalo")).toHaveCSS("animation-name", "protocol-pulse");
  await expect(page.locator(".selectedTrace")).toHaveCSS("animation-name", "protocol-flow");
  await expect(page.locator(".protocolHalo")).toHaveCSS("animation-iteration-count", "3");
  await page.setViewportSize({ width: 390, height: 844 });
  await guide.getByRole("button", { name: "Profile", exact: true }).click();
  await guide.getByRole("button", { name: "Validate", exact: true }).press("Enter");
  await expect(page.getByRole("heading", { name: "Ask the evidence to hold up." })).toBeVisible();
  await expect.poll(() => page.locator(".topologyViewport").evaluate(el => el.scrollLeft)).toBeGreaterThan(0);
  await expect(page.locator("[data-node=validate] .nodeValue")).toBeVisible();
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect(page.locator(".protocolHalo")).toHaveCSS("animation-name", "none");
  await expect(page.locator(".selectedTrace")).toHaveCSS("animation-name", "none");
  for (const width of [768, 1024, 1280, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await page.locator(".topologyViewport").evaluate(el => el.querySelector("svg")!.getBoundingClientRect().width <= el.clientWidth + 1)).toBeTruthy();
  }
  await page.locator(".ledgerDisclosure > summary").click();
  await expect(
    page.getByRole("combobox", { name: "Previous investigations" }),
  ).toBeVisible();
});

test("plotted regions connect to atlas evidence and their source experiment", async ({
  page,
}) => {
  await go(page, true);
  const failure = result.failure_atlas[1];
  await page
    .getByRole("button", {
      name: `Inspect region 2: ${failure.condition}`,
      exact: true,
    })
    .click();
  await expect(page.locator(".failureDetail h2")).toHaveText(failure.condition);
  await page.getByRole("button", { name: "Inspect source experiment" }).click();
  await expect(
    page.getByLabel("Search hypotheses", { exact: true }),
  ).toHaveValue(failure.experiment_id);
  await expect(
    page.getByRole("table", { name: "Executed hypotheses" }).getByRole("row"),
  ).toHaveCount(2);
  await expect(
    page.getByRole("table", { name: "Executed hypotheses" }),
  ).toContainText(failure.experiment_id);
});

test("experiment filtering, reset and chart expansion retain actual backend entries", async ({
  page,
}) => {
  await go(page, true);
  await navigate(page, "Experiments");
  const observations = result.active_investigation.observations;
  await expect(
    page.getByText(
      `Plotting 12 of ${observations.length} entries. The evidence table includes every entry.`,
      { exact: true },
    ),
  ).toBeVisible();
  await page
    .getByRole("button", {
      name: `Show all ${observations.length} entries`,
      exact: true,
    })
    .click();
  await expect(
    page.getByRole("button", { name: "Show compact plot" }),
  ).toHaveAttribute("aria-expanded", "true");
  await page
    .getByLabel("Validation state", { exact: true })
    .selectOption("VALIDATED");
  await expect(
    page.getByRole("table", { name: "Executed hypotheses" }).getByRole("row"),
  ).toHaveCount(
    observations.filter((o: { validated: boolean }) => o.validated).length + 1,
  );
  await page
    .getByLabel("Search hypotheses", { exact: true })
    .fill("no-such-hypothesis");
  await expect(
    page.getByRole("heading", { name: "No matching hypotheses." }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Reset experiment filters" }).click();
  await expect(
    page.getByRole("table", { name: "Executed hypotheses" }).getByRole("row"),
  ).toHaveCount(observations.length + 1);
});

test("coverage exposes unexecuted features from the returned candidate set", async ({
  page,
}) => {
  await go(page, true);
  await navigate(page, "Search Coverage");
  const families = Object.values(
    result.active_investigation.search_coverage.families,
  ) as Array<{ candidate_features: string[]; executed_features: string[] }>;
  const index = families.findIndex((f) =>
    f.candidate_features.some(
      (feature) => !f.executed_features.includes(feature),
    ),
  );
  const expected = families[index].candidate_features.filter(
    (feature) => !families[index].executed_features.includes(feature),
  );
  const card = page.locator(".familyCard").nth(index);
  await card.locator(".coverageBlindSpots summary").click();
  expect(
    await card.locator(".coverageBlindSpots code").allTextContents(),
  ).toEqual(expected);
  await expect(
    page.getByText(/behaviors outside that set cannot be quantified/),
  ).toBeVisible();
});

test("regression transition filters expose every returned match", async ({
  page,
}) => {
  await go(page);
  await navigate(page, "Regression");
  await page.getByRole("button", { name: "Compare demo versions" }).click();
  const filters = page.getByRole("group", {
    name: "Filter regression transitions",
  });
  await filters.getByRole("button", { name: /^New / }).click();
  await expect(page.locator(".regressionListRow")).toHaveCount(
    comparison.summary.deltas.filter(
      (delta: { status: string }) => delta.status === "NEW",
    ).length,
  );
  await filters.getByRole("button", { name: /^All transitions / }).click();
  await expect(page.locator(".regressionListRow")).toHaveCount(
    comparison.summary.deltas.length,
  );
});
test("all five workspaces have distinct empty states and working navigation", async ({
  page,
}) => {
  await go(page);
  for (const [tab, title] of [
    ["Failure Atlas", "No investigation loaded."],
    ["Experiments", "No experiments to inspect."],
    ["Search Coverage", "No search coverage to inspect."],
    ["Regression", "No comparison has been run yet."],
  ]) {
    await navigate(page, tab);
    await expect(
      page.getByRole("heading", { name: title, exact: true }),
    ).toBeVisible();
  }
  await navigate(page, "Overview");
  await expect(
    page.getByRole("heading", { name: "See the behavior behind the score." }),
  ).toBeVisible();
});
test("light theme overrides saved dark and system preferences without hydration errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", e => errors.push(e.message));
  await page.emulateMedia({ colorScheme: "dark" });
  await page.addInitScript(() => localStorage.setItem("modelxray-theme", "dark"));
  await go(page);
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(page.getByRole("button", { name: /Switch to .* theme/ })).toHaveCount(0);
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  expect(errors).toEqual([]);
});

test("API health starts unknown and errors remain recoverable", async ({
  page,
}) => {
  await mockApi(page);
  await page.route(`${base}/health`, (r) =>
    r.fulfill({ status: 503, json: { detail: "Offline" } }),
  );
  await page.route(`${base}/api/v1/jobs/demo/investigate`, (r) =>
    r.fulfill({
      status: 422,
      json: { detail: { message: "Budget rejected by backend" } },
    }),
  );
  await page.goto("/");
  await expect(
    page.locator(".headerContext").getByText("API OFFLINE"),
  ).toBeVisible();
  await page.getByRole("button", { name: "Explore benchmark demo" }).click();
  await expect(page.locator(".globalNotice[role=alert]")).toContainText(
    "Budget rejected by backend",
  );
  await expect(
    page.getByRole("button", { name: "Investigate a model", exact: true }),
  ).toBeEnabled();
});
test("upload form registers actual files and submits selected target and metric", async ({
  page,
}) => {
  await go(page);
  await page
    .getByRole("button", { name: "Investigate a model", exact: true })
    .click();
  await page
    .getByLabel("MODEL ARTIFACT")
    .setInputFiles(path.resolve(".test-artifacts/model-v1.joblib"));
  await page
    .getByLabel("EVALUATION DATASET", { exact: true })
    .setInputFiles(path.resolve(".test-artifacts/evaluation.csv"));
  await page.getByLabel("TARGET COLUMN", { exact: true }).fill("label");
  await page
    .getByLabel("METRIC", { exact: true })
    .selectOption("balanced_accuracy");
  const submission = page.waitForRequest(
    `${base}/api/v1/jobs/investigate/uploaded`,
  );
  await page
    .getByRole("button", { name: "Investigate uploaded model" })
    .click();
  expect((await submission).postDataJSON()).toMatchObject({
    target_column: "label",
    metric: "balanced_accuracy",
    model_id: "MODEL-TEST",
    dataset_id: "DATASET-TEST",
  });
  await expect(
    page.getByRole("heading", { name: "Model overview", exact: true }),
  ).toBeVisible();
});
test("upload API rejection shows the backend message and preserves input", async ({
  page,
}) => {
  await go(page);
  await page.route(`${base}/api/v1/assets/model`, (r) =>
    r.fulfill({
      status: 400,
      json: { detail: { message: "Unsupported model artifact" } },
    }),
  );
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
  await expect(page.locator(".globalNotice[role=alert]")).toContainText(
    "Unsupported model artifact",
  );
  await expect(
    page.getByRole("button", { name: "Investigate uploaded model" }),
  ).toBeEnabled();
});
test("server progress, recovery and cancellation never infer completion", async ({
  page,
}) => {
  await mockApi(page);
  let cancelled = false;
  await page.route(`${base}/api/v1/jobs/TEST-JOB`, (r) =>
    r.fulfill({
      json: {
        job_id: "TEST-JOB",
        kind: "DEMO_INVESTIGATION",
        status: cancelled ? "CANCELLED" : "RUNNING",
        phase: "SEARCHING",
        progress: 0.375,
        completed_units: 3,
        total_units: 8,
      },
    }),
  );
  await page.route(`${base}/api/v1/jobs/TEST-JOB/cancel`, (r) => {
    cancelled = true;
    return r.fulfill({
      json: {
        job_id: "TEST-JOB",
        kind: "DEMO_INVESTIGATION",
        status: "CANCELLING",
        cancel_requested: true,
        progress: 0.375,
      },
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Explore benchmark demo" }).click();
  await expect(page.getByRole("progressbar")).toHaveAttribute("value", "0.375");
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "RUNNING", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Cancel investigation", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "CANCELLED", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Model overview", exact: true }),
  ).toHaveCount(0);
});
test("completed investigation recovers, visualizes backend evidence and filters atlas", async ({
  page,
}) => {
  await go(page, true);
  await expect(
    page.getByRole("heading", { name: "Model overview", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Region effect intervals" }),
  ).toBeVisible();
  await page
    .getByText(
      `Accessible evidence table · ${result.failure_atlas.length} entries`,
      { exact: true },
    )
    .click();
  await expect(page.getByRole("table").first()).toContainText(
    result.failure_atlas[0].condition,
  );
  await navigate(page, "Failure Atlas");
  await page.getByLabel("Search regions").fill("no-matching-region");
  await expect(page.getByText("No regions match these filters.")).toBeVisible();
  await page.getByLabel("Search regions").fill("");
  await expect(
    page.getByText(/Prediction changes under small perturbations/),
  ).toBeVisible();
  await navigate(page, "Experiments");
  await expect(
    page.getByRole("heading", { name: "Experiment effect outcomes" }),
  ).toBeVisible();
  await navigate(page, "Search Coverage");
  await expect(
    page.getByText(result.active_investigation.search_coverage.interpretation),
  ).toBeVisible();
});
test("regression uses real fixture deltas, score units and warnings", async ({
  page,
}) => {
  await go(page);
  await navigate(page, "Regression");
  await page.getByRole("button", { name: "Compare demo versions" }).click();
  await expect(page.getByText("SYNTHETIC BENCHMARK COMPARISON")).toBeVisible();
  await expect(page.getByText("baseline score change")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Failure movement" }),
  ).toBeVisible();
  await expect(
    page.getByText(/does not prove total behavioral dissimilarity/),
  ).toBeVisible();
});
test("shared topology edge endpoints exactly coincide with node centers at widths and zoom", async ({
  page,
}) => {
  await go(page, true);
  for (const width of [390, 768, 1024, 1280, 1440])
    for (const zoom of [1, 1.5, 2]) {
      await page.setViewportSize({ width, height: 900 });
      await page.evaluate((z) => (document.body.style.zoom = String(z)), zoom);
      const detached = await page.locator(".investigationMap svg").evaluate(
        (svg) =>
          Array.from(svg.querySelectorAll("path[data-from]")).filter((edge) => {
            const node = (id: string | null) =>
              svg.querySelector(`[data-node="${id}"]`)! as SVGGraphicsElement;
            const from = node(
              edge.getAttribute("data-from"),
            ).transform.baseVal.consolidate()!.matrix;
            const to = node(
              edge.getAttribute("data-to"),
            ).transform.baseVal.consolidate()!.matrix;
            const p = edge as SVGPathElement;
            const a = p.getPointAtLength(0),
              b = p.getPointAtLength(p.getTotalLength());
            return (
              Math.hypot(a.x - from.e, a.y - from.f) > 0.001 ||
              Math.hypot(b.x - to.e, b.y - to.f) > 0.001
            );
          }).length,
      );
      expect(detached).toBe(0);
    }
});
for (const theme of ["light"])
  test(`enhanced landing is accessible in ${theme} theme`, async ({ page }) => {
    await go(page);

    await expect(page.locator("html")).toHaveCSS(
      "color",
      "rgb(27, 33, 25)",
    );
    await page.evaluate(() => { if (document.activeElement instanceof HTMLElement) document.activeElement.blur(); window.scrollTo(0, 0); });
    await page.screenshot({
      path: path.resolve(`.test-artifacts/landing-${theme}.png`),
      fullPage: true,
    });
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"])
          .analyze()
      ).violations,
    ).toEqual([]);
    await page.setViewportSize({ width: 390, height: 844 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    ).toBeTruthy();
    await page.evaluate(() => { if (document.activeElement instanceof HTMLElement) document.activeElement.blur(); window.scrollTo(0, 0); });
    await page.screenshot({
      path: path.resolve(`.test-artifacts/landing-${theme}-mobile.png`),
      fullPage: true,
    });
  });

for (const theme of ["light"])
  test(`accessibility, responsive containment and focus in ${theme} theme`, async ({
    page,
  }) => {
    await go(page, true);


    for (const tab of [
      "Overview",
      "Failure Atlas",
      "Experiments",
      "Search Coverage",
      "Regression",
    ]) {
      await navigate(page, tab);
      if (tab === "Regression") {
        await page.getByRole("button", { name: "Compare demo versions" }).click();
        await expect(page.getByRole("heading", { name: "Failure movement" })).toBeVisible();
      }
      for (const width of [390, 768, 1024, 1280, 1440]) {
        await page.setViewportSize({ width, height: 900 });
        expect(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= window.innerWidth + 1,
          ),
          `${tab} at ${width}px`,
        ).toBeTruthy();
      }
      await page.setViewportSize({ width: 1280, height: 900 });
      await page.evaluate(() => { if (document.activeElement instanceof HTMLElement) document.activeElement.blur(); window.scrollTo(0, 0); });
    await page.screenshot({
        path: path.resolve(
          `.test-artifacts/${tab.toLowerCase().replaceAll(" ", "-")}-${theme}.png`,
        ),
        fullPage: true,
      });
      const scan = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"])
        .analyze();
      expect(
        scan.violations,
        JSON.stringify(
          scan.violations.map((v) => ({
            id: v.id,
            nodes: v.nodes.map((n) => n.target),
          })),
        ),
      ).toEqual([]);
    }
    await navigate(page, "Overview");
    await page
      .getByRole("button", { name: "New investigation", exact: true })
      .first()
      .click();
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"])
          .analyze()
      ).violations,
    ).toEqual([]);
    for (const width of [390, 768, 1024, 1280, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth + 1,
        ),
      ).toBeTruthy();
    }
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.evaluate(() => { if (document.activeElement instanceof HTMLElement) document.activeElement.blur(); window.scrollTo(0, 0); });
    await page.screenshot({ path: path.resolve(`.test-artifacts/setup-${theme}.png`), fullPage: true });
    expect(
      await page
        .locator(".primaryButton")
        .first()
        .evaluate((el) => getComputedStyle(el).transitionDuration),
    ).toBe("0s");
  });

test("Ink Olive tokens, local fonts and readable disconnected state", async ({ page }) => {
  await page.route(`${base}/**`, (route) => route.abort("connectionrefused"));
  await page.goto("/");
  await expect(page.getByRole("alert").first()).toContainText("Cannot reach the ModelXray API");
  await expect(page.getByText("Failed to fetch", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Refresh history" })).toBeEnabled();
  for (const theme of ["light"]) {

    await expect(page.locator("html")).toHaveCSS("background-color", "rgb(245, 244, 239)");
    await page.evaluate(() => document.fonts.ready);
    expect(await page.locator("h1").evaluate(el => getComputedStyle(el).fontFamily)).toContain("geist");
    expect((await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"]).analyze()).violations).toEqual([]);
    await page.evaluate(() => { if (document.activeElement instanceof HTMLElement) document.activeElement.blur(); window.scrollTo(0, 0); });
    await page.screenshot({ path: path.resolve(`.test-artifacts/disconnected-${theme}.png`), fullPage: true });
  }
});
