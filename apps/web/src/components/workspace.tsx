"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type {
  Result,
  CompareResult,
  UploadedArtifact,
  Tab,
} from "../lib/types";
import { API_BASE, api, readApiError, storage, errorMessage } from "../lib/api";
import { Icon, StatusDot, EmptyState } from "./ui";
import { useInvestigationJob } from "../hooks/use-investigation-job";
import { JobProgress, RunHistory } from "./lifecycle";

import { Overview } from "./overview";
import { FailureAtlas } from "./failure-atlas";
import { Experiments } from "./experiments";
import { SearchCoverage } from "./search-coverage";

import { InvestigationSetup } from "./investigation-setup";
import { Regression } from "./regression";
import { Landing } from "./landing";

export default function Workspace() {
  const [result, setResult] = useState<Result | null>(null);
  const [comparison, setComparison] = useState<CompareResult | null>(null);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [loading, setLoading] = useState<
    "investigate" | "compare" | "compare-uploaded" | null
  >(null);
  const [activeTab, setActiveTab] = useState<Tab>("overview");
  const [selectedFailureId, setSelectedFailureId] = useState<string | null>(
    null,
  );
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [uploadedModels, setUploadedModels] = useState<UploadedArtifact[]>([]);
  const [uploadedDatasets, setUploadedDatasets] = useState<UploadedArtifact[]>(
    [],
  );
  const [cmpModelV1, setCmpModelV1] = useState("");
  const [cmpModelV2, setCmpModelV2] = useState("");
  const [cmpDataset, setCmpDataset] = useState("");
  const [regModelFile, setRegModelFile] = useState<File | null>(null);
  const [regDatasetFile, setRegDatasetFile] = useState<File | null>(null);
  const [targetColumn, setTargetColumn] = useState("target");
  const [metric, setMetric] = useState<"accuracy" | "balanced_accuracy">(
    "accuracy",
  );
  const [uploading, setUploading] = useState(false);
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [showInvestigationSetup, setShowInvestigationSetup] = useState(false);

  const [health, setHealth] = useState<"checking" | "online" | "offline">(
    "checking",
  );
  const [filter, setFilter] = useState("");
  const [severityFilter, setSeverityFilter] = useState("ALL");
  const [sort, setSort] = useState("evidence");
  const [experimentQuery, setExperimentQuery] = useState("");
  const setupRef = useRef<HTMLElement>(null);
  const receiveResult = useCallback((data: Result) => {
    setResult(data);
    setSelectedFailureId(data.failure_atlas?.[0]?.failure_id ?? null);
    setActiveTab("overview");
    setShowInvestigationSetup(false);
    setFilter("");
    setSeverityFilter("ALL");
    setExperimentQuery("");
  }, []);
  const lifecycle = useInvestigationJob(receiveResult);
  const busy = lifecycle.busy || loading !== null || uploading;
  useEffect(() => {
    let saved: string | null = null;
    try {
      saved = window.localStorage.getItem("modelxray-theme");
    } catch {}
    const next =
      saved === "light" || saved === "dark"
        ? saved
        : window.matchMedia("(prefers-color-scheme: light)").matches
          ? "light"
          : "dark";
    setTheme(next);
    document.documentElement.dataset.theme = next;
    const models = storage.get("models");
    const datasets = storage.get("datasets");
    try {
      if (models) setUploadedModels(JSON.parse(models));
      if (datasets) setUploadedDatasets(JSON.parse(datasets));
    } catch {}
  }, []);
  useEffect(() => {
    if (uploadedModels.length)
      storage.set("models", JSON.stringify(uploadedModels));
  }, [uploadedModels]);
  useEffect(() => {
    if (uploadedDatasets.length)
      storage.set("datasets", JSON.stringify(uploadedDatasets));
  }, [uploadedDatasets]);
  useEffect(() => {
    let disposed = false;
    async function check() {
      try {
        const data = await api<{ status: string }>("/health");
        if (!disposed) setHealth(data.status === "ok" ? "online" : "offline");
      } catch {
        if (!disposed) setHealth("offline");
      }
    }
    void check();
    const timer = setInterval(check, 30000);
    return () => {
      disposed = true;
      clearInterval(timer);
    };
  }, []);
  useEffect(() => {
    if (showInvestigationSetup) {
      setupRef.current?.scrollIntoView({ block: "start" });
      setupRef.current?.querySelector<HTMLInputElement>("input")?.focus();
    }
  }, [showInvestigationSetup]);
  const filteredFailures = useMemo(
    () =>
      (result?.failure_atlas ?? [])
        .filter(
          (f) =>
            (severityFilter === "ALL" ||
              f.severity.toUpperCase() === severityFilter) &&
            `${f.condition} ${f.failure_id}`
              .toLowerCase()
              .includes(filter.toLowerCase()),
        )
        .slice()
        .sort((a, b) =>
          sort === "gap"
            ? b.gap - a.gap
            : sort === "support"
              ? b.support - a.support
              : b.evidence_score - a.evidence_score,
        ),
    [result, filter, severityFilter, sort],
  );

  function openInvestigationSetup() {
    setComparisonError(null);
    setActiveTab("overview");
    setShowInvestigationSetup(true);
  }

  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.dataset.theme = next;
    try {
      window.localStorage.setItem("modelxray-theme", next);
    } catch {}
  }

  function navigateWorkspace(tab: Tab) {
    setActiveTab(tab);
    setShowInvestigationSetup(false);
    requestAnimationFrame(() => {
      const workspace = document.getElementById("workspace-content");
      workspace?.focus({ preventScroll: true });
      workspace?.scrollIntoView({ block: "start" });
    });
  }

  async function runInvestigation() {
    if (busy) return;
    setLoading("investigate");
    setComparisonError(null);
    try {
      await lifecycle.start("/api/v1/jobs/demo/investigate", {
        rows: 2500,
        random_state: 42,
        budget: 24,
        metric,
      });
    } catch (error) {
      setComparisonError(
        errorMessage(error, "The investigation request failed."),
      );
    } finally {
      setLoading(null);
    }
  }
  async function uploadAndInvestigate() {
    if (busy) return;
    if (!modelFile || !datasetFile || !targetColumn.trim())
      return setComparisonError(
        "Choose a model, CSV dataset, and target column first.",
      );
    setUploading(true);
    setComparisonError(null);
    try {
      const modelForm = new FormData();
      modelForm.append("file", modelFile);
      const modelResponse = await fetch(`${API_BASE}/api/v1/assets/model`, {
        method: "POST",
        body: modelForm,
      });
      if (!modelResponse.ok) throw new Error(await readApiError(modelResponse));
      const modelAsset = await modelResponse.json();
      setUploadedModels((prev) =>
        prev.some((m) => m.asset_id === modelAsset.asset_id)
          ? prev
          : [
              ...prev,
              {
                asset_id: modelAsset.asset_id,
                filename: modelAsset.filename ?? modelFile.name,
              },
            ],
      );
      const datasetForm = new FormData();
      datasetForm.append("file", datasetFile);
      const datasetResponse = await fetch(`${API_BASE}/api/v1/assets/dataset`, {
        method: "POST",
        body: datasetForm,
      });
      if (!datasetResponse.ok)
        throw new Error(await readApiError(datasetResponse));
      const datasetAsset = await datasetResponse.json();
      setUploadedDatasets((prev) =>
        prev.some((d) => d.asset_id === datasetAsset.asset_id)
          ? prev
          : [
              ...prev,
              {
                asset_id: datasetAsset.asset_id,
                filename: datasetAsset.filename ?? datasetFile.name,
                rows: datasetAsset.metadata?.rows,
              },
            ],
      );
      await lifecycle.start("/api/v1/jobs/investigate/uploaded", {
        model_id: modelAsset.asset_id,
        dataset_id: datasetAsset.asset_id,
        target_column: targetColumn.trim(),
        budget: 24,
        random_state: 42,
        metric,
      });
    } catch (error) {
      setComparisonError(
        errorMessage(error, "ModelXray could not process the uploaded artifacts."),
      );
    } finally {
      setUploading(false);
    }
  }
  async function compareVersions() {
    setLoading("compare");
    setComparisonError(null);
    try {
      const response = await fetch(`${API_BASE}/api/v1/demo/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows: 2500, random_state: 42 }),
      });
      if (!response.ok) throw new Error(await readApiError(response));
      setComparison(await response.json());
      setActiveTab("regression");
    } catch (error) {
      setComparisonError(
        error instanceof Error
          ? `${error.message}. Is the ModelXray API running on port 8000?`
          : "The comparison request failed.",
      );
    } finally {
      setLoading(null);
    }
  }
  async function registerModelArtifact() {
    if (!regModelFile)
      return setComparisonError(
        "Choose a .joblib/.pkl model artifact to register.",
      );
    setUploading(true);
    setComparisonError(null);
    try {
      const form = new FormData();
      form.append("file", regModelFile);
      const response = await fetch(`${API_BASE}/api/v1/assets/model`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) throw new Error(await readApiError(response));
      const asset = await response.json();
      const id: string = asset.asset_id;
      setUploadedModels((prev) =>
        prev.some((m) => m.asset_id === id)
          ? prev
          : [
              ...prev,
              { asset_id: id, filename: asset.filename ?? regModelFile.name },
            ],
      );
      if (!cmpModelV1) setCmpModelV1(id);
      else if (!cmpModelV2 && cmpModelV1 !== id) setCmpModelV2(id);
      setRegModelFile(null);
    } catch (error) {
      setComparisonError(
        errorMessage(error, "Model artifact registration failed."),
      );
    } finally {
      setUploading(false);
    }
  }
  async function registerDatasetArtifact() {
    if (!regDatasetFile)
      return setComparisonError("Choose a labeled .csv dataset to register.");
    setUploading(true);
    setComparisonError(null);
    try {
      const form = new FormData();
      form.append("file", regDatasetFile);
      const response = await fetch(`${API_BASE}/api/v1/assets/dataset`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) throw new Error(await readApiError(response));
      const asset = await response.json();
      const id: string = asset.asset_id;
      setUploadedDatasets((prev) =>
        prev.some((d) => d.asset_id === id)
          ? prev
          : [
              ...prev,
              {
                asset_id: id,
                filename: asset.filename ?? regDatasetFile.name,
                rows: asset.metadata?.rows,
              },
            ],
      );
      if (!cmpDataset) setCmpDataset(id);
      setRegDatasetFile(null);
    } catch (error) {
      setComparisonError(
        errorMessage(error, "Dataset registration failed."),
      );
    } finally {
      setUploading(false);
    }
  }
  async function compareUploadedModels() {
    if (!cmpModelV1 || !cmpModelV2 || !cmpDataset || !targetColumn.trim())
      return setComparisonError(
        "Select two uploaded model artifacts and a dataset first.",
      );
    if (cmpModelV1 === cmpModelV2)
      return setComparisonError(
        "Model v1 and model v2 must be different artifacts.",
      );
    setLoading("compare-uploaded");
    setComparisonError(null);
    try {
      const response = await fetch(`${API_BASE}/api/v1/regression/uploaded`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_v1_id: cmpModelV1,
          model_v2_id: cmpModelV2,
          dataset_id: cmpDataset,
          target_column: targetColumn.trim(),
          budget: 24,
          random_state: 42,
          metric,
        }),
      });
      if (!response.ok) throw new Error(await readApiError(response));
      setComparison(await response.json());
      setActiveTab("regression");
    } catch (error) {
      setComparisonError(
        errorMessage(error, "The uploaded-model comparison failed."),
      );
    } finally {
      setLoading(null);
    }
  }

  const failureCount = result?.failure_atlas.length ?? 0;
  const high =
    result?.failure_atlas.filter((x) => x.severity.toUpperCase() === "HIGH")
      .length ?? 0;
  const validated =
    result?.active_investigation.observations.filter((x) => x.validated)
      .length ?? 0;
  const reproducible =
    result?.failure_atlas.filter((x) => x.reproducible).length ?? 0;
  const selectedFailure =
    result?.failure_atlas.find((x) => x.failure_id === selectedFailureId) ??
    result?.failure_atlas[0] ??
    null;
  const selectedCounterexamples = selectedFailure
    ? (result?.counterexamples.filter((x) =>
        selectedFailure.counterexample_ids.includes(x.counterexample_id ?? ""),
      ) ?? [])
    : [];
  const accuracyDelta = comparison
    ? (comparison.models.v2.accuracy - comparison.models.v1.accuracy) * 100
    : 0;
  const coverage = result?.active_investigation.search_coverage;
  const severityCounts = useMemo(
    () =>
      ["HIGH", "MEDIUM", "LOW"].map((label) => ({
        label,
        count: (result?.failure_atlas ?? []).filter(
          (x) => x.severity.toUpperCase() === label,
        ).length,
      })),
    [result],
  );

  const navItems: Array<{
    id: Tab;
    label: string;
    icon: "overview" | "failures" | "experiments" | "coverage" | "regression";
  }> = [
    { id: "overview", label: "Overview", icon: "overview" },
    { id: "failures", label: "Failure Atlas", icon: "failures" },
    { id: "experiments", label: "Experiments", icon: "experiments" },
    { id: "coverage", label: "Search Coverage", icon: "coverage" },
    { id: "regression", label: "Regression", icon: "regression" },
  ];
  const tabTitle =
    activeTab === "overview"
      ? "Model overview"
      : activeTab === "failures"
        ? "Failure Atlas"
        : activeTab === "experiments"
          ? "Experiment Lab"
          : activeTab === "coverage"
            ? "Search coverage"
            : "Model regression";
  const tabCopy =
    activeTab === "overview"
      ? "See the behavioral regions behind the aggregate score."
      : activeTab === "failures"
        ? "Validated regions, evidence, and the search strategies that surfaced them."
        : activeTab === "experiments"
          ? "The hypotheses actually executed under the investigation budget."
          : activeTab === "coverage"
            ? "Understand what ModelXray explored — and what remained outside the executed search."
            : "Track how validated failure regions move between model versions.";

  return (
    <main className="appShell">
      <a className="skipLink" href="#workspace-content">
        Skip to workspace
      </a>
      <aside className="sidebar">
        <div className="brandLockup">
          <div>
            <div className="brandName">
              Model<span>Xray</span>
            </div>
            <div className="brandCaption">
              Model assurance
            </div>
          </div>
        </div>
        <div className="sidebarRule" />
        <div className="sidebarSectionLabel">Workspace</div>
        <nav className="navStack" aria-label="Workspaces">
          {navItems.map((item) => (
            <button
              key={item.id}
              aria-label={item.label}
              className={`navButton ${activeTab === item.id ? "active" : ""}`}
              aria-current={activeTab === item.id ? "page" : undefined}
              onClick={() => {
                navigateWorkspace(item.id);
                setComparisonError(null);
              }}
            >
              <span className="navIcon">
                <Icon name={item.icon} />
              </span>
              <span>{item.label}</span>
              {item.id === "failures" && failureCount > 0 ? (
                <em>{failureCount}</em>
              ) : null}
            </button>
          ))}
        </nav>
          <div className="headerActions">
            <button
              className="themeButton"
              onClick={toggleTheme}
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
              title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            >
              <Icon name={theme === "dark" ? "sun" : "moon"} />
              <span>{theme === "dark" ? "Light" : "Dark"}</span>
            </button>
            <button
              className="primaryButton"
              onClick={openInvestigationSetup}
              disabled={busy}
            >
              New investigation
              <Icon name="arrow" />
            </button>
          </div>
      </aside>

      <section className="mainArea">
        <header className="appHeader">
          <div className="headerContext">
            <span className="headerKicker">{result ? `Investigation ${result.investigation_id}` : "Local workspace"}</span>
            <span className="headerDivider" />
            <span>
              <StatusDot tone={health === "online" ? "green" : "amber"} /> API{" "}
              {health.toUpperCase()}
            </span>
          </div>
          <div className="headerActions">
            <button
              className="quietButton"
              onClick={compareVersions}
              disabled={busy}
            >
              {loading === "compare" ? "Comparing…" : "Demo comparison"}
            </button>
          </div>
        </header>

        <div className="contentWrap" id="workspace-content" tabIndex={-1}>
          <RunHistory
            onLoad={receiveResult}
            currentId={result?.investigation_id}
            disabled={busy}
          />
          {lifecycle.job &&
            (lifecycle.busy ||
              !result ||
              lifecycle.job.investigation_id === result.investigation_id) && (
              <JobProgress job={lifecycle.job} cancel={lifecycle.cancel} />
            )}
          {lifecycle.error && (
            <div className="globalNotice" role="alert">
              <p>{lifecycle.error}</p>
              <button className="quietButton" onClick={lifecycle.retryRecovery}>
                Retry recovery
              </button>
            </div>
          )}

          {comparisonError && (
            <div className="globalNotice" role="alert">
              <div>
                <StatusDot tone="red" />
                <strong>{comparisonError}</strong>
              </div>
              <button
                onClick={() => setComparisonError(null)}
                aria-label="Dismiss"
              >
                <Icon name="close" />
              </button>
            </div>
          )}

          {showInvestigationSetup && (
            <InvestigationSetup
              setupRef={setupRef}
              modelFile={modelFile}
              datasetFile={datasetFile}
              setModelFile={setModelFile}
              setDatasetFile={setDatasetFile}
              targetColumn={targetColumn}
              setTargetColumn={setTargetColumn}
              metric={metric}
              setMetric={setMetric}
              busy={busy}
              uploading={uploading}
              uploadAndInvestigate={uploadAndInvestigate}
              setShowInvestigationSetup={setShowInvestigationSetup}
              runInvestigation={runInvestigation}
            />
          )}

          {!result &&
          !comparison &&
          activeTab === "overview" &&
          !showInvestigationSetup ? (
            <Landing
              busy={busy}
              onInvestigate={openInvestigationSetup}
              onDemo={runInvestigation}
            />
          ) : null}

          {!result &&
          !comparison &&
          activeTab !== "overview" &&
          activeTab !== "regression" &&
          !showInvestigationSetup ? (
            <section className="contentPanel workspaceEmpty">
              <EmptyState
                eyebrow={
                  activeTab === "failures"
                    ? "FAILURE ATLAS"
                    : activeTab === "experiments"
                      ? "EXPERIMENT LAB"
                      : "SEARCH COVERAGE"
                }
                title={
                  activeTab === "failures"
                    ? "No investigation loaded."
                    : activeTab === "experiments"
                      ? "No experiments to inspect."
                      : "No search coverage to inspect."
                }
                copy="Start a new investigation to populate this workspace with real model-specific evidence."
                primary={
                  <button
                    className="primaryButton"
                    onClick={openInvestigationSetup}
                  >
                    Start new investigation
                    <Icon name="arrow" />
                  </button>
                }
                secondary={
                  <button
                    className="quietButton"
                    onClick={() => {
                      setShowInvestigationSetup(false);
                      runInvestigation();
                    }}
                    disabled={busy}
                  >
                    Run benchmark demo
                  </button>
                }
              />
            </section>
          ) : null}

          {!result && comparison && activeTab !== "regression" ? (
            <section className="contentPanel workspaceEmpty">
              <EmptyState
                eyebrow="NO INVESTIGATION LOADED"
                title="This workspace is waiting for a model investigation."
                copy="A regression comparison is loaded, but this screen needs a single-model investigation. Choose a model and evaluation dataset to populate the evidence panels."
                primary={
                  <button
                    className="primaryButton"
                    onClick={openInvestigationSetup}
                  >
                    Choose model
                    <Icon name="arrow" />
                  </button>
                }
                secondary={
                  <button
                    className="quietButton"
                    onClick={() => setActiveTab("regression")}
                  >
                    View regression
                  </button>
                }
              />
            </section>
          ) : null}

          {result && activeTab !== "regression" && (
            <>
              <div className="pageIntro">
                <div>
                  <div className="eyebrow blue">
                    {result.mode === "DEMO"
                      ? "SYNTHETIC BENCHMARK"
                      : result.mode === "UPLOADED_MODEL"
                        ? "UPLOADED MODEL"
                        : "SAVED INVESTIGATION"}{" "}
                    / {result.investigation_id}
                  </div>
                  <h1>{tabTitle}</h1>
                  <p>{tabCopy}</p>
                </div>
                <div className="introActions">
                  <div className="introModel">
                    <span>MODEL</span>
                    <strong>{result.model.model_type}</strong>
                    <small>
                      {result.dataset.rows.toLocaleString()} rows ·{" "}
                      {result.dataset.columns} features
                    </small>
                  </div>
                  <button
                    className="quietButton"
                    disabled={busy}
                    onClick={openInvestigationSetup}
                  >
                    New investigation
                  </button>
                </div>
              </div>

              {activeTab === "overview" && (
                <Overview
                  result={result}
                  failureCount={failureCount}
                  high={high}
                  validated={validated}
                  reproducible={reproducible}
                  coverage={coverage}
                  selectedFailureId={selectedFailureId}
                  setSelectedFailureId={(id) => {
                    setSelectedFailureId(id);
                    setFilter("");
                    setSeverityFilter("ALL");
                  }}
                  setActiveTab={navigateWorkspace}
                  severityCounts={severityCounts}
                />
              )}

              {activeTab === "failures" && (
                <FailureAtlas
                  failureCount={failureCount}
                  filteredFailures={filteredFailures}
                  selectedFailure={selectedFailure}
                  selectedCounterexamples={selectedCounterexamples}
                  setSelectedFailureId={setSelectedFailureId}
                  filter={filter}
                  setFilter={setFilter}
                  severityFilter={severityFilter}
                  setSeverityFilter={setSeverityFilter}
                  sort={sort}
                  setSort={setSort}
                  onInspectExperiment={(id) => {
                    setExperimentQuery(id);
                    navigateWorkspace("experiments");
                  }}
                />
              )}

              {activeTab === "experiments" && (
                <Experiments
                  result={result}
                  query={experimentQuery}
                  setQuery={setExperimentQuery}
                />
              )}

              {activeTab === "coverage" && (
                <SearchCoverage coverage={coverage} />
              )}
            </>
          )}

          {activeTab === "regression" && (
            <Regression
              loading={loading}
              busy={busy}
              uploading={uploading}
              comparison={comparison}
              accuracyDelta={accuracyDelta}
              compareVersions={compareVersions}
              regModelFile={regModelFile}
              regDatasetFile={regDatasetFile}
              setRegModelFile={setRegModelFile}
              setRegDatasetFile={setRegDatasetFile}
              registerModelArtifact={registerModelArtifact}
              registerDatasetArtifact={registerDatasetArtifact}
              cmpModelV1={cmpModelV1}
              cmpModelV2={cmpModelV2}
              cmpDataset={cmpDataset}
              setCmpModelV1={setCmpModelV1}
              setCmpModelV2={setCmpModelV2}
              setCmpDataset={setCmpDataset}
              uploadedModels={uploadedModels}
              uploadedDatasets={uploadedDatasets}
              targetColumn={targetColumn}
              setTargetColumn={setTargetColumn}
              metric={metric}
              setMetric={setMetric}
              compareUploadedModels={compareUploadedModels}
            />
          )}
        </div>
      </section>
    </main>
  );
}
