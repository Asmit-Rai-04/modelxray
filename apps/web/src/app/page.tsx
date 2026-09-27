"use client";

import { useEffect, useMemo, useState } from "react";
import type { ChangeEvent, ReactNode } from "react";

const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

type Observation = {
  experiment_id: string; condition: string; kind?: string; support: number; gap: number; p_value: number;
  severity: string; validated: boolean; validation_gap: number; adjusted_p_value?: number; evidence_score?: number;
  holdout_effect_ci_low?: number; holdout_effect_ci_high?: number; holdout_ci_credible?: boolean;
};
type AtlasFailure = {
  failure_id: string; experiment_id: string; condition: string; severity: string; support: number; gap: number;
  adjusted_p_value: number; effect_size: number; ci_low: number; ci_high: number; validation_gap: number;
  reproducible: boolean; evidence_score: number; counterexample_ids: string[]; strategy_kinds?: string[]; strategy_diversity?: number;
};
type Counterexample = {
  counterexample_id?: string; source_experiment_id: string; source_condition: string; source_row: number;
  original_prediction: string | number; counterfactual_prediction: string | number; changed_feature: string;
  original_value: number; counterfactual_value: number; absolute_change: number; relative_change: number; search_status: string;
};
type Result = {
  model: { model_type: string; classes: unknown[] };
  dataset: { rows: number; columns: number; numeric_features?: string[]; categorical_features?: string[] };
  baseline: { accuracy: number; error_rate: number }; metric?: string;
  instabilities: Array<{ feature: string; relative_change: number; original_prediction: string; perturbed_prediction: string }>;
  experiment_count: number;
  active_investigation: {
    budget: number; experiments_considered: number; experiments_executed: number; observations: Observation[];
    search_coverage?: { scope: string; interpretation: string; candidate_pool_size: number; executed: number;
      unexplored_candidates: number; candidate_execution_ratio: number; available_families: string[]; explored_families: string[];
      family_breadth_ratio: number; candidate_feature_count: number; executed_feature_count: number; feature_breadth_ratio: number;
      families: Record<string, { generated_candidates: number; executed: number; unexplored_candidates: number;
        candidate_execution_ratio: number; validated_failures: number; candidate_features: string[]; executed_features: string[] }>;
    };
  };
  investigation_id: string; failure_atlas: AtlasFailure[]; counterexamples: Counterexample[];
};
type ComparisonModel = { accuracy: number; model_type?: string; asset_id?: string; investigation_id?: string };
type RegressionDelta = {
  experiment_id_v1: string | null; experiment_id_v2: string | null; condition_v1: string | null; condition_v2: string | null;
  match_type: string; status: "FIXED" | "PERSISTENT" | "NEW"; severity_v1: string | null; severity_v2: string | null;
  gap_v1: number | null; gap_v2: number | null; evidence_score_v1: number | null; evidence_score_v2: number | null;
  evidence_delta: number | null; validation_gap_v1: number | null; validation_gap_v2: number | null;
  failure_cluster_id_v1?: string | null; failure_cluster_id_v2?: string | null;
};
type CompareResult = {
  mode: string; status?: string; dataset?: { rows: number; features: string[] };
  models: { v1: ComparisonModel; v2: ComparisonModel };
  summary: { fixed: number; persistent: number; new: number; failures_v1: number; failures_v2: number; net_failure_change: number;
    failure_reduction_ratio: number; deltas: RegressionDelta[]; comparison_basis?: string };
  failure_deltas?: RegressionDelta[];
  comparability?: { comparable: boolean; reason: string | null; warnings: string[] } | null;
};
type UploadedArtifact = { asset_id: string; filename: string; rows?: number };
type Tab = "overview" | "failures" | "experiments" | "coverage" | "regression";

function pct(value: number) { return `${(value * 100).toFixed(1)}%`; }
function pp(value: number) { return `${(value * 100).toFixed(1)} pp`; }
function basisLabel(basis?: string) { return (basis ?? "failure clusters, then hypotheses").replaceAll("_", " "); }
function isDemoComparison(comparison: CompareResult) { return comparison.mode !== "UPLOADED_MODEL_REGRESSION"; }
function familyLabel(key: string) {
  const labels: Record<string, string> = { grid: "NUMERIC", categorical: "CATEGORICAL", interaction: "INTERACTION", oblique_band: "OBLIQUE", prototype: "NOVELTY", prototype_region: "NOVELTY" };
  return labels[key] ?? key.replaceAll("_", " ").toUpperCase();
}
async function readApiError(response: Response) {
  try { const payload = await response.json(); const detail = payload?.detail; if (typeof detail === "string") return detail; if (detail?.message) return detail.message; } catch {}
  return `Request failed (HTTP ${response.status})`;
}

function Icon({ name }: { name: "overview" | "failures" | "experiments" | "coverage" | "regression" | "docs" | "arrow" | "close" | "sun" | "moon" }) {
  const paths: Record<string, string> = {
    overview: "M4 4h7v7H4z M13 4h7v7h-7z M4 13h7v7H4z M13 13h7v7h-7z",
    failures: "M5 19 9 5l6 8 4-10 M4 19h16",
    experiments: "M5 5h14v14H5z M8 9h8 M8 13h6 M8 17h4",
    coverage: "M4 18V9 M9 18V5 M14 18v-7 M19 18V3",
    regression: "M4 16l5-5 4 3 7-8 M4 20h16",
    docs: "M6 4h10l3 3v13H6z M16 4v4h4 M9 12h6 M9 16h6",
    arrow: "M5 12h13 M13 7l5 5-5 5",
    close: "M6 6l12 12 M18 6 6 18",
    sun: "M12 4v2 M12 18v2 M4 12h2 M18 12h2 M6.3 6.3l1.4 1.4 M16.3 16.3l1.4 1.4 M17.7 6.3l-1.4 1.4 M7.7 16.3l-1.4 1.4 M15.5 12a3.5 3.5 0 1 1-7 0 3.5 3.5 0 0 1 7 0",
    moon: "M20 15.2A8.2 8.2 0 0 1 8.8 4 8.3 8.3 0 1 0 20 15.2z",
  };
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d={paths[name]} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}
function StatusDot({ tone = "blue" }: { tone?: "blue" | "green" | "red" | "amber" }) { return <span className={`statusDot ${tone}`} />; }
function Severity({ value }: { value: string }) {
  const tone = value.toUpperCase() === "HIGH" ? "danger" : value.toUpperCase() === "MEDIUM" ? "warning" : "neutral";
  return <span className={`severityPill ${tone}`}><i />{value.toUpperCase()}</span>;
}
function SectionHeader({ eyebrow, title, copy, action }: { eyebrow: string; title: string; copy?: string; action?: ReactNode }) {
  return <div className="sectionHeader"><div><div className="eyebrow">{eyebrow}</div><h2>{title}</h2>{copy && <p>{copy}</p>}</div>{action}</div>;
}
function Metric({ label, value, detail, tone = "blue" }: { label: string; value: string; detail: string; tone?: string }) {
  return <div className={`metric tone-${tone}`}><div className="metricTop"><span>{label}</span><i /></div><strong>{value}</strong><small>{detail}</small></div>;
}
function EmptyState({ eyebrow, title, copy, primary, secondary }: { eyebrow: string; title: string; copy: string; primary?: ReactNode; secondary?: ReactNode }) {
  return <div className="emptyState"><div className="emptyOrb"><span /></div><div><div className="eyebrow">{eyebrow}</div><h2>{title}</h2><p>{copy}</p><div className="emptyActions">{primary}{secondary}</div></div></div>;
}
function Node({ x, y, label, value, active, tone = "blue" }: { x: number; y: number; label: string; value: string; active?: boolean; tone?: "blue" | "cyan" | "red" | "green" }) {
  return <g className={`mapNode ${active ? "active" : ""} ${tone}`} transform={`translate(${x} ${y})`}>
    <circle r="34" className="nodeHalo" /><circle r="24" className="nodeCore" /><circle r="4" className="nodePoint" />
    <text x="0" y="54" textAnchor="middle" className="nodeLabel">{label}</text><text x="0" y="69" textAnchor="middle" className="nodeValue">{value}</text>
  </g>;
}
function InvestigationMap({ result }: { result?: Result | null }) {
  const failures = result?.failure_atlas.length ?? 0;
  const experiments = result?.active_investigation.experiments_executed ?? 0;
  const evidence = result?.counterexamples.length ?? 0;
  const coverage = result?.active_investigation.search_coverage?.candidate_execution_ratio;
  return <div className="investigationMap">
    <div className="mapHeader"><div><div className="eyebrow">BEHAVIORAL TOPOLOGY</div><h3>{result ? "Investigation trace" : "The ModelXray loop"}</h3></div><span className="mapLegend"><i /> evidence path</span></div>
    <svg viewBox="0 0 900 330" role="img" aria-label="ModelXray investigation flow">
      <defs><filter id="softGlow"><feGaussianBlur stdDeviation="5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter><linearGradient id="trace" x1="0" x2="1"><stop offset="0"/><stop offset=".5"/><stop offset="1"/></linearGradient></defs>
      <path className="mapGrid" d="M0 55H900 M0 110H900 M0 165H900 M0 220H900 M0 275H900" />
      <path className="mapTrace" d="M100 165 C145 90 235 90 280 165 C325 240 405 240 450 165 C495 90 575 90 620 165 C665 240 755 240 800 165" />
      <path className="mapTrace faint" d="M100 165 C145 240 235 240 280 165 C325 90 405 90 450 165 C495 240 575 240 620 165 C665 90 755 90 800 165" />
      <Node x={100} y={165} label="MODEL" value={result ? result.model.model_type : "black-box"} active={!!result} />
      <Node x={280} y={165} label="PROBES" value={result ? `${result.active_investigation.experiments_considered} generated` : "hypotheses"} active={!!result} tone="cyan" />
      <Node x={450} y={165} label="EXPERIMENTS" value={result ? `${experiments} executed` : "budgeted search"} active={!!result} tone="cyan" />
      <Node x={620} y={165} label="EVIDENCE" value={result ? `${evidence} boundary items` : "statistical gates"} active={!!result} tone="green" />
      <Node x={800} y={165} label="FAILURES" value={result ? `${failures} validated` : "behavioral regions"} active={!!result} tone={failures ? "red" : "blue"} />
      {result && <text x="450" y="300" textAnchor="middle" className="mapCaption">coverage {coverage === undefined ? "—" : pct(coverage)} · {result.active_investigation.budget} experiment budget · {result.investigation_id}</text>}
      {!result && <text x="450" y="300" textAnchor="middle" className="mapCaption">PROFILE → SEARCH → VALIDATE → CONSOLIDATE → EVIDENCE</text>}
    </svg>
  </div>;
}

export default function Home() {
  const [result, setResult] = useState<Result | null>(null);
  const [comparison, setComparison] = useState<CompareResult | null>(null);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [loading, setLoading] = useState<"investigate" | "compare" | "compare-uploaded" | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("overview");
  const [selectedFailureId, setSelectedFailureId] = useState<string | null>(null);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [uploadedModels, setUploadedModels] = useState<UploadedArtifact[]>([]);
  const [uploadedDatasets, setUploadedDatasets] = useState<UploadedArtifact[]>([]);
  const [cmpModelV1, setCmpModelV1] = useState(""); const [cmpModelV2, setCmpModelV2] = useState(""); const [cmpDataset, setCmpDataset] = useState("");
  const [regModelFile, setRegModelFile] = useState<File | null>(null); const [regDatasetFile, setRegDatasetFile] = useState<File | null>(null);
  const [targetColumn, setTargetColumn] = useState("target");
  const [metric, setMetric] = useState<"accuracy" | "balanced_accuracy">("accuracy");
  const [uploading, setUploading] = useState(false);
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [showInvestigationSetup, setShowInvestigationSetup] = useState(false);

  useEffect(() => {
    const saved = window.localStorage.getItem("modelxray-theme");
    const next = saved === "light" ? "light" : "dark";
    setTheme(next);
    document.documentElement.dataset.theme = next;
  }, []);

  function openInvestigationSetup() {
    setComparisonError(null);
    setActiveTab("overview");
    setShowInvestigationSetup(true);
  }

  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.dataset.theme = next;
    window.localStorage.setItem("modelxray-theme", next);
  }

  async function runInvestigation() {
    setLoading("investigate"); setComparisonError(null);
    try {
      const response = await fetch(`${API_BASE}/api/v1/demo/investigate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ rows: 2500, random_state: 42 }) });
      if (!response.ok) throw new Error(await readApiError(response));
      const data = await response.json(); setResult(data); setSelectedFailureId(data.failure_atlas?.[0]?.failure_id ?? null); setActiveTab("overview");
    } catch (error) { console.error(error); setComparisonError(error instanceof Error ? error.message : "The investigation request failed."); }
    finally { setLoading(null); }
  }
  async function uploadAndInvestigate() {
    if (!modelFile || !datasetFile || !targetColumn.trim()) return setComparisonError("Choose a model, CSV dataset, and target column first.");
    setUploading(true); setComparisonError(null);
    try {
      const modelForm = new FormData(); modelForm.append("file", modelFile);
      const modelResponse = await fetch(`${API_BASE}/api/v1/assets/model`, { method: "POST", body: modelForm }); if (!modelResponse.ok) throw new Error(await readApiError(modelResponse));
      const modelAsset = await modelResponse.json();
      setUploadedModels((prev) => prev.some((m) => m.asset_id === modelAsset.asset_id) ? prev : [...prev, { asset_id: modelAsset.asset_id, filename: modelAsset.filename ?? modelFile.name }]);
      const datasetForm = new FormData(); datasetForm.append("file", datasetFile);
      const datasetResponse = await fetch(`${API_BASE}/api/v1/assets/dataset`, { method: "POST", body: datasetForm }); if (!datasetResponse.ok) throw new Error(await readApiError(datasetResponse));
      const datasetAsset = await datasetResponse.json();
      setUploadedDatasets((prev) => prev.some((d) => d.asset_id === datasetAsset.asset_id) ? prev : [...prev, { asset_id: datasetAsset.asset_id, filename: datasetAsset.filename ?? datasetFile.name, rows: datasetAsset.metadata?.rows }]);
      const investigationResponse = await fetch(`${API_BASE}/api/v1/investigate/uploaded`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_id: modelAsset.asset_id, dataset_id: datasetAsset.asset_id, target_column: targetColumn.trim(), budget: 24, random_state: 42, metric }) });
      if (!investigationResponse.ok) throw new Error(await readApiError(investigationResponse));
      const data = await investigationResponse.json(); setResult(data); setSelectedFailureId(data.failure_atlas?.[0]?.failure_id ?? null); setActiveTab("overview");
    } catch (error) { console.error(error); setComparisonError(error instanceof Error ? error.message : "ModelXray could not process the uploaded artifacts."); }
    finally { setUploading(false); }
  }
  async function compareVersions() {
    setLoading("compare"); setComparisonError(null);
    try { const response = await fetch(`${API_BASE}/api/v1/demo/compare`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ rows: 2500, random_state: 42 }) }); if (!response.ok) throw new Error(await readApiError(response)); setComparison(await response.json()); setActiveTab("regression"); }
    catch (error) { setComparisonError(error instanceof Error ? `${error.message}. Is the ModelXray API running on port 8000?` : "The comparison request failed."); }
    finally { setLoading(null); }
  }
  async function registerModelArtifact() {
    if (!regModelFile) return setComparisonError("Choose a .joblib/.pkl model artifact to register."); setUploading(true); setComparisonError(null);
    try { const form = new FormData(); form.append("file", regModelFile); const response = await fetch(`${API_BASE}/api/v1/assets/model`, { method: "POST", body: form }); if (!response.ok) throw new Error(await readApiError(response)); const asset = await response.json(); const id: string = asset.asset_id; setUploadedModels((prev) => prev.some((m) => m.asset_id === id) ? prev : [...prev, { asset_id: id, filename: asset.filename ?? regModelFile.name }]); if (!cmpModelV1) setCmpModelV1(id); else if (!cmpModelV2 && cmpModelV1 !== id) setCmpModelV2(id); setRegModelFile(null); }
    catch (error) { setComparisonError(error instanceof Error ? error.message : "Model artifact registration failed."); } finally { setUploading(false); }
  }
  async function registerDatasetArtifact() {
    if (!regDatasetFile) return setComparisonError("Choose a labeled .csv dataset to register."); setUploading(true); setComparisonError(null);
    try { const form = new FormData(); form.append("file", regDatasetFile); const response = await fetch(`${API_BASE}/api/v1/assets/dataset`, { method: "POST", body: form }); if (!response.ok) throw new Error(await readApiError(response)); const asset = await response.json(); const id: string = asset.asset_id; setUploadedDatasets((prev) => prev.some((d) => d.asset_id === id) ? prev : [...prev, { asset_id: id, filename: asset.filename ?? regDatasetFile.name, rows: asset.metadata?.rows }]); if (!cmpDataset) setCmpDataset(id); setRegDatasetFile(null); }
    catch (error) { setComparisonError(error instanceof Error ? error.message : "Dataset registration failed."); } finally { setUploading(false); }
  }
  async function compareUploadedModels() {
    if (!cmpModelV1 || !cmpModelV2 || !cmpDataset) return setComparisonError("Select two uploaded model artifacts and a dataset first.");
    if (cmpModelV1 === cmpModelV2) return setComparisonError("Model v1 and model v2 must be different artifacts."); setLoading("compare-uploaded"); setComparisonError(null);
    try { const response = await fetch(`${API_BASE}/api/v1/regression/uploaded`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_v1_id: cmpModelV1, model_v2_id: cmpModelV2, dataset_id: cmpDataset, target_column: targetColumn.trim() || "target", budget: 24, random_state: 42, metric }) }); if (!response.ok) throw new Error(await readApiError(response)); setComparison(await response.json()); setActiveTab("regression"); }
    catch (error) { setComparisonError(error instanceof Error ? error.message : "The uploaded-model comparison failed."); } finally { setLoading(null); }
  }

  const failureCount = result?.failure_atlas.length ?? 0;
  const high = result?.failure_atlas.filter((x) => x.severity.toUpperCase() === "HIGH").length ?? 0;
  const validated = result?.active_investigation.observations.filter((x) => x.validated).length ?? 0;
  const reproducible = result?.failure_atlas.filter((x) => x.reproducible).length ?? 0;
  const selectedFailure = result?.failure_atlas.find((x) => x.failure_id === selectedFailureId) ?? result?.failure_atlas[0] ?? null;
  const selectedCounterexamples = selectedFailure ? (result?.counterexamples.filter((x) => selectedFailure.counterexample_ids.includes(x.counterexample_id ?? "")) ?? []) : [];
  const accuracyDelta = comparison ? (comparison.models.v2.accuracy - comparison.models.v1.accuracy) * 100 : 0;
  const coverage = result?.active_investigation.search_coverage;
  const severityCounts = useMemo(() => ["HIGH", "MEDIUM", "LOW"].map((label) => ({ label, count: (result?.failure_atlas ?? []).filter((x) => x.severity.toUpperCase() === label).length })), [result]);

  const navItems: Array<{ id: Tab; label: string; icon: "overview" | "failures" | "experiments" | "coverage" | "regression" }> = [
    { id: "overview", label: "Overview", icon: "overview" }, { id: "failures", label: "Failure Atlas", icon: "failures" },
    { id: "experiments", label: "Experiments", icon: "experiments" }, { id: "coverage", label: "Search Coverage", icon: "coverage" }, { id: "regression", label: "Regression", icon: "regression" },
  ];
  const tabTitle = activeTab === "overview" ? "Model overview" : activeTab === "failures" ? "Failure Atlas" : activeTab === "experiments" ? "Experiment Lab" : activeTab === "coverage" ? "Search coverage" : "Model regression";
  const tabCopy = activeTab === "overview" ? "See the behavioral regions behind the aggregate score." : activeTab === "failures" ? "Validated regions, evidence, and the search strategies that surfaced them." : activeTab === "experiments" ? "The hypotheses actually executed under the investigation budget." : activeTab === "coverage" ? "Understand what ModelXray explored — and what remained outside the executed search." : "Track how validated failure regions move between model versions.";

  return <main className="appShell">
    <aside className="sidebar">
      <div className="brandLockup"><div><div className="brandName">Model<span>Xray</span></div><div className="brandCaption">MODEL ASSURANCE / FAILURE DISCOVERY</div></div></div>
      <div className="sidebarRule" />
      <div className="sidebarSectionLabel">Workspace</div>
      <nav className="navStack">{navItems.map((item) => <button key={item.id} className={`navButton ${activeTab === item.id ? "active" : ""}`} onClick={() => setActiveTab(item.id)}><span className="navIcon"><Icon name={item.icon} /></span><span>{item.label}</span>{item.id === "failures" && failureCount > 0 ? <em>{failureCount}</em> : null}</button>)}</nav>
      <div className="sidebarSpacer" />
      <div className="sidebarSectionLabel">System</div>
      <button className="navButton mutedNav"><span className="navIcon"><Icon name="docs" /></span><span>Documentation</span></button>
      <div className="systemCard"><div><StatusDot tone="green" /> <span>ENGINE READY</span></div><code>API :8000</code><small>Local workspace · deterministic core</small></div>
    </aside>

    <section className="mainArea">
      <header className="appHeader">
        <div className="headerContext"><span className="headerKicker">ML MODEL ASSURANCE</span><span className="headerDivider" /><span><StatusDot tone="green" /> ENGINE READY</span></div>
        <div className="headerActions"><button className="themeButton" onClick={toggleTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}><Icon name={theme === "dark" ? "sun" : "moon"} /><span>{theme === "dark" ? "Light" : "Dark"}</span></button><button className="quietButton" onClick={compareVersions} disabled={loading !== null}>{loading === "compare" ? "Comparing…" : "Quick compare"}</button><button className="primaryButton" onClick={openInvestigationSetup} disabled={loading !== null}>New investigation<Icon name="arrow" /></button></div>
      </header>

      <div className="contentWrap">
        {comparisonError && <div className="globalNotice"><div><StatusDot tone="red" /><strong>{comparisonError}</strong></div><button onClick={() => setComparisonError(null)} aria-label="Dismiss"><Icon name="close" /></button></div>}

        {showInvestigationSetup && <section className="investigationSetup contentPanel">
          <div className="setupHeader"><div><div className="eyebrow blue">NEW INVESTIGATION</div><h2>Choose what ModelXray should inspect.</h2><p>Upload a trained scikit-learn-compatible model and a labeled evaluation dataset. This creates a new single-model investigation; it does not alter your regression comparison.</p></div><button className="quietButton" onClick={() => setShowInvestigationSetup(false)}>Close</button></div>
          <div className="setupGrid">
            <label className="setupField"><span>MODEL ARTIFACT</span><input type="file" accept=".joblib,.pkl,.pickle" onChange={(e: ChangeEvent<HTMLInputElement>) => setModelFile(e.target.files?.[0] ?? null)} /><strong>{modelFile?.name ?? "Choose .joblib / .pkl"}</strong><small>Existing trained model · scikit-learn compatible</small></label>
            <label className="setupField"><span>EVALUATION DATASET</span><input type="file" accept=".csv" onChange={(e: ChangeEvent<HTMLInputElement>) => setDatasetFile(e.target.files?.[0] ?? null)} /><strong>{datasetFile?.name ?? "Choose labeled .csv"}</strong><small>Must contain the target column used for evaluation</small></label>
          </div>
          <div className="setupControls"><label><span>TARGET COLUMN</span><input value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} placeholder="target" /></label><label><span>METRIC</span><select value={metric} onChange={(e) => setMetric(e.target.value as "accuracy" | "balanced_accuracy")}><option value="accuracy">Accuracy</option><option value="balanced_accuracy">Balanced accuracy</option></select></label></div>
          <div className="setupActions"><button className="primaryButton" onClick={uploadAndInvestigate} disabled={uploading || !modelFile || !datasetFile}>{uploading ? "Starting investigation…" : "Investigate uploaded model"}<Icon name="arrow" /></button><button className="quietButton" onClick={() => { setShowInvestigationSetup(false); runInvestigation(); }} disabled={loading !== null}>Use benchmark demo</button><span className="helperText">The demo uses the built-in synthetic benchmark and is clearly labeled.</span></div>
        </section>}

        {!result && !comparison && activeTab === "overview" && !showInvestigationSetup ? <section className="landing">
          <div className="landingCopy"><div className="eyebrow blue">MODEL FAILURE DISCOVERY</div><h1>See the behavior<br /><span>behind the score.</span></h1><p>ModelXray systematically probes a trained model, validates failure regions, and records the evidence behind every finding.</p><div className="heroActions"><button className="primaryButton large" onClick={openInvestigationSetup} disabled={loading !== null}>Investigate a model<Icon name="arrow" /></button><button className="quietButton large" onClick={() => setActiveTab("regression")}>Compare versions</button></div><div className="heroMeta"><span><StatusDot tone="green" /> deterministic search</span><span><StatusDot tone="blue" /> BH-FDR + holdout</span><span><StatusDot tone="blue" /> evidence ledger</span></div></div>
          <InvestigationMap />
          <div className="landingRail"><span>01 / MODEL</span><span>02 / PROBE</span><span>03 / VALIDATE</span><span>04 / EVIDENCE</span><span>05 / REGRESSION</span></div>
          <section className="landingBottom"><div><span className="eyebrow">WHAT IT DOES</span><h2>Not an explanation dashboard.<br />An investigation engine.</h2></div><p>It searches across multiple hypothesis families, allocates a bounded experiment budget, applies statistical gates, and consolidates overlapping findings into behavioral regions. Coverage is reported relative to the hypotheses that were generated — not as a claim of completeness.</p></section>
        </section> : null}

        {!result && !comparison && activeTab !== "overview" && activeTab !== "regression" && !showInvestigationSetup ? <section className="contentPanel workspaceEmpty"><EmptyState eyebrow={activeTab === "failures" ? "FAILURE ATLAS" : activeTab === "experiments" ? "EXPERIMENT LAB" : "SEARCH COVERAGE"} title={activeTab === "failures" ? "No investigation loaded." : activeTab === "experiments" ? "No experiments to inspect." : "No search coverage to inspect."} copy="Start a new investigation to populate this workspace with real model-specific evidence." primary={<button className="primaryButton" onClick={openInvestigationSetup}>Start new investigation<Icon name="arrow" /></button>} secondary={<button className="quietButton" onClick={() => { setShowInvestigationSetup(false); runInvestigation(); }} disabled={loading !== null}>Run benchmark demo</button>} /></section> : null}

        {(!result && comparison && activeTab !== "regression") ? <section className="contentPanel workspaceEmpty"><EmptyState eyebrow="NO INVESTIGATION LOADED" title="This workspace is waiting for a model investigation." copy="A regression comparison is loaded, but this screen needs a single-model investigation. Choose a model and evaluation dataset to populate the evidence panels." primary={<button className="primaryButton" onClick={openInvestigationSetup}>Choose model<Icon name="arrow" /></button>} secondary={<button className="quietButton" onClick={() => setActiveTab("regression")}>View regression</button>} /></section> : null}

        {result && activeTab !== "regression" && <>
          <div className="pageIntro"><div><div className="eyebrow blue">INVESTIGATION / {result.investigation_id}</div><h1>{tabTitle}</h1><p>{tabCopy}</p></div><div className="introActions"><div className="introModel"><span>MODEL</span><strong>{result.model.model_type}</strong><small>{result.dataset.rows.toLocaleString()} rows · {result.dataset.columns} features</small></div><button className="quietButton" onClick={openInvestigationSetup}>Change model</button></div></div>

          {activeTab === "overview" && <>
            <div className="metricGrid"><Metric label={(result.metric ?? "accuracy").replace("_", " ").toUpperCase()} value={pct(result.baseline.accuracy)} detail="validation baseline" tone="cyan" /><Metric label="VALIDATED FAILURES" value={`${failureCount}`} detail={`${reproducible} reproducible regions`} tone="red" /><Metric label="HIGH SEVERITY" value={`${high}`} detail="validated priority regions" tone="amber" /><Metric label="EXPERIMENTS" value={`${result.active_investigation.experiments_executed}/${result.active_investigation.budget}`} detail={`${validated} cleared validation`} /><Metric label="SEARCH COVERAGE" value={coverage ? pct(coverage.candidate_execution_ratio) : "—"} detail="generated candidates executed" tone="green" /></div>
            <InvestigationMap result={result} />
            <div className="overviewGrid">
              <section className="contentPanel wide"><SectionHeader eyebrow="FAILURE REGIONS" title="Where the model becomes unreliable" copy="Overlapping validated hypotheses are consolidated into behavioral regions." action={<button className="textButton" onClick={() => setActiveTab("failures")}>Open atlas <Icon name="arrow" /></button>} />{failureCount === 0 ? <div className="cleanState"><StatusDot tone="green" /><div><strong>No validated failures found.</strong><p>The investigation did not find a region that survived the statistical and holdout gates.</p></div></div> : <div className="clusterList">{result.failure_atlas.slice(0, 8).map((f, index) => <button key={f.failure_id} className={`clusterRow ${selectedFailureId === f.failure_id ? "selected" : ""}`} onClick={() => { setSelectedFailureId(f.failure_id); setActiveTab("failures"); }}><span className="clusterIndex">0{index + 1}</span><span className="clusterMain"><strong>{f.condition}</strong><small>{f.failure_id} · {f.strategy_diversity ?? f.strategy_kinds?.length ?? 1} strategy families</small></span><Severity value={f.severity} /><span className="clusterMetric"><b>{pp(f.gap)}</b><small>gap</small></span><span className="clusterMetric"><b>{f.evidence_score.toFixed(1)}</b><small>evidence</small></span><span className="clusterMetric"><b>{pct(f.support)}</b><small>support</small></span></button>)}</div>}</section>
              <section className="contentPanel"><SectionHeader eyebrow="MODEL SNAPSHOT" title="Investigation context" /><div className="factStack"><div><span>MODEL</span><strong>{result.model.model_type}</strong></div><div><span>NUMERIC FEATURES</span><strong>{result.dataset.numeric_features?.length ?? 0}</strong></div><div><span>CATEGORICAL FEATURES</span><strong>{result.dataset.categorical_features?.length ?? 0}</strong></div><div><span>HYPOTHESES CONSIDERED</span><strong>{result.active_investigation.experiments_considered}</strong></div><div><span>BOUNDARY EVIDENCE</span><strong>{result.counterexamples.length}</strong></div></div><div className="panelFootnote">Coverage is relative to the generated candidate pool.</div></section>
              <section className="contentPanel"><SectionHeader eyebrow="SEVERITY MIX" title="Failure distribution" />{severityCounts.map((item) => <div className="severityLine" key={item.label}><div><Severity value={item.label} /><strong>{item.count}</strong></div><div className="severityTrack"><span className={item.label.toLowerCase()} style={{ width: `${Math.min(100, (item.count / Math.max(1, failureCount)) * 100)}%` }} /></div></div>)}<div className="panelFootnote">Severity is a transparent heuristic, not a probability of harm.</div></section>
            </div>
          </>}

          {activeTab === "failures" && <section className="failureWorkspace"><div className="failureList contentPanel"><SectionHeader eyebrow="FAILURE ATLAS" title={`${failureCount} validated regions`} copy="Select a region to inspect its evidence chain." />{failureCount === 0 ? <div className="cleanState"><StatusDot tone="green" /><div><strong>Atlas is clean.</strong><p>No validated failures were promoted in this run.</p></div></div> : result.failure_atlas.map((f) => <button key={f.failure_id} className={`failureListItem ${selectedFailure?.failure_id === f.failure_id ? "selected" : ""}`} onClick={() => setSelectedFailureId(f.failure_id)}><div className="failureListTop"><Severity value={f.severity} /><span>{f.reproducible ? "REPRODUCIBLE" : "PENDING"}</span></div><strong>{f.condition}</strong><div className="failureMeta"><span>{pp(f.gap)} gap</span><span>{f.evidence_score.toFixed(1)} evidence</span><span>{pct(f.support)} support</span></div></button>)}</div><div className="failureDetail contentPanel">{selectedFailure ? <><SectionHeader eyebrow="SELECTED REGION" title={selectedFailure.condition} copy={`${selectedFailure.failure_id} · ${selectedFailure.experiment_id}`} action={<Severity value={selectedFailure.severity} />} /><div className="detailGrid"><div><small>PERFORMANCE GAP</small><strong>{pp(selectedFailure.gap)}</strong></div><div><small>HOLDOUT GAP</small><strong>{pp(selectedFailure.validation_gap)}</strong></div><div><small>EVIDENCE</small><strong>{selectedFailure.evidence_score.toFixed(1)}</strong></div><div><small>SUPPORT</small><strong>{pct(selectedFailure.support)}</strong></div></div><div className="detailSection"><div className="subHeading">SEARCH SUPPORT</div><div className="supportList">{(selectedFailure.strategy_kinds ?? []).map((kind) => <div key={kind}><span>{familyLabel(kind)}</span><StatusDot tone="green" /></div>)}{!(selectedFailure.strategy_kinds?.length) && <div><span>Single validated hypothesis family</span><StatusDot tone="blue" /></div>}</div></div><div className="detailSection"><div className="subHeading">STATISTICAL EVIDENCE</div><div className="statLine"><span>Adjusted p-value</span><strong>{selectedFailure.adjusted_p_value.toExponential(2)}</strong></div><div className="statLine"><span>Effect interval</span><strong>{pp(selectedFailure.ci_low)} → {pp(selectedFailure.ci_high)}</strong></div><div className="statLine"><span>Reproducible</span><strong>{selectedFailure.reproducible ? "YES" : "PENDING"}</strong></div></div><div className="detailSection"><div className="subHeading">BOUNDARY-SENSITIVITY EVIDENCE</div>{selectedCounterexamples.length ? selectedCounterexamples.slice(0, 4).map((c, i) => <div className="miniEvidence" key={c.counterexample_id ?? i}><span>{c.changed_feature}</span><strong>{c.original_value.toFixed(4)} → {c.counterfactual_value.toFixed(4)}</strong><small>{c.original_prediction} → {c.counterfactual_prediction} · {c.search_status}</small></div>) : <div className="panelFootnote">No boundary-sensitivity evidence attached to this region.</div>}</div></> : <EmptyState eyebrow="FAILURE ATLAS" title="Select a validated region" copy="Choose a failure from the list to inspect its evidence chain." />}</div></section>}

          {activeTab === "experiments" && <section className="contentPanel"><SectionHeader eyebrow="EXPERIMENT LAB" title="Executed hypotheses" copy={`${result.active_investigation.experiments_considered} candidates considered · ${result.active_investigation.experiments_executed} executed · budget ${result.active_investigation.budget}`} /><div className="experimentTable"><div className="experimentHead"><span>ID</span><span>FAMILY</span><span>HYPOTHESIS</span><span>GAP</span><span>ADJ. P</span><span>STATE</span></div>{result.active_investigation.observations.map((x) => <div className="experimentRow" key={x.experiment_id}><span className="monoTiny">{x.experiment_id}</span><span className="strategyText">{familyLabel(x.kind ?? "grid")}</span><span className="truncate">{x.condition}</span><span>{pp(x.gap)}</span><span className="monoTiny">{x.adjusted_p_value !== undefined ? x.adjusted_p_value.toExponential(1) : "—"}</span><span>{x.validated ? <span className="statusInline success">VALIDATED</span> : <span className="statusInline">SCREENED</span>}</span></div>)}</div></section>}

          {activeTab === "coverage" && <section className="coveragePage"><section className="contentPanel coverageHero"><SectionHeader eyebrow="SEARCH COVERAGE" title={coverage ? `${pct(coverage.candidate_execution_ratio)} of generated candidates executed` : "Coverage unavailable"} copy={coverage?.interpretation ?? "The backend did not return search coverage for this investigation."} /><div className="coverageRing" style={{ background: `conic-gradient(var(--accent2) ${coverage ? coverage.candidate_execution_ratio * 100 : 0}%, rgba(255,255,255,.05) 0)` }}><div><strong>{coverage ? pct(coverage.candidate_execution_ratio) : "—"}</strong><span>candidate execution</span></div></div><div className="coverageFacts"><div><span>GENERATED</span><strong>{coverage?.candidate_pool_size ?? "—"}</strong></div><div><span>EXECUTED</span><strong>{coverage?.executed ?? "—"}</strong></div><div><span>UNEXPLORED</span><strong>{coverage?.unexplored_candidates ?? "—"}</strong></div><div><span>FAMILY BREADTH</span><strong>{coverage ? pct(coverage.family_breadth_ratio) : "—"}</strong></div><div><span>FEATURE BREADTH</span><strong>{coverage ? pct(coverage.feature_breadth_ratio) : "—"}</strong></div></div></section>{coverage && <div className="familyGrid">{Object.entries(coverage.families).map(([key, family]) => <section className="contentPanel familyCard" key={key}><div className="familyTop"><span>{familyLabel(key)}</span><b>{pct(family.candidate_execution_ratio)}</b></div><div className="familyBar"><span style={{ width: `${Math.min(100, family.candidate_execution_ratio * 100)}%` }} /></div><div className="familyStats"><span>{family.generated_candidates} generated</span><span>{family.executed} executed</span><span>{family.unexplored_candidates} unexplored</span><span>{family.validated_failures} validated</span></div><div className="familyFeatures">{family.candidate_features.slice(0, 6).map((feature) => <code key={feature}>{feature}</code>)}</div></section>)}</div>}</section>}
        </>}

        {activeTab === "regression" && <section className="regressionPage">
          {loading === "compare" || loading === "compare-uploaded" ? <div className="runBanner"><div><span className="eyebrow blue">MODEL REGRESSION</span><strong>Running comparison…</strong><small>{loading === "compare" ? "Investigating two synthetic benchmark versions." : "Investigating both uploaded versions against the selected dataset."}</small></div><div className="loaderBar" /></div> : null}
          <div className="pageIntro"><div><div className="eyebrow blue">MODEL REGRESSION</div><h1>Compare behavior across versions.</h1><p>Track which validated failure regions disappear, persist, or emerge after a model change.</p></div><button className="quietButton" onClick={compareVersions} disabled={loading !== null}>{loading === "compare" ? "Comparing…" : "Run demo comparison"}</button></div>
          {comparison ? <section className="comparisonScene"><div className="comparisonVisual contentPanel"><div className={`comparisonBadge ${isDemoComparison(comparison) ? "demo" : "uploaded"}`}>{isDemoComparison(comparison) ? "SYNTHETIC BENCHMARK COMPARISON" : "UPLOADED MODEL COMPARISON"}</div><div className="comparisonBasis">Basis · {basisLabel(comparison.summary.comparison_basis)}</div><div className="versionRail"><div><span>MODEL V1</span><strong>{pct(comparison.models.v1.accuracy)}</strong><small>{comparison.models.v1.model_type ?? comparison.models.v1.asset_id ?? "v1"}</small></div><div className={`deltaArrow ${accuracyDelta >= 0 ? "positive" : "negative"}`}><b>{accuracyDelta >= 0 ? "+" : ""}{accuracyDelta.toFixed(1)} pp</b><i>behavior delta</i></div><div><span>MODEL V2</span><strong>{pct(comparison.models.v2.accuracy)}</strong><small>{comparison.models.v2.model_type ?? comparison.models.v2.asset_id ?? "v2"}</small></div></div><div className="regressionCounts"><div><b>{comparison.summary.fixed}</b><span>fixed regions</span></div><div><b>{comparison.summary.persistent}</b><span>persistent</span></div><div><b>{comparison.summary.new}</b><span>new regions</span></div><div><b>{comparison.summary.failures_v1} → {comparison.summary.failures_v2}</b><span>validated total</span></div></div></div><div className="comparisonMap contentPanel"><div className="eyebrow">REGRESSION FLOW</div><div className="regFlow"><span>V1<br /><b>{comparison.summary.failures_v1}</b></span><i /><span>CLUSTER MATCH<br /><b>{comparison.summary.deltas.length}</b></span><i /><span>V2<br /><b>{comparison.summary.failures_v2}</b></span></div><p>{comparison.comparability?.reason ?? "Comparison uses the backend's cluster-aware matching basis."}</p></div></section> : <section className="contentPanel"><EmptyState eyebrow="MODEL REGRESSION" title="No comparison has been run yet." copy="Compare the synthetic demo models or register two of your own model artifacts and one evaluation dataset." primary={<button className="primaryButton" onClick={compareVersions} disabled={loading !== null}>Compare demo versions</button>} /></section>}

          <section className="contentPanel registrationPanel"><SectionHeader eyebrow="UPLOADED MODEL REGRESSION" title="Bring your own model versions" copy="Register two compatible model artifacts and one labeled evaluation dataset. Comparison remains grounded in the real backend path." /><div className="regUploadGrid"><label className="fileField"><span>MODEL VERSION ARTIFACT</span><input type="file" accept=".joblib,.pkl,.pickle" onChange={(e: ChangeEvent<HTMLInputElement>) => setRegModelFile(e.target.files?.[0] ?? null)} /><b>{regModelFile?.name ?? "Choose .joblib / .pkl"}</b></label><label className="fileField"><span>EVALUATION DATASET</span><input type="file" accept=".csv" onChange={(e: ChangeEvent<HTMLInputElement>) => setRegDatasetFile(e.target.files?.[0] ?? null)} /><b>{regDatasetFile?.name ?? "Choose labeled .csv"}</b></label></div><div className="regUploadActions"><button className="quietButton" onClick={registerModelArtifact} disabled={uploading || !regModelFile}>{uploading ? "Uploading…" : "Register model"}</button><button className="quietButton" onClick={registerDatasetArtifact} disabled={uploading || !regDatasetFile}>{uploading ? "Uploading…" : "Register dataset"}</button><span className="helperText">Registration stores the artifact; investigation begins only when comparison runs.</span></div><div className="regSelectors"><label><span>MODEL V1</span><select value={cmpModelV1} onChange={(e: ChangeEvent<HTMLSelectElement>) => setCmpModelV1(e.target.value)} disabled={uploadedModels.length < 2}><option value="">Select model</option>{uploadedModels.map((m) => <option key={m.asset_id} value={m.asset_id}>{m.filename}</option>)}</select></label><label><span>MODEL V2</span><select value={cmpModelV2} onChange={(e: ChangeEvent<HTMLSelectElement>) => setCmpModelV2(e.target.value)} disabled={uploadedModels.length < 2}><option value="">Select model</option>{uploadedModels.map((m) => <option key={m.asset_id} value={m.asset_id}>{m.filename}</option>)}</select></label><label><span>DATASET</span><select value={cmpDataset} onChange={(e: ChangeEvent<HTMLSelectElement>) => setCmpDataset(e.target.value)} disabled={!uploadedDatasets.length}><option value="">Select dataset</option>{uploadedDatasets.map((d) => <option key={d.asset_id} value={d.asset_id}>{d.filename}</option>)}</select></label></div><div className="regCompareActions"><button className="primaryButton" onClick={compareUploadedModels} disabled={loading !== null || !cmpModelV1 || !cmpModelV2 || !cmpDataset}>{loading === "compare-uploaded" ? "Comparing…" : "Compare models"}<Icon name="arrow" /></button><div className="sessionCount">{uploadedModels.length} models · {uploadedDatasets.length} datasets registered this session</div></div></section>
          {comparison && <section className="contentPanel"><SectionHeader eyebrow="REGRESSION DELTAS" title="Failure movement" copy={`Showing the first 20 of ${comparison.summary.deltas.length} matched transitions.`} /><div className="regressionList">{comparison.summary.deltas.slice(0, 20).map((d) => <div className="regressionListRow" key={`${d.experiment_id_v1 ?? "none"}-${d.experiment_id_v2 ?? "none"}-${d.status}-${d.condition_v1 ?? d.condition_v2 ?? "region"}`}><div className="regressionStatus"><span className={`statusInline ${d.status.toLowerCase()}`}>{d.status}</span><strong>{d.condition_v2 ?? d.condition_v1 ?? "Unmatched region"}</strong><small>{d.experiment_id_v1 ?? "—"} → {d.experiment_id_v2 ?? "—"}</small></div><div className="regressionDeltaValue">{d.gap_v1 !== null && d.gap_v2 !== null ? `${pp(d.gap_v1)} → ${pp(d.gap_v2)}` : d.gap_v2 !== null ? `NEW · ${pp(d.gap_v2)}` : d.gap_v1 !== null ? `FIXED · was ${pp(d.gap_v1)}` : "—"}</div></div>)}</div></section>}
        </section>}
      </div>
    </section>
  </main>;
}
