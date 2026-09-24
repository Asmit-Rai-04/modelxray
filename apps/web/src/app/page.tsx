"use client";

import { useMemo, useState } from "react";

const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

type Observation = {
  experiment_id: string;
  condition: string;
  kind?: string;
  support: number;
  gap: number;
  p_value: number;
  severity: string;
  validated: boolean;
  validation_gap: number;
  adjusted_p_value?: number;
  evidence_score?: number;
  holdout_effect_ci_low?: number;
  holdout_effect_ci_high?: number;
  holdout_ci_credible?: boolean;
};

type AtlasFailure = {
  failure_id: string;
  experiment_id: string;
  condition: string;
  severity: string;
  support: number;
  gap: number;
  adjusted_p_value: number;
  effect_size: number;
  ci_low: number;
  ci_high: number;
  validation_gap: number;
  reproducible: boolean;
  evidence_score: number;
  counterexample_ids: string[];
  strategy_kinds?: string[];
  strategy_diversity?: number;
};

type Counterexample = {
  counterexample_id?: string;
  source_experiment_id: string;
  source_condition: string;
  source_row: number;
  original_prediction: string | number;
  counterfactual_prediction: string | number;
  changed_feature: string;
  original_value: number;
  counterfactual_value: number;
  absolute_change: number;
  relative_change: number;
  search_status: string;
};

type Result = {
  model: { model_type: string; classes: unknown[] };
  dataset: { rows: number; columns: number; numeric_features?: string[]; categorical_features?: string[] };
  baseline: { accuracy: number; error_rate: number };
  metric?: string;
  instabilities: Array<{ feature: string; relative_change: number; original_prediction: string; perturbed_prediction: string }>;
  experiment_count: number;
  active_investigation: {
    budget: number;
    experiments_considered: number;
    experiments_executed: number;
    observations: Observation[];
    search_coverage?: {
      scope: string;
      interpretation: string;
      candidate_pool_size: number;
      executed: number;
      unexplored_candidates: number;
      candidate_execution_ratio: number;
      available_families: string[];
      explored_families: string[];
      family_breadth_ratio: number;
      candidate_feature_count: number;
      executed_feature_count: number;
      feature_breadth_ratio: number;
      families: Record<string, {
        generated_candidates: number;
        executed: number;
        unexplored_candidates: number;
        candidate_execution_ratio: number;
        validated_failures: number;
        candidate_features: string[];
        executed_features: string[];
      }>;
    };
  };
  investigation_id: string;
  failure_atlas: AtlasFailure[];
  counterexamples: Counterexample[];
};

type ComparisonModel = {
  accuracy: number;
  model_type?: string;
  asset_id?: string;
  investigation_id?: string;
};

type RegressionDelta = {
  experiment_id_v1: string | null;
  experiment_id_v2: string | null;
  condition_v1: string | null;
  condition_v2: string | null;
  match_type: string;
  status: "FIXED" | "PERSISTENT" | "NEW";
  severity_v1: string | null;
  severity_v2: string | null;
  gap_v1: number | null;
  gap_v2: number | null;
  evidence_score_v1: number | null;
  evidence_score_v2: number | null;
  evidence_delta: number | null;
  validation_gap_v1: number | null;
  validation_gap_v2: number | null;
  failure_cluster_id_v1?: string | null;
  failure_cluster_id_v2?: string | null;
};

type CompareResult = {
  mode: string;
  status?: string;
  dataset?: { rows: number; features: string[] };
  models: { v1: ComparisonModel; v2: ComparisonModel };
  summary: {
    fixed: number;
    persistent: number;
    new: number;
    failures_v1: number;
    failures_v2: number;
    net_failure_change: number;
    failure_reduction_ratio: number;
    deltas: RegressionDelta[];
    comparison_basis?: string;
  };
  failure_deltas?: RegressionDelta[];
  comparability?: { comparable: boolean; reason: string | null; warnings: string[] } | null;
};

type UploadedArtifact = { asset_id: string; filename: string; rows?: number };
type Tab = "overview" | "failures" | "experiments" | "regression";

function pct(value: number) { return `${(value * 100).toFixed(1)}%`; }
function pp(value: number) { return `${(value * 100).toFixed(1)} pp`; }
function basisLabel(basis?: string) { return basis === "failure_clusters_then_hypotheses" ? "failure clusters, then hypotheses" : (basis ?? "failure clusters, then hypotheses").replaceAll("_", " "); }
function isDemoComparison(comparison: CompareResult) { return comparison.mode !== "UPLOADED_MODEL_REGRESSION"; }
function familyLabel(key: string) {
  const labels: Record<string, string> = { grid: "NUMERIC", categorical: "CATEGORICAL", interaction: "INTERACTION", oblique_band: "OBLIQUE", prototype: "NOVELTY", prototype_region: "NOVELTY" };
  return labels[key] ?? key.replaceAll("_", " ").toUpperCase();
}
async function readApiError(response: Response) {
  try {
    const payload = await response.json();
    const detail = payload?.detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail.message === "string") return detail.message;
  } catch {}
  return `Request failed (HTTP ${response.status})`;
}

function Icon({ name }: { name: "overview" | "failures" | "experiments" | "regression" | "docs" }) {
  const paths: Record<string, string> = {
    overview: "M4 4h7v7H4z M13 4h7v7h-7z M4 13h7v7H4z M13 13h7v7h-7z",
    failures: "M5 19 9 5l6 8 4-10 M4 19h16",
    experiments: "M5 5h14v14H5z M8 9h8 M8 13h6 M8 17h4",
    regression: "M4 16l5-5 4 3 7-8 M4 20h16",
    docs: "M6 4h10l3 3v13H6z M16 4v4h4 M9 12h6 M9 16h6",
  };
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d={paths[name]} fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}

function StatusDot({ tone = "blue" }: { tone?: "blue" | "green" | "red" | "amber" }) { return <span className={`statusDot ${tone}`} />; }

function Severity({ value }: { value: string }) {
  const tone = value.toUpperCase() === "HIGH" ? "danger" : value.toUpperCase() === "MEDIUM" ? "warning" : "neutral";
  return <span className={`severityPill ${tone}`}><i />{value.toUpperCase()}</span>;
}

function SectionHeader({ eyebrow, title, copy, action }: { eyebrow: string; title: string; copy?: string; action?: React.ReactNode }) {
  return <div className="sectionHeader"><div><div className="eyebrow">{eyebrow}</div><h2>{title}</h2>{copy && <p>{copy}</p>}</div>{action}</div>;
}

function MetricCard({ label, value, sub, tone = "blue" }: { label: string; value: string; sub: string; tone?: string }) {
  return <div className={`metricCard tone-${tone}`}><div className="metricLabel">{label}</div><div className="metricValue">{value}</div><div className="metricSub">{sub}</div></div>;
}

function EmptyState({ eyebrow, title, copy, primary, secondary }: { eyebrow: string; title: string; copy: string; primary?: React.ReactNode; secondary?: React.ReactNode }) {
  return <div className="emptyState"><div className="emptyIcon"><div className="emptyIconCore" /></div><div><div className="eyebrow">{eyebrow}</div><h2>{title}</h2><p>{copy}</p><div className="emptyActions">{primary}{secondary}</div></div></div>;
}

export default function Home() {
  const [result, setResult] = useState<Result | null>(null);
  const [comparison, setComparison] = useState<CompareResult | null>(null);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [loading, setLoading] = useState<"investigate" | "compare" | "compare-uploaded" | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("overview");
  const [selectedFailureId, setSelectedFailureId] = useState<string | null>(null);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [uploadedModels, setUploadedModels] = useState<UploadedArtifact[]>([]);
  const [uploadedDatasets, setUploadedDatasets] = useState<UploadedArtifact[]>([]);
  const [cmpModelV1, setCmpModelV1] = useState("");
  const [cmpModelV2, setCmpModelV2] = useState("");
  const [cmpDataset, setCmpDataset] = useState("");
  const [regModelFile, setRegModelFile] = useState<File | null>(null);
  const [regDatasetFile, setRegDatasetFile] = useState<File | null>(null);
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [modelAssetId, setModelAssetId] = useState<string | null>(null);
  const [datasetAssetId, setDatasetAssetId] = useState<string | null>(null);
  const [targetColumn, setTargetColumn] = useState("target");
  const [metric, setMetric] = useState<"accuracy" | "balanced_accuracy">("accuracy");
  const [uploading, setUploading] = useState(false);

  async function runInvestigation() {
    setLoading("investigate");
    try {
      const response = await fetch(`${API_BASE}/api/v1/demo/investigate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ rows: 2500, random_state: 42 }) });
      if (!response.ok) throw new Error(`API ${response.status}`);
      const data = await response.json();
      setResult(data);
      setSelectedFailureId(data.failure_atlas?.[0]?.failure_id ?? null);
      setActiveTab("overview");
    } catch (error) {
      console.error(error);
      setComparisonError("Start the ModelXray API on port 8000 first.");
    } finally { setLoading(null); }
  }

  async function uploadAndInvestigate() {
    if (!modelFile || !datasetFile || !targetColumn.trim()) return setComparisonError("Choose a model, CSV dataset, and target column first.");
    setUploading(true);
    try {
      const modelForm = new FormData(); modelForm.append("file", modelFile);
      const modelResponse = await fetch(`${API_BASE}/api/v1/assets/model`, { method: "POST", body: modelForm });
      if (!modelResponse.ok) throw new Error(await readApiError(modelResponse));
      const modelAsset = await modelResponse.json();
      setModelAssetId(modelAsset.asset_id);
      setUploadedModels((prev) => prev.some((m) => m.asset_id === modelAsset.asset_id) ? prev : [...prev, { asset_id: modelAsset.asset_id, filename: modelAsset.filename ?? modelFile.name }]);

      const datasetForm = new FormData(); datasetForm.append("file", datasetFile);
      const datasetResponse = await fetch(`${API_BASE}/api/v1/assets/dataset`, { method: "POST", body: datasetForm });
      if (!datasetResponse.ok) throw new Error(await readApiError(datasetResponse));
      const datasetAsset = await datasetResponse.json();
      setDatasetAssetId(datasetAsset.asset_id);
      setUploadedDatasets((prev) => prev.some((d) => d.asset_id === datasetAsset.asset_id) ? prev : [...prev, { asset_id: datasetAsset.asset_id, filename: datasetAsset.filename ?? datasetFile.name, rows: datasetAsset.metadata?.rows }]);

      const investigationResponse = await fetch(`${API_BASE}/api/v1/investigate/uploaded`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_id: modelAsset.asset_id, dataset_id: datasetAsset.asset_id, target_column: targetColumn.trim(), budget: 24, random_state: 42, metric }) });
      if (!investigationResponse.ok) throw new Error(await readApiError(investigationResponse));
      const data = await investigationResponse.json();
      setResult(data); setSelectedFailureId(data.failure_atlas?.[0]?.failure_id ?? null); setActiveTab("overview");
    } catch (error) {
      console.error(error); setComparisonError(error instanceof Error ? error.message : "ModelXray could not process the uploaded artifacts.");
    } finally { setUploading(false); }
  }

  async function compareVersions() {
    setLoading("compare"); setComparisonError(null);
    try {
      const response = await fetch(`${API_BASE}/api/v1/demo/compare`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ rows: 2500, random_state: 42 }) });
      if (!response.ok) throw new Error(await readApiError(response));
      setComparison(await response.json()); setActiveTab("regression");
    } catch (error) {
      setComparisonError(error instanceof Error ? `${error.message}. Is the ModelXray API running on port 8000?` : "The comparison request failed.");
    } finally { setLoading(null); }
  }

  async function registerModelArtifact() {
    if (!regModelFile) return setComparisonError("Choose a .joblib/.pkl model artifact to register.");
    setUploading(true); setComparisonError(null);
    try {
      const form = new FormData(); form.append("file", regModelFile);
      const response = await fetch(`${API_BASE}/api/v1/assets/model`, { method: "POST", body: form });
      if (!response.ok) throw new Error(await readApiError(response));
      const asset = await response.json(); const assetId: string = asset.asset_id;
      setUploadedModels((prev) => prev.some((m) => m.asset_id === assetId) ? prev : [...prev, { asset_id: assetId, filename: asset.filename ?? regModelFile.name }]);
      if (!cmpModelV1) setCmpModelV1(assetId); else if (!cmpModelV2 && cmpModelV1 !== assetId) setCmpModelV2(assetId);
      setRegModelFile(null);
    } catch (error) { setComparisonError(error instanceof Error ? error.message : "Model artifact registration failed."); }
    finally { setUploading(false); }
  }

  async function registerDatasetArtifact() {
    if (!regDatasetFile) return setComparisonError("Choose a labeled .csv dataset to register.");
    setUploading(true); setComparisonError(null);
    try {
      const form = new FormData(); form.append("file", regDatasetFile);
      const response = await fetch(`${API_BASE}/api/v1/assets/dataset`, { method: "POST", body: form });
      if (!response.ok) throw new Error(await readApiError(response));
      const asset = await response.json(); const assetId: string = asset.asset_id;
      setUploadedDatasets((prev) => prev.some((d) => d.asset_id === assetId) ? prev : [...prev, { asset_id: assetId, filename: asset.filename ?? regDatasetFile.name, rows: asset.metadata?.rows }]);
      if (!cmpDataset) setCmpDataset(assetId); setRegDatasetFile(null);
    } catch (error) { setComparisonError(error instanceof Error ? error.message : "Dataset registration failed."); }
    finally { setUploading(false); }
  }

  async function compareUploadedModels() {
    if (!cmpModelV1 || !cmpModelV2 || !cmpDataset) return setComparisonError("Select two uploaded model artifacts and a dataset first.");
    if (cmpModelV1 === cmpModelV2) return setComparisonError("Model v1 and model v2 must be different artifacts.");
    setLoading("compare-uploaded"); setComparisonError(null);
    try {
      const response = await fetch(`${API_BASE}/api/v1/regression/uploaded`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_v1_id: cmpModelV1, model_v2_id: cmpModelV2, dataset_id: cmpDataset, target_column: targetColumn.trim() || "target", budget: 24, random_state: 42, metric }) });
      if (!response.ok) throw new Error(await readApiError(response));
      setComparison(await response.json()); setActiveTab("regression");
    } catch (error) { setComparisonError(error instanceof Error ? error.message : "The uploaded-model comparison failed."); }
    finally { setLoading(null); }
  }

  const failureCount = result?.failure_atlas.length ?? 0;
  const high = result?.failure_atlas.filter((x) => x.severity.toUpperCase() === "HIGH").length ?? 0;
  const validated = result?.active_investigation.observations.filter((x) => x.validated).length ?? 0;
  const reproducible = result?.failure_atlas.filter((x) => x.reproducible).length ?? 0;
  const selectedFailure = result?.failure_atlas.find((x) => x.failure_id === selectedFailureId) ?? result?.failure_atlas[0] ?? null;
  const selectedCounterexamples = selectedFailure ? (result?.counterexamples.filter((x) => selectedFailure.counterexample_ids.includes(x.counterexample_id ?? "")) ?? []) : [];
  const accuracyDelta = comparison ? (comparison.models.v2.accuracy - comparison.models.v1.accuracy) * 100 : 0;

  const severityCounts = useMemo(() => {
    const source = result?.failure_atlas ?? [];
    return ["HIGH", "MEDIUM", "LOW"].map((label) => ({ label, count: source.filter((x) => x.severity.toUpperCase() === label).length }));
  }, [result]);

  const coverage = result?.active_investigation.search_coverage;
  const evidenceTrend = (result?.failure_atlas ?? []).slice(0, 8).map((item) => Math.min(100, item.evidence_score));

  const navItems: Array<{ id: Tab; label: string; icon: "overview" | "failures" | "experiments" | "regression" }> = [
    { id: "overview", label: "Overview", icon: "overview" },
    { id: "failures", label: "Failure Atlas", icon: "failures" },
    { id: "experiments", label: "Experiments", icon: "experiments" },
    { id: "regression", label: "Regression", icon: "regression" },
  ];

  return (
    <main className="appShell">
      <aside className="sidebar">
        <div className="brandLockup">
          <div className="brandGlyph">M<span>X</span></div>
          <div><div className="brandName">ModelXray</div><div className="brandCaption">MODEL ASSURANCE</div></div>
        </div>
        <div className="sidebarSectionLabel">Workspace</div>
        <nav className="navStack">
          {navItems.map((item) => <button key={item.id} className={`navButton ${activeTab === item.id ? "active" : ""}`} onClick={() => setActiveTab(item.id)}><span className="navIcon"><Icon name={item.icon} /></span><span>{item.label}</span></button>)}
        </nav>
        <div className="sidebarSpacer" />
        <div className="sidebarSectionLabel">System</div>
        <button className="navButton mutedNav"><span className="navIcon"><Icon name="docs" /></span><span>Documentation</span></button>
        <div className="systemCard"><div className="systemCardTop"><span><StatusDot tone="green" /> API online</span><span className="monoTiny">:8000</span></div><small>Local workspace · trusted artifacts</small></div>
      </aside>

      <section className="mainArea">
        <header className="appHeader">
          <div className="headerContext"><span className="headerKicker">ML MODEL ASSURANCE</span><span className="headerDivider" /><span className="headerStatus"><StatusDot tone="green" /> ENGINE READY</span></div>
          <div className="headerActions"><button className="quietButton" onClick={compareVersions} disabled={loading !== null}>{loading === "compare" ? "Comparing…" : "Quick compare"}</button><button className="primaryButton" onClick={runInvestigation} disabled={loading !== null}>{loading === "investigate" ? "Running…" : "New investigation"}</button></div>
        </header>

        <div className="contentWrap">
          {!result && !comparison && activeTab !== "regression" ? (
            <section className="heroLayout">
              <div className="heroCopy">
                <div className="eyebrow blue">MODEL FAILURE DISCOVERY</div>
                <h1>Find where your model breaks.</h1>
                <p className="heroLead">ModelXray systematically probes a trained model, validates suspicious regions, and preserves the evidence behind every finding.</p>
                <div className="heroActions"><button className="primaryButton large" onClick={runInvestigation} disabled={loading !== null}>{loading === "investigate" ? "Running investigation…" : "Run demo investigation"}</button><button className="quietButton large" onClick={() => setActiveTab("regression")}>Explore regression</button></div>
                <div className="heroTrust"><span><StatusDot tone="green" /> Deterministic core</span><span><StatusDot tone="blue" /> FDR + holdout validation</span><span><StatusDot tone="blue" /> Evidence ledger</span></div>
              </div>

              <div className="probePreview">
                <div className="previewTop"><span className="previewLabel">INVESTIGATION PIPELINE</span><span className="previewState"><StatusDot tone="blue" /> READY</span></div>
                <div className="previewModel"><div><small>SUPPORTED INPUT</small><strong>Trained tabular classifier</strong></div><span className="modelTag">BLACK-BOX</span></div>
                <div className="pipeline">
                  {[['01','Profile','Model + dataset contract'],['02','Search','Multiple hypothesis families'],['03','Validate','FDR + holdout gate'],['04','Record','Failure cluster + evidence']].map(([num,title,copy]) => <div className="pipelineStep" key={num}><span className="pipelineNum">{num}</span><div><strong>{title}</strong><small>{copy}</small></div><span className="pipelineRail" /></div>)}
                </div>
                <div className="previewFooter"><span>coverage is measured against generated hypotheses</span><span className="monoTiny">MODELXRAY / CORE</span></div>
              </div>
            </section>
          ) : null}

          {(!result && comparison && activeTab !== "regression") ? <section className="contentPanel"><EmptyState eyebrow="NO INVESTIGATION LOADED" title="This workspace is waiting for a model investigation." copy="A comparison is loaded, but this screen needs a single-model investigation. Run one to populate the evidence panels." primary={<button className="primaryButton" onClick={runInvestigation} disabled={loading !== null}>{loading === "investigate" ? "Running…" : "Run investigation"}</button>} secondary={<button className="quietButton" onClick={() => setActiveTab("regression")}>View regression</button>} /></section> : null}

          {result && activeTab !== "regression" && (
            <>
              <div className="pageIntro"><div><div className="eyebrow blue">INVESTIGATION</div><h1>{activeTab === "failures" ? "Failure Atlas" : activeTab === "experiments" ? "Experiment Lab" : "Model overview"}</h1><p>{activeTab === "overview" ? "A measured view of where the current model becomes unreliable." : activeTab === "failures" ? "Validated behavioral regions with their supporting evidence and search provenance." : "The hypotheses ModelXray actually executed under the investigation budget."}</p></div><div className="pageIntroMeta"><span className="metaLabel">INVESTIGATION ID</span><code>{result.investigation_id}</code></div></div>

              {activeTab === "overview" && (
                <>
                  <div className="metricGridTop">
                    <MetricCard label={(result.metric ?? "accuracy").replace("_", " ").toUpperCase()} value={pct(result.baseline.accuracy)} sub="validation baseline" tone="cyan" />
                    <MetricCard label="VALIDATED FAILURES" value={`${failureCount}`} sub={`${reproducible} reproducible`} tone="blue" />
                    <MetricCard label="HIGH SEVERITY" value={`${high}`} sub="priority regions" tone="red" />
                    <MetricCard label="EXPERIMENTS" value={`${result.active_investigation.experiments_executed}/${result.active_investigation.budget}`} sub={`${validated} cleared validation`} tone="violet" />
                    <MetricCard label="SEARCH COVERAGE" value={coverage ? pct(coverage.candidate_execution_ratio) : "—"} sub="generated candidates tested" tone="green" />
                  </div>

                  <div className="mainGrid">
                    <section className="contentPanel span2"><SectionHeader eyebrow="MODEL SNAPSHOT" title="What was investigated" copy={`${result.model.model_type} · ${result.dataset.rows.toLocaleString()} rows · ${result.dataset.columns} features`} action={<span className="pill quiet">{result.metric === "balanced_accuracy" ? "BALANCED ACCURACY" : "ACCURACY"}</span>} /><div className="snapshotGrid"><div><small>NUMERIC FEATURES</small><strong>{result.dataset.numeric_features?.length ?? 0}</strong></div><div><small>CATEGORICAL FEATURES</small><strong>{result.dataset.categorical_features?.length ?? 0}</strong></div><div><small>HYPOTHESES CONSIDERED</small><strong>{result.active_investigation.experiments_considered}</strong></div><div><small>BOUNDARY EVIDENCE</small><strong>{result.counterexamples.length}</strong></div></div></section>
                    <section className="contentPanel"><SectionHeader eyebrow="SEVERITY" title="Failure mix" /><div className="severityList">{severityCounts.map((item) => <div className="severityItem" key={item.label}><div><Severity value={item.label} /><strong>{item.count}</strong></div><div className="severityTrack"><span className={`fill ${item.label.toLowerCase()}`} style={{ width: `${Math.min(100, (item.count / Math.max(1, failureCount)) * 100)}%` }} /></div></div>)}</div><div className="panelNote">Severity is a transparent heuristic derived from validated evidence; it is not a probability of harm.</div></section>

                    <section className="contentPanel span2"><SectionHeader eyebrow="FAILURE CLUSTERS" title="Where the model becomes unreliable" copy="Overlapping validated hypotheses are consolidated into behavioral clusters." action={<button className="linkButton" onClick={() => setActiveTab("failures")}>Open atlas ↗</button>} />{failureCount === 0 ? <div className="cleanState"><StatusDot tone="green" /><div><strong>No validated failures found.</strong><p>The current investigation did not find a region that survived the statistical and holdout gates.</p></div></div> : <div className="clusterTable"><div className="clusterHead"><span>REGION</span><span>SEVERITY</span><span>GAP</span><span>EVIDENCE</span><span>SUPPORT</span></div>{result.failure_atlas.slice(0, 7).map((f) => <button key={f.failure_id} className={`clusterRow ${selectedFailureId === f.failure_id ? "selected" : ""}`} onClick={() => { setSelectedFailureId(f.failure_id); setActiveTab("failures"); }}><span><strong>{f.condition}</strong><small>{f.failure_id} · {f.strategy_diversity ?? f.strategy_kinds?.length ?? 1} search strategies</small></span><Severity value={f.severity} /><span>{pp(f.gap)}</span><span className="monoScore">{f.evidence_score.toFixed(1)}</span><span>{pct(f.support)}</span></button>)}</div>}</section>

                    <section className="contentPanel"><SectionHeader eyebrow="SEARCH COVERAGE" title="What was actually explored" /><div className="coverageCompact">{coverage ? Object.entries(coverage.families).slice(0, 5).map(([key, fam]) => <div className="coverageMini" key={key}><div><span>{familyLabel(key)}</span><strong>{pct(fam.candidate_execution_ratio)}</strong></div><div className="miniTrack"><span style={{ width: `${Math.min(100, fam.candidate_execution_ratio * 100)}%` }} /></div><small>{fam.executed} / {fam.generated_candidates} tested</small></div>) : <div className="panelNote">Run an investigation to measure coverage.</div>}</div></section>

                    <section className="contentPanel"><SectionHeader eyebrow="INVESTIGATION TRACE" title="Evidence lifecycle" /><div className="traceList">{[['01','PROFILE','Model + dataset contract'],['02','SEARCH',`${result.active_investigation.experiments_executed} experiments executed`],['03','VALIDATE',`${validated} observations cleared the gates`],['04','CONSOLIDATE',`${failureCount} failure clusters persisted`],['05','EVIDENCE',`${result.counterexamples.length} boundary checks attached`]].map(([n,t,c]) => <div className="traceRow" key={n}><span>{n}</span><div><strong>{t}</strong><small>{c}</small></div><StatusDot tone="green" /></div>)}</div></section>
                  </div>

                  {result.counterexamples.length > 0 && <section className="contentPanel"><SectionHeader eyebrow="BOUNDARY-SENSITIVITY EVIDENCE" title="Concrete prediction flips near validated failures" copy="These are attached to validated regions only. A boundary flip by itself is ordinary classifier behavior, not proof of breakage." action={<span className="pill">{result.counterexamples.length} ATTACHED</span>} /><div className="evidenceCards">{result.counterexamples.slice(0, 4).map((c, i) => <div className="evidenceCard" key={c.counterexample_id ?? i}><div className="evidenceCardTop"><span className="monoTiny">{c.counterexample_id ?? `CX-${i + 1}`}</span><span className="flipValue">{c.original_prediction} → {c.counterfactual_prediction}</span></div><h3>{c.changed_feature}</h3><div className="evidenceNumbers"><div><small>ORIGINAL</small><strong>{c.original_value.toFixed(4)}</strong></div><div><small>FLIPPED</small><strong>{c.counterfactual_value.toFixed(4)}</strong></div><div><small>DELTA</small><strong>{c.absolute_change.toFixed(4)}</strong></div></div></div>)}</div></section>}
                </>
              )}

              {activeTab === "failures" && (
                <section className="failureWorkspace"><div className="failureList contentPanel"><SectionHeader eyebrow="FAILURE ATLAS" title={`${failureCount} validated regions`} copy="Select a region to inspect its evidence chain." />{result.failure_atlas.map((f) => <button key={f.failure_id} className={`failureListItem ${selectedFailure?.failure_id === f.failure_id ? "selected" : ""}`} onClick={() => setSelectedFailureId(f.failure_id)}><div className="failureListTop"><span className="monoTiny">{f.failure_id}</span><Severity value={f.severity} /></div><strong>{f.condition}</strong><div className="failureMeta"><span>{pp(f.gap)} gap</span><span>evidence {f.evidence_score.toFixed(1)}</span><span>{pct(f.support)} support</span></div></button>)}</div><div className="failureDetail contentPanel">{selectedFailure ? <><SectionHeader eyebrow="SELECTED FAILURE" title={selectedFailure.condition} copy={selectedFailure.experiment_id} action={<Severity value={selectedFailure.severity} />} /><div className="detailGrid"><div><small>PERFORMANCE GAP</small><strong>{pp(selectedFailure.gap)}</strong></div><div><small>HOLDOUT GAP</small><strong>{pp(selectedFailure.validation_gap)}</strong></div><div><small>EVIDENCE</small><strong>{selectedFailure.evidence_score.toFixed(1)}</strong></div><div><small>SUPPORT</small><strong>{pct(selectedFailure.support)}</strong></div></div><div className="detailSection"><div className="subHeading">SEARCH SUPPORT</div><div className="supportList">{(selectedFailure.strategy_kinds ?? []).map((kind) => <div key={kind}><span>{familyLabel(kind)}</span><StatusDot tone="green" /></div>)}{!(selectedFailure.strategy_kinds?.length) && <div><span>Single validated hypothesis family</span><StatusDot tone="blue" /></div>}</div></div><div className="detailSection"><div className="subHeading">STATISTICAL EVIDENCE</div><div className="statLine"><span>Adjusted p-value</span><strong>{selectedFailure.adjusted_p_value.toExponential(2)}</strong></div><div className="statLine"><span>Effect interval</span><strong>{pp(selectedFailure.ci_low)} to {pp(selectedFailure.ci_high)}</strong></div><div className="statLine"><span>Reproducible</span><strong>{selectedFailure.reproducible ? "YES" : "PENDING"}</strong></div></div><div className="detailSection"><div className="subHeading">BOUNDARY EVIDENCE</div>{selectedCounterexamples.length ? selectedCounterexamples.slice(0, 3).map((c, i) => <div className="miniEvidence" key={c.counterexample_id ?? i}><span>{c.changed_feature}</span><strong>{c.original_value.toFixed(4)} → {c.counterfactual_value.toFixed(4)}</strong><small>{c.original_prediction} → {c.counterfactual_prediction}</small></div>) : <div className="panelNote">No counterexample attached to this cluster.</div>}</div></> : <EmptyState eyebrow="FAILURE ATLAS" title="Select a validated region" copy="Choose a failure from the list to inspect its evidence chain." />}</div></section>
              )}

              {activeTab === "experiments" && (
                <section className="contentPanel"><SectionHeader eyebrow="EXPERIMENT LAB" title="Executed hypotheses" copy={`${result.active_investigation.experiments_considered} candidates considered · ${result.active_investigation.experiments_executed} executed · budget ${result.active_investigation.budget}`} /><div className="experimentTable"><div className="experimentHead"><span>ID</span><span>STRATEGY</span><span>HYPOTHESIS</span><span>GAP</span><span>ADJ. P</span><span>STATUS</span></div>{result.active_investigation.observations.map((x) => <div className="experimentRow" key={x.experiment_id}><span className="monoTiny">{x.experiment_id}</span><span className="strategyText">{familyLabel(x.kind ?? "grid")}</span><span className="truncate">{x.condition}</span><span>{pp(x.gap)}</span><span className="monoTiny">{x.adjusted_p_value !== undefined ? x.adjusted_p_value.toExponential(1) : "—"}</span><span>{x.validated ? <span className="statusInline success">VALIDATED</span> : <span className="statusInline">SCREENED</span>}</span></div>)}</div></section>
              )}
            </>
          )}

          {activeTab === "regression" && (
            <section className="regressionPage">
              {comparisonError && <div className="notice danger"><div><div className="eyebrow red">COMPARISON FAILED</div><strong>{comparisonError}</strong><small>{comparison ? "Previous comparison preserved." : "Retry after fixing the issue."}</small></div></div>}
              {(loading === "compare" || loading === "compare-uploaded") && <div className="notice"><div><div className="eyebrow blue">MODEL REGRESSION</div><strong>Running comparison…</strong><small>{loading === "compare" ? "Investigating two synthetic benchmark versions." : "Investigating both uploaded model versions against the selected dataset."}</small></div><div className="loaderBar" /></div>}

              <div className="regHeader"><div><div className="eyebrow blue">MODEL REGRESSION</div><h1>Compare model behavior across versions.</h1><p>Track which validated failure regions disappear, persist, or emerge after a model change.</p></div><div className="regHeaderActions"><button className="quietButton" onClick={compareVersions} disabled={loading !== null}>{loading === "compare" ? "Comparing…" : "Run demo comparison"}</button></div></div>

              {comparison ? <div className="comparisonHero contentPanel"><div className="comparisonSummary"><div className={`comparisonBadge ${isDemoComparison(comparison) ? "demo" : "uploaded"}`}>{isDemoComparison(comparison) ? "SYNTHETIC BENCHMARK" : "UPLOADED MODELS"}</div><div className="comparisonBasis">Basis · {basisLabel(comparison.summary.comparison_basis)}</div><div className="comparisonVersions"><div><small>MODEL V1</small><strong>{pct(comparison.models.v1.accuracy)}</strong><span>{comparison.models.v1.model_type ?? comparison.models.v1.asset_id ?? "v1"}</span></div><div className={`versionDelta ${accuracyDelta >= 0 ? "positive" : "negative"}`}>{accuracyDelta >= 0 ? "+" : ""}{accuracyDelta.toFixed(1)} pp</div><div><small>MODEL V2</small><strong>{pct(comparison.models.v2.accuracy)}</strong><span>{comparison.models.v2.model_type ?? comparison.models.v2.asset_id ?? "v2"}</span></div></div></div><div className="comparisonKpis"><div><small>FIXED</small><strong>{comparison.summary.fixed}</strong></div><div><small>PERSISTENT</small><strong>{comparison.summary.persistent}</strong></div><div><small>NEW</small><strong>{comparison.summary.new}</strong></div><div><small>REGIONS</small><strong>{comparison.summary.failures_v1} → {comparison.summary.failures_v2}</strong></div></div></div> : <EmptyState eyebrow="MODEL REGRESSION" title="No comparison has been run yet." copy="Compare the synthetic demo models or register two of your own model artifacts and one evaluation dataset." primary={<button className="primaryButton" onClick={compareVersions} disabled={loading !== null}>{loading === "compare" ? "Comparing…" : "Compare demo versions"}</button>} />}

              <section className="contentPanel registrationPanel"><SectionHeader eyebrow="UPLOADED MODEL REGRESSION" title="Bring your own model versions" copy="Upload two compatible artifacts and one labeled evaluation dataset. Nothing is hidden behind mock data." /><div className="regUploadGrid"><label className="fileField"><span>MODEL VERSION ARTIFACT</span><input type="file" accept=".joblib,.pkl,.pickle" onChange={(e) => setRegModelFile(e.target.files?.[0] ?? null)} /><b>{regModelFile?.name ?? "Choose .joblib / .pkl"}</b></label><label className="fileField"><span>EVALUATION DATASET</span><input type="file" accept=".csv" onChange={(e) => setRegDatasetFile(e.target.files?.[0] ?? null)} /><b>{regDatasetFile?.name ?? "Choose labeled .csv"}</b></label></div><div className="regUploadActions"><button className="quietButton" onClick={registerModelArtifact} disabled={uploading || !regModelFile}>{uploading ? "Uploading…" : "Register model"}</button><button className="quietButton" onClick={registerDatasetArtifact} disabled={uploading || !regDatasetFile}>{uploading ? "Uploading…" : "Register dataset"}</button><span className="helperText">Registration only stores the artifact; comparison runs only when you press Compare models.</span></div><div className="regSelectors"><label><span>MODEL V1</span><select value={cmpModelV1} onChange={(e) => setCmpModelV1(e.target.value)} disabled={uploadedModels.length < 2}><option value="">Select model</option>{uploadedModels.map((m) => <option key={m.asset_id} value={m.asset_id}>{m.filename}</option>)}</select></label><label><span>MODEL V2</span><select value={cmpModelV2} onChange={(e) => setCmpModelV2(e.target.value)} disabled={uploadedModels.length < 2}><option value="">Select model</option>{uploadedModels.map((m) => <option key={m.asset_id} value={m.asset_id}>{m.filename}</option>)}</select></label><label><span>DATASET</span><select value={cmpDataset} onChange={(e) => setCmpDataset(e.target.value)} disabled={!uploadedDatasets.length}><option value="">Select dataset</option>{uploadedDatasets.map((d) => <option key={d.asset_id} value={d.asset_id}>{d.filename}</option>)}</select></label></div><div className="regCompareActions"><button className="primaryButton" onClick={compareUploadedModels} disabled={loading !== null || !cmpModelV1 || !cmpModelV2 || !cmpDataset}>{loading === "compare-uploaded" ? "Comparing…" : "Compare models"}</button><div className="sessionCount">{uploadedModels.length} models · {uploadedDatasets.length} datasets registered this session</div></div></section>

              {comparison && <section className="contentPanel"><SectionHeader eyebrow="REGRESSION DELTAS" title="Failure movement" copy={`Showing the first 20 of ${comparison.summary.deltas.length} matched transitions.`} /><div className="regressionList">{comparison.summary.deltas.slice(0, 20).map((d) => <div className="regressionListRow" key={`${d.experiment_id_v1 ?? "none"}-${d.experiment_id_v2 ?? "none"}-${d.status}`}><div className="regressionStatus"><span className={`statusInline ${d.status.toLowerCase()}`}>{d.status}</span><strong>{d.condition_v2 ?? d.condition_v1 ?? "Unmatched region"}</strong><small>{d.experiment_id_v1 ?? "—"} → {d.experiment_id_v2 ?? "—"}</small></div><div className="regressionDeltaValue">{d.gap_v1 !== null && d.gap_v2 !== null ? `${pp(d.gap_v1)} → ${pp(d.gap_v2)}` : d.gap_v2 !== null ? `NEW · ${pp(d.gap_v2)}` : d.gap_v1 !== null ? `FIXED · was ${pp(d.gap_v1)}` : "—"}</div></div>)}</div></section>}
            </section>
          )}
        </div>
      </section>
    </main>
  );
}
