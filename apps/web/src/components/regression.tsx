import { useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import type { CompareResult, UploadedArtifact } from "../lib/types";
import { pct, basisLabel, isDemoComparison } from "../lib/format";
import { Icon, SectionHeader, EmptyState } from "./ui";
import { RegressionChart } from "./charts";
import { RegressionTransition } from "./regression-transition";
export function Regression({
  loading,
  busy,
  uploading,
  comparison,
  accuracyDelta,
  compareVersions,
  regModelFile,
  regDatasetFile,
  setRegModelFile,
  setRegDatasetFile,
  registerModelArtifact,
  registerDatasetArtifact,
  cmpModelV1,
  cmpModelV2,
  cmpDataset,
  setCmpModelV1,
  setCmpModelV2,
  setCmpDataset,
  uploadedModels,
  uploadedDatasets,
  targetColumn,
  setTargetColumn,
  metric,
  setMetric,
  compareUploadedModels,
}: {
  loading: "investigate" | "compare" | "compare-uploaded" | null;
  busy: boolean;
  uploading: boolean;
  comparison: CompareResult | null;
  accuracyDelta: number;
  compareVersions: () => void;
  regModelFile: File | null;
  regDatasetFile: File | null;
  setRegModelFile: (file: File | null) => void;
  setRegDatasetFile: (file: File | null) => void;
  registerModelArtifact: () => void;
  registerDatasetArtifact: () => void;
  cmpModelV1: string;
  cmpModelV2: string;
  cmpDataset: string;
  setCmpModelV1: (id: string) => void;
  setCmpModelV2: (id: string) => void;
  setCmpDataset: (id: string) => void;
  uploadedModels: UploadedArtifact[];
  uploadedDatasets: UploadedArtifact[];
  targetColumn: string;
  setTargetColumn: (value: string) => void;
  metric: "accuracy" | "balanced_accuracy";
  setMetric: (value: "accuracy" | "balanced_accuracy") => void;
  compareUploadedModels: () => void;
}) {
  const [transition, setTransition] = useState("ALL");
  const modelInput = useRef<HTMLInputElement>(null);
  const datasetInput = useRef<HTMLInputElement>(null);
  useEffect(() => { if (!regModelFile && modelInput.current) modelInput.current.value = ""; }, [regModelFile]);
  useEffect(() => { if (!regDatasetFile && datasetInput.current) datasetInput.current.value = ""; }, [regDatasetFile]);
  const visibleDeltas =
    comparison?.summary.deltas.filter(
      (delta) => transition === "ALL" || delta.status === transition,
    ) ?? [];
  return (
    <section className="regressionPage">
      {loading === "compare" || loading === "compare-uploaded" ? (
        <div className="runBanner">
          <div>
            <span className="eyebrow blue">MODEL REGRESSION</span>
            <strong>Running comparison…</strong>
            <small>
              {loading === "compare"
                ? "Investigating two synthetic benchmark versions."
                : "Investigating both uploaded versions against the selected dataset."}
            </small>
          </div>
          <p className="panelFootnote">
            The synchronous comparison API does not report progress or support
            cancellation.
          </p>
        </div>
      ) : null}
      <div className="pageIntro">
        <div>
          <div className="eyebrow blue">MODEL REGRESSION</div>
          <h1>Compare behavior across versions.</h1>
          <p>
            Track which validated failure regions disappear, persist, or emerge
            after a model change.
          </p>
        </div>
        <button
          className="quietButton"
          onClick={compareVersions}
          disabled={busy}
        >
          {loading === "compare" ? "Comparing…" : "Run demo comparison"}
        </button>
      </div>
      {comparison ? (
        <section className="comparisonScene">
          <div className="comparisonVisual contentPanel">
            <div
              className={`comparisonBadge ${isDemoComparison(comparison) ? "demo" : "uploaded"}`}
            >
              {isDemoComparison(comparison)
                ? "SYNTHETIC BENCHMARK COMPARISON"
                : "UPLOADED MODEL COMPARISON"}
            </div>
            <p className="panelFootnote">
              Zero persistent regions means no matches under the backend
              matching criteria; it does not prove total behavioral
              dissimilarity.
            </p>
            <div className="comparisonBasis">
              Basis · {basisLabel(comparison.summary.comparison_basis)}
            </div>
            <div className="versionRail">
              <div>
                <span>MODEL V1</span>
                <strong>{pct(comparison.models.v1.accuracy)}</strong>
                <small>
                  {comparison.models.v1.model_type ??
                    comparison.models.v1.asset_id ??
                    "v1"}
                </small>
              </div>
              <div
                className={`deltaArrow ${accuracyDelta >= 0 ? "positive" : "negative"}`}
              >
                <b>
                  {accuracyDelta >= 0 ? "+" : ""}
                  {accuracyDelta.toFixed(1)} pp
                </b>
                <i>baseline score change</i>
              </div>
              <div>
                <span>MODEL V2</span>
                <strong>{pct(comparison.models.v2.accuracy)}</strong>
                <small>
                  {comparison.models.v2.model_type ??
                    comparison.models.v2.asset_id ??
                    "v2"}
                </small>
              </div>
            </div>
            <div className="regressionCounts">
              <div>
                <b>{comparison.summary.fixed}</b>
                <span>fixed regions</span>
              </div>
              <div>
                <b>{comparison.summary.persistent}</b>
                <span>persistent</span>
              </div>
              <div>
                <b>{comparison.summary.new}</b>
                <span>new regions</span>
              </div>
              <div>
                <b>
                  {comparison.summary.failures_v1} →{" "}
                  {comparison.summary.failures_v2}
                </b>
                <span>validated total</span>
              </div>
            </div>
          </div>
          <div className="comparisonMap contentPanel">
            <div className="eyebrow">REGRESSION FLOW</div>
            <div className="regFlow">
              <span>
                V1
                <br />
                <b>{comparison.summary.failures_v1}</b>
              </span>
              <i />
              <span>
                CLUSTER MATCH
                <br />
                <b>{comparison.summary.deltas.length}</b>
              </span>
              <i />
              <span>
                V2
                <br />
                <b>{comparison.summary.failures_v2}</b>
              </span>
            </div>
            <p>
              {comparison.comparability?.reason ??
                "Comparison uses the backend's cluster-aware matching basis."}
            </p>
            {comparison.comparability && (
              <p role="status">
                {comparison.comparability.comparable
                  ? "Backend reports comparable runs."
                  : "Backend reports these runs are not comparable."}
              </p>
            )}
            {comparison.comparability?.warnings.map((warning) => (
              <p className="comparabilityWarning" key={warning}>
                {warning}
              </p>
            ))}
          </div>
        </section>
      ) : (
        <section className="contentPanel regressionEmpty">
          <EmptyState
            eyebrow="MODEL REGRESSION"
            title="No comparison has been run yet."
            copy="Compare the synthetic demo models or register two of your own model artifacts and one evaluation dataset."
            primary={
              <button
                className="primaryButton"
                onClick={compareVersions}
                disabled={busy}
              >
                Compare demo versions
              </button>
            }
          />
        </section>
      )}

      <section className="contentPanel registrationPanel">
        <SectionHeader
          eyebrow="UPLOADED MODEL REGRESSION"
          title="Bring your own model versions"
          copy="Register two compatible model artifacts and one labeled evaluation dataset. Comparison remains grounded in the real backend path."
        />
        <div className="regUploadGrid">
          <label className="fileField">
            <span>MODEL VERSION ARTIFACT</span>
            <input
              aria-label="MODEL VERSION ARTIFACT"
              ref={modelInput}
              disabled={busy}
              type="file"
              accept=".joblib,.pkl,.pickle"
              onChange={(e: ChangeEvent<HTMLInputElement>) =>
                setRegModelFile(e.target.files?.[0] ?? null)
              }
            />
            <b>{regModelFile?.name ?? "Choose .joblib / .pkl"}</b>
          </label>
          <label className="fileField">
            <span>EVALUATION DATASET</span>
            <input
              aria-label="EVALUATION DATASET"
              ref={datasetInput}
              disabled={busy}
              type="file"
              accept=".csv"
              onChange={(e: ChangeEvent<HTMLInputElement>) =>
                setRegDatasetFile(e.target.files?.[0] ?? null)
              }
            />
            <b>{regDatasetFile?.name ?? "Choose labeled .csv"}</b>
          </label>
        </div>
        <div className="regUploadActions">
          <button
            className="quietButton"
            onClick={registerModelArtifact}
            disabled={busy || !regModelFile}
          >
            {uploading ? "Uploading…" : "Register model"}
          </button>
          <button
            className="quietButton"
            onClick={registerDatasetArtifact}
            disabled={busy || !regDatasetFile}
          >
            {uploading ? "Uploading…" : "Register dataset"}
          </button>
          <span className="helperText">
            Registration stores the artifact; investigation begins only when
            comparison runs.
          </span>
        </div>
        <div className="regSelectors">
          <label>
            <span>MODEL V1</span>
            <select
              aria-label="MODEL V1"
              value={cmpModelV1}
              onChange={(e: ChangeEvent<HTMLSelectElement>) =>
                setCmpModelV1(e.target.value)
              }
              disabled={busy || uploadedModels.length < 2}
            >
              <option value="">Select model</option>
              {uploadedModels.map((m) => (
                <option key={m.asset_id} value={m.asset_id}>
                  {m.filename} · {m.asset_id}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>MODEL V2</span>
            <select
              aria-label="MODEL V2"
              value={cmpModelV2}
              onChange={(e: ChangeEvent<HTMLSelectElement>) =>
                setCmpModelV2(e.target.value)
              }
              disabled={busy || uploadedModels.length < 2}
            >
              <option value="">Select model</option>
              {uploadedModels.map((m) => (
                <option key={m.asset_id} value={m.asset_id}>
                  {m.filename} · {m.asset_id}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>DATASET</span>
            <select
              aria-label="DATASET"
              value={cmpDataset}
              onChange={(e: ChangeEvent<HTMLSelectElement>) =>
                setCmpDataset(e.target.value)
              }
              disabled={busy || !uploadedDatasets.length}
            >
              <option value="">Select dataset</option>
              {uploadedDatasets.map((d) => (
                <option key={d.asset_id} value={d.asset_id}>
                  {d.filename} · {d.asset_id}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="setupControls">
          <label>
            <span>REGRESSION TARGET COLUMN</span>
            <input
              disabled={busy}
            value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
            />
          </label>
          <label>
            <span>REGRESSION METRIC</span>
            <select
              disabled={busy}
              aria-label="REGRESSION METRIC"
              value={metric}
              onChange={(e) =>
                setMetric(e.target.value as "accuracy" | "balanced_accuracy")
              }
            >
              <option value="accuracy">Accuracy</option>
              <option value="balanced_accuracy">Balanced accuracy</option>
            </select>
          </label>
        </div>
        {cmpModelV1 && cmpModelV1 === cmpModelV2 && <p id="regression-selection-help" className="selectionHint" role="status">Choose different model assets for V1 and V2 to compare versions.</p>}
        <div className="regCompareActions">
          <button
            className="primaryButton"
            aria-describedby={cmpModelV1 && cmpModelV1 === cmpModelV2 ? "regression-selection-help" : undefined}
            onClick={compareUploadedModels}
            disabled={
              busy ||
              !cmpModelV1 ||
              !cmpModelV2 ||
              !cmpDataset ||
              cmpModelV1 === cmpModelV2 ||
              !targetColumn.trim()
            }
          >
            {loading === "compare-uploaded" ? "Comparing…" : "Compare models"}
            <Icon name="arrow" />
          </button>
          <div className="sessionCount">
            {uploadedModels.length} models · {uploadedDatasets.length} datasets
            available in this workspace
          </div>
        </div>
      </section>
      {comparison && (
        <section className="contentPanel regressionResults">
          <SectionHeader
            eyebrow="REGRESSION DELTAS"
            title="Failure movement"
            copy={`${comparison.summary.deltas.length} transitions returned by the backend matching process.`}
          />
          <RegressionChart comparison={comparison} />
          <div
            className="transitionFilters"
            role="group"
            aria-label="Filter regression transitions"
          >
            {["ALL", "FIXED", "PERSISTENT", "NEW"].map((status) => (
              <button
                className="quietButton"
                aria-pressed={transition === status}
                key={status}
                onClick={() => setTransition(status)}
              >
                {status === "ALL"
                  ? "All transitions"
                  : status[0] + status.slice(1).toLowerCase()}{" "}
                <span>
                  {status === "ALL"
                    ? comparison.summary.deltas.length
                    : comparison.summary.deltas.filter(
                        (delta) => delta.status === status,
                      ).length}
                </span>
              </button>
            ))}
          </div>
          <p className="filterSummary" role="status">
            Showing {visibleDeltas.length} of {comparison.summary.deltas.length}{" "}
            returned transitions.
          </p>
          {!visibleDeltas.length && (
            <div className="filteredEmpty">
              <h3>
                No{" "}
                {transition === "ALL" ? "returned" : transition.toLowerCase()}{" "}
                transitions.
              </h3>
              <p>
                This describes the backend’s returned matches, not every
                possible model behavior.
              </p>
            </div>
          )}
          <div className="regressionList" role="region" aria-label="Regression transition list" tabIndex={0}>
            {visibleDeltas.map((d) => (
              <RegressionTransition key={`${d.experiment_id_v1 ?? "none"}-${d.experiment_id_v2 ?? "none"}-${d.status}-${d.condition_v1 ?? d.condition_v2 ?? "region"}`} delta={d} />
            ))}
          </div>
        </section>
      )}
    </section>
  );
}
