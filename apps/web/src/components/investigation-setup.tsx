import type { ChangeEvent, RefObject } from "react";
import { Icon } from "./ui";
export function InvestigationSetup({
  setupRef,
  modelFile,
  datasetFile,
  setModelFile,
  setDatasetFile,
  targetColumn,
  setTargetColumn,
  metric,
  setMetric,
  busy,
  uploading,
  uploadAndInvestigate,
  setShowInvestigationSetup,
  runInvestigation,
}: {
  setupRef: RefObject<HTMLElement | null>;
  modelFile: File | null;
  datasetFile: File | null;
  setModelFile: (file: File | null) => void;
  setDatasetFile: (file: File | null) => void;
  targetColumn: string;
  setTargetColumn: (value: string) => void;
  metric: "accuracy" | "balanced_accuracy";
  setMetric: (value: "accuracy" | "balanced_accuracy") => void;
  busy: boolean;
  uploading: boolean;
  uploadAndInvestigate: () => void;
  setShowInvestigationSetup: (value: boolean) => void;
  runInvestigation: () => void;
}) {
  return (
    <section
      ref={setupRef}
      className="investigationSetup contentPanel"
      aria-label="New investigation"
    >
      <div className="setupHeader">
        <div>
          <div className="eyebrow blue">NEW INVESTIGATION</div>
          <h2>Choose what ModelXray should inspect.</h2>
          <p>
            Upload a trained scikit-learn-compatible model and a labeled
            evaluation dataset. This creates a new single-model investigation;
            it does not alter your regression comparison.
          </p>
        </div>
        <button
          className="quietButton"
          onClick={() => setShowInvestigationSetup(false)}
        >
          Close
        </button>
      </div>
      <div className="setupGrid">
        <label className="setupField">
          <span>MODEL ARTIFACT</span>
          <input
            aria-label="MODEL ARTIFACT"
            disabled={busy}
            type="file"
            accept=".joblib,.pkl,.pickle"
            onChange={(e: ChangeEvent<HTMLInputElement>) =>
              setModelFile(e.target.files?.[0] ?? null)
            }
          />
          <strong>{modelFile?.name ?? "Choose .joblib / .pkl"}</strong>
          <small>Existing trained model · scikit-learn compatible</small>
        </label>
        <label className="setupField">
          <span>EVALUATION DATASET</span>
          <input
            aria-label="EVALUATION DATASET"
            disabled={busy}
            type="file"
            accept=".csv"
            onChange={(e: ChangeEvent<HTMLInputElement>) =>
              setDatasetFile(e.target.files?.[0] ?? null)
            }
          />
          <strong>{datasetFile?.name ?? "Choose labeled .csv"}</strong>
          <small>Must contain the target column used for evaluation</small>
        </label>
      </div>
      <div className="setupControls">
        <label>
          <span>TARGET COLUMN</span>
          <input
            disabled={busy}
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            placeholder="target"
          />
        </label>
        <label>
          <span>METRIC</span>
          <select
            disabled={busy}
            aria-label="METRIC"
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
      <div className="setupActions">
        <button
          className="primaryButton"
          onClick={uploadAndInvestigate}
          disabled={busy || !modelFile || !datasetFile || !targetColumn.trim()}
        >
          {uploading ? "Starting investigation…" : "Investigate uploaded model"}
          <Icon name="arrow" />
        </button>
        <button
          className="quietButton"
          onClick={() => {
            setShowInvestigationSetup(false);
            runInvestigation();
          }}
          disabled={busy}
        >
          Use benchmark demo
        </button>
        <span className="helperText">
          The demo uses the built-in synthetic benchmark and is clearly labeled.
        </span>
      </div>
    </section>
  );
}
