from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from modelxray.core.adapters import CountingClassifierAdapter, SklearnClassifierAdapter
from modelxray.core.certificate import build_run_certificate
from modelxray.core.profiler import SUPPORTED_METRICS, evaluate_baseline, profile_dataset
from modelxray.investigation.controller import run_active_investigation
from modelxray.validation.contracts import ContractError, validate_prediction_contract

WORKER_PROTOCOL_VERSION = 2  # keep in sync with execution.sandbox


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def run(
    model_path: str,
    dataset_path: str,
    target_column: str,
    budget: int,
    random_state: int,
    metric: str = "accuracy",
    feature_manifest: list[str] | None = None,
) -> dict[str, Any]:
    if metric not in SUPPORTED_METRICS:
        raise ContractError(f"Unsupported metric '{metric}'. Supported: {', '.join(SUPPORTED_METRICS)}.")

    model = joblib.load(model_path)
    frame = pd.read_csv(dataset_path)
    if target_column not in frame.columns:
        raise ContractError(f"Target column '{target_column}' was not found in the dataset.")

    X = frame.drop(columns=[target_column]).copy()
    y = frame[target_column].copy()
    if X.isna().any().any() or y.isna().any():
        raise ContractError(
            "Uploaded evaluation data contains missing values. The current engine "
            "requires complete rows; drop or impute missing cells before uploading."
        )
    if len(X) < 100:
        raise ContractError("Dataset must contain at least 100 evaluation rows.")
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    if not numeric:
        raise ContractError("At least one numeric feature is required by the current investigation engine.")

    adapter = CountingClassifierAdapter(SklearnClassifierAdapter(model))
    validate_prediction_contract(adapter, X, y, feature_manifest=feature_manifest)

    X_discovery, X_validation, y_discovery, y_validation = train_test_split(
        X.reset_index(drop=True),
        y.reset_index(drop=True),
        test_size=0.40,
        stratify=y,
        random_state=random_state,
    )
    dataset_profile = profile_dataset(X_validation)
    baseline = evaluate_baseline(adapter, X_validation, y_validation, metric=metric)
    active_run = run_active_investigation(
        adapter,
        X_discovery,
        y_discovery,
        X_validation,
        y_validation,
        budget=budget,
        metric=metric,
    )
    import sklearn

    run_id = uuid.uuid4().hex[:12]
    certificate = build_run_certificate(
        run_id=run_id,
        engine_version=_project_version(),
        model_hash=_sha256_file(model_path),
        dataset_hash=_sha256_file(dataset_path),
        feature_manifest=X.columns.tolist(),
        metric=metric,
        budget=budget,
        random_state=random_state,
        configuration={"target_column": target_column, "worker_protocol_version": WORKER_PROTOCOL_VERSION},
    )

    return {
        "baseline": {"accuracy": baseline.accuracy, "error_rate": baseline.error_rate},
        "model_type": type(model).__name__,
        "classes": getattr(getattr(model, "classes_", None), "tolist", lambda: [])(),
        "rows": int(dataset_profile.rows),
        "columns": int(dataset_profile.columns),
        "features": X.columns.tolist(),
        "numeric_features": dataset_profile.numeric_features,
        "categorical_features": dataset_profile.categorical_features,
        "missing_cells": int(dataset_profile.missing_cells),
        "instabilities": [],
        "compute": adapter.summary(),
        "active_investigation": {
            "budget": active_run.budget,
            "experiments_considered": active_run.experiments_considered,
            "experiments_executed": active_run.experiments_executed,
            "observations": active_run.observations,
            "search_coverage": active_run.search_coverage,
            "failure_clusters": active_run.failure_clusters or [],
            "failure_cluster_count": len(active_run.failure_clusters or []),
        },
        "counterexamples": active_run.counterexamples,
        "run_certificate": certificate.to_dict(),
        "manifest": {
            "project_version": _project_version(),
            "python_version": sys.version.split(" ")[0],
            "sklearn_version": sklearn.__version__,
            "model_hash": _sha256_file(model_path),
            "dataset_hash": _sha256_file(dataset_path),
            "metric": metric,
            "budget": int(budget),
            "random_state": int(random_state),
            "feature_manifest": X.columns.tolist(),
        },
    }


def _project_version() -> str:
    try:
        from importlib.metadata import version

        return version("modelxray")
    except Exception:
        try:
            from modelxray import __version__

            return __version__
        except Exception:
            return "unknown"


def _sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--random-state", type=int, required=True)
    parser.add_argument("--metric", default="accuracy", choices=list(SUPPORTED_METRICS))
    parser.add_argument("--feature-manifest", default="", help="JSON list of expected feature names (schema manifest).")
    parser.add_argument(
        "--result-path",
        required=True,
        help="Path where the JSON result is written. stdout is reserved for model logging.",
    )
    args = parser.parse_args()

    feature_manifest: list[str] | None = None
    if args.feature_manifest:
        try:
            parsed = json.loads(args.feature_manifest)
            if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
                raise ValueError("manifest must be a JSON list of strings")
            feature_manifest = parsed
        except (json.JSONDecodeError, ValueError):
            print(json.dumps({"error": "Invalid feature manifest JSON."}, separators=(",", ":")))
            raise SystemExit(2) from None

    try:
        result = run(
            args.model,
            args.dataset,
            args.target,
            args.budget,
            args.random_state,
            metric=args.metric,
            feature_manifest=feature_manifest,
        )
        envelope = {
            "protocol_version": WORKER_PROTOCOL_VERSION,
            "run_id": uuid.uuid4().hex[:12],
            "result": _json_safe(result),
        }
    except ContractError as exc:
        envelope = {"protocol_version": WORKER_PROTOCOL_VERSION, "error": str(exc)}
    except Exception as exc:  # internal failure: report and exit nonzero
        envelope = {"protocol_version": WORKER_PROTOCOL_VERSION, "error": f"Internal worker failure: {exc}"}
        _write_result(args.result_path, envelope)
        print(json.dumps(envelope, separators=(",", ":")))
        raise SystemExit(2) from exc

    _write_result(args.result_path, envelope)


def _write_result(result_path: str, envelope: dict[str, Any]) -> None:
    """Write the result envelope; resilient to stdout pollution from models."""
    Path(result_path).write_text(
        json.dumps(envelope, separators=(",", ":"), default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
