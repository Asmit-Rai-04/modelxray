from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable
from uuid import uuid4
from threading import Event

import pandas as pd

from modelxray.core.adapters import CountingClassifierAdapter, SklearnClassifierAdapter
from modelxray.core.ingestion import AssetError, AssetStore
from modelxray.core.certificate import build_run_certificate
from modelxray.core.profiler import evaluate_baseline, profile_dataset, profile_model
from modelxray.detectors.perturbation import discover_local_instability
from modelxray.execution.sandbox import run_isolated_investigation
from modelxray.investigation.controller import run_active_investigation
from modelxray.investigation.regression import (
    build_regression_summary,
    check_comparability,
    compare_failure_snapshots,
)
from modelxray.storage import EvidenceStore, FailureRecord
from modelxray.storage.evidence import InvestigationBundle
from modelxray.validation.contracts import InvestigationResult


def new_investigation_id() -> str:
    return f"INV-{uuid4().hex[:10].upper()}"


def _assemble_evidence(
    investigation_id: str,
    observations: list[dict[str, Any]],
    counterexamples: list[dict[str, Any]],
    failure_clusters: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], list[dict[str, Any]]]:
    """Assemble consolidated failure records + counterexample payloads.

    One failure record represents one consolidated FailureCluster. Individual
    validated hypotheses remain in the experiment ledger as supporting evidence.
    """
    counterexample_payloads: list[dict[str, Any]] = []
    counterexamples_by_experiment: dict[str, list[str]] = {}
    for idx, counterexample in enumerate(counterexamples, start=1):
        counterexample_id = f"CX-{investigation_id}-{idx:03d}"
        counterexamples_by_experiment.setdefault(counterexample.get("source_experiment_id", ""), []).append(counterexample_id)
        counterexample_payloads.append({**counterexample, "counterexample_id": counterexample_id})

    obs_by_id = {obs["experiment_id"]: obs for obs in observations}
    if failure_clusters is None:
        # Backward-compatible fallback for older worker payloads.
        failure_clusters = []
        for obs in observations:
            if obs.get("reproducible"):
                failure_clusters.append({
                    "cluster_id": obs.get("failure_cluster_id") or f"FC-{len(failure_clusters)+1:03d}",
                    "representative_experiment_id": obs["experiment_id"],
                    "supporting_experiment_ids": [obs["experiment_id"]],
                    "condition": obs["condition"],
                    "severity": obs["severity"],
                    "hypothesis_count": 1,
                    "support": obs["support"],
                    "gap": obs["gap"],
                    "validation_gap": obs["validation_gap"],
                    "best_adjusted_p_value": obs["adjusted_p_value"],
                    "best_evidence_score": obs["evidence_score"],
                    "mean_evidence_score": obs["evidence_score"],
                    "region_jaccard_to_representative": 1.0,
                })

    failure_atlas: list[dict[str, Any]] = []
    for cluster in failure_clusters:
        supporting_ids = list(cluster.get("supporting_experiment_ids", []))
        representative_id = cluster["representative_experiment_id"]
        rep = obs_by_id.get(representative_id)
        if rep is None or not rep.get("reproducible"):
            continue
        cluster_counterexamples = [
            cx_id
            for experiment_id in supporting_ids
            for cx_id in counterexamples_by_experiment.get(experiment_id, [])
        ]
        failure_id = f"F-{investigation_id}-{len(failure_atlas)+1:03d}"
        record = FailureRecord(
            failure_id=failure_id,
            investigation_id=investigation_id,
            experiment_id=representative_id,
            detector="active_investigation",
            condition=cluster.get("condition", rep["condition"]),
            severity=cluster.get("severity", rep["severity"]),
            support=float(cluster.get("support", rep["support"])),
            gap=float(cluster.get("gap", rep["gap"])),
            adjusted_p_value=float(cluster.get("best_adjusted_p_value", rep["adjusted_p_value"])),
            effect_size=float(rep.get("effect_size", 0.0)),
            ci_low=float(rep.get("effect_ci_low", 0.0)),
            ci_high=float(rep.get("effect_ci_high", 0.0)),
            validation_gap=float(cluster.get("validation_gap", rep["validation_gap"])),
            reproducible=True,
            evidence_score=float(cluster.get("best_evidence_score", rep["evidence_score"])),
            counterexample_ids=cluster_counterexamples,
            failure_cluster_id=cluster.get("cluster_id"),
            supporting_experiment_ids=supporting_ids,
            hypothesis_count=int(cluster.get("hypothesis_count", len(supporting_ids) or 1)),
        )
        failure_atlas.append({**record.__dict__, "mean_evidence_score": float(cluster.get("mean_evidence_score", record.evidence_score)), "region_jaccard_to_representative": float(cluster.get("region_jaccard_to_representative", 1.0))})
    return failure_atlas, counterexamples_by_experiment, counterexample_payloads


def finalize_investigation(
    investigation_id: str,
    result_payload: dict[str, Any],
    failure_atlas: list[dict[str, Any]],
    counterexample_payloads: list[dict[str, Any]],
    evidence_store: EvidenceStore,
) -> None:
    """Persist the complete investigation bundle atomically."""
    bundle = InvestigationBundle(
        investigation_id=investigation_id,
        payload=result_payload,
        failures=[
            FailureRecord(**failure) for failure in failure_atlas
        ],
        counterexamples=counterexample_payloads,
    )
    evidence_store.save_bundle(bundle)


def run_demo_investigation(
    model: Any,
    X_discovery: pd.DataFrame,
    y_discovery: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    budget: int = 24,
    metric: str = "accuracy",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """One in-process demo investigation (demo models are trusted fixtures)."""
    adapter = CountingClassifierAdapter(SklearnClassifierAdapter(model))
    dataset_profile = profile_dataset(X_validation)
    model_profile = profile_model(adapter)
    baseline = evaluate_baseline(adapter, X_validation, y_validation, metric=metric)
    instabilities = discover_local_instability(adapter, X_validation)
    active_run = run_active_investigation(
        adapter,
        X_discovery,
        y_discovery,
        X_validation,
        y_validation,
        budget=budget,
        metric=metric,
        progress_callback=progress_callback,
    )

    investigation_id = new_investigation_id()
    failure_atlas, _cx_map, cx_payloads = _assemble_evidence(
        investigation_id, active_run.observations, active_run.counterexamples, active_run.failure_clusters
    )

    result = InvestigationResult.build(
        model={
            "model_type": model_profile.model_type,
            "classes": model_profile.classes,
            "supports_probabilities": model_profile.supports_probabilities,
        },
        dataset={
            "rows": dataset_profile.rows,
            "columns": dataset_profile.columns,
            "numeric_features": dataset_profile.numeric_features,
            "categorical_features": dataset_profile.categorical_features,
            "missing_cells": dataset_profile.missing_cells,
        },
        baseline={
            "accuracy": baseline.accuracy,
            "error_rate": baseline.error_rate,
        },
        instabilities=instabilities,
        experiment_count=active_run.experiments_executed,
        active_investigation={
            "budget": active_run.budget,
            "experiments_considered": active_run.experiments_considered,
            "experiments_executed": active_run.experiments_executed,
            "observations": active_run.observations,
            "search_coverage": active_run.search_coverage,
            "failure_clusters": active_run.failure_clusters or [],
            "failure_cluster_count": len(active_run.failure_clusters or []),
        },
        counterexamples=active_run.counterexamples,
        failure_atlas=failure_atlas,
        investigation_id=investigation_id,
        metric=metric,
    )
    payload = result.__dict__
    payload["status"] = "COMPLETED"
    payload["mode"] = "DEMO"
    payload["api_version"] = "1.0.0"
    payload["metric"] = metric
    payload["compute"] = adapter.summary()
    payload["run_certificate"] = build_run_certificate(
        run_id=investigation_id,
        engine_version="1.0.0",
        model_hash=None,
        dataset_hash=None,
        feature_manifest=X_validation.columns.tolist(),
        metric=metric,
        budget=budget,
        random_state=42,
        configuration={"mode": "demo", "rows_discovery": len(X_discovery), "rows_validation": len(X_validation)},
    ).to_dict()
    return payload, active_run.observations, failure_atlas, cx_payloads


def run_uploaded_investigation(
    asset_store: AssetStore,
    evidence_store: EvidenceStore,
    *,
    model_id: str,
    dataset_id: str,
    target_column: str,
    budget: int = 24,
    random_state: int = 42,
    metric: str = "accuracy",
    cancel_event: Event | None = None,
) -> dict[str, Any]:
    """Full uploaded-model investigation: worker execution + persistence."""
    model_asset = asset_store.get(model_id)
    dataset_asset = asset_store.get(dataset_id)
    if model_asset is None or model_asset.asset_type != "model":
        raise AssetError(f"Model asset not found: {model_id}")
    if dataset_asset is None or dataset_asset.asset_type != "dataset":
        raise AssetError(f"Dataset asset not found: {dataset_id}")

    dataset_manifest = dataset_asset.metadata.get("feature_manifest") or dataset_asset.metadata.get("features")

    worker = run_isolated_investigation(
        model_path=model_asset.path,
        dataset_path=dataset_asset.path,
        target_column=target_column,
        budget=budget,
        random_state=random_state,
        metric=metric,
        # Feature manifest for models without feature_names_in_ (schema policy C):
        # the upload-time CSV column order minus the target column.
        feature_manifest=[c for c in (dataset_manifest or []) if c != target_column] or None,
        cancel_event=cancel_event,
    )

    investigation_id = new_investigation_id()
    observations = worker["active_investigation"]["observations"]
    counterexamples = worker["counterexamples"]
    failure_clusters = worker["active_investigation"].get("failure_clusters", [])
    # Single shared assembly (same as the demo path): one severity/ID/persistence
    # semantics for every investigation mode.
    failure_atlas, _cx_map, counterexample_payloads = _assemble_evidence(
        investigation_id, observations, counterexamples, failure_clusters
    )

    result = InvestigationResult.build(
        model={
            "model_type": worker["model_type"],
            "classes": worker["classes"],
            "supports_probabilities": worker.get("supports_probabilities", True),
            "source_model_id": model_id,
        },
        dataset={
            # Full upload size (audit finding #19), not the validation split.
            "rows": worker["rows"],
            "columns": worker["columns"],
            "numeric_features": worker["numeric_features"],
            "categorical_features": worker["categorical_features"],
            "missing_cells": worker["missing_cells"],
            "source_dataset_id": dataset_id,
            "target_column": target_column,
        },
        baseline=worker["baseline"],
        instabilities=[],
        experiment_count=worker["active_investigation"]["experiments_executed"],
        active_investigation=worker["active_investigation"],
        counterexamples=counterexample_payloads,
        failure_atlas=failure_atlas,
        investigation_id=investigation_id,
        metric=metric,
        manifest=worker.get("manifest"),
    )
    payload = result.__dict__
    payload["status"] = "COMPLETED"
    payload["mode"] = "UPLOADED_MODEL"
    payload["api_version"] = "1.0.0"
    payload["metric"] = metric
    payload["compute"] = worker.get("compute", {})
    payload["run_certificate"] = worker.get("run_certificate")
    payload["manifest"] = {**(worker.get("manifest") or {}), "model_asset_id": model_id, "dataset_asset_id": dataset_id, "target_column": target_column}

    finalize_investigation(investigation_id, payload, failure_atlas, counterexample_payloads, evidence_store)
    return payload


def compare_models(
    observations_v1: list[dict[str, Any]],
    observations_v2: list[dict[str, Any]],
    manifest_v1: dict[str, Any] | None = None,
    manifest_v2: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Regression comparison with manifest comparability check."""
    comparability = check_comparability(manifest_v1, manifest_v2)
    comparison = compare_failure_snapshots(observations_v1, observations_v2)
    summary = build_regression_summary(comparison)
    return {
        "summary": summary,
        "failure_deltas": [asdict(item) for item in comparison],
        "comparability": comparability,
    }
