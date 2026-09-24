from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from modelxray.core.adapters import SklearnClassifierAdapter
from modelxray.core.ingestion import AssetError, AssetStore
from modelxray.service.errors import to_investigation_error
from modelxray.service.job_service import InvestigationJobManager
from modelxray.service.replay_service import replay_investigation
from modelxray.service.investigation_service import (
    compare_models,
    finalize_investigation,
    run_demo_investigation,
    run_uploaded_investigation,
)
from modelxray.storage import EvidenceStore

app = FastAPI(title="ModelXray", version="1.0.0")

asset_store = AssetStore()
evidence_store = EvidenceStore()
job_manager = InvestigationJobManager(evidence_store, max_workers=2)


def _http_error(exc: Exception) -> HTTPException:
    err = to_investigation_error(exc)
    return HTTPException(status_code=err.status_code, detail={"kind": err.kind, "message": str(err)})


# --------------------------------------------------------------------------- 
# Demo fixtures: benchmark data with a genuinely discoverable seeded failure.
# The seeded region is a single-feature threshold region (discoverable by the
# baseline grid), unlike the v1.0 fixture whose interaction region produced an
# empty atlas on the shipped demo artifacts.
# ---------------------------------------------------------------------------


def build_demo_case(rows: int, random_state: int):
    X_arr, y = make_classification(
        n_samples=rows,
        n_features=8,
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[0.62, 0.38],
        class_sep=1.1,
        random_state=random_state,
    )
    X = pd.DataFrame(X_arr, columns=[f"feature_{i+1}" for i in range(X_arr.shape[1])])

    # Seed a *marginal* (single-threshold) failure region so the deterministic
    # engine reliably rediscovers it: feature_1 > q80 with 60% label corruption.
    threshold = float(X["feature_1"].quantile(0.80))
    failure_region = (X["feature_1"] > threshold).to_numpy()
    rng = np.random.default_rng(random_state)
    y_mut = y.copy()
    y_mut[failure_region & (rng.random(rows) < 0.60)] ^= 1

    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y_mut, test_size=0.40, stratify=y_mut, random_state=random_state
    )
    X_discovery, X_validation, y_discovery, y_validation = train_test_split(
        X_eval, y_eval, test_size=0.50, stratify=y_eval, random_state=random_state
    )
    model = RandomForestClassifier(
        n_estimators=160,
        min_samples_leaf=4,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return (
        model,
        X_discovery.reset_index(drop=True),
        pd.Series(np.asarray(y_discovery)).reset_index(drop=True),
        X_validation.reset_index(drop=True),
        pd.Series(np.asarray(y_validation)).reset_index(drop=True),
    )


class DemoFailureModel:
    """Black-box benchmark wrapper used only by the demo regression endpoint."""

    def __init__(self, model: Any, failure_masks: list[Any] | None = None):
        self.model = model
        self.failure_masks = failure_masks or []
        self.classes_ = np.asarray(getattr(model, "classes_", []))

    def predict(self, X: Any):
        pred = np.asarray(self.model.predict(X)).copy()
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for mask_fn in self.failure_masks:
            mask = np.asarray(mask_fn(frame), dtype=bool)
            if np.any(mask):
                if self.classes_.size != 2:
                    continue
                pred[mask] = np.where(pred[mask] == self.classes_[0], self.classes_[1], self.classes_[0])
        return pred

    def predict_proba(self, X: Any):
        proba = np.asarray(self.model.predict_proba(X)).copy()
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.classes_.size == 2:
            for mask_fn in self.failure_masks:
                mask = np.asarray(mask_fn(frame), dtype=bool)
                if np.any(mask):
                    proba[mask] = proba[mask][:, ::-1]
        return proba


class DemoInvestigationRequest(BaseModel):
    rows: int = Field(default=2500, ge=500, le=10000)
    random_state: int = 42
    budget: int = Field(default=24, ge=5, le=100)
    metric: Literal["accuracy", "balanced_accuracy"] = "accuracy"


class RealInvestigationRequest(BaseModel):
    model_id: str
    dataset_id: str
    target_column: str
    budget: int = Field(default=24, ge=5, le=100)
    random_state: int = 42
    metric: Literal["accuracy", "balanced_accuracy"] = "accuracy"


class JobSubmissionResponse(BaseModel):
    job_id: str
    kind: str
    status: str


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:59300", "http://127.0.0.1:59300"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/assets/model")
async def upload_model(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        asset = asset_store.save_model_bytes(file.filename or "model.joblib", file.file)
        return {"status": "uploaded", **asset.__dict__}
    except AssetError as exc:
        raise _http_error(exc) from exc


@app.post("/api/v1/assets/dataset")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        asset = asset_store.save_dataset_bytes(file.filename or "dataset.csv", file.file)
        return {"status": "uploaded", **asset.__dict__}
    except AssetError as exc:
        raise _http_error(exc) from exc


@app.get("/api/v1/assets/{asset_id}")
def get_asset(asset_id: str) -> dict[str, Any]:
    try:
        asset = asset_store.get(asset_id)
    except AssetError as exc:
        raise _http_error(exc) from exc
    if asset is None:
        raise HTTPException(status_code=404, detail={"kind": "asset_not_found", "message": "Asset not found"})
    return asset.__dict__


@app.post("/api/v1/investigate/uploaded", response_model=dict[str, Any])
def investigate_uploaded(request: RealInvestigationRequest) -> dict[str, Any]:
    try:
        return run_uploaded_investigation(
            asset_store,
            evidence_store,
            model_id=request.model_id,
            dataset_id=request.dataset_id,
            target_column=request.target_column,
            budget=request.budget,
            random_state=request.random_state,
            metric=request.metric,
        )
    except (AssetError, Exception) as exc:
        raise _http_error(exc) from exc


@app.post("/api/v1/jobs/demo/investigate", response_model=JobSubmissionResponse, status_code=202)
def submit_demo_job(request: DemoInvestigationRequest) -> dict[str, Any]:
    def run(progress, cancel_event) -> dict[str, Any]:
        model, X_discovery, y_discovery, X_validation, y_validation = build_demo_case(
            request.rows, request.random_state
        )
        payload, _observations, failure_atlas, cx_payloads = run_demo_investigation(
            model, X_discovery, y_discovery, X_validation, y_validation,
            budget=request.budget, metric=request.metric, progress_callback=progress,
        )
        finalize_investigation(payload["investigation_id"], payload, failure_atlas, cx_payloads, evidence_store)
        return payload

    submission = job_manager.submit("DEMO_INVESTIGATION", run, total_units=request.budget)
    return submission.__dict__


@app.post("/api/v1/jobs/investigate/uploaded", response_model=JobSubmissionResponse, status_code=202)
def submit_uploaded_job(request: RealInvestigationRequest) -> dict[str, Any]:
    def run(progress, cancel_event) -> dict[str, Any]:
        progress({"phase": "PREPARING_WORKER", "progress": 0.05, "completed_units": 0, "total_units": request.budget, "current_unit": "WORKER_START", "current_family": "worker"})
        result = run_uploaded_investigation(
            asset_store, evidence_store,
            model_id=request.model_id,
            dataset_id=request.dataset_id,
            target_column=request.target_column,
            budget=request.budget,
            random_state=request.random_state,
            metric=request.metric,
            cancel_event=cancel_event,
        )
        compute = result.get("compute", {}) if isinstance(result, dict) else {}
        progress({"phase": "FINALIZING", "progress": 0.97, "completed_units": request.budget, "total_units": request.budget, "current_unit": "PERSISTENCE", "current_family": "persistence", "predict_calls": compute.get("predict_calls"), "predict_rows": compute.get("predict_rows")})
        return result

    submission = job_manager.submit("UPLOADED_INVESTIGATION", run, total_units=request.budget)
    return submission.__dict__


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"kind": "job_not_found", "message": "Job not found"})
    return job


@app.post("/api/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    job = job_manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"kind": "job_not_found", "message": "Job not found"})
    if job["status"] in {"COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"}:
        return job
    return job


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "modelxray-api"}


@app.post("/api/v1/demo/investigate", response_model=dict[str, Any])
def investigate_demo(request: DemoInvestigationRequest) -> dict[str, Any]:
    model, X_discovery, y_discovery, X_validation, y_validation = build_demo_case(request.rows, request.random_state)
    payload, observations, failure_atlas, cx_payloads = run_demo_investigation(
        model, X_discovery, y_discovery, X_validation, y_validation,
        budget=request.budget, metric=request.metric,
    )
    finalize_investigation(payload["investigation_id"], payload, failure_atlas, cx_payloads, evidence_store)
    return payload


def build_demo_versions(rows: int, random_state: int):
    """Build two benchmark versions over the same clean holdout.

    V1 contains a deterministic single-threshold failure region A. V2 fixes A
    but introduces B. The underlying predictor is a real sklearn classifier;
    the wrapper is a controlled benchmark fixture.
    """
    X_arr, y_clean = make_classification(
        n_samples=rows,
        n_features=8,
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[0.62, 0.38],
        class_sep=1.1,
        random_state=random_state,
    )
    X = pd.DataFrame(X_arr, columns=[f"feature_{i+1}" for i in range(X_arr.shape[1])])
    q1 = float(X["feature_1"].quantile(0.80))
    q3 = float(X["feature_5"].quantile(0.70))

    region_a = lambda frame: frame["feature_1"] > q1
    region_b = lambda frame: frame["feature_5"] > q3

    idx_train, idx_eval = train_test_split(
        np.arange(rows), test_size=0.40, stratify=np.asarray(y_clean), random_state=random_state
    )
    idx_discovery, idx_validation = train_test_split(
        idx_eval, test_size=0.50, stratify=np.asarray(y_clean)[idx_eval], random_state=random_state
    )

    base_model = RandomForestClassifier(
        n_estimators=160, min_samples_leaf=4, random_state=random_state, n_jobs=-1
    ).fit(X.iloc[idx_train], np.asarray(y_clean)[idx_train])
    model_v1 = DemoFailureModel(base_model, [region_a])
    model_v2 = DemoFailureModel(base_model, [region_b])

    return (
        model_v1,
        model_v2,
        X.iloc[idx_discovery].reset_index(drop=True),
        pd.Series(np.asarray(y_clean)[idx_discovery]).reset_index(drop=True),
        X.iloc[idx_validation].reset_index(drop=True),
        pd.Series(np.asarray(y_clean)[idx_validation]).reset_index(drop=True),
    )


@app.post("/api/v1/demo/compare", response_model=dict[str, Any])
def compare_demo(request: DemoInvestigationRequest) -> dict[str, Any]:
    model_v1, model_v2, X_discovery, y_discovery, X_validation, y_validation = build_demo_versions(
        request.rows, request.random_state
    )
    adapter_v1 = SklearnClassifierAdapter(model_v1)
    adapter_v2 = SklearnClassifierAdapter(model_v2)
    from modelxray.investigation.controller import run_active_investigation

    run_v1 = run_active_investigation(
        adapter_v1, X_discovery, y_discovery, X_validation, y_validation,
        budget=request.budget, metric=request.metric,
    )
    run_v2 = run_active_investigation(
        adapter_v2, X_discovery, y_discovery, X_validation, y_validation,
        budget=request.budget, metric=request.metric,
    )
    comparison = compare_models(
        run_v1.observations, run_v2.observations,
        manifest_v1={"metric": request.metric, "budget": run_v1.budget, "random_state": request.random_state},
        manifest_v2={"metric": request.metric, "budget": run_v2.budget, "random_state": request.random_state},
    )
    return {
        "mode": "MODEL_REGRESSION",
        "dataset": {"rows": len(X_validation), "features": list(X_validation.columns)},
        "models": {
            "v1": {"model_type": type(model_v1).__name__, "accuracy": float(np.mean(adapter_v1.predict(X_validation) == np.asarray(y_validation)))},
            "v2": {"model_type": type(model_v2).__name__, "accuracy": float(np.mean(adapter_v2.predict(X_validation) == np.asarray(y_validation)))},
        },
        "investigations": {
            "v1": {"experiments_executed": run_v1.experiments_executed, "observations": run_v1.observations},
            "v2": {"experiments_executed": run_v2.experiments_executed, "observations": run_v2.observations},
        },
        **comparison,
    }


@app.get("/api/v1/investigations")
def list_investigations(limit: int = 25) -> dict[str, Any]:
    items = evidence_store.list_investigations(limit)
    return {"items": items, "count": len(items)}


@app.get("/api/v1/investigations/{investigation_id}")
def get_investigation(investigation_id: str) -> dict[str, Any]:
    payload = evidence_store.get_investigation(investigation_id)
    if payload is None:
        raise HTTPException(status_code=404, detail={"kind": "investigation_not_found", "message": "Investigation not found"})
    payload["failure_atlas"] = evidence_store.list_failures(investigation_id)
    payload["counterexamples"] = evidence_store.list_counterexamples(investigation_id)
    payload["experiments"] = evidence_store.list_experiments(investigation_id)
    return payload


@app.post("/api/v1/investigations/{investigation_id}/replay")
def replay_saved_investigation(investigation_id: str) -> dict[str, Any]:
    try:
        return replay_investigation(asset_store, evidence_store, investigation_id)
    except Exception as exc:
        raise _http_error(exc) from exc



class UploadedComparisonRequest(BaseModel):
    model_v1_id: str
    model_v2_id: str
    dataset_id: str
    target_column: str
    budget: int = Field(default=24, ge=5, le=100)
    random_state: int = 42
    metric: Literal["accuracy", "balanced_accuracy"] = "accuracy"


@app.post("/api/v1/regression/uploaded")
def compare_uploaded_models(request: UploadedComparisonRequest) -> dict[str, Any]:
    if request.model_v1_id == request.model_v2_id:
        raise HTTPException(
            status_code=409,
            detail={"kind": "invalid_state", "message": "Model v1 and model v2 must be different assets."},
        )
    try:
        v1 = run_uploaded_investigation(
            asset_store, evidence_store,
            model_id=request.model_v1_id, dataset_id=request.dataset_id,
            target_column=request.target_column, budget=request.budget,
            random_state=request.random_state, metric=request.metric,
        )
        v2 = run_uploaded_investigation(
            asset_store, evidence_store,
            model_id=request.model_v2_id, dataset_id=request.dataset_id,
            target_column=request.target_column, budget=request.budget,
            random_state=request.random_state, metric=request.metric,
        )
    except Exception as exc:
        raise _http_error(exc) from exc
    comparison = compare_models(
        v1["active_investigation"]["observations"],
        v2["active_investigation"]["observations"],
        manifest_v1=v1.get("manifest"),
        manifest_v2=v2.get("manifest"),
    )
    return {
        "mode": "UPLOADED_MODEL_REGRESSION",
        "status": "COMPLETED",
        "models": {
            "v1": {"asset_id": request.model_v1_id, "investigation_id": v1["investigation_id"], "accuracy": v1["baseline"]["accuracy"]},
            "v2": {"asset_id": request.model_v2_id, "investigation_id": v2["investigation_id"], "accuracy": v2["baseline"]["accuracy"]},
        },
        **comparison,
    }
