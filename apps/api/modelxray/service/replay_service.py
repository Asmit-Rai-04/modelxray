from __future__ import annotations

from typing import Any

from modelxray.core.certificate import environment_manifest, stable_hash
from modelxray.core.ingestion import AssetError, AssetStore
from modelxray.execution.sandbox import run_isolated_investigation
from modelxray.storage import EvidenceStore


def _normalize_observation(obs: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "family",
        "condition",
        "metric",
        "gap",
        "support",
        "p_value",
        "adjusted_p_value",
        "validation_gap",
        "reproducible",
        "severity",
        "evidence_score",
    ]
    result: dict[str, Any] = {}
    for key in keys:
        value = obs.get(key)
        if isinstance(value, float):
            value = round(value, 10)
        result[key] = value
    return result



def _run_signature(worker: dict[str, Any]) -> str:
    active = worker.get("active_investigation", {})
    observations = [_normalize_observation(x) for x in active.get("observations", [])]
    observations.sort(key=lambda x: (str(x.get("family")), str(x.get("condition"))))
    clusters = []
    for c in active.get("failure_clusters", []):
        clusters.append({
            "condition": c.get("condition"),
            "supporting_conditions": sorted(
                str(x.get("condition", "")) for x in c.get("supporting_hypotheses", [])
            ),
            "hypothesis_count": c.get("hypothesis_count"),
        })
    clusters.sort(key=lambda x: str(x.get("condition")))
    baseline = worker.get("baseline") or {}
    baseline = {k: (round(v, 10) if isinstance(v, float) else v) for k, v in baseline.items()}
    return stable_hash({"baseline": baseline, "observations": observations, "clusters": clusters})


def replay_investigation(
    asset_store: AssetStore,
    evidence_store: EvidenceStore,
    investigation_id: str,
) -> dict[str, Any]:
    original = evidence_store.get_investigation(investigation_id)
    if original is None:
        raise AssetError(f"Investigation not found: {investigation_id}")

    cert = original.get("run_certificate") or {}
    manifest = original.get("manifest") or {}
    dataset = original.get("dataset") or {}
    model = original.get("model") or {}
    model_id = manifest.get("model_asset_id") or model.get("source_model_id")
    dataset_id = manifest.get("dataset_asset_id") or dataset.get("source_dataset_id")
    target_column = manifest.get("target_column") or dataset.get("target_column")

    if not model_id or not dataset_id or not target_column:
        return {
            "status": "NOT_REPLAYABLE",
            "investigation_id": investigation_id,
            "reason": "This investigation does not retain the model asset, dataset asset, and target-column references required for replay.",
        }

    model_asset = asset_store.get(model_id)
    dataset_asset = asset_store.get(dataset_id)
    if model_asset is None or dataset_asset is None:
        return {
            "status": "NOT_REPLAYABLE",
            "investigation_id": investigation_id,
            "reason": "Referenced model or dataset asset is no longer available.",
        }

    # Hash checks are done before executing a replay so the report can distinguish
    # environmental/input drift from actual behavioral drift.
    from modelxray.execution.worker import _sha256_file

    current_model_hash = _sha256_file(model_asset.path)
    current_dataset_hash = _sha256_file(dataset_asset.path)
    expected_model_hash = cert.get("model_hash") or manifest.get("model_hash")
    expected_dataset_hash = cert.get("dataset_hash") or manifest.get("dataset_hash")

    checks = {
        "model_hash": {"expected": expected_model_hash, "actual": current_model_hash, "match": expected_model_hash == current_model_hash},
        "dataset_hash": {"expected": expected_dataset_hash, "actual": current_dataset_hash, "match": expected_dataset_hash == current_dataset_hash},
        "metric": {"expected": cert.get("metric"), "actual": original.get("metric", "accuracy"), "match": cert.get("metric") == original.get("metric", "accuracy")},
        "budget": {"expected": cert.get("budget"), "actual": int((manifest or {}).get("budget", cert.get("budget", 24))), "match": int(cert.get("budget", -1)) == int((manifest or {}).get("budget", cert.get("budget", 24)))},
        "random_state": {"expected": cert.get("random_state"), "actual": int((manifest or {}).get("random_state", cert.get("random_state", 42))), "match": int(cert.get("random_state", -1)) == int((manifest or {}).get("random_state", cert.get("random_state", 42)))},
        "environment": {"expected": cert.get("environment", {}), "actual": environment_manifest(), "match": cert.get("environment", {}) == environment_manifest()},
    }

    if not checks["model_hash"]["match"] or not checks["dataset_hash"]["match"]:
        return {
            "status": "MISMATCH",
            "investigation_id": investigation_id,
            "checks": checks,
            "reason": "Model or dataset artifact hash differs from the recorded run certificate.",
        }

    replay_worker = run_isolated_investigation(
        model_path=model_asset.path,
        dataset_path=dataset_asset.path,
        target_column=target_column,
        budget=int(cert.get("budget", 24)),
        random_state=int(cert.get("random_state", 42)),
        metric=cert.get("metric", "accuracy"),
        feature_manifest=manifest.get("feature_manifest"),
    )

    original_observations = original.get("active_investigation", {}).get("observations", [])
    original_obs = [_normalize_observation(x) for x in original_observations]
    original_obs.sort(key=lambda x: (str(x.get("family")), str(x.get("condition"))))
    original_clusters = []
    for c in original.get("active_investigation", {}).get("failure_clusters", []):
        original_clusters.append({
            "condition": c.get("condition"),
            "supporting_conditions": sorted(
                str(x.get("condition", "")) for x in c.get("supporting_hypotheses", [])
            ),
            "hypothesis_count": c.get("hypothesis_count"),
        })
    original_clusters.sort(key=lambda x: str(x.get("condition")))
    original_baseline = original.get("baseline") or {}
    original_baseline = {k: (round(v, 10) if isinstance(v, float) else v) for k, v in original_baseline.items()}
    original_signature = stable_hash({"baseline": original_baseline, "observations": original_obs, "clusters": original_clusters})
    replay_signature = _run_signature(replay_worker)
    checks["behavior_signature"] = {
        "expected": original_signature,
        "actual": replay_signature,
        "match": original_signature == replay_signature,
    }

    status = "VERIFIED" if all(item["match"] for item in checks.values()) else "MISMATCH"
    return {
        "status": status,
        "investigation_id": investigation_id,
        "checks": checks,
        "replay": {
            "model_hash": current_model_hash,
            "dataset_hash": current_dataset_hash,
            "experiment_count": replay_worker.get("active_investigation", {}).get("experiments_executed", 0),
            "failure_cluster_count": len(replay_worker.get("active_investigation", {}).get("failure_clusters", [])),
            "counterexample_count": len(replay_worker.get("counterexamples", [])),
        },
    }
