from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class FailureSnapshot:
    experiment_id: str
    condition: str
    kind: str
    severity: str
    gap: float
    support: float
    adjusted_p_value: float
    validation_gap: float
    evidence_score: float
    reproducible: bool
    failure_cluster_id: str | None = None

    @classmethod
    def from_observation(cls, obs: dict[str, Any]) -> "FailureSnapshot":
        return cls(
            experiment_id=str(obs["experiment_id"]),
            condition=str(obs["condition"]),
            kind=str(obs.get("kind", "subgroup_threshold")),
            severity=str(obs["severity"]),
            gap=float(obs["gap"]),
            support=float(obs["support"]),
            adjusted_p_value=float(obs.get("adjusted_p_value", 1.0)),
            validation_gap=float(obs.get("validation_gap", 0.0)),
            evidence_score=float(obs.get("evidence_score", 0.0)),
            reproducible=bool(obs.get("reproducible", False)),
            failure_cluster_id=obs.get("failure_cluster_id"),
        )


@dataclass(frozen=True)
class FailureDelta:
    experiment_id_v1: str | None
    experiment_id_v2: str | None
    condition_v1: str | None
    condition_v2: str | None
    match_type: str  # "exact" or "region_overlap"
    status: str  # "PERSISTENT", "FIXED", "NEW"
    severity_v1: str | None
    severity_v2: str | None
    gap_v1: float | None
    gap_v2: float | None
    evidence_score_v1: float | None
    evidence_score_v2: float | None
    evidence_delta: float | None
    validation_gap_v1: float | None
    validation_gap_v2: float | None
    failure_cluster_id_v1: str | None = None
    failure_cluster_id_v2: str | None = None


def _condition_key(condition: str) -> tuple:
    """Parse a condition string into a comparable region key.

    `feature op threshold` (or conjunction) → (feature, operator, rounded
    threshold-bucket). Threshold buckets use 0.05-wide bins so a threshold that
    drifted by a quantile bin still matches; this addresses the audit finding
    that exact-ID matching flips PERSISTENT into FIXED+NEW on small drift.
    """
    def _bucket(value: str) -> tuple[str, str, int]:
        parts = value.strip().rsplit(" ", 2)
        if len(parts) != 3:
            return (value.strip(), "", -1)
        feature, op, threshold = parts
        try:
            bucket = int(float(threshold) / 0.05)
        except ValueError:
            bucket = -1
        return (feature, op, bucket)

    return tuple(_bucket(part) for part in condition.split(" & "))


def _snapshot_maps(observations: Iterable[dict[str, Any]]) -> tuple[dict[str, FailureSnapshot], dict[tuple, FailureSnapshot]]:
    by_id: dict[str, FailureSnapshot] = {}
    by_region: dict[tuple, FailureSnapshot] = {}
    for obs in observations:
        snapshot = FailureSnapshot.from_observation(obs)
        if snapshot.reproducible:
            by_id[snapshot.experiment_id] = snapshot
            by_region[_condition_key(snapshot.condition)] = snapshot
    return by_id, by_region


def compare_failure_snapshots(
    v1_observations: Iterable[dict[str, Any]],
    v2_observations: Iterable[dict[str, Any]],
) -> list[FailureDelta]:
    """Compare reproducible failures between two model versions.

    Matching strategy (v1.0.1): exact experiment IDs first; then a
    region-overlap pass (feature + operator + 0.05-wide threshold bucket) for
    failures whose IDs differ due to threshold drift. Each delta reports its
    match_type so consumers know how confident the pairing is.
    """
    v1_by_id, v1_by_region = _snapshot_maps(v1_observations)
    v2_by_id, v2_by_region = _snapshot_maps(v2_observations)

    deltas: list[FailureDelta] = []
    matched_v1: set[str] = set()
    matched_v2: set[str] = set()

    def _delta(left: FailureSnapshot | None, right: FailureSnapshot | None, match_type: str) -> FailureDelta:
        if left and right:
            status = "PERSISTENT"
        elif left and not right:
            status = "FIXED"
        else:
            status = "NEW"
        return FailureDelta(
            experiment_id_v1=left.experiment_id if left else None,
            experiment_id_v2=right.experiment_id if right else None,
            failure_cluster_id_v1=getattr(left, "failure_cluster_id", None) if left else None,
            failure_cluster_id_v2=getattr(right, "failure_cluster_id", None) if right else None,
            condition_v1=left.condition if left else None,
            condition_v2=right.condition if right else None,
            match_type=match_type,
            status=status,
            severity_v1=left.severity if left else None,
            severity_v2=right.severity if right else None,
            gap_v1=left.gap if left else None,
            gap_v2=right.gap if right else None,
            evidence_score_v1=left.evidence_score if left else None,
            evidence_score_v2=right.evidence_score if right else None,
            evidence_delta=(right.evidence_score - left.evidence_score) if left and right else None,
            validation_gap_v1=left.validation_gap if left else None,
            validation_gap_v2=right.validation_gap if right else None,
        )


    # Cluster-aware pass: compare consolidated behavioral failures before individual
    # hypothesis IDs. This makes regression resilient to new hypothesis IDs and
    # overlapping descriptions that belong to the same Failure Atlas cluster.
    def _cluster_groups(by_id: dict[str, FailureSnapshot]) -> dict[str, list[FailureSnapshot]]:
        groups: dict[str, list[FailureSnapshot]] = {}
        for snap in by_id.values():
            key = snap.failure_cluster_id
            if key:
                groups.setdefault(key, []).append(snap)
        return groups

    groups_v1 = _cluster_groups(v1_by_id)
    groups_v2 = _cluster_groups(v2_by_id)

    def _cluster_similarity(left: list[FailureSnapshot], right: list[FailureSnapshot]) -> float:
        left_keys = {_condition_key(s.condition) for s in left}
        right_keys = {_condition_key(s.condition) for s in right}
        if not left_keys or not right_keys:
            return 0.0
        return len(left_keys & right_keys) / len(left_keys | right_keys)

    cluster_pairs: list[tuple[float, str, str]] = []
    for c1, g1 in groups_v1.items():
        for c2, g2 in groups_v2.items():
            sim = _cluster_similarity(g1, g2)
            if sim >= 0.50:
                cluster_pairs.append((sim, c1, c2))
    cluster_pairs.sort(reverse=True)
    matched_clusters_v1: set[str] = set()
    matched_clusters_v2: set[str] = set()
    for sim, c1, c2 in cluster_pairs:
        if c1 in matched_clusters_v1 or c2 in matched_clusters_v2:
            continue
        left = max(groups_v1[c1], key=lambda s: s.evidence_score)
        right = max(groups_v2[c2], key=lambda s: s.evidence_score)
        deltas.append(_delta(left, right, "cluster_overlap"))
        matched_clusters_v1.add(c1)
        matched_clusters_v2.add(c2)
        matched_v1.update(s.experiment_id for s in groups_v1[c1])
        matched_v2.update(s.experiment_id for s in groups_v2[c2])

    # Pass 1: exact experiment-ID matches.
    for experiment_id in sorted(set(v1_by_id) & set(v2_by_id)):
        deltas.append(_delta(v1_by_id[experiment_id], v2_by_id[experiment_id], "exact"))
        matched_v1.add(experiment_id)
        matched_v2.add(experiment_id)

    # Pass 2: region-overlap matches for unmatched failures (threshold drift).
    for left in sorted(v1_by_id.values(), key=lambda s: s.experiment_id):
        if left.experiment_id in matched_v1:
            continue
        key = _condition_key(left.condition)
        right = v2_by_region.get(key)
        if right is not None and right.experiment_id not in matched_v2:
            deltas.append(_delta(left, right, "region_overlap"))
            matched_v1.add(left.experiment_id)
            matched_v2.add(right.experiment_id)

    for left in sorted(v1_by_id.values(), key=lambda s: s.experiment_id):
        if left.experiment_id not in matched_v1:
            deltas.append(_delta(left, None, "exact"))
    for right in sorted(v2_by_id.values(), key=lambda s: s.experiment_id):
        if right.experiment_id not in matched_v2:
            deltas.append(_delta(None, right, "exact"))

    order = {"NEW": 0, "PERSISTENT": 1, "FIXED": 2}
    deltas.sort(key=lambda x: (order[x.status], -(abs(x.evidence_delta or 0.0))))
    return deltas


def build_regression_summary(deltas: list[FailureDelta]) -> dict[str, Any]:
    fixed = [d for d in deltas if d.status == "FIXED"]
    persistent = [d for d in deltas if d.status == "PERSISTENT"]
    new = [d for d in deltas if d.status == "NEW"]
    before = len(fixed) + len(persistent)
    after = len(persistent) + len(new)
    net_change = after - before
    return {
        "fixed": len(fixed),
        "persistent": len(persistent),
        "new": len(new),
        "failures_v1": before,
        "failures_v2": after,
        "net_failure_change": net_change,
        "failure_reduction_ratio": (len(fixed) / before) if before else 0.0,
        "deltas": [asdict(d) for d in deltas],
        "comparison_basis": "failure_clusters_then_hypotheses",
    }


def check_comparability(manifest_v1: dict[str, Any] | None, manifest_v2: dict[str, Any] | None) -> dict[str, Any]:
    """Verify two run manifests are comparable for regression purposes.

    Hard requirements: same dataset hash, same feature manifest, same metric.
    Soft warnings (comparison still allowed, flagged): different budget,
    random state, library versions.
    """
    warnings: list[str] = []
    if manifest_v1 is None or manifest_v2 is None:
        return {"comparable": False, "reason": "missing manifest (legacy investigation)", "warnings": []}
    hard_fields = ("dataset_hash", "feature_manifest", "metric")
    for field in hard_fields:
        if manifest_v1.get(field) != manifest_v2.get(field):
            return {
                "comparable": False,
                "reason": f"manifest mismatch on '{field}': {manifest_v1.get(field)!r} vs {manifest_v2.get(field)!r}",
                "warnings": warnings,
            }
    for field in ("budget", "random_state", "sklearn_version", "python_version", "project_version"):
        if manifest_v1.get(field) != manifest_v2.get(field):
            warnings.append(f"{field} differs: {manifest_v1.get(field)!r} (v1) vs {manifest_v2.get(field)!r} (v2)")
    return {"comparable": True, "reason": None, "warnings": warnings}
