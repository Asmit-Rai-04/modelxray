from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from modelxray.investigation.schemas import Experiment, mask_for_experiment


@dataclass(frozen=True)
class FailureCluster:
    """A consolidated group of overlapping validated failure hypotheses.

    A cluster is NOT a claim that all hypotheses have the same causal mechanism.
    It means their validated regions overlap strongly enough on the evaluation
    frame that presenting them as separate failures would likely double-count
    the same behavioral region.
    """

    cluster_id: str
    representative_experiment_id: str
    supporting_experiment_ids: tuple[str, ...]
    condition: str
    severity: str
    hypothesis_count: int
    support: float
    gap: float
    validation_gap: float
    best_adjusted_p_value: float
    best_evidence_score: float
    mean_evidence_score: float
    region_jaccard_to_representative: float
    strategy_kinds: tuple[str, ...]
    strategy_diversity: int


_SEVERITY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def _union_find(n: int):
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return parent, find, union


def cluster_validated_failures(
    observations: list[Any],
    candidate_by_id: dict[str, Experiment],
    X: pd.DataFrame,
    *,
    jaccard_threshold: float = 0.60,
) -> tuple[list[FailureCluster], dict[str, str]]:
    """Consolidate strongly-overlapping validated hypotheses.

    Only observations already marked reproducible are considered. Region overlap
    is measured on the same evaluation frame used for holdout validation. The
    cluster relationship is deliberately geometric rather than causal.

    Returns (clusters, experiment_id -> cluster_id mapping).
    """
    eligible = [o for o in observations if bool(o.reproducible)]
    if not eligible:
        return [], {}

    masks: list[np.ndarray] = []
    kept: list[Any] = []
    for obs in eligible:
        exp = candidate_by_id.get(obs.experiment_id)
        if exp is None:
            continue
        mask = mask_for_experiment(exp, X)
        if mask.any():
            masks.append(mask)
            kept.append(obs)
    if not kept:
        return [], {}

    # Use complete-linkage style grouping rather than transitive connected components.
    # Connected components can create false merges through chaining (A overlaps B,
    # B overlaps C, but A and C are disjoint). A cluster is therefore accepted only
    # when every member overlaps every existing member at or above the threshold.
    similarity = np.eye(len(kept), dtype=float)
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            similarity[i, j] = similarity[j, i] = _jaccard(masks[i], masks[j])

    order = sorted(
        range(len(kept)),
        key=lambda i: (float(kept[i].evidence_score), float(kept[i].gap * kept[i].support)),
        reverse=True,
    )
    groups_list: list[list[int]] = []
    for idx in order:
        placed = False
        for group in groups_list:
            if all(similarity[idx, member] >= jaccard_threshold for member in group):
                group.append(idx)
                placed = True
                break
        if not placed:
            groups_list.append([idx])

    groups: dict[int, list[int]] = {i: group for i, group in enumerate(groups_list)}

    # Sort clusters by strongest evidence so IDs remain deterministic within a run.
    ordered_groups = sorted(
        groups.values(),
        key=lambda inds: max(float(kept[i].evidence_score) for i in inds),
        reverse=True,
    )

    clusters: list[FailureCluster] = []
    experiment_to_cluster: dict[str, str] = {}
    for cluster_num, inds in enumerate(ordered_groups, start=1):
        members = [kept[i] for i in inds]
        rep = max(members, key=lambda o: (float(o.evidence_score), float(o.gap * o.support)))
        rep_idx = kept.index(rep)
        cluster_id = f"FC-{cluster_num:03d}"
        representative_mask = masks[rep_idx]
        support_obs = [float(o.support) for o in members]
        severity_value = max((str(o.severity) for o in members), key=lambda s: _SEVERITY_RANK.get(s, 0))
        best_adj = min(float(o.adjusted_p_value) for o in members)
        best_score = max(float(o.evidence_score) for o in members)
        mean_score = float(np.mean([float(o.evidence_score) for o in members]))
        overlap_values = [_jaccard(representative_mask, masks[i]) for i in inds]
        strategy_kinds = tuple(sorted({str(o.kind) for o in members}))

        cluster = FailureCluster(
            cluster_id=cluster_id,
            representative_experiment_id=rep.experiment_id,
            supporting_experiment_ids=tuple(o.experiment_id for o in members),
            condition=str(rep.condition),
            severity=severity_value,
            hypothesis_count=len(members),
            support=float(rep.support),
            gap=float(rep.gap),
            validation_gap=float(rep.validation_gap),
            best_adjusted_p_value=best_adj,
            best_evidence_score=best_score,
            mean_evidence_score=mean_score,
            region_jaccard_to_representative=float(np.mean(overlap_values)),
            strategy_kinds=strategy_kinds,
            strategy_diversity=len(strategy_kinds),
        )
        clusters.append(cluster)
        for obs in members:
            experiment_to_cluster[obs.experiment_id] = cluster_id

    return clusters, experiment_to_cluster


def clusters_as_dict(clusters: list[FailureCluster]) -> list[dict[str, Any]]:
    return [asdict(cluster) for cluster in clusters]
