from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from modelxray.investigation.controller import ExperimentObservation
from modelxray.investigation.failure_clustering import cluster_validated_failures
from modelxray.investigation.schemas import Experiment


@dataclass(frozen=True)
class Hyp:
    eid: str
    mechanism: str
    lo: float
    hi: float
    evidence: float


def _obs(h: Hyp) -> ExperimentObservation:
    gap = 0.18 + 0.02 * (h.evidence % 3)
    return ExperimentObservation(
        experiment_id=h.eid,
        condition=f"{h.mechanism}:{h.lo:.3f}-{h.hi:.3f}",
        kind="oblique_band",
        support=max(0.01, h.hi - h.lo),
        subgroup_accuracy=0.7,
        baseline_accuracy=0.9,
        gap=gap,
        p_value=1e-8,
        severity="HIGH",
        validated=True,
        validation_gap=gap - 0.01,
        adjusted_p_value=1e-7,
        effect_size=gap,
        effect_ci_low=gap * 0.5,
        effect_ci_high=gap * 1.5,
        holdout_effect_ci_low=gap * 0.4,
        holdout_effect_ci_high=gap * 1.4,
        holdout_ci_credible=True,
        reproducible=True,
        evidence_score=h.evidence,
    )


def _make_case(seed: int, n_per_mechanism: int = 8) -> tuple[pd.DataFrame, list[Hyp]]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 2001)})
    specs = {
        "M1": (0.10, 0.34),
        "M2": (0.44, 0.66),
        "M3": (0.78, 0.94),
    }
    hyps: list[Hyp] = []
    i = 0
    for mechanism, (lo, hi) in specs.items():
        width = hi - lo
        for _ in range(n_per_mechanism):
            shrink = rng.uniform(0.02, 0.14) * width
            start = lo + rng.uniform(0.0, max(width - shrink, 1e-6))
            end = min(start + width - shrink, hi)
            # Force enough overlap for same-mechanism variants while keeping
            # different mechanisms clearly disjoint.
            if end - start < 0.7 * width:
                start = lo
                end = hi - rng.uniform(0.0, 0.08) * width
            hyps.append(Hyp(
                eid=f"S{seed}-{mechanism}-{i}",
                mechanism=mechanism,
                lo=float(start),
                hi=float(end),
                evidence=float(rng.uniform(70, 98)),
            ))
            i += 1
    return X, hyps


def _exp(h: Hyp) -> Experiment:
    return Experiment(
        h.eid,
        "oblique_band",
        "x",
        ">",
        0.0,
        1.0,
        f"{h.mechanism}:{h.lo:.3f}-{h.hi:.3f}",
        oblique_terms=(("x", 1.0),),
        oblique_lower=h.lo,
        oblique_upper=h.hi,
    )


def _pair_metrics(hyps: list[Hyp], mapping: dict[str, str]) -> tuple[float, float]:
    tp = fp = fn = 0
    for i in range(len(hyps)):
        for j in range(i + 1, len(hyps)):
            same_truth = hyps[i].mechanism == hyps[j].mechanism
            same_pred = mapping[hyps[i].eid] == mapping[hyps[j].eid]
            if same_truth and same_pred:
                tp += 1
            elif (not same_truth) and same_pred:
                fp += 1
            elif same_truth and (not same_pred):
                fn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return precision, recall


def _make_adversarial_case(seed: int):
    # Two truly different mechanisms occupy highly overlapping regions.
    # Geometric clustering cannot know causality, so this case measures a known
    # ambiguity rather than treating overlap as ground-truth mechanism identity.
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 2001)})
    specs = [
        Hyp(f"A-{seed}-1", "M1", 0.10, 0.70, 95.0),
        Hyp(f"A-{seed}-2", "M1", 0.12, 0.68, 90.0),
        Hyp(f"A-{seed}-3", "M2", 0.14, 0.66, 88.0),
        Hyp(f"A-{seed}-4", "M2", 0.16, 0.64, 84.0),
    ]
    return X, specs


def run(seeds: int, out: Path) -> dict:
    rows = []
    for seed in range(100, 100 + seeds):
        X, hyps = _make_case(seed)
        candidates = {h.eid: _exp(h) for h in hyps}
        observations = [_obs(h) for h in hyps]
        clusters, mapping = cluster_validated_failures(observations, candidates, X, jaccard_threshold=0.60)
        precision, recall = _pair_metrics(hyps, mapping)
        rows.append({
            "seed": seed,
            "clusters": len(clusters),
            "precision": precision,
            "recall": recall,
        })

    ambiguous_rows = []
    for seed in range(100, 100 + seeds):
        X, hyps = _make_adversarial_case(seed)
        candidates = {h.eid: _exp(h) for h in hyps}
        observations = [_obs(h) for h in hyps]
        clusters, mapping = cluster_validated_failures(observations, candidates, X, jaccard_threshold=0.60)
        precision, recall = _pair_metrics(hyps, mapping)
        ambiguous_rows.append({
            "seed": seed,
            "clusters": len(clusters),
            "precision": precision,
            "recall": recall,
        })

    result = {
        "seeds": seeds,
        "mean_clusters": float(np.mean([r["clusters"] for r in rows])),
        "mean_pair_precision": float(np.mean([r["precision"] for r in rows])),
        "mean_pair_recall": float(np.mean([r["recall"] for r in rows])),
        "adversarial_ambiguous": {
            "mean_clusters": float(np.mean([r["clusters"] for r in ambiguous_rows])),
            "mean_pair_precision": float(np.mean([r["precision"] for r in ambiguous_rows])),
            "mean_pair_recall": float(np.mean([r["recall"] for r in ambiguous_rows])),
            "interpretation": "Expected ambiguity: geometric overlap alone cannot distinguish different causal mechanisms when regions overlap strongly.",
        },
        "rows": rows,
        "ambiguous_rows": ambiguous_rows,
        "interpretation": "Pairwise clustering agreement against seeded mechanism IDs; not a causal claim.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "failure_cluster_benchmark.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("benchmark/cluster"))
    args = parser.parse_args()
    result = run(args.seeds, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
