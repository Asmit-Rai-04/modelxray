from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from modelxray.core.adapters import SklearnClassifierAdapter
from modelxray.investigation.controller import run_active_investigation
from modelxray.investigation.regression import compare_failure_snapshots, build_regression_summary


@dataclass(frozen=True)
class Transition:
    name: str
    regions_v1: tuple[tuple[float, float], ...]
    regions_v2: tuple[tuple[float, float], ...]
    expected_fixed: int
    expected_persistent: int
    expected_new: int


TRANSITIONS = (
    Transition("persistent", ((0.80, 2.20),), ((0.80, 2.20),), 0, 1, 0),
    Transition("fixed", ((0.80, 2.20),), tuple(), 1, 0, 0),
    Transition("new", tuple(), ((0.80, 2.20),), 0, 0, 1),
    Transition("replacement", ((0.80, 2.20),), ((-2.20, -0.80),), 1, 0, 1),
    Transition("drift", ((0.80, 2.20),), ((0.90, 2.30),), 0, 1, 0),
)


class FaultInjectedModel:
    """Black-box model wrapper that flips predictions in explicit regions."""

    def __init__(self, base: Any, regions: tuple[tuple[float, float], ...]):
        self.base = base
        self.regions = tuple(regions)

    def predict(self, X: Any) -> np.ndarray:
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        base_frame = frame.drop(columns=["fault_feature"], errors="ignore")
        pred = np.asarray(self.base.predict(base_frame)).copy()
        if not self.regions:
            return pred
        x = frame["fault_feature"].to_numpy(dtype=float)
        mask = np.zeros(len(frame), dtype=bool)
        for lo, hi in self.regions:
            mask |= (x >= lo) & (x <= hi)
        # binary fixture: flip 0 <-> 1 in the seeded region
        pred[mask] = 1 - pred[mask]
        return pred

    def predict_proba(self, X: Any) -> np.ndarray:
        pred = self.predict(X)
        return np.column_stack([1.0 - pred, pred])


def _make_data(seed: int) -> tuple[pd.DataFrame, pd.Series, RandomForestClassifier]:
    X, y = make_classification(
        n_samples=900,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        class_sep=2.2,
        random_state=seed,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(X.shape[1])])
    rng = np.random.default_rng(seed + 9000)
    X["fault_feature"] = rng.uniform(-3.0, 3.0, size=len(X))
    y = pd.Series(y, name="target")
    base = RandomForestClassifier(n_estimators=40, random_state=seed, n_jobs=1, max_depth=8)
    base.fit(X.iloc[:500].drop(columns=["fault_feature"]), y.iloc[:500])
    return X.iloc[500:].reset_index(drop=True), y.iloc[500:].reset_index(drop=True), base


def _run_one(model: Any, X: pd.DataFrame, y: pd.Series, budget: int) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    n = len(X)
    split = int(n * 0.6)
    adapter = SklearnClassifierAdapter(model)
    run = run_active_investigation(
        adapter,
        X.iloc[:split].reset_index(drop=True),
        y.iloc[:split].reset_index(drop=True),
        X.iloc[split:].reset_index(drop=True),
        y.iloc[split:].reset_index(drop=True),
        budget=budget,
        metric="accuracy",
    )
    return run.observations, X.iloc[split:].reset_index(drop=True)


def _expected_transition(name: str) -> tuple[int, int, int]:
    t = next(t for t in TRANSITIONS if t.name == name)
    return t.expected_fixed, t.expected_persistent, t.expected_new


def _status_counts(deltas: list[Any]) -> dict[str, int]:
    out = {"FIXED": 0, "PERSISTENT": 0, "NEW": 0}
    for d in deltas:
        out[d.status] += 1
    return out


def _condition_mask(condition: str | None, X: pd.DataFrame) -> np.ndarray:
    if not condition:
        return np.zeros(len(X), dtype=bool)
    mask = np.ones(len(X), dtype=bool)
    for part in condition.split(" & "):
        bits = part.strip().rsplit(" ", 2)
        if len(bits) != 3:
            return np.zeros(len(X), dtype=bool)
        feature, op, threshold = bits
        try:
            value = float(threshold)
        except ValueError:
            return np.zeros(len(X), dtype=bool)
        if feature not in X.columns:
            return np.zeros(len(X), dtype=bool)
        series = pd.to_numeric(X[feature], errors="coerce").to_numpy(dtype=float)
        if op == ">":
            part_mask = series > value
        elif op == "<=":
            part_mask = series <= value
        else:
            return np.zeros(len(X), dtype=bool)
        mask &= np.isfinite(series) & part_mask
    return mask


def _overlap_metrics(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    intersection = np.logical_and(predicted, target).sum()
    target_n = target.sum()
    predicted_n = predicted.sum()
    recall = float(intersection / target_n) if target_n else 0.0
    precision = float(intersection / predicted_n) if predicted_n else 0.0
    return recall, precision


def _target_transition_detected(
    deltas: list[Any],
    X_v1: pd.DataFrame,
    X_v2: pd.DataFrame,
    region_v1: tuple[float, float],
    region_v2: tuple[float, float],
    transition: str,
    threshold: float = 0.50,
) -> tuple[bool, dict[str, Any]]:
    target_masks_v1: np.ndarray | None = None
    target_masks_v2: np.ndarray | None = None
    if region_v1:
        s = X_v1["fault_feature"].to_numpy(dtype=float)
        target_masks_v1 = (s >= region_v1[0]) & (s <= region_v1[1])
    if region_v2:
        s = X_v2["fault_feature"].to_numpy(dtype=float)
        target_masks_v2 = (s >= region_v2[0]) & (s <= region_v2[1])

    diagnostics: dict[str, Any] = {"best_left_overlap": 0.0, "best_right_overlap": 0.0}
    for d in deltas:
        left_overlap = 0.0
        right_overlap = 0.0
        left_recall = left_precision = 0.0
        right_recall = right_precision = 0.0
        if target_masks_v1 is not None and d.condition_v1:
            left_recall, left_precision = _overlap_metrics(_condition_mask(d.condition_v1, X_v1), target_masks_v1)
        if target_masks_v2 is not None and d.condition_v2:
            right_recall, right_precision = _overlap_metrics(_condition_mask(d.condition_v2, X_v2), target_masks_v2)
        diagnostics["best_left_overlap"] = max(diagnostics["best_left_overlap"], left_recall)
        diagnostics["best_right_overlap"] = max(diagnostics["best_right_overlap"], right_recall)

        left_good = left_recall >= threshold and left_precision >= 0.10
        right_good = right_recall >= threshold and right_precision >= 0.10

        if transition == "persistent" and d.status == "PERSISTENT" and left_good and right_good:
            return True, {**diagnostics, "match": "persistent"}
        if transition == "fixed" and d.status == "FIXED" and left_good:
            return True, {**diagnostics, "match": "fixed"}
        if transition == "new" and d.status == "NEW" and right_good:
            return True, {**diagnostics, "match": "new"}
        if transition == "replacement":
            if d.status == "FIXED" and left_good:
                diagnostics["fixed_target"] = True
            if d.status == "NEW" and right_good:
                diagnostics["new_target"] = True
            if diagnostics.get("fixed_target") and diagnostics.get("new_target"):
                return True, {**diagnostics, "match": "replacement"}
        if transition == "drift" and d.status == "PERSISTENT" and left_good and right_good:
            return True, {**diagnostics, "match": "drift"}
    return False, diagnostics


def run(seeds: int, budget: int, out: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in range(100, 100 + seeds):
        X, y, base = _make_data(seed)
        for transition in TRANSITIONS:
            v1_model = FaultInjectedModel(base, transition.regions_v1)
            v2_model = FaultInjectedModel(base, transition.regions_v2)
            obs_v1, X_v1 = _run_one(v1_model, X, y, budget)
            obs_v2, X_v2 = _run_one(v2_model, X, y, budget)
            deltas = compare_failure_snapshots(obs_v1, obs_v2)
            counts = _status_counts(deltas)
            expected = {
                "FIXED": transition.expected_fixed,
                "PERSISTENT": transition.expected_persistent,
                "NEW": transition.expected_new,
            }
            # This benchmark checks whether the intended seeded transition is represented
            # in the regression output. Extra real failures are recorded, not suppressed.
            seeded_detected, seeded_diag = _target_transition_detected(
                deltas, X_v1, X_v2,
                transition.regions_v1[0] if transition.regions_v1 else tuple(),
                transition.regions_v2[0] if transition.regions_v2 else tuple(),
                transition.name,
            )
            row = {
                "seed": seed,
                "transition": transition.name,
                "counts": counts,
                "expected_transition": expected,
                "exact": counts == expected,
                "seeded_transition_detected": seeded_detected,
                "seeded_match_diagnostics": seeded_diag,
                "v1_validated": sum(bool(o.get("reproducible")) for o in obs_v1),
                "v2_validated": sum(bool(o.get("reproducible")) for o in obs_v2),
                "deltas": [
                    {
                        "status": d.status,
                        "match_type": d.match_type,
                        "condition_v1": d.condition_v1,
                        "condition_v2": d.condition_v2,
                    }
                    for d in deltas
                ],
            }
            rows.append(row)

    exact_rate = sum(r["exact"] for r in rows) / len(rows) if rows else 1.0
    seeded_rate = sum(r["seeded_transition_detected"] for r in rows) / len(rows) if rows else 1.0
    transition_rates = {}
    for name in [t.name for t in TRANSITIONS]:
        subset = [r for r in rows if r["transition"] == name]
        transition_rates[name] = (sum(r["seeded_transition_detected"] for r in subset) / len(subset)) if subset else 0.0
    result = {
        "seeds": seeds,
        "budget": budget,
        "transitions": [t.name for t in TRANSITIONS],
        "total_cases": len(rows),
        "exact_status_count_rate": exact_rate,
        "seeded_transition_detection_rate": seeded_rate,
        "seeded_transition_detection_by_case": transition_rates,
        "rows": rows,
        "interpretation": (
            "End-to-end benchmark runs actual black-box model wrappers through the real ModelXray "
            "investigation engine before cluster-aware regression. Exact status counts can be affected "
            "by unrelated real failures; this benchmark is a controlled validation, not a guarantee "
            "about arbitrary production models."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "e2e_regression_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("benchmark/e2e_regression"))
    args = parser.parse_args()
    result = run(args.seeds, args.budget, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
