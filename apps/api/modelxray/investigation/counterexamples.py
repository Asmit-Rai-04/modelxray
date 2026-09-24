from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from modelxray.core.adapters import ClassifierAdapter
from modelxray.investigation.schemas import Experiment, mask_for_experiment


@dataclass(frozen=True)
class Counterexample:
    """Boundary-sensitivity evidence for a *validated* failure region.

    Semantics (v1.0.1): a minimal-ish one-feature perturbation that flips the
    model's prediction near a validated failure region. This is NOT proof the
    model is broken on its own — decision boundaries always flip under some
    perturbation. It is attached only to statistically validated failure
    regions, where it localizes the boundary sensitivity.
    """

    source_experiment_id: str
    source_condition: str
    source_row: int
    frame: str
    original_prediction: Any
    counterfactual_prediction: Any
    changed_feature: str
    original_value: float
    counterfactual_value: float
    absolute_change: float
    relative_change: float
    search_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prediction(adapter: ClassifierAdapter, row: pd.DataFrame, feature: str, value: float) -> Any:
    candidate = row.copy()
    candidate.loc[candidate.index[0], feature] = value
    return adapter.predict(candidate)[0]


def _batched_refinement_predictions(
    adapter: ClassifierAdapter,
    row: pd.DataFrame,
    active: list[dict[str, Any]],
) -> list[Any]:
    """Predict one binary-search midpoint for every active feature in one call.

    This turns N independent per-feature binary searches into N initial grid
    calls + O(iterations) batched calls, which materially reduces model-call
    overhead while preserving the exact one-dimensional search semantics.
    """
    if not active:
        return []
    candidates = pd.concat([row] * len(active), ignore_index=True)
    for idx, item in enumerate(active):
        midpoint = (item["lo"] + item["hi"]) / 2.0
        candidates.loc[idx, item["feature"]] = midpoint
    return list(np.asarray(adapter.predict(candidates)))


def _batched_predictions(
    adapter: ClassifierAdapter,
    row: pd.DataFrame,
    feature: str,
    values: list[float],
) -> list[Any]:
    """Predict a grid of one-feature perturbations in one model call."""
    if not values:
        return []
    candidates = pd.concat([row] * len(values), ignore_index=True)
    candidates[feature] = np.asarray(values, dtype=float)
    predictions = adapter.predict(candidates)
    return list(np.asarray(predictions))


def _min_flip_upward(adapter, row, feature, original_value, boundary, original_prediction, iterations=16):
    lo, hi = original_value, boundary
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if _prediction(adapter, row, feature, mid) != original_prediction:
            hi = mid
        else:
            lo = mid
    return hi


def _min_flip_downward(adapter, row, feature, original_value, boundary, original_prediction, iterations=16):
    lo, hi = boundary, original_value
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if _prediction(adapter, row, feature, mid) != original_prediction:
            lo = mid
        else:
            hi = mid
    return lo


def find_minimal_counterexample(
    adapter: ClassifierAdapter,
    X: pd.DataFrame,
    experiment: Experiment,
    row_index: int,
    max_grid_points: int = 15,
    row_frame_name: str = "discovery",
) -> Counterexample | None:
    """Greedy approximate minimal single-feature flip.

    Grid probing remains per-feature, but all binary-refinement midpoints are
    evaluated in batches across features. The result is still the same
    approximate one-feature boundary search, with substantially fewer model
    invocations.
    """
    row = X.iloc[[row_index]].copy()
    numeric_columns = list(X.select_dtypes(include=np.number).columns)
    if not numeric_columns:
        return None

    original_prediction = adapter.predict(row)[0]
    best: Counterexample | None = None
    active: list[dict[str, Any]] = []

    for feature in numeric_columns:
        original_value = float(row.iloc[0][feature])
        series = pd.to_numeric(X[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        low, high = float(series.min()), float(series.max())
        if not low < high:
            continue

        grid = np.linspace(low, high, max_grid_points)
        grid_values = [float(value) for value in grid if not np.isclose(value, original_value)]
        grid_predictions = _batched_predictions(adapter, row, feature, grid_values)
        flip_points = [
            value for value, prediction in zip(grid_values, grid_predictions)
            if prediction != original_prediction
        ]
        if not flip_points:
            continue

        boundary = min(flip_points, key=lambda v: abs(v - original_value))
        if boundary > original_value:
            lo, hi = original_value, boundary
            direction = "up"
        else:
            lo, hi = boundary, original_value
            direction = "down"
        active.append({
            "feature": feature,
            "original_value": original_value,
            "lo": lo,
            "hi": hi,
            "direction": direction,
            "flip_value": boundary,
        })

    # Binary refinement across every feature in parallel.
    for _ in range(16):
        if not active:
            break
        predictions = _batched_refinement_predictions(adapter, row, active)
        next_active: list[dict[str, Any]] = []
        for item, prediction in zip(active, predictions):
            midpoint = (item["lo"] + item["hi"]) / 2.0
            if prediction != original_prediction:
                item["flip_value"] = midpoint
                item["hi"] = midpoint
            else:
                item["lo"] = midpoint
            next_active.append(item)
        active = next_active

    # Verify all refined endpoints in one batch; only boundary-rounding nudges
    # need individual calls.
    endpoint_rows: list[pd.DataFrame] = []
    endpoint_values: list[float] = []
    for item in active:
        value = float(item["flip_value"])
        candidate = row.copy()
        candidate.loc[candidate.index[0], item["feature"]] = value
        endpoint_rows.append(candidate)
        endpoint_values.append(value)
    endpoint_predictions = list(np.asarray(adapter.predict(pd.concat(endpoint_rows, ignore_index=True)))) if endpoint_rows else []

    verified: list[tuple[dict[str, Any], float, Any]] = []
    for item, value, counter_pred in zip(active, endpoint_values, endpoint_predictions):
        if counter_pred == original_prediction:
            # Numerical boundary effects can make the last midpoint land exactly
            # on the classifier's decision threshold. Move one representable
            # float toward the known flip side and verify again.
            direction = np.inf if item["direction"] == "up" else -np.inf
            nudged = float(np.nextafter(value, direction))
            counter_pred = _prediction(adapter, row, item["feature"], nudged)
            if counter_pred != original_prediction:
                value = nudged
        if counter_pred == original_prediction:
            continue
        verified.append((item, value, counter_pred))

    for item, value, counter_pred in verified:
        delta = abs(value - float(item["original_value"]))
        relative = delta / max(abs(float(item["original_value"])), 1e-9)
        original_clean = original_prediction.item() if hasattr(original_prediction, "item") else original_prediction
        counter_clean = counter_pred.item() if hasattr(counter_pred, "item") else counter_pred
        finding = Counterexample(
            source_experiment_id=experiment.id,
            source_condition=experiment.describe(),
            source_row=int(row_index),
            frame=row_frame_name,
            original_prediction=original_clean,
            counterfactual_prediction=counter_clean,
            changed_feature=item["feature"],
            original_value=float(item["original_value"]),
            counterfactual_value=value,
            absolute_change=float(delta),
            relative_change=float(relative),
            search_status="approx_minimal_flip",
        )
        if best is None or finding.absolute_change < best.absolute_change:
            best = finding

    return best


def discover_counterexamples(
    adapter: ClassifierAdapter,
    X: pd.DataFrame,
    experiments: list[Experiment],
    limit: int = 5,
    row_frame_name: str = "discovery",
) -> list[dict[str, Any]]:
    """Boundary-sensitivity probes for VALIDATED experiments only.

    The controller passes only validated (reproducible) experiments here, so a
    clean model with zero promoted failures produces zero counterexamples.
    `row_frame_name` labels the coordinate system of `source_row`.
    """
    findings: list[Counterexample] = []
    for experiment in experiments:
        mask = mask_for_experiment(experiment, X)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            continue
        values = pd.to_numeric(X[experiment.feature], errors="coerce").to_numpy(dtype=float)
        distances = np.abs(values[indices] - experiment.threshold)
        row_index = int(indices[int(np.argmin(distances))])
        finding = find_minimal_counterexample(
            adapter, X, experiment, row_index, row_frame_name=row_frame_name
        )
        if finding is not None:
            findings.append(finding)
        if len(findings) >= limit:
            break

    findings.sort(key=lambda item: item.absolute_change)
    return [item.to_dict() for item in findings]
