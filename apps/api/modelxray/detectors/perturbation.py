from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from modelxray.core.adapters import ClassifierAdapter


@dataclass(frozen=True)
class InstabilityFinding:
    """A local boundary-sensitivity observation (NOT a failure by itself)."""

    detector: str
    feature: str
    row_index: int
    original_value: float
    perturbed_value: float
    original_prediction: str
    perturbed_prediction: str
    relative_change: float


def discover_local_instability(
    adapter: ClassifierAdapter,
    X: pd.DataFrame,
    max_rows: int = 250,
) -> list[InstabilityFinding]:
    """Probe small local changes around numeric features and report label flips.

    These findings describe ordinary decision-boundary sensitivity. They are
    reported as descriptive context and never enter the failure atlas.
    """
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    if not numeric:
        return []

    findings: list[InstabilityFinding] = []
    working = X.head(max_rows).copy()
    baseline_pred = adapter.predict(working)

    for feature in numeric:
        values = working[feature].to_numpy(dtype=float)
        scale = float(np.nanstd(values)) or 1.0
        step = max(scale * 0.005, 1e-9)
        perturbed = working.copy()
        perturbed[feature] = values + step
        perturbed_pred = adapter.predict(perturbed)
        flips = np.where(baseline_pred != perturbed_pred)[0]
        for idx in flips[:10]:
            original = float(values[idx])
            change = abs(step) / max(abs(original), 1e-9)
            findings.append(
                InstabilityFinding(
                    detector="perturbation",
                    feature=feature,
                    row_index=int(working.index[idx]),
                    original_value=original,
                    perturbed_value=float(original + step),
                    original_prediction=str(baseline_pred[idx]),
                    perturbed_prediction=str(perturbed_pred[idx]),
                    relative_change=float(change),
                )
            )

    findings.sort(key=lambda f: f.relative_change)
    return findings[:20]
