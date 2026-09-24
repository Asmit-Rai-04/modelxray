from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Experiment:
    """One candidate hypothesis.

    Kinds:
      - "subgroup_threshold": single-feature threshold region.
      - "interaction": pairwise conjunction `feature_a cond_a AND feature_b cond_b`.
      - "oblique_band": weighted linear projection inside a numeric interval.
      - "categorical_subgroup": rows matching one categorical value.
      - "prototype_region": local neighborhood around an unsupervised prototype.

    Sign/condition convention: the region selects rows where the model is
    suspected to underperform. `gap` is reported as
    `baseline_score - subgroup_score` (positive = subgroup is worse).
    """

    id: str
    kind: str
    feature: str
    operator: str  # "<=" or ">"
    threshold: float
    priority: float
    rationale: str
    feature_b: str | None = None
    operator_b: str | None = None
    threshold_b: float | None = None
    oblique_terms: tuple[tuple[str, float], ...] = ()
    oblique_intercept: float = 0.0
    oblique_lower: float | None = None
    oblique_upper: float | None = None
    category: object | None = None
    cluster_centroid: tuple[float, ...] = ()
    cluster_scales: tuple[float, ...] = ()
    cluster_radius: float | None = None
    cluster_k: int | None = None
    cluster_id: int | None = None

    @property
    def features(self) -> tuple[str, ...]:
        if self.kind == "oblique_band" and self.oblique_terms:
            return tuple(feature for feature, _ in self.oblique_terms)
        return (self.feature, self.feature_b) if self.feature_b else (self.feature,)

    def describe(self) -> str:
        if self.kind == "categorical_subgroup" and self.category is not None:
            return f"{self.feature} == {self.category!r}"
        if self.kind == "prototype_region" and self.cluster_centroid:
            terms = ", ".join(f"{feature}={value:.2f}" for feature, value in zip(self.features, self.cluster_centroid))
            return f"prototype neighborhood [{terms}] radius≤{float(self.cluster_radius):.3f}"
        if self.kind == "oblique_band" and self.oblique_terms:
            terms = " ".join(
                f"{weight:+.3f}*{feature}" for feature, weight in self.oblique_terms
            ).lstrip("+")
            return (
                f"{terms} ∈ [{float(self.oblique_lower):.4g}, "
                f"{float(self.oblique_upper):.4g}]"
            )
        base = f"{self.feature} {self.operator} {self.threshold:.4g}"
        if self.kind == "interaction" and self.feature_b is not None:
            return f"{base} & {self.feature_b} {self.operator_b} {self.threshold_b:.4g}"
        return base


def numeric_mask(values: np.ndarray, operator: str, threshold: float) -> np.ndarray:
    finite = np.isfinite(values)
    if operator == "<=":
        return finite & (values <= threshold)
    if operator == ">":
        return finite & (values > threshold)
    raise ValueError(f"Unsupported operator {operator!r}")


def mask_for_experiment(experiment: Experiment, X: pd.DataFrame) -> np.ndarray:
    """Region mask for an experiment; single shared implementation."""
    if experiment.kind == "categorical_subgroup" and experiment.category is not None:
        values = X[experiment.feature].to_numpy(dtype=object)
        return pd.notna(values) & (values == experiment.category)

    if experiment.kind == "prototype_region" and experiment.cluster_centroid:
        features = experiment.features
        if not experiment.cluster_scales or experiment.cluster_radius is None:
            return np.zeros(len(X), dtype=bool)
        distances = np.zeros(len(X), dtype=float)
        finite = np.ones(len(X), dtype=bool)
        for feature, center, scale in zip(features, experiment.cluster_centroid, experiment.cluster_scales):
            values = pd.to_numeric(X[feature], errors="coerce").to_numpy(dtype=float)
            finite &= np.isfinite(values)
            safe_scale = max(float(scale), 1e-9)
            distances += ((values - float(center)) / safe_scale) ** 2
        return finite & (np.sqrt(distances) <= float(experiment.cluster_radius))

    if experiment.kind == "oblique_band" and experiment.oblique_terms:
        score = np.full(len(X), float(experiment.oblique_intercept), dtype=float)
        finite = np.ones(len(X), dtype=bool)
        for feature, weight in experiment.oblique_terms:
            values = pd.to_numeric(X[feature], errors="coerce").to_numpy(dtype=float)
            finite &= np.isfinite(values)
            score += float(weight) * values
        lower = float(experiment.oblique_lower)
        upper = float(experiment.oblique_upper)
        return finite & (score >= lower) & (score <= upper)

    values = pd.to_numeric(X[experiment.feature], errors="coerce").to_numpy(dtype=float)
    mask = numeric_mask(values, experiment.operator, experiment.threshold)
    if experiment.kind == "interaction" and experiment.feature_b is not None:
        values_b = pd.to_numeric(X[experiment.feature_b], errors="coerce").to_numpy(dtype=float)
        mask = mask & numeric_mask(values_b, experiment.operator_b or ">", float(experiment.threshold_b or 0.0))
    return mask
