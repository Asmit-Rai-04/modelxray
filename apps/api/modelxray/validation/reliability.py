from __future__ import annotations

from math import sqrt
from typing import Iterable

import numpy as np
from statsmodels.stats.multitest import multipletests

from modelxray.core.profiler import SUPPORTED_METRICS  # noqa: F401  (re-export convenience)

# ---------------------------------------------------------------------------
# Sign convention (single, project-wide):
#   gap = baseline_score - subgroup_score   (positive => subgroup performs worse)
# This applies to `gap`, `validation_gap`, `effect_size`, and CI bounds, where
# `effect_size` here is the *disadvantage* of the subgroup:
#   effect_size = complement_score - subgroup_score
# (Legacy `p1 - p2` subgroup-minus-complement convention was removed.)
# ---------------------------------------------------------------------------


def two_proportion_effect_and_ci(
    subgroup_correct: int,
    subgroup_total: int,
    complement_correct: int,
    complement_total: int,
) -> tuple[float, float, float]:
    """Disadvantage effect = complement_score - subgroup_score, with Wald CI.

    Returns (effect, ci_low, ci_high); positive effect means the subgroup is
    worse than the complement. The CI uses the unpooled Wald standard error.
    """
    if subgroup_total <= 0 or complement_total <= 0:
        return 0.0, 0.0, 0.0
    p_sub = subgroup_correct / subgroup_total
    p_comp = complement_correct / complement_total
    effect = p_comp - p_sub
    z = 1.96
    se = sqrt(max(0.0, p_sub * (1 - p_sub) / subgroup_total + p_comp * (1 - p_comp) / complement_total))
    return effect, effect - z * se, effect + z * se


def metric_effect_and_pvalue(
    subgroup_pred: np.ndarray,
    subgroup_y: np.ndarray,
    complement_pred: np.ndarray,
    complement_y: np.ndarray,
    metric: str,
    *,
    seed: int = 0,
    permutations: int = 499,
    bootstrap: int = 499,
) -> tuple[float, float, float, float]:
    """Return (disadvantage, p_value, ci_low, ci_high) for a subgroup metric.

    Accuracy uses the exact 2x2 Fisher test + Wald CI elsewhere. For balanced
    accuracy, row-level correctness is not a sufficient statistic because the
    metric averages per-class recall. We therefore use a stratified permutation
    test (shuffle subgroup membership within each true class) and a stratified
    bootstrap CI. The subgroup-minus-complement direction is converted to the
    project convention: positive means subgroup is worse.
    """
    if metric != "balanced_accuracy":
        raise ValueError("metric_effect_and_pvalue is intended for balanced_accuracy")

    def score(pred, y):
        classes = np.unique(np.concatenate([np.unique(y), np.unique(pred)]))
        recalls = []
        for cls in classes:
            m = y == cls
            if m.any():
                recalls.append(float(np.mean(pred[m] == cls)))
        return float(np.mean(recalls)) if recalls else 0.0

    sub_score = score(subgroup_pred, subgroup_y)
    comp_score = score(complement_pred, complement_y)
    observed = comp_score - sub_score

    rng = np.random.default_rng(seed)
    y_all = np.concatenate([subgroup_y, complement_y])
    pred_all = np.concatenate([subgroup_pred, complement_pred])
    n_sub = len(subgroup_y)
    sub_membership = np.zeros(len(y_all), dtype=bool)
    sub_membership[:n_sub] = True

    # Stratified permutation: shuffle membership within each true class.
    extreme = 0
    valid_perm = 0
    for _ in range(max(99, permutations)):
        perm = sub_membership.copy()
        for cls in np.unique(y_all):
            idx = np.flatnonzero(y_all == cls)
            if idx.size < 2:
                continue
            shuffled = rng.permutation(perm[idx])
            perm[idx] = shuffled
        if perm.sum() == 0 or (~perm).sum() == 0:
            continue
        stat = score(pred_all[~perm], y_all[~perm]) - score(pred_all[perm], y_all[perm])
        extreme += int(stat >= observed - 1e-12)
        valid_perm += 1
    p_value = (extreme + 1) / (valid_perm + 1)

    # Stratified bootstrap CI: resample rows with replacement within each true
    # class separately for subgroup and complement.
    boot = []
    for _ in range(max(99, bootstrap)):
        sub_parts = []
        comp_parts = []
        for cls in np.unique(y_all):
            si = np.flatnonzero(subgroup_y == cls)
            ci = np.flatnonzero(complement_y == cls)
            if si.size:
                take = rng.choice(si, size=si.size, replace=True)
                sub_parts.append((subgroup_pred[take], subgroup_y[take]))
            if ci.size:
                take = rng.choice(ci, size=ci.size, replace=True)
                comp_parts.append((complement_pred[take], complement_y[take]))
        if not sub_parts or not comp_parts:
            continue
        sp = np.concatenate([a for a, _ in sub_parts])
        sy = np.concatenate([b for _, b in sub_parts])
        cp = np.concatenate([a for a, _ in comp_parts])
        cy = np.concatenate([b for _, b in comp_parts])
        boot.append(score(cp, cy) - score(sp, sy))
    if boot:
        ci_low, ci_high = np.percentile(np.asarray(boot), [2.5, 97.5]).tolist()
    else:
        ci_low = ci_high = observed
    return float(observed), float(p_value), float(ci_low), float(ci_high)


def adjust_p_values(p_values: Iterable[float], alpha: float = 0.05) -> list[tuple[float, bool]]:
    """Benjamini-Hochberg FDR correction over a *pre-specified* test family.

    Guarantee (documented honestly): the controller executes the entire family
    passed here — selection of *which* tests to run happens by a fixed ranking
    inside the family, and BH is applied to all executed tests. FDR control
    holds for the executed family under the standard BH conditions (independent
    or PRDS p-values). The full candidate pool from which the family was drawn
    is larger; the controller therefore also reports `family_size` vs
    `candidate_pool_size` so the guarantee's scope is explicit.
    """
    values = [float(np.clip(p, 0.0, 1.0)) for p in p_values]
    if not values:
        return []
    rejected, adjusted, _, _ = multipletests(values, alpha=alpha, method="fdr_bh")
    return [(float(p), bool(r)) for p, r in zip(adjusted, rejected)]


def evidence_score(
    *,
    adjusted_p_value: float,
    support: float,
    effect_gap: float,
    validation_gap: float,
    counterexample_verified: bool = False,
) -> float:
    """Transparent 0-100 heuristic evidence strength (NOT a probability, NOT a
    calibrated severity, NOT comparable across unrelated failure types).

    Weights sum to 1.0. The counterexample term is a fixed bonus when a
    boundary-sensitivity probe agrees with the region.
    """
    significance = max(0.0, min(1.0, -np.log10(max(adjusted_p_value, 1e-12)) / 4.0))
    support_term = max(0.0, min(1.0, support / 0.25))
    effect_term = max(0.0, min(1.0, effect_gap / 0.25))
    validation_term = max(0.0, min(1.0, validation_gap / 0.20))
    counter_term = 0.10 if counterexample_verified else 0.0
    score = 100.0 * (
        0.30 * significance
        + 0.20 * support_term
        + 0.30 * effect_term
        + 0.20 * validation_term
        + counter_term
    )
    return float(min(100.0, score))


def severity(gap: float, support: float) -> str:
    """Single canonical severity classifier (heuristic thresholds, documented).

    HIGH: gap >= 0.15 and support >= 0.08
    MEDIUM: gap >= 0.08 or gap*support >= 0.015
    LOW otherwise.
    """
    if gap >= 0.15 and support >= 0.08:
        return "HIGH"
    if gap >= 0.08 or gap * support >= 0.015:
        return "MEDIUM"
    return "LOW"
