from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Iterable
import re
import hashlib
import time

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


from modelxray.core.adapters import ClassifierAdapter
from modelxray.core.profiler import SUPPORTED_METRICS, compute_metric
from modelxray.investigation.counterexamples import discover_counterexamples
from modelxray.investigation.failure_clustering import cluster_validated_failures, clusters_as_dict
from modelxray.investigation.schemas import Experiment, mask_for_experiment, numeric_mask
from modelxray.validation.reliability import (
    adjust_p_values,
    evidence_score,
    metric_effect_and_pvalue,
    severity,
    two_proportion_effect_and_ci,
)

# Holdout gate configuration (documented in README):
#   MIN_DISCOVERY_GAP   discovery-stage gap for a candidate to earn holdout validation
#   MIN_VALIDATION_GAP  holdout gap must reach this minimum practical effect
#   MIN_HOLDOUT_ROWS    holdout gate refuses to judge on tiny slices
#   CI_LOW_MIN          holdout disadvantage-effect CI lower bound (CREDIBLE effect)
#   ALPHA               BH-FDR level applied to the executed family
MIN_DISCOVERY_GAP = 0.05
MIN_VALIDATION_GAP = 0.03
MIN_HOLDOUT_ROWS = 10
CI_LOW_MIN = 0.0
ALPHA = 0.05


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "big")


@dataclass(frozen=True)
class ExperimentObservation:
    experiment_id: str
    condition: str
    kind: str
    support: float
    subgroup_accuracy: float
    baseline_accuracy: float
    gap: float
    p_value: float
    severity: str
    validated: bool
    validation_gap: float
    adjusted_p_value: float = 1.0
    effect_size: float = 0.0
    effect_ci_low: float = 0.0
    effect_ci_high: float = 0.0
    holdout_effect_ci_low: float = 0.0
    holdout_effect_ci_high: float = 0.0
    holdout_ci_credible: bool = False
    reproducible: bool = False
    evidence_score: float = 0.0
    failure_cluster_id: str | None = None
    evaluation_ms: float = 0.0


@dataclass(frozen=True)
class InvestigationRun:
    budget: int
    experiments_considered: int
    experiments_executed: int
    observations: list[dict[str, Any]]
    counterexamples: list[dict[str, Any]]
    candidate_pool_size: int = 0
    metric: str = "accuracy"
    search_coverage: dict[str, Any] | None = None
    failure_clusters: list[dict[str, Any]] | None = None


def generate_initial_experiments(X: pd.DataFrame) -> list[Experiment]:
    """Baseline hypothesis family: single-feature quantile thresholds.

    Model-independent and cheap. The controller owns this family; interaction
    experiments are generated on top by `generate_interaction_experiments` and
    share the same validation pipeline (no separate detector).
    """
    experiments: list[Experiment] = []
    for feature in X.select_dtypes(include=np.number).columns:
        values = X[feature].to_numpy(dtype=float)
        finite = np.isfinite(values)
        unique = np.unique(values[finite])
        if unique.size < 5:
            continue
        for qi, q in enumerate((0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9), start=1):
            threshold = float(np.quantile(unique, q))
            for direction in ("<=", ">"):
                support = q if direction == "<=" else 1.0 - q
                priority = 1.0 - abs(support - 0.2)
                experiments.append(
                    Experiment(
                        id=_experiment_id(feature, qi, direction),
                        kind="subgroup_threshold",
                        feature=feature,
                        operator=direction,
                        threshold=threshold,
                        priority=float(priority),
                        rationale=f"Probe the {feature} {direction} quantile boundary ({q:.0%}).",
                    )
                )
    return experiments


def _experiment_id(feature: str, qi: int, direction: str) -> str:
    suffix = direction.replace(">", "gt").replace("<", "lt").replace("=", "eq")
    safe_feature = feature.replace("-", "_dash_")
    return f"E-{safe_feature}-{qi}-{suffix}"


def generate_categorical_experiments(
    X: pd.DataFrame,
    y: pd.Series,
    predictions: np.ndarray,
    baseline: float,
    *,
    max_experiments: int = 80,
    min_support: float = 0.06,
    max_support: float = 0.60,
    max_categories_per_feature: int = 8,
) -> list[Experiment]:
    """Generate interpretable single-category subgroup hypotheses.

    Only categorical columns are considered. Rare levels and near-global levels
    are skipped so the family remains useful under the investigation budget.
    Every candidate enters the same discovery/FDR/holdout pipeline as numeric
    hypotheses; this function only constructs hypotheses.
    """
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    if not categorical:
        return []

    y_arr = np.asarray(y)
    experiments: list[Experiment] = []
    ranked: list[tuple[float, str, object, float]] = []
    for feature in categorical:
        series = X[feature]
        if series.isna().all():
            continue
        counts = series.value_counts(dropna=True)
        for category, count in counts.head(max_categories_per_feature).items():
            support = float(count / len(series))
            if not (min_support <= support <= max_support):
                continue
            mask = series.to_numpy(dtype=object) == category
            if not mask.any() or (~mask).sum() == 0:
                continue
            subgroup_score = compute_metric(predictions[mask], y_arr[mask], "accuracy")
            gap = baseline - subgroup_score
            if gap < 0.5 * MIN_DISCOVERY_GAP:
                continue
            ranked.append((float(gap * (1.0 + support)), feature, category, support))

    ranked.sort(reverse=True, key=lambda item: item[0])
    seen: set[tuple[str, str, str]] = set()
    for rank, (signal, feature, category, support) in enumerate(ranked[:max_experiments]):
        key = (feature, type(category).__name__, repr(category))
        if key in seen:
            continue
        seen.add(key)
        token = hashlib.sha256(f"{feature}\0{type(category).__name__}\0{category!r}".encode("utf-8")).hexdigest()[:10]
        experiments.append(
            Experiment(
                id=f"EC-{token}",
                kind="categorical_subgroup",
                feature=feature,
                operator="==",
                threshold=0.0,
                category=category,
                priority=float(1.25 + min(signal, 1.0) + 0.25 * support - rank * 1e-5),
                rationale=(
                    f"Categorical subgroup: {feature} == {category!r}; "
                    f"support {support:.0%}, estimated accuracy gap {signal / (1.0 + support):.3f}."
                ),
            )
        )
    return experiments


def generate_interaction_experiments(
    X: pd.DataFrame,
    y: pd.Series,
    predictions: np.ndarray,
    baseline: float,
    *,
    max_features: int = 8,
    max_experiments: int = 120,
    min_branch_support: float = 0.06,
) -> list[Experiment]:
    """Interaction-capable hypothesis generation (tree-guided, budget-aware).

    Strategy (fits ModelXray's architecture instead of a second detector):
      1. Build a shallow, interpretable error tree: fit a depth-3
         DecisionTreeClassifier on `correct` (1 = model wrong) over the numeric
         features. Leaves with error rate clearly above baseline become
         candidate error regions.
      2. Extract each leaf's root-to-leaf path as a conjunction of up to 3
         threshold conditions — an interpretable interaction hypothesis.
      3. Convert each path into an `Experiment` of kind "interaction" (V1 keeps
         the two strongest conditions per path; deeper conditions are folded
         into priority/rationale).

    This is a *candidate generator*, not a validator: every experiment still
    goes through the same Fisher screening, BH correction, and holdout gate as
    the baseline grid — one canonical validation pipeline.
    """
    from sklearn.tree import DecisionTreeClassifier

    numeric = X.select_dtypes(include=np.number).columns.tolist()
    if len(numeric) < 2:
        return []

    correct = (predictions == np.asarray(y)).astype(int)
    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=max(12, int(min_branch_support * len(X))),
        class_weight="balanced",
        random_state=0,
    )
    try:
        tree.fit(X[numeric], correct)
    except ValueError:
        return []

    tree_ = tree.tree_
    paths: list[tuple[list[tuple[str, str, float]], np.ndarray]] = []

    def _walk(node: int, conditions: list[tuple[str, str, float]], row_mask: np.ndarray) -> None:
        if tree_.children_left[node] == -1:  # leaf
            if row_mask.sum() > 0:
                paths.append((list(conditions), row_mask))
            return
        feature_name = numeric[tree_.feature[node]]
        threshold = float(tree_.threshold[node])
        left_mask = row_mask & (X[feature_name].to_numpy(dtype=float) <= threshold)
        right_mask = row_mask & (X[feature_name].to_numpy(dtype=float) > threshold)
        _walk(tree_.children_left[node], conditions + [(feature_name, "<=", threshold)], left_mask)
        _walk(tree_.children_right[node], conditions + [(feature_name, ">", threshold)], right_mask)

    root_mask = np.ones(len(X), dtype=bool)
    _walk(0, [], root_mask)

    experiments: list[Experiment] = []
    seen: set[tuple] = set()
    for conditions, row_mask in paths:
        if len(conditions) < 2:
            continue  # single-condition paths belong to the baseline grid
        # Degenerate-path filter: a path that splits the SAME feature twice
        # collapses to one threshold condition (a grid candidate), not an
        # interaction. Keep only paths with two distinct features.
        first_feature = conditions[0][0]
        second = next(((f, o, t) for f, o, t in conditions[1:] if f != first_feature), None)
        if second is None:
            continue
        leaf_rate = float(correct[row_mask].mean()) if row_mask.sum() else 0.0
        support = float(row_mask.mean())
        if support < min_branch_support:
            continue
        leaf_gap = baseline - leaf_rate
        if leaf_gap < 0.5 * MIN_DISCOVERY_GAP:
            continue
        key = (first_feature, conditions[0][1], conditions[0][2], second)
        if key in seen:
            continue
        seen.add(key)
        (fa, op_a, ta) = conditions[0]
        (fb, op_b, tb) = second
        safe_a = fa.replace("-", "_dash_")
        safe_b = fb.replace("-", "_dash_")
        idx = len(experiments)
        experiments.append(
            Experiment(
                id=f"EI-{safe_a}-{safe_b}-{idx}",
                kind="interaction",
                feature=fa,
                operator=op_a,
                threshold=float(ta),
                feature_b=fb,
                operator_b=op_b,
                threshold_b=float(tb),
                priority=float(1.0 + min(leaf_gap, 0.5) + support),
                rationale=(
                    f"Tree-guided interaction: error tree leaf with support {support:.0%} "
                    f"and estimated gap {leaf_gap:.3f} (discovery split)."
                ),
            )
        )
        if len(experiments) >= max_experiments:
            break

    if len(experiments) > max_features * 40:
        experiments.sort(key=lambda e: e.priority, reverse=True)
        experiments = experiments[: max_features * 40]
    return experiments


def generate_pairwise_interaction_experiments(
    X: pd.DataFrame,
    y: pd.Series,
    predictions: np.ndarray,
    baseline: float,
    *,
    max_features: int = 6,
    max_experiments: int = 120,
    min_support: float = 0.06,
) -> list[Experiment]:
    """Targeted pairwise conjunction family for interaction faults.

    Rank numeric features by their univariate association with the model-error
    indicator, then search a small, interpretable quantile grid over the top
    features. This complements the tree-guided generator: the tree can miss a
    conjunction when neither branch is individually strong, while this targeted
    pairwise search explicitly tests two-feature regions without enumerating the
    full combinatorial feature space.
    """
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    if len(numeric) < 2:
        return []
    correct = (predictions == np.asarray(y)).astype(int)
    error = 1 - correct

    rankings: list[tuple[float, str]] = []
    for feature in numeric:
        values = pd.to_numeric(X[feature], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() < 20 or np.unique(values[finite]).size < 5:
            continue
        xf = values[finite]
        ef = error[finite]
        if ef.sum() == 0 or ef.sum() == len(ef):
            assoc = 0.0
        else:
            mu_x = float(xf.mean())
            mu_e = float(ef.mean())
            denom = float(xf.std(ddof=0) * ef.std(ddof=0))
            assoc = abs(float(((xf - mu_x) * (ef - mu_e)).mean()) / denom) if denom > 1e-12 else 0.0
        rankings.append((assoc, feature))

    rankings.sort(reverse=True)
    selected = [f for _, f in rankings[:max_features]]
    if len(selected) < 2:
        return []

    quantiles = (0.20, 0.30, 0.70, 0.80)
    experiments: list[Experiment] = []
    seen: set[tuple] = set()
    for i, fa in enumerate(selected):
        va = pd.to_numeric(X[fa], errors="coerce").to_numpy(dtype=float)
        for fb in selected[i + 1 :]:
            vb = pd.to_numeric(X[fb], errors="coerce").to_numpy(dtype=float)
            qa = {q: float(np.quantile(va[np.isfinite(va)], q)) for q in quantiles}
            qb = {q: float(np.quantile(vb[np.isfinite(vb)], q)) for q in quantiles}
            for qa_q, ta in qa.items():
                for oa in ("<=", ">"):
                    for qb_q, tb in qb.items():
                        for ob in ("<=", ">"):
                            ma = numeric_mask(va, oa, ta)
                            mb = numeric_mask(vb, ob, tb)
                            mask = ma & mb
                            support = float(mask.mean())
                            if support < min_support or support > 0.55:
                                continue
                            subgroup_score = compute_metric(predictions[mask], np.asarray(y)[mask], "accuracy")
                            gap = baseline - subgroup_score
                            if gap < 0.5 * MIN_DISCOVERY_GAP:
                                continue
                            key = (fa, oa, round(ta, 6), fb, ob, round(tb, 6))
                            if key in seen:
                                continue
                            seen.add(key)
                            idx = len(experiments)
                            experiments.append(
                                Experiment(
                                    id=f"EP-{fa.replace('-', '_dash_')}-{fb.replace('-', '_dash_')}-{idx}",
                                    kind="interaction",
                                    feature=fa,
                                    operator=oa,
                                    threshold=float(ta),
                                    feature_b=fb,
                                    operator_b=ob,
                                    threshold_b=float(tb),
                                    priority=float(1.5 + min(gap, 0.5) + support + 0.25 * (rankings[i][0] + rankings[selected.index(fb)][0])),
                                    rationale=(
                                        f"Targeted pairwise interaction: {fa} {oa} {ta:.4g} AND "
                                        f"{fb} {ob} {tb:.4g}; support {support:.0%}, gap {gap:.3f}."
                                    ),
                                )
                            )
                            if len(experiments) >= max_experiments:
                                return experiments
    return experiments


def generate_oblique_band_experiments(
    X: pd.DataFrame,
    y: pd.Series,
    predictions: np.ndarray,
    baseline: float,
    *,
    max_features: int = 8,
    max_experiments: int = 40,
    min_band_support: float = 0.06,
) -> list[Experiment]:
    """Generate interpretable weighted-projection bands for non-axis-aligned faults.

    Candidate directions come from three deterministic sources: a raw equal-weight
    projection, PCA directions, and an error-classifier direction. Each direction
    is converted into a raw-feature linear score and scanned for narrow quantile
    bands. This is candidate generation only; all candidates share the canonical
    discovery -> BH-FDR -> holdout validation pipeline.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    numeric = X.select_dtypes(include=np.number).columns.tolist()
    if len(numeric) < 2 or len(X) < 100:
        return []
    numeric = numeric[:max_features]
    frame = X[numeric].apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any():
        return []

    values = frame.to_numpy(dtype=float)
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales[scales < 1e-9] = 1.0
    z = (values - means) / scales
    correct = (predictions == np.asarray(y)).astype(int)
    error = 1 - correct
    directions: list[tuple[np.ndarray, bool]] = []

    # Raw equal-weight score catches simple aggregate/diagonal bands directly.
    # Keep a second standardized variant because either representation can be the
    # meaningful score depending on feature scale.
    directions.append((np.ones(len(numeric), dtype=float) / np.sqrt(len(numeric)), False))
    directions.append((np.ones(len(numeric), dtype=float) / np.sqrt(len(numeric)), True))

    # PCA directions capture dominant correlated axes.
    try:
        n_components = min(3, len(numeric))
        pca = PCA(n_components=n_components, random_state=0).fit(z)
        directions.extend([(np.asarray(v, dtype=float), True) for v in pca.components_])
    except (ValueError, TypeError):
        pass

    # A linear classifier over the error indicator adds a supervised direction.
    if np.unique(error).size == 2 and int(error.sum()) >= 10:
        try:
            clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=500, random_state=0).fit(z, error)
            directions.append((np.asarray(clf.coef_[0], dtype=float), True))
        except (ValueError, TypeError):
            pass

    experiments: list[Experiment] = []
    seen: set[tuple] = set()
    band_centers = (0.50, 0.42, 0.58)
    band_widths = (0.06, 0.08, 0.10, 0.14)

    for direction, standardized in directions:
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            continue
        direction = direction / norm
        if standardized:
            raw_weights = direction / scales
            raw_intercept = float(-np.sum(raw_weights * means))
        else:
            raw_weights = direction
            raw_intercept = 0.0
        projection = values @ raw_weights + raw_intercept
        finite_projection = projection[np.isfinite(projection)]
        if finite_projection.size < 100:
            continue
        for center_q in band_centers:
            center = float(np.quantile(finite_projection, center_q))
            for width_q in band_widths:
                half = width_q / 2.0
                lower = float(np.quantile(finite_projection, max(0.001, center_q - half)))
                upper = float(np.quantile(finite_projection, min(0.999, center_q + half)))
                if not lower < upper:
                    continue
                mask = (projection >= lower) & (projection <= upper)
                support = float(mask.mean())
                if support < min_band_support:
                    continue
                band_score = compute_metric(predictions[mask], np.asarray(y)[mask], "accuracy")
                gap = baseline - band_score
                if gap < 0.5 * MIN_DISCOVERY_GAP:
                    continue
                dominant = tuple(np.argsort(np.abs(raw_weights))[-min(4, len(raw_weights)):].tolist())
                key = (tuple(np.round(raw_weights, 5)), round(lower, 5), round(upper, 5))
                if key in seen:
                    continue
                seen.add(key)
                terms = tuple(
                    (numeric[i], float(raw_weights[i]))
                    for i in np.argsort(np.abs(raw_weights))[::-1]
                    if abs(float(raw_weights[i])) >= 1e-6
                )
                idx = len(experiments)
                experiments.append(
                    Experiment(
                        id=f"EO-{idx}",
                        kind="oblique_band",
                        feature=terms[0][0],
                        operator="band",
                        threshold=lower,
                        priority=float(2.0 + min(gap, 0.5) + support),
                        rationale=(
                            f"Oblique projection band over {len(terms)} features; "
                            f"support {support:.0%}, estimated gap {gap:.3f}."
                        ),
                        oblique_terms=terms,
                        oblique_intercept=raw_intercept,
                        oblique_lower=lower,
                        oblique_upper=upper,
                    )
                )
                if len(experiments) >= max_experiments:
                    return experiments
    return experiments


def generate_prototype_region_experiments(
    X: pd.DataFrame,
    y: pd.Series,
    predictions: np.ndarray,
    baseline: float,
    *,
    max_features: int = 6,
    max_experiments: int = 24,
    min_support: float = 0.06,
) -> list[Experiment]:
    """Novelty-search family: error-heavy local neighborhoods discovered via KMeans.

    This intentionally does not assume axis-aligned thresholds or a linear
    decision boundary.  Prototypes are learned from the input distribution only;
    the model-error signal is used solely to rank candidate neighborhoods.  The
    resulting region is a standardized Euclidean neighborhood around a concrete
    prototype, then enters the same canonical validation/FDR pipeline.
    """
    from sklearn.cluster import KMeans

    numeric = X.select_dtypes(include=np.number).columns.tolist()
    if len(numeric) < 2 or len(X) < 120:
        return []
    numeric = numeric[:max_features]
    frame = X[numeric].apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any():
        return []
    values = frame.to_numpy(dtype=float)
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales[scales < 1e-9] = 1.0
    z = (values - means) / scales
    error = (predictions != np.asarray(y)).astype(int)

    ranked: list[tuple[float, int, int, np.ndarray, float, float]] = []
    max_k = min(5, max(2, len(X) // max(40, int(len(X) * min_support))))
    for k in range(2, max_k + 1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=17 + k)
            labels = km.fit_predict(z)
        except (ValueError, TypeError):
            continue
        for cluster_id in range(k):
            mask = labels == cluster_id
            support = float(mask.mean())
            if support < min_support or support > 0.60:
                continue
            gap = baseline - compute_metric(predictions[mask], np.asarray(y)[mask], "accuracy")
            if gap < 0.5 * MIN_DISCOVERY_GAP:
                continue
            pts = z[mask]
            distances = np.linalg.norm(pts - km.cluster_centers_[cluster_id], axis=1)
            radius = float(np.quantile(distances, 0.90))
            ranked.append((gap * (1.0 + support), k, cluster_id, km.cluster_centers_[cluster_id], radius, support))

    ranked.sort(reverse=True, key=lambda item: item[0])
    experiments: list[Experiment] = []
    seen: set[tuple] = set()
    for rank, (signal, k, cluster_id, center_z, radius, support) in enumerate(ranked):
        center = means + center_z * scales
        key = (k, cluster_id, tuple(np.round(center, 5)), round(radius, 5))
        if key in seen:
            continue
        seen.add(key)
        experiments.append(
            Experiment(
                id=f"EP-{k}-{cluster_id}-{rank}",
                kind="prototype_region",
                feature=numeric[0],
                operator="prototype",
                threshold=0.0,
                priority=float(1.6 + min(signal, 1.0) + 0.15 * support),
                rationale=(
                    f"Novelty-search prototype neighborhood k={k}, cluster={cluster_id}; "
                    f"support {support:.0%}, estimated gap {signal / (1.0 + support):.3f}."
                ),
                cluster_centroid=tuple(float(v) for v in center),
                cluster_scales=tuple(float(v) for v in scales),
                cluster_radius=radius,
                cluster_k=k,
                cluster_id=cluster_id,
            )
        )
        if len(experiments) >= max_experiments:
            break
    return experiments


def _family_for_experiment(exp: Experiment) -> str:
    if exp.kind == "subgroup_threshold":
        return "grid"
    if exp.kind == "categorical_subgroup":
        return "categorical"
    if exp.kind == "interaction":
        return "interaction"
    if exp.kind == "oblique_band":
        return "oblique_band"
    if exp.kind == "prototype_region":
        return "prototype_region"
    return exp.kind


def _family_signal(family: str, observations: Iterable[ExperimentObservation]) -> tuple[float, int, int]:
    family_obs = [o for o in observations if o.kind == ("subgroup_threshold" if family == "grid" else family)]
    if not family_obs:
        return 0.0, 0, 0
    best_gap = max((max(0.0, float(o.gap)) for o in family_obs), default=0.0)
    validated = sum(1 for o in family_obs if o.reproducible)
    return best_gap, len(family_obs), validated


def _rank_key(exp: Experiment, observations: Iterable[ExperimentObservation], *, cost_aware: bool = True) -> tuple[float, float, float]:
    """Adaptive candidate score used by the sequential controller.

    The controller deliberately does not claim formal value-of-information. It
    uses a lightweight explore/exploit rule: each available hypothesis family
    is sampled at least once, then families with stronger observed error gaps or
    validated discoveries receive more remaining budget. Within a family, the
    candidate's static priority is combined with evidence observed on related
    candidates.
    """
    obs = list(observations)
    family = _family_for_experiment(exp)
    best_gap, family_trials, validated = _family_signal(family, obs)

    # Exploration bonus is deliberately large enough to give untouched families
    # an early probe, then decays as evidence accumulates. The validated bonus
    # encourages continued probing after a family has demonstrated a real fault.
    exploration = 1.25 / np.sqrt(max(1, family_trials))
    # Soft allocation prior: preserve breadth while still allowing evidence-driven
    # exploitation. The prior is not a hard quota; it simply raises the score of
    # families that have consumed less than their intended share of the budget.
    target_share = {
        "grid": 0.20,
        "categorical": 0.10,
        "interaction": 0.35,
        "oblique_band": 0.20,
        "prototype_region": 0.15,
    }.get(family, 0.10)
    total_trials = max(1, len(obs))
    current_share = family_trials / total_trials
    allocation_gap = max(0.0, target_share - current_share)
    allocation_bonus = 1.5 * allocation_gap
    # Strong observed error evidence should outweigh the soft exploration prior.
    family_bonus = 3.5 * best_gap + 0.35 * validated + exploration + allocation_bonus

    # Use a deterministic structural cost estimate for ranking. Wall-clock
    # timing is retained as telemetry (`evaluation_ms`), but must not influence
    # experiment selection because runtime jitter would make otherwise identical
    # investigations choose different hypotheses and break replay determinism.
    estimated_costs = {
        "grid": 1.00,
        "categorical": 1.10,
        "interaction": 1.35,
        "oblique_band": 1.75,
        "prototype_region": 2.00,
    }
    cost_penalty = np.sqrt(estimated_costs.get(family, 1.25)) if cost_aware else 1.0

    # Keep the existing local exploitation behavior, but parse the actual
    # Experiment object rather than inferring features from serialized IDs.
    related_gap = 0.0
    exp_features = set(exp.features)
    for item in obs:
        observed_features = set(_features_from_condition(item.condition))
        if exp_features & observed_features:
            related_gap = max(related_gap, abs(float(item.gap)))
    candidate_score = exp.priority + 1.5 * related_gap
    raw_score = family_bonus + candidate_score
    cost_adjusted_score = raw_score / max(cost_penalty, 0.5) if cost_aware else raw_score
    return (cost_adjusted_score, raw_score, exp.priority)


def _features_from_condition(condition: str) -> list[str]:
    """Extract feature names from a serialized hypothesis condition.

    Works for simple threshold conjunctions and oblique projection bands.
    Uses the condition text as the source of truth instead of trying to infer
    feature names from experiment IDs (which is fragile for hyphenated names).
    """
    features: list[str] = []
    if condition.startswith("prototype neighborhood"):
        for feature in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=", condition):
            features.append(feature)
        return list(dict.fromkeys(features))
    if " ∈ [" in condition:
        terms_text = condition.rsplit(" ∈ [", 1)[0]
        for feature in re.findall(r"\*([A-Za-z_][A-Za-z0-9_]*(?:_dash_[A-Za-z0-9_]+)*)", terms_text):
            features.append(feature.replace("_dash_", "-"))
        return list(dict.fromkeys(features))
    for part in condition.split(" & "):
        tokens = part.strip().rsplit(" ", 2)
        if len(tokens) == 3:
            features.append(tokens[0])
    return list(dict.fromkeys(features))


def _choose_experiment(candidates: list[Experiment], observations: list[ExperimentObservation], *, cost_aware: bool = True) -> Experiment:
    executed = {o.experiment_id for o in observations}
    remaining = [c for c in candidates if c.id not in executed]
    if not remaining:
        raise RuntimeError("No experiments remain")
    remaining.sort(key=lambda e: _rank_key(e, observations, cost_aware=cost_aware), reverse=True)
    return remaining[0]


def _evaluate(
    experiment: Experiment,
    predictions: np.ndarray,
    X: pd.DataFrame,
    y: pd.Series,
    baseline: float,
    metric: str,
) -> ExperimentObservation:
    mask = mask_for_experiment(experiment, X)
    support = float(mask.mean())
    condition = experiment.describe()
    if support <= 0.0 or support >= 1.0:
        return ExperimentObservation(
            experiment.id, condition, experiment.kind, support, 0.0, baseline, 0.0, 1.0, severity(0.0, support), False, 0.0
        )

    y_arr = np.asarray(y)
    subgroup_pred = predictions[mask]
    subgroup_y = y_arr[mask]
    complement_pred = predictions[~mask]
    complement_y = y_arr[~mask]
    subgroup_score = compute_metric(subgroup_pred, subgroup_y, metric)
    gap = baseline - subgroup_score
    if metric == "balanced_accuracy":
        effect, p, ci_low, ci_high = metric_effect_and_pvalue(
            subgroup_pred, subgroup_y, complement_pred, complement_y, metric,
            seed=_stable_seed(experiment.id),
        )
    else:
        correct = (predictions == y_arr).astype(int)
        a = int(correct[mask].sum())
        b = int(mask.sum() - a)
        c = int(correct[~mask].sum())
        d = int((~mask).sum() - c)
        _, p = fisher_exact([[a, b], [c, d]], alternative="less")
        effect, ci_low, ci_high = two_proportion_effect_and_ci(a, a + b, c, c + d)
    return ExperimentObservation(
        experiment_id=experiment.id,
        condition=condition,
        kind=experiment.kind,
        support=support,
        subgroup_accuracy=float(subgroup_score),
        baseline_accuracy=baseline,
        gap=float(gap),
        p_value=float(p),
        severity=severity(float(gap), support),
        validated=False,
        validation_gap=0.0,
        effect_size=float(effect),
        effect_ci_low=float(ci_low),
        effect_ci_high=float(ci_high),
    )


def _validate(
    observation: ExperimentObservation,
    experiment: Experiment,
    predictions: np.ndarray,
    X: pd.DataFrame,
    y: pd.Series,
    baseline: float,
    metric: str,
) -> ExperimentObservation:
    mask = mask_for_experiment(experiment, X)
    n_in = int(mask.sum())
    n_out = int((~mask).sum())
    if n_in < MIN_HOLDOUT_ROWS or n_out < MIN_HOLDOUT_ROWS:
        return observation
    y_arr = np.asarray(y)
    subgroup_pred = predictions[mask]
    subgroup_y = y_arr[mask]
    complement_pred = predictions[~mask]
    complement_y = y_arr[~mask]
    validation_score = compute_metric(subgroup_pred, subgroup_y, metric)
    validation_gap = float(baseline - validation_score)
    # Disadvantage effect + CI on holdout rows (same sign convention as discovery).
    if metric == "balanced_accuracy":
        effect, _, ci_low, ci_high = metric_effect_and_pvalue(
            subgroup_pred, subgroup_y, complement_pred, complement_y, metric,
            seed=_stable_seed(f"{experiment.id}:holdout"),
        )
    else:
        correct = (predictions == y_arr).astype(int)
        a = int(correct[mask].sum())
        b = int(n_in - a)
        c = int(correct[~mask].sum())
        d = int(n_out - c)
        effect, ci_low, ci_high = two_proportion_effect_and_ci(a, a + b, c, c + d)
    # Holdout gate: minimum practical effect AND sign stability AND the CI's
    # lower bound above zero (credible effect, not just a noisy point estimate).
    validated = (
        observation.gap >= MIN_DISCOVERY_GAP
        and validation_gap >= MIN_VALIDATION_GAP
        and np.sign(observation.gap) == np.sign(validation_gap)
        and ci_low >= CI_LOW_MIN
    )
    return ExperimentObservation(
        **{
            **asdict(observation),
            "validated": bool(validated),
            "validation_gap": validation_gap,
            "holdout_effect_ci_low": ci_low,
            "holdout_effect_ci_high": ci_high,
            "holdout_ci_credible": bool(ci_low > 0.0),
        }
    )


def _build_search_coverage(
    candidates: list[Experiment],
    observations: list[ExperimentObservation],
) -> dict[str, Any]:
    """Describe how much of the *generated candidate pool* was actually tested.

    This is intentionally not called model-safety coverage or search-space recall.
    It quantifies only the finite hypotheses ModelXray generated for this run.
    """
    family_order = ("grid", "categorical", "interaction", "oblique_band", "prototype_region")
    candidate_by_family: dict[str, int] = {family: 0 for family in family_order}
    executed_by_family: dict[str, int] = {family: 0 for family in family_order}
    validated_by_family: dict[str, int] = {family: 0 for family in family_order}

    candidate_features: dict[str, set[str]] = {family: set() for family in family_order}
    executed_features: dict[str, set[str]] = {family: set() for family in family_order}
    candidate_ids = {c.id: c for c in candidates}

    for candidate in candidates:
        family = _family_for_experiment(candidate)
        candidate_by_family[family] = candidate_by_family.get(family, 0) + 1
        candidate_features.setdefault(family, set()).update(candidate.features)

    for obs in observations:
        candidate = candidate_ids.get(obs.experiment_id)
        if candidate is None:
            continue
        family = _family_for_experiment(candidate)
        executed_by_family[family] = executed_by_family.get(family, 0) + 1
        executed_features.setdefault(family, set()).update(candidate.features)
        if obs.reproducible:
            validated_by_family[family] = validated_by_family.get(family, 0) + 1

    family_details: dict[str, Any] = {}
    for family in family_order:
        generated = candidate_by_family.get(family, 0)
        executed = executed_by_family.get(family, 0)
        family_details[family] = {
            "generated_candidates": generated,
            "executed": executed,
            "unexplored_candidates": max(0, generated - executed),
            "candidate_execution_ratio": round(executed / generated, 4) if generated else 0.0,
            "validated_failures": validated_by_family.get(family, 0),
            "candidate_features": sorted(candidate_features.get(family, set())),
            "executed_features": sorted(executed_features.get(family, set())),
        }

    available_families = [f for f in family_order if candidate_by_family.get(f, 0) > 0]
    explored_families = [f for f in available_families if executed_by_family.get(f, 0) > 0]
    total_generated = len(candidates)
    total_executed = len(observations)
    all_candidate_features = set().union(*(candidate_features[f] for f in available_families)) if available_families else set()
    all_executed_features = set().union(*(executed_features[f] for f in available_families)) if explored_families else set()

    return {
        "scope": "generated_candidate_pool",
        "interpretation": (
            "Coverage measures the finite hypotheses generated for this run; it is not a "
            "claim about the full model behavior space or absence of undiscovered failures."
        ),
        "candidate_pool_size": total_generated,
        "executed": total_executed,
        "unexplored_candidates": max(0, total_generated - total_executed),
        "candidate_execution_ratio": round(total_executed / total_generated, 4) if total_generated else 0.0,
        "available_families": available_families,
        "explored_families": explored_families,
        "family_breadth_ratio": round(len(explored_families) / len(available_families), 4) if available_families else 0.0,
        "candidate_feature_count": len(all_candidate_features),
        "executed_feature_count": len(all_executed_features),
        "feature_breadth_ratio": round(len(all_executed_features) / len(all_candidate_features), 4) if all_candidate_features else 0.0,
        "families": family_details,
    }


def run_active_investigation(
    adapter: ClassifierAdapter,
    X_discovery: pd.DataFrame,
    y_discovery: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    budget: int = 24,
    metric: str = "accuracy",
    interaction_fraction: float = 0.35,
    oblique_fraction: float = 0.20,
    novelty_fraction: float = 0.15,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cost_aware: bool = True,
) -> InvestigationRun:
    """Budgeted sequential search with one canonical validation pipeline.

    Phases:
      1. Baseline grid family (single-feature quantile thresholds).
      2. Tree-guided interaction family (generated once, model-independent
         fitting on discovery rows only — no leakage; the tree uses
         discovery labels only).
      3. Sequential selection over the union, Fisher screening on discovery,
         holdout gate + BH on the executed family, evidence scoring.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric '{metric}'. Supported: {', '.join(SUPPORTED_METRICS)}")

    grid = generate_initial_experiments(X_discovery)
    discovery_predictions = adapter.predict(X_discovery)
    validation_predictions = adapter.predict(X_validation)
    if progress_callback:
        progress_callback({
            "phase": "PROFILING",
            "progress": 0.05,
            "completed_units": 0,
            "total_units": int(budget),
            "current_unit": "BASELINE",
            "current_family": "baseline",
        })
    baseline = compute_metric(discovery_predictions, np.asarray(y_discovery), metric)
    validation_baseline = compute_metric(validation_predictions, np.asarray(y_validation), metric)

    categorical = generate_categorical_experiments(
        X_discovery, y_discovery, discovery_predictions, baseline,
        max_experiments=max(12, int(round(budget * 3.0))),
    )
    interactions: list[Experiment] = []
    oblique: list[Experiment] = []
    interaction_budget = max(0, int(round(budget * interaction_fraction)))
    oblique_budget = max(0, int(round(budget * oblique_fraction)))
    if len(X_discovery.select_dtypes(include=np.number).columns) >= 2:
        interactions = generate_interaction_experiments(
            X_discovery, y_discovery, discovery_predictions, baseline,
            max_experiments=max(interaction_budget * 4, 40),
        )
        targeted_interactions = generate_pairwise_interaction_experiments(
            X_discovery, y_discovery, discovery_predictions, baseline,
            max_features=min(6, len(X_discovery.select_dtypes(include=np.number).columns)),
            max_experiments=max(interaction_budget * 8, 96),
        )
        # Merge tree-guided and targeted pairwise hypotheses into one canonical
        # interaction family. Exact duplicate predicates are removed by ID key.
        seen_interactions = set()
        merged_interactions: list[Experiment] = []
        for exp in [*interactions, *targeted_interactions]:
            key = (exp.feature, exp.operator, round(exp.threshold, 8), exp.feature_b, exp.operator_b,
                   round(float(exp.threshold_b), 8) if exp.threshold_b is not None else None)
            if key in seen_interactions:
                continue
            seen_interactions.add(key)
            merged_interactions.append(exp)
            if len(merged_interactions) >= max(interaction_budget * 10, 120):
                break
        interactions = merged_interactions
        oblique = generate_oblique_band_experiments(
            X_discovery, y_discovery, discovery_predictions, baseline,
            max_features=min(8, len(X_discovery.select_dtypes(include=np.number).columns)),
            max_experiments=max(oblique_budget * 5, 30),
        )
    novelty_budget = max(0, int(round(budget * novelty_fraction)))
    prototype_regions = generate_prototype_region_experiments(
        X_discovery, y_discovery, discovery_predictions, baseline,
        max_features=min(6, len(X_discovery.select_dtypes(include=np.number).columns)),
        max_experiments=max(novelty_budget * 5, 20),
    )
    candidates = grid + categorical + interactions + oblique + prototype_regions
    original_candidate_pool_size = len(candidates)
    candidate_by_id = {c.id: c for c in candidates}

    # Adaptive allocation replaces hard family caps. We still reserve one
    # exploratory slot for each available family so a promising but initially
    # weak family cannot be starved forever. Remaining slots are allocated by
    # _rank_key's family evidence score. interaction_fraction/oblique_fraction
    # remain as soft priors through candidate-generation depth, not hard quotas.
    available_families = {
        _family_for_experiment(c) for c in candidates
    }
    family_trials: dict[str, int] = {family: 0 for family in available_families}

    observations: list[ExperimentObservation] = []
    slots = min(budget, len(candidates))
    total_units = slots
    while slots > 0 and candidates:
        # Force one probe per available family before pure exploitation.
        unprobed = [c for c in candidates if family_trials.get(_family_for_experiment(c), 0) == 0]
        if unprobed:
            unprobed.sort(key=lambda e: (e.priority, e.id), reverse=True)
            exp = unprobed[0]
        else:
            exp = _choose_experiment(candidates, observations, cost_aware=cost_aware)
        slots -= 1
        family = _family_for_experiment(exp)
        family_trials[family] = family_trials.get(family, 0) + 1
        if progress_callback:
            progress_callback({
                "phase": "INVESTIGATING",
                "progress": 0.10 + (0.65 * (len(observations) / max(total_units, 1))),
                "completed_units": len(observations),
                "total_units": total_units,
                "current_unit": exp.id,
                "current_family": family,
            })
        started = time.perf_counter()
        obs = _evaluate(exp, discovery_predictions, X_discovery, y_discovery, baseline, metric)
        obs = replace(obs, evaluation_ms=(time.perf_counter() - started) * 1000.0)
        if obs.gap >= MIN_DISCOVERY_GAP and obs.p_value <= 0.10:
            obs = _validate(obs, exp, validation_predictions, X_validation, y_validation, validation_baseline, metric)
        observations.append(obs)
        if progress_callback:
            progress_callback({
                "phase": "INVESTIGATING",
                "progress": 0.10 + (0.65 * (len(observations) / max(total_units, 1))),
                "completed_units": len(observations),
                "total_units": total_units,
                "current_unit": exp.id,
                "current_family": family,
            })

    # BH over the executed family (the confirmatory test set). See
    # adjust_p_values for the precise scope of the FDR guarantee.
    if progress_callback:
        progress_callback({
            "phase": "STATISTICAL_VALIDATION",
            "progress": 0.78,
            "completed_units": len(observations),
            "total_units": total_units,
            "current_unit": "BH_FDR",
            "current_family": "validation",
        })
    adjusted = adjust_p_values([o.p_value for o in observations], alpha=ALPHA)
    corrected: list[ExperimentObservation] = []
    for obs, (adj_p, rejected) in zip(observations, adjusted):
        final_validated = bool(obs.validated and rejected)
        corrected.append(ExperimentObservation(**{**asdict(obs), "adjusted_p_value": adj_p, "validated": final_validated}))
    observations = corrected

    scored: list[ExperimentObservation] = []
    y_disc_arr = np.asarray(y_discovery)
    for obs in observations:
        exp = candidate_by_id[obs.experiment_id]
        mask = mask_for_experiment(exp, X_discovery)
        sub_pred = discovery_predictions[mask]
        sub_y = y_disc_arr[mask]
        comp_pred = discovery_predictions[~mask]
        comp_y = y_disc_arr[~mask]
        if metric == "balanced_accuracy":
            effect, _, ci_low, ci_high = metric_effect_and_pvalue(
                sub_pred, sub_y, comp_pred, comp_y, metric,
                seed=_stable_seed(f"{obs.experiment_id}:score"),
            )
        else:
            correct = (discovery_predictions == y_disc_arr).astype(int)
            a = int(correct[mask].sum())
            b = int(mask.sum() - a)
            c = int(correct[~mask].sum())
            d = int((~mask).sum() - c)
            effect, ci_low, ci_high = two_proportion_effect_and_ci(a, a + b, c, c + d)
        reproducible = bool(
            obs.validated
            and obs.validation_gap >= MIN_VALIDATION_GAP
            and obs.holdout_effect_ci_low >= CI_LOW_MIN
        )
        score = evidence_score(
            adjusted_p_value=obs.adjusted_p_value,
            support=obs.support,
            effect_gap=obs.gap,
            validation_gap=obs.validation_gap,
            counterexample_verified=False,
        )
        scored.append(
            ExperimentObservation(
                **{
                    **asdict(obs),
                    "effect_size": effect,
                    "effect_ci_low": ci_low,
                    "effect_ci_high": ci_high,
                    "reproducible": reproducible,
                    "evidence_score": score,
                }
            )
        )
    observations = scored

    # Consolidate overlapping validated hypotheses before presenting them as
    # separate failures. This prevents a threshold family such as
    # x > 2.1, x > 2.2, x > 2.3 from inflating the reported failure count.
    clusters, experiment_to_cluster = cluster_validated_failures(
        observations, candidate_by_id, X_validation, jaccard_threshold=0.60
    )
    clustered_observations: list[ExperimentObservation] = []
    for obs in observations:
        clustered_observations.append(
            ExperimentObservation(
                **{
                    **asdict(obs),
                    "failure_cluster_id": experiment_to_cluster.get(obs.experiment_id),
                }
            )
        )
    observations = clustered_observations
    observations.sort(key=lambda o: (o.reproducible, o.evidence_score, o.gap * o.support), reverse=True)

    if progress_callback:
        progress_callback({
            "phase": "BOUNDARY_EVIDENCE",
            "progress": 0.90,
            "completed_units": len(observations),
            "total_units": total_units,
            "current_unit": "COUNTEREXAMPLES",
            "current_family": "boundary",
        })
    # Boundary-sensitivity probes are attached ONLY for validated failures, and
    # record which frame `source_row` indexes (validation split here). Probe one
    # representative hypothesis per consolidated cluster so the evidence count
    # does not scale with duplicate descriptions of the same failure region.
    cluster_representatives = {cluster.cluster_id: cluster.representative_experiment_id for cluster in clusters}
    selected_ids = list(cluster_representatives.values())
    selected_experiments = [candidate_by_id[eid] for eid in selected_ids[:4] if eid in candidate_by_id]
    counterexamples = discover_counterexamples(
        adapter, X_validation, selected_experiments, limit=3, row_frame_name="validation"
    )

    if progress_callback:
        progress_callback({
            "phase": "FINALIZING",
            "progress": 0.97,
            "completed_units": len(observations),
            "total_units": total_units,
            "current_unit": "ASSEMBLING_EVIDENCE",
            "current_family": "persistence",
        })
    return InvestigationRun(
        budget=budget,
        experiments_considered=original_candidate_pool_size,
        experiments_executed=len(observations),
        observations=[asdict(x) for x in observations],
        counterexamples=counterexamples,
        candidate_pool_size=len(candidates),
        metric=metric,
        search_coverage=_build_search_coverage(candidates, observations),
        failure_clusters=clusters_as_dict(clusters),
    )
