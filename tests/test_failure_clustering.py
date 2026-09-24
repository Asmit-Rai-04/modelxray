import numpy as np
import pandas as pd

from modelxray.investigation.failure_clustering import cluster_validated_failures
from modelxray.investigation.schemas import Experiment
from modelxray.investigation.controller import ExperimentObservation


def _obs(eid, condition, evidence, gap, support=0.2, cluster=None):
    return ExperimentObservation(
        experiment_id=eid,
        condition=condition,
        kind="subgroup_threshold",
        support=support,
        subgroup_accuracy=0.7,
        baseline_accuracy=0.9,
        gap=gap,
        p_value=1e-6,
        severity="HIGH" if gap >= 0.15 else "MEDIUM",
        validated=True,
        validation_gap=gap - 0.01,
        adjusted_p_value=1e-5,
        effect_size=gap,
        effect_ci_low=gap * 0.5,
        effect_ci_high=gap * 1.5,
        holdout_effect_ci_low=gap * 0.4,
        holdout_effect_ci_high=gap * 1.4,
        holdout_ci_credible=True,
        reproducible=True,
        evidence_score=evidence,
        failure_cluster_id=cluster,
    )


def test_overlapping_hypotheses_are_consolidated():
    X = pd.DataFrame({"x": np.arange(100, dtype=float) / 100.0})
    e1 = Experiment("E-x-7-gt", "subgroup_threshold", "x", ">", 0.50, 1.0, "x > .5")
    e2 = Experiment("E-x-8-gt", "subgroup_threshold", "x", ">", 0.55, 1.0, "x > .55")
    e3 = Experiment("E-x-1-lt", "subgroup_threshold", "x", "<=", 0.20, 1.0, "x <= .2")
    candidates = {e.id: e for e in (e1, e2, e3)}
    observations = [
        _obs(e1.id, e1.describe(), 80.0, 0.22),
        _obs(e2.id, e2.describe(), 90.0, 0.25),
        _obs(e3.id, e3.describe(), 70.0, 0.20),
    ]

    clusters, mapping = cluster_validated_failures(observations, candidates, X, jaccard_threshold=0.60)

    assert len(clusters) == 2
    assert mapping[e1.id] == mapping[e2.id]
    assert mapping[e1.id] != mapping[e3.id]
    grouped = next(c for c in clusters if c.cluster_id == mapping[e1.id])
    assert grouped.hypothesis_count == 2
    assert set(grouped.supporting_experiment_ids) == {e1.id, e2.id}
    assert grouped.representative_experiment_id == e2.id


def test_disjoint_validated_failures_remain_separate():
    X = pd.DataFrame({"x": np.arange(100, dtype=float) / 100.0})
    e1 = Experiment("E-x-9-gt", "subgroup_threshold", "x", ">", 0.80, 1.0, "x > .8")
    e2 = Experiment("E-x-2-lt", "subgroup_threshold", "x", "<=", 0.10, 1.0, "x <= .1")
    candidates = {e.id: e for e in (e1, e2)}
    observations = [
        _obs(e1.id, e1.describe(), 90.0, 0.21),
        _obs(e2.id, e2.describe(), 80.0, 0.19),
    ]

    clusters, mapping = cluster_validated_failures(observations, candidates, X, jaccard_threshold=0.60)

    assert len(clusters) == 2
    assert mapping[e1.id] != mapping[e2.id]


def test_clustering_does_not_chain_distinct_regions():
    """A-B and B-C overlap must not force A/C into one cluster."""
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 101)})
    # A=[0,.79], B=[.20,.99], C=[.40,.99]:
    # A/B≈.75, B/C≈.75, A/C≈.50.
    e1 = Experiment(
        "A", "oblique_band", "x", ">", 0.0, 1.0, "A",
        oblique_terms=(("x", 1.0),), oblique_lower=0.0, oblique_upper=0.79,
    )
    e2 = Experiment(
        "B", "oblique_band", "x", ">", 0.0, 1.0, "B",
        oblique_terms=(("x", 1.0),), oblique_lower=0.20, oblique_upper=0.99,
    )
    e3 = Experiment(
        "C", "oblique_band", "x", ">", 0.0, 1.0, "C",
        oblique_terms=(("x", 1.0),), oblique_lower=0.40, oblique_upper=0.99,
    )
    candidates = {e.id: e for e in (e1, e2, e3)}
    observations = [
        _obs(e1.id, e1.describe(), 90.0, 0.21),
        _obs(e2.id, e2.describe(), 85.0, 0.20),
        _obs(e3.id, e3.describe(), 80.0, 0.19),
    ]
    clusters, mapping = cluster_validated_failures(observations, candidates, X, jaccard_threshold=0.60)
    assert len(clusters) == 2
    assert mapping[e1.id] == mapping[e2.id]
    assert mapping[e1.id] != mapping[e3.id]


def test_cluster_records_strategy_diversity():
    import pandas as pd
    from modelxray.investigation.controller import ExperimentObservation
    from modelxray.investigation.failure_clustering import cluster_validated_failures
    from modelxray.investigation.schemas import Experiment

    X = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4], "y": [0.1, 0.2, 0.3, 0.4]})
    e1 = Experiment("E-grid", "subgroup_threshold", "x", ">", 0.0, 1.0, "grid")
    e2 = Experiment("E-int", "interaction", "x", ">", 0.0, 1.0, "interaction", feature_b="y", operator_b=">", threshold_b=0.0)
    o1 = ExperimentObservation(
        experiment_id="E-grid", condition="x > 0", kind="subgroup_threshold", support=1.0,
        subgroup_accuracy=0.7, baseline_accuracy=0.9, gap=0.2, p_value=1e-8, severity="HIGH",
        validated=True, validation_gap=0.2, adjusted_p_value=1e-7, effect_size=0.2,
        effect_ci_low=0.1, effect_ci_high=0.3, holdout_effect_ci_low=0.1,
        holdout_effect_ci_high=0.3, holdout_ci_credible=True, reproducible=True, evidence_score=90.0)
    o2 = ExperimentObservation(
        experiment_id="E-int", condition="x > 0 & y > 0", kind="interaction", support=1.0,
        subgroup_accuracy=0.7, baseline_accuracy=0.9, gap=0.2, p_value=1e-8, severity="HIGH",
        validated=True, validation_gap=0.2, adjusted_p_value=1e-7, effect_size=0.2,
        effect_ci_low=0.1, effect_ci_high=0.3, holdout_effect_ci_low=0.1,
        holdout_effect_ci_high=0.3, holdout_ci_credible=True, reproducible=True, evidence_score=85.0)
    clusters, mapping = cluster_validated_failures([o1, o2], {"E-grid": e1, "E-int": e2}, X, jaccard_threshold=0.60)
    assert len(clusters) == 1
    assert clusters[0].strategy_diversity == 2
    assert set(clusters[0].strategy_kinds) == {"subgroup_threshold", "interaction"}
