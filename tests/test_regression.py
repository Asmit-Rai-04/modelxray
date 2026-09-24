from modelxray.investigation.regression import build_regression_summary, compare_failure_snapshots


def _obs(eid, gap, score, reproducible=True, condition=None):
    return {
        "experiment_id": eid,
        "condition": condition or eid,
        "severity": "HIGH" if gap >= 0.15 else "MEDIUM",
        "gap": gap,
        "support": 0.12,
        "adjusted_p_value": 0.01,
        "validation_gap": gap * 0.9,
        "evidence_score": score,
        "reproducible": reproducible,
    }


def test_compare_classifies_fixed_persistent_and_new():
    v1 = [_obs("E-a", .20, 80), _obs("E-b", .10, 60), _obs("E-only-v1", .16, 72)]
    v2 = [_obs("E-a", .06, 35, reproducible=False), _obs("E-b", .11, 61), _obs("E-new", .18, 77)]
    result = compare_failure_snapshots(v1, v2)
    statuses = {d.experiment_id_v1 or d.experiment_id_v2: d.status for d in result}
    assert statuses["E-a"] == "FIXED"
    assert statuses["E-b"] == "PERSISTENT"
    assert statuses["E-only-v1"] == "FIXED"
    assert statuses["E-new"] == "NEW"


def test_regression_summary_counts_correctly():
    v1 = [_obs("E-a", .2, 80), _obs("E-b", .1, 60)]
    v2 = [_obs("E-b", .11, 61), _obs("E-c", .18, 77)]
    deltas = compare_failure_snapshots(v1, v2)
    summary = build_regression_summary(deltas)
    assert summary["fixed"] == 1
    assert summary["persistent"] == 1
    assert summary["new"] == 1
    assert summary["failures_v1"] == 2
    assert summary["failures_v2"] == 2
    assert summary["net_failure_change"] == 0


def test_regression_prefers_failure_cluster_matching_over_experiment_ids():
    v1 = [
        _obs("v1-e1", .20, 80, condition="x > 1.0"),
        _obs("v1-e2", .18, 76, condition="x > 1.1"),
    ]
    v2 = [
        _obs("v2-new-id", .19, 82, condition="x > 1.01"),
    ]
    for obs in v1:
        obs["failure_cluster_id"] = "FC-001"
    v2[0]["failure_cluster_id"] = "FC-007"

    deltas = compare_failure_snapshots(v1, v2)
    assert len(deltas) == 1
    assert deltas[0].status == "PERSISTENT"
    assert deltas[0].match_type == "cluster_overlap"
    assert deltas[0].failure_cluster_id_v1 == "FC-001"
    assert deltas[0].failure_cluster_id_v2 == "FC-007"
