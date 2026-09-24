import numpy as np
from modelxray.validation.reliability import adjust_p_values, evidence_score, metric_effect_and_pvalue, two_proportion_effect_and_ci


def test_bh_adjustment_controls_false_discoveries():
    result = adjust_p_values([0.001, 0.01, 0.2, 0.8])
    assert len(result) == 4
    assert result[0][0] <= result[1][0]
    assert result[0][1] is True
    assert result[3][1] is False


def test_effect_size_confidence_interval_contains_difference():
    # v1.0.1 sign convention: effect = complement_score - subgroup_score
    # (positive = subgroup is worse).
    effect, low, high = two_proportion_effect_and_ci(70, 100, 450, 500)
    assert np.isclose(effect, 0.90 - 0.70)
    assert low < effect < high


def test_evidence_score_is_bounded_and_increases_with_validation():
    weak = evidence_score(adjusted_p_value=0.2, support=0.05, effect_gap=0.05, validation_gap=0.0)
    strong = evidence_score(adjusted_p_value=0.001, support=0.2, effect_gap=0.2, validation_gap=0.15, counterexample_verified=True)
    assert 0 <= weak <= 100
    assert 0 <= strong <= 100
    assert strong > weak


def test_balanced_accuracy_effect_uses_stratified_inference():
    y_sub = np.array([0, 0, 1, 1, 1, 0])
    p_sub = np.array([0, 0, 0, 1, 0, 0])
    y_comp = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    p_comp = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    effect, p, low, high = metric_effect_and_pvalue(p_sub, y_sub, p_comp, y_comp, "balanced_accuracy", seed=7, permutations=199, bootstrap=199)
    assert effect > 0
    assert 0 <= p <= 1
    assert low < effect < high


def test_balanced_accuracy_inference_differs_from_raw_accuracy_when_classes_are_imbalanced():
    y_sub = np.array([0] * 9 + [1])
    p_sub = np.array([0] * 10)
    y_comp = np.array([0] * 45 + [1] * 5)
    p_comp = np.array([0] * 45 + [1] * 5)
    effect, _, _, _ = metric_effect_and_pvalue(p_sub, y_sub, p_comp, y_comp, "balanced_accuracy", seed=3, permutations=99, bootstrap=99)
    assert effect > 0.4
