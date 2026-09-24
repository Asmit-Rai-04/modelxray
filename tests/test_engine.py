import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from modelxray.core.adapters import SklearnClassifierAdapter
from modelxray.detectors.perturbation import discover_local_instability


def test_perturbation_detector_does_not_crash():
    x = np.linspace(-2, 2, 100)
    X = pd.DataFrame({"x": x})
    y = (x > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    adapter = SklearnClassifierAdapter(model)
    findings = discover_local_instability(adapter, X)
    assert isinstance(findings, list)


def test_hypothesis_generation_is_model_independent():
    from modelxray.investigation.controller import generate_initial_experiments

    X = pd.DataFrame({"a": np.linspace(-1, 1, 100), "b": np.linspace(1, -1, 100)})
    candidates = generate_initial_experiments(X)
    assert candidates
    assert {c.feature for c in candidates} == {"a", "b"}


def test_interaction_generation_produces_conjunctions():
    from modelxray.investigation.controller import generate_interaction_experiments

    rng = np.random.default_rng(31)
    n = 1200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    y = (x1 + x2 > 0).astype(int)
    hidden = (x1 > 1.1) & (x2 < -0.7)
    y_corrupt = y.copy()
    y_corrupt[hidden] ^= 1
    model = LogisticRegression().fit(X.iloc[:800], y_corrupt[:800])
    adapter = SklearnClassifierAdapter(model)
    preds = adapter.predict(X.iloc[800:])
    baseline = float(np.mean(preds == np.asarray(y_corrupt[800:])))
    experiments = generate_interaction_experiments(
        X.iloc[800:].reset_index(drop=True), pd.Series(y_corrupt[800:]), preds, baseline
    )
    assert experiments, "tree-guided generator should propose interaction candidates"
    assert all(e.kind == "interaction" for e in experiments)
    assert all(e.feature_b is not None for e in experiments)


def test_targeted_pairwise_generation_finds_conjunction_candidate():
    from modelxray.investigation.controller import generate_pairwise_interaction_experiments

    rng = np.random.default_rng(123)
    n = 1800
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": rng.normal(size=n), "x4": rng.normal(size=n)})
    y = (x1 + x2 > 0).astype(int)
    hidden = (x1 > np.quantile(x1, 0.70)) & (x2 < np.quantile(x2, 0.25))
    y_eval = y.copy()
    y_eval[hidden] ^= 1
    model = LogisticRegression(max_iter=1000).fit(X.iloc[:1000], y_eval[:1000])
    preds = model.predict(X.iloc[1000:])
    X_eval = X.iloc[1000:].reset_index(drop=True)
    y_eval_frame = pd.Series(y_eval[1000:]).reset_index(drop=True)
    baseline = float(np.mean(preds == y_eval_frame.to_numpy()))
    experiments = generate_pairwise_interaction_experiments(
        X_eval, y_eval_frame, preds, baseline, max_features=4, max_experiments=96
    )
    assert experiments
    assert all(e.kind == "interaction" and e.feature_b is not None for e in experiments)


def test_active_investigation_respects_budget_and_validates_on_holdout():
    from modelxray.investigation.controller import run_active_investigation

    rng = np.random.default_rng(21)
    x1 = rng.normal(size=1600)
    x2 = rng.normal(size=1600)
    y = (x1 + x2 > 0).astype(int)
    hidden = (x1 > 1.1) & (x2 < -0.7)
    y[hidden] ^= 1
    X = pd.DataFrame({"x1": x1, "x2": x2})
    model = LogisticRegression().fit(X.iloc[:1000], y[:1000])
    adapter = SklearnClassifierAdapter(model)

    run = run_active_investigation(
        adapter,
        X.iloc[1000:1300], pd.Series(y[1000:1300]),
        X.iloc[1300:], pd.Series(y[1300:]),
        budget=7,
    )
    assert run.experiments_executed <= 7
    assert run.experiments_considered >= run.experiments_executed
    assert all("validated" in item for item in run.observations)
    assert all("kind" in item for item in run.observations)


def test_counterexample_search_finds_verified_flip():
    from modelxray.investigation.counterexamples import find_minimal_counterexample
    from modelxray.investigation.schemas import Experiment

    x = np.linspace(-2, 2, 400)
    X = pd.DataFrame({"x": x})
    y = (x > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    adapter = SklearnClassifierAdapter(model)

    exp = Experiment("E-x", "subgroup_threshold", "x", ">", 0.5, 1.0, "boundary probe")
    finding = find_minimal_counterexample(adapter, X, exp, row_index=260)
    assert finding is not None
    assert finding.search_status == "approx_minimal_flip"
    assert finding.original_prediction != finding.counterfactual_prediction
    assert finding.absolute_change > 0
    # Declared coordinate system + self-replay validity contract.
    assert finding.frame == "discovery"
    row = X.iloc[[260]].copy()
    row[finding.changed_feature] = finding.counterfactual_value
    assert adapter.predict(row)[0] != finding.original_prediction, (
        "stored counterfactual must actually flip the model on the declared row"
    )


def test_active_run_emits_counterexamples_only_for_validated_failures():
    from modelxray.investigation.controller import run_active_investigation

    rng = np.random.default_rng(99)
    x = rng.normal(size=500)
    X = pd.DataFrame({"x": x, "z": rng.normal(size=500)})
    y = (x > 0).astype(int)
    model = LogisticRegression().fit(X.iloc[:250], y[:250])
    adapter = SklearnClassifierAdapter(model)
    run = run_active_investigation(
        adapter, X.iloc[:250], pd.Series(y[:250]),
        X.iloc[250:], pd.Series(y[250:]), budget=5,
    )
    assert isinstance(run.counterexamples, list)
    # Boundary-sensitivity evidence may only reference validated failures, and
    # every item must declare + honor its frame ("validation" here).
    repro_ids = {o["experiment_id"] for o in run.observations if o["reproducible"]}
    X_validation = X.iloc[250:].reset_index(drop=True)
    for cx in run.counterexamples:
        assert cx["source_experiment_id"] in repro_ids
        assert cx["frame"] == "validation"
        row = X_validation.iloc[[int(cx["source_row"])]].copy()
        row[cx["changed_feature"]] = float(cx["counterfactual_value"])
        assert adapter.predict(row)[0] != cx["original_prediction"], (
            "counterexample must replay against its declared frame"
        )


def test_oblique_band_generation_produces_linear_region_candidate():
    from modelxray.investigation.controller import generate_oblique_band_experiments

    rng = np.random.default_rng(44)
    n = 1800
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(1, 7)})
    score = X.sum(axis=1)
    y = (score > np.median(score)).astype(int)
    # Make a narrow diagonal failure band around the score boundary.
    band = np.abs(score - np.median(score)) < np.quantile(np.abs(score - np.median(score)), 0.10)
    y_eval = y.copy()
    y_eval[band.to_numpy()] ^= 1
    model = LogisticRegression(max_iter=1000).fit(X.iloc[:1000], y_eval[:1000])
    preds = model.predict(X.iloc[1000:])
    baseline = float(np.mean(preds == y_eval[1000:]))
    exps = generate_oblique_band_experiments(
        X.iloc[1000:].reset_index(drop=True),
        pd.Series(y_eval[1000:]),
        preds,
        baseline,
        max_features=6,
        max_experiments=20,
    )
    assert exps
    assert all(e.kind == "oblique_band" for e in exps)
    assert all(len(e.oblique_terms) >= 2 for e in exps)
    assert all(e.oblique_lower is not None and e.oblique_upper is not None for e in exps)


def test_balanced_accuracy_active_run_uses_metric_specific_inference():
    from modelxray.investigation.controller import run_active_investigation

    rng = np.random.default_rng(202)
    x = rng.normal(size=900)
    z = rng.normal(size=900)
    y = np.where(x > 1.1, 1, 0)
    # Create an imbalanced evaluation population with a localized minority fault.
    y[(x > 1.1) & (z > 0.8)] = 0
    X = pd.DataFrame({"x": x, "z": z})
    model = LogisticRegression().fit(X.iloc[:500], y[:500])
    adapter = SklearnClassifierAdapter(model)
    run = run_active_investigation(
        adapter, X.iloc[500:700], pd.Series(y[500:700]),
        X.iloc[700:], pd.Series(y[700:]), budget=8, metric="balanced_accuracy"
    )
    assert all("adjusted_p_value" in row for row in run.observations)
    # The returned score must be based on balanced accuracy, not an accuracy-only
    # effect-size shortcut.
    assert run.metric == "balanced_accuracy"


def test_stable_seed_does_not_depend_on_python_hash_randomization():
    from modelxray.investigation.controller import _stable_seed
    assert _stable_seed("E-x-1") == _stable_seed("E-x-1")
    assert _stable_seed("E-x-1") != _stable_seed("E-x-2")


def test_categorical_generation_discovers_category_subgroup():
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from modelxray.investigation.controller import generate_categorical_experiments
    from modelxray.investigation.schemas import mask_for_experiment

    rng = np.random.default_rng(55)
    n = 900
    frame = pd.DataFrame({
        "segment": rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.3, 0.3]),
        "age": rng.normal(35, 8, size=n),
    })
    y_clean = ((frame["age"] > 35).astype(int)).to_numpy()
    eval_model = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["segment"]),
            ("num", "passthrough", ["age"]),
        ])),
        ("clf", LogisticRegression(max_iter=500)),
    ])
    train = frame.iloc[:500].copy()
    eval_frame = frame.iloc[500:].reset_index(drop=True)
    eval_y = y_clean[500:].copy()
    # Seed a real categorical failure: labels in segment B are corrupted while
    # the model itself was trained on clean labels.
    eval_y[eval_frame["segment"].to_numpy() == "B"] ^= 1
    eval_model.fit(train, y_clean[:500])
    preds = eval_model.predict(eval_frame)
    baseline = float(np.mean(preds == eval_y))

    experiments = generate_categorical_experiments(
        eval_frame, pd.Series(eval_y), preds, baseline, max_experiments=20
    )
    assert experiments
    assert all(e.kind == "categorical_subgroup" for e in experiments)
    b_candidates = [e for e in experiments if e.feature == "segment" and e.category == "B"]
    assert b_candidates, "expected category B candidate"
    b_mask = mask_for_experiment(b_candidates[0], eval_frame)
    assert b_mask.mean() > 0.15


def test_active_investigation_can_validate_categorical_failure():
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from modelxray.investigation.controller import run_active_investigation

    rng = np.random.default_rng(56)
    n = 1400
    frame = pd.DataFrame({
        "segment": rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.3, 0.3]),
        "age": rng.normal(35, 8, size=n),
        "income": rng.normal(60, 12, size=n),
    })
    y_clean = ((frame["age"] + 0.4 * frame["income"] > 60).astype(int)).to_numpy()
    train = frame.iloc[:800].copy()
    eval_frame = frame.iloc[800:].reset_index(drop=True)
    eval_y = y_clean[800:].copy()
    eval_y[eval_frame["segment"].to_numpy() == "B"] ^= 1

    model = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["segment"]),
            ("num", "passthrough", ["age", "income"]),
        ])),
        ("clf", LogisticRegression(max_iter=600)),
    ]).fit(train, y_clean[:800])
    adapter = SklearnClassifierAdapter(model)

    discovery = eval_frame.iloc[:300].reset_index(drop=True)
    validation = eval_frame.iloc[300:].reset_index(drop=True)
    y_disc = pd.Series(eval_y[:300])
    y_val = pd.Series(eval_y[300:])

    run = run_active_investigation(
        adapter,
        discovery,
        y_disc,
        validation,
        y_val,
        budget=12,
        metric="accuracy",
    )
    categorical = [o for o in run.observations if o["kind"] == "categorical_subgroup"]
    assert categorical, "categorical family should receive an investigation slot"
    validated = [o for o in categorical if o["reproducible"]]
    assert validated, "seeded categorical failure should validate on holdout"


def test_categorical_only_dataset_is_supported():
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from modelxray.investigation.controller import run_active_investigation

    rng = np.random.default_rng(57)
    n = 700
    frame = pd.DataFrame({
        "plan": rng.choice(["basic", "plus", "pro"], size=n, p=[0.45, 0.35, 0.20]),
    })
    y_clean = (frame["plan"].isin(["plus", "pro"])).astype(int).to_numpy()
    model = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["plan"]),
        ])),
        ("clf", LogisticRegression(max_iter=400)),
    ]).fit(frame.iloc[:400], y_clean[:400])

    y_eval = y_clean[400:].copy()
    basic = frame.iloc[400:]["plan"].to_numpy() == "basic"
    y_eval[basic] ^= 1
    discovery = frame.iloc[400:550].reset_index(drop=True)
    validation = frame.iloc[550:].reset_index(drop=True)
    run = run_active_investigation(
        SklearnClassifierAdapter(model),
        discovery, pd.Series(y_eval[:150]),
        validation, pd.Series(y_eval[150:]),
        budget=6,
    )
    assert run.experiments_considered > 0
    assert any(o["kind"] == "categorical_subgroup" for o in run.observations)


def test_search_coverage_is_explicit_and_does_not_claim_global_coverage():
    from modelxray.investigation.controller import run_active_investigation

    rng = np.random.default_rng(303)
    X = pd.DataFrame({
        "x1": rng.normal(size=900),
        "x2": rng.normal(size=900),
        "x3": rng.normal(size=900),
    })
    y = (X["x1"] + X["x2"] > 0).astype(int).to_numpy()
    model = LogisticRegression(max_iter=1000).fit(X.iloc[:500], y[:500])
    run = run_active_investigation(
        SklearnClassifierAdapter(model),
        X.iloc[500:700].reset_index(drop=True), pd.Series(y[500:700]).reset_index(drop=True),
        X.iloc[700:].reset_index(drop=True), pd.Series(y[700:]).reset_index(drop=True),
        budget=6,
    )

    coverage = run.search_coverage
    assert coverage is not None
    assert coverage["scope"] == "generated_candidate_pool"
    assert "full model behavior space" in coverage["interpretation"]
    assert coverage["executed"] == run.experiments_executed
    assert coverage["candidate_pool_size"] == run.candidate_pool_size
    assert coverage["unexplored_candidates"] == max(0, run.candidate_pool_size - run.experiments_executed)
    assert 0.0 <= coverage["candidate_execution_ratio"] <= 1.0
    assert 0.0 <= coverage["family_breadth_ratio"] <= 1.0
    assert set(coverage["families"]).issubset({"grid", "categorical", "interaction", "oblique_band", "prototype_region"})


def test_counterexample_search_batches_model_calls():
    from modelxray.investigation.counterexamples import find_minimal_counterexample
    from modelxray.investigation.schemas import Experiment
    from modelxray.core.adapters import CountingClassifierAdapter

    x = np.linspace(-2, 2, 500)
    X = pd.DataFrame({f"x{i}": x + i * 0.01 for i in range(6)})
    y = (x > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    adapter = CountingClassifierAdapter(SklearnClassifierAdapter(model))
    exp = Experiment("E-x", "subgroup_threshold", "x0", ">", 0.5, 1.0, "boundary probe")
    finding = find_minimal_counterexample(adapter, X, exp, row_index=280)
    assert finding is not None
    # One initial prediction + one batched grid call per feature + one batched
    # refinement call per iteration + one batched endpoint verification.
    assert adapter.predict_calls <= 1 + 6 + 16 + 1 + 6


def test_cost_aware_planner_uses_deterministic_family_cost():
    from modelxray.investigation.controller import ExperimentObservation, _rank_key
    from modelxray.investigation.schemas import Experiment

    candidate = Experiment(
        id="EI-x1-x2-1", kind="interaction", feature="x1", operator=">", threshold=1.0,
        feature_b="x2", operator_b="<", threshold_b=0.0, priority=1.0, rationale="test",
    )
    grid = ExperimentObservation(
        experiment_id="EG-x0-1", condition="x0 > 1", kind="subgroup_threshold",
        support=0.2, subgroup_accuracy=0.8, baseline_accuracy=0.9, gap=0.1, p_value=0.01,
        severity="medium", validated=False, validation_gap=0.0, evaluation_ms=1.0,
    )
    interaction = ExperimentObservation(
        experiment_id="EI-fast", condition="x1 > 1 & x2 < 0", kind="interaction",
        support=0.1, subgroup_accuracy=0.8, baseline_accuracy=0.9, gap=0.1, p_value=0.01,
        severity="medium", validated=False, validation_gap=0.0, evaluation_ms=100.0,
    )
    score_cost_aware = _rank_key(candidate, [grid, interaction], cost_aware=True)[0]
    score_unadjusted = _rank_key(candidate, [grid, interaction], cost_aware=False)[0]
    assert score_cost_aware < score_unadjusted


def test_cost_aware_ranking_ignores_wall_clock_jitter():
    from modelxray.investigation.controller import ExperimentObservation, _rank_key
    from modelxray.investigation.schemas import Experiment

    candidate = Experiment(
        id="EI-x1-x2-1", kind="interaction", feature="x1", operator=">", threshold=1.0,
        feature_b="x2", operator_b="<", threshold_b=0.0, priority=1.0, rationale="test",
    )
    base = dict(
        experiment_id="EG-x0-1", condition="x0 > 1", kind="subgroup_threshold",
        support=0.2, subgroup_accuracy=0.8, baseline_accuracy=0.9, gap=0.1, p_value=0.01,
        severity="medium", validated=False, validation_gap=0.0,
    )
    a = ExperimentObservation(**base, evaluation_ms=1.0)
    b = ExperimentObservation(**base, evaluation_ms=9999.0)
    assert _rank_key(candidate, [a], cost_aware=True) == _rank_key(candidate, [b], cost_aware=True)
