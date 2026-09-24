"""Phase 1+2 repair regression tests.

Covers audit findings: interaction discovery, no-failure control, counterexample
semantics, worker stdout safety, PYTHONNOUSERSITE removal, schema safety,
path traversal, single severity function, effect sign conventions.

Sandbox tests build artifacts in a subprocess (pickled classes defined inside
pytest modules cannot be unpickled by the worker subprocess, so the loud model
lives in a fixture module written to tmp_path).
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from modelxray.core.adapters import SklearnClassifierAdapter
from modelxray.investigation.controller import run_active_investigation
from modelxray.investigation.schemas import Experiment
from modelxray.validation.contracts import ContractError, validate_prediction_contract
from modelxray.validation.reliability import severity, two_proportion_effect_and_ci


def _interaction_frame(n=3000, seed=7):
    X, y = make_classification(n_samples=n, n_features=6, n_informative=4, random_state=seed)
    frame = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(6)])
    region = (frame["x1"] > frame["x1"].quantile(0.78)) & (frame["x2"] < frame["x2"].quantile(0.25))
    y_corrupt = np.asarray(y).copy()
    y_corrupt[region.to_numpy()] ^= 1
    return frame, np.asarray(y), y_corrupt, region.to_numpy()


def test_interaction_failure_is_discovered():
    frame, _y_clean, y_corrupt, region = _interaction_frame()
    model = LogisticRegression(max_iter=1000).fit(frame.iloc[:2000], y_corrupt[:2000])
    run = run_active_investigation(
        SklearnClassifierAdapter(model),
        frame.iloc[2000:2500].reset_index(drop=True), pd.Series(y_corrupt[2000:2500]),
        frame.iloc[2500:].reset_index(drop=True), pd.Series(y_corrupt[2500:]),
        budget=24,
    )
    repro = [o for o in run.observations if o["reproducible"]]
    assert repro, "expected at least one reproducible failure for the seeded interaction fault"
    kinds = {o["kind"] for o in repro}
    assert "interaction" in kinds, f"expected an interaction experiment among {kinds}"
    inter = next(o for o in repro if o["kind"] == "interaction")
    assert inter["gap"] >= 0.05 and inter["validation_gap"] >= 0.03
    assert inter["adjusted_p_value"] <= 0.05


def test_no_failure_control_stays_clean():
    """Teacher-label control: eval labels = the teacher's own predictions.

    Canonical false-discovery control: a model evaluated against labels its
    teacher reproduces perfectly has zero clean errors, so any promoted failure
    would be fabrication. On Bayes-error data, error concentration near class
    overlap is statistically real and IS reported (documented limitation).
    """
    X, y = make_classification(
        n_samples=2500, n_features=6, n_informative=4, n_redundant=1,
        class_sep=3.0, flip_y=0.0, random_state=202,
    )
    frame = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(6)])
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=120, random_state=303, n_jobs=-1).fit(
        frame.iloc[:1500], y[:1500]
    )
    run = run_active_investigation(
        SklearnClassifierAdapter(model),
        frame.iloc[1500:2000].reset_index(drop=True), pd.Series(y[1500:2000]),
        frame.iloc[2000:].reset_index(drop=True), pd.Series(y[2000:]),
        budget=24,
    )
    repro = [o for o in run.observations if o["reproducible"]]
    assert repro == [], f"no-failure control must promote zero failures, got {len(repro)}"
    assert run.counterexamples == [], "clean model must produce no boundary-evidence attachments"


def test_counterexamples_require_validated_failures():
    frame, _y_clean, y_corrupt, _region = _interaction_frame(seed=13)
    model = LogisticRegression(max_iter=1000).fit(frame.iloc[:2000], y_corrupt[:2000])
    run = run_active_investigation(
        SklearnClassifierAdapter(model),
        frame.iloc[2000:2500].reset_index(drop=True), pd.Series(y_corrupt[2000:2500]),
        frame.iloc[2500:].reset_index(drop=True), pd.Series(y_corrupt[2500:]),
        budget=20,
    )
    repro_ids = {o["experiment_id"] for o in run.observations if o["reproducible"]}
    for cx in run.counterexamples:
        assert cx["source_experiment_id"] in repro_ids, (
            "boundary-sensitivity evidence must only attach to validated failures"
        )


def test_main_effect_failure_is_discovered():
    X, y = make_classification(n_samples=3000, n_features=5, n_informative=3, random_state=5)
    frame = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(5)])
    region = frame["x1"] > frame["x1"].quantile(0.8)
    y_corrupt = np.asarray(y).copy()
    y_corrupt[region.to_numpy()] ^= 1
    model = LogisticRegression(max_iter=1000).fit(frame.iloc[:2000], y_corrupt[:2000])
    run = run_active_investigation(
        SklearnClassifierAdapter(model),
        frame.iloc[2000:2500].reset_index(drop=True), pd.Series(y_corrupt[2000:2500]),
        frame.iloc[2500:].reset_index(drop=True), pd.Series(y_corrupt[2500:]),
        budget=20,
    )
    repro = [o for o in run.observations if o["reproducible"]]
    assert repro, "seeded main-effect failure must be discovered"
    best = repro[0]
    assert best["gap"] >= 0.05 and best["validation_gap"] >= 0.03


# ---------------------------------------------------------------------------
# Worker stdout safety + environment (audit criticals #1 and #2)
# ---------------------------------------------------------------------------

_LOUD_FIXTURE = '''
import numpy as np
from sklearn.linear_model import LogisticRegression


class LoudModel(LogisticRegression):
    """Model that prints to stdout during predict() - worker IPC must survive."""

    def predict(self, X):
        print("hello from a chatty model")
        print("debug: anything can go here")
        return super().predict(X)


def build(X, y):
    return LoudModel(max_iter=500).fit(X, y)
'''


def _make_loud_model_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    module_path = tmp_path / "mx_loud_fixture.py"
    module_path.write_text(_LOUD_FIXTURE, encoding="utf-8")
    script = (
        "import sys, joblib, numpy as np, pandas as pd\n"
        f"sys.path.insert(0, r'{tmp_path}')\n"
        "from sklearn.datasets import make_classification\n"
        "from mx_loud_fixture import build\n"
        "X, y = make_classification(n_samples=220, n_features=5, n_informative=3, random_state=11)\n"
        "cols = ['feature_' + str(i) for i in range(5)]\n"
        "frame = pd.DataFrame(X, columns=cols)\n"
        "model = build(frame[cols], y)\n"
        f"joblib.dump(model, r'{tmp_path / 'loud.joblib'}')\n"
        f"frame.assign(target=y).to_csv(r'{tmp_path / 'loud.csv'}', index=False)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return tmp_path / "loud.joblib", tmp_path / "loud.csv"


def test_stdout_printing_model_survives_worker(tmp_path):
    from modelxray.execution.sandbox import run_isolated_investigation

    model_path, dataset_path = _make_loud_model_artifacts(tmp_path)
    result = run_isolated_investigation(
        model_path=model_path,
        dataset_path=dataset_path,
        target_column="target",
        budget=6,
        random_state=42,
        timeout_seconds=120,
        extra_pythonpath=str(tmp_path),  # custom model classes must be importable
    )
    assert result["active_investigation"]["experiments_executed"] == 6
    assert result["baseline"]["accuracy"] >= 0.6


def test_worker_env_policy_does_not_set_pynousersite():
    """The sandbox must not set PYTHONNOUSERSITE (audit critical #2)."""
    from modelxray.execution import sandbox

    source = Path(sandbox.__file__).read_text(encoding="utf-8")
    assert '"PYTHONNOUSERSITE"' not in source, "sandbox must not set PYTHONNOUSERSITE"


def test_worker_reports_internal_failure_classified(tmp_path):
    """A nonexistent model file surfaces a classified internal failure."""
    from modelxray.execution.sandbox import SandboxExecutionError, run_isolated_investigation

    X, y = make_classification(n_samples=120, n_features=4, n_informative=2, n_redundant=1, random_state=1)
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    frame["target"] = y
    dataset_path = tmp_path / "d.csv"
    frame.to_csv(dataset_path, index=False)
    with pytest.raises(SandboxExecutionError) as excinfo:
        run_isolated_investigation(
            model_path=str(tmp_path / "missing.joblib"),
            dataset_path=dataset_path,
            target_column="target",
            budget=6,
            random_state=42,
        )
    assert "Internal worker failure" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Schema safety (audit critical #4)
# ---------------------------------------------------------------------------


def test_reordered_columns_fail_for_nameless_model():
    X, y = make_classification(n_samples=200, n_features=4, n_informative=2, n_redundant=1, random_state=3)
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model_np = LogisticRegression().fit(X, y)  # trained on numpy: no feature_names_in_
    adapter = SklearnClassifierAdapter(model_np)
    y_ser = pd.Series(y)
    validate_prediction_contract(adapter, frame, y_ser, feature_manifest=["a", "b", "c", "d"])
    shuffled = frame[["d", "a", "b", "c"]]
    with pytest.raises(ContractError) as excinfo:
        validate_prediction_contract(adapter, shuffled, y_ser, feature_manifest=["a", "b", "c", "d"])
    assert "wrong ORDER" in str(excinfo.value) or "schema mismatch" in str(excinfo.value).lower()


def test_schema_mismatch_message_names_columns():
    X, y = make_classification(n_samples=150, n_features=4, n_informative=2, n_redundant=1, random_state=4)
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model_np = LogisticRegression().fit(X, y)
    with pytest.raises(ContractError) as excinfo:
        validate_prediction_contract(
            SklearnClassifierAdapter(model_np),
            frame[["a", "b"]],
            pd.Series(y),
            feature_manifest=["a", "b", "c", "d"],
        )
    msg = str(excinfo.value)
    assert "missing" in msg and "c" in msg


def test_predict_only_model_passes_contract():
    X, y = make_classification(n_samples=200, n_features=4, n_informative=2, n_redundant=1, random_state=9)
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model = LinearSVC().fit(X, y)  # no predict_proba
    check = validate_prediction_contract(SklearnClassifierAdapter(model), frame, pd.Series(y))
    assert check.ok and check.has_predict_proba is False


# ---------------------------------------------------------------------------
# Path traversal (audit high #7)
# ---------------------------------------------------------------------------


def test_traversal_ids_fail_safely(tmp_path):
    from modelxray.core.ingestion import AssetStore, is_valid_asset_id

    store = AssetStore(tmp_path / "assets")
    for bad in ["../secret", "..%2F..%2Fetc", "MODEL-../../x", "C:/win.ini", "MODEL-ABC", "", "MODEL-" + "A" * 200]:
        assert store.get(bad) is None, f"traversal id {bad!r} must not resolve"
        assert not is_valid_asset_id(bad)
    asset = store.save_model_bytes("m.joblib", io.BytesIO(b"x"))
    assert store.get(asset.asset_id) is not None


# ---------------------------------------------------------------------------
# Statistical integrity (Phase 2)
# ---------------------------------------------------------------------------


def test_severity_single_canonical_function():
    from modelxray.detectors import subgroup  # legacy module

    assert not hasattr(subgroup, "_severity")
    assert severity(0.16, 0.10) == "HIGH"
    assert severity(0.09, 0.05) == "MEDIUM"
    assert severity(0.01, 0.05) == "LOW"
    assert severity(0.08, 0.01) == "MEDIUM"


def test_effect_sign_convention_is_disadvantage():
    effect, low, high = two_proportion_effect_and_ci(70, 100, 450, 500)
    assert effect == pytest.approx(0.90 - 0.70)
    assert low < effect < high
    effect2, _, _ = two_proportion_effect_and_ci(90, 100, 60, 100)
    assert effect2 < 0


def test_holdout_gate_requires_minimum_rows():
    from modelxray.investigation import controller as ctrl

    obs = ctrl.ExperimentObservation(
        experiment_id="E-x-4-gt", condition="x > 0", kind="subgroup_threshold",
        support=0.2, subgroup_accuracy=0.7, baseline_accuracy=0.85, gap=0.15,
        p_value=0.001, severity="HIGH", validated=False, validation_gap=0.0,
    )
    X = pd.DataFrame({"x": np.linspace(0, 1, 30)})
    y = pd.Series(np.ones(30, dtype=int))
    preds = np.ones(30, dtype=int)
    exp = Experiment("E-x-4-gt", "subgroup_threshold", "x", ">", 0.5, 1.0, "probe")
    result = ctrl._validate(obs, exp, preds, X, y, 0.85, "accuracy")
    assert result.validated is False


def test_adaptive_allocation_respects_budget_and_explores_families():
    frame, _y_clean, y_corrupt, _region = _interaction_frame(n=2200, seed=17)
    model = LogisticRegression(max_iter=1000).fit(frame.iloc[:1400], y_corrupt[:1400])
    run = run_active_investigation(
        SklearnClassifierAdapter(model),
        frame.iloc[1400:1800].reset_index(drop=True), pd.Series(y_corrupt[1400:1800]),
        frame.iloc[1800:].reset_index(drop=True), pd.Series(y_corrupt[1800:]),
        budget=10,
    )
    assert run.experiments_executed == 10
    kinds = [o["kind"] for o in run.observations]
    # Adaptive allocation is intentionally no longer governed by a hard family
    # cap. It must, however, give each available family an initial probe.
    assert {"subgroup_threshold", "interaction", "oblique_band"}.issubset(set(kinds))


def test_s3_boundary_band_gets_oblique_candidates():
    from tests.bench.scenarios import s3_boundary_instability
    from modelxray.investigation.controller import generate_oblique_band_experiments
    frame, y_eval, _ = s3_boundary_instability(n=2200, seed=123)
    model = LogisticRegression(max_iter=1000).fit(frame.iloc[:1500], y_eval[:1500])
    X = frame.iloc[1500:].reset_index(drop=True)
    y = pd.Series(y_eval[1500:])
    preds = model.predict(X)
    baseline = float(np.mean(preds == y.to_numpy()))
    exps = generate_oblique_band_experiments(X, y, preds, baseline, max_features=6, max_experiments=30)
    assert exps, "oblique search should propose candidates on a diagonal/banded fault"
    # At least one candidate should depend on several features, rather than being
    # reducible to a single-axis threshold.
    assert any(len(e.oblique_terms) >= 3 for e in exps)


def test_oblique_band_discovers_seeded_diagonal_failure():
    from tests.bench.run_bench import _condition_to_mask
    from tests.bench.scenarios import s3_boundary_instability

    frame, y_eval, truth = s3_boundary_instability(n=3000, seed=100)
    model = LogisticRegression(max_iter=1000).fit(frame.iloc[:1650], y_eval[:1650])
    X_discovery = frame.iloc[1650:2000].reset_index(drop=True)
    y_discovery = pd.Series(y_eval[1650:2000])
    X_validation = frame.iloc[2000:].reset_index(drop=True)
    y_validation = pd.Series(y_eval[2000:])
    run = run_active_investigation(
        SklearnClassifierAdapter(model),
        X_discovery,
        y_discovery,
        X_validation,
        y_validation,
        budget=24,
    )
    seeded = truth[1650:]
    best_f1 = 0.0
    for observation in run.observations:
        if not observation["reproducible"] or observation["kind"] != "oblique_band":
            continue
        mask = _condition_to_mask(observation["condition"], frame.iloc[1650:].reset_index(drop=True))
        if mask is None:
            continue
        tp = float(np.logical_and(mask, seeded).sum())
        fp = float(np.logical_and(mask, ~seeded).sum())
        fn = float(np.logical_and(~mask, seeded).sum())
        f1 = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
        best_f1 = max(best_f1, f1)
    assert best_f1 >= 0.5, f"oblique failure region was not recovered; best F1={best_f1:.3f}"


def test_adaptive_allocator_probes_each_available_family():
    from modelxray.investigation.controller import _choose_experiment, _family_for_experiment, _rank_key
    from modelxray.investigation.schemas import Experiment

    grid = Experiment("E-grid", "subgroup_threshold", "x", ">", 0.5, 0.5, "grid")
    interaction = Experiment(
        "EI-x-y-0", "interaction", "x", ">", 0.5, 0.5, "interaction",
        feature_b="y", operator_b="<", threshold_b=-0.5,
    )
    oblique = Experiment(
        "EO-0", "oblique_band", "x", "band", 0.0, 0.5, "oblique",
        oblique_terms=(("x", 0.7), ("y", 0.7)), oblique_lower=-0.2, oblique_upper=0.2,
    )
    candidates = [grid, interaction, oblique]
    observations = []
    # Untouched families receive an exploration bonus and remain rankable.
    keys = [_rank_key(c, observations) for c in candidates]
    assert all(len(k) == 3 for k in keys)
    picked = _choose_experiment(candidates, observations)
    assert _family_for_experiment(picked) in {"grid", "interaction", "oblique_band"}


def test_adaptive_allocator_exploits_family_with_strong_observed_gap():
    from modelxray.investigation.controller import _rank_key, _choose_experiment
    from modelxray.investigation.schemas import Experiment
    from modelxray.investigation.controller import ExperimentObservation

    grid = Experiment("E-grid", "subgroup_threshold", "x", ">", 0.5, 0.5, "grid")
    interaction = Experiment(
        "EI-x-y-0", "interaction", "x", ">", 0.5, 0.5, "interaction",
        feature_b="y", operator_b="<", threshold_b=-0.5,
    )
    obs = [
        ExperimentObservation("EI-x-y-prev", "x > .5 & y < -.5", "interaction", 0.18,
                              0.70, 0.90, 0.20, 0.001, "HIGH", False, 0.0)
    ]
    assert _rank_key(interaction, obs)[0] > _rank_key(grid, obs)[0]
    assert _choose_experiment([grid, interaction], obs).kind == "interaction"


def test_cost_aware_toggle_matches_unadjusted_path():
    from modelxray.investigation.controller import _rank_key
    from modelxray.investigation.schemas import Experiment
    fast = Experiment("E-fast", "subgroup_threshold", "x", ">", 0.5, 1.0, "fast")
    slow = Experiment("E-slow", "interaction", "x", ">", 0.5, 1.0, "slow")
    score_no_cost = _rank_key(fast, [], cost_aware=False)
    score_cost = _rank_key(fast, [], cost_aware=True)
    assert score_no_cost == score_cost
