from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def test_isolated_worker_runs_investigation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from modelxray.execution.sandbox import run_isolated_investigation

    X, y = make_classification(n_samples=220, n_features=5, n_informative=3, random_state=11)
    columns = [f"feature_{i}" for i in range(5)]
    frame = pd.DataFrame(X, columns=columns)
    frame["target"] = y
    model = RandomForestClassifier(n_estimators=20, random_state=11).fit(frame[columns], y)
    model_path = tmp_path / "model.joblib"
    dataset_path = tmp_path / "dataset.csv"
    joblib.dump(model, model_path)
    frame.to_csv(dataset_path, index=False)

    result = run_isolated_investigation(
        model_path=model_path,
        dataset_path=dataset_path,
        target_column="target",
        budget=6,
        random_state=42,
        timeout_seconds=60,
    )
    assert result["active_investigation"]["experiments_executed"] == 6
    assert result["baseline"]["accuracy"] >= 0.6


def test_dataset_row_limit_is_enforced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from modelxray.core.ingestion import AssetError, AssetStore
    import io

    store = AssetStore(tmp_path / "assets")
    monkeypatch.setenv("MODELXRAY_MAX_DATASET_ROWS", "3")
    frame = pd.DataFrame({"a": range(4), "b": range(4)})
    try:
        store.save_dataset_bytes("too-many.csv", io.BytesIO(frame.to_csv(index=False).encode()))
    except AssetError as exc:
        assert "row limit" in str(exc)
    else:
        raise AssertionError("Expected dataset row limit to reject the upload")
