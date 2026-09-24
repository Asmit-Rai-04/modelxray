from pathlib import Path
import io

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def test_upload_and_investigate_real_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from modelxray.core.ingestion import AssetStore
    import modelxray.main as main

    main.asset_store = AssetStore(tmp_path / "assets")
    main.evidence_store = main.EvidenceStore(tmp_path / "evidence.db")
    client = TestClient(main.app)

    X, y = make_classification(n_samples=400, n_features=6, n_informative=4, random_state=7)
    columns = [f"feature_{i+1}" for i in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=columns)
    frame["target"] = y
    model = RandomForestClassifier(n_estimators=50, random_state=7).fit(frame[columns], y)

    model_buf = io.BytesIO()
    joblib.dump(model, model_buf)
    model_buf.seek(0)
    response = client.post(
        "/api/v1/assets/model",
        files={"file": ("credit.joblib", model_buf, "application/octet-stream")},
    )
    assert response.status_code == 200, response.text
    model_id = response.json()["asset_id"]

    dataset_buf = io.BytesIO(frame.to_csv(index=False).encode())
    response = client.post(
        "/api/v1/assets/dataset",
        files={"file": ("credit.csv", dataset_buf, "text/csv")},
    )
    assert response.status_code == 200, response.text
    dataset_id = response.json()["asset_id"]

    response = client.post(
        "/api/v1/investigate/uploaded",
        json={
            "model_id": model_id,
            "dataset_id": dataset_id,
            "target_column": "target",
            "budget": 8,
            "random_state": 42,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["dataset"]["source_dataset_id"] == dataset_id
    assert payload["dataset"]["target_column"] == "target"
    assert payload["experiment_count"] == 8
    assert payload["baseline"]["accuracy"] >= 0.70
