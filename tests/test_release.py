import io
from pathlib import Path
import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


def _client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import modelxray.main as main
    from modelxray.core.ingestion import AssetStore
    main.asset_store = AssetStore(tmp_path / "assets")
    main.evidence_store = main.EvidenceStore(tmp_path / "evidence.db")
    return main, TestClient(main.app)


def _upload_model(client, model, name):
    buf = io.BytesIO(); joblib.dump(model, buf); buf.seek(0)
    r = client.post("/api/v1/assets/model", files={"file": (name, buf, "application/octet-stream")})
    assert r.status_code == 200, r.text
    return r.json()["asset_id"]


def test_investigation_history_and_complete_detail(tmp_path, monkeypatch):
    main, client = _client(tmp_path, monkeypatch)
    X, y = make_classification(n_samples=260, n_features=5, random_state=19)
    cols = [f"f{i}" for i in range(5)]
    frame = pd.DataFrame(X, columns=cols); frame["target"] = y
    model = LogisticRegression(max_iter=500).fit(frame[cols], y)
    model_id = _upload_model(client, model, "m.joblib")
    r = client.post("/api/v1/assets/dataset", files={"file": ("d.csv", io.BytesIO(frame.to_csv(index=False).encode()), "text/csv")})
    dataset_id = r.json()["asset_id"]
    r = client.post("/api/v1/investigate/uploaded", json={"model_id": model_id, "dataset_id": dataset_id, "target_column": "target", "budget": 6})
    assert r.status_code == 200, r.text
    inv_id = r.json()["investigation_id"]
    history = client.get("/api/v1/investigations").json()
    assert any(item["investigation_id"] == inv_id for item in history["items"])
    detail = client.get(f"/api/v1/investigations/{inv_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["status"] == "COMPLETED"
    assert isinstance(body["experiments"], list)
    assert all("counterexample_id" in cx for cx in body["counterexamples"])
    coverage = body["active_investigation"]["search_coverage"]
    assert coverage["scope"] == "generated_candidate_pool"
    assert coverage["executed"] == body["active_investigation"]["experiments_executed"]
    assert coverage["unexplored_candidates"] >= 0


def test_uploaded_regression_endpoint(tmp_path, monkeypatch):
    main, client = _client(tmp_path, monkeypatch)
    X, y = make_classification(n_samples=360, n_features=6, random_state=23)
    cols = [f"f{i}" for i in range(6)]
    frame = pd.DataFrame(X, columns=cols); frame["target"] = y
    model1 = LogisticRegression(max_iter=500).fit(frame[cols], y)
    model2 = LogisticRegression(C=0.25, max_iter=500).fit(frame[cols], y)
    m1 = _upload_model(client, model1, "v1.joblib")
    m2 = _upload_model(client, model2, "v2.joblib")
    r = client.post("/api/v1/assets/dataset", files={"file": ("d.csv", io.BytesIO(frame.to_csv(index=False).encode()), "text/csv")})
    d = r.json()["asset_id"]
    r = client.post("/api/v1/regression/uploaded", json={"model_v1_id": m1, "model_v2_id": m2, "dataset_id": d, "target_column": "target", "budget": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "COMPLETED"
    assert "summary" in body and "failure_deltas" in body


def test_run_certificate_is_deterministic_and_bindings_are_explicit():
    from modelxray.core.certificate import build_run_certificate

    cert1 = build_run_certificate(
        run_id="RUN-1",
        engine_version="1.0.0",
        model_hash="a" * 64,
        dataset_hash="b" * 64,
        feature_manifest=["a", "b"],
        metric="accuracy",
        budget=24,
        random_state=42,
        configuration={"target": "label"},
    )
    cert2 = build_run_certificate(
        run_id="RUN-1",
        engine_version="1.0.0",
        model_hash="a" * 64,
        dataset_hash="b" * 64,
        feature_manifest=["a", "b"],
        metric="accuracy",
        budget=24,
        random_state=42,
        configuration={"target": "label"},
    )
    assert cert1.certificate_hash == cert2.certificate_hash
    payload = cert1.to_dict()
    assert payload["metric"] == "accuracy"
    assert payload["budget"] == 24
    assert payload["feature_manifest_hash"]
    assert payload["environment"]["python_version"]
    assert len(payload["certificate_hash"]) == 64


def test_uploaded_investigation_replay_verifies(tmp_path, monkeypatch):
    main, client = _client(tmp_path, monkeypatch)
    X, y = make_classification(n_samples=320, n_features=5, random_state=31)
    cols = [f"f{i}" for i in range(5)]
    frame = pd.DataFrame(X, columns=cols); frame["target"] = y
    model = LogisticRegression(max_iter=500).fit(frame[cols], y)
    model_id = _upload_model(client, model, "replay.joblib")
    r = client.post("/api/v1/assets/dataset", files={"file": ("d.csv", io.BytesIO(frame.to_csv(index=False).encode()), "text/csv")})
    assert r.status_code == 200, r.text
    dataset_id = r.json()["asset_id"]
    r = client.post("/api/v1/investigate/uploaded", json={"model_id": model_id, "dataset_id": dataset_id, "target_column": "target", "budget": 5, "random_state": 42})
    assert r.status_code == 200, r.text
    inv_id = r.json()["investigation_id"]
    replay = client.post(f"/api/v1/investigations/{inv_id}/replay")
    assert replay.status_code == 200, replay.text
    body = replay.json()
    assert body["status"] == "VERIFIED", body
    assert body["checks"]["model_hash"]["match"]
    assert body["checks"]["dataset_hash"]["match"]
    assert body["checks"]["behavior_signature"]["match"]


def test_replay_detects_artifact_mismatch(tmp_path, monkeypatch):
    main, client = _client(tmp_path, monkeypatch)
    X, y = make_classification(n_samples=300, n_features=4, random_state=41)
    cols = [f"f{i}" for i in range(4)]
    frame = pd.DataFrame(X, columns=cols); frame["target"] = y
    model = LogisticRegression(max_iter=500).fit(frame[cols], y)
    model_id = _upload_model(client, model, "replay-mismatch.joblib")
    r = client.post("/api/v1/assets/dataset", files={"file": ("d.csv", io.BytesIO(frame.to_csv(index=False).encode()), "text/csv")})
    dataset_id = r.json()["asset_id"]
    r = client.post("/api/v1/investigate/uploaded", json={"model_id": model_id, "dataset_id": dataset_id, "target_column": "target", "budget": 5})
    inv_id = r.json()["investigation_id"]
    # Mutate the dataset bytes after the investigation. Replay must refuse before execution.
    asset = main.asset_store.get(dataset_id)
    assert asset is not None
    Path(asset.path).write_text(Path(asset.path).read_text() + "\n", encoding="utf-8")
    replay = client.post(f"/api/v1/investigations/{inv_id}/replay")
    assert replay.status_code == 200, replay.text
    body = replay.json()
    assert body["status"] == "MISMATCH"
    assert not body["checks"]["dataset_hash"]["match"]
