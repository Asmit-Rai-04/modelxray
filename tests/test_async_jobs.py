import time

from fastapi.testclient import TestClient


def _client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import modelxray.main as main
    from modelxray.core.ingestion import AssetStore
    from modelxray.service.job_service import InvestigationJobManager

    main.job_manager.shutdown(wait=False)
    main.asset_store = AssetStore(tmp_path / "assets")
    main.evidence_store = main.EvidenceStore(tmp_path / "evidence.db")
    main.job_manager = InvestigationJobManager(main.evidence_store, max_workers=1)
    return main, TestClient(main.app)


def _wait(client: TestClient, job_id: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in {"COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"}:
            return body
        time.sleep(0.15)
    raise AssertionError("job did not reach a terminal state before timeout")


def test_demo_job_is_asynchronous_and_persisted(tmp_path, monkeypatch):
    main, client = _client(tmp_path, monkeypatch)
    try:
        response = client.post(
            "/api/v1/jobs/demo/investigate",
            json={"rows": 500, "random_state": 7, "budget": 5},
        )
        assert response.status_code == 202, response.text
        submitted = response.json()
        assert submitted["job_id"].startswith("JOB-")
        assert submitted["status"] == "PENDING"

        job = _wait(client, submitted["job_id"])
        assert job["status"] == "COMPLETED"
        assert job["investigation_id"].startswith("INV-")

        investigation = client.get(f"/api/v1/investigations/{job['investigation_id']}")
        assert investigation.status_code == 200, investigation.text
        assert investigation.json()["status"] == "COMPLETED"
    finally:
        main.job_manager.shutdown(wait=False)


def test_uploaded_job_failure_is_persisted_as_failed_job(tmp_path, monkeypatch):
    main, client = _client(tmp_path, monkeypatch)
    try:
        response = client.post(
            "/api/v1/jobs/investigate/uploaded",
            json={
                "model_id": "does-not-exist",
                "dataset_id": "does-not-exist",
                "target_column": "target",
                "budget": 5,
            },
        )
        assert response.status_code == 202, response.text
        job = _wait(client, response.json()["job_id"])
        assert job["status"] == "FAILED"
        assert job["error"]["kind"] == "asset_not_found"
        assert job["error"]["status_code"] == 404
    finally:
        main.job_manager.shutdown(wait=False)


def test_unknown_job_returns_404(tmp_path, monkeypatch):
    main, client = _client(tmp_path, monkeypatch)
    try:
        response = client.get("/api/v1/jobs/JOB-NOTFOUND")
        assert response.status_code == 404
        assert response.json()["detail"]["kind"] == "job_not_found"
    finally:
        main.job_manager.shutdown(wait=False)


def test_job_reports_progress_and_compute_accounting(tmp_path, monkeypatch):
    main, client = _client(tmp_path, monkeypatch)
    try:
        response = client.post(
            "/api/v1/jobs/demo/investigate",
            json={"rows": 500, "random_state": 13, "budget": 5},
        )
        assert response.status_code == 202
        job = _wait(client, response.json()["job_id"])
        assert job["status"] == "COMPLETED"
        assert job["progress"] == 1.0
        assert job["phase"] == "COMPLETED"
        assert job["completed_units"] == 5
        assert job["total_units"] == 5
        assert job["predict_calls"] >= 2
        assert job["predict_rows"] >= 500
    finally:
        main.job_manager.shutdown(wait=False)


def test_pending_job_can_be_cancelled(tmp_path, monkeypatch):
    main, client = _client(tmp_path, monkeypatch)
    try:
        # Occupy the single worker so the second job stays PENDING.
        first = main.job_manager.submit("BLOCK", lambda progress, cancel: __import__("time").sleep(0.4) or {"active_investigation": {"budget": 0, "experiments_executed": 0}})
        second = main.job_manager.submit("BLOCK2", lambda progress, cancel: {"active_investigation": {"budget": 0, "experiments_executed": 0}})
        cancelled = client.post(f"/api/v1/jobs/{second.job_id}/cancel")
        assert cancelled.status_code == 200
        body = cancelled.json()
        assert body["status"] == "CANCELLED"
        assert body["cancel_requested"] is True
    finally:
        main.job_manager.shutdown(wait=True)


def test_running_job_can_be_cooperatively_cancelled(tmp_path, monkeypatch):
    main, client = _client(tmp_path, monkeypatch)
    try:
        import time
        def slow(progress, cancel):
            while not cancel.is_set():
                progress({"phase": "RUNNING", "progress": 0.4, "completed_units": 1, "total_units": 4, "current_unit": "SLOW", "current_family": "test"})
                time.sleep(0.02)
            # The next progress checkpoint turns the cancel event into JobCancelled.
            progress({"phase": "RUNNING", "progress": 0.5, "completed_units": 2, "total_units": 4, "current_unit": "CANCEL_CHECK", "current_family": "test"})
            return {"active_investigation": {"budget": 4, "experiments_executed": 2}}
        submitted = main.job_manager.submit("SLOW", slow, total_units=4)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            job = client.get(f"/api/v1/jobs/{submitted.job_id}").json()
            if job["status"] == "RUNNING":
                break
            time.sleep(0.02)
        response = client.post(f"/api/v1/jobs/{submitted.job_id}/cancel")
        assert response.status_code == 200
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            job = client.get(f"/api/v1/jobs/{submitted.job_id}").json()
            if job["status"] == "CANCELLED":
                assert job["cancel_requested"] is True
                return
            time.sleep(0.02)
        raise AssertionError(f"job was not cancelled: {job}")
    finally:
        main.job_manager.shutdown(wait=True)
