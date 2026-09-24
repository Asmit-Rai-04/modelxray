from pathlib import Path

from modelxray.storage import EvidenceStore, FailureRecord


def test_evidence_store_persists_investigation_and_failure(tmp_path: Path):
    store = EvidenceStore(tmp_path / "modelxray.db")
    payload = {
        "investigation_id": "INV-TEST",
        "active_investigation": {"observations": [{"experiment_id": "E-1", "gap": 0.2}]},
    }
    store.save_investigation("INV-TEST", payload)
    record = FailureRecord(
        failure_id="F-TEST-001",
        investigation_id="INV-TEST",
        experiment_id="E-1",
        detector="active_investigation",
        condition="x > 1",
        severity="HIGH",
        support=0.12,
        gap=0.2,
        adjusted_p_value=0.001,
        effect_size=-0.2,
        ci_low=-0.3,
        ci_high=-0.1,
        validation_gap=0.18,
        reproducible=True,
        evidence_score=88.0,
        counterexample_ids=["CX-1"],
    )
    store.save_failure(record)
    assert store.get_investigation("INV-TEST")["investigation_id"] == "INV-TEST"
    failures = store.list_failures("INV-TEST")
    assert failures[0]["failure_id"] == "F-TEST-001"
    assert failures[0]["counterexample_ids"] == ["CX-1"]
