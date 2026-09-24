"""MXBench smoke test (CI-grade, not the full 30-seed run).

Runs 2 seeds x budget 12 and asserts the benchmark machinery works and the
fabrication-FDR control stays clean. The full run lives in
`python -m tests.bench.run_bench --seeds 30 --budget 20`.
"""
from __future__ import annotations

import json

from tests.bench.run_bench import run_benchmark


def test_bench_smoke_produces_valid_results(tmp_path):
    payload = run_benchmark(seeds=2, budget=12, base_seed=900)

    results = payload["results"]
    assert set(results) >= {
        "S1_main_effect", "S2_interaction", "C0_no_failure", "C0b_self_consistency",
    }
    for name, entry in results.items():
        assert entry["crashed"] == 0, f"{name} crashed on smoke seeds"
        assert 0.0 <= entry["discovery_rate"] <= 1.0
        assert entry["mean_runtime_s"] >= 0
        assert entry["mean_predict_calls"] is not None

    # The fabrication-FDR control must be exactly clean.
    assert results["C0b_self_consistency"]["false_discovery_rate"] == 0.0

    # Machine-readable output must be JSON-serializable end to end.
    blob = json.dumps(payload)
    assert "discovery_rate" in blob


def test_bench_smoke_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from pathlib import Path

    from tests.bench.run_bench import write_report

    payload = run_benchmark(seeds=1, budget=10, base_seed=950)
    Path("benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload, Path("."))
    assert Path("benchmark_results.json").exists()
    report = Path("benchmark_report.md").read_text(encoding="utf-8")
    assert "MXBench Report" in report
    assert "Fabrication-only FDR" in report
