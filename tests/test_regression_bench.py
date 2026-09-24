from pathlib import Path

from tests.bench.regression_bench import run


def test_regression_benchmark_smoke(tmp_path: Path):
    result = run(2, tmp_path)
    assert result["exact_case_accuracy"] >= 0.90
    assert (tmp_path / "regression_benchmark.json").exists()
