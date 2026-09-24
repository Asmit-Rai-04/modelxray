from pathlib import Path

from tests.bench.failure_cluster_bench import run


def test_failure_cluster_benchmark_smoke(tmp_path: Path):
    result = run(3, tmp_path)
    assert result["mean_pair_precision"] >= 0.95
    assert result["mean_pair_recall"] >= 0.95
    assert (tmp_path / "failure_cluster_benchmark.json").exists()
