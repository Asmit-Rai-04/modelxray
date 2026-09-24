from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modelxray.investigation.regression import compare_failure_snapshots


@dataclass(frozen=True)
class Case:
    name: str
    v1: list[dict[str, Any]]
    v2: list[dict[str, Any]]
    expected: dict[str, int]


def _obs(eid: str, condition: str, *, cluster: str | None = None, gap: float = 0.2, score: float = 80.0, reproducible: bool = True) -> dict[str, Any]:
    return {
        "experiment_id": eid,
        "condition": condition,
        "kind": "subgroup_threshold",
        "severity": "HIGH" if gap >= 0.15 else "MEDIUM",
        "gap": gap,
        "support": 0.12,
        "adjusted_p_value": 1e-4,
        "validation_gap": gap * 0.9,
        "evidence_score": score,
        "reproducible": reproducible,
        "failure_cluster_id": cluster,
    }


def _cases(seed: int) -> list[Case]:
    t = 1.00 + (seed % 5) * 0.004
    return [
        Case(
            "persistent_exact_cluster",
            [_obs("v1-a", "x > 1.00", cluster="FC-A")],
            [_obs("v2-a", "x > 1.00", cluster="FC-Z")],
            {"PERSISTENT": 1, "FIXED": 0, "NEW": 0},
        ),
        Case(
            "persistent_threshold_drift",
            [_obs("v1-a", "x > 1.00", cluster="FC-A")],
            [_obs("v2-a", f"x > {t:.3f}", cluster="FC-Z")],
            {"PERSISTENT": 1, "FIXED": 0, "NEW": 0},
        ),
        Case(
            "fixed",
            [_obs("v1-a", "x > 1.00", cluster="FC-A")],
            [_obs("v2-a", "x > 1.00", cluster="FC-Z", reproducible=False)],
            {"PERSISTENT": 0, "FIXED": 1, "NEW": 0},
        ),
        Case(
            "new",
            [],
            [_obs("v2-a", "x > 2.00", cluster="FC-Z")],
            {"PERSISTENT": 0, "FIXED": 0, "NEW": 1},
        ),
        Case(
            "replacement",
            [_obs("v1-a", "x > 1.00", cluster="FC-A")],
            [_obs("v2-b", "x > 3.00", cluster="FC-Z")],
            {"PERSISTENT": 0, "FIXED": 1, "NEW": 1},
        ),
        Case(
            "split_region",
            [
                _obs("v1-a1", "x > 1.00", cluster="FC-A", score=90),
                _obs("v1-a2", "x > 1.10", cluster="FC-A", score=85),
            ],
            [
                _obs("v2-b1", "x > 1.00", cluster="FC-X", score=91),
                _obs("v2-b2", "x > 1.10", cluster="FC-Y", score=84),
            ],
            {"PERSISTENT": 1, "FIXED": 0, "NEW": 1},
        ),
        Case(
            "merge_regions",
            [
                _obs("v1-a1", "x > 1.00", cluster="FC-A", score=90),
                _obs("v1-a2", "x > 2.00", cluster="FC-B", score=85),
            ],
            [
                _obs("v2-c1", "x > 1.00", cluster="FC-X", score=92),
                _obs("v2-c2", "x > 2.00", cluster="FC-X", score=86),
            ],
            {"PERSISTENT": 1, "FIXED": 1, "NEW": 0},
        ),
        Case(
            "unrelated_clusters",
            [
                _obs("v1-a", "x > 1.00", cluster="FC-A"),
                _obs("v1-b", "x > 4.00", cluster="FC-B"),
            ],
            [
                _obs("v2-c", "x > 1.00", cluster="FC-X"),
                _obs("v2-d", "x > 7.00", cluster="FC-Y"),
            ],
            {"PERSISTENT": 1, "FIXED": 1, "NEW": 1},
        ),
    ]


def _counts(deltas) -> dict[str, int]:
    out = {"PERSISTENT": 0, "FIXED": 0, "NEW": 0}
    for d in deltas:
        out[d.status] += 1
    return out


def run(seeds: int, out: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in range(100, 100 + seeds):
        for case in _cases(seed):
            deltas = compare_failure_snapshots(case.v1, case.v2)
            actual = _counts(deltas)
            exact = actual == case.expected
            rows.append({"seed": seed, "case": case.name, "expected": case.expected, "actual": actual, "exact": exact})

    exact_rate = sum(r["exact"] for r in rows) / len(rows) if rows else 1.0
    result = {
        "seeds": seeds,
        "cases_per_seed": len(_cases(100)),
        "total_cases": len(rows),
        "exact_case_accuracy": exact_rate,
        "failed_cases": [r for r in rows if not r["exact"]],
        "interpretation": "Regression benchmark validates status-count behavior on controlled version transitions. It is not a causal correctness guarantee.",
        "rows": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "regression_benchmark.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("benchmark/regression"))
    args = parser.parse_args()
    result = run(args.seeds, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
