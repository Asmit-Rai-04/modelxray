# ModelXray — End-to-End Model Regression Benchmark

This checkpoint adds a controlled end-to-end benchmark that runs real black-box model wrappers through the actual ModelXray investigation engine and then through the cluster-aware regression comparator.

## What it validates

- persistent seeded failure
- fixed seeded failure
- newly introduced seeded failure
- replacement (old failure fixed + new failure introduced)
- threshold/region drift

The benchmark uses a dedicated `fault_feature` that the base model does not use. The wrapper flips predictions in explicit regions, so the transition ground truth is controlled while the investigation engine remains unchanged.

## Benchmark semantics

The benchmark does **not** require the total regression output to contain only the seeded failure. ModelXray can discover other real behavioral weaknesses. Instead, it asks whether the **seeded transition itself is represented with the expected status**.

A seeded match requires sufficient overlap with the known seeded region using target recall >= 0.50 and predicted-region precision >= 0.10. This is a benchmark-specific validation criterion, not a general production guarantee.

## Result

Run:

```bash
PYTHONPATH=.:apps/api python -m tests.bench.e2e_model_regression_bench --seeds 5 --budget 12 --out benchmark/e2e_regression_5seed_fast
```

Observed:

- 5 seeds
- 5 transition cases per seed
- 25 total cases
- seeded transition detection rate: **1.00**
- persistent: **1.00**
- fixed: **1.00**
- new: **1.00**
- replacement: **1.00**
- drift: **1.00**
- exact total status-count equality: **0.00** (expected, because unrelated real failures may also be discovered)

This demonstrates that the cluster-aware regression layer can preserve the intended seeded version transition through the full investigation pipeline under this controlled benchmark. It does not prove perfect regression correctness for arbitrary real models.
