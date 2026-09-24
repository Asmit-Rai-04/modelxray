# ModelXray

Autonomous ML failure discovery, boundary-evidence, and model-regression engine for black-box tabular classifiers.

## v1.0 — Repaired release

This release repairs the findings of an independent audit (interaction discovery,
worker IPC, schema safety, counterexample semantics, statistical gates, path
traversal, error taxonomy, persistence integrity) and adds MXBench, a reproducible
benchmark harness with machine-readable results.

## Investigation flow

`ingest → profile → discovery/validation split → hypothesis family (quantile grid + tree-guided pairwise interactions) → budgeted sequential selection → Fisher screening → BH-FDR over the executed family → holdout credibility gate → failure atlas → boundary-sensitivity evidence → model regression`

## What changed vs the audited v1.0 (honest summary)

- **Worker IPC**: results travel via a unique tempfile (`--result-path`), never
  stdout. Models that `print()` during `predict()` cannot corrupt the protocol
  (regression-tested).
- **Worker environment**: `PYTHONNOUSERSITE` is no longer set; the worker uses
  the same interpreter/environment as the API. Import failures produce an
  actionable, classified environment error.
- **Interaction discovery**: the controller owns one hypothesis family. A
  depth-3 error tree (fit on discovery rows only) proposes pairwise-conjunction
  candidates which go through the SAME Fisher screening, BH-FDR, and holdout
  gate as the quantile grid. No second, competing detector exists anymore.
- **Schema safety**: models carrying `feature_names_in_` enforce it; models
  without it use the upload-time CSV column order as a persisted manifest;
  reordered uploads fail loudly instead of silently producing wrong predictions.
- **Counterexamples → boundary-sensitivity evidence**: attached ONLY to
  validated failures; a clean model produces zero of them. Renamed away from
  "proof of breakage" everywhere (API payload semantics, UI, docs).
- **Statistics**: single sign convention (`gap = baseline − subgroup`, effect =
  subgroup disadvantage); holdout gate = minimum practical effect + sign
  stability + CI lower bound > 0; FDR honestly scoped to the executed family
  (documented); one canonical severity function; `balanced_accuracy` metric
  option for imbalanced data.
- **Security**: strict asset-ID validation (regex + resolve check) closes the
  path-traversal hole; worker remains defense-in-depth, NOT a sandbox (below).
- **Errors**: 404 asset/investigation not found, 409 conflict, 422 contract
  violation, 504 timeout, 500 worker/environment failure — no more everything-422.
- **Persistence**: one transaction per investigation (no orphan records);
  connections always closed; run manifests (versions, hashes, config) stored
  and checked for regression comparability.
- **MXBench**: `python -m tests.bench.run_bench` produces `benchmark_results.json`
  and `benchmark_report.md` (see Benchmark below).

## Statistical guarantees — exact scope

- BH-FDR (α = 0.05, configurable constant `ALPHA`) is applied to the **executed
  family** (the experiments actually run within the budget). The candidate pool
  is larger; `experiments_considered` vs `experiments_executed` are reported so
  the scope is explicit. Selection inside the family is ranking-based; FDR
  control holds for the executed family under standard BH conditions.
- The holdout gate requires: discovery gap ≥ 0.05, holdout gap ≥ 0.03, sign
  stability, and holdout disadvantage-CI lower bound > 0. Thresholds are
  documented constants (`controller.MIN_*`), not statistically derived optima.
- `evidence_score` (0–100) is a transparent heuristic — NOT calibrated, NOT a
  probability, NOT comparable across unrelated failure types. Severity labels
  use fixed thresholds (`reliability.severity`).
- Discovery/validation are disjoint row splits; thresholds are computed on
  discovery rows only; no label or threshold leakage into holdout.

## Model & dataset contract

- Binary/multiclass sklearn-compatible classifiers exposing `predict()`
  (`predict_proba` probed but not required by the engine).
- Numeric features support threshold, interaction, and oblique hypotheses;
  categorical features support interpretable single-value subgroup hypotheses.
- `feature_names_in_` enforced when present; otherwise upload-time column order
  is the manifest. Reordered uploads are rejected with a message naming the
  mismatch (missing/extra/wrong ORDER).
- Datasets with missing values are rejected with an explicit, documented
  policy message (drop or impute before upload).
- Minimum 100 evaluation rows; at least one supported feature column is required.
  Numeric-only datasets enable numeric hypothesis families; categorical-only datasets
  can still be investigated through categorical subgroup hypotheses.

### Metric-specific inference

When `balanced_accuracy` is selected, subgroup inference uses stratified permutation
tests and stratified bootstrap confidence intervals because balanced accuracy is
not a simple row-level binomial proportion. Accuracy retains the Fisher-exact /
two-proportion path. Results are deterministic for the same experiment IDs.

## Security boundary (read before deploying)

- **Pickle/joblib deserialization executes arbitrary code.** The isolated worker
  is defense-in-depth (separate process, POSIX rlimits for CPU/memory/FD/file
  size, single-threaded BLAS), **not a sandbox**: a malicious model artifact can
  do anything the worker's OS user can, including network access. On Windows,
  rlimits are unavailable (wall-clock timeout only). There is no authentication,
  no rate limiting, no network isolation, and no per-tenant separation in this
  codebase. For any untrusted/multi-tenant deployment: run the worker in a
  dedicated unprivileged container/VM with network isolation and no secrets,
  or do not accept public uploads.
- Model bytes are never deserialized in the API process; validation happens in
  the worker only.

## Benchmark (MXBench)

```bash
PYTHONPATH=apps/api python -m tests.bench.run_bench --seeds 30 --budget 20 --out benchmark
```

Cells: S1 main-effect, S2 interaction, S3 boundary instability, S4 covariate
shift, S5 noisy feature, S6 correlated features, S7 class imbalance;
controls C0 (clean flexible model), C1 global noise, C2 duplicated features,
C0b self-consistency (labels = teacher predictions; any promotion is a
fabricated failure).

Metrics: discovery rate (region-overlap F1 ≥ 0.5 against the seeded region),
region F1, tests-to-discovery, discovery efficiency, no-fault alarm rate,
fabrication-only FDR (C0b), counterexample distance/validity, reproducibility
(Jaccard across seeds), predict()-call count, runtime. Results are written to
`benchmark/benchmark_results.json` and `benchmark/benchmark_report.md`.

Honest headline numbers (30 seeds, budget 24; see `benchmark_report.md` for the
run in this repo — regenerate rather than trusting any snapshot): main-effect
discovery 0.87, covariate shift 0.90, correlated 0.90, noisy feature 0.70,
interaction 0.30, imbalance 0.53, boundary-instability 0.0; fabrication-only
FDR (C0b) 0.00; counterexample replay validity 1.00. Interaction faults are
discoverable but far from reliably; axis-aligned-threshold discovery cannot
express a diagonal boundary-instability band — a **documented capability
boundary**, not a bug.

## Development checkpoint — oblique failure-region search

The current working tree adds an `oblique_band` hypothesis family for non-axis-aligned
failure regions. This is a development checkpoint, not a new release claim: the full
30-seed MXBench has **not** been rerun after this change. A targeted 5-seed S3 run
(seeds 100–104, budget 24) recovered the seeded diagonal failure in **5/5** cases
with mean region-overlap F1 **0.853**. The clean self-consistency control still
produced **0 promoted failures across 5/5 seeds**. Pairwise interaction discovery
remains a known weakness and still requires a full benchmark before any v1.x
release claim. See `BENCHMARK_DEV_NOTES.md`.

## Run backend

```bash
PYTHONPATH=apps/api uvicorn modelxray.main:app --reload --port 8000
```

## Run tests

```bash
PYTHONPATH=apps/api pytest -q
```

CI (GitHub Actions) installs a clean environment, lints, runs the suite, and
executes a benchmark smoke test on every push.

## Run web

```bash
cd apps/web
npm install
npm run dev
```

The web app expects the API at `http://localhost:8000`.

## API surface

- `POST /api/v1/assets/model` — upload `.joblib/.pkl/.pickle`
- `POST /api/v1/assets/dataset` — upload labeled `.csv` (column order becomes the schema manifest)
- `POST /api/v1/investigate/uploaded` — `{model_id, dataset_id, target_column, budget, random_state, metric}`
- `POST /api/v1/demo/investigate` — synthetic demo (seeds a *marginal* failure region the engine reliably finds)
- `POST /api/v1/demo/compare`, `POST /api/v1/regression/uploaded` — regression with manifest comparability check
- `GET /api/v1/investigations`, `GET /api/v1/investigations/{id}` — evidence ledger
- `GET /health`


## Cost-aware planner comparison (development benchmark)

A same-seed, same-budget comparison is available at `benchmark/cost_aware_compare_10.json` (10 seeds, budget 24). The cost-aware planner improved discovery/F1 for interaction faults in this run (0.80 → 1.00 discovery; mean F1 0.784 → 0.907) and modestly improved F1 on several other cells, but it did **not** consistently reduce wall-clock time or `predict()` calls. Therefore the cost-aware policy is currently treated as an evidence-prioritization improvement, not a guaranteed compute-saving optimization. A 30-seed comparison remains a release-quality validation task.

## Dashboard

The web app keeps the black/charcoal + neon-blue engineering aesthetic and now
states only what the backend proves: "validated failures" (never "verified
breakage"), boundary-sensitivity evidence attached only to validated regions, a
real investigation trace built from backend fields, an explicit empty state when
no failure survives the gates, and interaction/threshold experiment lineage.


## Development checkpoint — interaction and oblique search

The current development branch extends hypothesis generation with:

- targeted pairwise interaction search over a ranked subset of numeric features;
- oblique weighted-projection bands for non-axis-aligned failure regions.

Both are part of the same canonical discovery → FDR → holdout → evidence pipeline.

Development verification (not a final release benchmark):
- full tests: 38 passed;
- 30-seed targeted pairwise interaction check: 96.7% discovery, mean F1 0.877;
- 30-seed targeted oblique S3 check: 90.0% discovery, mean F1 0.795;
- 10-seed fabrication-only C0b check: 0 promoted failures.

The full multi-scenario 30-seed MXBench remains the release gate.

### Search coverage semantics

ModelXray reports how much of the **generated candidate pool** it actually tested.
This is intentionally not presented as coverage of the full model behavior space and
does not imply that unobserved failure modes do not exist. The UI exposes candidate
coverage, family breadth, unexplored candidates, and per-family execution ratios so
the user can see both what was investigated and what remained unexplored.

### Multi-strategy discovery (v1.0.8-dev)
The development engine now searches across numeric thresholds, categorical subgroups, pairwise interactions, oblique projection bands, and prototype neighborhoods. Prototype neighborhoods are a novelty-search family intended to surface local error concentrations that are not naturally axis-aligned or linear. All candidates use the same discovery → FDR → holdout validation pipeline.

Family allocation uses a soft prior so interaction discovery cannot be starved by higher-scoring oblique candidates, while observed evidence can still reallocate remaining budget. This is an experimental development feature and is not yet a final benchmarked release.

## Failure consolidation (development)

Validated hypotheses are now consolidated into `FailureCluster` objects using region overlap on the validation frame. This prevents multiple nearby threshold/interaction descriptions of the same behavioral region from being counted as separate failures. The cluster is explicitly geometric, not causal: a cluster means the hypotheses overlap strongly enough to present together, not that they share a proven mechanism. The default Jaccard threshold is 0.60.

The API exposes `active_investigation.failure_clusters` and `failure_cluster_count`. Each persisted Failure Atlas record carries the representative experiment plus `supporting_experiment_ids`, `hypothesis_count`, and overlap metadata.

## Failure-aware model regression benchmark (development)

A controlled regression benchmark is available at `tests/bench/regression_bench.py`.
It tests version-to-version status transitions for persistent, fixed, new,
threshold-drift, split-region, merge-region, and unrelated-cluster cases while
allowing experiment and cluster IDs to change between versions.

Run:

```bash
PYTHONPATH=apps/api python -m tests.bench.regression_bench --seeds 30 --out benchmark/regression
```

Current controlled result in this checkpoint: **240/240 cases exact (1.00)**.
This validates the regression comparator's behavior on controlled transition
fixtures; it is **not** a claim of causal equivalence between model versions.
Model-level regression still requires end-to-end benchmark runs over the actual
investigation engine.

## End-to-end regression benchmark checkpoint

A controlled end-to-end regression benchmark is available at `tests/bench/e2e_model_regression_bench.py`. It runs black-box fault-injected model versions through the real investigation engine and cluster-aware comparator. The latest 5-seed, budget-12 run detected all 25 seeded transitions (1.00 seeded transition detection rate); total status-count equality is intentionally not used as the correctness criterion because the engine may discover additional real failure regions.

## Cross-strategy agreement (development)
Failure clusters now record the distinct discovery strategy kinds that independently support the same behavioral region. This is descriptive evidence diversity, not a causal proof: a cluster supported by `subgroup_threshold` + `interaction` is not automatically more severe or more causal, but it gives the investigator visibility into whether multiple search strategies converged on the same region.

## Async investigation jobs (backend development)
Long-running investigations can now be submitted as durable background jobs without changing the synchronous endpoints. The job status is persisted in SQLite and can be polled independently from the HTTP request.

Endpoints:

```text
POST /api/v1/jobs/demo/investigate
POST /api/v1/jobs/investigate/uploaded
GET  /api/v1/jobs/{job_id}
```

A submission returns `202 Accepted` with a `JOB-...` identifier. Terminal states are `COMPLETED`, `FAILED`, and `INTERRUPTED`. An application restart marks jobs that were `PENDING` or `RUNNING` as `INTERRUPTED`; callers can retry them safely. The current implementation uses a bounded in-process thread pool plus SQLite persistence and intentionally avoids adding Redis/Celery at this stage.

Verification checkpoint:

```text
54 non-benchmark tests passed
4 benchmark regression/smoke tests passed
3 sklearn warnings
```
## Backend job lifecycle (v1.0.13-dev)

Long investigations run as durable jobs. `GET /api/v1/jobs/{job_id}` now exposes phase, progress, experiment units, current strategy/unit, model prediction-call accounting, and cancellation state. `POST /api/v1/jobs/{job_id}/cancel` requests cooperative cancellation; demo investigations check cancellation at experiment checkpoints and uploaded-model jobs can terminate their isolated worker process.

Compute accounting is reported as model `predict()` calls and rows scored. These are execution counts, not unique data rows.



### Counterexample compute optimization

Boundary-sensitivity search now batches initial grid predictions, binary-refinement midpoints, and final endpoint verification across candidate features. This reduces model invocation overhead without changing the approximate single-feature flip semantics. A model-call-count regression test protects the optimization.

## Run certificates

Each completed investigation emits a deterministic run certificate containing the
engine/environment versions, metric, budget, random seed, feature-manifest hash,
input model/dataset hashes when available, and a canonical configuration hash.
The certificate itself has a SHA-256 `certificate_hash`, allowing a saved run to
be checked for accidental configuration drift before replay or model regression.

A certificate records what was run; it does not make a stochastic investigation
mathematically immutable or guarantee that the search found every possible model
failure.

## Replay verification (development)

Saved uploaded-model investigations can be replayed with:

```bash
curl -X POST http://localhost:8000/api/v1/investigations/<INVESTIGATION_ID>/replay
```

The endpoint compares the recorded run certificate against the current model/dataset
artifacts and environment, then re-runs the investigation when the artifacts still
match. It reports `VERIFIED`, `MISMATCH`, or `NOT_REPLAYABLE` and exposes the specific
checks that differ. A replay verification confirms reproducibility of the recorded
investigation under the declared configuration; it does not establish completeness
of failure discovery.

## Containerized deployment (v1.0.20-dev)

A production-oriented Docker Compose profile is available under `deploy/`.
It runs the API as a non-root user with a read-only application filesystem,
resource limits, dropped Linux capabilities, and a persistent data volume.
This is defense-in-depth, not a hostile-code sandbox for arbitrary pickle/joblib
models; public multi-tenant deployments still require a stronger isolation boundary.

```bash
docker compose -f deploy/docker-compose.prod.yml up --build
```
