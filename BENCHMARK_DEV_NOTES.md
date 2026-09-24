# ModelXray — oblique-search development checkpoint

This checkout is based on the independently repaired v1.0 baseline. It adds an
**oblique-band hypothesis family** plus a targeted pairwise interaction family
to address the previously measured S3 boundary-instability and weak S2
interaction capability gaps.

## What changed

- Added `oblique_band` experiments using interpretable weighted linear projections.
- Candidate directions include raw equal-weight projection, standardized equal-weight
  projection, PCA directions, and an error-classifier direction.
- Candidates are narrow projection bands and use the same canonical pipeline as
  other hypotheses: discovery screen → BH-FDR → holdout gate → failure atlas.
- Added a benchmark parser for oblique region conditions.
- Added an end-to-end regression test for a seeded diagonal failure.
- Added `tests/__init__.py` so `PYTHONPATH=apps/api pytest -q` reliably imports
  the benchmark package in clean environments.

## Verification

- Full suite: **38 passed**.
- Self-consistency fabrication control: **0 promoted failures on 5/5 seeds**.
- Main-effect quick check: unchanged within expected seed variance.
- Targeted 30-seed interaction check: **96.7% discovery**, mean F1 **0.877**.
- Targeted 30-seed oblique S3 check: **90.0% discovery**, mean F1 **0.795**.
- S3 diagonal-band targeted check: **5/5 discoveries**, mean region-overlap F1
  **0.853**, seeds 100–104, budget 24.

These S3 numbers are a **targeted 5-seed development result**, not a replacement
for the full 30-seed MXBench. Run the full benchmark before making any release or
README claim about final discovery rates.

## Known limitations

- The targeted pairwise interaction family substantially improves the audited S2
  weakness in a dedicated 30-seed check. The combined full 30-seed MXBench was
  not completed in this environment because the full suite is computationally
  expensive; do not treat the targeted result as the final benchmark score.
- Oblique-band search is deliberately interpretable and limited; it is not a
  general nonlinear region learner.
- No new security boundary is claimed beyond the repaired v1.0 worker design.

## Search Coverage (v1.0.7-dev)

The investigation response now reports `active_investigation.search_coverage`.
Coverage is explicitly scoped to the finite candidate hypotheses generated for a run.
It is not model-safety coverage, exhaustive search coverage, or a probability that
no undiscovered failure exists.

Reported values include candidate pool size, executed/unexplored candidates, family
breadth, feature breadth, and per-family generated/executed/validated counts.

## v1.0.8 development checkpoint
- Added `prototype_region` novelty-search family using KMeans local neighborhoods.
- Added soft family-allocation prior to keep interaction search from being crowded out by oblique candidates.
- Unified prototype candidates through the existing validation/FDR/holdout pipeline.
- 48/48 tests pass.
- Interaction seeded regression: 10/10 seeds produced at least one reproducible failure at budget 24.
- This is not a full 30-seed MXBench result; run the full benchmark before making release claims.

## v1.0.9b failure-cluster benchmark

Added `tests/bench/failure_cluster_bench.py` with 30 seeded runs.

Results:
- Unambiguous same-mechanism families: pair precision 1.000, pair recall 1.000, mean 3 clusters.
- Adversarial ambiguity case: pair precision 0.333, recall 1.000 because two genuinely different mechanisms were deliberately assigned strongly overlapping regions. This is expected and demonstrates that geometric overlap cannot identify causal mechanism.

Product interpretation: FailureCluster must be described as an **overlapping behavioral-region consolidation**, not as mechanism discovery. The system must not claim that clustered hypotheses share a causal mechanism.

## v1.0.15 cost-aware planner development
- Added measured `evaluation_ms` to experiment observations.
- `_rank_key` now applies a soft family-level cost penalty based on observed marginal evaluation time.
- Predictions remain cached; cost therefore represents incremental hypothesis evaluation work rather than invented model-call costs.
- Added regression test ensuring a slower family is penalized relative to identical evidence from a faster run.
- Backend verification: 59 tests passed, 3 existing sklearn warnings.

## v1.0.15 cost-aware planner comparison
- Added `cost_aware` toggle to `run_active_investigation` so the existing planner can be benchmarked against the cost-unaware baseline without duplicating controller logic.
- Added `tests/bench/cost_aware_bench.py` for same-seed/same-budget comparison.
- 5-seed, budget-24 comparison: cost-aware improved S2 interaction discovery from 0.60 -> 1.00 and mean F1 0.655 -> 0.901; S4 mean F1 0.895 -> 0.914; S6 mean predict calls 69.2 -> 64.4. Runtime was broadly similar and not consistently lower.
- Result: cost-aware allocation is promising, but this is not evidence of a universal compute win. Keep it as the default while requiring larger benchmark runs before release claims.
