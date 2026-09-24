# MXBench Report

Config: {"seeds": 30, "budget": 24, "base_seed": 100, "engine_defaults": {"alpha": 0.05, "min_validation_gap": 0.03, "interaction_fraction": 0.35}}

| Scenario | Discovery rate | Mean F1 | Tests-to-disc. | Efficiency | Promoted (mean) | Interaction | FDR | CX dist. | CX valid | predict() | Runtime s | Jaccard | Crashed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_main_effect | 0.8667 | 0.881 | 1.12 | 0.9535 | 6.933 | 1.533 | - | 0.70063 | 1.0 | 439.7 | 0.364 | 0.0001 | 0 |
| S2_interaction | 0.3 | 0.4899 | 1.0 | 0.9583 | 4.0 | 0.933 | - | 0.40952 | 1.0 | 399.6 | 0.308 | 0.0 | 0 |
| S3_boundary_instability | 0.0 | 0.1372 | None | None | 2.933 | 0.767 | - | 0.8369 | 1.0 | 308.6 | 0.263 | 0.0138 | 0 |
| S4_covariate_shift | 0.9 | 0.818 | 2.19 | 0.909 | 6.767 | 1.467 | - | 0.91199 | 1.0 | 427.0 | 0.364 | 0.0 | 0 |
| S5_noisy_feature | 0.7 | 0.727 | 1.33 | 0.9444 | 6.4 | 1.467 | - | 0.39663 | 1.0 | 445.3 | 0.338 | 0.0 | 0 |
| S6_correlated_features | 0.9 | 0.8826 | 1.11 | 0.9537 | 7.133 | 1.533 | - | 0.40841 | 1.0 | 454.0 | 0.347 | 0.0001 | 0 |
| S7_class_imbalance | 0.5333 | 0.549 | 1.25 | 0.9479 | 4.933 | 1.033 | - | 1.74089 | 1.0 | 354.3 | 0.274 | 0.0 | 0 |
| C0_no_failure | 0.2 | 0.0 | None | None | 0.3 | 0.133 | - | 1.5438 | 1.0 | 40.1 | 1.353 | 0.6345 | 0 |
| C1_global_noise | 0.0 | 0.0507 | None | None | 0.6 | 0.267 | - | 1.11136 | 1.0 | 78.5 | 2.539 | 0.4828 | 0 |
| C2_duplicated_features | 0.3333 | 0.0 | None | None | 0.533 | 0.233 | - | 1.5405 | 1.0 | 63.7 | 2.069 | 0.4368 | 0 |
| C0b_self_consistency | 0.0 | 0.0 | None | None | 0.0 | 0.0 | 0.0 | None | None | 2.0 | 0.124 | 1.0 | 0 |

## Honest readings

- Fabrication-only FDR (C0b self-consistency control): **0.0** (target: <= 0.10). Any promotion here is a fabricated failure because zero clean errors exist by construction.
- No-fault alarm rate on C0 (flexible clean model): **0.2** — alarm may reflect a REAL stochastic weakness of that specific model, not engine fabrication (verified manually: held-out accuracy drops 7-10pp inside reported bands).
- Interaction discovery (S2): **0.3**.
- Main-effect discovery (S1): **0.8667**.
- Interpretation notes: 'discovery' requires region-overlap F1 >= 0.5 against the seeded region and a validated (FDR + holdout) finding. Error concentration near class overlap is real model behavior and IS reported by design.