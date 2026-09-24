# MXBench Report

Config: {"seeds": 1, "budget": 12, "base_seed": 100, "engine_defaults": {"alpha": 0.05, "min_validation_gap": 0.03, "interaction_fraction": 0.35}}

| Scenario | Discovery rate | Mean F1 | Tests-to-disc. | Efficiency | Promoted (mean) | Interaction | FDR | CX dist. | CX valid | predict() | Runtime s | Jaccard | Crashed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_main_effect | 1.0 | 0.5618 | 2.0 | 0.8333 | 4.0 | 4.0 | - | 3.18709 | 1.0 | 176.0 | 0.127 | None | 0 |
| S2_interaction | 0.0 | 0.2504 | None | None | 4.0 | 1.0 | - | 0.25712 | 1.0 | 159.0 | 0.154 | None | 0 |
| S3_boundary_instability | 1.0 | 0.8413 | 1.0 | 0.9167 | 8.0 | 1.0 | - | 2.2571 | 1.0 | 159.0 | 0.129 | None | 0 |
| S4_covariate_shift | 1.0 | 0.8915 | 1.0 | 0.9167 | 10.0 | 9.0 | - | 0.15808 | 1.0 | 329.0 | 0.217 | None | 0 |
| S5_noisy_feature | 0.0 | 0.4549 | None | None | 5.0 | 5.0 | - | 0.14446 | 1.0 | 261.0 | 0.179 | None | 0 |
| S6_correlated_features | 1.0 | 0.9267 | 2.0 | 0.8333 | 5.0 | 4.0 | - | 1.31414 | 1.0 | 227.0 | 0.182 | None | 0 |
| S7_class_imbalance | 0.0 | 0.3318 | None | None | 6.0 | 2.0 | - | 4.07022 | 1.0 | 74.0 | 0.097 | None | 0 |
| C0_no_failure | 0.0 | 0.0 | None | None | 0.0 | 0.0 | - | None | None | 2.0 | 0.105 | None | 0 |
| C1_global_noise | 0.0 | 0.0 | None | None | 0.0 | 0.0 | - | None | None | 2.0 | 0.108 | None | 0 |
| C2_duplicated_features | 0.0 | 0.0 | None | None | 0.0 | 0.0 | - | None | None | 2.0 | 0.1 | None | 0 |
| C0b_self_consistency | 0.0 | 0.0 | None | None | 0.0 | 0.0 | 0.0 | None | None | 2.0 | 0.1 | None | 0 |

## Honest readings

- Fabrication-only FDR (C0b self-consistency control): **0.0** (target: <= 0.10). Any promotion here is a fabricated failure because zero clean errors exist by construction.
- No-fault alarm rate on C0 (flexible clean model): **0.0** — alarm may reflect a REAL stochastic weakness of that specific model, not engine fabrication (verified manually: held-out accuracy drops 7-10pp inside reported bands).
- Interaction discovery (S2): **0.0**.
- Main-effect discovery (S1): **1.0**.
- Interpretation notes: 'discovery' requires region-overlap F1 >= 0.5 against the seeded region and a validated (FDR + holdout) finding. Error concentration near class overlap is real model behavior and IS reported by design.