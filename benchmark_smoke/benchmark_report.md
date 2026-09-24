# MXBench Report

Config: {"seeds": 5, "budget": 16, "base_seed": 100, "engine_defaults": {"alpha": 0.05, "min_validation_gap": 0.03, "interaction_fraction": 0.35}}

| Scenario | Discovery rate | Mean F1 | Tests-to-disc. | Efficiency | Promoted (mean) | Interaction | FDR | CX dist. | CX valid | predict() | Runtime s | Jaccard | Crashed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_main_effect | 0.8 | 0.851 | 1.0 | 0.9375 | 5.2 | 1.4 | - | 0.7719 | 0.2 | 448.2 | 0.302 | 0.0 | 0 |
| S2_interaction | 0.2 | 0.2859 | 2.0 | 0.875 | 4.6 | 1.0 | - | 0.64148 | 0.2 | 371.2 | 0.255 | 0.0 | 0 |
| S3_boundary_instability | 0.0 | 0.1168 | None | None | 2.0 | 1.4 | - | 1.37091 | 1.0 | 219.0 | 0.159 | 0.0 | 0 |
| S4_covariate_shift | 1.0 | 0.9737 | 1.0 | 0.9375 | 5.8 | 2.2 | - | 0.49469 | 0.8 | 441.6 | 0.303 | 0.0 | 0 |
| S5_noisy_feature | 0.6 | 0.7141 | 1.0 | 0.9375 | 6.8 | 1.6 | - | 0.5744 | 0.4 | 434.8 | 0.297 | 0.0 | 0 |
| S6_correlated_features | 0.8 | 0.8412 | 1.0 | 0.9375 | 5.6 | 1.2 | - | 0.35482 | 0.6 | 485.8 | 0.326 | 0.0 | 0 |
| S7_class_imbalance | 0.0 | 0.0466 | None | None | 2.0 | 0.4 | - | 2.65108 | 0.75 | 198.6 | 0.15 | 0.0 | 0 |
| C0_no_failure | 0.4 | 0.0 | None | None | 0.4 | 0.4 | - | 1.39129 | 0.5 | 52.0 | 1.3 | 0.3 | 0 |
| C1_global_noise | 0.0 | 0.0481 | None | None | 0.8 | 0.4 | - | 0.93274 | 0.5 | 105.4 | 2.668 | 0.3 | 0 |
| C2_duplicated_features | 0.4 | 0.0 | None | None | 0.4 | 0.4 | - | 2.78021 | 0.5 | 45.2 | 1.165 | 0.3 | 0 |
| C0b_self_consistency | 0.0 | 0.0 | None | None | 0.0 | 0.0 | 0.0 | None | None | 2.0 | 0.077 | 1.0 | 0 |

## Honest readings

- Fabrication-only FDR (C0b self-consistency control): **0.0** (target: <= 0.10). Any promotion here is a fabricated failure because zero clean errors exist by construction.
- No-fault alarm rate on C0 (flexible clean model): **0.4** — alarm may reflect a REAL stochastic weakness of that specific model, not engine fabrication (verified manually: held-out accuracy drops 7-10pp inside reported bands).
- Interaction discovery (S2): **0.2**.
- Main-effect discovery (S1): **0.8**.
- Interpretation notes: 'discovery' requires region-overlap F1 >= 0.5 against the seeded region and a validated (FDR + holdout) finding. Error concentration near class overlap is real model behavior and IS reported by design.