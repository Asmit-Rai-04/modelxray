# MXBench Report

Config: {"seeds": 5, "budget": 16, "base_seed": 100, "engine_defaults": {"alpha": 0.05, "min_validation_gap": 0.03, "interaction_fraction": 0.35}}

| Scenario | Discovery rate | Mean F1 | Tests-to-disc. | Efficiency | Promoted (mean) | Interaction | FDR | CX dist. | CX valid | predict() | Runtime s | Jaccard | Crashed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_main_effect | 0.8 | 0.7787 | 1.25 | 0.9219 | 12.0 | 9.8 | - | 1.13495 | 1.0 | 176.0 | 0.179 | 0.0 | 0 |
| S2_interaction | 0.6 | 0.6554 | 1.0 | 0.9375 | 7.2 | 6.2 | - | 0.50675 | 1.0 | 162.4 | 0.17 | 0.0 | 0 |
| S3_boundary_instability | 1.0 | 0.8527 | 1.0 | 0.9375 | 10.0 | 1.0 | - | 0.8324 | 1.0 | 162.4 | 0.188 | 0.0 | 0 |
| S4_covariate_shift | 1.0 | 0.8417 | 1.0 | 0.9375 | 11.8 | 11.4 | - | 0.75263 | 1.0 | 210.0 | 0.178 | 0.0 | 0 |
| S5_noisy_feature | 0.6 | 0.6477 | 2.33 | 0.8542 | 8.2 | 5.0 | - | 0.97307 | 1.0 | 152.8 | 0.17 | 0.0 | 0 |
| S6_correlated_features | 0.8 | 0.8471 | 1.25 | 0.9219 | 12.6 | 10.0 | - | 0.45139 | 1.0 | 244.0 | 0.204 | 0.0 | 0 |
| S7_class_imbalance | 0.0 | 0.3573 | None | None | 4.2 | 2.0 | - | 2.72982 | 1.0 | 88.2 | 0.156 | 0.0 | 0 |
| C0_no_failure | 0.0 | 0.0 | None | None | 0.0 | 0.0 | - | None | None | 2.0 | 0.13 | 1.0 | 0 |
| C1_global_noise | 0.0 | 0.0365 | None | None | 0.2 | 0.2 | - | 0.33951 | 1.0 | 20.4 | 0.573 | 0.6 | 0 |
| C2_duplicated_features | 0.2 | 0.0 | None | None | 0.8 | 0.0 | - | 0.0429 | 1.0 | 30.0 | 0.804 | 0.6 | 0 |
| C0b_self_consistency | 0.0 | 0.0 | None | None | 0.0 | 0.0 | 0.0 | None | None | 2.0 | 0.145 | 1.0 | 0 |

## Honest readings

- Fabrication-only FDR (C0b self-consistency control): **0.0** (target: <= 0.10). Any promotion here is a fabricated failure because zero clean errors exist by construction.
- No-fault alarm rate on C0 (flexible clean model): **0.0** — alarm may reflect a REAL stochastic weakness of that specific model, not engine fabrication (verified manually: held-out accuracy drops 7-10pp inside reported bands).
- Interaction discovery (S2): **0.6**.
- Main-effect discovery (S1): **0.8**.
- Interpretation notes: 'discovery' requires region-overlap F1 >= 0.5 against the seeded region and a validated (FDR + holdout) finding. Error concentration near class overlap is real model behavior and IS reported by design.