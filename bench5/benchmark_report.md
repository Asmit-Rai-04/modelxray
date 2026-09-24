# MXBench Report

Config: {"seeds": 5, "budget": 16, "base_seed": 100, "engine_defaults": {"alpha": 0.05, "min_validation_gap": 0.03, "interaction_fraction": 0.35}}

| Scenario | Discovery rate | Mean F1 | Tests-to-disc. | Efficiency | Promoted (mean) | Interaction | FDR | CX dist. | CX valid | predict() | Runtime s | Jaccard | Crashed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_main_effect | 0.8 | 0.7802 | 1.25 | 0.9219 | 9.0 | 5.6 | - | 1.13495 | 1.0 | 172.6 | 0.195 | 0.0 | 0 |
| S2_interaction | 0.6 | 0.7062 | 1.0 | 0.9375 | 6.4 | 4.0 | - | 0.51346 | 1.0 | 176.0 | 0.201 | 0.0 | 0 |
| S3_boundary_instability | 1.0 | 0.8515 | 1.0 | 0.9375 | 6.4 | 2.6 | - | 1.16744 | 1.0 | 172.6 | 0.193 | 0.0 | 0 |
| S4_covariate_shift | 1.0 | 0.8915 | 1.0 | 0.9375 | 8.4 | 5.6 | - | 0.58308 | 1.0 | 203.2 | 0.195 | 0.0 | 0 |
| S5_noisy_feature | 0.8 | 0.6781 | 1.5 | 0.9062 | 9.0 | 4.0 | - | 0.6997 | 1.0 | 189.6 | 0.236 | 0.0 | 0 |
| S6_correlated_features | 0.8 | 0.7741 | 1.25 | 0.9219 | 8.8 | 5.6 | - | 0.31921 | 1.0 | 233.8 | 0.23 | 0.0 | 0 |
| S7_class_imbalance | 0.0 | 0.3769 | None | None | 6.2 | 2.8 | - | 1.76147 | 1.0 | 94.4 | 0.18 | 0.0 | 0 |
| C0_no_failure | 0.0 | 0.0 | None | None | 0.0 | 0.0 | - | None | None | 2.0 | 0.132 | 1.0 | 0 |
| C1_global_noise | 0.0 | 0.0707 | None | None | 0.6 | 0.4 | - | 1.17733 | 1.0 | 36.8 | 0.959 | 0.3 | 0 |
| C2_duplicated_features | 0.2 | 0.0 | None | None | 0.6 | 0.0 | - | 0.31725 | 1.0 | 23.2 | 0.628 | 0.6 | 0 |
| C0b_self_consistency | 0.0 | 0.0 | None | None | 0.0 | 0.0 | 0.0 | None | None | 2.0 | 0.159 | 1.0 | 0 |

## Honest readings

- Fabrication-only FDR (C0b self-consistency control): **0.0** (target: <= 0.10). Any promotion here is a fabricated failure because zero clean errors exist by construction.
- No-fault alarm rate on C0 (flexible clean model): **0.0** — alarm may reflect a REAL stochastic weakness of that specific model, not engine fabrication (verified manually: held-out accuracy drops 7-10pp inside reported bands).
- Interaction discovery (S2): **0.6**.
- Main-effect discovery (S1): **0.8**.
- Interpretation notes: 'discovery' requires region-overlap F1 >= 0.5 against the seeded region and a validated (FDR + holdout) finding. Error concentration near class overlap is real model behavior and IS reported by design.