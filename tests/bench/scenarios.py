"""Shared scenario generators for MXBench (Phase 5).

Each generator returns (X_frame, y_eval, region_mask_or_None) where region_mask
is the ground-truth fault region (None for controls). Labels follow the
"black-box evaluation" convention: the model under test is trained on clean or
corrupted training labels (per scenario), then evaluated against y_eval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def base_frame(n: int, d: int, seed: int, class_sep: float = 1.1, weights=(0.6, 0.4)) -> tuple[pd.DataFrame, np.ndarray]:
    X, y = make_classification(
        n_samples=n, n_features=d, n_informative=max(2, d - 2), n_redundant=1,
        class_sep=class_sep, weights=list(weights), random_state=seed,
    )
    frame = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(d)])
    return frame, np.asarray(y)


def _q(frame: pd.DataFrame, feature: str, q: float) -> float:
    return float(frame[feature].quantile(q))


def s1_main_effect(n=3000, seed=7, flip=0.55):
    frame, y = base_frame(n, 6, seed)
    region = (frame["x1"] > _q(frame, "x1", 0.80)).to_numpy()
    y_eval = y.copy()
    y_eval[region] ^= 1
    return frame, y_eval, region


def s2_interaction(n=3000, seed=7, flip=0.60):
    """Pairwise-interaction fault on an INDEPENDENT-coordinate design.

    make_classification rotates informative directions per seed, which made
    the seeded conjunction's support unstable (measured 0.0%-9.2% across
    seeds; an empty region is undiscoverable by construction). Here x1 and
    x2 are independent N(0,1) by construction, so the conjunction support is
    stable at ~7.5% (above the engine's 6% discovery floor) on every seed.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
            "x4": rng.normal(size=n),
            "x5": rng.normal(size=n),
            "x6": rng.normal(size=n),
        }
    )
    score = 1.8 * frame["x1"] + 1.8 * frame["x2"] + 0.8 * frame["x3"] + rng.normal(0, 1.2, size=n)
    y = (score > np.quantile(score, 0.6)).astype(int)
    region = ((frame["x1"] > _q(frame, "x1", 0.70)) & (frame["x2"] < _q(frame, "x2", 0.25))).to_numpy()
    y_eval = np.asarray(y).copy()
    y_eval[region] ^= 1
    return frame, y_eval, region


def s3_boundary_instability(n=3000, seed=7):
    frame, y = base_frame(n, 6, seed)
    # Errors hug the class-1 probability boundary: corrupt labels where a linear
    # score is near zero (jittered decision boundary). The band support equals
    # the quantile level used for the distance cutoff; 0.08 gives ~8% support,
    # safely above the engine's documented 6% discovery floor (at 0.05 the band
    # measured ~5.6%, making the cell unmeasurable rather than informative).
    score = frame.sum(axis=1)
    band = np.quantile(np.abs(score - score.median()), 0.08)
    region = (np.abs(score - score.median()) < band).to_numpy()
    y_eval = y.copy()
    y_eval[region] ^= 1
    return frame, y_eval, region


def s4_covariate_shift(n=3000, seed=7):
    frame, y = base_frame(n, 6, seed)
    # Shift x3 upward on 20% of rows; the model trained pre-shift fails there.
    region = (frame["x3"] > _q(frame, "x3", 0.80)).to_numpy()
    y_eval = y.copy()
    y_eval[region] ^= 1
    return frame, y_eval, region


def s5_noisy_feature(n=3000, seed=7, noise=0.6):
    frame, y = base_frame(n, 6, seed)
    # Additive evaluation-time noise on x4 degrades predictions where x4 is large.
    rng = np.random.default_rng(seed)
    scale = float(frame["x4"].std())
    noisy = frame["x4"] + rng.normal(0, noise * scale, size=n)
    region = (noisy > _q(frame, "x4", 0.75)).to_numpy()
    frame = frame.copy()
    frame["x4"] = noisy
    y_eval = y.copy()
    y_eval[region] ^= 1
    return frame, y_eval, region


def s6_correlated_features(n=3000, seed=7, rho=0.95):
    frame, y = base_frame(n, 6, seed)
    # Make x6 a near-copy of x1, then fault on a condition of x1 alone: the
    # engine may find either feature (or their shared region).
    rng = np.random.default_rng(seed + 1)
    frame = frame.copy()
    frame["x6"] = rho * frame["x1"] + np.sqrt(1 - rho**2) * rng.normal(size=n)
    region = (frame["x1"] > _q(frame, "x1", 0.80)).to_numpy()
    y_eval = y.copy()
    y_eval[region] ^= 1
    return frame, y_eval, region


def s7_class_imbalance(n=3000, seed=7):
    """Severe class imbalance (95/5).

    The seeded fault flips MAJORITY-class labels to minority inside the 4x4
    (x1, x2) quartile cell with the LOWEST empirical minority rate. A learner
    dominated by the 95/5 prior predicts majority there, so the flips are real
    model failures. Fixed corners proved unreliable (a corner can be genuinely
    minority-dense, in which case the model already predicts minority and the
    seeded labels are not a fault - measured in-region error 0.0000).
    """
    frame, y = base_frame(n, 6, seed, weights=(0.95, 0.05))
    x1, x2 = frame["x1"], frame["x2"]
    b1 = pd.qcut(x1, 4, labels=False, duplicates="drop")
    b2 = pd.qcut(x2, 4, labels=False, duplicates="drop")
    best_mask, best_rate, best_support = None, 2.0, 0.0
    for i in range(4):
        for j in range(4):
            m = ((b1 == i) & (b2 == j)).to_numpy()
            support = float(m.mean())
            if support < 0.03:  # discovery floor
                continue
            rate = float(y[m].mean())
            if rate < best_rate or (rate <= best_rate + 1e-12 and support > best_support):
                best_mask, best_rate, best_support = m, rate, support
    if best_mask is None:
        best_mask = np.zeros(n, dtype=bool)
    region = best_mask & (y == 0)
    y_eval = y.copy()
    y_eval[region] ^= 1
    return frame, y_eval, region


def c0_no_failure(n=3000, seed=7):
    """No-failure control.

    Uses a flexible learner (RandomForest) on well-separated classes with no
    label noise: the model under test is genuinely clean, so any concentrated,
    validated error region the engine reports is a FALSE discovery. With a
    misspecified linear model, error concentration near class overlap is real
    and the engine SHOULD report it (that is true discovery, not FDR).
    """
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(
        n_samples=n, n_features=6, n_informative=4, n_redundant=1,
        class_sep=2.0, flip_y=0.0, weights=[0.6, 0.4], random_state=seed,
    )
    frame = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(6)])
    return frame, np.asarray(y), None


def c1_global_noise(n=3000, seed=7, flip=0.10):
    frame, y = base_frame(n, 6, seed)
    rng = np.random.default_rng(seed + 9)
    region = rng.random(n) < flip
    y_eval = y.copy()
    y_eval[region] ^= 1
    return frame, y_eval, region


def c2_duplicated_features(n=3000, seed=7):
    from sklearn.datasets import make_classification as _mc

    X, y = _mc(
        n_samples=n, n_features=6, n_informative=4, n_redundant=1,
        class_sep=2.0, flip_y=0.0, weights=[0.6, 0.4], random_state=seed,
    )
    frame = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(6)])
    frame["x6"] = frame["x1"]  # exact duplicate
    return frame, np.asarray(y), None


SCENARIOS = {
    "S1_main_effect": s1_main_effect,
    "S2_interaction": s2_interaction,
    "S3_boundary_instability": s3_boundary_instability,
    "S4_covariate_shift": s4_covariate_shift,
    "S5_noisy_feature": s5_noisy_feature,
    "S6_correlated_features": s6_correlated_features,
    "S7_class_imbalance": s7_class_imbalance,
    "C0_no_failure": c0_no_failure,
    "C1_global_noise": c1_global_noise,
    "C2_duplicated_features": c2_duplicated_features,
}

FAULT_SCENARIOS = ("S1_main_effect", "S2_interaction", "S3_boundary_instability",
                   "S4_covariate_shift", "S5_noisy_feature", "S6_correlated_features",
                   "S7_class_imbalance")
CONTROL_SCENARIOS = ("C0_no_failure", "C1_global_noise", "C2_duplicated_features")
