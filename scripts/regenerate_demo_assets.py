"""Regenerate data/demo_assets with a model that HAS a discoverable failure.

The audited v1.0 shipped a fixture whose best single-threshold gap was ~0.027
(noise), so the golden path produced an empty atlas. This script seeds a
*marginal* failure region (feature_1 > q80, 60% label corruption in training
labels) so the deterministic engine rediscovers it end-to-end.

Usage: PYTHONPATH=apps/api python scripts/regenerate_demo_assets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main() -> None:
    from sklearn.datasets import make_classification

    rows = 2500
    seed = 42
    X_arr, y = make_classification(
        n_samples=rows, n_features=8, n_informative=4, n_redundant=1,
        n_clusters_per_class=2, weights=[0.62, 0.38], class_sep=1.1, random_state=seed,
    )
    X = pd.DataFrame(X_arr, columns=[f"feature_{i+1}" for i in range(X_arr.shape[1])])
    threshold = float(X["feature_1"].quantile(0.80))
    rng = np.random.default_rng(seed)
    y_mut = np.asarray(y).copy()
    y_mut[(X["feature_1"] > threshold).to_numpy() & (rng.random(rows) < 0.60)] ^= 1

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y_mut, test_size=0.40, stratify=y_mut, random_state=seed
    )
    model = RandomForestClassifier(n_estimators=160, min_samples_leaf=4, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)

    frame = X_rest.copy()
    frame["target"] = y_rest
    out_dir = Path("data/demo_assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "demo_model.joblib")
    frame.to_csv(out_dir / "demo_dataset.csv", index=False)

    # Verify the seeded failure is actually discoverable on the rest split.
    pred = model.predict(X_rest)
    correct = pred == y_rest
    region = (X_rest["feature_1"] > X_rest["feature_1"].quantile(0.80)).to_numpy()
    gap = correct[~region].mean() - correct[region].mean()
    print(f"demo assets regenerated; holdout gap in seeded region: {gap:.3f}")
    if gap < 0.05:
        print("WARNING: seeded region gap below discovery threshold; adjust corruption rate.")
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())
