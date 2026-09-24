from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .adapters import ClassifierAdapter

SUPPORTED_METRICS = ("accuracy", "balanced_accuracy")


@dataclass(frozen=True)
class DatasetProfile:
    rows: int
    columns: int
    numeric_features: list[str]
    categorical_features: list[str]
    missing_cells: int


@dataclass(frozen=True)
class ModelProfile:
    model_type: str
    classes: list[Any]
    supports_probabilities: bool


@dataclass(frozen=True)
class BaselineMetrics:
    accuracy: float
    error_rate: float
    metric: str = "accuracy"


def _accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y))


def _balanced_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    classes = np.unique(np.concatenate([np.unique(y), np.unique(pred)]))
    recalls = []
    for cls in classes:
        cls_mask = y == cls
        if cls_mask.sum() == 0:
            continue
        recalls.append(float(np.mean(pred[cls_mask] == cls)))
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def compute_metric(pred: np.ndarray, y: np.ndarray, metric: str = "accuracy") -> float:
    if metric == "accuracy":
        return _accuracy(pred, y)
    if metric == "balanced_accuracy":
        return _balanced_accuracy(pred, y)
    raise ValueError(f"Unsupported metric '{metric}'. Supported: {', '.join(SUPPORTED_METRICS)}")


def profile_dataset(X: pd.DataFrame) -> DatasetProfile:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return DatasetProfile(
        rows=len(X),
        columns=len(X.columns),
        numeric_features=numeric,
        categorical_features=categorical,
        missing_cells=int(X.isna().sum().sum()),
    )


def profile_model(adapter: ClassifierAdapter) -> ModelProfile:
    classes = getattr(adapter, "classes_", np.array([]))
    return ModelProfile(
        model_type=type(getattr(adapter, "model", adapter)).__name__,
        classes=np.asarray(classes).tolist(),
        supports_probabilities=hasattr(getattr(adapter, "model", adapter), "predict_proba"),
    )


def evaluate_baseline(adapter: ClassifierAdapter, X: pd.DataFrame, y: pd.Series, metric: str = "accuracy") -> BaselineMetrics:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric '{metric}'. Supported: {', '.join(SUPPORTED_METRICS)}")
    pred = adapter.predict(X)
    score = compute_metric(pred, np.asarray(y), metric)
    return BaselineMetrics(accuracy=score, error_rate=1.0 - score, metric=metric)
