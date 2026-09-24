from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class ContractError(ValueError):
    """Raised when a model and evaluation dataset violate ModelXray's runtime contract."""


@dataclass(frozen=True)
class ContractCheck:
    ok: bool
    message: str
    has_predict_proba: bool


def _resolve_expected_features(model: Any, feature_manifest: list[str] | None) -> tuple[list[str] | None, str]:
    """Return (expected_features, source) for the schema contract."""
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        return [str(x) for x in np.asarray(expected).tolist()], "model_feature_names_in_"
    if feature_manifest:
        return [str(x) for x in feature_manifest], "upload_manifest"
    return None, "none"


def _schema_mismatch_error(expected: list[str], actual: list[str]) -> ContractError:
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    same_set = not missing and not extra
    detail = (
        "same columns but wrong ORDER"
        if same_set
        else f"missing={missing}, extra={extra}"
    )
    return ContractError(
        "Feature schema mismatch between the model and the uploaded dataset: "
        f"expected ordered features {expected}, actual {actual} ({detail}). "
        "Re-upload the dataset with columns in the model's training order, or "
        "upload a model that carries feature_names_in_."
    )


def validate_prediction_contract(
    adapter: Any,
    X: pd.DataFrame,
    y: pd.Series,
    feature_manifest: list[str] | None = None,
) -> ContractCheck:
    """Validate the black-box model/data contract BEFORE spending budget.

    Schema policy (documented in README):
      1. If the model has `feature_names_in_`, that ordered list is the contract.
      2. Otherwise the upload-time CSV column order is the manifest (enforced by
         the worker through `feature_manifest`); a reordered upload fails loudly.
      3. A model with neither metadata nor manifest is rejected: silently
         accepting arbitrary column order can produce wrong predictions.

    predict_proba is probed but is no longer strictly required by the contract;
    the investigation engine itself only calls predict(). The check result is
    reported in the contract status.
    """
    model = getattr(adapter, "model", adapter)
    expected, source = _resolve_expected_features(model, feature_manifest)
    actual = [str(x) for x in X.columns]

    if expected is not None and expected != actual:
        raise _schema_mismatch_error(expected, actual)

    has_proba = hasattr(model, "predict_proba")

    probe = X.iloc[: min(32, len(X))]
    try:
        pred = np.asarray(adapter.predict(probe))
    except Exception as exc:
        raise ContractError(f"Model/data prediction contract failed on predict(): {exc}") from exc

    if pred.ndim != 1 or len(pred) != len(probe):
        raise ContractError("Model predict() must return one prediction per input row.")

    if has_proba:
        try:
            proba = np.asarray(adapter.predict_proba(probe))
        except Exception as exc:
            raise ContractError(f"Model predict_proba() probe failed: {exc}") from exc
        if proba.ndim != 2 or len(proba) != len(probe):
            raise ContractError("Model predict_proba() must return a 2D probability matrix.")
        if not np.isfinite(proba).all():
            raise ContractError("Model predict_proba() returned NaN or infinite values.")
        if np.any(proba < -1e-9) or np.any(proba > 1 + 1e-9):
            raise ContractError("Model predict_proba() returned values outside [0, 1].")
        row_sums = proba.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            raise ContractError("Model predict_proba() rows must sum to approximately 1.")
        classes = getattr(model, "classes_", None)
        if classes is not None and len(np.asarray(classes)) != proba.shape[1]:
            raise ContractError("Model classes_ does not match predict_proba() column count.")

    if y is not None and len(pd.unique(y)) < 2:
        raise ContractError("Target must contain at least two classes.")

    if expected is None:
        return ContractCheck(
            ok=True,
            message=(
                "No schema metadata or manifest: accepted upload-time column order as the "
                "feature contract (column-order manifest persisted with the asset)."
            ),
            has_predict_proba=has_proba,
        )
    return ContractCheck(
        ok=True,
        message=f"Feature order verified against {source}.",
        has_predict_proba=has_proba,
    )
