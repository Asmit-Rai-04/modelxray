from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class ClassifierAdapter(Protocol):
    """Minimum black-box contract ModelXray needs from a classifier."""

    def predict(self, X: Any) -> np.ndarray: ...

    def predict_proba(self, X: Any) -> np.ndarray: ...


@dataclass
class SklearnClassifierAdapter:
    model: Any

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("Model must expose predict_proba for the current investigation engine.")
        return np.asarray(self.model.predict_proba(X))

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(getattr(self.model, "classes_", []))

@dataclass
class CountingClassifierAdapter:
    """Transparent adapter that records model inference work."""

    base: ClassifierAdapter
    predict_calls: int = 0
    predict_rows: int = 0
    predict_proba_calls: int = 0
    predict_proba_rows: int = 0

    @property
    def model(self) -> Any:
        return getattr(self.base, "model", self.base)

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(getattr(self.base, "classes_", []))

    def predict(self, X: Any) -> np.ndarray:
        self.predict_calls += 1
        try:
            self.predict_rows += len(X)
        except TypeError:
            self.predict_rows += 1
        return self.base.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        self.predict_proba_calls += 1
        try:
            self.predict_proba_rows += len(X)
        except TypeError:
            self.predict_proba_rows += 1
        return self.base.predict_proba(X)

    def summary(self) -> dict[str, int]:
        return {
            "predict_calls": int(self.predict_calls),
            "predict_rows": int(self.predict_rows),
            "predict_proba_calls": int(self.predict_proba_calls),
            "predict_proba_rows": int(self.predict_proba_rows),
        }

