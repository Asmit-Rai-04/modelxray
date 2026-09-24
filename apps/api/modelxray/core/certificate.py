from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn


CERTIFICATE_VERSION = "1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def environment_manifest() -> dict[str, str]:
    return {
        "python_version": sys.version.split(" ")[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
    }


@dataclass(frozen=True)
class RunCertificate:
    certificate_version: str
    run_id: str
    engine_version: str
    model_hash: str | None
    dataset_hash: str | None
    feature_manifest_hash: str | None
    configuration_hash: str
    metric: str
    budget: int
    random_state: int
    environment: dict[str, str]

    @property
    def certificate_hash(self) -> str:
        payload = asdict(self)
        return stable_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["certificate_hash"] = self.certificate_hash
        return result


def build_run_certificate(
    *,
    run_id: str,
    engine_version: str,
    model_hash: str | None,
    dataset_hash: str | None,
    feature_manifest: list[str] | None,
    metric: str,
    budget: int,
    random_state: int,
    configuration: dict[str, Any] | None = None,
) -> RunCertificate:
    return RunCertificate(
        certificate_version=CERTIFICATE_VERSION,
        run_id=run_id,
        engine_version=engine_version,
        model_hash=model_hash,
        dataset_hash=dataset_hash,
        feature_manifest_hash=stable_hash(feature_manifest or []),
        configuration_hash=stable_hash(configuration or {}),
        metric=metric,
        budget=int(budget),
        random_state=int(random_state),
        environment=environment_manifest(),
    )
