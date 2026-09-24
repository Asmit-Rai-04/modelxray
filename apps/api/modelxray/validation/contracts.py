from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from modelxray.detectors.perturbation import InstabilityFinding


@dataclass(frozen=True)
class InvestigationResult:
    model: dict[str, Any]
    dataset: dict[str, Any]
    baseline: dict[str, Any]
    instabilities: list[dict[str, Any]]
    experiment_count: int
    active_investigation: dict[str, Any]
    counterexamples: list[dict[str, Any]]
    failure_atlas: list[dict[str, Any]]
    investigation_id: str
    metric: str = "accuracy"
    manifest: dict[str, Any] | None = None
    status: str = "completed"

    @classmethod
    def build(
        cls,
        model: dict[str, Any],
        dataset: dict[str, Any],
        baseline: dict[str, Any],
        instabilities: list[InstabilityFinding],
        experiment_count: int,
        active_investigation: dict[str, Any],
        counterexamples: list[dict[str, Any]],
        failure_atlas: list[dict[str, Any]],
        investigation_id: str,
        metric: str = "accuracy",
        manifest: dict[str, Any] | None = None,
    ) -> "InvestigationResult":
        return cls(
            model=model,
            dataset=dataset,
            baseline=baseline,
            instabilities=[asdict(x) if hasattr(x, "__dataclass_fields__") else x for x in instabilities],
            experiment_count=experiment_count,
            active_investigation=active_investigation,
            counterexamples=counterexamples,
            failure_atlas=failure_atlas,
            investigation_id=investigation_id,
            metric=metric,
            manifest=manifest,
        )

from modelxray.validation.contracts_runtime import (  # noqa: F401  (re-export shim: worker/service/tests import from here)
    ContractError,
    validate_prediction_contract,
)
