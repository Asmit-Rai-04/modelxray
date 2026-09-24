from __future__ import annotations

from modelxray.core.ingestion import AssetError
from modelxray.execution.sandbox import SandboxExecutionError
from modelxray.validation.contracts import ContractError


class InvestigationError(Exception):
    """Base class for investigation failures with a defined HTTP mapping."""

    status_code = 500
    kind = "internal_error"


class AssetNotFoundError(InvestigationError):
    status_code = 404
    kind = "asset_not_found"


class ContractViolationError(InvestigationError):
    """User/model contract problem: bad request semantics (422)."""

    status_code = 422
    kind = "contract_violation"


class AssetConflictError(InvestigationError):
    status_code = 409
    kind = "invalid_state"


class WorkerTimeoutError(InvestigationError):
    status_code = 504
    kind = "worker_timeout"


class WorkerEnvironmentError(InvestigationError):
    """Worker infrastructure problem, NOT a user error (500)."""

    status_code = 500
    kind = "worker_environment"


class WorkerInternalError(InvestigationError):
    status_code = 500
    kind = "worker_internal"


def to_investigation_error(exc: Exception) -> InvestigationError:
    """Translate low-level failures into the canonical taxonomy."""
    if isinstance(exc, InvestigationError):
        return exc
    if isinstance(exc, AssetError):
        message = str(exc)
        if "not found" in message.lower() or "Invalid asset id" in message or "path traversal" in message:
            return AssetNotFoundError(message)
        return ContractViolationError(message)
    if isinstance(exc, ContractError):
        return ContractViolationError(str(exc))
    if isinstance(exc, SandboxExecutionError):
        if exc.is_timeout:
            return WorkerTimeoutError(str(exc))
        if exc.is_environment:
            return WorkerEnvironmentError(str(exc))
        # Non-environment sandbox failures are worker-reported errors: contract
        # problems surface as ContractError from the worker; anything else is
        # an internal worker failure.
        message = str(exc)
        if "Internal worker failure" in message:
            return WorkerInternalError(message)
        return ContractViolationError(message)
    return WorkerInternalError(str(exc))
