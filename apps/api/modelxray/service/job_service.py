from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any, Callable
from uuid import uuid4

from modelxray.service.errors import to_investigation_error
from modelxray.storage import EvidenceStore


class JobCancelled(RuntimeError):
    """Raised at a cooperative investigation checkpoint after cancellation."""


@dataclass(frozen=True)
class JobSubmission:
    job_id: str
    kind: str
    status: str


ProgressCallback = Callable[[dict[str, Any]], None]
JobCallable = Callable[[ProgressCallback, Event], dict[str, Any]]


class InvestigationJobManager:
    """Persistent bounded job runner with cooperative progress/cancellation."""

    def __init__(self, store: EvidenceStore, max_workers: int = 2) -> None:
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="modelxray-job")
        self._cancel_events: dict[str, Event] = {}
        self._lock = Lock()
        self.store.mark_incomplete_jobs_interrupted()

    def submit(self, kind: str, fn: JobCallable, *, total_units: int = 0) -> JobSubmission:
        job_id = f"JOB-{uuid4().hex[:12].upper()}"
        cancel_event = Event()
        with self._lock:
            self._cancel_events[job_id] = cancel_event
        self.store.create_job(job_id, kind, total_units=total_units)

        def progress(update: dict[str, Any]) -> None:
            if cancel_event.is_set():
                raise JobCancelled("Investigation cancellation requested.")
            self.store.update_job(
                job_id,
                status="RUNNING",
                phase=str(update.get("phase")) if update.get("phase") is not None else None,
                progress=update.get("progress"),
                total_units=update.get("total_units"),
                completed_units=update.get("completed_units"),
                current_unit=update.get("current_unit"),
                current_family=update.get("current_family"),
                predict_calls=update.get("predict_calls"),
                predict_rows=update.get("predict_rows"),
            )

        def runner() -> None:
            self.store.update_job(job_id, status="RUNNING", started=True, phase="STARTING", progress=0.0)
            try:
                result = fn(progress, cancel_event)
                investigation_id = result.get("investigation_id") if isinstance(result, dict) else None
                active = result.get("active_investigation", {}) if isinstance(result, dict) else {}
                compute = result.get("compute") if isinstance(result, dict) else None
                self.store.update_job(
                    job_id,
                    status="COMPLETED",
                    investigation_id=investigation_id,
                    completed=True,
                    phase="COMPLETED",
                    progress=1.0,
                    completed_units=active.get("experiments_executed") if isinstance(active, dict) else None,
                    total_units=active.get("budget") if isinstance(active, dict) else total_units or None,
                    predict_calls=compute.get("predict_calls") if isinstance(compute, dict) else None,
                    predict_rows=compute.get("predict_rows") if isinstance(compute, dict) else None,
                    cancel_requested=False,
                )
            except JobCancelled as exc:
                self.store.update_job(
                    job_id, status="CANCELLED", error={"kind": "job_cancelled", "message": str(exc), "status_code": 499},
                    completed=True, phase="CANCELLED", cancel_requested=True,
                )
            except Exception as exc:
                if bool(getattr(exc, "is_cancelled", False)):
                    self.store.update_job(
                        job_id, status="CANCELLED", error={"kind": "job_cancelled", "message": str(exc), "status_code": 499},
                        completed=True, phase="CANCELLED", cancel_requested=True,
                    )
                    return  # pragma: no cover - asserted through polling/API tests
                err = to_investigation_error(exc)
                self.store.update_job(
                    job_id,
                    status="FAILED",
                    error={"kind": err.kind, "message": str(err), "status_code": err.status_code},
                    completed=True,
                    phase="FAILED",
                )
            finally:
                with self._lock:
                    self._cancel_events.pop(job_id, None)

        self.executor.submit(runner)
        return JobSubmission(job_id=job_id, kind=kind, status="PENDING")

    def get(self, job_id: str) -> dict[str, Any] | None:
        return self.store.get_job(job_id)

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        job = self.store.get_job(job_id)
        if job is None:
            return None
        if job["status"] in {"COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"}:
            return job
        with self._lock:
            event = self._cancel_events.get(job_id)
            if event is not None:
                event.set()
        if job["status"] == "PENDING":
            self.store.update_job(
                job_id, status="CANCELLED", completed=True, phase="CANCELLED", cancel_requested=True
            )
        else:
            self.store.update_job(job_id, status="CANCELLING", phase="CANCELLING", cancel_requested=True)
        return self.store.get_job(job_id)

    def shutdown(self, wait: bool = False) -> None:
        self.executor.shutdown(wait=wait, cancel_futures=not wait)
