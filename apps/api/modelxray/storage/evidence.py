from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FailureRecord:
    failure_id: str
    investigation_id: str
    experiment_id: str
    detector: str
    condition: str
    severity: str
    support: float
    gap: float
    adjusted_p_value: float
    effect_size: float
    ci_low: float
    ci_high: float
    validation_gap: float
    reproducible: bool
    evidence_score: float
    counterexample_ids: list[str]
    failure_cluster_id: str | None = None
    supporting_experiment_ids: list[str] | None = None
    hypothesis_count: int = 1
    mean_evidence_score: float | None = None
    region_jaccard_to_representative: float | None = None


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    kind: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    investigation_id: str | None = None
    error: dict[str, Any] | None = None


@dataclass(frozen=True)
class InvestigationBundle:
    """Everything persisted for one investigation, in one transaction.

    Writing the investigation row, its experiment observations, failures, and
    counterexamples atomically prevents half-created orphan records if the
    process crashes mid-save.
    """

    investigation_id: str
    payload: dict[str, Any]
    failures: list[FailureRecord]
    counterexamples: list[dict[str, Any]]  # each must contain counterexample_id


class EvidenceStore:
    """Small SQLite evidence ledger for reproducible local investigations.

    The store is intentionally append-oriented: an investigation snapshot and its
    failures/counterexamples can be inspected later without depending on
    in-memory controller state. It is a V1 persistence layer, not a distributed
    event store. All writes for one investigation happen in a single
    transaction (see InvestigationBundle).
    """

    def __init__(self, path: str | Path = "data/modelxray.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.executescript(
                    """
                    PRAGMA journal_mode=WAL;
                    CREATE TABLE IF NOT EXISTS investigations (
                        investigation_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS experiments (
                        investigation_id TEXT NOT NULL,
                        experiment_id TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        PRIMARY KEY (investigation_id, experiment_id)
                    );
                    CREATE TABLE IF NOT EXISTS counterexamples (
                        counterexample_id TEXT PRIMARY KEY,
                        investigation_id TEXT NOT NULL,
                        experiment_id TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS failures (
                        failure_id TEXT PRIMARY KEY,
                        investigation_id TEXT NOT NULL,
                        experiment_id TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_failures_investigation ON failures(investigation_id);
                    CREATE INDEX IF NOT EXISTS idx_counterexamples_investigation ON counterexamples(investigation_id);
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        completed_at TEXT,
                        investigation_id TEXT,
                        error_json TEXT,
                        phase TEXT,
                        progress REAL NOT NULL DEFAULT 0,
                        total_units INTEGER NOT NULL DEFAULT 0,
                        completed_units INTEGER NOT NULL DEFAULT 0,
                        current_unit TEXT,
                        current_family TEXT,
                        predict_calls INTEGER NOT NULL DEFAULT 0,
                        predict_rows INTEGER NOT NULL DEFAULT 0,
                        cancel_requested INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
                    """
                )
                existing = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
                migrations = {
                    "phase": "ALTER TABLE jobs ADD COLUMN phase TEXT",
                    "progress": "ALTER TABLE jobs ADD COLUMN progress REAL NOT NULL DEFAULT 0",
                    "total_units": "ALTER TABLE jobs ADD COLUMN total_units INTEGER NOT NULL DEFAULT 0",
                    "completed_units": "ALTER TABLE jobs ADD COLUMN completed_units INTEGER NOT NULL DEFAULT 0",
                    "current_unit": "ALTER TABLE jobs ADD COLUMN current_unit TEXT",
                    "current_family": "ALTER TABLE jobs ADD COLUMN current_family TEXT",
                    "predict_calls": "ALTER TABLE jobs ADD COLUMN predict_calls INTEGER NOT NULL DEFAULT 0",
                    "predict_rows": "ALTER TABLE jobs ADD COLUMN predict_rows INTEGER NOT NULL DEFAULT 0",
                    "cancel_requested": "ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0",
                }
                for name, statement in migrations.items():
                    if name not in existing:
                        conn.execute(statement)
        finally:
            conn.close()

    def save_bundle(self, bundle: InvestigationBundle) -> None:
        """Persist an entire investigation atomically (single transaction)."""
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO investigations VALUES (?, datetime('now'), ?)",
                    (bundle.investigation_id, json.dumps(bundle.payload, default=str)),
                )
                for observation in bundle.payload.get("active_investigation", {}).get("observations", []):
                    conn.execute(
                        "INSERT OR REPLACE INTO experiments VALUES (?, ?, ?)",
                        (bundle.investigation_id, observation["experiment_id"], json.dumps(observation, default=str)),
                    )
                for record in bundle.failures:
                    conn.execute(
                        "INSERT OR REPLACE INTO failures VALUES (?, ?, ?, ?)",
                        (
                            record.failure_id,
                            record.investigation_id,
                            record.experiment_id,
                            json.dumps(asdict(record)),
                        ),
                    )
                for payload in bundle.counterexamples:
                    conn.execute(
                        "INSERT OR REPLACE INTO counterexamples VALUES (?, ?, ?, ?)",
                        (
                            payload["counterexample_id"],
                            bundle.investigation_id,
                            payload.get("source_experiment_id", ""),
                            json.dumps(payload, default=str),
                        ),
                    )
        finally:
            conn.close()

    def create_job(self, job_id: str, kind: str, status: str = "PENDING", total_units: int = 0) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO jobs(job_id, kind, status, created_at, total_units, progress) VALUES (?, ?, ?, datetime('now'), ?, ?)",
                    (job_id, kind, status, int(total_units), 0.0),
                )
        finally:
            conn.close()

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        investigation_id: str | None = None,
        error: dict[str, Any] | None = None,
        started: bool = False,
        completed: bool = False,
        phase: str | None = None,
        progress: float | None = None,
        total_units: int | None = None,
        completed_units: int | None = None,
        current_unit: str | None = None,
        current_family: str | None = None,
        predict_calls: int | None = None,
        predict_rows: int | None = None,
        cancel_requested: bool | None = None,
    ) -> None:
        conn = self._connect()
        try:
            fields = ["status = ?"]
            values: list[Any] = [status]
            optional = {
                "phase": phase,
                "progress": None if progress is None else max(0.0, min(1.0, float(progress))),
                "total_units": total_units,
                "completed_units": completed_units,
                "current_unit": current_unit,
                "current_family": current_family,
                "predict_calls": predict_calls,
                "predict_rows": predict_rows,
                "cancel_requested": None if cancel_requested is None else int(bool(cancel_requested)),
            }
            for field, value in optional.items():
                if value is not None:
                    fields.append(f"{field} = ?")
                    values.append(value)
            if investigation_id is not None:
                fields.append("investigation_id = ?")
                values.append(investigation_id)
            if error is not None:
                fields.append("error_json = ?")
                values.append(json.dumps(error, default=str))
            if started:
                fields.append("started_at = datetime('now')")
            if completed:
                fields.append("completed_at = datetime('now')")
            values.append(job_id)
            with conn:
                conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?", values)
        finally:
            conn.close()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT job_id, kind, status, created_at, started_at, completed_at, investigation_id, error_json, phase, progress, total_units, completed_units, current_unit, current_family, predict_calls, predict_rows, cancel_requested FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {
            "job_id": row[0],
            "kind": row[1],
            "status": row[2],
            "created_at": row[3],
            "started_at": row[4],
            "completed_at": row[5],
            "investigation_id": row[6],
            "error": json.loads(row[7]) if row[7] else None,
            "phase": row[8],
            "progress": float(row[9] or 0.0),
            "total_units": int(row[10] or 0),
            "completed_units": int(row[11] or 0),
            "current_unit": row[12],
            "current_family": row[13],
            "predict_calls": int(row[14] or 0),
            "predict_rows": int(row[15] or 0),
            "cancel_requested": bool(row[16]),
        }

    def mark_incomplete_jobs_interrupted(self) -> int:
        conn = self._connect()
        try:
            with conn:
                cursor = conn.execute(
                    "UPDATE jobs SET status = 'INTERRUPTED', completed_at = datetime('now') WHERE status IN ('PENDING', 'RUNNING')"
                )
                return int(cursor.rowcount)
        finally:
            conn.close()

    def save_investigation(self, investigation_id: str, payload: dict[str, Any]) -> None:
        """Backward-compatible save (investigation + experiments only)."""
        self.save_bundle(
            InvestigationBundle(investigation_id=investigation_id, payload=payload, failures=[], counterexamples=[])
        )

    def save_failure(self, record: FailureRecord) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO failures VALUES (?, ?, ?, ?)",
                    (record.failure_id, record.investigation_id, record.experiment_id, json.dumps(asdict(record))),
                )
        finally:
            conn.close()

    def save_counterexample(self, counterexample_id: str, investigation_id: str, payload: dict[str, Any]) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO counterexamples VALUES (?, ?, ?, ?)",
                    (counterexample_id, investigation_id, payload.get("source_experiment_id", ""), json.dumps(payload, default=str)),
                )
        finally:
            conn.close()

    def list_investigations(self, limit: int = 25) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT investigation_id, created_at, payload_json FROM investigations ORDER BY created_at DESC, investigation_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        items = []
        for row in rows:
            payload = json.loads(row[2])
            items.append(
                {
                    "investigation_id": row[0],
                    "created_at": row[1],
                    "status": payload.get("status", "COMPLETED"),
                    "mode": payload.get("mode", "INVESTIGATION"),
                    "baseline_accuracy": payload.get("baseline", {}).get("accuracy"),
                    "metric": payload.get("metric", "accuracy"),
                    "failure_count": len(payload.get("failure_atlas", [])),
                    "experiment_count": payload.get("experiment_count", 0),
                    "model": payload.get("model", {}),
                    "dataset": payload.get("dataset", {}),
                }
            )
        return items

    def list_counterexamples(self, investigation_id: str) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT payload_json FROM counterexamples WHERE investigation_id = ? ORDER BY counterexample_id",
                (investigation_id,),
            ).fetchall()
        finally:
            conn.close()
        return [json.loads(row[0]) for row in rows]

    def list_experiments(self, investigation_id: str) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT payload_json FROM experiments WHERE investigation_id = ?",
                (investigation_id,),
            ).fetchall()
        finally:
            conn.close()
        # Numeric-aware ordering: E-x1-2 before E-x1-10.
        def _order(row_json: dict[str, Any]) -> tuple:
            experiment_id = str(row_json.get("experiment_id", ""))
            tail = experiment_id.rsplit("-", 1)[-1]
            return (experiment_id, int(tail) if tail.isdigit() else -1)

        return sorted((json.loads(row[0]) for row in rows), key=_order)

    def get_investigation(self, investigation_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload_json FROM investigations WHERE investigation_id = ?", (investigation_id,)
            ).fetchone()
        finally:
            conn.close()
        return json.loads(row[0]) if row else None

    def list_failures(self, investigation_id: str) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT payload_json FROM failures WHERE investigation_id = ? ORDER BY CAST(json_extract(payload_json, '$.evidence_score') AS REAL) DESC",
                (investigation_id,),
            ).fetchall()
        finally:
            conn.close()
        return [json.loads(row[0]) for row in rows]
