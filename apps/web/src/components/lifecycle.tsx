"use client";
import { useEffect, useState } from "react";
import { api, storage, errorMessage } from "../lib/api";
import type { Result } from "../lib/types";
import type { Job } from "../hooks/use-investigation-job";

export function JobProgress({ job, cancel }: { job: Job; cancel: () => void }) {
  const active = ["PENDING", "RUNNING", "CANCELLING"].includes(job.status);
  return (
    <section className="jobPanel" data-status={job.status} aria-label="Investigation progress">
      <div role="status" aria-live="polite">
        <span className="eyebrow">{job.kind?.replaceAll("_", " ")}</span>
        <h2>{job.status.replaceAll("_", " ")}</h2>
        <p>
          {job.phase?.replaceAll("_", " ") ?? "Waiting for server progress"} ·{" "}
          {job.job_id}
        </p>
      </div>
      {typeof job.progress === "number" && (
        <div>
          <label htmlFor="job-progress">
            Server-reported progress · {(job.progress * 100).toFixed(0)}%
          </label>
          <progress id="job-progress" max={1} value={job.progress} />
        </div>
      )}
      {job.completed_units != null && (
        <p>
          {job.completed_units} / {job.total_units ?? "—"} experiment units ·{" "}
          {job.current_family ?? "No strategy reported"}
        </p>
      )}
      {active && (
        <button
          className="quietButton"
          onClick={cancel}
          disabled={job.status === "CANCELLING" || job.cancel_requested}
        >
          {" "}
          {job.cancel_requested
            ? "Cancellation requested"
            : "Cancel investigation"}
        </button>
      )}
      {job.status === "CANCELLING" && (
        <p>Waiting for a cooperative server checkpoint.</p>
      )}
    </section>
  );
}

type HistoryItem = {
  investigation_id: string;
  mode: string;
  created_at: string;
  model: { model_type?: string };
  status: string;
};
export function RunHistory({
  onLoad,
  currentId,
  disabled,
}: {
  onLoad: (result: Result) => void;
  currentId?: string;
  disabled: boolean;
}) {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      setItems(
        (
          await api<{ items: HistoryItem[] }>(
            "/api/v1/investigations?limit=100",
          )
        ).items,
      );
    } catch (e) {
      setError(errorMessage(e, "History unavailable"));
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    void refresh();
  }, [currentId]);
  async function load(id: string) {
    if (!id) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api<Result>(
        `/api/v1/investigations/${encodeURIComponent(id)}`,
      );
      onLoad(result);
      storage.set("last-run", id);
    } catch (e) {
      setError(errorMessage(e, "Could not load investigation"));
    } finally {
      setLoading(false);
    }
  }
  return (
    <details
      className="ledgerDisclosure"
      open={Boolean(currentId) || Boolean(error)}
    >
      <summary>
        <span>Evidence ledger</span>
        <span className="ledgerSummary">
          {error
            ? "History unavailable"
            : loading
              ? "Loading saved investigations"
              : `${items.length} saved investigations`}
        </span>
      </summary>
      <section className="historyBar" aria-label="Evidence ledger navigation">
        <label>
          Evidence ledger
          <select
            aria-label="Previous investigations"
            value={
              items.some((i) => i.investigation_id === currentId)
                ? currentId
                : ""
            }
            onChange={(e) => void load(e.target.value)}
            disabled={disabled || loading}
          >
            <option value="">
              {loading
                ? "Loading saved runs…"
                : items.length
                  ? "Choose a saved investigation"
                  : "No saved investigations"}
            </option>
            {items.map((item) => (
              <option key={item.investigation_id} value={item.investigation_id}>
                {item.mode} · {item.model.model_type ?? "Model"} ·{" "}
                {item.investigation_id} · {item.created_at}
              </option>
            ))}
          </select>
        </label>
        <button className="quietButton" disabled={loading} onClick={refresh}>
          Refresh history
        </button>
        {error && <p role="alert">{error}</p>}
      </section>
    </details>
  );
}
