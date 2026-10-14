"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, storage, errorMessage } from "../lib/api";
import type { Result } from "../lib/types";

export type Job = {
  job_id: string;
  kind: string;
  status: string;
  phase?: string;
  progress?: number;
  completed_units?: number;
  total_units?: number;
  current_family?: string;
  investigation_id?: string;
  cancel_requested?: boolean;
  error?: { message: string };
};
const terminal = new Set(["COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"]);

export function useInvestigationJob(onResult: (result: Result) => void) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [recovering, setRecovering] = useState(true);
  const callback = useRef(onResult);
  callback.current = onResult;

  useEffect(() => {
    let disposed = false;
    const id = storage.get("active-job");
    const last = storage.get("last-run");
    (async () => {
      try {
        if (id) {
          const saved = await api<Job>(
            `/api/v1/jobs/${encodeURIComponent(id)}`,
          );
          if (!disposed) setJob(saved);
        } else if (last) {
          const saved = await api<Result>(
            `/api/v1/investigations/${encodeURIComponent(last)}`,
          );
          if (!disposed) callback.current(saved);
        }
      } catch (e) {
        if (!disposed)
          setError(
            `Recovery failed: ${errorMessage(e, "API unavailable")}. Saved identifiers are retained; retry recovery when the API is available.`,
          );
      } finally {
        if (!disposed) setRecovering(false);
      }
    })();
    return () => {
      disposed = true;
    };
  }, [retry]);

  useEffect(() => {
    if (!job?.job_id) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    const id = job.job_id;
    async function poll() {
      try {
        const current = await api<Job>(
          `/api/v1/jobs/${encodeURIComponent(id)}`,
        );
        if (disposed) return;
        setJob(current);
        setError(null);
        if (current.status === "COMPLETED") {
          if (!current.investigation_id)
            throw new Error("Completed job has no investigation identifier.");
          const result = await api<Result>(
            `/api/v1/investigations/${encodeURIComponent(current.investigation_id)}`,
          );
          if (disposed) return;
          callback.current(result);
          storage.set("last-run", result.investigation_id);
          storage.remove("active-job");
        } else if (terminal.has(current.status)) {
          storage.remove("active-job");
          if (current.error) setError(current.error.message);
        } else timer = setTimeout(poll, 1200);
      } catch (e) {
        if (!disposed) {
          setError(
            `Job connection interrupted: ${errorMessage(e, "API unavailable")}. Retrying; the server job may still be running.`,
          );
          timer = setTimeout(poll, 4000);
        }
      }
    }
    void poll();
    return () => {
      disposed = true;
      clearTimeout(timer);
    };
  }, [job?.job_id]);

  const start = useCallback(async (path: string, body: object) => {
    setSubmitting(true);
    setError(null);
    try {
      const submitted = await api<Job>(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      storage.set("active-job", submitted.job_id);
      setJob(submitted);
    } finally {
      setSubmitting(false);
    }
  }, []);
  async function cancel() {
    if (!job) return;
    try {
      setJob(
        await api<Job>(
          `/api/v1/jobs/${encodeURIComponent(job.job_id)}/cancel`,
          { method: "POST" },
        ),
      );
    } catch (e) {
      setError(errorMessage(e, "Cancellation request failed."));
    }
  }
  return {
    job,
    error,
    start,
    cancel,
    recovering,
    retryRecovery: () => setRetry((n) => n + 1),
    busy: recovering || submitting || (!!job && !terminal.has(job.status)),
  };
}
