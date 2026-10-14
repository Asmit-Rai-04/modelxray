"use client";
import { useState } from "react";
import type { AtlasFailure, Observation, CompareResult } from "../lib/types";
import { pp, familyLabel } from "../lib/format";

type Interval = {
  id: string;
  label: string;
  value: number;
  low?: number;
  high?: number;
  validated?: boolean;
};
function IntervalChart({
  rows,
  title,
  description,
  unit = "pp",
}: {
  rows: Interval[];
  title: string;
  description: string;
  unit?: string;
}) {
  const [expanded, setExpanded] = useState(false);
  if (!rows.length)
    return (
      <p className="panelFootnote">No evidence available for this chart.</p>
    );
  const values = rows
    .flatMap((r) => [r.value, r.low ?? r.value, r.high ?? r.value])
    .filter(Number.isFinite);
  const min = Math.min(0, ...values);
  const max = Math.max(0.01, ...values);
  const x = (value: number) => 255 + ((value - min) / (max - min)) * 510;
  const format = (value: number) =>
    unit === "pp" ? pp(value) : value.toFixed(2);
  const plotted = expanded ? rows : rows.slice(0, 12);
  const hasIntervals = rows.some(
    (r) => Number.isFinite(r.low) && Number.isFinite(r.high),
  );
  return (
    <figure className="analysisFigure">
      <figcaption>
        <h3>{title}</h3>
        <p>{description}</p>
        <p className="chartLegend">
          {rows.some((row) => row.validated === false) ? "● Validated effect · ○ Screened effect · " : "● Reported value · "}
          {hasIntervals ? "━ Backend confidence interval · " : ""}dashed line:
          zero · units: {unit === "pp" ? "percentage points (pp)" : unit}
        </p>
      </figcaption>
      <div className="chartScroll" role="region" aria-label={`${title} plot`} tabIndex={0}>
        <svg
          viewBox={`0 0 820 ${plotted.length * 44 + 60}`}
          role="img"
          aria-label={`${title}. Exact values are in the following evidence table.`}
        >
          <line
            className="zeroLine"
            x1={x(0)}
            x2={x(0)}
            y1={10}
            y2={plotted.length * 44 + 15}
          />
          {plotted.map((r, i) => (
            <g key={r.id} transform={`translate(0 ${i * 44 + 30})`}>
              <text x={8} y={4}>
                {r.id.length > 30 ? `${r.id.slice(0, 27)}…` : r.id}
              </text>
              <line className="chartGuide" x1={255} x2={765} />
              {Number.isFinite(r.low) && Number.isFinite(r.high) && (
                <>
                  <line
                    className="intervalLine"
                    x1={x(r.low!)}
                    x2={x(r.high!)}
                  />
                  <line
                    className="intervalLine"
                    x1={x(r.low!)}
                    x2={x(r.low!)}
                    y1={-7}
                    y2={7}
                  />
                  <line
                    className="intervalLine"
                    x1={x(r.high!)}
                    x2={x(r.high!)}
                    y1={-7}
                    y2={7}
                  />
                </>
              )}
              <circle
                className={
                  r.validated === false ? "screenedPoint" : "effectPoint"
                }
                cx={x(r.value)}
                r={5}
              >
                <title>{`${r.label}: ${format(r.value)}`}</title>
              </circle>
            </g>
          ))}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <text
              key={t}
              x={255 + t * 510}
              y={plotted.length * 44 + 43}
              textAnchor="middle"
            >
              {format(min + t * (max - min))}
            </text>
          ))}
        </svg>
      </div>
      {rows.length > 12 && (
        <div className="plotControls">
          <span>
            Plotting {plotted.length} of {rows.length} entries. The evidence
            table includes every entry.
          </span>
          <button
            className="quietButton"
            aria-expanded={expanded}
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? "Show compact plot" : `Show all ${rows.length} entries`}
          </button>
        </div>
      )}
      <details>
        <summary>Accessible evidence table · {rows.length} entries</summary>
        <div className="tableScroll" role="region" aria-label={`${title} data`} tabIndex={0}>
          <table>
            <caption>{title} — exact backend values</caption>
            <thead>
              <tr>
                <th scope="col">Region / hypothesis</th>
                <th scope="col">Observed ({unit})</th>
                <th scope="col">Interval lower ({unit})</th>
                <th scope="col">Interval upper ({unit})</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id}>
                  <th scope="row">
                    {r.id} · {r.label}
                  </th>
                  <td>{format(r.value)}</td>
                  <td>{r.low == null ? "Not reported" : format(r.low)}</td>
                  <td>{r.high == null ? "Not reported" : format(r.high)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

export function EvidenceChart({
  failures,
  onSelect,
}: {
  failures: AtlasFailure[];
  onSelect?: (id: string) => void;
}) {
  return (
    <>
      <IntervalChart
        rows={failures.map((f) => ({
          id: f.failure_id,
          label: f.condition,
          value: f.effect_size,
          low: f.ci_low,
          high: f.ci_high,
        }))}
        title="Region effect intervals"
        description="Discovery-set effects and confidence intervals returned by the backend. Reproducibility and holdout gaps are reported separately in each region."
      />
      {onSelect && failures.length > 0 && (
        <div
          className="evidenceLinks"
          role="group"
          aria-label="Inspect plotted failure regions"
        >
          {failures.map((failure, index) => (
            <button
              className="evidenceLink"
              key={failure.failure_id}
              onClick={() => onSelect(failure.failure_id)}
              aria-label={`Inspect region ${index + 1}: ${failure.condition}`}
            >
              <span className="evidenceOrdinal">
                {String(index + 1).padStart(2, "0")}
              </span>
              <span>{failure.condition}</span>
              <span aria-hidden="true">↗</span>
            </button>
          ))}
        </div>
      )}
    </>
  );
}
export function ExperimentChart({
  observations,
}: {
  observations: Observation[];
}) {
  return (
    <IntervalChart
      rows={observations
        .filter((o) => o.effect_size != null)
        .map((o) => ({
          id: o.experiment_id,
          label: `${familyLabel(o.kind ?? "unknown")} · ${o.condition} · ${o.validated ? "validated" : "screened"}`,
          value: o.effect_size!,
          low: o.effect_ci_low,
          high: o.effect_ci_high,
          validated: o.validated,
        }))}
      title="Experiment effect outcomes"
      description="Discovery effects (complement score minus subgroup score) and confidence intervals for executed hypotheses. Holdout gaps appear in region details. Filled circles: validated; outlined circles: screened. A gap alone does not establish a failure."
    />
  );
}
export function RegressionChart({ comparison }: { comparison: CompareResult }) {
  const rows = comparison.summary.deltas
    .filter((d) => d.gap_v1 != null && d.gap_v2 != null)
    .map((d, i) => ({
      id: `MATCH-${i + 1}`,
      label: `${d.status} · ${d.condition_v2 ?? d.condition_v1}`,
      value: d.gap_v2! - d.gap_v1!,
    }));
  return (
    <IntervalChart
      rows={rows}
      title="Matched region gap changes"
      description="V2 minus V1 performance gap for matched regions. Positive values mean a larger gap. New and fixed regions have no paired gap and appear in the transition list below. No confidence intervals for deltas were supplied."
    />
  );
}
