"use client";
import { useMemo, useState } from "react";
import type { Result } from "../lib/types";
import { pct, pp, familyLabel } from "../lib/format";
import { SectionHeader } from "./ui";
import { ExperimentChart } from "./charts";

export function Experiments({
  result,
  query,
  setQuery,
}: {
  result: Result;
  query: string;
  setQuery: (value: string) => void;
}) {
  const [family, setFamily] = useState("ALL");
  const [state, setState] = useState("ALL");
  const [sort, setSort] = useState("execution");
  const observations = result.active_investigation.observations;
  const families = [...new Set(observations.map((o) => o.kind ?? "unknown"))];
  const filtered = useMemo(
    () =>
      observations
        .filter(
          (o) =>
            `${o.experiment_id} ${o.condition}`
              .toLowerCase()
              .includes(query.toLowerCase()) &&
            (family === "ALL" || (o.kind ?? "unknown") === family) &&
            (state === "ALL" || o.validated === (state === "VALIDATED")),
        )
        .slice()
        .sort((a, b) =>
          sort === "gap"
            ? b.gap - a.gap
            : sort === "support"
              ? b.support - a.support
              : 0,
        ),
    [observations, query, family, state, sort],
  );
  function clear() {
    setQuery("");
    setFamily("ALL");
    setState("ALL");
    setSort("execution");
  }
  return (
    <section className="contentPanel experimentWorkspace">
      <SectionHeader
        eyebrow="EXPERIMENT LAB"
        title="Executed hypotheses"
        copy={`${result.active_investigation.experiments_considered} candidates considered · ${result.active_investigation.experiments_executed} executed · budget ${result.active_investigation.budget}`}
      />
      <div className="analysisToolbar">
        <label className="toolbarSearch">
          Search hypotheses
          <input
            aria-label="Search hypotheses"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Condition or experiment ID"
          />
        </label>
        <label>
          Strategy family
          <select
            aria-label="Strategy family"
            value={family}
            onChange={(e) => setFamily(e.target.value)}
          >
            <option value="ALL">All families</option>
            {families.map((f) => (
              <option key={f} value={f}>
                {familyLabel(f)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Validation state
          <select
            aria-label="Validation state"
            value={state}
            onChange={(e) => setState(e.target.value)}
          >
            <option value="ALL">All outcomes</option>
            <option value="VALIDATED">Validated</option>
            <option value="SCREENED">Screened</option>
          </select>
        </label>
        <label>
          Order hypotheses
          <select
            aria-label="Order hypotheses"
            value={sort}
            onChange={(e) => setSort(e.target.value)}
          >
            <option value="execution">Execution order</option>
            <option value="gap">Greatest discovery gap</option>
            <option value="support">Highest support</option>
          </select>
        </label>
      </div>
      <div className="filterSummary">
        <p role="status">
          Showing {filtered.length} of {observations.length} returned
          hypotheses.
        </p>
        <button
          className="textButton"
          onClick={clear}
          disabled={
            !query &&
            family === "ALL" &&
            state === "ALL" &&
            sort === "execution"
          }
        >
          Clear filters
        </button>
      </div>
      {filtered.length ? (
        <ExperimentChart observations={filtered} />
      ) : (
        <div className="filteredEmpty">
          <h3>No matching hypotheses.</h3>
          <p>
            Change the search, strategy or validation filters to inspect the
            returned experiments.
          </p>
          <button className="quietButton" onClick={clear}>
            Reset experiment filters
          </button>
        </div>
      )}
      <div
        className="experimentTable"
        role="table"
        aria-label="Executed hypotheses"
      >
        <div className="experimentHead" role="row">
          <span role="columnheader">ID</span>
          <span role="columnheader">FAMILY</span>
          <span role="columnheader">HYPOTHESIS</span>
          <span role="columnheader">GAP (pp)</span>
          <span role="columnheader">SUPPORT (%)</span>
          <span role="columnheader">ADJ. P</span>
          <span role="columnheader">STATE</span>
        </div>
        {filtered.map((x) => (
          <div className="experimentRow" role="row" key={x.experiment_id}>
            <span role="cell" className="monoTiny">
              {x.experiment_id}
            </span>
            <span role="cell" className="strategyText">
              {familyLabel(x.kind ?? "unknown")}
            </span>
            <span role="cell" className="hypothesisText">
              {x.condition}
            </span>
            <span role="cell">{pp(x.gap)}</span>
            <span role="cell">{pct(x.support)}</span>
            <span role="cell" className="monoTiny">
              {x.adjusted_p_value !== undefined
                ? x.adjusted_p_value.toExponential(1)
                : "—"}
            </span>
            <span role="cell">
              <span className={`statusInline ${x.validated ? "success" : ""}`}>
                {x.validated ? "VALIDATED" : "SCREENED"}
              </span>
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
