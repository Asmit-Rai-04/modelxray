import type { Result, Tab } from "../lib/types";
import { pct, pp } from "../lib/format";
import { Icon, StatusDot, Severity, SectionHeader, Metric } from "./ui";
import { InvestigationMap } from "./investigation-map";
import { EvidenceChart } from "./charts";
export function Overview({
  result,
  failureCount,
  high,
  validated,
  reproducible,
  coverage,
  selectedFailureId,
  setSelectedFailureId,
  setActiveTab,
  severityCounts,
}: {
  result: Result;
  failureCount: number;
  high: number;
  validated: number;
  reproducible: number;
  coverage: Result["active_investigation"]["search_coverage"];
  selectedFailureId: string | null;
  setSelectedFailureId: (id: string) => void;
  setActiveTab: (tab: Tab) => void;
  severityCounts: Array<{ label: string; count: number }>;
}) {
  return (
    <>
      <div className="metricGrid">
        <Metric
          label={(result.metric ?? "accuracy").replace("_", " ").toUpperCase()}
          value={pct(result.baseline.accuracy)}
          detail="validation baseline"
          tone="cyan"
        />
        <Metric
          label="VALIDATED FAILURES"
          value={`${failureCount}`}
          detail={`${reproducible} reproducible regions`}
          tone="red"
        />
        <Metric
          label="HIGH SEVERITY"
          value={`${high}`}
          detail="validated priority regions"
          tone="amber"
        />
        <Metric
          label="EXPERIMENTS"
          value={`${result.active_investigation.experiments_executed}/${result.active_investigation.budget}`}
          detail={`${validated} cleared validation`}
        />
        <Metric
          label="SEARCH COVERAGE"
          value={coverage ? pct(coverage.candidate_execution_ratio) : "—"}
          detail="generated candidates executed"
          tone="green"
        />
      </div>
      <InvestigationMap result={result} />
      <section className="contentPanel">
        <EvidenceChart
          failures={result.failure_atlas}
          onSelect={(id) => {
            setSelectedFailureId(id);
            setActiveTab("failures");
          }}
        />
      </section>
      <div className="overviewGrid">
        <section className="contentPanel wide">
          <SectionHeader
            eyebrow="FAILURE REGIONS"
            title="Where the model becomes unreliable"
            copy="Overlapping validated hypotheses are consolidated into behavioral regions."
            action={
              <button
                className="textButton"
                onClick={() => setActiveTab("failures")}
              >
                Open atlas <Icon name="arrow" />
              </button>
            }
          />
          {failureCount === 0 ? (
            <div className="cleanState">
              <StatusDot tone="green" />
              <div>
                <strong>No validated failures found.</strong>
                <p>
                  The investigation did not find a region that survived the
                  statistical and holdout gates.
                </p>
              </div>
            </div>
          ) : (
            <div className="clusterList">
              {result.failure_atlas.slice(0, 8).map((f, index) => (
                <button
                  key={f.failure_id}
                  className={`clusterRow ${selectedFailureId === f.failure_id ? "selected" : ""}`}
                  onClick={() => {
                    setSelectedFailureId(f.failure_id);
                    setActiveTab("failures");
                  }}
                >
                  <span className="clusterIndex">0{index + 1}</span>
                  <span className="clusterMain">
                    <strong>{f.condition}</strong>
                    <small>
                      {f.failure_id} ·{" "}
                      {f.strategy_diversity != null || f.strategy_kinds?.length
                        ? `${f.strategy_diversity ?? f.strategy_kinds?.length} strategy families`
                        : "strategy breadth not reported"}
                    </small>
                  </span>
                  <Severity value={f.severity} />
                  <span className="clusterMetric">
                    <b>{pp(f.gap)}</b>
                    <small>gap</small>
                  </span>
                  <span className="clusterMetric">
                    <b>{f.evidence_score.toFixed(1)}</b>
                    <small>evidence</small>
                  </span>
                  <span className="clusterMetric">
                    <b>{pct(f.support)}</b>
                    <small>support</small>
                  </span>
                </button>
              ))}
            </div>
          )}
        </section>
        <section className="contentPanel">
          <SectionHeader
            eyebrow="MODEL SNAPSHOT"
            title="Investigation context"
          />
          <div className="factStack">
            <div>
              <span>MODEL</span>
              <strong>{result.model.model_type}</strong>
            </div>
            <div>
              <span>NUMERIC FEATURES</span>
              <strong>{result.dataset.numeric_features?.length ?? 0}</strong>
            </div>
            <div>
              <span>CATEGORICAL FEATURES</span>
              <strong>
                {result.dataset.categorical_features?.length ?? 0}
              </strong>
            </div>
            <div>
              <span>HYPOTHESES CONSIDERED</span>
              <strong>
                {result.active_investigation.experiments_considered}
              </strong>
            </div>
            <div>
              <span>BOUNDARY EVIDENCE</span>
              <strong>{result.counterexamples.length}</strong>
            </div>
          </div>
          <div className="panelFootnote">
            Coverage is relative to the generated candidate pool.
          </div>
        </section>
        <section className="contentPanel">
          <SectionHeader eyebrow="SEVERITY MIX" title="Failure distribution" />
          {severityCounts.map((item) => (
            <div className="severityLine" key={item.label}>
              <div>
                <Severity value={item.label} />
                <strong>{item.count}</strong>
              </div>
              <div className="severityTrack">
                <span
                  className={item.label.toLowerCase()}
                  style={{
                    width: `${Math.min(100, (item.count / Math.max(1, failureCount)) * 100)}%`,
                  }}
                />
              </div>
            </div>
          ))}
          <div className="panelFootnote">
            Severity is a transparent heuristic, not a probability of harm.
          </div>
        </section>
      </div>
    </>
  );
}
