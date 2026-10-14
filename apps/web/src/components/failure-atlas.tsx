import type { AtlasFailure, Counterexample } from "../lib/types";
import { pct, pp, familyLabel } from "../lib/format";
import { Icon, StatusDot, Severity, SectionHeader, EmptyState } from "./ui";
export function FailureAtlas({
  failureCount,
  filteredFailures,
  selectedFailure,
  selectedCounterexamples,
  setSelectedFailureId,
  filter,
  setFilter,
  severityFilter,
  setSeverityFilter,
  sort,
  setSort,
  onInspectExperiment,
}: {
  failureCount: number;
  filteredFailures: AtlasFailure[];
  selectedFailure: AtlasFailure | null;
  selectedCounterexamples: Counterexample[];
  setSelectedFailureId: (id: string) => void;
  filter: string;
  setFilter: (v: string) => void;
  severityFilter: string;
  setSeverityFilter: (v: string) => void;
  sort: string;
  setSort: (v: string) => void;
  onInspectExperiment: (id: string) => void;
}) {
  return (
    <section className="failureWorkspace">
      <div className="failureList contentPanel">
        <SectionHeader
          eyebrow="FAILURE ATLAS"
          title={`${failureCount} validated regions`}
          copy="Select a region to inspect its evidence chain."
        />
        <div className="atlasFilters">
          <label>
            Search regions
            <input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Condition or region ID"
            />
          </label>
          <label>
            Severity
            <select
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
            >
              <option value="ALL">All severities</option>
              {["HIGH", "MEDIUM", "LOW"].map((v) => (
                <option key={v}>{v}</option>
              ))}
            </select>
          </label>
          <label>
            Sort by
            <select value={sort} onChange={(e) => setSort(e.target.value)}>
              <option value="evidence">Evidence score</option>
              <option value="gap">Performance gap</option>
              <option value="support">Support</option>
            </select>
          </label>
          <p role="status">
            {filteredFailures.length} of {failureCount} regions
          </p>
        </div>
        {failureCount > 0 && !filteredFailures.length && (
          <p>No regions match these filters.</p>
        )}
        {failureCount === 0 ? (
          <div className="cleanState">
            <StatusDot tone="green" />
            <div>
              <strong>Atlas is clean.</strong>
              <p>No validated failures were promoted in this run.</p>
            </div>
          </div>
        ) : (
          filteredFailures.map((f) => (
            <button
              key={f.failure_id}
              className={`failureListItem ${selectedFailure?.failure_id === f.failure_id ? "selected" : ""}`}
              onClick={() => setSelectedFailureId(f.failure_id)}
            >
              <div className="failureListTop">
                <Severity value={f.severity} />
                <span>
                  {f.reproducible ? "REPRODUCIBLE" : "NOT REPRODUCED"}
                </span>
              </div>
              <strong>{f.condition}</strong>
              <div className="failureMeta">
                <span>{pp(f.gap)} gap</span>
                <span>{f.evidence_score.toFixed(1)} evidence</span>
                <span>{pct(f.support)} support</span>
              </div>
            </button>
          ))
        )}
      </div>
      <div className="failureDetail contentPanel">
        {selectedFailure ? (
          <>
            <SectionHeader
              eyebrow="SELECTED REGION"
              title={selectedFailure.condition}
              copy={`${selectedFailure.failure_id} · ${selectedFailure.experiment_id}`}
              action={<Severity value={selectedFailure.severity} />}
            />
            <div className="detailGrid">
              <div>
                <small>PERFORMANCE GAP</small>
                <strong>{pp(selectedFailure.gap)}</strong>
              </div>
              <div>
                <small>HOLDOUT GAP</small>
                <strong>{pp(selectedFailure.validation_gap)}</strong>
              </div>
              <div>
                <small>EVIDENCE</small>
                <strong>{selectedFailure.evidence_score.toFixed(1)}</strong>
              </div>
              <div>
                <small>SUPPORT</small>
                <strong>{pct(selectedFailure.support)}</strong>
              </div>
            </div>
            <div className="detailActions">
              <button
                className="quietButton"
                onClick={() =>
                  onInspectExperiment(selectedFailure.experiment_id)
                }
              >
                Inspect source experiment <Icon name="arrow" />
              </button>
              <span>Trace this region back to its executed hypothesis.</span>
            </div>
            <div className="detailSection">
              <div className="subHeading">SEARCH SUPPORT</div>
              <div className="supportList">
                {(selectedFailure.strategy_kinds ?? []).map((kind) => (
                  <div key={kind}>
                    <span>{familyLabel(kind)}</span>
                    <StatusDot tone="green" />
                  </div>
                ))}
                {!selectedFailure.strategy_kinds?.length && (
                  <div>
                    <span>Strategy breadth not reported</span>
                    <StatusDot tone="blue" />
                  </div>
                )}
              </div>
            </div>
            <div className="detailSection">
              <div className="subHeading">STATISTICAL EVIDENCE</div>
              <div className="statLine">
                <span>Adjusted p-value</span>
                <strong>
                  {selectedFailure.adjusted_p_value.toExponential(2)}
                </strong>
              </div>
              <div className="statLine">
                <span>Discovery effect interval</span>
                <strong>
                  {pp(selectedFailure.ci_low)} → {pp(selectedFailure.ci_high)}
                </strong>
              </div>
              <div className="statLine">
                <span>Reproducible</span>
                <strong>
                  {selectedFailure.reproducible ? "YES" : "NOT REPRODUCED"}
                </strong>
              </div>
            </div>
            <div className="detailSection">
              <p className="panelFootnote">
                Discovery effects compare subgroup and complement performance.
                Performance gaps compare the subgroup with the aggregate
                baseline.
              </p>
              <div className="subHeading">BOUNDARY-SENSITIVITY EVIDENCE</div>
              <p className="panelFootnote">
                Prediction changes under small perturbations indicate
                sensitivity; they do not independently prove model failure.
              </p>
              {selectedCounterexamples.length ? (
                selectedCounterexamples.slice(0, 4).map((c, i) => (
                  <div className="miniEvidence" key={c.counterexample_id ?? i}>
                    <span>{c.changed_feature}</span>
                    <strong>
                      {c.original_value.toFixed(4)} →{" "}
                      {c.counterfactual_value.toFixed(4)}
                    </strong>
                    <small>
                      {c.original_prediction} → {c.counterfactual_prediction} ·{" "}
                      {c.search_status}
                    </small>
                  </div>
                ))
              ) : (
                <div className="panelFootnote">
                  No boundary-sensitivity evidence attached to this region.
                </div>
              )}
            </div>
          </>
        ) : (
          <EmptyState
            eyebrow="FAILURE ATLAS"
            title="Select a validated region"
            copy="Choose a failure from the list to inspect its evidence chain."
          />
        )}
      </div>
    </section>
  );
}
