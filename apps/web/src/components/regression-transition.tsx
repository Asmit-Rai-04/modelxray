import type { RegressionDelta } from "../lib/types";
import { pp } from "../lib/format";

export function RegressionTransition({ delta: d }: { delta: RegressionDelta }) {
  return (
    <article className="regressionListRow">
      <div className="regressionStatus">
        <span className={`statusInline ${d.status.toLowerCase()}`}>{d.status}</span>
        <strong>{d.condition_v2 ?? d.condition_v1 ?? "Unmatched region"}</strong>
        <small>{d.experiment_id_v1 ?? "—"} → {d.experiment_id_v2 ?? "—"}</small>
        <details className="transitionEvidence">
          <summary>Inspect transition evidence</summary>
          <div className="transitionEvidenceGrid">
            {(["v1", "v2"] as const).map(version => (
              <div key={version}>
                <h3>Model {version.toUpperCase()}</h3>
                <p className="transitionCondition">{d[`condition_${version}`] ?? "No matched region"}</p>
                <dl>
                  <div><dt>Discovery gap</dt><dd>{d[`gap_${version}`] != null ? pp(d[`gap_${version}`]!) : "—"}</dd></div>
                  <div><dt>Holdout gap</dt><dd>{d[`validation_gap_${version}`] != null ? pp(d[`validation_gap_${version}`]!) : "—"}</dd></div>
                  <div><dt>Evidence score</dt><dd>{d[`evidence_score_${version}`] != null ? d[`evidence_score_${version}`]!.toFixed(1) : "—"}</dd></div>
                  <div><dt>Severity</dt><dd>{d[`severity_${version}`] ?? "—"}</dd></div>
                </dl>
              </div>
            ))}
          </div>
          <p className="panelFootnote">Backend match: {d.match_type?.replaceAll("_", " ") ?? "Not reported"}. Missing values indicate no measurement was returned.</p>
        </details>
      </div>
      <div className="regressionDeltaValue">
        {d.gap_v1 != null && d.gap_v2 != null ? `${pp(d.gap_v1)} → ${pp(d.gap_v2)}` : d.gap_v2 != null ? `NEW · ${pp(d.gap_v2)}` : d.gap_v1 != null ? `FIXED · was ${pp(d.gap_v1)}` : "—"}
      </div>
    </article>
  );
}
