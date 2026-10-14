import type { Result } from "../lib/types";
import { pct, familyLabel } from "../lib/format";
import { SectionHeader } from "./ui";
export function SearchCoverage({
  coverage,
}: {
  coverage: Result["active_investigation"]["search_coverage"];
}) {
  return (
    <section className="coveragePage">
      <section className="contentPanel coverageHero">
        <SectionHeader
          eyebrow="SEARCH COVERAGE"
          title={
            coverage
              ? `${pct(coverage.candidate_execution_ratio)} of generated candidates executed`
              : "Coverage unavailable"
          }
          copy={
            coverage?.interpretation ??
            "The backend did not return search coverage for this investigation."
          }
        />
        <div
          className="coverageRing"
          style={{
            background: `conic-gradient(var(--accent2) ${coverage ? coverage.candidate_execution_ratio * 100 : 0}%, var(--line) 0)`,
          }}
        >
          <div>
            <strong>
              {coverage ? pct(coverage.candidate_execution_ratio) : "—"}
            </strong>
            <span>candidate execution</span>
          </div>
        </div>
        <div className="coverageFacts">
          <div>
            <span>GENERATED</span>
            <strong>{coverage?.candidate_pool_size ?? "—"}</strong>
          </div>
          <div>
            <span>EXECUTED</span>
            <strong>{coverage?.executed ?? "—"}</strong>
          </div>
          <div>
            <span>UNEXPLORED</span>
            <strong>{coverage?.unexplored_candidates ?? "—"}</strong>
          </div>
          <div>
            <span>FAMILY BREADTH</span>
            <strong>
              {coverage ? pct(coverage.family_breadth_ratio) : "—"}
            </strong>
          </div>
          <div>
            <span>FEATURE BREADTH</span>
            <strong>
              {coverage ? pct(coverage.feature_breadth_ratio) : "—"}
            </strong>
          </div>
        </div>
      </section>
      {coverage && (
        <div className="familyGrid">
          {Object.entries(coverage.families).map(([key, family]) => (
            <section className="contentPanel familyCard" key={key}>
              <div className="familyTop">
                <span>{familyLabel(key)}</span>
                <b>{pct(family.candidate_execution_ratio)}</b>
              </div>
              <div className="familyBar">
                <span
                  style={{
                    width: `${Math.min(100, family.candidate_execution_ratio * 100)}%`,
                  }}
                />
              </div>
              <div className="familyStats">
                <span>{family.generated_candidates} generated</span>
                <span>{family.executed} executed</span>
                <span>{family.unexplored_candidates} unexplored</span>
                <span>{family.validated_failures} validated</span>
              </div>
              <div className="familyFeatures">
                {family.candidate_features.map((feature) => (
                  <code key={feature}>{feature}</code>
                ))}
              </div>
              <details className="coverageBlindSpots">
                <summary>
                  Unexecuted feature breadth ·{" "}
                  {
                    family.candidate_features.filter(
                      (feature) => !family.executed_features.includes(feature),
                    ).length
                  }
                </summary>
                <p>
                  Features present in this family’s candidate set but absent
                  from its executed hypotheses:
                </p>
                <div className="familyFeatures">
                  {family.candidate_features
                    .filter(
                      (feature) => !family.executed_features.includes(feature),
                    )
                    .map((feature) => (
                      <code key={feature}>{feature}</code>
                    ))}
                </div>
                {family.candidate_features.every((feature) =>
                  family.executed_features.includes(feature),
                ) && (
                  <p>
                    {family.generated_candidates === 0
                      ? "No candidates were generated for this family."
                      : "Every candidate feature appeared in at least one executed hypothesis. This does not mean every condition was tested."}
                  </p>
                )}
              </details>
            </section>
          ))}
        </div>
      )}
      {coverage && (
        <aside className="coverageCaveat">
          <span className="eyebrow">THE LIMIT OF THIS MAP</span>
          <p>
            {coverage.unexplored_candidates} generated candidates remained
            unexecuted. Feature and strategy breadth describe this finite
            candidate set; behaviors outside that set cannot be quantified by
            these ratios.
          </p>
        </aside>
      )}
    </section>
  );
}
