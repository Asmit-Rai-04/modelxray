"use client";
import { useState } from "react";
import { InvestigationMap } from "./investigation-map";
import { Icon, StatusDot } from "./ui";

const protocol = [
  {
    node: "profile",
    label: "Profile",
    title: "Start with the evaluation context.",
    copy: "ModelXray profiles the model and labeled dataset, establishes the selected metric baseline, and identifies the feature types available to the search.",
  },
  {
    node: "experiment",
    label: "Experiment",
    title: "Spend the budget on hypotheses.",
    copy: "Generated hypotheses compete for a bounded experiment budget. The executed search spans the strategy families supported by the data; unexplored candidates remain visible in Coverage.",
  },
  {
    node: "validate",
    label: "Validate",
    title: "Ask the evidence to hold up.",
    copy: "Discovery effects, multiple-testing correction and independent holdout checks determine which findings survive. A prediction flip on its own does not establish model failure.",
  },
  {
    node: "atlas",
    label: "Consolidate",
    title: "Inspect regions, then compare versions.",
    copy: "Validated hypotheses are consolidated into behavioral regions with traceable evidence. A separate regression investigation compares two compatible model versions.",
  },
] as const;

export function Landing({
  busy,
  onInvestigate,
  onDemo,
}: {
  busy: boolean;
  onInvestigate: () => void;
  onDemo: () => void;
}) {
  const [stage, setStage] = useState(0);
  return (
    <section className="landing enhancedLanding">
      <div className="landingCopy">
        <div className="eyebrow blue">
          <span className="instrumentMark" aria-hidden="true" /> MODEL FAILURE
          DISCOVERY
        </div>
        <h1>
          See the behavior
          <br />
          <span>behind the score.</span>
        </h1>
        <p>
          A model can score well and still fail in specific regions. Investigate
          those regions, test their reliability, and follow the evidence behind
          each finding.
        </p>
        <div className="heroActions">
          <button
            className="primaryButton large"
            onClick={onInvestigate}
            disabled={busy}
          >
            Investigate a model
            <Icon name="arrow" />
          </button>
          <button
            className="quietButton large"
            disabled={busy}
            onClick={onDemo}
          >
            Explore benchmark demo
          </button>
        </div>
        <div className="heroMeta">
          <span>
            <StatusDot tone="green" /> bounded search
          </span>
          <span>
            <StatusDot tone="blue" /> BH-FDR + holdout
          </span>
          <span>
            <StatusDot tone="blue" /> persistent evidence
          </span>
        </div>
        <div className="artifactBrief">
          <span className="eyebrow">YOUR STARTING MATERIAL</span>
          <p>
            A trained <code>.joblib / .pkl</code> model.
            <br />A labeled <code>.csv</code> evaluation dataset.
          </p>
          <button
            className="textButton"
            onClick={onInvestigate}
            disabled={busy}
          >
            Configure an investigation <Icon name="arrow" />
          </button>
        </div>
      </div>
      <div className="protocolInstrument">
        <InvestigationMap activeNode={protocol[stage].node} />
        <div
          className="protocolControls"
          role="group"
          aria-label="Explore the investigation protocol"
        >
          {protocol.map((item, i) => (
            <button
              key={item.node}
              aria-pressed={stage === i}
              onClick={() => setStage(i)}
            >
              <span aria-hidden="true">0{i + 1}</span>
              {item.label}
            </button>
          ))}
        </div>
        <div className="protocolDescription" role="status" aria-live="polite">
          <span className="eyebrow">PROTOCOL / 0{stage + 1}</span>
          <h3>{protocol[stage].title}</h3>
          <p>{protocol[stage].copy}</p>
        </div>
      </div>
      <section className="methodStrip" aria-label="Investigation principles">
        <article>
          <span className="methodNumber" aria-hidden="true">
            01
          </span>
          <h2>Search deliberately.</h2>
          <p>
            Explore multiple hypothesis families within a bounded budget.
            Coverage describes the generated search, not every possible
            behavior.
          </p>
        </article>
        <article>
          <span className="methodNumber" aria-hidden="true">
            02
          </span>
          <h2>Keep the evidence.</h2>
          <p>
            Inspect effect intervals, support, adjusted p-values and holdout
            gaps. Boundary sensitivity remains a separate form of evidence.
          </p>
        </article>
        <article>
          <span className="methodNumber" aria-hidden="true">
            03
          </span>
          <h2>Follow the change.</h2>
          <p>
            Compare model versions to identify fixed, persistent and newly
            discovered regions under the backend’s matching criteria.
          </p>
        </article>
      </section>
    </section>
  );
}
