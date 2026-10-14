"use client";
import { useEffect, useState } from "react";
import "./workflow-guide.css";

const steps = [
  { label: "Model", detail: "Trained artifact", stage: 0 },
  { label: "Profile", detail: "Evaluation context", stage: 0 },
  { label: "Probe", detail: "Generate hypotheses", stage: 1 },
  { label: "Experiment", detail: "Budgeted search", stage: 1 },
  { label: "Validate", detail: "FDR + holdout", stage: 2 },
  { label: "Atlas", detail: "Consolidate regions", stage: 3 },
];

export function WorkflowGuide({ stage, onSelect }: { stage: number; onSelect: (stage: number) => void }) {
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    const preference = matchMedia("(prefers-reduced-motion: reduce)");
    const stop = () => { if (preference.matches) setPlaying(false); };
    preference.addEventListener("change", stop);
    if (!playing) return () => preference.removeEventListener("change", stop);
    const timer = window.setInterval(() => onSelect((stage + 1) % 4), 2600);
    return () => { clearInterval(timer); preference.removeEventListener("change", stop); };
  }, [playing, stage, onSelect]);
  return (
    <section className="workflowGuide" aria-label="Investigation workflow guide">
      <div className="mapHeader">
        <div><div className="eyebrow">Investigation anatomy</div><h3>A disciplined path to discovery</h3></div>
      </div>
      <div className="workflowPlayback">
        <span>Explore the workflow</span>
        <button className="textButton" onClick={() => setPlaying(!playing)} aria-pressed={playing}>{playing ? "Pause walkthrough" : "Play walkthrough"}</button>
      </div>
      <ol className="workflowSteps">
        {steps.map((step, index) => (
          <li key={step.label}>
            <button className={`workflowStep ${stage === step.stage ? "selected" : ""}`} aria-label={`Explore ${step.label} step`} aria-pressed={stage === step.stage} onClick={() => { setPlaying(false); onSelect(step.stage); }}>
              <span className="workflowStepNumber" aria-hidden="true">0{index + 1}</span>
              <strong>{step.label}</strong><small>{step.detail}</small>
              <span className="workflowStepArrow" aria-hidden="true">{index < 5 ? "→" : "◎"}</span>
            </button>
          </li>
        ))}
      </ol>
      <div className="workflowBranches">
        <div><span aria-hidden="true">↳</span><div><strong>Boundary sensitivity</strong><small>Branches from validation</small></div></div>
        <div><span aria-hidden="true">⇄</span><div><strong>Regression</strong><small>Compare two model atlases</small></div></div>
      </div>
      <p className="mapAlternative">An illustrative walkthrough of the method, not live job progress. Select a step to inspect its purpose.</p>
    </section>
  );
}
