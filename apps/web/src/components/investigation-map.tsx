"use client";
import { useEffect, useRef } from "react";
import type { Result } from "../lib/types";
// One coordinate table defines both node centers and every edge endpoint.
export const topology = [
  { id: "model", x: 80, y: 110, label: "MODEL" },
  { id: "profile", x: 240, y: 110, label: "PROFILE" },
  { id: "probe", x: 400, y: 110, label: "PROBE" },
  { id: "experiment", x: 560, y: 110, label: "EXPERIMENT" },
  { id: "validate", x: 720, y: 110, label: "VALIDATE" },
  { id: "atlas", x: 880, y: 110, label: "ATLAS" },
  { id: "evidence", x: 720, y: 275, label: "BOUNDARY" },
  { id: "regression", x: 880, y: 275, label: "REGRESSION" },
] as const;
const edges = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [4, 5],
  [4, 6],
  [5, 7],
];
export function InvestigationMap({
  result,
  activeNode,
}: {
  result?: Result | null;
  activeNode?: string;
}) {
  const mapRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const container = mapRef.current;
    const node = topology.find((item) => item.id === activeNode);
    const svg = container?.querySelector("svg");
    if (container && node && svg) {
      // SVG clientWidth is zero in some browsers. Normalize the rendered width
      // back to scroll coordinates so the same calculation also handles zoom.
      const viewportWidth = container.getBoundingClientRect().width;
      const drawingWidth =
        viewportWidth > 0
          ? (svg.getBoundingClientRect().width * container.clientWidth) /
            viewportWidth
          : 0;
      container.scrollTo({
        left: Math.max(
          0,
          (node.x / 960) * drawingWidth - container.clientWidth / 2,
        ),
        behavior: "auto",
      });
    }
  }, [activeNode]);
  const values = result
    ? [
        "trained model",
        `${result.dataset.columns} features`,
        `${result.active_investigation.experiments_considered} candidates`,
        `${result.active_investigation.experiments_executed} executed`,
        "FDR + holdout",
        `${result.failure_atlas.length} regions`,
        `${result.counterexamples.length} items`,
        "compare versions",
      ]
    : [
        "trained artifact",
        "data structure",
        "hypotheses",
        "budgeted search",
        "FDR + holdout",
        "failure regions",
        "sensitivity",
        "two models",
      ];
  return (
    <section className="investigationMap" aria-label="Investigation topology">
      <div className="mapHeader">
        <div>
          <div className="eyebrow">INVESTIGATION ANATOMY</div>
          <h3>
            {result
              ? "From model to evidence"
              : "A disciplined path to discovery"}
          </h3>
        </div>
        <span className="mapLegend">
          <i /> Workflow relationships
        </span>
      </div>
      <div
        className="topologyViewport"
        ref={mapRef}
        role="region"
        aria-label="Investigation diagram"
        tabIndex={0}
      >
        <svg
          viewBox="0 0 960 370"
          role="img"
          aria-label="Model, profile, probe, experiment, validate, atlas. Boundary sensitivity branches from validation; regression compares atlas findings across versions."
        >
          {edges.map(([a, b]) => (
            <path
              key={`${a}-${b}`}
              data-from={topology[a].id}
              data-to={topology[b].id}
              className={`mapTrace ${b > 5 ? "branch" : ""}`}
              d={
                b > 5
                  ? `M${topology[a].x} ${topology[a].y} C${topology[a].x + 100} ${topology[a].y} ${topology[b].x + 100} ${topology[b].y} ${topology[b].x} ${topology[b].y}`
                  : `M${topology[a].x} ${topology[a].y} L${topology[b].x} ${topology[b].y}`
              }
            />
          ))}
          {topology.map((n, i) => (
            <g
              key={n.id}
              data-node={n.id}
              className={`mapNode ${result ? "active" : ""} ${activeNode === n.id ? "protocolSelected" : ""} ${n.id === "atlas" && result && result.failure_atlas.length > 0 ? "red" : "neutral"}`}
              transform={`translate(${n.x} ${n.y})`}
            >
              <title>{`${n.label}: ${values[i]}`}</title>
              {activeNode === n.id && (
                <circle r={35} className="protocolHalo" />
              )}
              <circle r={26} className="nodeCore" />
              <circle r={4} className="nodePoint" />
              <text y={47} textAnchor="middle" className="nodeLabel">
                {n.label}
              </text>
              <text y={64} textAnchor="middle" className="nodeValue">
                {values[i]}
              </text>
            </g>
          ))}
        </svg>
      </div>
      <p className="mapAlternative">
        Model → profile → probe → experiment → validate → atlas. Boundary items
        describe sensitivity. Regression requires a separate two-version
        comparison. Paths show workflow relationships, not measured flows.
      </p>
    </section>
  );
}
