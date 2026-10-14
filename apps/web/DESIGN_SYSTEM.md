# ModelXray visual system

An evidence instrument: a compact navigation rail above an editorial analytical canvas. Overview connects findings to detail; Atlas uses a master/detail layout; Experiments uses an interval plot and dense table; Coverage exposes bounded search breadth; Regression compares versions and transitions.

The landing page includes a four-stage protocol guide with pressed-state buttons, live explanatory text, and a matching SVG highlight. These highlights explain workflow steps; they are never job-progress states. The evidence ledger is collapsed before an investigation and opens when a run is loaded. Three editorial columns explain search, evidence and regression without decorative analytics.

## Tokens and typography

`src/app/globals.css` owns semantic tokens and base component geometry. `instrument.css` owns the instrument shell, readable component scale, responsive rules and accessibility states. Components use CSS variables rather than theme-specific chart colors.

| Role | Dark | Light |
| --- | --- | --- |
| Canvas | Aubergine `#17131c` | Porcelain `#f2efe8` |
| Primary text | Warm neutral `#f4eee7` | Aubergine `#241b2c` |
| Secondary text | Lilac neutral `#c0b5c7` | Plum `#55465f` |
| Accent / positive | Mineral green `#b6d4b0` | Olive `#455e38` |
| Analytical accent | Lilac `#c9b5e0` | Plum `#6b477c` |
| Severity | Rose `#ff9ca8` | Burgundy `#a62f4b` |

Segoe UI/system sans carries controls and prose; Georgia carries editorial headings; Cascadia Code/Consolas carries identifiers and technical labels. Fonts require no network fetch. Body text is 16px; analytical metadata is 12–14px. Headings scale with viewport width. Long conditions wrap instead of hiding the evidence.

Spacing tokens: 4, 8, 12, 16, 24, 32px. Panels use 16–24px padding, 1px semantic borders and a restrained 4px radius. Metrics form a continuous rail rather than floating cards. Severity always includes a text label.

## Component states

Primary actions use `--accent` / `--on-accent`; secondary controls use neutral surfaces and borders. Active navigation uses a border accent and `aria-current`. Disabled actions retain readable labels. Keyboard focus is a 3px outline with offset; upload surfaces expose focus within. Inputs and touch targets are at least 44px high. Errors use `role=alert`; job states and result counts use polite live regions. A skip link targets the workspace.

Empty workspaces explain their own data requirements. The primary investigation action opens upload setup; a separately labeled demo action submits the benchmark. A health label reflects `/health`, initially checking. Upload and comparison failures expose backend messages.

## Evidence graphics

SVG node and edge coordinates share one topology table. Workflow links express relationships, never measured throughput. Boundary sensitivity is a branch from validation; regression is a separate comparison. Narrow screens scroll the diagram locally to preserve readable labels.

Effect plots use the backend's discovery effects and associated confidence intervals in percentage points. Holdout gaps are labeled separately. Regression plots contain only paired V2–V1 gaps, with no invented confidence intervals. Legends explain marks and units; expandable HTML tables expose exact values. Strategy bars display execution divided by generated candidates, with adjacent numerator/denominator text. Search coverage is explicitly bounded by generated hypotheses.

Long plots default to 12 visible entries; the complete axis domain and exact-value table still use every returned entry. An explicit control expands the plot. Confidence-interval legends are shown only when intervals exist. Region buttons connect the plot to Atlas; source-experiment actions open Experiments filtered by the actual experiment ID. Experiment filters operate on returned observations, and transition filters never discard backend regression results. Coverage lists all candidate features and identifies which were absent from a family's executed feature set.

Scrollable topology regions accept keyboard focus. Workspace navigation moves focus to the workspace and returns its context to view. Support percentages appear next to hypothesis gaps so support-based sorting can be assessed from the table.

## Theme, motion and responsiveness

Theme preference is applied before paint with a small initialization script, then maintained by React. Saved choice takes precedence over OS preference; storage restrictions are tolerated. Light and dark use separate semantic palettes, including SVG text and chart lines.

At 1180px the navigation wraps and the landing stacks. At 900px layouts collapse to one column, metrics to two columns, and forms stack. Tables and diagrams scroll within their panels. Desktop content is capped at 1480px. Controls retain visible labels on mobile.

Motion is limited to brief control feedback. No chart animation represents inferred progress. Reduced-motion preference disables animations and transitions.

Automated axe checks cover both themes and all five workspaces plus setup; viewport checks cover 390, 768 and 1440px. Manual screen-reader and cross-browser checks remain separate acceptance work, not implied by automated scans.
