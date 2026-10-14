import type { CompareResult } from "./types";
export function pct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}
export function pp(value: number) {
  return `${(value * 100).toFixed(1)} pp`;
}
export function basisLabel(basis?: string) {
  return (basis ?? "failure clusters, then hypotheses").replaceAll("_", " ");
}
export function isDemoComparison(comparison: CompareResult) {
  return comparison.mode !== "UPLOADED_MODEL_REGRESSION";
}
export function familyLabel(key: string) {
  const labels: Record<string, string> = {
    grid: "NUMERIC",
    categorical: "CATEGORICAL",
    interaction: "INTERACTION",
    oblique_band: "OBLIQUE",
    prototype: "NOVELTY",
    prototype_region: "NOVELTY",
  };
  return labels[key] ?? key.replaceAll("_", " ").toUpperCase();
}
