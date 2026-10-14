import type { ReactNode } from "react";
export function Icon({
  name,
}: {
  name:
    | "overview"
    | "failures"
    | "experiments"
    | "coverage"
    | "regression"
    | "docs"
    | "arrow"
    | "close"
    | "sun"
    | "moon";
}) {
  const paths: Record<string, string> = {
    overview: "M4 4h7v7H4z M13 4h7v7h-7z M4 13h7v7H4z M13 13h7v7h-7z",
    failures: "M5 19 9 5l6 8 4-10 M4 19h16",
    experiments: "M5 5h14v14H5z M8 9h8 M8 13h6 M8 17h4",
    coverage: "M4 18V9 M9 18V5 M14 18v-7 M19 18V3",
    regression: "M4 16l5-5 4 3 7-8 M4 20h16",
    docs: "M6 4h10l3 3v13H6z M16 4v4h4 M9 12h6 M9 16h6",
    arrow: "M5 12h13 M13 7l5 5-5 5",
    close: "M6 6l12 12 M18 6 6 18",
    sun: "M12 4v2 M12 18v2 M4 12h2 M18 12h2 M6.3 6.3l1.4 1.4 M16.3 16.3l1.4 1.4 M17.7 6.3l-1.4 1.4 M7.7 16.3l-1.4 1.4 M15.5 12a3.5 3.5 0 1 1-7 0 3.5 3.5 0 0 1 7 0",
    moon: "M20 15.2A8.2 8.2 0 0 1 8.8 4 8.3 8.3 0 1 0 20 15.2z",
  };
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d={paths[name]}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
export function StatusDot({
  tone = "blue",
}: {
  tone?: "blue" | "green" | "red" | "amber";
}) {
  return <span className={`statusDot ${tone}`} />;
}
export function Severity({ value }: { value: string }) {
  const tone =
    value.toUpperCase() === "HIGH"
      ? "danger"
      : value.toUpperCase() === "MEDIUM"
        ? "warning"
        : "neutral";
  return (
    <span className={`severityPill ${tone}`}>
      <i />
      {value.toUpperCase()}
    </span>
  );
}
export function SectionHeader({
  eyebrow,
  title,
  copy,
  action,
}: {
  eyebrow: string;
  title: string;
  copy?: string;
  action?: ReactNode;
}) {
  return (
    <div className="sectionHeader">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h2>{title}</h2>
        {copy && <p>{copy}</p>}
      </div>
      {action}
    </div>
  );
}
export function Metric({
  label,
  value,
  detail,
  tone = "blue",
}: {
  label: string;
  value: string;
  detail: string;
  tone?: string;
}) {
  return (
    <div className={`metric tone-${tone}`}>
      <div className="metricTop">
        <span>{label}</span>
        <i />
      </div>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}
export function EmptyState({
  eyebrow,
  title,
  copy,
  primary,
  secondary,
}: {
  eyebrow: string;
  title: string;
  copy: string;
  primary?: ReactNode;
  secondary?: ReactNode;
}) {
  return (
    <div className="emptyState">
      <div className="emptyOrb">
        <span />
      </div>
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h2>{title}</h2>
        <p>{copy}</p>
        <div className="emptyActions">
          {primary}
          {secondary}
        </div>
      </div>
    </div>
  );
}
