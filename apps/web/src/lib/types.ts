export type Observation = {
  experiment_id: string;
  condition: string;
  kind?: string;
  support: number;
  gap: number;
  p_value: number;
  effect_size?: number;
  effect_ci_low?: number;
  effect_ci_high?: number;
  severity: string;
  validated: boolean;
  validation_gap: number;
  adjusted_p_value?: number;
  evidence_score?: number;
  holdout_effect_ci_low?: number;
  holdout_effect_ci_high?: number;
  holdout_ci_credible?: boolean;
};
export type AtlasFailure = {
  failure_id: string;
  experiment_id: string;
  condition: string;
  severity: string;
  support: number;
  gap: number;
  adjusted_p_value: number;
  effect_size: number;
  ci_low: number;
  ci_high: number;
  validation_gap: number;
  reproducible: boolean;
  evidence_score: number;
  counterexample_ids: string[];
  strategy_kinds?: string[];
  strategy_diversity?: number;
};
export type Counterexample = {
  counterexample_id?: string;
  source_experiment_id: string;
  source_condition: string;
  source_row: number;
  original_prediction: string | number;
  counterfactual_prediction: string | number;
  changed_feature: string;
  original_value: number;
  counterfactual_value: number;
  absolute_change: number;
  relative_change: number;
  search_status: string;
};
export type Result = {
  mode?: string;
  model: { model_type: string; classes: unknown[] };
  dataset: {
    rows: number;
    columns: number;
    numeric_features?: string[];
    categorical_features?: string[];
  };
  baseline: { accuracy: number; error_rate: number };
  metric?: string;
  instabilities: Array<{
    feature: string;
    relative_change: number;
    original_prediction: string;
    perturbed_prediction: string;
  }>;
  experiment_count: number;
  active_investigation: {
    budget: number;
    experiments_considered: number;
    experiments_executed: number;
    observations: Observation[];
    search_coverage?: {
      scope: string;
      interpretation: string;
      candidate_pool_size: number;
      executed: number;
      unexplored_candidates: number;
      candidate_execution_ratio: number;
      available_families: string[];
      explored_families: string[];
      family_breadth_ratio: number;
      candidate_feature_count: number;
      executed_feature_count: number;
      feature_breadth_ratio: number;
      families: Record<
        string,
        {
          generated_candidates: number;
          executed: number;
          unexplored_candidates: number;
          candidate_execution_ratio: number;
          validated_failures: number;
          candidate_features: string[];
          executed_features: string[];
        }
      >;
    };
  };
  investigation_id: string;
  failure_atlas: AtlasFailure[];
  counterexamples: Counterexample[];
};
export type ComparisonModel = {
  accuracy: number;
  model_type?: string;
  asset_id?: string;
  investigation_id?: string;
};
export type RegressionDelta = {
  experiment_id_v1: string | null;
  experiment_id_v2: string | null;
  condition_v1: string | null;
  condition_v2: string | null;
  match_type: string;
  status: "FIXED" | "PERSISTENT" | "NEW";
  severity_v1: string | null;
  severity_v2: string | null;
  gap_v1: number | null;
  gap_v2: number | null;
  evidence_score_v1: number | null;
  evidence_score_v2: number | null;
  evidence_delta: number | null;
  validation_gap_v1: number | null;
  validation_gap_v2: number | null;
  failure_cluster_id_v1?: string | null;
  failure_cluster_id_v2?: string | null;
};
export type CompareResult = {
  mode: string;
  status?: string;
  dataset?: { rows: number; features: string[] };
  models: { v1: ComparisonModel; v2: ComparisonModel };
  summary: {
    fixed: number;
    persistent: number;
    new: number;
    failures_v1: number;
    failures_v2: number;
    net_failure_change: number;
    failure_reduction_ratio: number;
    deltas: RegressionDelta[];
    comparison_basis?: string;
  };
  failure_deltas?: RegressionDelta[];
  comparability?: {
    comparable: boolean;
    reason: string | null;
    warnings: string[];
  } | null;
};
export type UploadedArtifact = {
  asset_id: string;
  filename: string;
  rows?: number;
};
export type Tab =
  "overview" | "failures" | "experiments" | "coverage" | "regression";
