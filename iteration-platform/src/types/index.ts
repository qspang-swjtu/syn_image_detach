export * from './dataset';
export * from './iteration';

import type { TrainPlan } from './iteration';

export interface IterationFormValues {
  iteration_id: string;
  base_csv: string;
  increment_manifest: string;
  seed: number;
  nproc: number;
  val_real_total: number;
  val_fake_total: number;
  train_plan: TrainPlan;
}

export type TaskStatus = 'queued' | 'running' | 'completed' | 'failed';

export interface StageStep {
  key: string;
  title: string;
  status: 'wait' | 'process' | 'finish' | 'error';
}

export interface TrendPoint {
  step: number;
  loss: number;
  ap: number;
}

export interface IterationTask {
  id: string;
  status: TaskStatus;
  createdAt: string;
  currentStage: number;
  form: IterationFormValues;
  logs: string[];
  trends: TrendPoint[];
}

export type EvalSplit = 'val' | 'test_unseen' | 'test_all' | 'hard';

export interface MetricRecord {
  split: EvalSplit;
  acc: number;
  ap: number;
  auroc: number;
  recall_p95: number;
  recall_p98: number;
  recall_p99: number;
  real_fpr: number;
  fake_fnr: number;
}

export interface PredictionRecord {
  id: string;
  path: string;
  label: 'real' | 'fake';
  probability: number;
  prediction: 'real' | 'fake';
  source: string;
  generator: string;
  is_error: boolean;
}

export interface CompareRecord {
  metric: string;
  baseline: number;
  candidate: number;
  delta: number;
  result: 'pass' | 'warning' | 'fail';
}

export interface GateResult {
  status: 'pass' | 'review' | 'block';
  summary: string;
  recommendation: string;
  checks: Array<{
    name: string;
    passed: boolean;
    detail: string;
  }>;
}

export interface DashboardSummary {
  productionModel: string;
  latestIteration: string;
  latestStatus: 'success' | 'running' | 'failed';
  valAp: number;
  recallP98: number;
  hardRecallP98: number;
  latestTestUnseen: number;
}

export interface RecentTaskRecord {
  iteration_id: string;
  train_plan: TrainPlan;
  status: 'success' | 'running' | 'failed';
  val_ap: number;
  recall_p98: number;
  created_at: string;
}

export interface MetricTrendRecord {
  version: string;
  ap: number;
  recall_p98: number;
  auroc: number;
  real_fpr: number;
  fake_fnr: number;
}

export type ModelStatus = 'production' | 'candidate' | 'archived' | 'failed';

export interface ModelRecord {
  model_id: string;
  iteration_id: string;
  train_plan: TrainPlan;
  status: ModelStatus;
  ap: number;
  auroc: number;
  recall_p98: number;
  real_fpr: number;
  fake_fnr: number;
  created_at: string;
}

export interface ModelDetail extends ModelRecord {
  checkpoint_path: string;
  config_path: string;
  metrics_dir: string;
  threshold_path: string;
  data_version: string;
  git_commit: string;
}

export interface PlatformSettings {
  default_base_csv: string;
  default_increment_manifest: string;
  default_output_dir: string;
  default_model_registry_dir: string;
  default_log_dir: string;
  default_seed: number;
  default_nproc: number;
  default_train_plan: TrainPlan;
  default_val_real_total: number;
  default_val_fake_total: number;
  run_stage1: boolean;
  run_stage2: boolean;
  run_replay: boolean;
  run_stage3: boolean;
  run_eval: boolean;
  val_ap_min_delta: number;
  val_recall_p98_min_delta: number;
  hard_recall_p98_min_delta: number;
  real_fpr_max_delta: number;
  fake_fnr_max_delta: number;
  require_test_unseen: boolean;
}
