export type EvalDatasetName = 'val' | 'test_unseen' | 'test_all' | 'hard' | 'replay';

export type EvalMetrics = {
  dataset: EvalDatasetName;
  exists: boolean;
  numSamples?: number;
  acc?: number;
  real_acc?: number;
  fake_acc?: number;
  ap?: number;
  auroc?: number;
  recall_p95?: number;
  recall_p98?: number;
  recall_p99?: number;
  threshold_p95?: number;
  threshold_p98?: number;
  threshold_p99?: number;
  real_fpr?: number;
  fake_fnr?: number;
  metricsPath?: string;
  warning?: string;
};

export type EvaluationSummary = {
  iterationId: string;
  status: 'pending' | 'running' | 'success' | 'failed' | 'missing';
  outputDir: string;
  datasets: EvalMetrics[];
  warnings: string[];
};

export type PredictionRecord = {
  id: string;
  path: string;
  label: 0 | 1;
  probability: number;
  prediction: 0 | 1;
  source?: string;
  generator?: string;
  split_hint?: string;
  hard_type?: string;
  is_error: boolean;
  error_type?: 'false_positive' | 'false_negative' | 'correct';
};

export type PredictionQuery = {
  dataset: EvalDatasetName;
  page: number;
  pageSize: number;
  errorType?: 'false_positive' | 'false_negative' | 'all';
  source?: string;
  generator?: string;
  splitHint?: string;
};

export type PredictionPage = {
  iterationId: string;
  dataset: EvalDatasetName;
  total: number;
  page: number;
  pageSize: number;
  records: PredictionRecord[];
};
