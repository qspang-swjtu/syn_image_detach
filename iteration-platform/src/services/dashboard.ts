import type { DashboardSummary, MetricTrendRecord, RecentTaskRecord } from '../types';

function delay<T>(value: T, ms = 180): Promise<T> {
  return new Promise((resolve) => window.setTimeout(() => resolve(value), ms));
}

const dashboardSummary: DashboardSummary = {
  productionModel: 'sid-20260507-prod',
  latestIteration: 'iter_20260507_001',
  latestStatus: 'success',
  valAp: 0.9821,
  recallP98: 0.881,
  hardRecallP98: 0.793,
  latestTestUnseen: 0,
};

const recentTasks: RecentTaskRecord[] = [
  {
    iteration_id: 'iter_20260507_001',
    train_plan: 'hard_in_stage2',
    status: 'success',
    val_ap: 0.9821,
    recall_p98: 0.881,
    created_at: '2026-05-07 22:18:31',
  },
  {
    iteration_id: 'iter_20260505_003',
    train_plan: 'hard_in_stage1',
    status: 'success',
    val_ap: 0.9794,
    recall_p98: 0.8742,
    created_at: '2026-05-05 21:02:44',
  },
  {
    iteration_id: 'iter_20260504_002',
    train_plan: 'hard_in_stage2',
    status: 'running',
    val_ap: 0.9768,
    recall_p98: 0.8661,
    created_at: '2026-05-04 19:41:12',
  },
  {
    iteration_id: 'iter_20260502_001',
    train_plan: 'hard_in_stage1',
    status: 'failed',
    val_ap: 0.9612,
    recall_p98: 0.8427,
    created_at: '2026-05-02 18:29:03',
  },
  {
    iteration_id: 'iter_20260429_001',
    train_plan: 'hard_in_stage2',
    status: 'success',
    val_ap: 0.9748,
    recall_p98: 0.8619,
    created_at: '2026-04-29 20:12:16',
  },
  {
    iteration_id: 'iter_20260425_001',
    train_plan: 'hard_in_stage1',
    status: 'success',
    val_ap: 0.9706,
    recall_p98: 0.8532,
    created_at: '2026-04-25 22:55:09',
  },
];

const metricTrends: MetricTrendRecord[] = [
  { version: 'sid-20260415', ap: 0.9581, recall_p98: 0.814, auroc: 0.964, real_fpr: 0.048, fake_fnr: 0.079 },
  { version: 'sid-20260420', ap: 0.9664, recall_p98: 0.836, auroc: 0.971, real_fpr: 0.044, fake_fnr: 0.071 },
  { version: 'sid-20260425', ap: 0.9706, recall_p98: 0.8532, auroc: 0.975, real_fpr: 0.042, fake_fnr: 0.066 },
  { version: 'sid-20260429', ap: 0.9748, recall_p98: 0.8619, auroc: 0.978, real_fpr: 0.041, fake_fnr: 0.061 },
  { version: 'sid-20260505', ap: 0.9794, recall_p98: 0.8742, auroc: 0.982, real_fpr: 0.039, fake_fnr: 0.057 },
  { version: 'sid-20260507', ap: 0.9821, recall_p98: 0.881, auroc: 0.984, real_fpr: 0.038, fake_fnr: 0.054 },
];

export async function getDashboardSummary(): Promise<DashboardSummary> {
  return delay(dashboardSummary);
}

export async function getRecentTasks(): Promise<RecentTaskRecord[]> {
  return delay(recentTasks);
}

export async function getMetricTrends(): Promise<MetricTrendRecord[]> {
  return delay(metricTrends);
}
