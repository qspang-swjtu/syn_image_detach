import { USE_MOCK } from '../config/env';
import { get } from './apiClient';
import type {
  EvalDatasetName,
  EvalMetrics,
  EvaluationSummary,
  PredictionPage,
  PredictionQuery,
} from '../types/evaluation';

function delay<T>(value: T, ms = 180): Promise<T> {
  return new Promise((resolve) => window.setTimeout(() => resolve(value), ms));
}

const datasets: EvalMetrics[] = [
  {
    dataset: 'val',
    exists: true,
    numSamples: 6000,
    acc: 0.9412,
    ap: 0.9588,
    auroc: 0.9631,
    recall_p95: 0.9024,
    recall_p98: 0.8621,
    recall_p99: 0.8119,
    real_fpr: 0.0351,
    fake_fnr: 0.0442,
    metricsPath: 'outputs/iterations/mock/eval/metrics/val.json',
  },
  {
    dataset: 'test_unseen',
    exists: false,
    warning: '当前没有 test_unseen 评估结果，可能是输入数据中没有 split_hint=unseen 的样本。',
  },
  {
    dataset: 'test_all',
    exists: true,
    numSamples: 6000,
    acc: 0.9327,
    ap: 0.9522,
    auroc: 0.9583,
    recall_p95: 0.8908,
    recall_p98: 0.8462,
    recall_p99: 0.8035,
    real_fpr: 0.0412,
    fake_fnr: 0.0524,
    metricsPath: 'outputs/iterations/mock/eval/metrics/test_all.json',
  },
  { dataset: 'hard', exists: true, numSamples: 1200, acc: 0.8875, ap: 0.9137, auroc: 0.9256, recall_p98: 0.7729 },
  { dataset: 'replay', exists: false, warning: 'replay 评估集未生成。' },
];

export async function getEvaluationSummary(iterationId: string): Promise<EvaluationSummary> {
  if (USE_MOCK) {
    return delay({
      iterationId,
      status: 'success',
      outputDir: `outputs/iterations/${iterationId}`,
      datasets,
      warnings: ['当前没有 test_unseen 评估结果，可能是输入数据中没有 split_hint=unseen 的样本。'],
    });
  }
  return get<EvaluationSummary>(`/api/iterations/${iterationId}/evaluation/summary`);
}

export async function getEvalMetrics(iterationId: string, dataset: EvalDatasetName): Promise<EvalMetrics> {
  if (USE_MOCK) {
    return delay(datasets.find((item) => item.dataset === dataset) || { dataset, exists: false });
  }
  return get<EvalMetrics>(`/api/iterations/${iterationId}/evaluation/metrics`, { dataset });
}

export async function getPredictions(iterationId: string, query: PredictionQuery): Promise<PredictionPage> {
  if (USE_MOCK) {
    const allRecords: PredictionPage['records'] = [
      {
        id: 'p1',
        path: 'dataset/val/real/camera_00123.jpg',
        label: 0,
        probability: 0.0832,
        prediction: 0,
        source: 'base',
        generator: 'real',
        split_hint: 'seen',
        is_error: false,
        error_type: 'correct',
      },
      {
        id: 'p2',
        path: 'dataset/val/fake/sdxl_00812.png',
        label: 1,
        probability: 0.9441,
        prediction: 1,
        source: 'increment',
        generator: 'SDXL',
        split_hint: 'seen',
        is_error: false,
        error_type: 'correct',
      },
      {
        id: 'p3',
        path: 'dataset/hard/fake/mj_02091.png',
        label: 1,
        probability: 0.4117,
        prediction: 0,
        source: 'hard',
        generator: 'Midjourney',
        split_hint: 'hard',
        hard_type: 'texture',
        is_error: true,
        error_type: 'false_negative',
      },
      {
        id: 'p4',
        path: 'dataset/val/real/mobile_07412.jpg',
        label: 0,
        probability: 0.681,
        prediction: 1,
        source: 'reviewed_pool',
        generator: 'real',
        split_hint: 'seen',
        is_error: true,
        error_type: 'false_positive',
      },
    ];
    const filtered = allRecords.filter((record) => {
      if (query.errorType && query.errorType !== 'all' && record.error_type !== query.errorType) return false;
      if (query.source && record.source !== query.source) return false;
      if (query.generator && record.generator !== query.generator) return false;
      if (query.splitHint && record.split_hint !== query.splitHint) return false;
      return true;
    });
    const start = (query.page - 1) * query.pageSize;
    return delay({
      iterationId,
      dataset: query.dataset,
      total: filtered.length,
      page: query.page,
      pageSize: query.pageSize,
      records: filtered.slice(start, start + query.pageSize),
    });
  }
  return get<PredictionPage>(`/api/iterations/${iterationId}/evaluation/predictions`, {
    dataset: query.dataset,
    page: query.page,
    page_size: query.pageSize,
    error_type: query.errorType,
    source: query.source,
    generator: query.generator,
    split_hint: query.splitHint,
  });
}
