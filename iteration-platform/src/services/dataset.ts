import { USE_MOCK } from '../config/env';
import { get, post } from './apiClient';
import type {
  DatasetInfo,
  DatasetSplitRequest,
  DatasetSplitSummary,
  ManifestPreview,
  MergeIndexRequest,
  MergeIndexResponse,
  ScanIncrementRequest,
  ScanIncrementResponse,
  SaveManifestRequest,
  SaveManifestResponse,
} from '../types';

function delay<T>(value: T, ms = 180): Promise<T> {
  return new Promise((resolve) => window.setTimeout(() => resolve(value), ms));
}

const mockDatasetInfo: DatasetInfo = {
  csvPath: 'safepp_pytorch/manifests/base_index.csv',
  totalRows: 128640,
  realCount: 62300,
  fakeCount: 66340,
  seenCount: 114280,
  hardCount: 8640,
  unseenCount: 5720,
  reviewedPoolCount: 4200,
  lastModified: '2026-05-07 18:30:12',
};

const mockManifestPreview: ManifestPreview = {
  manifestPath: 'safepp_pytorch/manifests/increment_manifest.jsonl',
  estimatedRows: 6280,
  warnings: ['mock manifest 中包含 hard 数据，可用于本轮困难样本增强。'],
  sources: [
    {
      name: 'flux_product_pack_001',
      path: 'data/increment/flux/product_pack',
      label: 1,
      source: 'increment',
      generator: 'FLUX',
      split_hint: 'unseen',
      sample_weight: 1.2,
      is_hard_negative: 0,
      recursive: true,
    },
    {
      name: 'mobile_real_reviewed',
      path: 'data/reviewed/mobile_real',
      label: 0,
      source: 'reviewed_pool',
      generator: '-',
      split_hint: 'reviewed_pool',
      sample_weight: 1,
      is_hard_negative: 0,
      recursive: true,
    },
    {
      name: 'midjourney_hard_faces',
      path: 'data/hard/midjourney_faces',
      label: 1,
      source: 'hard_collection',
      generator: 'Midjourney',
      split_hint: 'hard',
      sample_weight: 1.6,
      is_hard_negative: 1,
      recursive: true,
    },
  ],
};

const mockMergeResponse: MergeIndexResponse = {
  iterationId: 'iter_20260511_001',
  allSamplesCsv: 'data/iterations/iter_20260511_001/all_samples.csv',
  summaryYaml: 'data/iterations/iter_20260511_001/all_samples_summary.yaml',
  totalRows: 134920,
  duplicateRemoved: 128,
  byLabel: { '0': 64260, '1': 70660 },
  bySplitHint: { seen: 119800, hard: 10240, reviewed_pool: 4880 },
};

const mockSplitSummary: DatasetSplitSummary = {
  allInput: 134920,
  seenForSplit: 119800,
  trainBase: 105600,
  trainHard: 10240,
  trainStage1: 105600,
  trainStage2: 115840,
  trainStage3Initial: 115840,
  val: 6000,
  testUnseen: 0,
  testAll: 6000,
  reviewedPool: 4880,
  outputDir: 'data/iterations/iter_20260511_001',
  files: {
    trainBaseCsv: 'data/iterations/iter_20260511_001/train_base.csv',
    trainHardCsv: 'data/iterations/iter_20260511_001/train_hard.csv',
    trainStage1Csv: 'data/iterations/iter_20260511_001/train_stage1.csv',
    trainStage2Csv: 'data/iterations/iter_20260511_001/train_stage2.csv',
    trainStage3Csv: 'data/iterations/iter_20260511_001/train_stage3.csv',
    valCsv: 'data/iterations/iter_20260511_001/val.csv',
    testUnseenCsv: 'data/iterations/iter_20260511_001/test_unseen.csv',
    testAllCsv: 'data/iterations/iter_20260511_001/test_all.csv',
    reviewedPoolCsv: 'data/iterations/iter_20260511_001/reviewed_pool.csv',
  },
  warnings: ['当前没有 split_hint=unseen 的数据，本轮不会生成泛化测试集。'],
};

function stringifyScalar(value: string | number | boolean): string {
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') return String(value);
  if (/^[A-Za-z0-9_./:\\-]+$/.test(value)) return value;
  return JSON.stringify(value);
}

function buildManifestYaml(sources: SaveManifestRequest['sources']): string {
  const lines = ['sources:'];
  sources.forEach((source) => {
    const normalized = {
      name: source.name,
      path: source.path,
      label: source.label,
      source: source.source || source.name,
      dataset: source.dataset || 'unknown',
      domain: source.domain || (source.label === 0 ? 'real' : 'fake'),
      generator: source.generator || (source.label === 0 ? 'real' : 'unknown'),
      split_hint: source.split_hint,
      sample_weight: source.sample_weight ?? 1.0,
      is_hard_negative: source.is_hard_negative ?? 0,
      recursive: source.recursive ?? true,
    };
    lines.push('  - name: ' + stringifyScalar(normalized.name));
    lines.push('    path: ' + stringifyScalar(normalized.path));
    lines.push('    label: ' + normalized.label);
    lines.push('    source: ' + stringifyScalar(normalized.source));
    lines.push('    dataset: ' + stringifyScalar(normalized.dataset));
    lines.push('    domain: ' + stringifyScalar(normalized.domain));
    lines.push('    generator: ' + stringifyScalar(normalized.generator));
    lines.push('    split_hint: ' + stringifyScalar(normalized.split_hint));
    lines.push('    sample_weight: ' + normalized.sample_weight);
    lines.push('    is_hard_negative: ' + normalized.is_hard_negative);
    lines.push('    recursive: ' + stringifyScalar(normalized.recursive));
  });
  return lines.join('\n') + '\n';
}

export async function getBaseDatasetInfo(csvPath: string): Promise<DatasetInfo> {
  if (USE_MOCK) {
    return delay({ ...mockDatasetInfo, csvPath });
  }
  return get<DatasetInfo>(`/api/datasets/base-info?csv_path=${encodeURIComponent(csvPath)}`);
}

export async function getManifestPreview(manifestPath: string): Promise<ManifestPreview> {
  if (USE_MOCK) {
    return delay({ ...mockManifestPreview, manifestPath });
  }
  return get<ManifestPreview>(`/api/datasets/manifest-preview?manifest_path=${encodeURIComponent(manifestPath)}`);
}

export async function scanIncrementManifest(
  iterationId: string,
  incrementManifest: string,
): Promise<ScanIncrementResponse> {
  const req: ScanIncrementRequest = { iterationId, incrementManifest };
  if (USE_MOCK) {
    return delay({
      iterationId,
      incrementCsv: `data/iterations/${iterationId}/increment_from_manifest.csv`,
      summaryYaml: `data/iterations/${iterationId}/increment_from_manifest_summary.yaml`,
      rows: 6280,
      byLabel: { '0': 1960, '1': 4320 },
      bySplitHint: { hard: 2140, reviewed_pool: 1820, seen: 2320 },
    });
  }
  return post<ScanIncrementRequest, ScanIncrementResponse>('/api/datasets/scan-increment', req);
}

export async function mergeDatasetIndex(req: MergeIndexRequest): Promise<MergeIndexResponse> {
  if (USE_MOCK) {
    return delay({
      ...mockMergeResponse,
      iterationId: req.iterationId,
      allSamplesCsv: `data/iterations/${req.iterationId}/all_samples.csv`,
      summaryYaml: `data/iterations/${req.iterationId}/all_samples_summary.yaml`,
    });
  }
  return post<MergeIndexRequest, MergeIndexResponse>('/api/datasets/merge-index', req);
}

export async function previewDatasetSplit(req: DatasetSplitRequest): Promise<DatasetSplitSummary> {
  if (USE_MOCK) {
    return delay({
      ...mockSplitSummary,
      outputDir: `data/iterations/${req.iterationId}`,
      files: {
        trainBaseCsv: `data/iterations/${req.iterationId}/train_base.csv`,
        trainHardCsv: `data/iterations/${req.iterationId}/train_hard.csv`,
        trainStage1Csv: `data/iterations/${req.iterationId}/train_stage1.csv`,
        trainStage2Csv: `data/iterations/${req.iterationId}/train_stage2.csv`,
        trainStage3Csv: `data/iterations/${req.iterationId}/train_stage3.csv`,
        valCsv: `data/iterations/${req.iterationId}/val.csv`,
        testUnseenCsv: `data/iterations/${req.iterationId}/test_unseen.csv`,
        testAllCsv: `data/iterations/${req.iterationId}/test_all.csv`,
        reviewedPoolCsv: `data/iterations/${req.iterationId}/reviewed_pool.csv`,
      },
    });
  }
  return post<DatasetSplitRequest, DatasetSplitSummary>('/api/datasets/split-preview', req);
}

export async function saveIncrementManifest(req: SaveManifestRequest): Promise<SaveManifestResponse> {
  if (USE_MOCK) {
    return delay({
      iterationId: req.iterationId,
      manifestPath: `data/iterations/${req.iterationId}/increment_manifest.yaml`,
      sourceCount: req.sources.length,
      yamlText: buildManifestYaml(req.sources),
      warnings: [],
    });
  }
  return post<SaveManifestRequest, SaveManifestResponse>('/api/datasets/save-increment-manifest', req);
}
