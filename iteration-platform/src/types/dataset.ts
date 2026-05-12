export type SplitHint = 'seen' | 'hard' | 'unseen' | 'reviewed_pool' | string;

export type DatasetInfo = {
  csvPath: string;
  totalRows: number;
  realCount: number;
  fakeCount: number;
  seenCount: number;
  hardCount: number;
  unseenCount: number;
  reviewedPoolCount: number;
  lastModified?: string;
};

export type ManifestSource = {
  name: string;
  path: string;
  label: 0 | 1;
  source?: string;
  dataset?: string;
  domain?: string;
  generator?: string;
  split_hint: SplitHint;
  sample_weight?: number;
  is_hard_negative?: 0 | 1;
  recursive?: boolean;
};

export type SaveManifestRequest = {
  iterationId: string;
  sources: ManifestSource[];
};

export type SaveManifestResponse = {
  iterationId: string;
  manifestPath: string;
  sourceCount: number;
  yamlText: string;
  warnings: string[];
};

export type ManifestPreview = {
  manifestPath: string;
  sources: ManifestSource[];
  estimatedRows?: number;
  warnings: string[];
};

export type ScanIncrementRequest = {
  iterationId: string;
  incrementManifest: string;
};

export type ScanIncrementResponse = {
  iterationId: string;
  incrementCsv: string;
  summaryYaml: string;
  rows: number;
  byLabel: Record<string, number>;
  bySplitHint: Record<string, number>;
};

export type MergeIndexRequest = {
  iterationId: string;
  baseCsv: string;
  incrementManifest?: string;
};

export type MergeIndexResponse = {
  iterationId: string;
  allSamplesCsv: string;
  summaryYaml: string;
  totalRows: number;
  duplicateRemoved: number;
  byLabel: Record<string, number>;
  bySplitHint: Record<string, number>;
};

export type DatasetSplitRequest = {
  iterationId: string;
  inputCsv: string;
  trainPlan: 'hard_in_stage1' | 'hard_in_stage2';
  valRealTotal: number;
  valFakeTotal: number;
  seed: number;
};

export type DatasetSplitSummary = {
  allInput: number;
  seenForSplit: number;
  trainBase: number;
  trainHard: number;
  trainStage1: number;
  trainStage2: number;
  trainStage3Initial: number;
  val: number;
  testUnseen: number;
  testAll: number;
  reviewedPool: number;
  outputDir: string;
  files: {
    trainBaseCsv: string;
    trainHardCsv: string;
    trainStage1Csv: string;
    trainStage2Csv: string;
    trainStage3Csv: string;
    valCsv: string;
    testUnseenCsv: string;
    testAllCsv: string;
    reviewedPoolCsv: string;
  };
  warnings: string[];
};
