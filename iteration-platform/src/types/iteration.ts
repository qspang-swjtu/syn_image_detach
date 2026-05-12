import type { DatasetSplitSummary } from './dataset';

export type TrainPlan = 'hard_in_stage1' | 'hard_in_stage2';

export type CreateIterationRequest = {
  iterationId: string;
  description?: string;
  baseCsv: string;
  incrementManifest?: string;
  trainPlan: TrainPlan;
  seed: number;
  nproc: number;
  valRealTotal: number;
  valFakeTotal: number;
  runStage1: boolean;
  runStage2: boolean;
  runReplay: boolean;
  runStage3: boolean;
  runEval: boolean;
};

export type IterationCreateResponse = {
  iterationId: string;
  status: 'created';
  dataDir: string;
  outputDir: string;
  allSamplesCsv: string;
  splitSummary: DatasetSplitSummary;
  configPath?: string;
};
