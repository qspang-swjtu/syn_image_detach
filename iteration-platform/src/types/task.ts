import type { TrainPlan } from './iteration';

export type TaskStatus = 'created' | 'pending' | 'running' | 'success' | 'failed' | 'stopped';

export type StageStatus = 'waiting' | 'running' | 'success' | 'failed' | 'skipped';

export type IterationStageName =
  | 'merge_index'
  | 'split_dataset'
  | 'stage1_train'
  | 'stage2_train'
  | 'replay_mining'
  | 'stage3_train'
  | 'evaluation'
  | 'model_compare'
  | 'package';

export type IterationStage = {
  name: IterationStageName;
  title: string;
  status: StageStatus;
  startedAt?: string;
  finishedAt?: string;
  durationSeconds?: number;
  message?: string;
};

export type IterationTaskDetail = {
  iterationId: string;
  status: TaskStatus;
  trainPlan: TrainPlan;
  description?: string;
  dataDir: string;
  outputDir: string;
  allSamplesCsv?: string;
  trainStage1Csv?: string;
  trainStage2Csv?: string;
  trainStage3Csv?: string;
  valCsv?: string;
  testUnseenCsv?: string;
  testAllCsv?: string;
  reviewedPoolCsv?: string;
  dataPrepared: boolean;
  missingPreparedFiles: string[];
  startedAt?: string;
  finishedAt?: string;
  elapsedSeconds?: number;
  progress: number;
  currentStage?: IterationStageName;
  stages: IterationStage[];
  errorMessage?: string;
};

export type TaskLogLevel = 'INFO' | 'WARN' | 'ERROR' | 'DEBUG';

export type TaskLogLine = {
  id: string;
  timestamp: string;
  level: TaskLogLevel;
  stage?: IterationStageName;
  message: string;
};

export type TaskLogsResponse = {
  iterationId: string;
  lines: TaskLogLine[];
  nextCursor?: string;
};

export type TaskMetricPoint = {
  step: number;
  epoch?: number;
  timestamp?: string;
  trainLoss?: number;
  valAp?: number;
  recallP98?: number;
  lr?: number;
};

export type TaskRuntimeMetrics = {
  iterationId: string;
  points: TaskMetricPoint[];
};
