import { USE_MOCK } from '../config/env';
import { get, post } from './apiClient';
import type {
  IterationStage,
  IterationTaskDetail,
  TaskLogLevel,
  TaskLogsResponse,
  TaskRuntimeMetrics,
} from '../types/task';

const stageTitles: Array<IterationStage['name']> = [
  'merge_index',
  'split_dataset',
  'stage1_train',
  'stage2_train',
  'replay_mining',
  'stage3_train',
  'evaluation',
  'model_compare',
  'package',
];

const titleMap: Record<IterationStage['name'], string> = {
  merge_index: '数据合并',
  split_dataset: '数据切分',
  stage1_train: 'Stage1 训练',
  stage2_train: 'Stage2 训练',
  replay_mining: 'Replay Mining',
  stage3_train: 'Stage3 训练',
  evaluation: '自动评估',
  model_compare: '模型对比',
  package: '模型保存',
};

function delay<T>(value: T, ms = 180): Promise<T> {
  return new Promise((resolve) => window.setTimeout(() => resolve(value), ms));
}

function mockStages(running = true): IterationStage[] {
  return stageTitles.map((name, index) => ({
    name,
    title: titleMap[name],
    status: index < 2 ? 'success' : index === 2 && running ? 'running' : index > 6 ? 'skipped' : 'waiting',
    message: index === 2 && running ? 'Stage1 正在训练' : undefined,
  }));
}

export async function startIterationTask(iterationId: string, force = false): Promise<IterationTaskDetail> {
  if (USE_MOCK) {
    return delay({
      iterationId,
      status: 'running',
      trainPlan: 'hard_in_stage2',
      description: 'Mock 训练任务',
      dataDir: `data/iterations/${iterationId}`,
      outputDir: `outputs/iterations/${iterationId}`,
      allSamplesCsv: `data/iterations/${iterationId}/all_samples.csv`,
      trainStage1Csv: `data/iterations/${iterationId}/train_stage1.csv`,
      trainStage2Csv: `data/iterations/${iterationId}/train_stage2.csv`,
      trainStage3Csv: `data/iterations/${iterationId}/train_stage3.csv`,
      valCsv: `data/iterations/${iterationId}/val.csv`,
      testUnseenCsv: `data/iterations/${iterationId}/test_unseen.csv`,
      testAllCsv: `data/iterations/${iterationId}/test_all.csv`,
      reviewedPoolCsv: `data/iterations/${iterationId}/reviewed_pool.csv`,
      dataPrepared: true,
      missingPreparedFiles: [],
      startedAt: new Date().toISOString(),
      elapsedSeconds: 18,
      progress: force ? 24 : 18,
      currentStage: 'stage1_train',
      stages: mockStages(true),
    });
  }
  return post<{ force: boolean }, IterationTaskDetail>(`/api/iterations/${iterationId}/start`, { force });
}

export async function getIterationTask(iterationId: string): Promise<IterationTaskDetail> {
  if (USE_MOCK) {
    return delay({
      iterationId,
      status: 'running',
      trainPlan: 'hard_in_stage2',
      description: 'Mock 训练任务',
      dataDir: `data/iterations/${iterationId}`,
      outputDir: `outputs/iterations/${iterationId}`,
      allSamplesCsv: `data/iterations/${iterationId}/all_samples.csv`,
      trainStage1Csv: `data/iterations/${iterationId}/train_stage1.csv`,
      trainStage2Csv: `data/iterations/${iterationId}/train_stage2.csv`,
      trainStage3Csv: `data/iterations/${iterationId}/train_stage3.csv`,
      valCsv: `data/iterations/${iterationId}/val.csv`,
      testUnseenCsv: `data/iterations/${iterationId}/test_unseen.csv`,
      testAllCsv: `data/iterations/${iterationId}/test_all.csv`,
      reviewedPoolCsv: `data/iterations/${iterationId}/reviewed_pool.csv`,
      dataPrepared: true,
      missingPreparedFiles: [],
      startedAt: new Date(Date.now() - 124000).toISOString(),
      elapsedSeconds: 124,
      progress: 35,
      currentStage: 'stage1_train',
      stages: mockStages(true),
    });
  }
  return get<IterationTaskDetail>(`/api/iterations/${iterationId}`);
}

export async function getIterationStages(iterationId: string): Promise<IterationStage[]> {
  if (USE_MOCK) {
    return delay(mockStages(true));
  }
  const response = await get<{ iterationId: string; stages: IterationStage[] }>(`/api/iterations/${iterationId}/stages`);
  return response.stages;
}

export async function getTaskLogs(
  iterationId: string,
  options?: { cursor?: string; level?: TaskLogLevel },
): Promise<TaskLogsResponse> {
  if (USE_MOCK) {
    const now = new Date().toISOString();
    const lines: TaskLogsResponse['lines'] = [
      { id: '1', timestamp: now, level: 'INFO', stage: 'merge_index', message: 'all_samples.csv 已准备完成' },
      { id: '2', timestamp: now, level: 'INFO', stage: 'split_dataset', message: '数据切分完成，val=6000' },
      { id: '3', timestamp: now, level: 'WARN', stage: 'evaluation', message: 'test_unseen 暂无样本' },
      { id: '4', timestamp: now, level: 'INFO', stage: 'stage1_train', message: 'epoch=1 train_loss=0.431 val_ap=0.942' },
    ];
    const filtered = lines.filter((line) => !options?.level || line.level === options.level);
    return delay({ iterationId, lines: filtered, nextCursor: String(lines.length) });
  }
  return get<TaskLogsResponse>(`/api/iterations/${iterationId}/logs`, options);
}

export async function getTaskRuntimeMetrics(iterationId: string): Promise<TaskRuntimeMetrics> {
  if (USE_MOCK) {
    return delay({
      iterationId,
      points: Array.from({ length: 10 }, (_, index) => ({
        step: index + 1,
        epoch: Math.floor(index / 2) + 1,
        trainLoss: Number((0.62 * Math.exp(-index / 8) + 0.08).toFixed(4)),
        valAp: Number((0.91 + index * 0.004).toFixed(4)),
        recallP98: Number((0.78 + index * 0.006).toFixed(4)),
        lr: Number((0.0002 * Math.exp(-index / 12)).toFixed(6)),
      })),
    });
  }
  return get<TaskRuntimeMetrics>(`/api/iterations/${iterationId}/runtime-metrics`);
}
