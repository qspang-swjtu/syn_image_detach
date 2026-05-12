import type {
  CompareRecord,
  GateResult,
  IterationFormValues,
  IterationTask,
  MetricRecord,
  PredictionRecord,
  StageStep,
  TaskStatus,
  TrendPoint,
} from '../types';

const STORAGE_KEY = 'safepp:iterationTasks';

const stageTitles = [
  '数据合并',
  '数据切分',
  'Stage1',
  'Stage2',
  'Replay',
  'Stage3',
  '评估',
  '模型对比',
  '模型保存',
];

function delay<T>(value: T, ms = 260): Promise<T> {
  return new Promise((resolve) => window.setTimeout(() => resolve(value), ms));
}

function readTasks(): IterationTask[] {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return [];
  try {
    return JSON.parse(raw) as IterationTask[];
  } catch {
    return [];
  }
}

function writeTasks(tasks: IterationTask[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(tasks));
}

function createTrends(): TrendPoint[] {
  return Array.from({ length: 18 }, (_, index) => {
    const step = index + 1;
    return {
      step,
      loss: Number((0.82 * Math.exp(-index / 8) + 0.08 + Math.sin(index) * 0.012).toFixed(4)),
      ap: Number((0.842 + index * 0.006 + Math.cos(index / 2) * 0.004).toFixed(4)),
    };
  });
}

function createLogs(values: IterationFormValues): string[] {
  const planLabel =
    values.train_plan === 'hard_in_stage1'
      ? 'Stage1=Base+Hard, Stage2=Stage1, Stage3=Stage2+Replay'
      : 'Stage1=Base, Stage2=Base+Hard, Stage3=Stage2+Replay';

  return [
    `[00:00:02] load base csv: ${values.base_csv}`,
    `[00:00:05] load increment manifest: ${values.increment_manifest}`,
    `[00:00:08] merged all_samples.csv generated: outputs/${values.iteration_id}/all_samples.csv`,
    `[00:00:14] split seed=${values.seed}, val real/fake=${values.val_real_total}/${values.val_fake_total}`,
    `[00:01:21] train plan resolved: ${planLabel}`,
    `[00:06:44] Stage1 completed, best val AP=0.9221`,
    `[00:12:18] Stage2 completed, hard recall@p98 improved +3.42%`,
    `[00:17:31] Replay pool sampled from reviewed_pool, size=2400`,
    `[00:25:09] Stage3 completed, candidate checkpoint saved`,
    `[00:28:40] evaluation finished on val/test_unseen/test_all/hard`,
    `[00:29:10] gate passed with 6/7 checks, candidate model ready for review`,
  ];
}

function fallbackTask(taskId: string): IterationTask {
  const form: IterationFormValues = {
    iteration_id: taskId,
    base_csv: 'safepp_pytorch/manifests/base_dataset.csv',
    increment_manifest: 'safepp_pytorch/manifests/increment_manifest.jsonl',
    seed: 3407,
    nproc: 8,
    val_real_total: 3000,
    val_fake_total: 3000,
    train_plan: 'hard_in_stage2',
  };

  return {
    id: taskId,
    status: 'completed',
    createdAt: new Date().toISOString(),
    currentStage: 8,
    form,
    logs: createLogs(form),
    trends: createTrends(),
  };
}

export const mockApi = {
  async createIteration(values: IterationFormValues): Promise<IterationTask> {
    const task: IterationTask = {
      id: values.iteration_id,
      status: 'running',
      createdAt: new Date().toISOString(),
      currentStage: 3,
      form: values,
      logs: createLogs(values).slice(0, 7),
      trends: createTrends(),
    };

    const tasks = readTasks().filter((item) => item.id !== task.id);
    writeTasks([task, ...tasks]);
    localStorage.setItem('safepp:lastTaskId', task.id);
    return delay(task);
  },

  async getTask(taskId: string): Promise<IterationTask> {
    const task = readTasks().find((item) => item.id === taskId) ?? fallbackTask(taskId);
    return delay(task);
  },

  async completeTask(taskId: string): Promise<IterationTask> {
    const tasks = readTasks();
    const index = tasks.findIndex((item) => item.id === taskId);
    const current = index >= 0 ? tasks[index] : fallbackTask(taskId);
    const completed: IterationTask = {
      ...current,
      status: 'completed',
      currentStage: 8,
      logs: createLogs(current.form),
    };
    if (index >= 0) {
      tasks[index] = completed;
      writeTasks(tasks);
    }
    return delay(completed, 180);
  },

  getStages(task: IterationTask): StageStep[] {
    return stageTitles.map((title, index) => ({
      key: title,
      title,
      status:
        task.status === 'failed' && index === task.currentStage
          ? 'error'
          : index < task.currentStage
            ? 'finish'
            : index === task.currentStage
              ? task.status === 'completed'
                ? 'finish'
                : 'process'
              : 'wait',
    }));
  },

  getStatusColor(status: TaskStatus) {
    return {
      queued: 'default',
      running: 'processing',
      completed: 'success',
      failed: 'error',
    }[status];
  },

  async getEvaluation(): Promise<{
    metrics: MetricRecord[];
    predictions: PredictionRecord[];
  }> {
    return delay({
      metrics: [
        {
          split: 'val',
          acc: 0.9412,
          ap: 0.9588,
          auroc: 0.9631,
          recall_p95: 0.9024,
          recall_p98: 0.8621,
          recall_p99: 0.8119,
          real_fpr: 0.0351,
          fake_fnr: 0.0442,
        },
        {
          split: 'test_unseen',
          acc: 0.9184,
          ap: 0.9445,
          auroc: 0.9518,
          recall_p95: 0.8732,
          recall_p98: 0.8314,
          recall_p99: 0.7826,
          real_fpr: 0.0469,
          fake_fnr: 0.0615,
        },
        {
          split: 'test_all',
          acc: 0.9327,
          ap: 0.9522,
          auroc: 0.9583,
          recall_p95: 0.8908,
          recall_p98: 0.8462,
          recall_p99: 0.8035,
          real_fpr: 0.0412,
          fake_fnr: 0.0524,
        },
        {
          split: 'hard',
          acc: 0.8875,
          ap: 0.9137,
          auroc: 0.9256,
          recall_p95: 0.8241,
          recall_p98: 0.7729,
          recall_p99: 0.7038,
          real_fpr: 0.0618,
          fake_fnr: 0.0921,
        },
      ],
      predictions: [
        {
          id: 'p1',
          path: 'dataset/val/real/camera_00123.jpg',
          label: 'real',
          probability: 0.0832,
          prediction: 'real',
          source: 'base',
          generator: '-',
          is_error: false,
        },
        {
          id: 'p2',
          path: 'dataset/test_unseen/fake/sdxl_00812.png',
          label: 'fake',
          probability: 0.9441,
          prediction: 'fake',
          source: 'increment',
          generator: 'SDXL',
          is_error: false,
        },
        {
          id: 'p3',
          path: 'dataset/hard/fake/mj_02091.png',
          label: 'fake',
          probability: 0.4117,
          prediction: 'real',
          source: 'hard',
          generator: 'Midjourney',
          is_error: true,
        },
        {
          id: 'p4',
          path: 'dataset/reviewed/real/mobile_07412.jpg',
          label: 'real',
          probability: 0.2864,
          prediction: 'real',
          source: 'reviewed_pool',
          generator: '-',
          is_error: false,
        },
        {
          id: 'p5',
          path: 'dataset/test_all/fake/flux_00031.webp',
          label: 'fake',
          probability: 0.7218,
          prediction: 'fake',
          source: 'increment',
          generator: 'FLUX',
          is_error: false,
        },
      ],
    });
  },

  async getComparison(): Promise<{
    records: CompareRecord[];
    gate: GateResult;
  }> {
    const rows = [
      ['val_ap', 0.9491, 0.9588, 'pass'],
      ['test_unseen_ap', 0.9324, 0.9445, 'pass'],
      ['hard_recall@p98', 0.7218, 0.7729, 'pass'],
      ['test_all_auroc', 0.9538, 0.9583, 'pass'],
      ['real_fpr', 0.0394, 0.0412, 'warning'],
      ['fake_fnr', 0.0587, 0.0524, 'pass'],
      ['hard_acc', 0.8616, 0.8875, 'pass'],
    ] as const;

    return delay({
      records: rows.map(([metric, baseline, candidate, result]) => ({
        metric,
        baseline,
        candidate,
        delta: Number((candidate - baseline).toFixed(4)),
        result,
      })),
      gate: {
        status: 'pass',
        summary: 'candidate 相比 baseline 在 hard 集和 test_unseen 集上有稳定收益，real_fpr 轻微上升但未超过阈值。',
        recommendation:
          '建议保存 candidate 模型与本轮报告，进入人工复核和灰度验证流程。',
        checks: [
          { name: 'test_unseen AP 不下降', passed: true, detail: '+1.21%' },
          { name: 'hard recall@p98 提升', passed: true, detail: '+5.11%' },
          { name: 'real_fpr 阈值', passed: true, detail: '0.0412 <= 0.0450' },
          { name: 'fake_fnr 不劣化', passed: true, detail: '-0.63%' },
        ],
      },
    });
  },
};
