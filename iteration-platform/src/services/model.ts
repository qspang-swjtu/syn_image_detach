import type { ModelDetail, ModelRecord, ModelStatus } from '../types';

function delay<T>(value: T, ms = 180): Promise<T> {
  return new Promise((resolve) => window.setTimeout(() => resolve(value), ms));
}

const modelDetails: ModelDetail[] = [
  {
    model_id: 'sid-20260507-prod',
    iteration_id: 'iter_20260507_001',
    train_plan: 'hard_in_stage2',
    status: 'production',
    ap: 0.9821,
    auroc: 0.984,
    recall_p98: 0.881,
    real_fpr: 0.038,
    fake_fnr: 0.054,
    checkpoint_path: 'registry/sid-20260507-prod/model/best.pt',
    config_path: 'registry/sid-20260507-prod/config/stage3.yaml',
    metrics_dir: 'registry/sid-20260507-prod/metrics',
    threshold_path: 'registry/sid-20260507-prod/threshold/thresholds.json',
    data_version: 'data-20260507-a',
    git_commit: '8f7c1a2',
    created_at: '2026-05-07 22:48:11',
  },
  {
    model_id: 'sid-20260509-cand',
    iteration_id: 'iter_20260509_001',
    train_plan: 'hard_in_stage1',
    status: 'candidate',
    ap: 0.9834,
    auroc: 0.9852,
    recall_p98: 0.8861,
    real_fpr: 0.0391,
    fake_fnr: 0.051,
    checkpoint_path: 'registry/sid-20260509-cand/model/best.pt',
    config_path: 'registry/sid-20260509-cand/config/stage3.yaml',
    metrics_dir: 'registry/sid-20260509-cand/metrics',
    threshold_path: 'registry/sid-20260509-cand/threshold/thresholds.json',
    data_version: 'data-20260509-b',
    git_commit: 'aa91db0',
    created_at: '2026-05-09 23:16:40',
  },
  {
    model_id: 'sid-20260505-arch',
    iteration_id: 'iter_20260505_003',
    train_plan: 'hard_in_stage1',
    status: 'archived',
    ap: 0.9794,
    auroc: 0.982,
    recall_p98: 0.8742,
    real_fpr: 0.039,
    fake_fnr: 0.057,
    checkpoint_path: 'registry/sid-20260505-arch/model/best.pt',
    config_path: 'registry/sid-20260505-arch/config/stage3.yaml',
    metrics_dir: 'registry/sid-20260505-arch/metrics',
    threshold_path: 'registry/sid-20260505-arch/threshold/thresholds.json',
    data_version: 'data-20260505-c',
    git_commit: '5e2f91c',
    created_at: '2026-05-05 21:52:09',
  },
  {
    model_id: 'sid-20260502-failed',
    iteration_id: 'iter_20260502_001',
    train_plan: 'hard_in_stage1',
    status: 'failed',
    ap: 0.9612,
    auroc: 0.968,
    recall_p98: 0.8427,
    real_fpr: 0.052,
    fake_fnr: 0.082,
    checkpoint_path: 'registry/sid-20260502-failed/model/last.pt',
    config_path: 'registry/sid-20260502-failed/config/stage2.yaml',
    metrics_dir: 'registry/sid-20260502-failed/metrics',
    threshold_path: 'registry/sid-20260502-failed/threshold/thresholds.json',
    data_version: 'data-20260502-a',
    git_commit: 'd91b3e8',
    created_at: '2026-05-02 19:11:33',
  },
];

function toModelRecord(detail: ModelDetail): ModelRecord {
  const { checkpoint_path, config_path, metrics_dir, threshold_path, data_version, git_commit, ...record } = detail;
  void checkpoint_path;
  void config_path;
  void metrics_dir;
  void threshold_path;
  void data_version;
  void git_commit;
  return record;
}

export async function getModelList(): Promise<ModelRecord[]> {
  return delay(modelDetails.map(toModelRecord));
}

export async function getModelDetail(modelId: string): Promise<ModelDetail> {
  const detail = modelDetails.find((item) => item.model_id === modelId) ?? modelDetails[0];
  return delay(detail);
}

export async function promoteModel(modelId: string): Promise<ModelRecord> {
  const detail = modelDetails.find((item) => item.model_id === modelId) ?? modelDetails[0];
  const promoted: ModelRecord = { ...toModelRecord(detail), status: 'production' };
  return delay(promoted, 260);
}

export async function archiveModel(modelId: string): Promise<ModelRecord> {
  const detail = modelDetails.find((item) => item.model_id === modelId) ?? modelDetails[0];
  const archivedStatus: ModelStatus = 'archived';
  const archived: ModelRecord = { ...toModelRecord(detail), status: archivedStatus };
  return delay(archived, 260);
}
