import type { PlatformSettings } from '../types';

const STORAGE_KEY = 'safepp:platformSettings';

const defaultSettings: PlatformSettings = {
  default_base_csv: 'safepp_pytorch/manifests/base_index.csv',
  default_increment_manifest: 'safepp_pytorch/manifests/increment_manifest.jsonl',
  default_output_dir: 'safepp_pytorch/outputs/iterations',
  default_model_registry_dir: 'safepp_pytorch/outputs/model_registry',
  default_log_dir: 'safepp_pytorch/outputs/logs',
  default_seed: 3407,
  default_nproc: 8,
  default_train_plan: 'hard_in_stage2',
  default_val_real_total: 3000,
  default_val_fake_total: 3000,
  run_stage1: true,
  run_stage2: true,
  run_replay: true,
  run_stage3: true,
  run_eval: true,
  val_ap_min_delta: 0,
  val_recall_p98_min_delta: -0.002,
  hard_recall_p98_min_delta: 0.01,
  real_fpr_max_delta: 0.003,
  fake_fnr_max_delta: 0.002,
  require_test_unseen: true,
};

function delay<T>(value: T, ms = 180): Promise<T> {
  return new Promise((resolve) => window.setTimeout(() => resolve(value), ms));
}

export async function getSettings(): Promise<PlatformSettings> {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return delay(defaultSettings);

  try {
    return delay({ ...defaultSettings, ...(JSON.parse(raw) as Partial<PlatformSettings>) });
  } catch {
    return delay(defaultSettings);
  }
}

export async function saveSettings(settings: PlatformSettings): Promise<PlatformSettings> {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  return delay(settings, 240);
}
