#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_BIN="${PYTHON_BIN//\\//}"
TORCHRUN_CMD="${TORCHRUN_CMD:-}"

run_python() {
  "${PYTHON_BIN}" "$@"
}

run_torchrun() {
  if [[ "${NPROC}" -le 1 ]]; then
    run_python "$@"
    return
  fi
  if [[ -n "${TORCHRUN_CMD}" ]]; then
    # shellcheck disable=SC2086
    ${TORCHRUN_CMD} --nproc_per_node="${NPROC}" "$@"
  else
    "${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node="${NPROC}" "$@"
  fi
}
# 定位项目根目录。
# 兼容两种放置方式：
#   1) 推荐：safepp_pytorch_v2.0/scripts/run_iteration.sh
#   2) 兼容：safepp_pytorch_v2.0/src/scripts/run_iteration.sh
# 之前如果脚本放在 src/scripts/ 下，只向上一级会把 ROOT_DIR 误判成 src，
# 后续 python src/tools/... 就会变成 src/src/tools/...，导致找不到文件。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/../src/tools" && -d "${SCRIPT_DIR}/../configs" ]]; then
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -d "${SCRIPT_DIR}/../../src/tools" && -d "${SCRIPT_DIR}/../../configs" ]]; then
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
elif [[ -d "$(pwd)/src/tools" && -d "$(pwd)/configs" ]]; then
  ROOT_DIR="$(pwd)"
else
  echo "[ERROR] Cannot locate project root from SCRIPT_DIR=${SCRIPT_DIR}" >&2
  echo "[ERROR] Please put this script under <project>/scripts/ or <project>/src/scripts/, or run it from project root." >&2
  exit 1
fi
cd "${ROOT_DIR}"

# =========================
# 参数配置区
# 使用方式：在命令行前通过环境变量覆盖，例如：
#   ITER_ID=iter_20260507 BASE_CSV=/data/base_index.csv INCREMENT_MANIFEST=manifests/new_data.yaml TRAIN_PLAN=hard_in_stage2 bash scripts/run_iteration.sh
# 未显式传入的参数会使用下面的默认值。
# =========================

# 本轮迭代 ID。用于区分每次训练的 data/outputs/release 目录。
ITER_ID="${ITER_ID:-$(date +%Y%m%d_%H%M%S)}"

# 数据切分模式：
# - full_seen_random_val：从 seen 数据中随机/分组扣出 val，剩余 seen 全量训练。当前推荐模式。
# - full_seen_heldout_val：旧模式，按完整 source/generator 扣出 val。
# - 其他值：回退到 make_small_splits.py 的 smoke/mini/pilot 抽样逻辑。
SPLIT_MODE="${SPLIT_MODE:-full_seen_random_val}"

# 训练方案：
# - hard_in_stage1：Stage1 使用基础数据 + 已收集 hard；Stage2 使用与 Stage1 相同数据做鲁棒微调。
# - hard_in_stage2：Stage1 只用基础数据；Stage2 加入已收集 hard 做微调。
TRAIN_PLAN="${TRAIN_PLAN:-hard_in_stage1}"

# 小规模抽样预设。仅在非 full_seen_* 切分模式下使用。
# 可选值通常为 smoke / mini / pilot。
PRESET="${PRESET:-mini}"

# torchrun 使用的进程数，通常等于使用的 GPU 数。
NPROC="${NPROC:-1}"
CUDA_DEVICE_COUNT="${CUDA_DEVICE_COUNT:-$(run_python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)}"
if [[ "${CUDA_DEVICE_COUNT}" =~ ^[0-9]+$ ]]; then
  if [[ "${CUDA_DEVICE_COUNT}" -gt 0 && "${NPROC}" -gt "${CUDA_DEVICE_COUNT}" ]]; then
    echo "[WARN] NPROC=${NPROC} exceeds visible CUDA devices=${CUDA_DEVICE_COUNT}; use NPROC=${CUDA_DEVICE_COUNT}"
    NPROC="${CUDA_DEVICE_COUNT}"
  elif [[ "${CUDA_DEVICE_COUNT}" -eq 0 && "${NPROC}" -gt 1 ]]; then
    echo "[WARN] CUDA is not available; use NPROC=1"
    NPROC="1"
  fi
fi

# 随机种子，用于数据切分、采样和训练复现。
SEED="${SEED:-3407}"

# Stage1 基础配置文件路径。
BASE_STAGE1_CONFIG="${BASE_STAGE1_CONFIG:-configs/stage1.yaml}"

# Stage2 基础配置文件路径。Stage3 默认也基于该配置改写。
BASE_STAGE2_CONFIG="${BASE_STAGE2_CONFIG:-configs/stage2.yaml}"

# 评估基础配置文件路径。
BASE_EVAL_CONFIG="${BASE_EVAL_CONFIG:-configs/eval.yaml}"

# 新旧模型对比 / 晋级门禁配置文件路径。
GATE_CONFIG="${GATE_CONFIG:-configs/gate.yaml}"

# 兼容模式：完整数据集 manifest。日常迭代不推荐使用，只在没有 BASE_CSV 且显式 RUN_FULL_SCAN=1 时才会全量扫描。
MANIFEST="${MANIFEST:-manifests/model_iteration_sources_example.yaml}"

# 已经展开到图片级别的完整 CSV。提供后会直接复制为本轮 all_samples.csv，跳过 BASE_CSV + INCREMENT_MANIFEST。
# 适合调试或复现实验，不是日常增量迭代的推荐入口。
INPUT_CSV="${INPUT_CSV:-}"

# =========================
# 增量索引模式：日常迭代推荐入口
# =========================

# 基础数据集图片级 CSV，长期稳定，不建议每轮修改。
# 这个 CSV 已经包含基础数据集中每张图片的 path、label、source、generator、split_hint 等字段。
# 每次迭代时，脚本不会重新扫描基础数据，只会读取这个 CSV。
BASE_CSV="${BASE_CSV:-/ai_paas_jf/pangqs/AIGC/safepp_pytorch_v2.0/src/data/iterations/20260507_093541/all_samples.csv}"

# 本轮新增数据 manifest，只记录“本轮新收集图片目录”的路径和元信息。
# 脚本会只扫描这个 manifest 中的新数据目录，生成临时 increment_from_manifest.csv，
# 然后与 BASE_CSV 合并成本轮 all_samples.csv。
# 如果本轮没有新增数据，可以留空；脚本会只用 BASE_CSV 继续切分和训练。
INCREMENT_MANIFEST="${INCREMENT_MANIFEST:-}"

# 合并去重保留策略：
# - last：同 path 重复时保留新增 manifest 扫描出的记录，适合修正标签/更新 split_hint。
# - first：同 path 重复时保留基础 CSV 中的记录。
MERGE_KEEP="${MERGE_KEEP:-last}"

# 合并去重依据，默认按图片 path 去重。
MERGE_DEDUP_KEY="${MERGE_DEDUP_KEY:-path}"

# 写入新增样本 added_iter 字段的值，默认使用本轮 ITER_ID。
MERGE_ADDED_ITER="${MERGE_ADDED_ITER:-${ITER_ID}}"

# 是否严格检查图片路径存在：
# - 0：不检查，速度快，适合路径在训练机器上才可见的场景。
# - 1：检查 path 是否存在，不存在则报错。
MERGE_STRICT_PATHS="${MERGE_STRICT_PATHS:-0}"

# 是否允许在没有 BASE_CSV / INPUT_CSV 时回退到 MANIFEST 全量扫描：
# - 0：不允许，避免误触发全量扫描。当前推荐。
# - 1：允许，兼容旧流程。
RUN_FULL_SCAN="${RUN_FULL_SCAN:-0}"

# 已收集 hard 样本 CSV，可选。会和主 CSV 中标记为 hard 的样本一起组成 train_hard.csv。
HARD_CSV="${HARD_CSV:-}"

# Stage3 replay 候选池 CSV，可选。REPLAY_CANDIDATE_CSV 是兼容旧变量名。
REVIEWED_POOL_CSV="${REVIEWED_POOL_CSV:-${REPLAY_CANDIDATE_CSV:-}}"

# 本轮数据输出目录。会生成 all_samples.csv、train_stage1.csv、val.csv 等。
DATA_DIR="${DATA_DIR:-data/iterations/${ITER_ID}}"

# 本轮训练、评估、打包输出目录。
OUT_DIR="${OUT_DIR:-outputs/iterations/${ITER_ID}}"

# 旧抽样切分模式中的 unseen generator/source 配置。full_seen_random_val 下通常不需要。
HOLDOUT_GENERATORS="${HOLDOUT_GENERATORS:-}"
HOLDOUT_SOURCES="${HOLDOUT_SOURCES:-}"

# 旧抽样切分模式中用于平衡采样的分组列。
GROUP_COL="${GROUP_COL:-source}"

# =========================
# full_seen_random_val 切分参数
# =========================

# 从 seen 的真实样本中抽取多少条作为验证集。
VAL_REAL_TOTAL="${VAL_REAL_TOTAL:-20000}"

# 从 seen 的合成样本中抽取多少条作为验证集。
VAL_FAKE_TOTAL="${VAL_FAKE_TOTAL:-20000}"

# 验证集抽样时是否允许 hard 样本进入 val：
# - 1：允许，验证集更贴近当前数据总体。
# - 0：不允许，hard 样本全部留给训练。
VAL_INCLUDE_HARD="${VAL_INCLUDE_HARD:-1}"

# 抽取真实验证集时按哪个列做分组均衡。
VAL_REAL_GROUP_COL="${VAL_REAL_GROUP_COL:-source}"

# 抽取合成验证集时按哪个列做分组均衡。
VAL_FAKE_GROUP_COL="${VAL_FAKE_GROUP_COL:-generator}"

# 哪些 split_hint 值会被识别为已收集 hard 样本。
HARD_HINTS="${HARD_HINTS:-hard,hard_negative,collected_hard,train_hard}"

# hard 标记列名，默认 is_hard_negative。
HARD_FLAG_COL="${HARD_FLAG_COL:-is_hard_negative}"

# hard 标记列中哪些值表示 true。
HARD_FLAG_VALUES="${HARD_FLAG_VALUES:-1,true,yes,y}"

# 哪些 split_hint 值会被识别为 reviewed_pool / replay 候选池。
REVIEWED_POOL_HINTS="${REVIEWED_POOL_HINTS:-reviewed,reviewed_pool,replay_candidate,candidate}"

# 哪些 split_hint 值会被识别为 unseen 泛化测试集。
TEST_UNSEEN_HINTS="${TEST_UNSEEN_HINTS:-unseen,test_unseen}"

# =========================
# 旧 whole-source/generator holdout 切分参数，保留兼容
# =========================

# 旧模式：整体扣出的真实 source 列表。默认兼容 HOLDOUT_SOURCES。
REAL_HOLDOUT_SOURCES="${REAL_HOLDOUT_SOURCES:-${HOLDOUT_SOURCES}}"

# 旧模式：整体扣出的合成 generator 列表。默认兼容 HOLDOUT_GENERATORS。
FAKE_HOLDOUT_GENERATORS="${FAKE_HOLDOUT_GENERATORS:-${HOLDOUT_GENERATORS}}"

# 旧模式：整体扣出的合成 source 列表。
FAKE_HOLDOUT_SOURCES="${FAKE_HOLDOUT_SOURCES:-}"

# 旧模式：是否自动选择真实 source 作为验证集。
AUTO_HOLDOUT_REAL_SOURCES="${AUTO_HOLDOUT_REAL_SOURCES:-0}"

# 旧模式：是否自动选择合成 generator 作为验证集。
AUTO_HOLDOUT_FAKE_GENERATORS="${AUTO_HOLDOUT_FAKE_GENERATORS:-0}"

# 当某个 source 的父目录分组过少时，是否用 hash bucket 重新分桶。
FLAT_DIR_BUCKET_THRESHOLD="${FLAT_DIR_BUCKET_THRESHOLD:-1}"

# hash 重新分桶数量。
HASH_BUCKETS="${HASH_BUCKETS:-128}"

# =========================
# 流程开关：1 表示执行，0 表示跳过
# =========================

# 是否执行完整 MANIFEST 扫描。
# 日常增量迭代不需要打开；只有 RUN_FULL_SCAN=1 且没有 BASE_CSV / INPUT_CSV 时才会使用。
RUN_SCAN="${RUN_SCAN:-${RUN_FULL_SCAN}}"

# 是否执行数据切分。
RUN_SPLIT="${RUN_SPLIT:-1}"

# 是否直接使用由平台提前准备好的 DATA_DIR。
# USE_PREPARED_DATA=1 时跳过 BASE_CSV + INCREMENT_MANIFEST 合并、MANIFEST 扫描和数据切分。
USE_PREPARED_DATA="${USE_PREPARED_DATA:-0}"

# 是否训练 Stage1。
RUN_STAGE1="${RUN_STAGE1:-1}"

# 是否训练 Stage2。
RUN_STAGE2="${RUN_STAGE2:-1}"

# 是否从 reviewed_pool 挖 replay 样本。
RUN_REPLAY="${RUN_REPLAY:-1}"

# 是否训练 Stage3。
RUN_STAGE3="${RUN_STAGE3:-1}"

# 是否评估最终模型。
RUN_EVAL="${RUN_EVAL:-1}"

# 是否执行新旧模型 gate 对比。
RUN_GATE="${RUN_GATE:-1}"

# 是否打包 release。
RUN_PACKAGE="${RUN_PACKAGE:-1}"

# 如果 train_stage3.csv 与 train_stage2.csv 完全相同，是否仍然训练 Stage3：
# - 0：跳过 Stage3，避免重复训练。
# - 1：即使数据相同也训练 Stage3。
TRAIN_STAGE3_IF_UNCHANGED="${TRAIN_STAGE3_IF_UNCHANGED:-0}"

# Stage1 初始 checkpoint，可选。用于从已有模型继续训练。
STAGE1_INIT_CKPT="${STAGE1_INIT_CKPT:-}"

# 线上/旧模型评估指标目录。提供后才会执行 gate 对比。
BASELINE_METRICS_DIR="${BASELINE_METRICS_DIR:-}"

# gate 失败时是否软失败：
# - 0：gate 不通过则脚本失败。
# - 1：只生成报告，不中断脚本。
GATE_SOFT_FAIL="${GATE_SOFT_FAIL:-0}"

# 评估设备，可选。例如 cuda:0 / cpu。为空时由评估脚本自动选择。
DEVICE="${DEVICE:-}"

# Stage3 训练 epoch 数。
STAGE3_EPOCHS="${STAGE3_EPOCHS:-3}"

# Stage3 学习率。
STAGE3_LR="${STAGE3_LR:-0.0001}"

mkdir -p "${DATA_DIR}" "${OUT_DIR}/configs" "${OUT_DIR}/warmstarts"
echo "[INFO] project root: ${ROOT_DIR}"
echo "[INFO] iteration : ${ITER_ID}"
echo "[INFO] split mode: ${SPLIT_MODE}"
echo "[INFO] train plan: ${TRAIN_PLAN}"
echo "[INFO] data dir  : ${DATA_DIR}"
echo "[INFO] out dir   : ${OUT_DIR}"
if [[ -n "${BASE_CSV}" ]]; then echo "[INFO] base csv  : ${BASE_CSV}"; fi
if [[ -n "${INCREMENT_MANIFEST}" ]]; then echo "[INFO] increment manifest: ${INCREMENT_MANIFEST}"; fi

row_count() {
  local csv_path="$1"
  run_python - "$csv_path" <<'PY'
import sys
from pathlib import Path
import pandas as pd
path = Path(sys.argv[1])
if not path.exists():
    print(0)
else:
    try:
        print(len(pd.read_csv(path)))
    except Exception:
        print(0)
PY
}

same_file_content() {
  local a="$1"
  local b="$2"
  if [[ ! -f "${a}" || ! -f "${b}" ]]; then
    return 1
  fi
  cmp -s "${a}" "${b}"
}

find_script() {
  local name="$1"
  if [[ -f "scripts/${name}" ]]; then
    echo "scripts/${name}"
  elif [[ -f "src/scripts/${name}" ]]; then
    echo "src/scripts/${name}"
  else
    echo "[ERROR] Cannot find helper script: ${name} under scripts/ or src/scripts/" >&2
    exit 1
  fi
}

make_cfg() { run_python "$(find_script make_runtime_config.py)" "$@"; }
warmstart() { run_python "$(find_script prepare_warmstart.py)" --input "$1" --output "$2" --prefer "${WARMSTART_PREFER:-ema}"; }

if [[ "${USE_PREPARED_DATA}" == "1" ]]; then
  echo "[1/9] Use prepared data directory and skip merge/split"
  required_files=(
    "${DATA_DIR}/all_samples.csv"
    "${DATA_DIR}/train_stage1.csv"
    "${DATA_DIR}/train_stage2.csv"
    "${DATA_DIR}/train_stage3.csv"
    "${DATA_DIR}/val.csv"
  )
  for f in "${required_files[@]}"; do
    if [[ ! -f "${f}" ]]; then
      echo "[ERROR] USE_PREPARED_DATA=1 but required file is missing: ${f}" >&2
      exit 1
    fi
  done
  RUN_SCAN=0
  RUN_SPLIT=0
else
# Resolve the canonical image-level index for this iteration.
# 日常推荐优先级：
#   1) INPUT_CSV：直接使用已经合并好的图片级 CSV，适合复现实验/调试。
#   2) BASE_CSV + INCREMENT_MANIFEST：只扫描本轮新增 manifest，再与基础 CSV 合并，推荐日常迭代使用。
#   3) MANIFEST 全量扫描：仅当 RUN_FULL_SCAN=1 时启用，避免误触发慢速全量扫描。
if [[ -n "${INPUT_CSV}" ]]; then
  echo "[1/9] Use provided canonical image-level CSV: ${INPUT_CSV}"
  mkdir -p "${DATA_DIR}"
  cp "${INPUT_CSV}" "${DATA_DIR}/all_samples.csv"
  RUN_SCAN=0
elif [[ -n "${BASE_CSV}" ]]; then
  echo "[1/9] Resolve canonical CSV from base index"
  mkdir -p "${DATA_DIR}"

  if [[ -z "${INCREMENT_MANIFEST}" ]]; then
    echo "[INFO] INCREMENT_MANIFEST is empty; copy BASE_CSV directly and skip merge."
    cp "${BASE_CSV}" "${DATA_DIR}/all_samples.csv"
    run_python - "${DATA_DIR}/all_samples.csv" "${DATA_DIR}/all_samples_summary.yaml" <<'PY'
from pathlib import Path
import sys
import pandas as pd

csv_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
df = pd.read_csv(csv_path, engine='pyarrow')
lines = [
    f"source_csv: {csv_path}",
    f"num_rows: {len(df)}",
]
if 'label' in df.columns:
    label_counts = df['label'].value_counts().sort_index().to_dict()
    lines.append('by_label:')
    for k, v in label_counts.items():
        lines.append(f"  {k}: {int(v)}")
if 'split_hint' in df.columns:
    split_counts = df['split_hint'].fillna('seen').astype(str).value_counts().to_dict()
    lines.append('by_split_hint:')
    for k, v in split_counts.items():
        lines.append(f"  {k}: {int(v)}")
summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f"[DONE] copied base index: {csv_path} -> summary {summary_path}")
PY
  else
    scanned_increment_csv="${DATA_DIR}/increment_from_manifest.csv"
    echo "[INFO] scan only new increment manifest: ${INCREMENT_MANIFEST}"
    run_python src/tools/scan_manifest_to_csv.py \
      --manifest "${INCREMENT_MANIFEST}" \
      --output_csv "${scanned_increment_csv}" \
      --summary_yaml "${DATA_DIR}/increment_from_manifest_summary.yaml"

    merge_args=(
      --base_csv "${BASE_CSV}"
      --append_csvs "${scanned_increment_csv}"
      --output_csv "${DATA_DIR}/all_samples.csv"
      --dedup_key "${MERGE_DEDUP_KEY}"
      --keep "${MERGE_KEEP}"
      --added_iter "${MERGE_ADDED_ITER}"
      --summary_yaml "${DATA_DIR}/all_samples_summary.yaml"
    )
    if [[ "${MERGE_STRICT_PATHS}" == "1" ]]; then
      merge_args+=(--strict_paths)
    fi
    run_python "$(find_script merge_dataset_index.py)" "${merge_args[@]}"
  fi
  RUN_SCAN=0
fi

if [[ "${RUN_SCAN}" == "1" ]]; then
  echo "[1/9] Build canonical CSV from full manifest"
  run_python src/tools/scan_manifest_to_csv.py \
    --manifest "${MANIFEST}" \
    --output_csv "${DATA_DIR}/all_samples.csv" \
    --summary_yaml "${DATA_DIR}/all_samples_summary.yaml"
else
  echo "[1/9] Skip full scan"
fi

if [[ ! -f "${DATA_DIR}/all_samples.csv" ]]; then
  echo "[ERROR] ${DATA_DIR}/all_samples.csv not found."
  echo "[ERROR] Please provide one of:"
  echo "        1) BASE_CSV=/path/to/base_index.csv [INCREMENT_MANIFEST=/path/to/new_data.yaml]"
  echo "        2) INPUT_CSV=/path/to/already_merged_index.csv"
  echo "        3) RUN_FULL_SCAN=1 MANIFEST=/path/to/full_manifest.yaml"
  exit 1
fi
fi

if [[ "${RUN_SPLIT}" == "1" ]]; then
  if [[ "${SPLIT_MODE}" == "full_seen_random_val" ]]; then
    echo "[2/9] Build full-seen random validation splits"
    split_args=(
      --source_csv "${DATA_DIR}/all_samples.csv"
      --output_dir "${DATA_DIR}"
      --train_plan "${TRAIN_PLAN}"
      --val_real_total "${VAL_REAL_TOTAL}"
      --val_fake_total "${VAL_FAKE_TOTAL}"
      --seed "${SEED}"
      --val_real_group_col "${VAL_REAL_GROUP_COL}"
      --val_fake_group_col "${VAL_FAKE_GROUP_COL}"
      --flat_dir_bucket_threshold "${FLAT_DIR_BUCKET_THRESHOLD}"
      --hash_buckets "${HASH_BUCKETS}"
      --test_unseen_hints "${TEST_UNSEEN_HINTS}"
      --hard_hints "${HARD_HINTS}"
      --hard_flag_col "${HARD_FLAG_COL}"
      --hard_flag_values "${HARD_FLAG_VALUES}"
      --reviewed_pool_hints "${REVIEWED_POOL_HINTS}"
    )
    if [[ "${VAL_INCLUDE_HARD}" == "1" ]]; then split_args+=(--val_include_hard); fi
    if [[ -n "${HARD_CSV}" ]]; then split_args+=(--hard_csv "${HARD_CSV}"); fi
    if [[ -n "${REVIEWED_POOL_CSV}" ]]; then split_args+=(--reviewed_pool_csv "${REVIEWED_POOL_CSV}"); fi
    run_python src/tools/build_full_seen_random_val.py "${split_args[@]}"
  elif [[ "${SPLIT_MODE}" == "full_seen_heldout_val" ]]; then
    echo "[2/9] Build full-seen train + whole-source/generator held-out validation splits"
    split_args=(
      --source_csv "${DATA_DIR}/all_samples.csv"
      --train_csv "${DATA_DIR}/train_stage1.csv"
      --val_csv "${DATA_DIR}/val.csv"
      --test_unseen_csv "${DATA_DIR}/test_unseen.csv"
      --summary_yaml "${DATA_DIR}/split_summary.yaml"
      --val_real_total "${VAL_REAL_TOTAL}"
      --val_fake_total "${VAL_FAKE_TOTAL}"
      --seed "${SEED}"
      --flat_dir_bucket_threshold "${FLAT_DIR_BUCKET_THRESHOLD}"
      --hash_buckets "${HASH_BUCKETS}"
    )
    if [[ -n "${REAL_HOLDOUT_SOURCES}" ]]; then split_args+=(--real_holdout_sources "${REAL_HOLDOUT_SOURCES}"); fi
    if [[ -n "${FAKE_HOLDOUT_GENERATORS}" ]]; then split_args+=(--fake_holdout_generators "${FAKE_HOLDOUT_GENERATORS}"); fi
    if [[ -n "${FAKE_HOLDOUT_SOURCES}" ]]; then split_args+=(--fake_holdout_sources "${FAKE_HOLDOUT_SOURCES}"); fi
    if [[ "${AUTO_HOLDOUT_REAL_SOURCES}" == "1" ]]; then split_args+=(--auto_holdout_real_sources); fi
    if [[ "${AUTO_HOLDOUT_FAKE_GENERATORS}" == "1" ]]; then split_args+=(--auto_holdout_fake_generators); fi
    run_python src/tools/build_full_seen_with_heldout_val.py "${split_args[@]}"
    run_python - "${DATA_DIR}" <<'PY'
from pathlib import Path
import pandas as pd
import sys

data_dir = Path(sys.argv[1])
train = pd.read_csv(data_dir / 'train_stage1.csv')
train.to_csv(data_dir / 'train_stage2.csv', index=False)
train.to_csv(data_dir / 'train_stage3.csv', index=False)
train.iloc[0:0].to_csv(data_dir / 'reviewed_pool.csv', index=False)
parts = []
for name in ['val.csv', 'test_unseen.csv']:
    path = data_dir / name
    if path.exists():
        df = pd.read_csv(path)
        if len(df) > 0:
            parts.append(df)
if parts:
    pd.concat(parts, axis=0, ignore_index=True).drop_duplicates(subset=['path'], keep='first').to_csv(data_dir / 'test_all.csv', index=False)
else:
    train.iloc[0:0].to_csv(data_dir / 'test_all.csv', index=False)
print('[DONE] full heldout split compatibility files: train_stage2.csv, train_stage3.csv, reviewed_pool.csv, test_all.csv')
PY
  else
    echo "[2/9] Build sampled train/val/test splits"
    split_args=(--input_csv "${DATA_DIR}/all_samples.csv" --output_dir "${DATA_DIR}" --preset "${PRESET}" --seed "${SEED}" --group_col "${GROUP_COL}")
    if [[ -n "${HOLDOUT_GENERATORS}" ]]; then split_args+=(--holdout_generators "${HOLDOUT_GENERATORS}"); fi
    if [[ -n "${HOLDOUT_SOURCES}" ]]; then split_args+=(--holdout_sources "${HOLDOUT_SOURCES}"); fi
    run_python src/tools/make_small_splits.py "${split_args[@]}"
  fi
else
  echo "[2/9] Skip split"
fi

STAGE1_CFG="${OUT_DIR}/configs/stage1.yaml"
STAGE2_CFG="${OUT_DIR}/configs/stage2.yaml"
STAGE3_CFG="${OUT_DIR}/configs/stage3.yaml"
EVAL_REPLAY_CFG="${OUT_DIR}/configs/eval_replay.yaml"

make_cfg --base "${BASE_STAGE1_CONFIG}" --output "${STAGE1_CFG}" --set "seed=${SEED}" --set "output_dir=${OUT_DIR}/stage1" --set "data.train_csv=${DATA_DIR}/train_stage1.csv" --set "data.val_csv=${DATA_DIR}/val.csv"
make_cfg --base "${BASE_STAGE2_CONFIG}" --output "${STAGE2_CFG}" --set "seed=${SEED}" --set "output_dir=${OUT_DIR}/stage2" --set "data.train_csv=${DATA_DIR}/train_stage2.csv" --set "data.val_csv=${DATA_DIR}/val.csv" --set "model.pretrained_rgb=false" --set "model.pretrained_forensic=false"

if [[ "${RUN_STAGE1}" == "1" ]]; then
  echo "[3/9] Train stage1"
  stage1_args=(--config "${STAGE1_CFG}")
  if [[ -n "${STAGE1_INIT_CKPT}" ]]; then
    STAGE1_WARM="${OUT_DIR}/warmstarts/stage1_init.pt"
    warmstart "${STAGE1_INIT_CKPT}" "${STAGE1_WARM}"
    stage1_args+=(--resume "${STAGE1_WARM}")
  fi
  run_torchrun src/train.py "${stage1_args[@]}"
else
  echo "[3/9] Skip stage1"
fi

STAGE1_CKPT="${OUT_DIR}/stage1/best.pt"
if [[ "${RUN_STAGE2}" == "1" ]]; then
  echo "[4/9] Train stage2 from stage1 weights"
  STAGE2_WARM="${OUT_DIR}/warmstarts/stage1_to_stage2.pt"
  warmstart "${STAGE1_CKPT}" "${STAGE2_WARM}"
  run_torchrun src/train.py --config "${STAGE2_CFG}" --resume "${STAGE2_WARM}"
else
  echo "[4/9] Skip stage2"
fi

STAGE2_CKPT="${OUT_DIR}/stage2/best.pt"
FINAL_CKPT="${STAGE2_CKPT}"
make_cfg --base "${BASE_EVAL_CONFIG}" --output "${EVAL_REPLAY_CFG}" --set "output_dir=${OUT_DIR}/eval_replay" --set "data.test_csv=${DATA_DIR}/reviewed_pool.csv" --set "model.pretrained_rgb=false" --set "model.pretrained_forensic=false" --set "eval.tta=${REPLAY_TTA:-1}"

if [[ "${RUN_REPLAY}" == "1" ]]; then
  echo "[5/9] Mine hard examples / replay buffer"
  reviewed_rows="$(row_count "${DATA_DIR}/reviewed_pool.csv")"
  if [[ "${reviewed_rows}" == "0" ]]; then
    echo "[WARN] reviewed_pool.csv is empty; copy train_stage2.csv to train_stage3.csv"
    cp "${DATA_DIR}/train_stage2.csv" "${DATA_DIR}/train_stage3.csv"
  else
    run_python src/tools/auto_replay.py --config "${EVAL_REPLAY_CFG}" --ckpt "${STAGE2_CKPT}" --candidate_csv "${DATA_DIR}/reviewed_pool.csv" --calib_csv "${DATA_DIR}/val.csv" --precision "${REPLAY_PRECISION:-0.98}" --output_buffer_csv "${DATA_DIR}/replay_buffer.csv" --base_train_csv "${DATA_DIR}/train_stage2.csv" --merged_output_csv "${DATA_DIR}/train_stage3.csv" --topk_real "${REPLAY_TOPK_REAL:-40000}" --topk_fake "${REPLAY_TOPK_FAKE:-40000}" --topk_uncertain "${REPLAY_TOPK_UNCERTAIN:-20000}" --max_buffer "${REPLAY_MAX_BUFFER:-200000}"
  fi
else
  echo "[5/9] Skip replay"
  cp "${DATA_DIR}/train_stage2.csv" "${DATA_DIR}/train_stage3.csv"
fi

if [[ "${RUN_STAGE3}" == "1" ]]; then
  if same_file_content "${DATA_DIR}/train_stage2.csv" "${DATA_DIR}/train_stage3.csv" && [[ "${TRAIN_STAGE3_IF_UNCHANGED}" != "1" ]]; then
    echo "[6/9] Skip stage3 because train_stage3.csv is identical to train_stage2.csv"
  else
    echo "[6/9] Train stage3 from stage2 weights on replay-merged data"
    make_cfg --base "${BASE_STAGE2_CONFIG}" --output "${STAGE3_CFG}" --set "seed=${SEED}" --set "output_dir=${OUT_DIR}/stage3" --set "data.train_csv=${DATA_DIR}/train_stage3.csv" --set "data.val_csv=${DATA_DIR}/val.csv" --set "model.pretrained_rgb=false" --set "model.pretrained_forensic=false" --set "optim.lr=${STAGE3_LR}" --set "train.epochs=${STAGE3_EPOCHS}"
    STAGE3_WARM="${OUT_DIR}/warmstarts/stage2_to_stage3.pt"
    warmstart "${STAGE2_CKPT}" "${STAGE3_WARM}"
    run_torchrun src/train.py --config "${STAGE3_CFG}" --resume "${STAGE3_WARM}"
    FINAL_CKPT="${OUT_DIR}/stage3/best.pt"
  fi
else
  echo "[6/9] Skip stage3"
fi

EVAL_DIR="${OUT_DIR}/eval"
METRICS_DIR="${EVAL_DIR}/metrics"
if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "[7/9] Evaluate final checkpoint"
  eval_args=(--ckpt "${FINAL_CKPT}" --base_config "${BASE_EVAL_CONFIG}" --data_dir "${DATA_DIR}" --output_dir "${EVAL_DIR}" --skip_missing --skip_empty --splits val=val.csv test_unseen=test_unseen.csv test_all=test_all.csv)
  if [[ -n "${DEVICE}" ]]; then eval_args+=(--device "${DEVICE}"); fi
  run_python "$(find_script eval_suite.py)" "${eval_args[@]}"
else
  echo "[7/9] Skip eval"
fi

if [[ "${RUN_GATE}" == "1" ]]; then
  echo "[8/9] Check promotion gate"
  if [[ -n "${BASELINE_METRICS_DIR}" && -d "${BASELINE_METRICS_DIR}" ]]; then
    gate_args=(--candidate_dir "${METRICS_DIR}" --baseline_dir "${BASELINE_METRICS_DIR}" --gate "${GATE_CONFIG}" --out "${OUT_DIR}/gate_report.json")
    if [[ "${GATE_SOFT_FAIL}" == "1" ]]; then gate_args+=(--soft_fail); fi
    run_python "$(find_script check_gate.py)" "${gate_args[@]}"
  else
    echo "[WARN] BASELINE_METRICS_DIR is empty or missing; skip gate."
  fi
else
  echo "[8/9] Skip gate"
fi

if [[ "${RUN_PACKAGE}" == "1" ]]; then
  echo "[9/9] Package release"
  run_python "$(find_script package_release.py)" --iteration_id "${ITER_ID}" --ckpt "${FINAL_CKPT}" --output_dir "${OUT_DIR}/release" --configs_dir "${OUT_DIR}/configs" --metrics_dir "${METRICS_DIR}" --data_dir "${DATA_DIR}" --status candidate
else
  echo "[9/9] Skip package"
fi

echo "[DONE] iteration ${ITER_ID}"
echo "[DONE] final checkpoint: ${FINAL_CKPT}"
echo "[DONE] outputs: ${OUT_DIR}"
