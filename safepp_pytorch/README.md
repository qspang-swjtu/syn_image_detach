#  PyTorch minimal training skeleton

A minimal, executable PyTorch implementation for a dual-branch synthetic image detector inspired by SAFE:
- RGB branch: ConvNeXt-Tiny
- Forensic branch: ResNet18 on DWT-HH
- Crop-first preprocessing
- ColorJitter + RandomRotation + RandomMask
- Optional robustness finetune with GaussianBlur + JPEG
- DDP-ready training with AMP, EMA, cosine scheduler
- Source-balanced sampling
- Hard-example replay buffer builder

## CSV format

The minimum CSV format is:

```csv
path,label
/data/imgs/0001.jpg,0
/data/imgs/fake_0002.png,1
```

Recommended extended format for source balancing and replay:

```csv
path,label,source,sample_weight,is_hard_negative
/data/real/wildfake/0001.jpg,0,wildfake,1.0,0
/data/fake/dragon/0002.png,1,dragon,1.0,0
/data/real/replay/0003.jpg,0,review_queue,3.0,1
```

Notes:
- `label=0` means real, `label=1` means fake.
- `source` is used by the source-balanced sampler. If missing, the sampler falls back to the parent directory name of `path`.
- `sample_weight` is optional. It is multiplied into the sampler weight.
- `is_hard_negative=1` marks replay-buffer samples and can be upweighted again through `data.sampler.hard_negative_boost`.

## Train

```bash
torchrun --nproc_per_node=2 src/train.py --config configs/stage1.yaml
torchrun --nproc_per_node=2 src/train.py --config configs/stage2.yaml --resume outputs/stage1/best.pt
torchrun --nproc_per_node=2 src/train.py --config configs/stage3.yaml --resume outputs/stage2/best.pt
```

Each stage config now contains:

```yaml
data:
  sampler:
    name: source_balanced
    group_by: [source, label]
    alpha: 1.0
    replacement: true
    weight_col: sample_weight
    hard_negative_col: is_hard_negative
    hard_negative_boost: 1.0   # stage3 uses 3.0 by default
```

This balances sampling mass across `source x label` groups so large sources do not dominate the epoch.

## Eval

```bash
torchrun --nproc_per_node=1 src/eval.py --config configs/eval.yaml --ckpt outputs/stage3/best.pt
```

## Build replay buffer automatically

Mine hard examples from a labeled candidate pool and merge them into a replay CSV:

```bash
python src/tools/auto_replay.py \
  --config configs/eval.yaml \
  --ckpt outputs/stage2/best.pt \
  --candidate_csv /path/to/reviewed_pool.csv \
  --calib_csv /path/to/val.csv \
  --precision 0.98 \
  --output_buffer_csv /path/to/replay_buffer.csv \
  --base_train_csv /path/to/train_stage2.csv \
  --merged_output_csv /path/to/train_stage3.csv \
  --topk_real 40000 \
  --topk_fake 40000 \
  --topk_uncertain 20000 \
  --max_buffer 200000
```

What the script does:
- scores a labeled candidate pool with your checkpoint
- derives a deployment threshold from `--calib_csv` at the requested precision, or uses `--decision_thr`
- mines three buckets:
  - hard real images (`real_false_positive` / `real_near_miss`)
  - hard fake images (`fake_false_negative` / `fake_near_miss`)
  - threshold-band uncertain samples (`uncertain_band`)
- assigns `sample_weight` and `is_hard_negative=1`
- merges with an existing replay buffer if present
- caps the buffer with a source-balanced keep policy
- optionally writes a merged stage-3 training CSV

Recommended loop:
1. Train stage 1 and stage 2.
2. Run `auto_replay.py` on a reviewed pool of new traffic or external hard sets.
3. Point `configs/stage3.yaml -> data.train_csv` to the merged CSV.
4. Run stage 3 hard-example tuning.

## Practical notes

- `auto_replay.py` expects **labeled** candidate data. For real production traffic, first human-review or rule-review the mined candidates, then feed the reviewed CSV back into the replay script.
- The sampler summary is saved to `output_dir/sampler_summary.yaml` at train start.
- Stage 3 already sets `hard_negative_boost: 3.0`; if replay dominates too much, reduce it to `2.0`.


## Quick small-data validation workflow

When you want to verify the pipeline before spending time on multi-million-image training, use a compact benchmark first.

Recommended scales:
- `smoke`: 5k real + 5k fake for training, mainly to check the code path and losses.
- `mini`: 20k real + 20k fake for stage-1 training, plus an unseen-generator test split. This is the recommended first serious run.
- `pilot`: 60k real + 60k fake if `mini` looks healthy and you want a more stable estimate before the full run.

### 1) Build a canonical index from raw folders

Edit `manifests/small_sources_example.yaml`, then run:

```bash
python src/tools/scan_manifest_to_csv.py   --manifest manifests/small_sources_example.yaml   --output_csv data/index/all_samples.csv
```

### 2) Create compact splits

```bash
python src/tools/make_small_splits.py   --input_csv data/index/all_samples.csv   --output_dir data/small   --preset mini
```

This writes:
- `train_stage1.csv`
- `train_stage2.csv`
- `train_stage3.csv`
- `val.csv`
- `test_seen.csv`
- `test_unseen.csv`
- `test_all.csv`
- `reviewed_pool.csv`

If your canonical CSV does not contain `split_hint`, you can force unseen holdouts by source or generator:

```bash
python src/tools/make_small_splits.py   --input_csv data/index/all_samples.csv   --output_dir data/small   --preset mini   --holdout_generators flux,pixart
```

### 3) Train the compact configs

```bash
torchrun --nproc_per_node=2 src/train.py --config configs/stage1_small.yaml
torchrun --nproc_per_node=2 src/train.py --config configs/stage2_small.yaml --resume outputs/stage1_small/best.pt
torchrun --nproc_per_node=2 src/train.py --config configs/stage3_small.yaml --resume outputs/stage2_small/best.pt
```

### 4) Evaluate unseen-generator generalization

```bash
torchrun --nproc_per_node=1 src/eval.py --config configs/eval_small.yaml --ckpt outputs/stage3_small/best.pt
```
