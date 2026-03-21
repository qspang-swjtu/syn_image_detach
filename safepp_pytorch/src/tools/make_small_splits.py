import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.common import ensure_dir, save_yaml  # noqa: E402


PRESETS = {
    'smoke': {
        'train_total_per_label': 5000,
        'val_total_per_label': 1000,
        'test_seen_total_per_label': 1000,
        'test_unseen_total_per_label': 2000,
        'stage2_extra_total_per_label': 0,
    },
    'mini': {
        'train_total_per_label': 20000,
        'val_total_per_label': 4000,
        'test_seen_total_per_label': 4000,
        'test_unseen_total_per_label': 4000,
        'stage2_extra_total_per_label': 4000,
    },
    'pilot': {
        'train_total_per_label': 60000,
        'val_total_per_label': 8000,
        'test_seen_total_per_label': 8000,
        'test_unseen_total_per_label': 8000,
        'stage2_extra_total_per_label': 12000,
    },
}

FIXED_VAL_HINTS = {'val', 'dev'}
FIXED_TEST_SEEN_HINTS = {'test_seen', 'seen_test'}
UNSEEN_HINTS = {'unseen', 'test_unseen'}
ROBUST_HINTS = {'robust', 'stage2', 'robustness'}


def parse_args():
    parser = argparse.ArgumentParser(description='Create compact train/val/test CSVs for quick SAFE++ validation.')
    parser.add_argument('--input_csv', type=str, required=True, help='Canonical CSV produced by scan_manifest_to_csv.py.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the split CSVs.')
    parser.add_argument('--preset', type=str, default='mini', choices=sorted(PRESETS.keys()))
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--group_col', type=str, default='source', help='Column used to balance source coverage within each label.')
    parser.add_argument('--split_hint_col', type=str, default='split_hint')
    parser.add_argument('--holdout_sources', type=str, default='', help='Comma-separated source names to reserve for test_unseen.')
    parser.add_argument('--holdout_generators', type=str, default='', help='Comma-separated fake generators to reserve for test_unseen.')
    parser.add_argument('--train_total_per_label', type=int, default=None)
    parser.add_argument('--val_total_per_label', type=int, default=None)
    parser.add_argument('--test_seen_total_per_label', type=int, default=None)
    parser.add_argument('--test_unseen_total_per_label', type=int, default=None)
    parser.add_argument('--stage2_extra_total_per_label', type=int, default=None)
    return parser.parse_args()


def parse_set(text: str) -> Set[str]:
    if not text:
        return set()
    return {x.strip() for x in text.split(',') if x.strip()}


def normalize_split_hint(x: object) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 'seen'
    return str(x).strip().lower() or 'seen'


def allocate_evenly(counts: Dict[str, int], total: int) -> Dict[str, int]:
    counts = {str(k): int(v) for k, v in counts.items() if int(v) > 0}
    total = min(int(total), int(sum(counts.values())))
    alloc = {k: 0 for k in counts}
    if total <= 0 or not counts:
        return alloc

    alive = set(counts.keys())
    while total > 0 and alive:
        per = max(1, total // len(alive))
        progressed = False
        for k in list(alive):
            room = counts[k] - alloc[k]
            take = min(room, per)
            if take > 0:
                alloc[k] += take
                total -= take
                progressed = True
            if alloc[k] >= counts[k]:
                alive.remove(k)
        if not progressed:
            break

    if total > 0:
        for k in sorted(counts.keys(), key=lambda x: counts[x] - alloc[x], reverse=True):
            if total <= 0:
                break
            room = counts[k] - alloc[k]
            if room <= 0:
                continue
            take = min(room, total)
            alloc[k] += take
            total -= take
    return alloc


def sample_balanced(df: pd.DataFrame, total_per_label: int, group_col: str, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    kept_parts: List[pd.DataFrame] = []
    remain_parts: List[pd.DataFrame] = []

    for label in [0, 1]:
        sub = df[df['label'].astype(int) == label].copy()
        if len(sub) == 0:
            continue
        if group_col not in sub.columns:
            sub[group_col] = 'unknown'
        sub[group_col] = sub[group_col].fillna('unknown').astype(str)
        counts = {str(k): int(v) for k, v in sub[group_col].value_counts().items()}
        alloc = allocate_evenly(counts, total_per_label)

        picked = []
        dropped = []
        for g, quota in alloc.items():
            gdf = sub[sub[group_col] == g].sample(frac=1.0, random_state=seed + label + abs(hash(g)) % 9973)
            picked.append(gdf.head(quota))
            dropped.append(gdf.iloc[quota:])
        if picked:
            kept_parts.append(pd.concat(picked, axis=0, ignore_index=False))
        if dropped:
            remain_parts.append(pd.concat(dropped, axis=0, ignore_index=False))

        unseen_groups = set(sub[group_col].astype(str).unique()) - set(alloc.keys())
        for g in unseen_groups:
            remain_parts.append(sub[sub[group_col] == g])

    kept_df = pd.concat(kept_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True) if kept_parts else df.iloc[0:0].copy()
    remain_df = pd.concat(remain_parts, axis=0).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True) if remain_parts else df.iloc[0:0].copy()
    kept_paths = set(kept_df['path'].astype(str).tolist())
    remain_df = df[~df['path'].astype(str).isin(kept_paths)].copy().reset_index(drop=True)
    return kept_df, remain_df


def split_by_role(df: pd.DataFrame, split_hint_col: str, holdout_sources: Set[str], holdout_generators: Set[str]) -> Dict[str, pd.DataFrame]:
    out = df.copy()
    if split_hint_col not in out.columns:
        out[split_hint_col] = 'seen'
    out[split_hint_col] = out[split_hint_col].map(normalize_split_hint)

    source_series = out['source'].fillna('unknown').astype(str) if 'source' in out.columns else pd.Series(['unknown'] * len(out), index=out.index)
    generator_series = out['generator'].fillna('unknown').astype(str) if 'generator' in out.columns else pd.Series(['unknown'] * len(out), index=out.index)

    is_unseen = out[split_hint_col].isin(UNSEEN_HINTS) | source_series.isin(holdout_sources) | generator_series.isin(holdout_generators)
    is_robust = out[split_hint_col].isin(ROBUST_HINTS)
    is_fixed_val = out[split_hint_col].isin(FIXED_VAL_HINTS)
    is_fixed_test_seen = out[split_hint_col].isin(FIXED_TEST_SEEN_HINTS)

    seen_df = out[~is_unseen & ~is_robust & ~is_fixed_val & ~is_fixed_test_seen].copy().reset_index(drop=True)
    robust_df = out[is_robust].copy().reset_index(drop=True)
    unseen_df = out[is_unseen].copy().reset_index(drop=True)
    fixed_val_df = out[is_fixed_val].copy().reset_index(drop=True)
    fixed_test_seen_df = out[is_fixed_test_seen].copy().reset_index(drop=True)

    return {
        'seen': seen_df,
        'robust': robust_df,
        'unseen': unseen_df,
        'fixed_val': fixed_val_df,
        'fixed_test_seen': fixed_test_seen_df,
    }


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        'source': 'unknown',
        'dataset': 'unknown',
        'domain': 'unknown',
        'generator': 'unknown',
        'sample_weight': 1.0,
        'is_hard_negative': 0,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].fillna(default)
    out['label'] = out['label'].astype(int)
    out['sample_weight'] = pd.to_numeric(out['sample_weight'], errors='coerce').fillna(1.0)
    out['is_hard_negative'] = pd.to_numeric(out['is_hard_negative'], errors='coerce').fillna(0).astype(int)
    return out


def summarize_split(df: pd.DataFrame) -> Dict:
    if len(df) == 0:
        return {'num_rows': 0, 'by_label': {}, 'by_source': {}, 'by_generator': {}}
    return {
        'num_rows': int(len(df)),
        'by_label': {int(k): int(v) for k, v in df['label'].value_counts().sort_index().items()},
        'by_source': {str(k): int(v) for k, v in df['source'].value_counts().head(20).items()},
        'by_generator': {str(k): int(v) for k, v in df['generator'].value_counts().head(20).items()},
    }


def write_csv(df: pd.DataFrame, path: Path):
    keep_cols = [
        'path', 'label', 'source', 'dataset', 'domain', 'generator',
        'sample_weight', 'is_hard_negative', 'split_hint'
    ]
    cols = [c for c in keep_cols if c in df.columns] + [c for c in df.columns if c not in keep_cols]
    df[cols].to_csv(path, index=False)


def main():
    args = parse_args()
    preset = PRESETS[args.preset].copy()
    for k in list(preset.keys()):
        v = getattr(args, k)
        if v is not None:
            preset[k] = int(v)

    df = pd.read_csv(args.input_csv)
    required = {'path', 'label'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    df = ensure_columns(df)

    roles = split_by_role(
        df,
        split_hint_col=args.split_hint_col,
        holdout_sources=parse_set(args.holdout_sources),
        holdout_generators=parse_set(args.holdout_generators),
    )

    fixed_val_df = roles['fixed_val']
    fixed_test_seen_df = roles['fixed_test_seen']

    seen_remaining = roles['seen'].copy()
    if len(fixed_val_df) > 0:
        val_df = fixed_val_df.copy()
        seen_remaining = seen_remaining[~seen_remaining['path'].isin(val_df['path'])].copy()
    else:
        val_df, seen_remaining = sample_balanced(
            seen_remaining,
            total_per_label=preset['val_total_per_label'],
            group_col=args.group_col,
            seed=args.seed + 11,
        )

    if len(fixed_test_seen_df) > 0:
        test_seen_df = fixed_test_seen_df.copy()
        seen_remaining = seen_remaining[~seen_remaining['path'].isin(test_seen_df['path'])].copy()
    else:
        test_seen_df, seen_remaining = sample_balanced(
            seen_remaining,
            total_per_label=preset['test_seen_total_per_label'],
            group_col=args.group_col,
            seed=args.seed + 23,
        )

    train_stage1_df, leftovers_seen = sample_balanced(
        seen_remaining,
        total_per_label=preset['train_total_per_label'],
        group_col=args.group_col,
        seed=args.seed + 37,
    )

    test_unseen_df, leftovers_unseen = sample_balanced(
        roles['unseen'],
        total_per_label=preset['test_unseen_total_per_label'],
        group_col=args.group_col,
        seed=args.seed + 51,
    )

    # If the unseen pool only contains fake generators (a common setup),
    # supplement the missing label from leftover seen data so the eval split remains binary.
    test_unseen_labels = set(test_unseen_df['label'].astype(int).unique().tolist()) if len(test_unseen_df) else set()
    for missing_label in sorted({0, 1} - test_unseen_labels):
        supplement_pool = leftovers_seen[leftovers_seen['label'].astype(int) == missing_label].copy()
        if len(supplement_pool) == 0:
            continue
        supplement_df, leftovers_seen = sample_balanced(
            supplement_pool,
            total_per_label=preset['test_unseen_total_per_label'],
            group_col=args.group_col,
            seed=args.seed + 59 + missing_label,
        )
        supplement_df['split_hint'] = 'supplement_unseen'
        test_unseen_df = pd.concat([test_unseen_df, supplement_df], axis=0, ignore_index=True)
        leftovers_seen = leftovers_seen[~leftovers_seen['path'].isin(supplement_df['path'])].copy()

    stage2_extra_df = roles['robust'].copy()
    if len(stage2_extra_df) > 0 and preset['stage2_extra_total_per_label'] > 0:
        stage2_extra_df, _ = sample_balanced(
            stage2_extra_df,
            total_per_label=preset['stage2_extra_total_per_label'],
            group_col=args.group_col,
            seed=args.seed + 71,
        )
        stage2_extra_df['sample_weight'] = np.maximum(stage2_extra_df['sample_weight'].astype(float).to_numpy(), 1.0)
    else:
        stage2_extra_df = stage2_extra_df.iloc[0:0].copy()

    train_stage2_df = pd.concat([train_stage1_df, stage2_extra_df], axis=0, ignore_index=True)
    train_stage2_df = train_stage2_df.drop_duplicates(subset=['path'], keep='first').reset_index(drop=True)
    train_stage3_df = train_stage2_df.copy()
    reviewed_pool_df = pd.concat([leftovers_seen, leftovers_unseen], axis=0, ignore_index=True)
    reviewed_pool_df = reviewed_pool_df.drop_duplicates(subset=['path'], keep='first').reset_index(drop=True)

    out_dir = Path(args.output_dir)
    ensure_dir(str(out_dir))
    write_csv(train_stage1_df, out_dir / 'train_stage1.csv')
    write_csv(train_stage2_df, out_dir / 'train_stage2.csv')
    write_csv(train_stage3_df, out_dir / 'train_stage3.csv')
    write_csv(val_df, out_dir / 'val.csv')
    write_csv(test_seen_df, out_dir / 'test_seen.csv')
    write_csv(test_unseen_df, out_dir / 'test_unseen.csv')
    write_csv(pd.concat([test_seen_df, test_unseen_df], axis=0, ignore_index=True).drop_duplicates(subset=['path'], keep='first'), out_dir / 'test_all.csv')
    write_csv(reviewed_pool_df, out_dir / 'reviewed_pool.csv')

    summary = {
        'preset': args.preset,
        'config': preset,
        'input_csv': args.input_csv,
        'group_col': args.group_col,
        'holdout_sources': sorted(parse_set(args.holdout_sources)),
        'holdout_generators': sorted(parse_set(args.holdout_generators)),
        'splits': {
            'train_stage1': summarize_split(train_stage1_df),
            'train_stage2': summarize_split(train_stage2_df),
            'train_stage3': summarize_split(train_stage3_df),
            'val': summarize_split(val_df),
            'test_seen': summarize_split(test_seen_df),
            'test_unseen': summarize_split(test_unseen_df),
            'reviewed_pool': summarize_split(reviewed_pool_df),
        },
    }
    save_yaml(summary, str(out_dir / 'split_summary.yaml'))
    print(f'[DONE] wrote split CSVs to {out_dir}')
    for name, info in summary['splits'].items():
        print(f'  - {name}: {info["num_rows"]} rows, labels={info["by_label"]}')


if __name__ == '__main__':
    main()
