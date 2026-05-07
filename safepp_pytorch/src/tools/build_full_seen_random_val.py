#!/usr/bin/env python3
"""Build full-seen splits with random/group-balanced validation.

This splitter is for model-iteration experiments where validation is sampled
from the seen portion instead of holding out entire sources or generators.

Rows with split_hint in --test_unseen_hints are written to test_unseen.csv and
are never used for train/val. Rows with split_hint in --reviewed_pool_hints are
written to reviewed_pool.csv for replay mining and are not used for train/val.
The remaining seen rows are split into val and training rows.

Training plans:
  hard_in_stage1: stage1 = base + collected hard; stage2 = stage1
  hard_in_stage2: stage1 = base; stage2 = base + collected hard

Stage3 starts as a copy of stage2. run_iteration.sh can merge mined replay rows
into train_stage3.csv before training stage3.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError('PyYAML is required for build_full_seen_random_val.py') from exc


def parse_csv_list(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).split(',') if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build full-seen training CSVs plus a random/group-balanced seen validation CSV.'
    )
    parser.add_argument('--source_csv', required=True, help='Canonical image-level CSV. Required columns: path,label.')
    parser.add_argument('--output_dir', required=True, help='Directory where split CSVs and summary YAML are written.')
    parser.add_argument('--hard_csv', default='', help='Optional extra collected-hard CSV. Rows are marked as hard.')
    parser.add_argument('--reviewed_pool_csv', default='', help='Optional labeled replay-candidate CSV used by auto_replay.py.')
    parser.add_argument('--train_plan', default='hard_in_stage1', choices=['hard_in_stage1', 'hard_in_stage2'])
    parser.add_argument('--val_real_total', type=int, default=20000)
    parser.add_argument('--val_fake_total', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--val_include_hard', action='store_true', help='Allow collected-hard rows to be sampled into val.')
    parser.add_argument('--val_real_group_col', default='source', help='Group column for balancing real validation samples.')
    parser.add_argument('--val_fake_group_col', default='generator', help='Group column for balancing fake validation samples.')
    parser.add_argument('--default_group_level', type=int, default=1)
    parser.add_argument('--flat_dir_bucket_threshold', type=int, default=1)
    parser.add_argument('--hash_buckets', type=int, default=128)
    parser.add_argument('--test_unseen_hints', default='unseen,test_unseen')
    parser.add_argument('--hard_hints', default='hard,hard_negative,collected_hard,train_hard')
    parser.add_argument('--reviewed_pool_hints', default='reviewed,reviewed_pool,replay_candidate,candidate')
    parser.add_argument('--hard_flag_col', default='is_hard_negative')
    parser.add_argument('--hard_flag_values', default='1,true,yes,y')
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_yaml(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def read_csv_compat(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, engine='pyarrow')
    except Exception as exc:
        print(f'[WARN] pyarrow CSV read failed for {path}: {exc}. Falling back to pandas default engine.')
        return pd.read_csv(path)


def normalize_split_hint(x: object) -> str:
    if x is None:
        return 'seen'
    if isinstance(x, float) and math.isnan(x):
        return 'seen'
    text = str(x).strip().lower()
    return text or 'seen'


def resolve_group_token(path: Path, level: int) -> str:
    level = max(1, int(level))
    parents = list(path.parents)
    if not parents:
        return path.stem or 'unknown'
    idx = min(level - 1, len(parents) - 1)
    token = parents[idx].name.strip()
    if token:
        return token
    for parent in parents[idx + 1:]:
        if parent.name.strip():
            return parent.name.strip()
    return path.stem or 'unknown'


def hash_bucket(text: str, num_buckets: int) -> str:
    digest = hashlib.md5(text.encode('utf-8')).hexdigest()
    return f'bucket_{int(digest, 16) % max(1, int(num_buckets)):03d}'


def ensure_columns(df: pd.DataFrame, default_group_level: int, mark_hard: bool = False, mark_reviewed: bool = False) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    missing = {'path', 'label'} - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    out = df.copy()
    out['path'] = out['path'].astype(str)
    out['label'] = pd.to_numeric(out['label'], errors='raise').astype(int)

    if 'source' not in out.columns:
        out['source'] = out['path'].map(lambda p: Path(str(p)).parent.name or 'unknown')
    out['source'] = out['source'].fillna('unknown').astype(str)

    defaults = {
        'dataset': 'unknown',
        'domain': 'unknown',
        'split_hint': 'seen',
        'sample_weight': 1.0,
        'is_hard_negative': 0,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    if 'generator' not in out.columns:
        out['generator'] = out['label'].map(lambda x: 'real' if int(x) == 0 else 'unknown')

    out['dataset'] = out['dataset'].fillna('unknown').astype(str)
    out['domain'] = out['domain'].fillna('unknown').astype(str)
    out['generator'] = out['generator'].fillna(out['label'].map(lambda x: 'real' if int(x) == 0 else 'unknown')).astype(str)
    out['split_hint'] = out['split_hint'].fillna('seen').astype(str)
    out['sample_weight'] = pd.to_numeric(out['sample_weight'], errors='coerce').fillna(1.0)
    out['is_hard_negative'] = pd.to_numeric(out['is_hard_negative'], errors='coerce').fillna(0).astype(int)

    if mark_hard:
        out['is_hard_negative'] = 1
        out['split_hint'] = out['split_hint'].map(lambda x: 'hard' if normalize_split_hint(x) == 'seen' else x)
        out['sample_weight'] = out['sample_weight'].clip(lower=1.0)
    if mark_reviewed:
        out['split_hint'] = out['split_hint'].map(lambda x: 'reviewed_pool' if normalize_split_hint(x) == 'seen' else x)

    if 'group_level' not in out.columns:
        out['group_level'] = int(default_group_level)
    out['group_level'] = pd.to_numeric(out['group_level'], errors='coerce').fillna(int(default_group_level)).astype(int)

    if 'group_token' not in out.columns:
        out['group_token'] = [
            resolve_group_token(Path(path), level=level)
            for path, level in zip(out['path'].tolist(), out['group_level'].tolist())
        ]
    out['group_token'] = out['group_token'].fillna('unknown').astype(str)

    if 'group_id' not in out.columns:
        out['group_id'] = out['source'].astype(str) + ':' + out['group_token'].astype(str)
    out['group_id'] = out['group_id'].fillna(out['source'].astype(str) + ':unknown').astype(str)
    return out


def maybe_rebucket_flat_sources(df: pd.DataFrame, threshold: int, num_buckets: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    by_source = out.groupby('source')['group_token'].nunique().to_dict()
    rebucket_sources = {src for src, n in by_source.items() if int(n) <= int(threshold)}
    if rebucket_sources:
        mask = out['source'].isin(rebucket_sources)
        out.loc[mask, 'group_token'] = out.loc[mask, 'path'].astype(str).map(lambda p: hash_bucket(p, num_buckets))
        out.loc[mask, 'group_id'] = out.loc[mask, 'source'].astype(str) + ':' + out.loc[mask, 'group_token'].astype(str)
    return out


def stable_seed_offset(text: str) -> int:
    total = 0
    for idx, ch in enumerate(str(text)):
        total += (idx + 1) * ord(ch)
    return total % 100003


def allocate_evenly(counts: Dict[str, int], total: int) -> Dict[str, int]:
    counts = {str(k): int(v) for k, v in counts.items() if int(v) > 0}
    remaining = min(int(total), int(sum(counts.values())))
    alloc = {k: 0 for k in counts}
    if remaining <= 0 or not counts:
        return alloc
    alive = set(counts.keys())
    while remaining > 0 and alive:
        progressed = False
        base = max(1, remaining // len(alive))
        for key in list(alive):
            if remaining <= 0:
                break
            room = counts[key] - alloc[key]
            take = min(base, room, remaining)
            if take > 0:
                alloc[key] += take
                remaining -= take
                progressed = True
            if alloc[key] >= counts[key]:
                alive.remove(key)
        if not progressed:
            break
    if remaining > 0:
        for key in sorted(counts.keys(), key=lambda x: counts[x] - alloc[x], reverse=True):
            if remaining <= 0:
                break
            room = counts[key] - alloc[key]
            if room <= 0:
                continue
            take = min(room, remaining)
            alloc[key] += take
            remaining -= take
    return alloc


def ensure_text_column(df: pd.DataFrame, col: str, default: str = 'unknown') -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns:
        out[col] = default
    out[col] = out[col].fillna(default).astype(str)
    return out


def sample_by_group(df: pd.DataFrame, total: int, group_col: str, seed: int) -> pd.DataFrame:
    if len(df) == 0 or int(total) <= 0:
        return df.iloc[0:0].copy()
    work = ensure_text_column(df, group_col)
    counts = {str(k): int(v) for k, v in work[group_col].value_counts().items()}
    alloc = allocate_evenly(counts, int(total))
    parts: List[pd.DataFrame] = []
    for group_name, quota in alloc.items():
        group_df = work[work[group_col] == group_name]
        shuffled = group_df.sample(frac=1.0, random_state=seed + stable_seed_offset(group_name))
        parts.append(shuffled.head(quota))
    out = pd.concat(parts, axis=0, ignore_index=True) if parts else work.iloc[0:0].copy()
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def truthy_series(series: pd.Series, truthy_values: Set[str]) -> pd.Series:
    text = series.fillna('').astype(str).str.strip().str.lower()
    return text.isin(truthy_values)


def split_hint_mask(df: pd.DataFrame, hints: Sequence[str]) -> pd.Series:
    hint_set = {normalize_split_hint(x) for x in hints if str(x).strip()}
    return df['split_hint_norm'].isin(hint_set)


def build_hard_mask(df: pd.DataFrame, hard_hints: Sequence[str], hard_flag_col: str, hard_flag_values: Set[str]) -> pd.Series:
    mask = split_hint_mask(df, hard_hints)
    if hard_flag_col and hard_flag_col in df.columns:
        mask = mask | truthy_series(df[hard_flag_col], hard_flag_values)
    if 'hard_type' in df.columns:
        mask = mask | df['hard_type'].fillna('').astype(str).str.strip().ne('')
    return mask


def ordered_columns(df: pd.DataFrame) -> List[str]:
    first = [
        'path', 'label', 'source', 'dataset', 'domain', 'generator', 'split_hint',
        'sample_weight', 'is_hard_negative', 'group_level', 'group_token', 'group_id',
        'hard_type', 'score', 'priority', 'decision_thr', 'mined_at', 'mined_model'
    ]
    return [c for c in first if c in df.columns] + [c for c in df.columns if c not in first and not c.endswith('_norm')]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    cols = ordered_columns(df)
    df[cols].to_csv(path, index=False)


def summarize(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {'num_rows': 0, 'by_label': {}, 'by_source': {}, 'fake_by_generator': {}, 'hard_rows': 0}
    hard_rows = int(df['is_collected_hard'].sum()) if 'is_collected_hard' in df.columns else 0
    fake_df = df[df['label'].astype(int) == 1]
    return {
        'num_rows': int(len(df)),
        'num_unique_paths': int(df['path'].astype(str).nunique()),
        'by_label': {int(k): int(v) for k, v in df['label'].value_counts().sort_index().items()},
        'by_source': {str(k): int(v) for k, v in df['source'].value_counts().head(50).items()},
        'fake_by_generator': {str(k): int(v) for k, v in fake_df['generator'].value_counts().head(50).items()},
        'hard_rows': hard_rows,
    }


def overlap_count(a: pd.DataFrame, b: pd.DataFrame, col: str) -> int:
    if col not in a.columns or col not in b.columns:
        return 0
    return len(set(a[col].astype(str)) & set(b[col].astype(str)))


def read_optional_csv(path: str, default_group_level: int, mark_hard: bool = False, mark_reviewed: bool = False) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'CSV not found: {path}')
    df = read_csv_compat(str(p))
    return ensure_columns(df, default_group_level=default_group_level, mark_hard=mark_hard, mark_reviewed=mark_reviewed)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    base_df = read_optional_csv(args.source_csv, default_group_level=args.default_group_level)
    hard_df = read_optional_csv(args.hard_csv, default_group_level=args.default_group_level, mark_hard=True)
    reviewed_extra_df = read_optional_csv(args.reviewed_pool_csv, default_group_level=args.default_group_level, mark_reviewed=True)

    # Put explicit hard/reviewed rows first, so duplicate paths keep their more specific role.
    all_df = pd.concat([hard_df, reviewed_extra_df, base_df], axis=0, ignore_index=True)
    if all_df.empty:
        raise RuntimeError('No rows found after reading source/hard/reviewed CSVs.')
    all_df = all_df.drop_duplicates(subset=['path'], keep='first').reset_index(drop=True)
    all_df['split_hint_norm'] = all_df['split_hint'].map(normalize_split_hint)
    all_df = maybe_rebucket_flat_sources(all_df, threshold=args.flat_dir_bucket_threshold, num_buckets=args.hash_buckets)

    unseen_hints = parse_csv_list(args.test_unseen_hints)
    reviewed_hints = parse_csv_list(args.reviewed_pool_hints)
    hard_hints = parse_csv_list(args.hard_hints)
    hard_values = {x.lower() for x in parse_csv_list(args.hard_flag_values)}

    unseen_mask = split_hint_mask(all_df, unseen_hints)
    reviewed_mask = split_hint_mask(all_df, reviewed_hints) & (~unseen_mask)
    seen_mask = (~unseen_mask) & (~reviewed_mask)
    test_unseen_df = all_df[unseen_mask].copy().reset_index(drop=True)
    reviewed_pool_df = all_df[reviewed_mask].copy().reset_index(drop=True)
    seen_df = all_df[seen_mask].copy().reset_index(drop=True)

    if seen_df.empty:
        raise RuntimeError('No seen rows remain after removing unseen/reviewed_pool rows.')
    hard_seen_mask = build_hard_mask(seen_df, hard_hints=hard_hints, hard_flag_col=args.hard_flag_col, hard_flag_values=hard_values)
    seen_df['is_collected_hard'] = hard_seen_mask.astype(int)
    if 'is_collected_hard' not in reviewed_pool_df.columns:
        reviewed_pool_df['is_collected_hard'] = 0
    if 'is_collected_hard' not in test_unseen_df.columns:
        test_unseen_df['is_collected_hard'] = 0

    val_pool = seen_df if args.val_include_hard else seen_df[seen_df['is_collected_hard'] == 0].copy()
    real_val_pool = val_pool[val_pool['label'].astype(int) == 0].copy()
    fake_val_pool = val_pool[val_pool['label'].astype(int) == 1].copy()
    if real_val_pool.empty:
        raise RuntimeError('Validation real pool is empty. Check labels or --val_include_hard.')
    if fake_val_pool.empty:
        raise RuntimeError('Validation fake pool is empty. Check labels or --val_include_hard.')

    val_real_df = sample_by_group(real_val_pool, total=args.val_real_total, group_col=args.val_real_group_col, seed=args.seed + 11)
    val_fake_df = sample_by_group(fake_val_pool, total=args.val_fake_total, group_col=args.val_fake_group_col, seed=args.seed + 23)
    val_df = pd.concat([val_real_df, val_fake_df], axis=0, ignore_index=True)
    val_df = val_df.drop_duplicates(subset=['path'], keep='first').sample(frac=1.0, random_state=args.seed + 89).reset_index(drop=True)

    val_paths = set(val_df['path'].astype(str).tolist())
    train_remaining = seen_df[~seen_df['path'].astype(str).isin(val_paths)].copy().reset_index(drop=True)
    base_train_df = train_remaining[train_remaining['is_collected_hard'] == 0].copy().reset_index(drop=True)
    hard_train_df = train_remaining[train_remaining['is_collected_hard'] == 1].copy().reset_index(drop=True)

    if base_train_df.empty:
        raise RuntimeError('Base train split is empty after validation sampling.')
    if hard_train_df.empty:
        print('[WARN] No collected-hard rows were found for train. The two train plans will be equivalent.')

    if args.train_plan == 'hard_in_stage1':
        train_stage1_df = pd.concat([base_train_df, hard_train_df], axis=0, ignore_index=True)
        train_stage2_df = train_stage1_df.copy()
    else:
        train_stage1_df = base_train_df.copy()
        train_stage2_df = pd.concat([base_train_df, hard_train_df], axis=0, ignore_index=True)

    train_stage1_df = train_stage1_df.drop_duplicates(subset=['path'], keep='first').sample(frac=1.0, random_state=args.seed + 101).reset_index(drop=True)
    train_stage2_df = train_stage2_df.drop_duplicates(subset=['path'], keep='first').sample(frac=1.0, random_state=args.seed + 103).reset_index(drop=True)
    train_stage3_df = train_stage2_df.copy()

    test_all_df = pd.concat([val_df, test_unseen_df], axis=0, ignore_index=True).drop_duplicates(subset=['path'], keep='first')

    write_csv(train_stage1_df, out_dir / 'train_stage1.csv')
    write_csv(train_stage2_df, out_dir / 'train_stage2.csv')
    write_csv(train_stage3_df, out_dir / 'train_stage3.csv')
    write_csv(base_train_df, out_dir / 'train_base.csv')
    write_csv(hard_train_df, out_dir / 'train_hard.csv')
    write_csv(val_df, out_dir / 'val.csv')
    write_csv(test_unseen_df, out_dir / 'test_unseen.csv')
    write_csv(test_all_df, out_dir / 'test_all.csv')
    write_csv(reviewed_pool_df, out_dir / 'reviewed_pool.csv')

    summary = {
        'source_csv': args.source_csv,
        'hard_csv': args.hard_csv,
        'reviewed_pool_csv': args.reviewed_pool_csv,
        'train_plan': args.train_plan,
        'val_include_hard': bool(args.val_include_hard),
        'targets': {
            'val_real_total': int(args.val_real_total),
            'val_fake_total': int(args.val_fake_total),
            'val_real_group_col': args.val_real_group_col,
            'val_fake_group_col': args.val_fake_group_col,
        },
        'hints': {
            'test_unseen_hints': unseen_hints,
            'hard_hints': hard_hints,
            'reviewed_pool_hints': reviewed_hints,
            'hard_flag_col': args.hard_flag_col,
            'hard_flag_values': sorted(hard_values),
        },
        'splits': {
            'all_input': summarize(all_df),
            'seen_for_split': summarize(seen_df),
            'train_base': summarize(base_train_df),
            'train_hard': summarize(hard_train_df),
            'train_stage1': summarize(train_stage1_df),
            'train_stage2': summarize(train_stage2_df),
            'train_stage3_initial': summarize(train_stage3_df),
            'val': summarize(val_df),
            'test_unseen': summarize(test_unseen_df),
            'test_all': summarize(test_all_df),
            'reviewed_pool': summarize(reviewed_pool_df),
        },
        'overlap_checks': {
            'train_stage1_val_path_overlap': overlap_count(train_stage1_df, val_df, 'path'),
            'train_stage2_val_path_overlap': overlap_count(train_stage2_df, val_df, 'path'),
            'stage1_stage2_path_overlap': overlap_count(train_stage1_df, train_stage2_df, 'path'),
        },
    }
    save_yaml(summary, out_dir / 'split_summary.yaml')
    print(f'[DONE] wrote splits -> {out_dir}')
    print(json.dumps({k: v['num_rows'] for k, v in summary['splits'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
