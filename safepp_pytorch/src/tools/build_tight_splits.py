from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import yaml


def load_yaml(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(obj: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def normalize_split_hint(x: object) -> str:
    if x is None:
        return 'seen'
    text = str(x).strip().lower()
    return text or 'seen'


def ensure_text(df: pd.DataFrame, col: str, default: str = 'unknown') -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=object)
    return df[col].fillna(default).astype(str)


def choose_unit_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    return 'path'


def hash_offset(text: str) -> int:
    digest = hashlib.md5(text.encode('utf-8')).hexdigest()
    return int(digest[:8], 16)


def unit_table(df: pd.DataFrame, unit_col: str) -> pd.DataFrame:
    work = df.copy()
    work['__rows'] = 1
    agg = work.groupby(unit_col, as_index=False).agg(
        rows=('__rows', 'sum'),
        label=('label', 'first'),
        source=('source', 'first'),
        dataset=('dataset', 'first'),
        domain=('domain', 'first'),
        generator=('generator', 'first'),
        split_hint=('split_hint', 'first'),
    )
    return agg


def allocate_by_bucket(ut: pd.DataFrame, bucket_col: str, rules: Sequence[Dict], seed: int) -> List[str]:
    chosen: List[str] = []
    if len(ut) == 0:
        return chosen
    for bucket, bucket_df in ut.groupby(bucket_col):
        bucket_rows = int(bucket_df['rows'].sum())
        target = 0
        for rule in rules:
            if bucket_rows >= int(rule['min_rows']):
                ratio = float(rule['ratio'])
                target = int(round(bucket_rows * ratio))
                target = max(target, int(rule['min_rows_out']))
                target = min(target, int(rule['cap_rows_out']))
                break
        if target <= 0:
            continue
        shuffled = bucket_df.sample(frac=1.0, random_state=seed + hash_offset(str(bucket)))
        acc = 0
        bucket_units: List[str] = []
        for _, row in shuffled.iterrows():
            bucket_units.append(str(row.iloc[0]))
            acc += int(row['rows'])
            if acc >= target:
                break
        chosen.extend(bucket_units)
    return chosen


def sample_fixed_rows_from_units(ut: pd.DataFrame, total_rows: int, bucket_col: str, seed: int) -> List[str]:
    chosen: List[str] = []
    if len(ut) == 0 or total_rows <= 0:
        return chosen
    bucket_counts = ut.groupby(bucket_col)['rows'].sum().to_dict()
    total_available = int(sum(int(v) for v in bucket_counts.values()))
    total_rows = min(total_rows, total_available)

    alloc: Dict[str, int] = {}
    remaining = total_rows
    alive = {str(k): int(v) for k, v in bucket_counts.items() if int(v) > 0}
    while remaining > 0 and alive:
        base = max(1, remaining // len(alive))
        progressed = False
        for k in list(alive.keys()):
            room = int(alive[k]) - int(alloc.get(k, 0))
            take = min(base, room, remaining)
            if take > 0:
                alloc[k] = int(alloc.get(k, 0)) + int(take)
                remaining -= int(take)
                progressed = True
            if int(alloc.get(k, 0)) >= int(alive[k]):
                alive.pop(k, None)
        if not progressed:
            break
    if remaining > 0:
        for k, v in sorted(bucket_counts.items(), key=lambda kv: int(kv[1]), reverse=True):
            room = int(v) - int(alloc.get(str(k), 0))
            if room <= 0:
                continue
            take = min(room, remaining)
            alloc[str(k)] = int(alloc.get(str(k), 0)) + int(take)
            remaining -= int(take)
            if remaining <= 0:
                break

    for bucket, bucket_df in ut.groupby(bucket_col):
        bucket_target = int(alloc.get(str(bucket), 0))
        if bucket_target <= 0:
            continue
        shuffled = bucket_df.sample(frac=1.0, random_state=seed + hash_offset(str(bucket)))
        acc = 0
        for _, row in shuffled.iterrows():
            chosen.append(str(row.iloc[0]))
            acc += int(row['rows'])
            if acc >= bucket_target:
                break
    return chosen


def subset_by_units(df: pd.DataFrame, unit_col: str, units: Sequence[str]) -> pd.DataFrame:
    units_set = set(str(x) for x in units)
    if not units_set:
        return df.iloc[0:0].copy()
    return df[df[unit_col].astype(str).isin(units_set)].copy().reset_index(drop=True)


def summarize(df: pd.DataFrame, unit_col: str) -> Dict:
    fake = df[df['label'].astype(int) == 1].copy()
    real = df[df['label'].astype(int) == 0].copy()
    return {
        'rows': int(len(df)),
        'units': int(df[unit_col].astype(str).nunique()) if len(df) else 0,
        'by_label': {int(k): int(v) for k, v in df['label'].value_counts().sort_index().items()},
        'real_by_source': {str(k): int(v) for k, v in real['source'].value_counts().items()},
        'fake_by_source': {str(k): int(v) for k, v in fake['source'].value_counts().items()},
        'fake_by_generator': {str(k): int(v) for k, v in fake['generator'].value_counts().items()},
        'by_dataset': {str(k): int(v) for k, v in df['dataset'].value_counts().items()},
        'by_domain': {str(k): int(v) for k, v in df['domain'].value_counts().items()},
    }


def overlap(a: pd.DataFrame, b: pd.DataFrame, col: str) -> Dict:
    sa = set(a[col].astype(str).tolist()) if col in a.columns else set()
    sb = set(b[col].astype(str).tolist()) if col in b.columns else set()
    inter = sorted(list(sa & sb))
    return {
        'column': col,
        'a_unique': int(len(sa)),
        'b_unique': int(len(sb)),
        'overlap_unique': int(len(inter)),
        'examples': inter[:50],
    }


def assert_binary(df: pd.DataFrame, name: str) -> None:
    if len(df) == 0:
        raise RuntimeError(f'{name} is empty.')
    labels = set(int(x) for x in df['label'].astype(int).unique().tolist())
    if labels != {0, 1}:
        raise RuntimeError(f'{name} must contain both labels, got {labels}.')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Build tighter train/val splits from stage-2 semantic dedup output.')
    p.add_argument('--input_csv', required=True, type=str)
    p.add_argument('--out_dir', required=True, type=str)
    p.add_argument('--config_yaml', required=False, type=str, default='')
    p.add_argument('--seed', required=False, type=int, default=3407)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config_yaml) if args.config_yaml else {}
    split_cfg = cfg.get('split', {}) or {}

    df = pd.read_csv(args.input_csv)
    if len(df) == 0:
        raise RuntimeError('input_csv is empty.')
    df['split_hint'] = ensure_text(df, 'split_hint', 'seen').map(normalize_split_hint)
    df['source'] = ensure_text(df, 'source')
    df['dataset'] = ensure_text(df, 'dataset')
    df['domain'] = ensure_text(df, 'domain')
    df['generator'] = ensure_text(df, 'generator')

    unit_candidates = split_cfg.get('unit_col_candidates', ['tight_split_component_id', 'split_component_id', 'dedup_component_id', 'group_id', 'path'])
    unit_col = choose_unit_col(df, unit_candidates)
    df[unit_col] = ensure_text(df, unit_col)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_train_datasets = set(str(x) for x in split_cfg.get('exclude_datasets_from_train', []))

    test_role_values = set(str(x).lower() for x in split_cfg.get('test_role_values', ['unseen', 'test_unseen']))
    explicit_test_sources = set(str(x) for x in split_cfg.get('explicit_test_sources', []))
    explicit_test_generators = set(str(x) for x in split_cfg.get('explicit_test_generators', []))

    test_mask = df['split_hint'].isin(test_role_values)
    if explicit_test_sources:
        test_mask |= df['source'].isin(explicit_test_sources)
    if explicit_test_generators:
        test_mask |= df['generator'].isin(explicit_test_generators)

    test_df = df[test_mask].copy().reset_index(drop=True)
    seen_df = df[~test_mask].copy().reset_index(drop=True)
    if exclude_train_datasets:
        seen_df = seen_df[~seen_df['dataset'].isin(exclude_train_datasets)].copy().reset_index(drop=True)

    val_unseen_cfg = split_cfg.get('val_unseen', {}) or {}
    val_unseen_real_sources = set(str(x) for x in val_unseen_cfg.get('real_sources', []))
    val_unseen_fake_generators = set(str(x) for x in val_unseen_cfg.get('fake_generators', []))
    val_unseen_fake_sources = set(str(x) for x in val_unseen_cfg.get('fake_sources', []))

    real_seen = seen_df[seen_df['label'].astype(int) == 0].copy().reset_index(drop=True)
    fake_seen = seen_df[seen_df['label'].astype(int) == 1].copy().reset_index(drop=True)

    real_val_pool = real_seen[real_seen['source'].isin(val_unseen_real_sources)].copy().reset_index(drop=True)
    fake_val_mask = fake_seen['generator'].isin(val_unseen_fake_generators)
    if val_unseen_fake_sources:
        fake_val_mask |= fake_seen['source'].isin(val_unseen_fake_sources)
    fake_val_pool = fake_seen[fake_val_mask].copy().reset_index(drop=True)

    real_train_pool = real_seen[~real_seen['source'].isin(val_unseen_real_sources)].copy().reset_index(drop=True)
    fake_train_mask = ~fake_seen['generator'].isin(val_unseen_fake_generators)
    if val_unseen_fake_sources:
        fake_train_mask &= ~fake_seen['source'].isin(val_unseen_fake_sources)
    fake_train_pool = fake_seen[fake_train_mask].copy().reset_index(drop=True)

    unit_real_val_pool = unit_table(real_val_pool, unit_col)
    unit_fake_val_pool = unit_table(fake_val_pool, unit_col)

    val_unseen_real_units = sample_fixed_rows_from_units(
        unit_real_val_pool,
        total_rows=int(val_unseen_cfg.get('real_target_rows', 6000)),
        bucket_col='source',
        seed=args.seed + 11,
    )
    val_unseen_fake_units = sample_fixed_rows_from_units(
        unit_fake_val_pool,
        total_rows=int(val_unseen_cfg.get('fake_target_rows', 6000)),
        bucket_col='generator',
        seed=args.seed + 23,
    )

    val_unseen_df = pd.concat([
        subset_by_units(real_val_pool, unit_col, val_unseen_real_units),
        subset_by_units(fake_val_pool, unit_col, val_unseen_fake_units),
    ], axis=0, ignore_index=True)

    used_units = set(val_unseen_df[unit_col].astype(str).tolist())
    train_pool = pd.concat([real_train_pool, fake_train_pool], axis=0, ignore_index=True)
    if used_units:
        train_pool = train_pool[~train_pool[unit_col].astype(str).isin(used_units)].copy().reset_index(drop=True)

    val_seen_cfg = split_cfg.get('val_seen', {}) or {}
    default_rules = [
        {'min_rows': 200000, 'ratio': 0.01, 'min_rows_out': 1200, 'cap_rows_out': 3000},
        {'min_rows': 50000, 'ratio': 0.02, 'min_rows_out': 600, 'cap_rows_out': 1600},
        {'min_rows': 0, 'ratio': 0.05, 'min_rows_out': 200, 'cap_rows_out': 800},
    ]
    rule_list = val_seen_cfg.get('rules', default_rules)
    rule_list = sorted(rule_list, key=lambda x: int(x['min_rows']), reverse=True)

    train_pool_real = train_pool[train_pool['label'].astype(int) == 0].copy().reset_index(drop=True)
    train_pool_fake = train_pool[train_pool['label'].astype(int) == 1].copy().reset_index(drop=True)

    real_bucket_col = str(val_seen_cfg.get('real_bucket_col', 'source'))
    fake_bucket_col = str(val_seen_cfg.get('fake_bucket_col', 'source'))

    ut_real = unit_table(train_pool_real, unit_col)
    ut_fake = unit_table(train_pool_fake, unit_col)
    val_seen_real_units = allocate_by_bucket(ut_real, bucket_col=real_bucket_col, rules=rule_list, seed=args.seed + 31)
    val_seen_fake_units = allocate_by_bucket(ut_fake, bucket_col=fake_bucket_col, rules=rule_list, seed=args.seed + 47)

    val_seen_df = pd.concat([
        subset_by_units(train_pool_real, unit_col, val_seen_real_units),
        subset_by_units(train_pool_fake, unit_col, val_seen_fake_units),
    ], axis=0, ignore_index=True)

    used_units |= set(val_seen_df[unit_col].astype(str).tolist())
    train_df = train_pool[~train_pool[unit_col].astype(str).isin(used_units)].copy().reset_index(drop=True)

    # final hygiene
    train_df = train_df.drop_duplicates(subset=['path'], keep='first').sample(frac=1.0, random_state=args.seed + 71).reset_index(drop=True)
    val_seen_df = val_seen_df.drop_duplicates(subset=['path'], keep='first').sample(frac=1.0, random_state=args.seed + 73).reset_index(drop=True)
    val_unseen_df = val_unseen_df.drop_duplicates(subset=['path'], keep='first').sample(frac=1.0, random_state=args.seed + 79).reset_index(drop=True)
    test_df = test_df.drop_duplicates(subset=['path'], keep='first').sample(frac=1.0, random_state=args.seed + 83).reset_index(drop=True)

    assert_binary(train_df, 'train')
    assert_binary(val_seen_df, 'val_seen')
    assert_binary(val_unseen_df, 'val_unseen')
    if len(test_df):
        assert_binary(test_df, 'test')

    train_df.to_csv(out_dir / 'train.csv', index=False)
    val_seen_df.to_csv(out_dir / 'val_seen.csv', index=False)
    val_unseen_df.to_csv(out_dir / 'val_unseen.csv', index=False)
    test_df.to_csv(out_dir / 'test.csv', index=False)

    summary = {
        'inputs': {
            'input_csv': str(args.input_csv),
            'config_yaml': str(args.config_yaml or ''),
            'unit_col': unit_col,
        },
        'split_settings': split_cfg,
        'splits': {
            'train': summarize(train_df, unit_col),
            'val_seen': summarize(val_seen_df, unit_col),
            'val_unseen': summarize(val_unseen_df, unit_col),
            'test': summarize(test_df, unit_col),
        },
        'overlaps': {
            'train_vs_val_seen_unit': overlap(train_df, val_seen_df, unit_col),
            'train_vs_val_unseen_unit': overlap(train_df, val_unseen_df, unit_col),
            'train_vs_test_unit': overlap(train_df, test_df, unit_col),
            'val_seen_vs_val_unseen_unit': overlap(val_seen_df, val_unseen_df, unit_col),
            'train_vs_val_seen_path': overlap(train_df, val_seen_df, 'path'),
            'train_vs_val_unseen_path': overlap(train_df, val_unseen_df, 'path'),
            'train_vs_test_path': overlap(train_df, test_df, 'path'),
        },
    }
    save_yaml(summary, str(out_dir / 'split_summary.yaml'))
    print(f'[DONE] wrote {out_dir / "train.csv"}')
    print(f'[DONE] wrote {out_dir / "val_seen.csv"}')
    print(f'[DONE] wrote {out_dir / "val_unseen.csv"}')
    print(f'[DONE] wrote {out_dir / "test.csv"}')
    print(f'[DONE] wrote {out_dir / "split_summary.yaml"}')


if __name__ == '__main__':
    main()
