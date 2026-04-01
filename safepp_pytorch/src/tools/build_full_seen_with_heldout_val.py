import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from utils.common import ensure_dir, load_yaml, save_yaml  # type: ignore
except Exception:
    import os
    import yaml

    def ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def load_yaml(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def save_yaml(obj, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


META_COL_CHOICES = ['source', 'dataset', 'domain', 'generator', 'group_id']


def parse_csv_list(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(',') if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build full-seen train CSV, plus a truly held-out source/generator validation CSV, and keep split_hint=unseen as test_unseen.'
    )
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--test_unseen_csv', type=str, required=True)
    parser.add_argument('--summary_yaml', type=str, required=True)

    parser.add_argument('--val_real_total', type=int, default=20000)
    parser.add_argument('--val_fake_total', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--strict', action='store_true')

    parser.add_argument('--default_group_level', type=int, default=1)
    parser.add_argument('--flat_dir_bucket_threshold', type=int, default=1,
                        help='If a source has <= this many unique parent-dir groups, fallback to hash buckets for that source.')
    parser.add_argument('--hash_buckets', type=int, default=128)

    parser.add_argument('--real_holdout_sources', type=str, default='',
                        help='Comma-separated real source names to hold out entirely for val. Example: real_ffhq,real_laion5b')
    parser.add_argument('--fake_holdout_generators', type=str, default='',
                        help='Comma-separated fake generator names to hold out entirely for val. Example: ADM,VQDM,MAE,SD_3')
    parser.add_argument('--fake_holdout_sources', type=str, default='',
                        help='Optional comma-separated fake source names to hold out entirely for val.')

    parser.add_argument('--auto_holdout_real_sources', action='store_true',
                        help='If set and --real_holdout_sources is empty, greedily choose real sources (smallest first) until covering val_real_total.')
    parser.add_argument('--auto_holdout_fake_generators', action='store_true',
                        help='If set and --fake_holdout_generators is empty, greedily choose fake generators (smallest first) until covering val_fake_total.')
    return parser.parse_args()


def iter_images(root: Path, recursive: bool, exts: Iterable[str]) -> List[Path]:
    valid_exts = {'.' + e.lower().lstrip('.') for e in exts}
    if recursive:
        files = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in valid_exts]
    else:
        files = [p for p in root.glob('*') if p.is_file() and p.suffix.lower() in valid_exts]
    return sorted(files)


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
    return f'bucket_{int(digest, 16) % max(1, num_buckets):03d}'


def make_row(path: Path, spec: Dict, default_group_level: int) -> Dict:
    label = int(spec['label'])
    source = str(spec.get('name', path.parent.name or 'unknown'))
    group_level = int(spec.get('group_level', default_group_level))
    group_token = resolve_group_token(path, level=group_level)
    return {
        'path': str(path.resolve()),
        'label': label,
        'source': source,
        'dataset': str(spec.get('dataset', 'unknown')),
        'domain': str(spec.get('domain', 'unknown')),
        'generator': str(spec.get('generator', 'real' if label == 0 else 'unknown')),
        'split_hint': str(spec.get('split_hint', 'seen')),
        'sample_weight': float(spec.get('sample_weight', 1.0)),
        'is_hard_negative': int(spec.get('is_hard_negative', 0)),
        'group_level': group_level,
        'group_token': group_token,
        'group_id': f'{source}:{group_token}',
    }


def scan_manifest(manifest: Dict, strict: bool, default_group_level: int) -> pd.DataFrame:
    recursive_default = bool(manifest.get('recursive', True))
    exts_default = manifest.get('exts', ['jpg', 'jpeg', 'png', 'webp', 'bmp'])
    sources = manifest.get('sources', [])
    if not sources:
        raise ValueError('Manifest must contain a non-empty `sources` list.`')

    rows = []
    missing = []
    empty = []
    for spec in sources:
        for key in ['path', 'label']:
            if key not in spec:
                raise ValueError(f'Missing `{key}` in source entry: {spec}')
        root = Path(spec['path'])
        recursive = bool(spec.get('recursive', recursive_default))
        exts = spec.get('exts', exts_default)
        if not root.exists():
            missing.append(str(root))
            if strict:
                continue
            print(f'[WARN] missing path: {root}')
            continue
        files = iter_images(root, recursive=recursive, exts=exts)
        if not files:
            empty.append(str(root))
            if strict:
                continue
            print(f'[WARN] no images found: {root}')
            continue
        rows.extend(make_row(path, spec, default_group_level=default_group_level) for path in files)
        print(f'[OK] {spec.get("name", root.name)} -> {len(files)} files')

    if strict and (missing or empty):
        raise RuntimeError(f'Strict mode failed. Missing={missing}, empty={empty}')
    if not rows:
        raise RuntimeError('No image rows were collected from the manifest.')

    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=['path'], keep='first').reset_index(drop=True)


def normalize_split_hint(x: object) -> str:
    if x is None:
        return 'seen'
    text = str(x).strip().lower()
    return text or 'seen'


def maybe_rebucket_flat_sources(df: pd.DataFrame, threshold: int, num_buckets: int) -> pd.DataFrame:
    out = df.copy()
    if len(out) == 0:
        return out
    by_source = out.groupby('source')['group_token'].nunique().to_dict()
    rebucket_sources = {src for src, n in by_source.items() if int(n) <= int(threshold)}
    if rebucket_sources:
        mask = out['source'].isin(rebucket_sources)
        out.loc[mask, 'group_token'] = out.loc[mask, 'path'].astype(str).map(lambda p: hash_bucket(p, num_buckets))
        out.loc[mask, 'group_id'] = out.loc[mask, 'source'].astype(str) + ':' + out.loc[mask, 'group_token'].astype(str)
    return out


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


def stable_seed_offset(text: str) -> int:
    total = 0
    for idx, ch in enumerate(text):
        total += (idx + 1) * ord(ch)
    return total % 100003


def ensure_text_column(df: pd.DataFrame, col: str, default: str = 'unknown') -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns:
        out[col] = default
    out[col] = out[col].fillna(default).astype(str)
    return out


def sample_by_group(df: pd.DataFrame, total: int, group_col: str, seed: int) -> pd.DataFrame:
    if len(df) == 0 or total <= 0:
        return df.iloc[0:0].copy()
    work = ensure_text_column(df, group_col)
    counts = {str(k): int(v) for k, v in work[group_col].value_counts().items()}
    alloc = allocate_evenly(counts, total)

    parts: List[pd.DataFrame] = []
    for group_name, quota in alloc.items():
        group_df = work[work[group_col] == group_name]
        shuffled = group_df.sample(frac=1.0, random_state=seed + stable_seed_offset(group_name))
        parts.append(shuffled.head(quota))
    out = pd.concat(parts, axis=0, ignore_index=True) if parts else work.iloc[0:0].copy()
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def greedy_cover(values: Dict[str, int], target: int) -> List[str]:
    items = [(str(k), int(v)) for k, v in values.items() if int(v) > 0]
    items.sort(key=lambda x: (x[1], x[0]))
    chosen: List[str] = []
    covered = 0
    for name, count in items:
        if covered >= target:
            break
        chosen.append(name)
        covered += count
    return chosen


def summarize_split(df: pd.DataFrame) -> Dict:
    fake_df = df[df['label'].astype(int) == 1].copy()
    real_df = df[df['label'].astype(int) == 0].copy()
    return {
        'num_rows': int(len(df)),
        'num_unique_paths': int(df['path'].astype(str).nunique()) if 'path' in df.columns else 0,
        'num_unique_group_id': int(df['group_id'].astype(str).nunique()) if 'group_id' in df.columns else 0,
        'by_label': {int(k): int(v) for k, v in df['label'].value_counts().sort_index().items()},
        'real_by_source': {str(k): int(v) for k, v in real_df['source'].value_counts().items()},
        'fake_by_source': {str(k): int(v) for k, v in fake_df['source'].value_counts().items()},
        'fake_by_generator': {str(k): int(v) for k, v in fake_df['generator'].value_counts().items()},
        'real_by_dataset': {str(k): int(v) for k, v in real_df['dataset'].value_counts().items()},
        'fake_by_dataset': {str(k): int(v) for k, v in fake_df['dataset'].value_counts().items()},
    }


def overlap_stats(train_df: pd.DataFrame, val_df: pd.DataFrame, col: str) -> Dict:
    train_vals = set(ensure_text_column(train_df, col)[col].tolist())
    val_vals = set(ensure_text_column(val_df, col)[col].tolist())
    overlap = train_vals & val_vals
    return {
        'column': col,
        'train_unique': int(len(train_vals)),
        'val_unique': int(len(val_vals)),
        'overlap_unique': int(len(overlap)),
        'overlap_values': sorted(list(overlap))[:50],
    }


def main():
    args = parse_args()
    manifest = load_yaml(args.manifest)
    all_df = scan_manifest(manifest, strict=args.strict, default_group_level=args.default_group_level)
    if 'split_hint' not in all_df.columns:
        all_df['split_hint'] = 'seen'
    all_df['split_hint'] = all_df['split_hint'].map(normalize_split_hint)
    all_df = maybe_rebucket_flat_sources(all_df, threshold=args.flat_dir_bucket_threshold, num_buckets=args.hash_buckets)

    test_unseen_df = all_df[all_df['split_hint'] == 'unseen'].copy().reset_index(drop=True)
    seen_df = all_df[all_df['split_hint'] != 'unseen'].copy().reset_index(drop=True)

    real_seen = seen_df[seen_df['label'].astype(int) == 0].copy().reset_index(drop=True)
    fake_seen = seen_df[seen_df['label'].astype(int) == 1].copy().reset_index(drop=True)
    if len(real_seen) == 0:
        raise RuntimeError('No real samples were found in the seen portion of the manifest.')
    if len(fake_seen) == 0:
        raise RuntimeError('No fake samples were found in the seen portion of the manifest.')

    real_holdout_sources = parse_csv_list(args.real_holdout_sources)
    fake_holdout_generators = parse_csv_list(args.fake_holdout_generators)
    fake_holdout_sources = parse_csv_list(args.fake_holdout_sources)

    if not real_holdout_sources and args.auto_holdout_real_sources:
        counts = real_seen['source'].value_counts().to_dict()
        real_holdout_sources = greedy_cover(counts, args.val_real_total)
    if not fake_holdout_generators and args.auto_holdout_fake_generators:
        counts = fake_seen['generator'].value_counts().to_dict()
        fake_holdout_generators = greedy_cover(counts, args.val_fake_total)

    real_holdout_sources = sorted(set(real_holdout_sources))
    fake_holdout_generators = sorted(set(fake_holdout_generators))
    fake_holdout_sources = sorted(set(fake_holdout_sources))

    if not real_holdout_sources:
        raise RuntimeError('Please provide --real_holdout_sources or use --auto_holdout_real_sources.')
    if not fake_holdout_generators and not fake_holdout_sources:
        raise RuntimeError('Please provide --fake_holdout_generators / --fake_holdout_sources or use --auto_holdout_fake_generators.')

    real_val_pool = real_seen[real_seen['source'].isin(real_holdout_sources)].copy().reset_index(drop=True)
    fake_val_mask = fake_seen['generator'].isin(fake_holdout_generators)
    if fake_holdout_sources:
        fake_val_mask = fake_val_mask | fake_seen['source'].isin(fake_holdout_sources)
    fake_val_pool = fake_seen[fake_val_mask].copy().reset_index(drop=True)

    if len(real_val_pool) == 0:
        raise RuntimeError('Held-out real val pool is empty. Check --real_holdout_sources.')
    if len(fake_val_pool) == 0:
        raise RuntimeError('Held-out fake val pool is empty. Check --fake_holdout_generators / --fake_holdout_sources.')

    train_real_df = real_seen[~real_seen['source'].isin(real_holdout_sources)].copy().reset_index(drop=True)
    train_fake_mask = (~fake_seen['generator'].isin(fake_holdout_generators))
    if fake_holdout_sources:
        train_fake_mask = train_fake_mask & (~fake_seen['source'].isin(fake_holdout_sources))
    train_fake_df = fake_seen[train_fake_mask].copy().reset_index(drop=True)

    if len(train_real_df) == 0:
        raise RuntimeError('Train real pool became empty. Reduce held-out real sources.')
    if len(train_fake_df) == 0:
        raise RuntimeError('Train fake pool became empty. Reduce held-out fake generators/sources.')

    val_real_df = sample_by_group(real_val_pool, total=args.val_real_total, group_col='source', seed=args.seed + 11)
    fake_group_col = 'generator' if len(fake_holdout_generators) > 0 else 'source'
    val_fake_df = sample_by_group(fake_val_pool, total=args.val_fake_total, group_col=fake_group_col, seed=args.seed + 23)

    train_df = pd.concat([train_real_df, train_fake_df], axis=0, ignore_index=True)
    train_df = train_df.drop_duplicates(subset=['path'], keep='first')
    train_df = train_df.sample(frac=1.0, random_state=args.seed + 71).reset_index(drop=True)

    val_df = pd.concat([val_real_df, val_fake_df], axis=0, ignore_index=True)
    val_df = val_df.drop_duplicates(subset=['path'], keep='first')
    val_df = val_df.sample(frac=1.0, random_state=args.seed + 89).reset_index(drop=True)

    ensure_dir(str(Path(args.train_csv).parent))
    train_df.to_csv(args.train_csv, index=False)
    ensure_dir(str(Path(args.val_csv).parent))
    val_df.to_csv(args.val_csv, index=False)
    ensure_dir(str(Path(args.test_unseen_csv).parent))
    test_unseen_df.to_csv(args.test_unseen_csv, index=False)

    summary = {
        'manifest': args.manifest,
        'targets': {
            'val_real_total': int(args.val_real_total),
            'val_fake_total': int(args.val_fake_total),
        },
        'holdout': {
            'real_sources': real_holdout_sources,
            'fake_generators': fake_holdout_generators,
            'fake_sources': fake_holdout_sources,
            'flat_dir_bucket_threshold': int(args.flat_dir_bucket_threshold),
            'hash_buckets': int(args.hash_buckets),
        },
        'splits': {
            'train': summarize_split(train_df),
            'val': summarize_split(val_df),
            'test_unseen': summarize_split(test_unseen_df),
        },
        'overlap_checks': {
            'path': overlap_stats(train_df, val_df, 'path'),
            'real_source': overlap_stats(train_df[train_df['label'].astype(int) == 0], val_df[val_df['label'].astype(int) == 0], 'source'),
            'fake_source': overlap_stats(train_df[train_df['label'].astype(int) == 1], val_df[val_df['label'].astype(int) == 1], 'source'),
            'fake_generator': overlap_stats(train_df[train_df['label'].astype(int) == 1], val_df[val_df['label'].astype(int) == 1], 'generator'),
        },
    }
    save_yaml(summary, args.summary_yaml)

    print(f'[DONE] wrote train {len(train_df)} rows -> {args.train_csv}')
    print(f'[INFO] train real={len(train_real_df)}, train fake={len(train_fake_df)}')
    print(f'[DONE] wrote val {len(val_df)} rows -> {args.val_csv}')
    print(f'[INFO] val real={len(val_real_df)}, val fake={len(val_fake_df)}')
    print(f'[DONE] wrote test_unseen {len(test_unseen_df)} rows -> {args.test_unseen_csv}')
    print(f'[INFO] held-out real sources={real_holdout_sources}')
    print(f'[INFO] held-out fake generators={fake_holdout_generators}')
    print(f'[INFO] held-out fake sources={fake_holdout_sources}')
    print(f'[DONE] wrote summary -> {args.summary_yaml}')


if __name__ == '__main__':
    main()
