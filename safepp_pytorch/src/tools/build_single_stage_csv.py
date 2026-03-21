import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.common import ensure_dir, load_yaml, save_yaml  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build a single-stage training CSV from a manifest by sampling balanced real/fake data.'
    )
    parser.add_argument('--manifest', type=str, required=True, help='YAML manifest in the same format as manifests/small_sources_example.yaml.')
    parser.add_argument('--output_csv', type=str, required=True, help='Output train CSV path.')
    parser.add_argument('--val_csv', type=str, default='', help='Optional val CSV path. Defaults to <output_csv stem>_val.csv.')
    parser.add_argument('--test_unseen_csv', type=str, default='', help='Optional unseen test CSV path. Defaults to <output_csv stem>_test_unseen.csv.')
    parser.add_argument('--summary_yaml', type=str, default='', help='Optional summary YAML path.')
    parser.add_argument('--real_total', type=int, default=60000, help='Target number of real samples.')
    parser.add_argument('--fake_total', type=int, default=60000, help='Target number of fake samples.')
    parser.add_argument('--val_real_total', type=int, default=8000, help='Target number of real validation samples.')
    parser.add_argument('--val_fake_total', type=int, default=8000, help='Target number of fake validation samples.')
    parser.add_argument('--real_group_col', type=str, default='source', choices=['source', 'dataset', 'domain', 'generator'])
    parser.add_argument('--fake_group_col', type=str, default='generator', choices=['source', 'dataset', 'domain', 'generator'])
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--strict', action='store_true', help='Fail when a listed directory does not exist or contains no images.')
    return parser.parse_args()


def iter_images(root: Path, recursive: bool, exts: Iterable[str]) -> List[Path]:
    valid_exts = {'.' + e.lower().lstrip('.') for e in exts}
    if recursive:
        files = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in valid_exts]
    else:
        files = [p for p in root.glob('*') if p.is_file() and p.suffix.lower() in valid_exts]
    return sorted(files)


def make_row(path: Path, spec: Dict) -> Dict:
    label = int(spec['label'])
    return {
        'path': str(path.resolve()),
        'label': label,
        'source': str(spec.get('name', path.parent.name or 'unknown')),
        'dataset': str(spec.get('dataset', 'unknown')),
        'domain': str(spec.get('domain', 'unknown')),
        'generator': str(spec.get('generator', 'real' if label == 0 else 'unknown')),
        'split_hint': str(spec.get('split_hint', 'seen')),
        'sample_weight': float(spec.get('sample_weight', 1.0)),
        'is_hard_negative': int(spec.get('is_hard_negative', 0)),
    }


def scan_manifest(manifest: Dict, strict: bool) -> pd.DataFrame:
    recursive_default = bool(manifest.get('recursive', True))
    exts_default = manifest.get('exts', ['jpg', 'jpeg', 'png', 'webp', 'bmp'])
    sources = manifest.get('sources', [])
    if not sources:
        raise ValueError('Manifest must contain a non-empty `sources` list.')

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
        rows.extend(make_row(path, spec) for path in files)
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


def allocate_evenly(counts: Dict[str, int], total: int) -> Dict[str, int]:
    counts = {str(k): int(v) for k, v in counts.items() if int(v) > 0}
    total = min(int(total), int(sum(counts.values())))
    alloc = {k: 0 for k in counts}
    if total <= 0 or not counts:
        return alloc

    alive = set(counts.keys())
    while total > 0 and alive:
        base = max(1, total // len(alive))
        progressed = False
        for key in list(alive):
            room = counts[key] - alloc[key]
            take = min(base, room)
            if take > 0:
                alloc[key] += take
                total -= take
                progressed = True
            if alloc[key] >= counts[key]:
                alive.remove(key)
        if not progressed:
            break

    if total > 0:
        for key in sorted(counts.keys(), key=lambda x: counts[x] - alloc[x], reverse=True):
            if total <= 0:
                break
            room = counts[key] - alloc[key]
            if room <= 0:
                continue
            take = min(room, total)
            alloc[key] += take
            total -= take
    return alloc


def stable_seed_offset(text: str) -> int:
    total = 0
    for idx, ch in enumerate(text):
        total += (idx + 1) * ord(ch)
    return total % 100003


def sample_by_group(df: pd.DataFrame, total: int, group_col: str, seed: int) -> pd.DataFrame:
    if len(df) == 0 or total <= 0:
        return df.iloc[0:0].copy()

    work = df.copy()
    if group_col not in work.columns:
        work[group_col] = 'unknown'
    work[group_col] = work[group_col].fillna('unknown').astype(str)

    counts = {str(k): int(v) for k, v in work[group_col].value_counts().items()}
    alloc = allocate_evenly(counts, total)

    parts: List[pd.DataFrame] = []
    for group_name, quota in alloc.items():
        group_df = work[work[group_col] == group_name]
        shuffled = group_df.sample(frac=1.0, random_state=seed + stable_seed_offset(group_name))
        parts.append(shuffled.head(quota))

    if not parts:
        return work.iloc[0:0].copy()
    out = pd.concat(parts, axis=0, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def split_selected(df: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0 or len(selected) == 0:
        return df.copy().reset_index(drop=True)
    selected_paths = set(selected['path'].astype(str).tolist())
    return df[~df['path'].astype(str).isin(selected_paths)].copy().reset_index(drop=True)


def summarize_split(df: pd.DataFrame) -> Dict:
    fake_df = df[df['label'].astype(int) == 1].copy()
    real_df = df[df['label'].astype(int) == 0].copy()
    return {
        'num_rows': int(len(df)),
        'by_label': {int(k): int(v) for k, v in df['label'].value_counts().sort_index().items()},
        'real_by_source': {str(k): int(v) for k, v in real_df['source'].value_counts().items()},
        'fake_by_source': {str(k): int(v) for k, v in fake_df['source'].value_counts().items()},
        'fake_by_generator': {str(k): int(v) for k, v in fake_df['generator'].value_counts().items()},
        'fake_by_dataset': {str(k): int(v) for k, v in fake_df['dataset'].value_counts().items()},
    }


def summarize(train_df: pd.DataFrame, val_df: pd.DataFrame, test_unseen_df: pd.DataFrame, manifest_path: str, real_group_col: str, fake_group_col: str, real_target: int, fake_target: int, val_real_target: int, val_fake_target: int) -> Dict:
    return {
        'manifest': manifest_path,
        'targets': {
            'real_total': int(real_target),
            'fake_total': int(fake_target),
            'val_real_total': int(val_real_target),
            'val_fake_total': int(val_fake_target),
            'real_group_col': real_group_col,
            'fake_group_col': fake_group_col,
        },
        'splits': {
            'train': summarize_split(train_df),
            'val': summarize_split(val_df),
            'test_unseen': summarize_split(test_unseen_df),
        },
    }


def main():
    args = parse_args()
    manifest = load_yaml(args.manifest)
    all_df = scan_manifest(manifest, strict=args.strict)
    if 'split_hint' not in all_df.columns:
        all_df['split_hint'] = 'seen'
    all_df['split_hint'] = all_df['split_hint'].map(normalize_split_hint)

    unseen_mask = all_df['split_hint'] == 'unseen'
    test_unseen_df = all_df[unseen_mask].copy().reset_index(drop=True)
    sampled_df = all_df[~unseen_mask].copy().reset_index(drop=True)

    real_pool = sampled_df[sampled_df['label'].astype(int) == 0].copy().reset_index(drop=True)
    fake_pool = sampled_df[sampled_df['label'].astype(int) == 1].copy().reset_index(drop=True)
    if len(real_pool) == 0:
        raise RuntimeError('No real samples were found in the seen portion of the manifest.')
    if len(fake_pool) == 0:
        raise RuntimeError('No fake samples were found in the seen portion of the manifest.')

    val_real_df = sample_by_group(real_pool, total=args.val_real_total, group_col=args.real_group_col, seed=args.seed + 11)
    val_fake_df = sample_by_group(fake_pool, total=args.val_fake_total, group_col=args.fake_group_col, seed=args.seed + 23)

    train_real_pool = split_selected(real_pool, val_real_df)
    train_fake_pool = split_selected(fake_pool, val_fake_df)

    real_df = sample_by_group(train_real_pool, total=args.real_total, group_col=args.real_group_col, seed=args.seed + 37)
    fake_df = sample_by_group(train_fake_pool, total=args.fake_total, group_col=args.fake_group_col, seed=args.seed + 53)

    train_df = pd.concat([real_df, fake_df], axis=0, ignore_index=True)
    train_df = train_df.drop_duplicates(subset=['path'], keep='first')
    train_df = train_df.sample(frac=1.0, random_state=args.seed + 71).reset_index(drop=True)

    val_df = pd.concat([val_real_df, val_fake_df], axis=0, ignore_index=True)
    val_df = val_df.drop_duplicates(subset=['path'], keep='first')
    val_df = val_df.sample(frac=1.0, random_state=args.seed + 89).reset_index(drop=True)

    out_csv = Path(args.output_csv)
    ensure_dir(str(out_csv.parent))
    train_df.to_csv(out_csv, index=False)

    val_csv = Path(args.val_csv) if args.val_csv else out_csv.with_name(out_csv.stem + '_val.csv')
    ensure_dir(str(val_csv.parent))
    val_df.to_csv(val_csv, index=False)

    test_unseen_csv = Path(args.test_unseen_csv) if args.test_unseen_csv else out_csv.with_name(out_csv.stem + '_test_unseen.csv')
    ensure_dir(str(test_unseen_csv.parent))
    test_unseen_df.to_csv(test_unseen_csv, index=False)

    summary_path = Path(args.summary_yaml) if args.summary_yaml else out_csv.with_name(out_csv.stem + '_summary.yaml')
    save_yaml(
        summarize(
            train_df,
            val_df,
            test_unseen_df,
            manifest_path=args.manifest,
            real_group_col=args.real_group_col,
            fake_group_col=args.fake_group_col,
            real_target=args.real_total,
            fake_target=args.fake_total,
            val_real_target=args.val_real_total,
            val_fake_target=args.val_fake_total,
        ),
        str(summary_path),
    )

    print(f'[DONE] wrote train {len(train_df)} rows -> {out_csv}')
    print(f'[INFO] train real={len(real_df)}, train fake={len(fake_df)}')
    print(f'[DONE] wrote val {len(val_df)} rows -> {val_csv}')
    print(f'[INFO] val real={len(val_real_df)}, val fake={len(val_fake_df)}')
    print(f'[DONE] wrote test_unseen {len(test_unseen_df)} rows -> {test_unseen_csv}')
    print(f'[DONE] wrote summary -> {summary_path}')


if __name__ == '__main__':
    main()
