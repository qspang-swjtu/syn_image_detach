import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

DEFAULTS: Dict[str, object] = {
    'source': 'unknown',
    'dataset': 'unknown',
    'domain': 'unknown',
    'generator': 'unknown',
    'split_hint': 'seen',
    'sample_weight': 1.0,
    'is_hard_negative': 0,
    'added_iter': 'unknown',
}

PREFERRED_COLS = [
    'path',
    'label',
    'source',
    'dataset',
    'domain',
    'generator',
    'split_hint',
    'sample_weight',
    'is_hard_negative',
    'hard_type',
    'added_iter',
]


def parse_csv_list(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(',') if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge a stable base image-level index CSV with one or more incremental CSVs.'
    )
    parser.add_argument('--base_csv', type=str, required=True, help='Stable base image-level CSV. Must contain path,label.')
    parser.add_argument('--append_csvs', type=str, default='', help='Comma-separated incremental CSV list.')
    parser.add_argument('--append_dir', type=str, default='', help='Optional directory; all *.csv files inside are appended in sorted order.')
    parser.add_argument('--output_csv', type=str, required=True, help='Merged output CSV for this iteration.')
    parser.add_argument('--summary_yaml', type=str, default='', help='Optional YAML/JSON summary path.')
    parser.add_argument('--dedup_key', type=str, default='path', choices=['path'])
    parser.add_argument('--keep', type=str, default='last', choices=['first', 'last'], help='When paths repeat, last lets increments override base metadata.')
    parser.add_argument('--added_iter', type=str, default='', help='Fill empty added_iter for appended rows.')
    parser.add_argument('--strict_paths', action='store_true', help='Fail if any image path in the merged CSV does not exist.')
    return parser.parse_args()


def read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, engine='pyarrow')
    except Exception:
        return pd.read_csv(path)


def normalize(df: pd.DataFrame, source_name: str, added_iter: str) -> pd.DataFrame:
    if 'path' not in df.columns or 'label' not in df.columns:
        raise ValueError(f'{source_name} must contain path,label columns.')

    out = df.copy()
    out['path'] = out['path'].astype(str).str.strip()
    out = out[out['path'] != ''].copy()
    out['label'] = pd.to_numeric(out['label'], errors='raise').astype(int)

    bad = ~out['label'].isin([0, 1])
    if bad.any():
        vals = sorted(out.loc[bad, 'label'].unique().tolist())
        raise ValueError(f'{source_name} contains non-binary labels: {vals}')

    for col, default in DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].fillna(default)

    if added_iter:
        mask = out['added_iter'].astype(str).isin(['', 'unknown', 'nan', 'None'])
        out.loc[mask, 'added_iter'] = added_iter

    out['sample_weight'] = pd.to_numeric(out['sample_weight'], errors='coerce').fillna(1.0)
    out['is_hard_negative'] = pd.to_numeric(out['is_hard_negative'], errors='coerce').fillna(0).astype(int)
    out['_index_source'] = source_name
    return out


def discover_append_csvs(append_csvs: str, append_dir: str) -> List[str]:
    paths = parse_csv_list(append_csvs)
    if append_dir:
        root = Path(append_dir)
        if not root.exists():
            raise FileNotFoundError(f'append_dir does not exist: {append_dir}')
        paths.extend(str(p) for p in sorted(root.glob('*.csv')))
    # Preserve order while deduplicating the list itself.
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def write_summary(summary: Dict, path: str) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == '.json' or yaml is None:
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        out.write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding='utf-8')


def main():
    args = parse_args()
    append_paths = discover_append_csvs(args.append_csvs, args.append_dir)

    frames = [normalize(read_csv(args.base_csv), source_name='base', added_iter='base')]
    for path in append_paths:
        frames.append(normalize(read_csv(path), source_name=str(path), added_iter=args.added_iter))

    merged_raw = pd.concat(frames, axis=0, ignore_index=True)
    before = int(len(merged_raw))
    merged = merged_raw.drop_duplicates(subset=[args.dedup_key], keep=args.keep).reset_index(drop=True)
    after = int(len(merged))

    if args.strict_paths:
        missing = [p for p in merged['path'].astype(str).tolist() if not Path(p).exists()]
        if missing:
            preview = missing[:20]
            raise FileNotFoundError(f'{len(missing)} image paths do not exist. First examples: {preview}')

    extra_cols = [c for c in merged.columns if c not in PREFERRED_COLS and c != '_index_source']
    cols = [c for c in PREFERRED_COLS if c in merged.columns] + extra_cols
    merged = merged[cols]

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)

    summary = {
        'base_csv': args.base_csv,
        'append_csvs': append_paths,
        'output_csv': str(output),
        'dedup_key': args.dedup_key,
        'keep': args.keep,
        'rows_before_dedup': before,
        'rows_after_dedup': after,
        'duplicates_removed': before - after,
        'by_label': {int(k): int(v) for k, v in merged['label'].value_counts().sort_index().items()},
        'by_split_hint': {str(k): int(v) for k, v in merged['split_hint'].value_counts().items()},
        'hard_rows': int((merged['is_hard_negative'].astype(int) == 1).sum()) if 'is_hard_negative' in merged.columns else 0,
        'num_unique_paths': int(merged['path'].astype(str).nunique()),
    }
    write_summary(summary, args.summary_yaml)

    print(f'[DONE] wrote merged index: {output}')
    print(f'[INFO] base rows + append rows before dedup: {before}')
    print(f'[INFO] final rows: {after}')
    print(f'[INFO] duplicates removed: {before - after}')
    print(f'[INFO] by label: {summary["by_label"]}')
    print(f'[INFO] by split_hint: {summary["by_split_hint"]}')
    print(f'[INFO] hard rows: {summary["hard_rows"]}')


if __name__ == '__main__':
    main()
