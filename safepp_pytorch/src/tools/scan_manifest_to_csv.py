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
    parser = argparse.ArgumentParser(description='Scan dataset roots from a YAML manifest and write a canonical CSV index.')
    parser.add_argument('--manifest', type=str, required=True, help='YAML manifest describing sources and labels.')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV path.')
    parser.add_argument('--summary_yaml', type=str, default='', help='Optional summary YAML path.')
    parser.add_argument('--strict', action='store_true', help='Fail when a listed directory does not exist or contains no images.')
    return parser.parse_args()


def iter_images(root: Path, recursive: bool, exts: Iterable[str]) -> List[Path]:
    exts = {'.' + e.lower().lstrip('.') for e in exts}
    if recursive:
        files = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in root.glob('*') if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def make_row(path: Path, spec: Dict) -> Dict:
    return {
        'path': str(path.resolve()),
        'label': int(spec['label']),
        'source': str(spec.get('name', path.parent.name or 'unknown')),
        'dataset': str(spec.get('dataset', 'unknown')),
        'domain': str(spec.get('domain', 'unknown')),
        'generator': str(spec.get('generator', 'real' if int(spec['label']) == 0 else 'unknown')),
        'split_hint': str(spec.get('split_hint', 'seen')),
        'sample_weight': float(spec.get('sample_weight', 1.0)),
        'is_hard_negative': int(spec.get('is_hard_negative', 0)),
    }


def build_summary(df: pd.DataFrame, manifest_path: str) -> Dict:
    summary = {
        'manifest': manifest_path,
        'num_rows': int(len(df)),
        'by_label': {int(k): int(v) for k, v in df['label'].value_counts().sort_index().items()},
        'by_source': {str(k): int(v) for k, v in df['source'].value_counts().items()},
        'by_dataset': {str(k): int(v) for k, v in df['dataset'].value_counts().items()},
        'by_split_hint': {str(k): int(v) for k, v in df['split_hint'].value_counts().items()},
    }
    return summary


def main():
    args = parse_args()
    manifest = load_yaml(args.manifest)
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
            if args.strict:
                continue
            else:
                print(f'[WARN] missing path: {root}')
                continue
        files = iter_images(root, recursive=recursive, exts=exts)
        if not files:
            empty.append(str(root))
            if args.strict:
                continue
            else:
                print(f'[WARN] no images found: {root}')
                continue
        for path in files:
            rows.append(make_row(path, spec))
        print(f'[OK] {spec.get("name", root.name)} -> {len(files)} files')

    if args.strict and (missing or empty):
        raise RuntimeError(f'Strict mode failed. Missing={missing}, empty={empty}')
    if not rows:
        raise RuntimeError('No image rows were collected from the manifest.')

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=['path'], keep='first').reset_index(drop=True)
    out_csv = Path(args.output_csv)
    ensure_dir(str(out_csv.parent))
    df.to_csv(out_csv, index=False)
    print(f'[DONE] wrote {len(df)} rows -> {out_csv}')

    summary_path = Path(args.summary_yaml) if args.summary_yaml else out_csv.with_name(out_csv.stem + '_summary.yaml')
    save_yaml(build_summary(df, args.manifest), str(summary_path))
    print(f'[DONE] wrote summary -> {summary_path}')


if __name__ == '__main__':
    main()
