#!/usr/bin/env python3
"""Package a candidate model release with checkpoint, configs, metrics and data checksums."""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def parse_args():
    parser = argparse.ArgumentParser(description='Package a model iteration release.')
    parser.add_argument('--iteration_id', required=True)
    parser.add_argument('--ckpt', required=True, help='Checkpoint to package as model/best.pt')
    parser.add_argument('--output_dir', required=True, help='Release directory.')
    parser.add_argument('--configs_dir', default='', help='Directory of rendered configs to copy.')
    parser.add_argument('--metrics_dir', default='', help='Directory of metrics JSON files to copy.')
    parser.add_argument('--data_dir', default='', help='Directory of split CSVs. Checksums will be recorded.')
    parser.add_argument('--copy_data_csv', action='store_true', help='Copy CSVs into release/data in addition to checksums.')
    parser.add_argument('--status', default='candidate', choices=['candidate', 'staged', 'production'])
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def copy_matching(src_dir: Path, dst_dir: Path, patterns):
    if not src_dir or not src_dir.exists():
        return []
    copied = []
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for src in sorted(src_dir.glob(pattern)):
            if src.is_file():
                dst = dst_dir / src.name
                shutil.copy2(src, dst)
                copied.append(str(dst))
    return copied


def git_sha() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_metrics_summary(metrics_dir: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if not metrics_dir or not metrics_dir.exists():
        return summary
    keys = ['n', 'ap', 'auroc', 'recall@p95', 'recall@p98', 'recall@p99', 'op_default_acc', 'op_p98_fpr', 'op_p98_threshold']
    for path in sorted(metrics_dir.glob('*.json')):
        data = load_json(path)
        summary[path.stem] = {k: data.get(k) for k in keys if k in data}
    return summary


def collect_thresholds(metrics_dir: Path) -> Dict[str, Any]:
    thresholds: Dict[str, Any] = {}
    if not metrics_dir or not metrics_dir.exists():
        return thresholds
    for path in sorted(metrics_dir.glob('*.json')):
        data = load_json(path)
        thresholds[path.stem] = {k: v for k, v in data.items() if k.startswith('thr@') or k.endswith('_threshold')}
    return thresholds


def main():
    args = parse_args()
    release_dir = Path(args.output_dir)
    model_dir = release_dir / 'model'
    config_dir = release_dir / 'config'
    metrics_out_dir = release_dir / 'metrics'
    data_out_dir = release_dir / 'data'
    threshold_dir = release_dir / 'threshold'
    for d in [model_dir, config_dir, metrics_out_dir, data_out_dir, threshold_dir]:
        d.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f'checkpoint not found: {ckpt}')
    shutil.copy2(ckpt, model_dir / 'best.pt')
    configs_dir = Path(args.configs_dir) if args.configs_dir else None
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else None
    data_dir = Path(args.data_dir) if args.data_dir else None
    copied_configs = copy_matching(configs_dir, config_dir, ['*.yaml', '*.yml']) if configs_dir else []
    copied_metrics = copy_matching(metrics_dir, metrics_out_dir, ['*.json', '*.csv']) if metrics_dir else []
    data_checksums: Dict[str, Any] = {}
    if data_dir and data_dir.exists():
        for path in sorted(data_dir.glob('*')):
            if path.is_file() and path.suffix.lower() in {'.csv', '.yaml', '.yml'}:
                data_checksums[path.name] = {'sha256': sha256_file(path), 'bytes': path.stat().st_size}
                if args.copy_data_csv and path.suffix.lower() == '.csv':
                    shutil.copy2(path, data_out_dir / path.name)
    with open(data_out_dir / 'checksums.json', 'w', encoding='utf-8') as f:
        json.dump(data_checksums, f, ensure_ascii=False, indent=2)
    thresholds = collect_thresholds(metrics_dir) if metrics_dir else {}
    with open(threshold_dir / 'thresholds.json', 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    manifest = {
        'artifact_format_version': 1,
        'iteration_id': args.iteration_id,
        'status': args.status,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'git_sha': git_sha(),
        'checkpoint': {'source': str(ckpt), 'release_path': 'model/best.pt', 'sha256': sha256_file(model_dir / 'best.pt'), 'bytes': (model_dir / 'best.pt').stat().st_size},
        'copied_configs': copied_configs,
        'copied_metrics': copied_metrics,
        'data_checksums': data_checksums,
        'metrics_summary': collect_metrics_summary(metrics_dir) if metrics_dir else {},
        'thresholds': thresholds,
    }
    with open(release_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f'[DONE] release packaged at {release_dir}')


if __name__ == '__main__':
    main()
