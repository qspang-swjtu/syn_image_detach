#!/usr/bin/env python3
"""Evaluate one checkpoint on multiple CSV splits and collect JSON metrics."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Run scripts/eval_with_outputs.py on a list of split CSVs.')
    parser.add_argument('--ckpt', required=True, help='Checkpoint to evaluate.')
    parser.add_argument('--base_config', default='configs/eval_small.yaml', help='Base eval YAML config.')
    parser.add_argument('--data_dir', required=True, help='Directory containing split CSVs.')
    parser.add_argument('--output_dir', required=True, help='Eval output root. Will contain metrics/, predictions/, eval_configs/.')
    parser.add_argument('--splits', nargs='*', default=['val=val.csv', 'test_seen=test_seen.csv', 'test_unseen=test_unseen.csv', 'test_all=test_all.csv'], help='Split specs in NAME=CSV format.')
    parser.add_argument('--device', default='', help='Optional device passed to eval script.')
    parser.add_argument('--tta', default='', help='Optional eval.tta override.')
    parser.add_argument('--threshold', default='0.5', help='Default threshold passed to eval script.')
    parser.add_argument('--skip_missing', action='store_true', help='Skip split CSVs that do not exist.')
    parser.add_argument('--python', default=sys.executable, help='Python executable.')
    return parser.parse_args()


def parse_split(spec: str) -> Tuple[str, str]:
    if '=' not in spec:
        raise ValueError(f'Invalid split spec {spec!r}; expected NAME=CSV')
    name, path = spec.split('=', 1)
    name, path = name.strip(), path.strip()
    if not name or not path:
        raise ValueError(f'Invalid split spec {spec!r}; expected NAME=CSV')
    return name, path


def set_by_dot_path(cfg: Dict[str, Any], key: str, value: Any):
    parts = [p for p in key.split('.') if p]
    cursor = cfg
    for part in parts[:-1]:
        if part not in cursor or cursor[part] is None:
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (repo_root / data_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    metrics_dir = output_dir / 'metrics'
    predictions_dir = output_dir / 'predictions'
    configs_dir = output_dir / 'eval_configs'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    base_config = Path(args.base_config)
    if not base_config.is_absolute():
        base_config = (repo_root / base_config).resolve()
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = (repo_root / ckpt).resolve()

    rows: List[Dict[str, Any]] = []
    for spec in args.splits:
        split_name, csv_path_text = parse_split(spec)
        csv_path = Path(csv_path_text)
        if not csv_path.is_absolute():
            csv_path = data_dir / csv_path
        csv_path = csv_path.resolve()
        if not csv_path.exists():
            message = f'[WARN] missing split {split_name}: {csv_path}'
            if args.skip_missing:
                print(message)
                continue
            raise FileNotFoundError(message)

        cfg = load_yaml(base_config)
        set_by_dot_path(cfg, 'data.test_csv', str(csv_path))
        set_by_dot_path(cfg, 'output_dir', str(output_dir / 'runs' / split_name))
        set_by_dot_path(cfg, 'model.pretrained_rgb', False)
        set_by_dot_path(cfg, 'model.pretrained_forensic', False)
        if args.tta:
            set_by_dot_path(cfg, 'eval.tta', int(args.tta))
        rendered_config = configs_dir / f'{split_name}.yaml'
        save_yaml(rendered_config, cfg)

        metrics_path = metrics_dir / f'{split_name}.json'
        predictions_path = predictions_dir / f'{split_name}.csv'
        cmd = [args.python, 'scripts/eval_with_outputs.py', '--config', str(rendered_config), '--ckpt', str(ckpt), '--out', str(metrics_path), '--predictions_out', str(predictions_path), '--threshold', str(args.threshold)]
        if args.device:
            cmd.extend(['--device', args.device])
        print('[RUN]', ' '.join(cmd))
        subprocess.run(cmd, cwd=repo_root, check=True)
        metrics = load_json(metrics_path)
        rows.append({
            'split': split_name,
            'n': metrics.get('n'),
            'ap': metrics.get('ap'),
            'auroc': metrics.get('auroc'),
            'recall@p95': metrics.get('recall@p95'),
            'recall@p98': metrics.get('recall@p98'),
            'recall@p99': metrics.get('recall@p99'),
            'op_p98_fpr': metrics.get('op_p98_fpr'),
            'op_p98_threshold': metrics.get('op_p98_threshold'),
            'op_default_acc': metrics.get('op_default_acc'),
        })

    summary_csv = output_dir / 'summary.csv'
    if rows:
        with open(summary_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f'[DONE] metrics -> {metrics_dir}')
    print(f'[DONE] summary -> {summary_csv}')


if __name__ == '__main__':
    main()
