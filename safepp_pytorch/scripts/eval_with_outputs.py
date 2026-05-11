#!/usr/bin/env python3
"""Enhanced SAFE++ evaluation with JSON metrics and per-image prediction CSV.

This keeps src/eval.py untouched and is used by the model iteration workflow.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SRC_ROOT = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CSVDataset  # noqa: E402
from data.transforms import build_val_transform, five_crop_tensor_views  # noqa: E402
from models.safepp import build_model  # noqa: E402
from utils.common import load_yaml  # noqa: E402
from utils.metrics import binary_metrics, recall_at_precision, threshold_for_precision  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SAFE++ checkpoint and write machine-readable outputs.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', required=True, help='Metrics JSON output path')
    parser.add_argument('--predictions_out', default='', help='Optional per-image prediction CSV')
    parser.add_argument('--threshold', type=float, default=0.5, help='Default operating threshold')
    parser.add_argument('--device', default='', help='cpu / cuda / cuda:0')
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return float(num / den)


def operating_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, prefix: str) -> Dict[str, Any]:
    y_true = y_true.astype(np.int64)
    y_pred = (y_prob >= threshold).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {
        f'{prefix}_threshold': float(threshold),
        f'{prefix}_acc': safe_div(tp + tn, tp + fp + tn + fn),
        f'{prefix}_precision': safe_div(tp, tp + fp),
        f'{prefix}_recall': safe_div(tp, tp + fn),
        f'{prefix}_fpr': safe_div(fp, fp + tn),
        f'{prefix}_fnr': safe_div(fn, fn + tp),
        f'{prefix}_real_acc': safe_div(tn, tn + fp),
        f'{prefix}_fake_acc': safe_div(tp, tp + fn),
        f'{prefix}_tp': tp,
        f'{prefix}_fp': fp,
        f'{prefix}_tn': tn,
        f'{prefix}_fn': fn,
    }


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def write_json(path: str, payload: Dict[str, Any]):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(sanitize_for_json(payload), f, ensure_ascii=False, indent=2, allow_nan=False)


def load_model(cfg: Dict[str, Any], ckpt_path: str, device: torch.device):
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['ema'] if 'ema' in ckpt else ckpt['model']
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def batch_to_records(batch: Dict[str, Any], probs: np.ndarray) -> List[Dict[str, Any]]:
    labels = batch['label'].detach().cpu().numpy().astype(np.int64).reshape(-1)
    paths = batch.get('path', [''] * len(labels))
    sources = batch.get('source', [''] * len(labels))
    generators = batch.get('generator', [''] * len(labels))
    domains = batch.get('domain', [''] * len(labels))
    datasets = batch.get('dataset', [''] * len(labels))
    rows = []
    for i in range(len(labels)):
        rows.append({
            'path': str(paths[i]),
            'label': int(labels[i]),
            'prob_fake': float(probs[i]),
            'source': str(sources[i]) if i < len(sources) else '',
            'generator': str(generators[i]) if i < len(generators) else '',
            'domain': str(domains[i]) if i < len(domains) else '',
            'dataset': str(datasets[i]) if i < len(datasets) else '',
        })
    return rows


def row_to_record(row: pd.Series, prob: float) -> Dict[str, Any]:
    return {
        'path': str(row.get('path', '')),
        'label': int(row.get('label')),
        'prob_fake': float(prob),
        'source': str(row.get('source', '')),
        'generator': str(row.get('generator', '')),
        'domain': str(row.get('domain', '')),
        'dataset': str(row.get('dataset', '')),
    }


def write_predictions(path: str, records: List[Dict[str, Any]], threshold: float):
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    if len(df) > 0:
        df['pred_at_threshold'] = (df['prob_fake'].astype(float) >= threshold).astype(int)
        df['threshold'] = float(threshold)
    df.to_csv(out, index=False)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    device = resolve_device(args.device)
    model = load_model(cfg, args.ckpt, device)
    tta = int(cfg.get('eval', {}).get('tta', 1))
    records: List[Dict[str, Any]] = []

    if tta <= 1:
        ds = CSVDataset(cfg['data']['test_csv'], transform=build_val_transform(cfg))
        num_workers = int(cfg['data'].get('num_workers', 8))
        loader = DataLoader(
            ds,
            batch_size=cfg['eval']['batch_size_per_gpu'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(cfg['data'].get('pin_memory', True)),
            persistent_workers=bool(cfg['data'].get('persistent_workers', False)) and num_workers > 0,
            drop_last=False,
        )
        y_true, y_prob = [], []
        for batch in tqdm(loader):
            x = batch['image'].to(device, non_blocking=True)
            labels_np = batch['label'].detach().cpu().numpy().astype(np.int64).reshape(-1)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                prob_np = torch.sigmoid(model(x)).detach().cpu().numpy().reshape(-1)
            y_true.extend(labels_np.tolist())
            y_prob.extend(prob_np.tolist())
            if args.predictions_out:
                records.extend(batch_to_records(batch, prob_np))
    else:
        df = pd.read_csv(cfg['data']['test_csv'])
        y_true, y_prob = [], []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img = Image.open(row['path']).convert('RGB')
            views = five_crop_tensor_views(img, cfg)
            x = torch.stack(views, dim=0).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                prob = torch.sigmoid(model(x)).mean().item()
            y_true.append(int(row['label']))
            y_prob.append(float(prob))
            if args.predictions_out:
                records.append(row_to_record(row, prob))

    y_true = np.array(y_true, dtype=np.int64)
    y_prob = np.array(y_prob, dtype=np.float64)
    metrics: Dict[str, Any] = binary_metrics(y_true, y_prob)
    metrics['n'] = int(len(y_true))
    metrics['num_real'] = int((y_true == 0).sum())
    metrics['num_fake'] = int((y_true == 1).sum())
    metrics.update(operating_metrics(y_true, y_prob, threshold=args.threshold, prefix='op_default'))

    for p in [0.95, 0.98, 0.99]:
        p_int = int(p * 100)
        rec = recall_at_precision(y_true, y_prob, p)
        thr = threshold_for_precision(y_true, y_prob, p)
        metrics[f'recall@p{p_int}'] = None if rec is None else rec
        metrics[f'thr@p{p_int}'] = None if thr is None else thr
        if thr is not None:
            metrics.update(operating_metrics(y_true, y_prob, threshold=thr, prefix=f'op_p{p_int}'))

    metrics = sanitize_for_json(metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
    write_json(args.out, metrics)
    write_predictions(args.predictions_out, records, threshold=args.threshold)


if __name__ == '__main__':
    main()
