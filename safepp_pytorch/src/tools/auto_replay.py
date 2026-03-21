import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.dataset import CSVDataset  # noqa: E402
from data.transforms import build_val_transform, five_crop_tensor_views  # noqa: E402
from models.safepp import build_model  # noqa: E402
from utils.common import load_yaml  # noqa: E402
from utils.metrics import threshold_for_precision  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Mine hard examples and build a replay buffer CSV.')
    parser.add_argument('--config', type=str, required=True, help='Model/eval yaml used to build transforms and network.')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path. EMA is preferred if present.')
    parser.add_argument('--candidate_csv', type=str, required=True, help='Labeled candidate pool. Must contain path,label.')
    parser.add_argument('--output_buffer_csv', type=str, required=True, help='Where to write/update replay buffer CSV.')
    parser.add_argument('--base_train_csv', type=str, default='', help='Optional base train CSV to merge with replay buffer.')
    parser.add_argument('--merged_output_csv', type=str, default='', help='Optional merged train CSV output path.')
    parser.add_argument('--calib_csv', type=str, default='', help='Optional calibration CSV used to derive threshold for target precision.')
    parser.add_argument('--precision', type=float, default=0.98, help='Target precision used to derive deployment threshold.')
    parser.add_argument('--decision_thr', type=float, default=None, help='Manual decision threshold. Overrides --calib_csv if set.')
    parser.add_argument('--tta', type=int, default=None, help='Override TTA views. Defaults to config eval.tta.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for center-crop inference.')
    parser.add_argument('--topk_real', type=int, default=40000, help='How many hard real images to keep.')
    parser.add_argument('--topk_fake', type=int, default=40000, help='How many hard fake images to keep.')
    parser.add_argument('--topk_uncertain', type=int, default=20000, help='How many uncertain samples to keep around threshold.')
    parser.add_argument('--errors_only', action='store_true', help='Only keep actual model mistakes, not near-miss examples.')
    parser.add_argument('--max_buffer', type=int, default=200000, help='Max replay buffer size after merge.')
    parser.add_argument('--source_col', type=str, default='source', help='CSV source column name. Falls back to parent dir if missing.')
    parser.add_argument('--weight_real', type=float, default=3.0, help='sample_weight assigned to hard real images.')
    parser.add_argument('--weight_fake', type=float, default=3.0, help='sample_weight assigned to hard fake images.')
    parser.add_argument('--weight_uncertain', type=float, default=1.5, help='sample_weight assigned to uncertain images.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_model(cfg, ckpt_path: str, device: torch.device):
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['ema'] if 'ema' in ckpt else ckpt['model']
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def infer_source_from_path(path: str) -> str:
    parent = Path(path).parent.name.strip()
    return parent if parent else 'unknown'


def ensure_source(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    out = df.copy()
    if source_col not in out.columns:
        out[source_col] = out['path'].astype(str).map(infer_source_from_path)
    out[source_col] = out[source_col].fillna('unknown').astype(str)
    return out


@torch.no_grad()
def score_csv(cfg, model, csv_path: str, device: torch.device, batch_size: int, tta: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if tta <= 1:
        ds = CSVDataset(csv_path, transform=build_val_transform(cfg))
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg['data'].get('num_workers', 8),
            pin_memory=cfg['data'].get('pin_memory', True),
            persistent_workers=cfg['data'].get('persistent_workers', True),
            drop_last=False,
        )
        scores = []
        for batch in tqdm(loader, desc=f'scoring {Path(csv_path).name}'):
            x = batch['image'].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                prob = torch.sigmoid(model(x)).detach().cpu().numpy().tolist()
            scores.extend(prob)
        df = df.copy()
        df['score'] = scores
        return df

    df = df.copy()
    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'scoring {Path(csv_path).name} tta={tta}'):
        img = Image.open(row['path']).convert('RGB')
        views = five_crop_tensor_views(img, cfg)
        if tta > len(views):
            raise ValueError(f'TTA={tta} requested, but only {len(views)} crop views are available.')
        x = torch.stack(views[:tta], dim=0).to(device)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(x)
            prob = torch.sigmoid(logits).mean().item()
        scores.append(prob)
    df['score'] = scores
    return df


def calibrate_threshold(cfg, model, calib_csv: str, device: torch.device, batch_size: int, tta: int, target_precision: float) -> float:
    calib_df = score_csv(cfg, model, calib_csv, device=device, batch_size=batch_size, tta=tta)
    y_true = calib_df['label'].astype(int).to_numpy()
    y_prob = calib_df['score'].astype(float).to_numpy()
    thr = threshold_for_precision(y_true, y_prob, target_precision)
    if thr is None:
        raise RuntimeError(f'Could not derive threshold for precision={target_precision} from {calib_csv}.')
    return float(thr)


def sort_and_take(df: pd.DataFrame, by: str, ascending: bool, k: int) -> pd.DataFrame:
    if k <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()
    return df.sort_values(by=by, ascending=ascending).head(k).copy()


def select_hard_examples(
    scored_df: pd.DataFrame,
    decision_thr: float,
    topk_real: int,
    topk_fake: int,
    topk_uncertain: int,
    weight_real: float,
    weight_fake: float,
    weight_uncertain: float,
    source_col: str,
    errors_only: bool,
) -> pd.DataFrame:
    df = ensure_source(scored_df, source_col=source_col)
    df = df.copy()
    df['label'] = df['label'].astype(int)
    df['score'] = df['score'].astype(float)
    df['pred_label'] = (df['score'] >= decision_thr).astype(int)
    df['is_error'] = (df['pred_label'] != df['label']).astype(int)

    real_df = df[df['label'] == 0].copy()
    fake_df = df[df['label'] == 1].copy()

    real_df['priority'] = real_df['score']
    real_df['margin_to_threshold'] = real_df['score'] - decision_thr
    real_df['hard_type'] = np.where(real_df['is_error'] == 1, 'real_false_positive', 'real_near_miss')
    if errors_only:
        real_df = real_df[real_df['is_error'] == 1]
    real_df = sort_and_take(real_df, by='priority', ascending=False, k=topk_real)
    real_df['sample_weight'] = float(weight_real)

    fake_df['priority'] = 1.0 - fake_df['score']
    fake_df['margin_to_threshold'] = decision_thr - fake_df['score']
    fake_df['hard_type'] = np.where(fake_df['is_error'] == 1, 'fake_false_negative', 'fake_near_miss')
    if errors_only:
        fake_df = fake_df[fake_df['is_error'] == 1]
    fake_df = sort_and_take(fake_df, by='score', ascending=True, k=topk_fake)
    fake_df['sample_weight'] = float(weight_fake)

    selected_paths = set(real_df['path'].astype(str).tolist()) | set(fake_df['path'].astype(str).tolist())
    uncertain_df = df[~df['path'].astype(str).isin(selected_paths)].copy()
    uncertain_df['priority'] = 1.0 - np.abs(uncertain_df['score'] - decision_thr)
    uncertain_df['margin_to_threshold'] = -np.abs(uncertain_df['score'] - decision_thr)
    uncertain_df['hard_type'] = 'uncertain_band'
    if errors_only:
        uncertain_df = uncertain_df.iloc[0:0].copy()
    uncertain_df = sort_and_take(uncertain_df, by='priority', ascending=False, k=topk_uncertain)
    uncertain_df['sample_weight'] = float(weight_uncertain)

    replay_df = pd.concat([real_df, fake_df, uncertain_df], axis=0, ignore_index=True)
    if len(replay_df) == 0:
        return replay_df

    replay_df['is_hard_negative'] = 1
    replay_df['decision_thr'] = float(decision_thr)
    replay_df['mined_at'] = datetime.now(timezone.utc).isoformat()
    replay_df['mined_model'] = 'ema'  # this script prefers EMA when available.
    replay_df = replay_df.drop_duplicates(subset=['path'], keep='first')
    replay_df = replay_df.sort_values(['priority', 'sample_weight'], ascending=False)
    return replay_df


def keep_balanced_buffer(df: pd.DataFrame, max_buffer: int, source_col: str) -> pd.DataFrame:
    if len(df) <= max_buffer:
        return df.copy()
    if source_col not in df.columns:
        return df.sort_values(['priority', 'sample_weight'], ascending=False).head(max_buffer).copy()

    groups = []
    for source, group in df.groupby(source_col, sort=False):
        group = group.sort_values(['priority', 'sample_weight'], ascending=False).copy()
        groups.append((source, group))

    n_groups = max(1, len(groups))
    base_quota = max_buffer // n_groups
    kept = []
    leftovers = []
    for _, group in groups:
        take_n = min(len(group), base_quota)
        kept.append(group.head(take_n))
        if take_n < len(group):
            leftovers.append(group.iloc[take_n:])

    kept_df = pd.concat(kept, axis=0, ignore_index=True) if kept else df.iloc[0:0].copy()
    remaining = max_buffer - len(kept_df)
    if remaining > 0 and leftovers:
        tail = pd.concat(leftovers, axis=0, ignore_index=True)
        tail = tail.sort_values(['priority', 'sample_weight'], ascending=False).head(remaining)
        kept_df = pd.concat([kept_df, tail], axis=0, ignore_index=True)

    return kept_df.sort_values(['priority', 'sample_weight'], ascending=False).head(max_buffer).copy()


def merge_buffer(existing_df: Optional[pd.DataFrame], new_df: pd.DataFrame, max_buffer: int, source_col: str) -> pd.DataFrame:
    if existing_df is not None and len(existing_df) > 0:
        merged = pd.concat([new_df, existing_df], axis=0, ignore_index=True)
    else:
        merged = new_df.copy()
    merged = ensure_source(merged, source_col=source_col)
    sort_cols = [c for c in ['priority', 'sample_weight', 'score', 'mined_at'] if c in merged.columns]
    merged = merged.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    merged = merged.drop_duplicates(subset=['path'], keep='first')
    merged = keep_balanced_buffer(merged, max_buffer=max_buffer, source_col=source_col)
    return merged.reset_index(drop=True)


def merge_with_base_train(base_train_csv: str, replay_df: pd.DataFrame, output_csv: str, source_col: str):
    base_df = pd.read_csv(base_train_csv)
    base_df = ensure_source(base_df, source_col=source_col)
    replay_cols = ['path', 'label', source_col, 'sample_weight', 'is_hard_negative']
    optional = [c for c in ['hard_type', 'score', 'priority', 'margin_to_threshold', 'decision_thr', 'mined_at', 'mined_model'] if c in replay_df.columns]
    replay_view = replay_df[replay_cols + optional].copy()
    merged = pd.concat([replay_view, base_df], axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset=['path'], keep='first')
    merged.to_csv(output_csv, index=False)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    device = torch.device(args.device)
    model = load_model(cfg, args.ckpt, device=device)

    tta = args.tta if args.tta is not None else int(cfg.get('eval', {}).get('tta', 1))
    if args.decision_thr is not None:
        decision_thr = float(args.decision_thr)
    elif args.calib_csv:
        decision_thr = calibrate_threshold(
            cfg,
            model,
            calib_csv=args.calib_csv,
            device=device,
            batch_size=args.batch_size,
            tta=tta,
            target_precision=args.precision,
        )
    else:
        decision_thr = 0.5

    scored_df = score_csv(cfg, model, args.candidate_csv, device=device, batch_size=args.batch_size, tta=tta)
    scored_df = ensure_source(scored_df, source_col=args.source_col)

    replay_new = select_hard_examples(
        scored_df=scored_df,
        decision_thr=decision_thr,
        topk_real=args.topk_real,
        topk_fake=args.topk_fake,
        topk_uncertain=args.topk_uncertain,
        weight_real=args.weight_real,
        weight_fake=args.weight_fake,
        weight_uncertain=args.weight_uncertain,
        source_col=args.source_col,
        errors_only=args.errors_only,
    )

    existing_df = None
    existing_path = args.output_buffer_csv
    if os.path.exists(existing_path):
        existing_df = pd.read_csv(existing_path)

    replay_merged = merge_buffer(existing_df, replay_new, max_buffer=args.max_buffer, source_col=args.source_col)
    Path(args.output_buffer_csv).parent.mkdir(parents=True, exist_ok=True)
    replay_merged.to_csv(args.output_buffer_csv, index=False)

    if args.base_train_csv and args.merged_output_csv:
        Path(args.merged_output_csv).parent.mkdir(parents=True, exist_ok=True)
        merge_with_base_train(args.base_train_csv, replay_merged, args.merged_output_csv, source_col=args.source_col)

    summary = {
        'decision_thr': round(float(decision_thr), 6),
        'candidate_rows': int(len(scored_df)),
        'new_replay_rows': int(len(replay_new)),
        'final_buffer_rows': int(len(replay_merged)),
        'real_errors_in_new': int(((replay_new['label'] == 0) & (replay_new['is_error'] == 1)).sum()) if len(replay_new) else 0,
        'fake_errors_in_new': int(((replay_new['label'] == 1) & (replay_new['is_error'] == 1)).sum()) if len(replay_new) else 0,
        'uncertain_in_new': int((replay_new['hard_type'] == 'uncertain_band').sum()) if len(replay_new) else 0,
    }
    print(summary)


if __name__ == '__main__':
    main()
