import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CSVDataset
from data.transforms import build_val_transform, five_crop_tensor_views
from models.safepp import build_model
from utils.common import load_yaml
from utils.metrics import binary_metrics, recall_at_precision, threshold_for_precision


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt['ema'] if 'ema' in ckpt else ckpt['model']
    model.load_state_dict(state, strict=True)
    model.eval()

    if cfg['eval'].get('tta', 1) == 1:
        ds = CSVDataset(cfg['data']['test_csv'], transform=build_val_transform(cfg))
        loader = DataLoader(ds, batch_size=cfg['eval']['batch_size_per_gpu'], shuffle=False, num_workers=cfg['data']['num_workers'])
        y_true, y_prob = [], []
        for batch in tqdm(loader):
            x = batch['image'].to(device)
            y = batch['label'].numpy().tolist()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                prob = torch.sigmoid(model(x)).cpu().numpy().tolist()
            y_true.extend(y)
            y_prob.extend(prob)
    else:
        import pandas as pd
        from PIL import Image
        df = pd.read_csv(cfg['data']['test_csv'])
        y_true, y_prob = [], []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img = Image.open(row['path']).convert('RGB')
            views = five_crop_tensor_views(img, cfg)
            x = torch.stack(views, dim=0).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                prob = torch.sigmoid(logits).mean().item()
            y_true.append(int(row['label']))
            y_prob.append(prob)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    metrics = binary_metrics(y_true, y_prob)
    for p in [0.95, 0.98, 0.99]:
        rec = recall_at_precision(y_true, y_prob, p)
        thr = threshold_for_precision(y_true, y_prob, p)
        metrics[f'recall@p{int(p*100)}'] = -1.0 if rec is None else rec
        metrics[f'thr@p{int(p*100)}'] = None if thr is None else thr
    print(metrics)


if __name__ == '__main__':
    main()
