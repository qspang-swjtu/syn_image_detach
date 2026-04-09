from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass
class TierRule:
    min_rows: int
    cosine_threshold: float
    topk: int


class ImageDataset:
    def __init__(self, df: pd.DataFrame, transform, fallback_size: int) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.fallback_size = int(fallback_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row['path']
        try:
            img = Image.open(path).convert('RGB')
            w, h = img.size
        except Exception:
            img = Image.new('RGB', (self.fallback_size, self.fallback_size), color=(0, 0, 0))
            w, h = self.fallback_size, self.fallback_size
        return self.transform(img), idx, w, h


def collate_fn(batch):
    import torch
    imgs = torch.stack([b[0] for b in batch], dim=0)
    idxs = [b[1] for b in batch]
    sizes = [(b[2], b[3]) for b in batch]
    return imgs, idxs, sizes


def load_encoder(model_name: str, device: str, input_size: int = 224):
    import torch
    try:
        import timm
        from timm.data import create_transform, resolve_model_data_config
    except Exception as e:
        raise RuntimeError(
            'timm is required for stage-2 embedding extraction. Install timm and torchvision on the training machine.'
        ) from e

    input_size = int(input_size) if input_size else 224
    model = None
    create_errors = []
    for kwargs in (
        {'pretrained': True, 'num_classes': 0, 'img_size': input_size},
        {'pretrained': True, 'num_classes': 0, 'img_size': input_size, 'dynamic_img_size': True},
        {'pretrained': True, 'num_classes': 0},
    ):
        try:
            model = timm.create_model(model_name, **kwargs)
            break
        except TypeError as e:
            create_errors.append(f'{kwargs}: {e}')
            continue
    if model is None:
        raise RuntimeError('Failed to create timm model with the requested input size. Errors: ' + ' | '.join(create_errors))

    model.eval().to(device)
    data_config = resolve_model_data_config(model)
    # Force the transform to match the real model input size after any img_size override.
    transform = create_transform(**data_config, is_training=False)
    input_size_final = data_config.get('input_size', (3, input_size, input_size))[-1]
    return model, transform, int(input_size_final)


def extract_embeddings(df: pd.DataFrame, model_name: str, batch_size: int, num_workers: int, device: str, input_size: int = 224) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    model, transform, fallback_size = load_encoder(model_name, device=device, input_size=input_size)
    ds = ImageDataset(df, transform=transform, fallback_size=fallback_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != 'cpu'),
        collate_fn=collate_fn,
        drop_last=False,
    )

    feats: List[np.ndarray] = []
    ws = np.zeros(len(df), dtype=np.int32)
    hs = np.zeros(len(df), dtype=np.int32)

    with torch.no_grad():
        for images, idxs, sizes in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            if outputs.ndim > 2:
                outputs = outputs.mean(dim=tuple(range(2, outputs.ndim)))
            outputs = torch.nn.functional.normalize(outputs.float(), dim=1)
            feats.append(outputs.cpu().numpy().astype(np.float32))
            for local_idx, (w, h) in zip(idxs, sizes):
                ws[local_idx] = int(w)
                hs[local_idx] = int(h)

    emb = np.concatenate(feats, axis=0)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-12, None)
    return emb.astype(np.float32), ws, hs


def knn_search(emb: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    n, d = emb.shape
    topk = int(max(2, min(topk, n)))
    try:
        import faiss  # type: ignore
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = max(64, topk * 2)
        index.add(emb)
        sims, nbrs = index.search(emb, topk)
        return sims, nbrs
    except Exception:
        if n > 150000:
            raise RuntimeError(
                'faiss is not available and the focus source is too large for brute-force sklearn fallback. Install faiss-cpu or faiss-gpu.'
            )
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=topk, metric='cosine', algorithm='brute', n_jobs=-1)
        nn.fit(emb)
        dist, nbrs = nn.kneighbors(emb, return_distance=True)
        sims = 1.0 - dist
        return sims.astype(np.float32), nbrs.astype(np.int64)


def choose_tier(num_rows: int, tier_rules: Sequence[TierRule]) -> TierRule:
    ordered = sorted(tier_rules, key=lambda x: x.min_rows, reverse=True)
    for rule in ordered:
        if num_rows >= rule.min_rows:
            return rule
    return ordered[-1]


def keep_budget(size: int) -> int:
    if size <= 1:
        return 1
    if size <= 4:
        return 1
    if size <= 16:
        return 2
    return 3


def stable_choice_order(df: pd.DataFrame, width_col: str, height_col: str) -> pd.Index:
    work = df.copy()
    if width_col not in work.columns:
        work[width_col] = 0
    if height_col not in work.columns:
        work[height_col] = 0
    work['__area'] = work[width_col].fillna(0).astype(int) * work[height_col].fillna(0).astype(int)
    work['__sample_weight'] = work.get('sample_weight', 1.0)
    return work.sort_values(['__area', '__sample_weight', 'path'], ascending=[False, False, True]).index


def derive_focus_sources(df: pd.DataFrame, report_yaml: Optional[str], cfg: Dict) -> List[str]:
    sem = cfg.get('semantic_dedup', {})
    explicit = [str(x) for x in sem.get('explicit_focus_sources', []) if str(x).strip()]
    explicit_set = set(explicit)
    if report_yaml is None:
        return explicit

    report = load_yaml(report_yaml)
    raw_by_source = (report.get('summary', {}) or {}).get('raw_by_source', {}) or {}
    clean_by_source = (report.get('summary', {}) or {}).get('clean_by_source', {}) or {}
    auto = sem.get('auto_focus', {}) or {}
    if not auto.get('enabled', True):
        return explicit

    min_clean_rows = int(auto.get('min_clean_rows', 100000))
    max_removed_ratio = float(auto.get('max_stage1_removed_ratio', 0.05))

    derived: List[str] = []
    for source, clean_rows in clean_by_source.items():
        clean_rows = int(clean_rows)
        raw_rows = int(raw_by_source.get(source, clean_rows))
        removed_ratio = 0.0 if raw_rows <= 0 else (raw_rows - clean_rows) / max(raw_rows, 1)
        if clean_rows >= min_clean_rows and removed_ratio <= max_removed_ratio:
            derived.append(str(source))

    merged = list(dict.fromkeys(explicit + derived))
    # only keep sources still present after any train/test exclusions already encoded in the CSV
    present = set(df['source'].astype(str).unique().tolist())
    return [s for s in merged if s in present]


def normalize_text_series(df: pd.DataFrame, col: str, default: str = 'unknown') -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=object)
    return df[col].fillna(default).astype(str)


def get_base_component_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    return 'path'


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Stage-2 semantic dedup for large AIGC detection training corpora.')
    p.add_argument('--clean_index_csv', required=True, type=str)
    p.add_argument('--out_dir', required=True, type=str)
    p.add_argument('--config_yaml', required=False, type=str, default='')
    p.add_argument('--report_yaml', required=False, type=str, default='')
    p.add_argument('--model_name', required=False, type=str, default='vit_small_patch14_dinov2.lvd142m')
    p.add_argument('--batch_size', required=False, type=int, default=256)
    p.add_argument('--num_workers', required=False, type=int, default=8)
    p.add_argument('--device', required=False, type=str, default='cuda')
    p.add_argument('--input_size', required=False, type=int, default=224, help='Target square input size used when creating the timm model and preprocessing images.')
    p.add_argument('--focus_sources', required=False, type=str, default='')
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(args.config_yaml) if args.config_yaml else {}
    df = pd.read_csv(args.clean_index_csv)
    if len(df) == 0:
        raise RuntimeError('clean_index_csv is empty.')

    if 'split_hint' not in df.columns:
        df['split_hint'] = 'seen'
    df['split_hint'] = df['split_hint'].map(normalize_split_hint)
    df['source'] = normalize_text_series(df, 'source')
    df['dataset'] = normalize_text_series(df, 'dataset')
    df['domain'] = normalize_text_series(df, 'domain')
    df['generator'] = normalize_text_series(df, 'generator')

    sem = cfg.get('semantic_dedup', {}) or {}
    seen_only = bool(sem.get('seen_only', True))
    skip_datasets = set(str(x) for x in sem.get('skip_datasets', []))
    skip_sources = set(str(x) for x in sem.get('skip_sources', []))
    skip_split_hints = set(str(x).lower() for x in sem.get('skip_split_hints', ['unseen', 'test_unseen']))
    base_candidates = sem.get('base_component_col_candidates', ['tight_split_component_id', 'split_component_id', 'dedup_component_id', 'group_id', 'path'])
    base_col = get_base_component_col(df, base_candidates)
    df['base_split_component_id'] = normalize_text_series(df, base_col)

    pool_mask = pd.Series([True] * len(df), index=df.index)
    if seen_only:
        pool_mask &= ~df['split_hint'].isin(skip_split_hints)
    if skip_datasets:
        pool_mask &= ~df['dataset'].isin(skip_datasets)
    if skip_sources:
        pool_mask &= ~df['source'].isin(skip_sources)

    pool_df = df[pool_mask].copy().reset_index(drop=False).rename(columns={'index': 'orig_row_id'})

    focus_sources_arg = [x.strip() for x in args.focus_sources.split(',') if x.strip()]
    if focus_sources_arg:
        focus_sources = focus_sources_arg
    else:
        focus_sources = derive_focus_sources(pool_df, args.report_yaml or None, cfg)
    if not focus_sources:
        raise RuntimeError('No focus sources were selected for stage-2 semantic dedup. Provide --focus_sources or configure explicit_focus_sources/auto_focus.')

    tier_rules_cfg = sem.get('tier_rules', {}) or {
        'huge': {'min_rows': 300000, 'cosine_threshold': 0.9920, 'topk': 64},
        'large': {'min_rows': 100000, 'cosine_threshold': 0.9935, 'topk': 48},
        'medium': {'min_rows': 50000, 'cosine_threshold': 0.9950, 'topk': 32},
    }
    tier_rules = [TierRule(int(v['min_rows']), float(v['cosine_threshold']), int(v['topk'])) for v in tier_rules_cfg.values()]

    all_edge_rows: List[Dict] = []
    semantic_component_labels = np.array([''] * len(df), dtype=object)
    semantic_component_sizes = np.ones(len(df), dtype=np.int32)
    drop_stage2 = np.zeros(len(df), dtype=np.int8)

    unique_base_ids = pd.Index(df['base_split_component_id'].astype(str).unique())
    base_id_to_pos = {k: i for i, k in enumerate(unique_base_ids.tolist())}
    base_uf = UnionFind(len(unique_base_ids))

    source_reports: List[Dict] = []

    for source in focus_sources:
        src_df = pool_df[pool_df['source'] == source].copy().reset_index(drop=True)
        n_src = len(src_df)
        if n_src <= 1:
            continue
        rule = choose_tier(n_src, tier_rules)
        emb_path = out_dir / f'embeddings_{source}.npy'
        meta_path = out_dir / f'embeddings_{source}_sizes.npz'
        if emb_path.exists() and meta_path.exists():
            emb = np.load(emb_path)
            meta = np.load(meta_path)
            widths = meta['widths']
            heights = meta['heights']
        else:
            emb_input = src_df[['path']].copy()
            emb_input['path'] = src_df['path'].astype(str)
            if 'sample_weight' in src_df.columns:
                emb_input['sample_weight'] = src_df['sample_weight']
            emb, widths, heights = extract_embeddings(
                emb_input,
                model_name=args.model_name,
                batch_size=int(sem.get('batch_size', args.batch_size)),
                num_workers=int(sem.get('num_workers', args.num_workers)),
                device=args.device,
            )
            np.save(emb_path, emb)
            np.savez_compressed(meta_path, widths=widths, heights=heights)

        src_df['width_stage2'] = widths
        src_df['height_stage2'] = heights
        sims, nbrs = knn_search(emb, topk=rule.topk)

        row_uf = UnionFind(n_src)
        edge_count = 0
        for i in range(n_src):
            for rank in range(1, sims.shape[1]):
                j = int(nbrs[i, rank])
                if j < 0 or j == i:
                    continue
                sim = float(sims[i, rank])
                if sim < rule.cosine_threshold:
                    continue
                if j < i:
                    continue
                oi = int(src_df.loc[i, 'orig_row_id'])
                oj = int(src_df.loc[j, 'orig_row_id'])
                row_uf.union(i, j)
                base_uf.union(base_id_to_pos[str(df.loc[oi, 'base_split_component_id'])], base_id_to_pos[str(df.loc[oj, 'base_split_component_id'])])
                all_edge_rows.append({
                    'source': source,
                    'path_a': str(df.loc[oi, 'path']),
                    'path_b': str(df.loc[oj, 'path']),
                    'row_a': oi,
                    'row_b': oj,
                    'cosine_sim': round(sim, 6),
                    'reason': 'semantic_knn',
                })
                edge_count += 1

        # row-level semantic components for keep/drop policy
        src_roots: Dict[int, List[int]] = {}
        for i in range(n_src):
            src_roots.setdefault(row_uf.find(i), []).append(i)

        kept_rows = 0
        dropped_rows = 0
        for comp_idx, members in enumerate(src_roots.values()):
            members_global = [int(src_df.loc[m, 'orig_row_id']) for m in members]
            semantic_id = f'sem:{source}:{comp_idx:07d}'
            member_df = src_df.loc[members].copy()
            semantic_component_labels[members_global] = semantic_id
            semantic_component_sizes[members_global] = len(members)
            budget = keep_budget(len(members))
            order = stable_choice_order(member_df, 'width_stage2', 'height_stage2').tolist()
            keep_global = set(int(src_df.loc[o, 'orig_row_id']) for o in order[:budget])
            kept_rows += len(keep_global)
            dropped_rows += max(0, len(members_global) - len(keep_global))
            for g in members_global:
                if len(members_global) > 1 and g not in keep_global:
                    drop_stage2[g] = 1

        source_reports.append({
            'source': source,
            'rows_in_source': int(n_src),
            'cosine_threshold': float(rule.cosine_threshold),
            'topk': int(rule.topk),
            'semantic_edges': int(edge_count),
            'semantic_removed_rows': int(dropped_rows),
            'semantic_kept_rows': int(kept_rows),
        })

    # non-focus rows still need semantic labels and sizes
    missing_mask = semantic_component_labels == ''
    for row_id in np.where(missing_mask)[0].tolist():
        semantic_component_labels[row_id] = f'sem:singleton:{row_id:09d}'
        semantic_component_sizes[row_id] = 1

    tight_split_ids = np.empty(len(df), dtype=object)
    for row_id in range(len(df)):
        base_pos = base_id_to_pos[str(df.loc[row_id, 'base_split_component_id'])]
        tight_split_ids[row_id] = f'tight:{base_uf.find(base_pos):09d}'

    out_df = df.copy()
    out_df['semantic_component_id'] = semantic_component_labels
    out_df['semantic_component_size'] = semantic_component_sizes
    out_df['tight_split_component_id'] = tight_split_ids
    out_df['drop_stage2_semantic'] = drop_stage2
    out_df['keep_stage2_semantic'] = 1 - drop_stage2

    stage2_clean = out_df[out_df['keep_stage2_semantic'].astype(int) == 1].copy().reset_index(drop=True)

    pd.DataFrame(all_edge_rows).to_csv(out_dir / 'stage2_semantic_pairs.csv', index=False)
    out_df.to_csv(out_dir / 'stage2_full_index.csv', index=False)
    stage2_clean.to_csv(out_dir / 'stage2_clean_index.csv', index=False)

    report = {
        'inputs': {
            'clean_index_csv': str(args.clean_index_csv),
            'report_yaml': str(args.report_yaml or ''),
            'config_yaml': str(args.config_yaml or ''),
            'base_component_col': base_col,
        },
        'settings': {
            'model_name': args.model_name,
            'batch_size': int(sem.get('batch_size', args.batch_size)),
            'num_workers': int(sem.get('num_workers', args.num_workers)),
            'device': args.device,
            'focus_sources': focus_sources,
            'skip_datasets': sorted(list(skip_datasets)),
            'skip_split_hints': sorted(list(skip_split_hints)),
        },
        'summary': {
            'rows_before_stage2': int(len(df)),
            'rows_after_stage2': int(len(stage2_clean)),
            'rows_removed_stage2': int(len(df) - len(stage2_clean)),
            'focus_source_count': int(len(source_reports)),
            'semantic_edge_count': int(len(all_edge_rows)),
        },
        'removed_by_source': {str(k): int(v) for k, v in out_df[out_df['drop_stage2_semantic'].astype(int) == 1]['source'].value_counts().items()},
        'source_reports': source_reports,
    }
    save_yaml(report, str(out_dir / 'stage2_report.yaml'))
    print(f'[DONE] stage2_full_index -> {out_dir / "stage2_full_index.csv"}')
    print(f'[DONE] stage2_clean_index -> {out_dir / "stage2_clean_index.csv"}')
    print(f'[DONE] stage2_semantic_pairs -> {out_dir / "stage2_semantic_pairs.csv"}')
    print(f'[DONE] stage2_report -> {out_dir / "stage2_report.yaml"}')


if __name__ == '__main__':
    main()
