import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, Sampler


class DistributedWeightedSampler(Sampler[int]):
    """DDP-safe weighted sampler with replacement by default."""

    def __init__(
        self,
        weights: Sequence[float],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        replacement: bool = True,
        num_samples: Optional[int] = None,
        seed: int = 0,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank >= num_replicas or rank < 0:
            raise ValueError(f'Invalid rank {rank} for num_replicas={num_replicas}')

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if torch.any(self.weights <= 0):
            raise ValueError('All sampling weights must be > 0')

        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0

        if num_samples is None:
            num_samples = int(math.ceil(len(self.weights) / self.num_replicas))
        self.num_samples = int(num_samples)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        replacement = self.replacement or (self.total_size > len(self.weights))
        indices = torch.multinomial(self.weights, self.total_size, replacement=replacement, generator=g).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class SamplerSummary:
    def __init__(self, mode: str, details: Optional[Dict] = None):
        self.mode = mode
        self.details = details or {}

    def to_dict(self) -> Dict:
        return {'mode': self.mode, **self.details}


def _infer_source_from_path(path: str) -> str:
    parent = Path(path).parent.name.strip()
    return parent if parent else 'unknown'


def _series_for_col(df: pd.DataFrame, col: str, source_col: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna('unknown').astype(str)
    if col == source_col and 'path' in df.columns:
        return df['path'].astype(str).map(_infer_source_from_path)
    if col == 'label' and 'label' in df.columns:
        return df['label'].astype(str)
    return pd.Series(['unknown'] * len(df), index=df.index)


def build_group_keys(df: pd.DataFrame, group_by: Sequence[str], source_col: str = 'source') -> pd.Series:
    if not group_by:
        return pd.Series(['all'] * len(df), index=df.index)
    parts = [_series_for_col(df, col, source_col) for col in group_by]
    keys = parts[0]
    for part in parts[1:]:
        keys = keys + '|' + part
    return keys


def compute_sample_weights(
    df: pd.DataFrame,
    group_by: Sequence[str],
    alpha: float = 1.0,
    source_col: str = 'source',
    weight_col: Optional[str] = None,
    hard_negative_col: Optional[str] = None,
    hard_negative_boost: float = 1.0,
) -> Tuple[np.ndarray, SamplerSummary]:
    weights = np.ones(len(df), dtype=np.float64)
    group_keys = build_group_keys(df, group_by=group_by, source_col=source_col)
    counts = group_keys.value_counts(dropna=False)
    inv = group_keys.map(lambda k: 1.0 / (float(counts[k]) ** alpha)).to_numpy(dtype=np.float64)
    weights *= inv

    if weight_col and weight_col in df.columns:
        raw = pd.to_numeric(df[weight_col], errors='coerce').fillna(1.0).to_numpy(dtype=np.float64)
        raw = np.clip(raw, a_min=1e-6, a_max=None)
        weights *= raw

    if hard_negative_col and hard_negative_col in df.columns and hard_negative_boost != 1.0:
        hard_mask = pd.to_numeric(df[hard_negative_col], errors='coerce').fillna(0).to_numpy(dtype=np.float64) > 0
        weights[hard_mask] *= float(hard_negative_boost)

    weights = np.clip(weights, a_min=1e-12, a_max=None)

    source_preview = None
    if source_col in df.columns or 'path' in df.columns:
        source_preview = _series_for_col(df, source_col, source_col).value_counts().head(10).to_dict()

    summary = SamplerSummary(
        mode='source_balanced',
        details={
            'group_by': list(group_by),
            'alpha': float(alpha),
            'num_groups': int(counts.shape[0]),
            'top_groups': {str(k): int(v) for k, v in counts.head(10).items()},
            'top_sources': None if source_preview is None else {str(k): int(v) for k, v in source_preview.items()},
            'hard_negative_boost': float(hard_negative_boost),
            'weight_min': float(weights.min()) if len(weights) else 0.0,
            'weight_max': float(weights.max()) if len(weights) else 0.0,
        },
    )
    return weights, summary


def build_train_sampler(cfg: Dict, train_dataset, distributed: bool, rank: int, world_size: int):
    scfg = cfg.get('data', {}).get('sampler', {}) or {}
    sampler_name = str(scfg.get('name', 'distributed')).lower()

    if sampler_name in {'source_balanced', 'weighted_source', 'group_balanced'}:
        group_by = scfg.get('group_by', ['source', 'label'])
        if isinstance(group_by, str):
            group_by = [group_by]
        weights, summary = compute_sample_weights(
            train_dataset.df,
            group_by=group_by,
            alpha=float(scfg.get('alpha', 1.0)),
            source_col=str(scfg.get('source_col', 'source')),
            weight_col=scfg.get('weight_col', 'sample_weight'),
            hard_negative_col=scfg.get('hard_negative_col', 'is_hard_negative'),
            hard_negative_boost=float(scfg.get('hard_negative_boost', 1.0)),
        )
        replacement = bool(scfg.get('replacement', True))
        samples_per_epoch = scfg.get('samples_per_epoch', None)
        if samples_per_epoch is None:
            samples_per_epoch = len(train_dataset)
        per_replica = int(math.ceil(int(samples_per_epoch) / max(1, world_size)))
        sampler = DistributedWeightedSampler(
            weights=weights,
            num_replicas=world_size if distributed else 1,
            rank=rank if distributed else 0,
            replacement=replacement,
            num_samples=per_replica,
            seed=int(cfg.get('seed', 0)),
        )
        return sampler, summary.to_dict()

    if distributed:
        return DistributedSampler(train_dataset, shuffle=True), SamplerSummary(mode='distributed').to_dict()
    return None, SamplerSummary(mode='random').to_dict()
