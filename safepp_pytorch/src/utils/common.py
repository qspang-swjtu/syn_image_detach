import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(1, self.count)

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n


class CosineWithWarmup:
    def __init__(self, optimizer, total_steps: int, warmup_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(0, warmup_steps)
        self.min_lr = min_lr
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps and self.warmup_steps > 0:
            scale = self.step_num / self.warmup_steps
        else:
            progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for lr0, group in zip(self.base_lrs, self.optimizer.param_groups):
            group['lr'] = self.min_lr + (lr0 - self.min_lr) * scale

    def state_dict(self):
        return {'step_num': self.step_num}

    def load_state_dict(self, state):
        self.step_num = state.get('step_num', 0)


class ModelEmaV2:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.module = self._clone_model(model)
        self.decay = decay

    @staticmethod
    def _clone_model(model: torch.nn.Module) -> torch.nn.Module:
        import copy
        ema = copy.deepcopy(model)
        ema.eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for k, v in ema_state.items():
            if not v.dtype.is_floating_point:
                v.copy_(model_state[k])
            else:
                v.mul_(self.decay).add_(model_state[k], alpha=1.0 - self.decay)

    def to(self, device):
        self.module.to(device)
        return self
