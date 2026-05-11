#!/usr/bin/env python3
"""Create a weights-only checkpoint for stage-to-stage fine-tuning.

The existing src/train.py --resume restores epoch/optimizer/scheduler. That is good for
crash recovery but not for moving from stage1 to stage2. This helper creates a compact
checkpoint with epoch=-1 and no optimizer/scheduler, so train.py starts the next stage
from epoch 0 while loading the previous weights.
"""

import argparse
from pathlib import Path
from typing import Dict

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Create a weights-only warm-start checkpoint.')
    parser.add_argument('--input', required=True, help='Source checkpoint, usually best.pt')
    parser.add_argument('--output', required=True, help='Output warm-start checkpoint')
    parser.add_argument('--prefer', default='ema', choices=['ema', 'model'], help='Which state dict to prefer when both exist')
    return parser.parse_args()


def pick_state(ckpt: Dict, prefer: str):
    if prefer == 'ema' and 'ema' in ckpt:
        return ckpt['ema'], 'ema'
    if 'model' in ckpt:
        return ckpt['model'], 'model'
    if 'ema' in ckpt:
        return ckpt['ema'], 'ema'
    raise KeyError('checkpoint has neither `model` nor `ema` state dict')


def main():
    args = parse_args()
    ckpt = torch.load(args.input, map_location='cpu')
    state, state_key = pick_state(ckpt, args.prefer)
    out = {
        'epoch': -1,
        'best_ap': -1.0,
        'model': state,
        'ema': state,
        'warmstart_from': str(args.input),
        'warmstart_state_key': state_key,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    print(f'[OK] wrote warm-start checkpoint: {output} using `{state_key}`')


if __name__ == '__main__':
    main()
