import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from data.dataset import CSVDataset
from data.samplers import build_train_sampler
from data.transforms import build_train_transform, build_val_transform
from models.safepp import build_model
from utils.common import load_yaml, set_seed, ensure_dir, AverageMeter, CosineWithWarmup, ModelEmaV2, save_yaml
from utils.metrics import binary_metrics, recall_at_precision, threshold_for_precision


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default='')
    return parser.parse_args()


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def gather_predictions(y_true_local, y_prob_local, world_size: int):
    if world_size == 1:
        return np.array(y_true_local), np.array(y_prob_local)
    y_true_list = [None for _ in range(world_size)]
    y_prob_list = [None for _ in range(world_size)]
    dist.all_gather_object(y_true_list, list(y_true_local))
    dist.all_gather_object(y_prob_list, list(y_prob_local))
    y_true = np.array([x for sub in y_true_list for x in sub])
    y_prob = np.array([x for sub in y_prob_list for x in sub])
    return y_true, y_prob


def smooth_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return targets
    return targets * (1.0 - smoothing) + 0.5 * smoothing


@torch.no_grad()
def evaluate(model, loader, device, world_size: int, amp: bool = True):
    model.eval()
    y_true_local = []
    y_prob_local = []
    for batch in tqdm(loader, disable=(not (not dist.is_initialized() or dist.get_rank() == 0)), desc='val'):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
        probs = torch.sigmoid(logits)
        y_true_local.extend(labels.detach().cpu().numpy().tolist())
        y_prob_local.extend(probs.detach().cpu().numpy().tolist())
    y_true, y_prob = gather_predictions(y_true_local, y_prob_local, world_size)
    metrics = binary_metrics(y_true.astype(np.int64), y_prob)
    for p in [0.95, 0.98, 0.99]:
        rec = recall_at_precision(y_true.astype(np.int64), y_prob, p)
        thr = threshold_for_precision(y_true.astype(np.int64), y_prob, p)
        metrics[f'recall@p{int(p*100)}'] = -1.0 if rec is None else rec
        metrics[f'thr@p{int(p*100)}'] = None if thr is None else thr
    return metrics


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    distributed, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg['seed'] + rank)

    if is_main_process(rank):
        ensure_dir(cfg['output_dir'])
        save_yaml(cfg, os.path.join(cfg['output_dir'], 'resolved_config.yaml'))

    train_ds = CSVDataset(cfg['data']['train_csv'], transform=build_train_transform(cfg))
    val_ds = CSVDataset(cfg['data']['val_csv'], transform=build_val_transform(cfg))

    train_sampler, sampler_summary = build_train_sampler(cfg, train_ds, distributed=distributed, rank=rank, world_size=world_size)
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    if is_main_process(rank):
        print('train_sampler:', sampler_summary)
        save_yaml({'train_sampler': sampler_summary}, os.path.join(cfg['output_dir'], 'sampler_summary.yaml'))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['train']['batch_size_per_gpu'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg['data']['num_workers'],
        pin_memory=cfg['data']['pin_memory'],
        persistent_workers=cfg['data']['persistent_workers'],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['eval']['batch_size_per_gpu'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg['data']['num_workers'],
        pin_memory=cfg['data']['pin_memory'],
        persistent_workers=cfg['data']['persistent_workers'],
        drop_last=False,
    )

    model = build_model(cfg).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['optim']['lr'],
        weight_decay=cfg['optim']['weight_decay'],
        betas=tuple(cfg['optim']['betas']),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['train']['amp'])
    total_steps = len(train_loader) * cfg['train']['epochs'] // cfg['train']['grad_accum_steps']
    warmup_steps = len(train_loader) * cfg['optim']['warmup_epochs'] // cfg['train']['grad_accum_steps']
    scheduler = CosineWithWarmup(optimizer, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=cfg['optim']['min_lr'])

    raw_model = model.module if distributed else model
    ema = ModelEmaV2(raw_model, decay=cfg['train']['ema_decay']).to(device)

    start_epoch = 0
    best_ap = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        raw_model.load_state_dict(ckpt['model'], strict=True)
        if 'ema' in ckpt:
            ema.module.load_state_dict(ckpt['ema'], strict=True)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt and scaler.is_enabled() and ckpt['scaler'] is not None:
            scaler.load_state_dict(ckpt['scaler'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_ap = ckpt.get('best_ap', -1.0)

    pos_weight = torch.tensor([cfg['loss']['pos_weight']], dtype=torch.float32, device=device)

    for epoch in range(start_epoch, cfg['train']['epochs']):
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        model.train()
        loss_meter = AverageMeter()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=(not is_main_process(rank)), desc=f'train {epoch}')
        for step, batch in pbar:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            labels = smooth_targets(labels, cfg['train']['label_smoothing'])

            with torch.cuda.amp.autocast(enabled=cfg['train']['amp']):
                logits = model(images)
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
                loss = loss / cfg['train']['grad_accum_steps']

            scaler.scale(loss).backward()

            if (step + 1) % cfg['train']['grad_accum_steps'] == 0:
                if cfg['train']['clip_grad_norm'] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['clip_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(raw_model)

            reduced_loss = reduce_mean(loss.detach(), world_size)
            loss_meter.update(reduced_loss.item() * cfg['train']['grad_accum_steps'], images.size(0))
            if is_main_process(rank):
                pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'lr': optimizer.param_groups[0]['lr']})

        metrics = evaluate(ema.module, val_loader, device, world_size=world_size, amp=cfg['train']['amp'])

        if is_main_process(rank):
            ckpt = {
                'epoch': epoch,
                'model': raw_model.state_dict(),
                'ema': ema.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler.is_enabled() else None,
                'scheduler': scheduler.state_dict(),
                'best_ap': best_ap,
                'metrics': metrics,
            }
            last_path = os.path.join(cfg['output_dir'], 'last.pt')
            torch.save(ckpt, last_path)
            if metrics['ap'] > best_ap:
                best_ap = metrics['ap']
                ckpt['best_ap'] = best_ap
                torch.save(ckpt, os.path.join(cfg['output_dir'], 'best.pt'))
            print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()})

    cleanup_distributed()


if __name__ == '__main__':
    main()
