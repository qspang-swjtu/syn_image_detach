import argparse
import json
from typing import Optional

import torch
from PIL import Image

from data.transforms import build_val_transform, five_crop_tensor_views
from models.safepp import build_model
from utils.common import load_yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Run single-image inference with SAFE++.')
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config.')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint (.pt).')
    parser.add_argument('--image', type=str, required=True, help='Path to input image.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold for fake class.')
    parser.add_argument('--tta', type=int, default=None, choices=[1, 5], help='Override config eval.tta with 1 or 5.')
    parser.add_argument('--device', type=str, default=None, help='Device override, e.g. cpu / cuda:0.')
    parser.add_argument('--json', action='store_true', help='Print result as JSON.')
    return parser.parse_args()


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(cfg, ckpt_path: str, device: torch.device):
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['ema'] if 'ema' in ckpt else ckpt['model']
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_single_image(model, cfg, image_path: str, device: torch.device, tta: int) -> float:
    img = Image.open(image_path).convert('RGB')
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and device.type == 'cuda'):
        if tta == 5:
            views = five_crop_tensor_views(img, cfg)
            x = torch.stack(views, dim=0).to(device)
            prob = torch.sigmoid(model(x)).mean().item()
        else:
            transform = build_val_transform(cfg)
            x = transform(img).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(x)).item()
    return float(prob)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    device = resolve_device(args.device)
    tta = args.tta if args.tta is not None else cfg.get('eval', {}).get('tta', 1)
    if tta not in (1, 5):
        raise ValueError(f'Unsupported tta={tta}, expected 1 or 5.')

    model = load_model(cfg, args.ckpt, device)
    prob = predict_single_image(model, cfg, args.image, device, tta)
    pred = 1 if prob >= args.threshold else 0

    result = {
        'image': args.image,
        'probability': prob,
        'threshold': args.threshold,
        'prediction': pred,
        'prediction_name': 'fake' if pred == 1 else 'real',
        'tta': tta,
        'device': str(device),
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"image: {result['image']}")
        print(f"probability_fake: {result['probability']:.6f}")
        print(f"threshold: {result['threshold']:.6f}")
        print(f"prediction: {result['prediction_name']} ({result['prediction']})")
        print(f"tta: {result['tta']}")
        print(f"device: {result['device']}")


if __name__ == '__main__':
    main()
