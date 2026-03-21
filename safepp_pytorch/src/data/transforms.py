import io
import random
from typing import Tuple, Dict, Any, Callable

from PIL import Image, ImageFilter
import torchvision.transforms as T
import torchvision.transforms.functional as F


class ReflectPadToMin:
    def __init__(self, min_size: int):
        self.min_size = min_size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        pad_w = max(0, self.min_size - w)
        pad_h = max(0, self.min_size - h)
        if pad_w == 0 and pad_h == 0:
            return img
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        return F.pad(img, [left, top, right, bottom], padding_mode='reflect')


class RandomMask:
    def __init__(self, p: float = 0.5, patch: int = 16, max_ratio: float = 0.75):
        self.p = p
        self.patch = patch
        self.max_ratio = max_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        x = F.pil_to_tensor(img).clone()
        _, h, w = x.shape
        ratio = random.uniform(0.0, self.max_ratio)
        num = int((h * w * ratio) // (self.patch * self.patch))
        occupied = set()
        for _ in range(num):
            gy = random.randint(0, max(0, h // self.patch - 1))
            gx = random.randint(0, max(0, w // self.patch - 1))
            if (gy, gx) in occupied:
                continue
            occupied.add((gy, gx))
            y0 = gy * self.patch
            x0 = gx * self.patch
            y1 = min(h, y0 + self.patch)
            x1 = min(w, x0 + self.patch)
            x[:, y0:y1, x0:x1] = 0
        return F.to_pil_image(x)


class RandomJPEG:
    def __init__(self, p: float = 0.2, quality: Tuple[int, int] = (70, 100)):
        self.p = p
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        q = random.randint(self.quality[0], max(self.quality[0], self.quality[1] - 1))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        out = Image.open(buf).convert('RGB')
        return out


class RandomGaussianBlur:
    def __init__(self, p: float = 0.5, sigma: Tuple[float, float] = (0.1, 2.0)):
        self.p = p
        self.sigma = sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        radius = random.uniform(*self.sigma)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


def build_train_transform(cfg: Dict[str, Any]) -> Callable:
    aug = cfg['augment']['train']
    dcfg = cfg['data']
    cj_alpha = aug['color_jitter_strength']
    return T.Compose([
        ReflectPadToMin(dcfg['image_size']),
        RandomGaussianBlur(p=aug['gaussian_blur_p'], sigma=tuple(aug['gaussian_blur_sigma'])),
        RandomJPEG(p=aug['jpeg_p'], quality=tuple(aug['jpeg_quality'])),
        T.RandomCrop(dcfg['image_size']),
        T.RandomHorizontalFlip(p=aug['hflip_p']),
        T.RandomApply([
            T.ColorJitter(brightness=cj_alpha, contrast=cj_alpha, saturation=cj_alpha, hue=0.0)
        ], p=aug['color_jitter_p']),
        T.RandomApply([
            T.RandomRotation(degrees=(-aug['rotation_deg'], aug['rotation_deg']), fill=0)
        ], p=aug['rotation_p']),
        RandomMask(
            p=aug['random_mask_p'],
            patch=aug['random_mask_patch'],
            max_ratio=aug['random_mask_ratio'],
        ),
        T.ToTensor(),
        T.Normalize(mean=dcfg['mean'], std=dcfg['std']),
    ])


def build_val_transform(cfg: Dict[str, Any]) -> Callable:
    dcfg = cfg['data']
    return T.Compose([
        ReflectPadToMin(dcfg['image_size']),
        T.CenterCrop(dcfg['image_size']),
        T.ToTensor(),
        T.Normalize(mean=dcfg['mean'], std=dcfg['std']),
    ])


def five_crop_tensor_views(img: Image.Image, cfg: Dict[str, Any]):
    dcfg = cfg['data']
    base = T.Compose([
        ReflectPadToMin(dcfg['image_size']),
    ])
    img = base(img)
    crops = T.FiveCrop(dcfg['image_size'])(img)
    norm = T.Compose([T.ToTensor(), T.Normalize(mean=dcfg['mean'], std=dcfg['std'])])
    return [norm(c) for c in crops]
