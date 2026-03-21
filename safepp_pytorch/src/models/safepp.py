from typing import Dict, Any, Tuple

import timm
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward


class DwtHH(nn.Module):
    def __init__(self, wave: str = 'bior1.3', mode: str = 'symmetric'):
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wave, mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        yl, yh = self.dwt(x)
        # yh[0]: [B, C, 3, H/2, W/2] -> LH, HL, HH
        hh = yh[0][:, :, 2, :, :]
        return hh


class TimmBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool, in_chans: int = 3, drop_rate: float = 0.0):
        super().__init__()
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg',
            in_chans=in_chans,
            drop_rate=drop_rate,
        )
        self.out_dim = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SafePPDual(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        mcfg = cfg['model']
        self.dwt = DwtHH(wave=mcfg['wave'], mode=mcfg['mode'])
        self.rgb = TimmBackbone(
            mcfg['rgb_backbone'],
            pretrained=mcfg.get('pretrained_rgb', True),
            in_chans=3,
            drop_rate=mcfg.get('rgb_drop', 0.0),
        )
        self.forensic = TimmBackbone(
            mcfg['forensic_backbone'],
            pretrained=mcfg.get('pretrained_forensic', False),
            in_chans=3,
            drop_rate=mcfg.get('forensic_drop', 0.0),
        )
        fusion_in = self.rgb.out_dim + self.forensic.out_dim
        hidden = mcfg.get('fusion_dim', 384)
        self.head = nn.Sequential(
            nn.Linear(fusion_in, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.rgb(x)
        hh = self.dwt(x)
        forensic_feat = self.forensic(hh)
        feat = torch.cat([rgb_feat, forensic_feat], dim=1)
        logit = self.head(feat).squeeze(1)
        return logit


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    name = cfg['model']['name']
    if name == 'safepp_dual':
        return SafePPDual(cfg)
    raise ValueError(f'Unknown model: {name}')
