# RegionBGC adaptation for BGC region prediction; upstream attribution is retained.
"""by lyuwenyu
"""

import torch.nn as nn

from src.core import register
from .interval_ops import INTERVAL_REPRESENTATION


__all__ = ['RegionBGC', ]


@register
class RegionBGC(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]
    interval_representation = INTERVAL_REPRESENTATION

    def __init__(self, backbone: nn.Module, encoder, decoder):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
