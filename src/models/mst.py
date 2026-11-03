"""MST pooling; pointwise projections are applied in the region feature encoder."""

import torch
import torch.nn as nn

from src.core import register


__all__ = ["MultiScalePooling"]


@register
class MultiScalePooling(nn.Module):
    """Average-pooling stage of Multi-Scale Transformation (MST)."""
    __share__ = []

    def __init__(
        self,
        in_ch: int = 960,
        output_strides=None,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        if output_strides is None:
            output_strides = [1, 2, 4]
        self.output_strides = [int(s) for s in output_strides]
        if len(self.output_strides) == 0:
            raise ValueError("output_strides must contain at least one level")
        if self.output_strides[0] != 1:
            raise ValueError("output_strides must start from 1 for the finest input level")
        for prev, curr in zip(self.output_strides[:-1], self.output_strides[1:]):
            if curr != prev * 2:
                raise ValueError(
                    f"Only dyadic pyramid strides are supported, but got {self.output_strides}"
                )
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < 2:
            return x
        return self.pool(x)

    def forward(self, x: torch.Tensor):
        if x.ndim != 3 or x.shape[1] != self.in_ch or x.shape[-1] < 1:
            raise ValueError(f"BGC input must have shape [B,{self.in_ch},L], L > 0")
        feats = [x.contiguous()]
        while len(feats) < len(self.output_strides):
            feats.append(self._downsample(feats[-1]))
        return feats
