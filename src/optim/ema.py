"""Exponential moving average for model parameters and buffers."""

from copy import deepcopy
import math

import torch
import torch.nn as nn

from src.core import register
from src.misc import distributed


__all__ = ["ModelEMA"]


@register
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, warmups: int = 2000):
        self.module = deepcopy(model).eval()
        self.decay = float(decay)
        self.warmups = int(warmups)
        self.updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.decay * (1.0 - math.exp(-self.updates / self.warmups))
        model_state = distributed.de_parallel(model).state_dict()
        for name, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                # Preserve the original operation sequence for numerical parity.
                value *= decay
                value += (1.0 - decay) * model_state[name].detach()

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        return self

    def state_dict(self) -> dict:
        return {
            "module": self.module.state_dict(),
            "updates": self.updates,
            "warmups": self.warmups,
        }

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state["module"], strict=True)
        self.updates = int(state.get("updates", 0))
        self.warmups = int(state.get("warmups", self.warmups))
