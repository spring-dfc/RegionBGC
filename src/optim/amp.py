import torch.cuda.amp as amp

from src.core import register


__all__ = ['GradScaler']

GradScaler = register(amp.grad_scaler.GradScaler)
