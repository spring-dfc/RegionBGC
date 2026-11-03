"""Strict loading of native interval training and fused deployment weights."""
import torch

from .deploy import DEPLOY_FORMATS, compact_structure
from .interval_ops import validate_interval_checkpoint


def load_evaluation_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") in DEPLOY_FORMATS:
        compact_structure(model)
        state = checkpoint["model"]
        source = "deploy:" + checkpoint.get("metadata", {}).get("weights_source", "model")
    else:
        if "format" in checkpoint:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint['format']}")
        if "ema" in checkpoint:
            state = checkpoint["ema"].get("module", checkpoint["ema"])
            source = "ema"
        else:
            state = checkpoint.get("model", checkpoint)
            source = "model"
    validate_interval_checkpoint(checkpoint, state)
    model.load_state_dict(state, strict=True)
    return checkpoint.get("last_epoch", checkpoint.get("epoch")), source
