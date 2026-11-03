"""Multilabel BGC product-classification metrics."""

from typing import Dict

import torch

from .gene import _auroc_sklearn_safe, _validate_metric_inputs


def compute_multilabel_micro_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Micro-averaged product-classification metrics from aligned logits and targets."""
    with torch.no_grad():
        _validate_metric_inputs(logits, targets)
        if logits.numel() == 0 or targets.numel() == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auroc": 0.5}

        probs = logits.sigmoid().detach().float().view(-1)
        labels = targets.detach().float().view(-1)

        preds = (probs >= threshold).to(torch.int64)
        y = (labels >= 0.5).to(torch.int64)

        tp = ((preds == 1) & (y == 1)).sum().item()
        fp = ((preds == 1) & (y == 0)).sum().item()
        fn = ((preds == 0) & (y == 1)).sum().item()

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0)

        probs_np = probs.detach().cpu().numpy().astype("float64")
        y_np = y.detach().cpu().numpy().astype("int64")
        auroc = _auroc_sklearn_safe(y_np, probs_np)

        return {"precision": precision, "recall": recall, "f1": f1, "auroc": auroc}



__all__ = ["compute_multilabel_micro_metrics"]
