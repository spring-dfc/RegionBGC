"""Gene-level BGC localization metrics."""

from typing import Dict

import numpy as np
import torch

try:
    from sklearn.metrics import roc_auc_score  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError('BGC metrics require scikit-learn; install requirements.txt') from exc


def _auroc_sklearn_safe(y_true_01: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC via sklearn.metrics.roc_auc_score with safe fallbacks.

    - If sklearn is unavailable, raises ImportError.
    - If AUROC is undefined (single-class y_true), return 0.5 (matches previous behavior).
    """
    if roc_auc_score is None:
        raise ImportError("scikit-learn is required for AUROC (roc_auc_score) but is not installed.")
    y = np.asarray(y_true_01, dtype=np.int64).reshape(-1)
    s = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y.size != s.size:
        raise ValueError('AUROC labels and scores must have matching lengths')
    if not np.isfinite(s).all():
        raise ValueError('AUROC scores must be finite')
    if y.size == 0:
        return 0.5
    # Undefined when only one class exists in y_true
    if int((y == 1).sum()) == 0 or int((y == 0).sum()) == 0:
        return 0.5
    return float(roc_auc_score(y, s))


def compute_gene_metrics(
    bgc_probs: torch.Tensor,
    bgc_labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Gene-level localization Precision, Recall, F1 and AUROC."""
    probs = bgc_probs.detach().float().view(-1)
    labels = bgc_labels.detach().float().view(-1)
    _validate_metric_inputs(probs, labels)

    preds = (probs >= threshold).to(torch.int64)
    y = (labels > 0.5).to(torch.int64)

    tp = int(((preds == 1) & (y == 1)).sum().item())
    fp = int(((preds == 1) & (y == 0)).sum().item())
    fn = int(((preds == 0) & (y == 1)).sum().item())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    probs_np = probs.detach().cpu().numpy().astype("float64")
    y_np = y.detach().cpu().numpy().astype("int64")
    auroc = _auroc_sklearn_safe(y_np, probs_np)
    return {"precision": float(precision), "recall": float(recall),
            "f1": float(f1), "auroc": float(auroc)}


def _validate_metric_inputs(probs: torch.Tensor, labels: torch.Tensor) -> None:
    if probs.shape != labels.shape:
        raise ValueError('Metric predictions and labels must have matching shapes')
    if not bool(torch.isfinite(probs).all()) or not bool(torch.isfinite(labels).all()):
        raise ValueError('Metric predictions and labels must be finite')


__all__ = ["compute_gene_metrics"]
