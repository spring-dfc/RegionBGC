"""Run interval inference on valid ORF sequences rather than padded axes."""

import math
from typing import Dict, List, Optional, Tuple

import torch


def probability_threshold(value):
    """Parse a finite probability threshold in [0, 1]."""
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("threshold must be finite and within [0, 1]")
    return result


@torch.no_grad()
def predict_valid_sequences(
    model,
    postprocessor,
    features: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor]:
    """Infer [B,L,C] features, crop right padding, and retain sample order.

    Each model input and its postprocessing length describe the same ORF axis.
    Equal-length samples share one forward pass. Full-length batches retain
    the original single-forward path, including batch composition and order.
    """
    if features.ndim != 3 or min(features.shape) < 1:
        raise ValueError('ORF features must have shape [B,L,C], with positive dimensions')
    batch_size, padded_length, _ = features.shape
    if padding_mask is None:
        lengths = torch.full((batch_size,), padded_length, dtype=torch.int64,
                             device=features.device)
    else:
        if padding_mask.dtype != torch.bool or padding_mask.shape != features.shape[:2]:
            raise ValueError('padding_mask must be boolean with shape [B,L]')
        lengths = padding_mask.sum(dim=1).to(torch.int64)
        expected = torch.arange(padded_length, device=padding_mask.device)[None, :] < lengths[:, None]
        if not torch.equal(padding_mask, expected):
            raise ValueError('padding_mask must mark a contiguous valid ORF prefix')
        if bool((lengths == 0).any()):
            raise ValueError('Cannot infer an empty ORF sequence')

    if bool((lengths == padded_length).all()):
        orf_features = features.transpose(1, 2).contiguous()
        return postprocessor(model(orf_features), lengths.to(features.device)), lengths

    groups = {}
    for index, sequence_length in enumerate(lengths.tolist()):
        groups.setdefault(int(sequence_length), []).append(index)
    results = [None] * batch_size
    for sequence_length, indices in groups.items():
        if len(indices) == batch_size:
            group = features[:, :sequence_length, :]
        else:
            index_tensor = torch.tensor(indices, dtype=torch.int64, device=features.device)
            group = features.index_select(0, index_tensor)[:, :sequence_length, :]
        orf_features = group.transpose(1, 2).contiguous()
        group_lengths = torch.full((len(indices),), sequence_length, dtype=torch.int64,
                                   device=features.device)
        group_results = postprocessor(model(orf_features), group_lengths)
        if len(group_results) != len(indices):
            raise ValueError('Postprocessor must return one result per ORF sequence')
        for index, result in zip(indices, group_results):
            results[index] = result
    if any(result is None for result in results):
        raise ValueError('Postprocessor returned an incomplete ORF batch')
    return results, lengths
