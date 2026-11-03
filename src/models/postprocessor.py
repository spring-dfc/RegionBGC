"""RegionBGC interval postprocessing on the ORF sequence axis
Converts normalized (center,length) to [start,end) on the ORF index axis.
"""
from typing import List, Dict

import torch
import torch.nn as nn
from .interval_ops import interval_cw_to_se

from src.core import register


@register
class RegionBGCPostProcessor(nn.Module):
    __share__ = ['num_classes', 'num_top_queries']

    def __init__(self, num_classes: int = 1,
                 num_top_queries: int = 300, round_to_int: bool = True,
                 clamp_to_len: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.round_to_int = round_to_int
        self.clamp_to_len = clamp_to_len

    def forward(self, outputs: Dict[str, torch.Tensor], sequence_lengths: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Args:
            outputs: 'pred_logits' [B,Q,C], 'pred_intervals' [B,Q,2] as normalized (center,length)
            sequence_lengths: [B], the number of valid ORFs per sample
        Returns: list of results per batch item:
            - segments: [N,2] tensor of [start,end] (int or float)
            - scores: [N]
            - labels: [N]
        """
        logits, intervals = outputs['pred_logits'], outputs['pred_intervals']
        pred_bgc_types = outputs.get('pred_bgc_types', None)  # optional: [B, Q, K]
        B = logits.shape[0]
        if sequence_lengths.ndim != 1 or sequence_lengths.shape[0] != B:
            raise ValueError('sequence_lengths must have shape [B]')

        # Convert directly to continuous endpoint coordinates on the ORF axis.
        scale = sequence_lengths[:, None, None]  # [B,1,1]
        intervals = interval_cw_to_se(intervals) * scale

        picked_bgc_type_probs = None

        probs = torch.sigmoid(logits)  # [B,Q,C]
        scores, flat_idx = torch.topk(probs.flatten(1), k=min(self.num_top_queries, probs.numel() // B), dim=1)
        labels = flat_idx % self.num_classes
        query_idx = flat_idx // self.num_classes
        # gather intervals
        gather_idx = query_idx.unsqueeze(-1).repeat(1, 1, 2)
        picked = torch.gather(intervals, dim=1, index=gather_idx)

        # optional: gather per-query multi-label bgc types
        if pred_bgc_types is not None:
            # pred_bgc_types are logits (multi-label); convert to probs with sigmoid
            K = int(pred_bgc_types.shape[-1])
            gather_t = query_idx.unsqueeze(-1).repeat(1, 1, K)
            picked_types = torch.gather(pred_bgc_types, dim=1, index=gather_t)
            picked_bgc_type_probs = torch.sigmoid(picked_types)

        # convert to interval endpoints on the ORF axis
        starts = picked[..., 0]
        ends = picked[..., 1]
        if self.clamp_to_len:
            valid_lengths = sequence_lengths[:, None]
            # PyTorch clamp requires min/max to be same type; do it in two steps for broadcasting safety
            starts = starts.clamp(min=0)
            ends = ends.clamp(min=0)
            starts = torch.minimum(starts, valid_lengths)
            ends = torch.minimum(ends, valid_lengths)
        if self.round_to_int:
            starts = starts.round().to(torch.int64)
            ends = ends.round().to(torch.int64)
        segments = torch.stack([starts, ends], dim=-1)  # [B,K,2]

        results: List[Dict[str, torch.Tensor]] = []
        for b in range(B):
            out = {
                'segments': segments[b],
                'scores': scores[b],
                'labels': labels[b],
            }
            # Attach optional BGC type probabilities per kept query/segment
            if picked_bgc_type_probs is not None:
                out['bgc_type_probs'] = picked_bgc_type_probs[b]
            results.append(out)
        return results
