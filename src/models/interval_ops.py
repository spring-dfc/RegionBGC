"""Continuous geometry for BGC regions.

Model/target coordinates are normalized (center, length). Endpoint intervals
use [start, end); clipping and integer rounding are reserved for inference.
"""

import torch


INTERVAL_REPRESENTATION = 'center_length_1d_v1'


def validate_interval_checkpoint(checkpoint, state):
    """Reject incompatible geometry before loading weights or optimizer state."""
    representation = checkpoint.get('interval_representation')
    if representation is not None and representation != INTERVAL_REPRESENTATION:
        raise ValueError(f'Unsupported BGC interval representation: {representation}')
    for name, value in state.items():
        is_output = name.endswith('enc_interval_head.layers.2.weight') or (
            'dec_interval_head.' in name and name.endswith('.layers.2.weight')
        )
        is_position = name.endswith('query_pos_head.layers.0.weight')
        if (is_output and value.shape[0] != 2) or (is_position and value.shape[1] != 2):
            raise ValueError(
                'Checkpoint uses incompatible BGC interval coordinates; this model requires '
                '(center,length) intervals. Old four-coordinate checkpoints cannot be '
                'used for resume/evaluation. Use a compatible native interval checkpoint.'
            )


def interval_cw_to_se(intervals):
    if intervals.shape[-1] != 2:
        raise ValueError('Expected BGC (center, length) intervals with last dimension 2')
    center, length = intervals.unbind(-1)
    return torch.stack((center - length * 0.5, center + length * 0.5), dim=-1)


def interval_iou(intervals1, intervals2):
    """Pairwise IoU and union for endpoint tensors [N,2] and [M,2]."""
    lengths1 = (intervals1[:, 1] - intervals1[:, 0]).clamp(min=0)
    lengths2 = (intervals2[:, 1] - intervals2[:, 0]).clamp(min=0)
    intersection = (
        torch.minimum(intervals1[:, None, 1], intervals2[None, :, 1])
        - torch.maximum(intervals1[:, None, 0], intervals2[None, :, 0])
    ).clamp(min=0)
    union = lengths1[:, None] + lengths2[None, :] - intersection
    return intersection / union.clamp(min=torch.finfo(union.dtype).eps), union


def generalized_interval_iou(intervals1, intervals2):
    """Pairwise generalized IoU on the sequence axis, including disjoint intervals."""
    iou, union = interval_iou(intervals1, intervals2)
    enclosing = (
        torch.maximum(intervals1[:, None, 1], intervals2[None, :, 1])
        - torch.minimum(intervals1[:, None, 0], intervals2[None, :, 0])
    ).clamp(min=torch.finfo(union.dtype).eps)
    return iou - (enclosing - union) / enclosing
