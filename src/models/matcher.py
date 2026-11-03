# RegionBGC adaptation for BGC region prediction; upstream attribution is retained.
"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

by lyuwenyu
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence

from scipy.optimize import linear_sum_assignment
from torch import nn

from .interval_ops import interval_cw_to_se, generalized_interval_iou

from src.core import register


@register
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, weight_dict, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_interval_l1: This is the relative weight of the L1 error of the BGC interval coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the BGC interval in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_interval_l1 = weight_dict['cost_interval_l1']
        self.cost_giou = weight_dict['cost_giou']

        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_interval_l1 != 0 or self.cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_intervals": [batch_size, num_queries, 2], normalized (center, length)

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_intervals] (where num_target_intervals is the number of ground-truth
                           objects in the target) containing the class labels
                 "intervals": [num_target_intervals, 2], normalized (center, length)

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_intervals)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))

        out_interval = outputs["pred_intervals"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Also concat the target labels and intervals
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_interval = torch.cat([v["intervals"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        out_prob = out_prob[:, tgt_ids]
        neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = self.alpha * ((1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class - neg_cost_class

        # Compute the L1 cost between intervals
        cost_interval_l1 = torch.cdist(out_interval, tgt_interval, p=1)

        # Compute the giou cost betwen intervals
        cost_giou = -generalized_interval_iou(interval_cw_to_se(out_interval), interval_cw_to_se(tgt_interval))

        # Final cost matrix
        C = self.cost_interval_l1 * cost_interval_l1 + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["intervals"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def match_queries_by_hungarian(
    matcher: torch.nn.Module,
    outputs: Dict[str, object],
    targets: Sequence[Dict[str, object]],
) -> List[List[Optional[int]]]:
    """Map each GT to its training-matcher query, without score/IoU cutoffs."""
    match_outputs = {key: value for key, value in outputs.items() if 'aux' not in key}
    assignments: List[List[Optional[int]]] = []
    for target, (query_indices, gt_indices) in zip(targets, matcher(match_outputs, targets)):
        matches: List[Optional[int]] = [None] * int(target['intervals'].shape[0])
        for query_index, gt_index in zip(query_indices.tolist(), gt_indices.tolist()):
            matches[int(gt_index)] = int(query_index)
        assignments.append(matches)
    return assignments
