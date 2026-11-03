# RegionBGC adaptation for BGC region prediction; upstream attribution is retained.
"""
reference:
https://github.com/facebookresearch/detr/blob/main/models/detr.py

by lyuwenyu
"""


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .interval_ops import interval_cw_to_se, interval_iou, generalized_interval_iou

from src.misc.dist import get_world_size, is_dist_available_and_initialized
from src.core import register


@register
class RegionBGCCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth intervals and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and interval)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0,
                 num_classes=1, bgc_alpha=0.75, bgc_gamma=None,
                 det_pos_weight=1.0):
        """IA-BCE detection, interval L1/GIoU and matched multi-label Focal."""
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        if not set(losses).issubset({'ia_bce', 'intervals', 'bgc_types'}):
            raise ValueError('Unsupported BGC interval loss')
        self.alpha = alpha
        self.gamma = gamma
        self.det_pos_weight = float(det_pos_weight)
        if not math.isfinite(self.det_pos_weight) or self.det_pos_weight <= 0:
            raise ValueError('det_pos_weight must be finite and greater than zero')
        self.bgc_alpha = bgc_alpha
        self.bgc_gamma = gamma if bgc_gamma is None else bgc_gamma

    def loss_labels_ia_bce(self, outputs, targets, indices, num_intervals, log=True):
        """IoU-aware BCE classification loss (RF-DETR / LW-DETR style).

        Key idea: for matched (positive) query-class positions, build a soft target
        t = p^alpha * IoU^(1-alpha) (clamped), and optimize a BCE with soft labels.
        For negatives, down-weight easy negatives with p^gamma (focal-like).
        """
        assert 'pred_logits' in outputs
        assert 'pred_intervals' in outputs

        src_logits = outputs['pred_logits']  # [B, Q, C]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # Empty matches still supervise every query as background. The empty
        # positive tensors below leave the negative focal-BCE terms intact.
        # IoU for matched pairs (detach: do not backprop through interval for classification loss)
        src_intervals = outputs['pred_intervals'][idx]
        target_intervals = torch.cat([t['intervals'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = interval_iou(interval_cw_to_se(src_intervals.detach()), interval_cw_to_se(target_intervals))
        pos_ious = torch.diag(ious).detach()

        prob = src_logits.sigmoid()
        gamma = self.gamma
        alpha = self.alpha

        pos_weights = torch.zeros_like(src_logits)
        neg_weights = prob.pow(gamma)

        # Positive indices: (batch_idx, query_idx, class_idx)
        pos_ind = (idx[0], idx[1], target_classes_o)

        # Soft target strength for positives: t = p^alpha * iou^(1-alpha)
        t = prob[pos_ind].pow(alpha) * pos_ious.pow(1.0 - alpha)
        t = torch.clamp(t, min=0.01).detach()

        pos_weights[pos_ind] = t.to(pos_weights.dtype)
        neg_weights[pos_ind] = (1.0 - t).to(neg_weights.dtype)

        # Stable reformulation of:
        #   loss = -pos_w * log(sigmoid(x)) - neg_w * log(1 - sigmoid(x))
        loss = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)
        if self.det_pos_weight != 1.0:
            # Weight the whole soft-label BCE for matched positives, retaining
            # their target t and the original loss/gradient of all negatives.
            sample_weights = torch.ones_like(loss)
            sample_weights[pos_ind] = self.det_pos_weight
            loss = loss * sample_weights
        loss = loss.sum() / num_intervals
        return {'loss_ia_bce': loss}


    def loss_bgc_types(self, outputs, targets, indices, num_intervals, log=True):
        """Multi-label BGC-type loss on top of an additional prediction head.

        This expects:
          - outputs['pred_bgc_types']: Tensor [B, Q, K] with logits for K BGC types
          - targets[i]['bgc_type_vec']: Tensor [num_gt_intervals_i, K] with multi-hot
            vectors inferred from sample['bgc_class'] (built in RegionBGCDataset).

        We only supervise the queries that are matched to ground-truth intervals.
        If the extra head is absent, or there are no positives, we return a
        zero loss tensor on the correct device to keep the pipeline stable.
        """
        if 'pred_bgc_types' not in outputs:
            # No extra head in this configuration / auxiliary output
            src_logits = outputs.get('pred_logits')
            if src_logits is not None:
                return {'loss_bgc_types': src_logits.sum() * 0.0}
            # Fallback: create a scalar zero on the first available device
            first = next(iter(outputs.values()))
            return {'loss_bgc_types': first.sum() * 0.0}

        src_types = outputs['pred_bgc_types']  # [B, Q, K]
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            # No matched predictions
            return {'loss_bgc_types': src_types.sum() * 0.0}

        # Gather matched predictions
        src_types = src_types[idx]  # [N_pos, K]

        # Build matched target multi-hot vectors per positive interval
        target_types = torch.cat(
            [t['bgc_type_vec'][J] for t, (_, J) in zip(targets, indices)],
            dim=0,
        )

        if target_types.numel() == 0:
            raise ValueError("Matched BGC regions require nonempty product targets")

        target_types = target_types.to(src_types.device).type_as(src_types)

        focal = torchvision.ops.sigmoid_focal_loss(
            src_types, target_types, alpha=self.bgc_alpha,
            gamma=self.bgc_gamma, reduction='none',
        )
        loss_raw = focal.sum()
        loss = loss_raw / max(num_intervals, 1)
        return {'loss_bgc_types': loss}


    def loss_intervals(self, outputs, targets, indices, num_intervals):
        """Compute the losses related to the BGC intervals, the L1 regression loss and the GIoU loss
           targets dicts contain "intervals" of shape [nb_target_intervals, 2].
           Coordinates are (center, length), normalized by the ORF sequence length.
        """
        assert 'pred_intervals' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_intervals = outputs['pred_intervals'][idx]
        target_intervals = torch.cat([t['intervals'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        loss_interval_l1 = F.l1_loss(src_intervals, target_intervals, reduction='none')
        losses['loss_interval_l1'] = loss_interval_l1.sum() / num_intervals

        loss_giou = 1 - torch.diag(generalized_interval_iou(
                interval_cw_to_se(src_intervals),
                interval_cw_to_se(target_intervals)))
        losses['loss_giou'] = loss_giou.sum() / num_intervals
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


    def get_loss(self, loss, outputs, targets, indices, num_intervals, **kwargs):
        loss_map = {
            'intervals': self.loss_intervals,

            'ia_bce': self.loss_labels_ia_bce,
            'bgc_types': self.loss_bgc_types,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_intervals, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target intervals accross all nodes, for normalization purposes
        num_intervals = sum(len(t["labels"]) for t in targets)
        num_intervals = torch.as_tensor([num_intervals], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_intervals)
        num_intervals = torch.clamp(num_intervals / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_intervals)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # Helper: compute auxiliary losses with optional scaling.
        def _apply_aux(outputs_list, suffix: str, scale: float = 1.0):
            if not outputs_list:
                return
            try:
                scale_f = float(scale)
            except Exception:
                scale_f = 1.0
            for i, aux_outputs in enumerate(outputs_list):
                indices_i = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_i, num_intervals)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    if scale_f != 1.0:
                        l_dict = {k: v * scale_f for k, v in l_dict.items()}
                    l_dict = {k + f'_{suffix}_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Decoder intermediate-layer aux losses (legacy key: aux_outputs).
        # Default scale is 1.0 for backwards compatibility.
        if 'aux_outputs' in outputs:
            _apply_aux(outputs.get('aux_outputs', []), suffix='aux', scale=outputs.get('aux_weight', 1.0))

        # Encoder top-k aux losses (new key: enc_aux_outputs).
        if 'enc_aux_outputs' in outputs:
            _apply_aux(outputs.get('enc_aux_outputs', []), suffix='enc_aux', scale=outputs.get('enc_aux_weight', 1.0))

        return losses
