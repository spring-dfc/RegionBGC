"""ORF-interval validation and nucleotide-coordinate independent region metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


Segment = Tuple[int, int]  # [start, end) half-open


def _extract_regions_half_open(binary: np.ndarray) -> List[Segment]:
    """Extract contiguous 1-runs as half-open [start, end)."""
    if binary is None:
        return []
    x = np.asarray(binary, dtype=np.int8)
    if x.size == 0:
        return []
    padded = np.pad(x, (1, 1), mode="constant")
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends_inclusive = np.where(diff == -1)[0] - 1
    if starts.size == 0:
        return []
    # convert inclusive end -> exclusive end
    return [(int(s), int(e) + 1) for s, e in zip(starts.tolist(), ends_inclusive.tolist())]


def interval_iou_half_open(a: Segment, b: Segment) -> float:
    """IoU for half-open regions [start, end)."""
    a0, a1 = int(a[0]), int(a[1])
    b0, b1 = int(b[0]), int(b[1])
    inter = max(0, min(a1, b1) - max(a0, b0))
    if inter <= 0:
        return 0.0
    union = (a1 - a0) + (b1 - b0) - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass(frozen=True)
class PredInterval:
    sample_id: str
    score: float
    seg: Segment


def compute_interval_prf_at_score(
    preds: List[PredInterval],
    gts_by_sample: Dict[str, List[Segment]],
    *,
    iou_thr: float = 0.5,
    score_thr: float = 0.3,
) -> Dict[str, float]:
    """Training validation on half-open ORF-index intervals."""
    # Validation matches predictions in descending confidence order.
    kept = [p for p in preds if float(p.score) >= float(score_thr)]
    kept.sort(key=lambda x: float(x.score), reverse=True)

    matched = {sid: np.zeros(len(gts_by_sample.get(sid, [])), dtype=bool) for sid in gts_by_sample.keys()}
    tp = 0
    fp = 0
    total_gt = int(sum(len(v) for v in gts_by_sample.values()))

    for p in kept:
        gts = gts_by_sample.get(p.sample_id, [])
        if not gts:
            fp += 1
            continue
        m = matched.get(p.sample_id)
        if m is None or m.size != len(gts):
            m = np.zeros(len(gts), dtype=bool)
            matched[p.sample_id] = m

        best_iou = 0.0
        best_j: Optional[int] = None
        for j, gt in enumerate(gts):
            if m[j]:
                continue
            iou = interval_iou_half_open(p.seg, gt)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j is not None and best_iou >= float(iou_thr):
            m[best_j] = True
            tp += 1
        else:
            fp += 1

    fn = total_gt - tp
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "interval_precision": float(precision),
        "interval_recall": float(recall),
        "interval_f1": float(f1),
        "interval_tp": int(tp),
        "interval_fp": int(fp),
        "interval_fn": int(fn),
    }


def build_gt_intervals_from_labels(labels: np.ndarray) -> List[Segment]:
    """labels: 0/1 or float array; threshold at >0.5."""
    y = (np.asarray(labels, dtype=float) > 0.5).astype(np.int8)
    return _extract_regions_half_open(y)


def normalize_pred_segments(segments: np.ndarray, scores: np.ndarray, *, W: int) -> List[Tuple[Segment, float]]:
    """Clamp, sanitize, convert to half-open [s,e) segments."""
    if segments is None or scores is None:
        return []
    segs = np.asarray(segments)
    scs = np.asarray(scores, dtype=float)
    if segs.size == 0 or scs.size == 0:
        return []
    segs = segs.reshape(-1, 2)
    n = min(segs.shape[0], scs.shape[0])
    out: List[Tuple[Segment, float]] = []
    for i in range(n):
        s = int(segs[i, 0])
        e = int(segs[i, 1])
        # Clamp to [0, W]
        s = max(0, min(s, int(W)))
        e = max(0, min(e, int(W)))
        if e <= s:
            continue
        out.append(((s, e), float(scs[i])))
    return out


__all__ = [
    "compute_bp_region_metrics",
    "PredInterval",
    "build_gt_intervals_from_labels",
    "normalize_pred_segments",
    "compute_interval_prf_at_score",
    "interval_iou_half_open",
]


def _bp_iou(a, b):
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = a[1] - a[0] + 1 + b[1] - b[0] + 1 - intersection
    return intersection / union if intersection else 0.0


def _match_bp_regions(predictions, truth, minimum_iou):
    candidates = []
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, target in enumerate(truth):
            iou = _bp_iou(prediction, target)
            if iou > 0 and iou >= minimum_iou:
                candidates.append((iou, prediction_index, truth_index))
    used_predictions, used_truth, matches = set(), set(), []
    for iou, prediction_index, truth_index in sorted(candidates, reverse=True):
        if prediction_index in used_predictions or truth_index in used_truth:
            continue
        used_predictions.add(prediction_index)
        used_truth.add(truth_index)
        matches.append(iou)
    return matches


def compute_bp_region_metrics(
    truth_by_sample, predictions_by_sample, positions_by_sample, *, iou_thresholds=(0.3, 0.5),
):
    """Independent region metrics using inclusive nucleotide coordinates."""
    if set(truth_by_sample) != set(predictions_by_sample) or set(truth_by_sample) != set(positions_by_sample):
        raise ValueError("Truth, predictions and coordinates must cover the same samples")
    thresholds = tuple(float(value) for value in iou_thresholds)
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("IoU thresholds must be within [0, 1]")
    true_positives = {value: 0 for value in thresholds}
    total_predicted = total_truth = overlap_count = 0
    overlap_sum = 0.0
    for sample_id, labels in truth_by_sample.items():
        truth = np.asarray(labels)
        predicted = np.asarray(predictions_by_sample[sample_id])
        positions = np.asarray(positions_by_sample[sample_id], dtype=float)
        if truth.ndim != 1 or predicted.shape != truth.shape:
            raise ValueError(f"{sample_id}: truth/prediction length mismatch")
        if not np.isin(truth, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
            raise ValueError(f"{sample_id}: labels and predictions must be binary")
        if positions.shape != (len(truth), 2) or not np.isfinite(positions).all():
            raise ValueError(f"{sample_id}: missing or invalid nucleotide coordinates")
        if (positions < 0).any() or not np.equal(positions, np.floor(positions)).all():
            raise ValueError(f"{sample_id}: nucleotide coordinates must be non-negative integers")
        def to_bp(binary):
            return [(int(positions[start:end].min()), int(positions[start:end].max()))
                    for start, end in _extract_regions_half_open(binary)]
        true_regions, predicted_regions = to_bp(truth), to_bp(predicted)
        total_truth += len(true_regions)
        total_predicted += len(predicted_regions)
        overlaps = _match_bp_regions(predicted_regions, true_regions, 0.0)
        overlap_sum += float(np.sum(overlaps)) if overlaps else 0.0
        overlap_count += len(overlaps)
        for cutoff in thresholds:
            true_positives[cutoff] += len(_match_bp_regions(predicted_regions, true_regions, cutoff))
    result = {}
    for cutoff, tp in true_positives.items():
        if total_predicted == total_truth == 0:
            precision = recall = f1 = 1.0
        else:
            precision = tp / total_predicted if total_predicted else 0.0
            recall = tp / total_truth if total_truth else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[f"{cutoff:g}"] = {
            "precision": precision, "recall": recall, "f1": f1,
            "avg_iou": overlap_sum / overlap_count if overlap_count else 0.0,
            "tp": tp, "fp": total_predicted - tp, "fn": total_truth - tp,
            "predicted": total_predicted, "ground_truth": total_truth,
        }
    return result
