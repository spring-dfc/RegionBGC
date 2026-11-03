from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from utils.bgc_labels import parse_binary_labels


BGC_TYPE_VOCAB = ("PKS", "NRPS", "RiPP", "saccharide", "terpene", "other")


def extract_positive_regions(scores: np.ndarray, threshold: float) -> List[tuple[int, int]]:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return []
    positive = (values >= float(threshold)).astype(np.int8)
    padded = np.pad(positive, (1, 1), mode="constant")
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def update_genome_context(
    context_by_genome: Dict[str, Dict],
    genome_id: str,
    total_length: int,
    *,
    segment_start: Optional[int] = None,
    orf_ids: Optional[List[str]] = None,
    true_labels: Optional[List[float]] = None,
) -> None:
    gid = str(genome_id)
    total_length = max(int(total_length), 0)
    ctx = context_by_genome.get(gid)
    if ctx is None:
        ctx = {
            "total_length": total_length,
            "orf_ids": [None] * total_length,
            "true_labels": None,
        }
        context_by_genome[gid] = ctx
    elif total_length > int(ctx.get("total_length", 0)):
        old_len = int(ctx.get("total_length", 0))
        ctx["total_length"] = total_length
        ctx.setdefault("orf_ids", [])
        ctx["orf_ids"].extend([None] * max(0, total_length - len(ctx["orf_ids"])))
        if isinstance(ctx.get("true_labels"), list):
            ctx["true_labels"].extend([None] * max(0, total_length - old_len))

    start = int(segment_start) if segment_start is not None else 0
    if orf_ids:
        end = min(total_length, start + len(orf_ids))
        for pos in range(start, end):
            local_idx = pos - start
            if ctx["orf_ids"][pos] is None:
                ctx["orf_ids"][pos] = str(orf_ids[local_idx])

    if true_labels is not None:
        true_labels = parse_binary_labels(true_labels, len(true_labels), context=f"Genome {gid}")
        if start < 0 or start + len(true_labels) > int(ctx['total_length']):
            raise ValueError(f"Genome {gid}: labeled segment exceeds genome bounds")
        if orf_ids is not None and len(orf_ids) != len(true_labels):
            raise ValueError(f"Genome {gid}: ORF IDs and labels must have equal lengths")
        if ctx.get("true_labels") is None:
            ctx["true_labels"] = [None] * int(ctx['total_length'])
        end = min(total_length, start + len(true_labels))
        for pos in range(start, end):
            local_idx = pos - start
            previous = ctx['true_labels'][pos]
            if previous is not None and previous != true_labels[local_idx]:
                raise ValueError(f"Genome {gid}: conflicting labels at ORF index {pos}")
            ctx["true_labels"][pos] = float(true_labels[local_idx])


def build_genome_consensus(
    intervals_by_genome: Dict[str, List[Dict]],
    context_by_genome: Dict[str, Dict],
    *,
    threshold: float,
    bgc_type_threshold: float = 0.5,
) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}

    genome_ids = sorted(set(intervals_by_genome.keys()) | set(context_by_genome.keys()))
    for genome_id in genome_ids:
        ctx = context_by_genome.get(genome_id, {})
        total_length = int(ctx.get("total_length", 0))
        intervals = list(intervals_by_genome.get(genome_id, []))

        if total_length <= 0:
            for interval in intervals:
                total_length = max(total_length, int(interval.get("end", 0)))

        token_scores = np.zeros(total_length, dtype=np.float32)
        winner_interval_idx = np.full(total_length, -1, dtype=np.int64)

        for idx, interval in enumerate(intervals):
            start = max(0, min(int(interval.get("start", 0)), total_length))
            end = max(0, min(int(interval.get("end", 0)), total_length))
            if end <= start:
                continue
            score = float(interval.get("score", 0.0))
            local_scores = token_scores[start:end]
            replace_mask = score > local_scores
            if np.any(replace_mask):
                local_scores[replace_mask] = score
                token_scores[start:end] = local_scores
                winner_slice = winner_interval_idx[start:end]
                winner_slice[replace_mask] = idx
                winner_interval_idx[start:end] = winner_slice

        final_regions = []
        for start, end in extract_positive_regions(token_scores, threshold):
            winner_ids = []
            for idx in winner_interval_idx[start:end].tolist():
                if idx >= 0 and idx not in winner_ids:
                    winner_ids.append(idx)

            class_names: List[str] = []
            for type_idx, type_name in enumerate(BGC_TYPE_VOCAB):
                for interval_idx in winner_ids:
                    probs = intervals[interval_idx].get("bgc_type_probs", [])
                    if type_idx < len(probs) and float(probs[type_idx]) >= float(bgc_type_threshold):
                        class_names.append(type_name)
                        break

            region_scores = token_scores[start:end]
            final_regions.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "confidence": float(region_scores.mean()) if region_scores.size > 0 else 0.0,
                    "class_name": ";".join(class_names) if class_names else "none",
                    "dominant_interval_indices": winner_ids,
                }
            )

        orf_ids = list(ctx.get("orf_ids", [])) if isinstance(ctx.get("orf_ids"), list) else [None] * total_length
        if len(orf_ids) < total_length:
            orf_ids.extend([None] * (total_length - len(orf_ids)))
        for pos in range(total_length):
            if orf_ids[pos] is None:
                orf_ids[pos] = f"{genome_id}:ORF_{pos + 1:06d}"

        result = {
            "predictions": token_scores.tolist(),
            "orf_ids": [str(x) for x in orf_ids],
            "final_regions": final_regions,
            "intervals": intervals,
            "is_interval_consensus": True,
        }
        true_labels = ctx.get("true_labels")
        if isinstance(true_labels, list):
            result['true_labels'] = parse_binary_labels(
                true_labels, total_length, context=f"Genome {genome_id} (fused labels)"
            )
        results[genome_id] = result

    return results


__all__ = [
    "BGC_TYPE_VOCAB",
    "build_genome_consensus",
    "extract_positive_regions",
    "update_genome_context",
]
