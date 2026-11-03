"""Normalize and aggregate split genome identifiers."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
from utils.bgc_labels import parse_binary_labels



# SSFG -> SSFG_1, SSFG_2
# SSMG -> SSMG_1, SSMG_2
# SSNG -> SSNG_1, SSNG_2, SSNG_3
_SPLIT_RE = re.compile(r"^(SSFG|SSMG|SSNG)_(\d+)$")


def base_genome_id(genome_id: str) -> str:
    """Map a known split identifier to its base genome identifier."""
    s = str(genome_id) if genome_id is not None else ""
    m = _SPLIT_RE.match(s)
    return m.group(1) if m else s


def _fragment_sort_key(genome_id: str) -> Tuple[str, int, str]:
    """Return a deterministic ordering key for genome fragments."""
    s = str(genome_id) if genome_id is not None else ""
    m = _SPLIT_RE.match(s)
    if m:
        return (m.group(1), int(m.group(2)), s)

    return (base_genome_id(s), 10**9, s)


def aggregate_fused_predictions_by_base(
    fused: Dict[str, Dict],
    *,
    keep_singleton_key: bool = False,
) -> Dict[str, Dict]:
    """Concatenate fragment-level outputs after per-fragment fusion.

    Fusion must happen before concatenation because sliding-window offsets are
    local to each fragment.
    """
    if not isinstance(fused, dict) or len(fused) == 0:
        return fused if isinstance(fused, dict) else {}

    groups: Dict[str, List[Tuple[str, Dict]]] = {}
    for gid, item in fused.items():
        if not isinstance(item, dict):
            raise ValueError(f"Genome {gid}: fused result must be a dict")
        if 'true_labels' in item:
            parse_binary_labels(item['true_labels'], len(item['predictions']), context=f"Genome {gid}")
        base = base_genome_id(gid)
        groups.setdefault(base, []).append((str(gid), item))

    out: Dict[str, Dict] = {}
    for base, items in groups.items():
        if len(items) == 1 and (keep_singleton_key and items[0][0] != base) is False:

            out[base] = items[0][1]
            continue

        items.sort(key=lambda t: _fragment_sort_key(t[0]))
        merged_preds: List[float] = []
        merged_trues: Optional[List[float]] = []  # only if any true_labels exists
        merged_orf_ids: Optional[List[str]] = []  # only if any orf_ids exists
        any_trues = any(isinstance(it.get("true_labels", None), (list, tuple)) for _, it in items if isinstance(it, dict))
        any_orf = any(isinstance(it.get("orf_ids", None), (list, tuple)) for _, it in items if isinstance(it, dict))

        num_segments = 0
        for gid, it in items:
            if not isinstance(it, dict):
                continue
            preds = it.get("predictions", []) or []
            merged_preds.extend(list(preds))

            if any_trues and merged_trues is not None:
                trues = it.get("true_labels", None)
                if isinstance(trues, (list, tuple)):
                    merged_trues.extend([float(x) for x in trues])
                else:

                    raise ValueError(f"Genome fragment {gid}: missing true_labels")

            if any_orf and merged_orf_ids is not None:
                orf_ids = it.get("orf_ids", None)
                if isinstance(orf_ids, (list, tuple)):
                    merged_orf_ids.extend([str(x) for x in orf_ids])
                else:

                    merged_orf_ids.extend([f"{gid}:ORF_{i+1:06d}" for i in range(len(preds))])

            try:
                num_segments += int(it.get("num_segments", 1) or 1)
            except Exception:
                num_segments += 1

        merged: Dict[str, object] = {
            "predictions": merged_preds,
            "is_fused": True,
            "num_segments": int(num_segments),
            "fusion_method": "concat_base_genome",
        }
        if any_trues and merged_trues is not None:
            merged["true_labels"] = merged_trues
        if any_orf and merged_orf_ids is not None:
            merged["orf_ids"] = merged_orf_ids
        out[base] = merged

    return out


__all__ = ["base_genome_id", "aggregate_fused_predictions_by_base"]
