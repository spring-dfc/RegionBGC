"""Strict parsing of observed per-ORF labels; missing labels are never background."""

import json
import math


def parse_binary_labels(raw_labels, length, *, context="sample"):
    if isinstance(raw_labels, str):
        try:
            raw_labels = json.loads(raw_labels)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: invalid bgc_labels JSON") from exc
    if not isinstance(raw_labels, (list, tuple)):
        raise ValueError(f"{context}: bgc_labels must be a binary list")
    if len(raw_labels) != length:
        raise ValueError(
            f"{context}: bgc_labels length {len(raw_labels)} != ORF count {length}"
        )
    labels = []
    for index, value in enumerate(raw_labels):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: invalid bgc_labels[{index}]={value!r}") from exc
        if not math.isfinite(number) or number not in (0.0, 1.0):
            raise ValueError(f"{context}: bgc_labels[{index}] must be 0 or 1, got {value!r}")
        labels.append(number)
    return labels
