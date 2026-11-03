"""Read ordered ORF IDs consistently across training, prediction and evaluation."""

import json


def sample_orf_ids(sample):
    """Use ``orf_ids`` when provided, otherwise use protein-dictionary keys."""
    context = f"Sample {sample.get('genome_id', '<unknown>')}"
    sequences = sample.get('orf_sequences')
    values = sample.get('orf_ids')
    if values is None:
        if not isinstance(sequences, dict):
            raise ValueError(f"{context}: orf_ids are required for sequence lists")
        values = list(sequences)
    elif isinstance(values, str):
        try:
            values = json.loads(values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: invalid orf_ids JSON") from exc
    if not isinstance(values, list) or not values:
        raise ValueError(f"{context}: orf_ids must be a nonempty list")
    ids = [str(value) for value in values]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{context}: orf_ids must be nonempty and unique")
    if isinstance(sequences, dict):
        if ids != list(sequences):
            raise ValueError(f"{context}: orf_ids must match protein-dictionary key order")
    elif sequences is not None and len(ids) != len(sequences):
        raise ValueError(f"{context}: orf_ids must align with protein sequences")
    return ids
