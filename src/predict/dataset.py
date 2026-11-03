from __future__ import annotations

import json
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from src.utils.bgc_labels import parse_binary_labels
from src.utils.bgc_data import sample_orf_ids


def _load_json_list(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Prediction JSON must be a list")
    return data


def _parse_labels(raw_labels, length: int) -> List[float]:
    return parse_binary_labels(raw_labels, length, context="Prediction sample")


class PredictionDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        features_manager,
        max_orfs: int = 128,
        overlap_ratio: float = 0.5,
        use_sliding_window: bool = True,
        strict_feature_match: bool = True,
    ) -> None:
        self.features_manager = features_manager
        self.max_orfs = max_orfs
        self.overlap_ratio = overlap_ratio
        if not strict_feature_match:
            raise ValueError("strict_feature_match=False is no longer supported")
        raw_data = _load_json_list(data_file)
        self.data = self._preprocess(raw_data, use_sliding_window=use_sliding_window)
        self.embedding_dim = int(features_manager.embedding_dim)

    def _preprocess(self, raw_data: List[Dict], use_sliding_window: bool) -> List[Dict]:
        processed: List[Dict] = []
        for sample in raw_data:
            genome_id = str(sample.get("original_genome_id") or sample.get("genome_id"))
            sample_id = str(sample.get("genome_id") or genome_id)
            orf_sequences_data = sample.get("orf_sequences", {})
            orf_ids = sample_orf_ids(sample)
            if isinstance(orf_sequences_data, dict):
                orf_sequences = [orf_sequences_data[orf_id] for orf_id in orf_ids]
            else:
                orf_sequences = list(orf_sequences_data)

            if len(orf_ids) != len(orf_sequences) or not orf_ids:
                raise ValueError(f"Genome {genome_id}: ORF IDs must align with nonempty sequences")
            bgc_labels = _parse_labels(sample['bgc_labels'], len(orf_sequences))
            if use_sliding_window and len(orf_sequences) > self.max_orfs:
                processed.extend(self._split_windows(
                    sample_id, orf_sequences, orf_ids, bgc_labels,
                    original_genome_id=genome_id,
                    base_start=int(sample.get("segment_start") or 0),
                    total_orfs=sample.get("total_orfs_in_genome", len(orf_sequences)),
                ))
            else:
                processed.append(
                    {
                        "genome_id": sample_id,
                        "original_genome_id": genome_id,
                        "orf_sequences": orf_sequences,
                        "orf_ids": orf_ids,
                        "bgc_labels": bgc_labels,
                        "segment_start": sample.get("segment_start"),
                        "segment_end": sample.get("segment_end"),
                        "total_orfs_in_genome": sample.get("total_orfs_in_genome", len(orf_sequences)),
                    }
                )
        return processed

    def _split_windows(
        self,
        genome_id: str,
        orf_sequences: List[str],
        orf_ids: List[str],
        bgc_labels: List[float],
        original_genome_id: Optional[str] = None,
        base_start: int = 0,
        total_orfs: Optional[int] = None,
    ) -> List[Dict]:
        step_size = max(1, int(self.max_orfs * (1 - self.overlap_ratio)))
        total = len(orf_sequences)
        original_genome_id = original_genome_id or genome_id
        total_orfs = int(total_orfs) if total_orfs is not None else total
        starts = list(range(0, total - self.max_orfs + 1, step_size))
        last_start = max(0, total - self.max_orfs)
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        segments: List[Dict] = []

        for start in starts:
            end = min(start + self.max_orfs, total)
            segments.append(
                {
                    "genome_id": f"{genome_id}_seg_{base_start + start}_{base_start + end}",
                    "original_genome_id": original_genome_id,
                    "orf_sequences": orf_sequences[start:end],
                    "orf_ids": orf_ids[start:end],
                    "bgc_labels": bgc_labels[start:end],
                    "segment_start": base_start + start,
                    "segment_end": base_start + end,
                    "total_orfs_in_genome": total_orfs,
                }
            )

        return segments

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]
        orf_sequences = list(sample["orf_sequences"])[: self.max_orfs]
        orf_ids = list(sample["orf_ids"])[: self.max_orfs]
        genome_id = sample["genome_id"]
        original_genome_id = sample["original_genome_id"]
        num_orfs = len(orf_ids)

        features = torch.zeros((self.max_orfs, self.embedding_dim), dtype=torch.float32)
        padding_mask = torch.zeros(self.max_orfs, dtype=torch.bool)
        padding_mask[:num_orfs] = True

        missing_features: List[str] = []
        for i, orf_id in enumerate(orf_ids):
            feature = self.features_manager.get_feature(orf_id, original_genome_id, copy=False)
            if feature is None:
                missing_features.append(f"{original_genome_id}_{orf_id}")
            else:
                features[i] = torch.from_numpy(feature).float()

        if missing_features:
            examples = ", ".join(missing_features[:10])
            raise KeyError(
                "Missing ORF features for prediction. "
                f"genome_id={original_genome_id}, "
                f"missing={len(missing_features)}/{num_orfs}. "
                f"Examples: {examples}. "
                "Check that the dataset IDs and feature-file IDs use the same "
                "composite format: {original_genome_id}_{orf_id}."
            )

        bgc_labels = torch.zeros(self.max_orfs, dtype=torch.float32)
        bgc_labels[:num_orfs] = torch.tensor(sample["bgc_labels"][:num_orfs], dtype=torch.float32)

        return {
            "orf_sequences": orf_sequences,
            "orf_ids": orf_ids,
            "padding_mask": padding_mask,
            "bgc_labels": bgc_labels,
            "features": features,
            "genome_id": genome_id,
            "original_genome_id": original_genome_id,
            "segment_start": sample.get("segment_start"),
            "segment_end": sample.get("segment_end"),
            "total_orfs_in_genome": sample.get("total_orfs_in_genome", num_orfs),
        }


def collate_prediction_batch(batch: List[Dict]) -> Dict:
    return {
        "orf_sequences": [sample["orf_sequences"] for sample in batch],
        "orf_ids": [sample["orf_ids"] for sample in batch],
        "padding_mask": torch.stack([sample["padding_mask"] for sample in batch]),
        "bgc_labels": torch.stack([sample["bgc_labels"] for sample in batch]),
        "features": torch.stack([sample["features"] for sample in batch]),
        "genome_id": [sample["genome_id"] for sample in batch],
        "original_genome_id": [sample["original_genome_id"] for sample in batch],
        "segment_start": [sample.get("segment_start") for sample in batch],
        "segment_end": [sample.get("segment_end") for sample in batch],
        "total_orfs_in_genome": [sample.get("total_orfs_in_genome") for sample in batch],
    }
