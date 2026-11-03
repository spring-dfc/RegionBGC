"""Offline evaluators for the two labeled RegionBGC test sets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.core import YAMLConfig
from src.models.loading import load_evaluation_checkpoint
from src.models.matcher import match_queries_by_hungarian
from src.data.bgc.bgc_dataset import (
    _bgc_multi_hot_from_sample,
    _segments_from_binary_labels,
    bgc_collate_fn,
)
from src.utils.bgc_genome import base_genome_id
from src.metrics.regions import compute_bp_region_metrics
from src.metrics.gene import compute_gene_metrics
from src.metrics.classification import compute_multilabel_micro_metrics
from src.utils.bgc_labels import parse_binary_labels
from src.utils.bgc_data import sample_orf_ids


def _load_labeled_samples(data_file: str) -> List[Dict[str, Any]]:
    with open(data_file, "r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list):
        raise ValueError(f"Labeled JSON must contain a list: {data_file}")
    return samples


def _sample_labels(sample: Dict[str, Any], expected: int) -> List[float]:
    values = sample.get("bgc_labels", [])
    if isinstance(values, str):
        values = json.loads(values)
    labels = [float(value) for value in values]
    if len(labels) != expected:
        raise ValueError(
            f"Label/ORF length mismatch for {sample.get('genome_id')}: "
            f"labels={len(labels)} orfs={expected}"
        )
    return labels


def prepare_localization_dataset(
    *, faa_dir: str, truth_file: str, output_json: str,
    converter: Callable[[str, str], str],
) -> Dict[str, Any]:
    """Convert protein FASTAs with the active engine and build an explicit truth ID map."""
    faa_root = Path(faa_dir).expanduser().resolve()
    if not faa_root.is_dir():
        raise FileNotFoundError(f"Localization FAA directory not found: {faa_root}")
    output_path = Path(output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_dir = output_path.parent / "converted_fragments"
    fragment_dir.mkdir(parents=True, exist_ok=True)
    truth_samples = _load_labeled_samples(truth_file)
    truth_by_genome = {str(sample.get("genome_id")): sample for sample in truth_samples}
    faa_files = sorted(path for path in faa_root.iterdir() if path.is_file() and path.suffix.lower() in {".fa", ".faa", ".fasta"})
    if not faa_files:
        raise FileNotFoundError(f"No protein FASTA files found in: {faa_root}")
    stems = {path.stem for path in faa_files}
    expected_stems = set(truth_by_genome)
    if stems != expected_stems:
        raise ValueError(
            f"FAA/truth genome mismatch: missing={sorted(expected_stems - stems)}, "
            f"extra={sorted(stems - expected_stems)}"
        )
    combined: List[Dict[str, Any]] = []
    prediction_to_truth: Dict[str, str] = {}
    for faa_path in faa_files:
        converted_path = fragment_dir / f"{faa_path.stem}.json"
        actual_path = Path(converter(str(faa_path), str(converted_path))).resolve()
        converted = _load_labeled_samples(str(actual_path))
        truth = truth_by_genome[faa_path.stem]
        truth_ids = sample_orf_ids(truth)
        truth_sequences = truth.get("orf_sequences", {})
        converted_pairs: List[Tuple[str, str]] = []
        for sample in converted:
            sequences = sample.get("orf_sequences", {})
            if not isinstance(sequences, dict):
                raise ValueError(f"Converted sample lacks orf_sequences: {faa_path.name}")
            converted_pairs.extend((str(key), str(value)) for key, value in sequences.items())
        if len(converted_pairs) != len(truth_ids):
            raise ValueError(
                f"Converted/truth ORF count mismatch for {faa_path.stem}: "
                f"converted={len(converted_pairs)} truth={len(truth_ids)}"
            )
        for (prediction_id, sequence), truth_id in zip(converted_pairs, truth_ids):
            if isinstance(truth_sequences, dict) and sequence != str(truth_sequences[truth_id]):
                raise ValueError(
                    f"Converted ORF order/sequence mismatch for {faa_path.stem}: "
                    f"prediction={prediction_id} truth={truth_id}"
                )
            written_id = prediction_id.split(":", 1)[-1]
            if written_id in prediction_to_truth:
                raise ValueError(f"Duplicate deployed ORF ID after output normalization: {written_id}")
            prediction_to_truth[written_id] = truth_id
        combined.extend(converted)
    output_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "dataset_json": str(output_path), "prediction_to_truth_orf": prediction_to_truth,
        "fragments": len(truth_samples),
        "genomes": len({base_genome_id(str(sample.get("genome_id"))) for sample in truth_samples}),
        "orfs": len(prediction_to_truth),
    }


def score_localization_output(
    *, prediction_file: str, truth_file: str,
    prediction_to_truth_orf: Dict[str, str], threshold: float,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Score the native deployed ORF artifact with strict one-to-one alignment."""
    predictions: Dict[str, float] = {}
    with open(prediction_file, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"ORF_ID", "Prob"}.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected ORF prediction header: {reader.fieldnames}")
        for row in reader:
            orf_id = str(row["ORF_ID"])
            if orf_id in predictions:
                raise ValueError(f"Duplicate ORF prediction row: {orf_id}")
            predictions[orf_id] = float(row["Prob"])
    expected, observed = set(prediction_to_truth_orf), set(predictions)
    if expected != observed:
        raise ValueError(
            f"ORF prediction alignment failed: missing {len(expected - observed)}, "
            f"extra {len(observed - expected)}"
        )
    truth_samples = _load_labeled_samples(truth_file)
    truth_values: Dict[str, float] = {}
    truth_order: List[str] = []
    for sample in truth_samples:
        orf_ids = sample_orf_ids(sample)
        labels = _sample_labels(sample, len(orf_ids))
        for orf_id, label in zip(orf_ids, labels):
            if orf_id in truth_values:
                raise ValueError(f"Duplicate truth ORF ID: {orf_id}")
            truth_values[orf_id] = label
            truth_order.append(orf_id)
    truth_to_prediction = {truth_id: pred_id for pred_id, truth_id in prediction_to_truth_orf.items()}
    if set(truth_order) != set(truth_to_prediction):
        raise ValueError("Prepared ORF mapping does not cover the complete localization truth set")
    probabilities = [predictions[truth_to_prediction[orf_id]] for orf_id in truth_order]
    labels = [truth_values[orf_id] for orf_id in truth_order]
    metrics = compute_gene_metrics(
        torch.tensor(probabilities, dtype=torch.float32).unsqueeze(0),
        torch.tensor(labels, dtype=torch.float32).unsqueeze(0), threshold=threshold,
    )
    predicted = np.asarray(probabilities) >= float(threshold)
    positive = np.asarray(labels) > 0.5
    counts = {
        "genomes": len({base_genome_id(str(sample.get("genome_id"))) for sample in truth_samples}),
        "fragments": len(truth_samples), "orfs": len(labels),
        "tp": int(np.logical_and(predicted, positive).sum()),
        "fp": int(np.logical_and(predicted, np.logical_not(positive)).sum()),
        "fn": int(np.logical_and(np.logical_not(predicted), positive).sum()),
    }
    return metrics, counts


def evaluate_deployed_localization(
    *,
    config: str,
    checkpoint: str,
    faa_dir: str,
    truth_file: str,
    esmc_checkpoint: str,
    work_dir: str,
    extract_features: Callable[..., Tuple[Any, Any]],
    converter: Callable[[str, str], str],
    run_prediction: Callable[..., Dict[str, str]],
    device: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_orfs: int = 128,
    toks_per_batch: int = 2560,
    truncation_seq_length: int = 2048,
    threshold: float = 0.3,
    overlap_ratio: float = 0.5,
    type_threshold: float = 0.5,
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, Any]]:
    """Run the active engine's deployed protein-to-prediction localization path."""
    work_path = Path(work_dir).expanduser().resolve()
    work_path.mkdir(parents=True, exist_ok=True)
    prepared = prepare_localization_dataset(
        faa_dir=faa_dir,
        truth_file=truth_file,
        output_json=str(work_path / "prediction_dataset.json"),
        converter=converter,
    )
    features_manager, _ = extract_features(
        input_json=prepared["dataset_json"],
        device=str(device),
        model_path=str(esmc_checkpoint),
        toks_per_batch=int(toks_per_batch),
        truncation_seq_length=int(truncation_seq_length),
        progress_name="ESMC localization end-to-end",
    )
    prediction_dir = work_path / "deployed_prediction"
    prediction_outputs = run_prediction(
        test_file=prepared["dataset_json"],
        features_manager=features_manager,
        config=str(config),
        resume=str(checkpoint),
        output_dir=str(prediction_dir),
        max_orfs=int(max_orfs),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        threshold=float(threshold),
        overlap_ratio=float(overlap_ratio),
        bgc_type_threshold=float(type_threshold),
        save_orf_txt=True,
        device=str(device),
        progress_name="Localization deployed predictor",
    )
    prediction_file = prediction_outputs.get("orf_txt_path")
    if not prediction_file:
        raise RuntimeError("Deployed predictor did not return an ORF prediction artifact")
    metrics, counts = score_localization_output(
        prediction_file=prediction_file,
        truth_file=truth_file,
        prediction_to_truth_orf=prepared["prediction_to_truth_orf"],
        threshold=threshold,
    )
    provenance = {
        "dataset_json": prepared["dataset_json"],
        "prepared": {key: prepared[key] for key in ("genomes", "fragments", "orfs")},
        "features": features_manager.get_stats(),
        "prediction_outputs": {key: str(value) for key, value in prediction_outputs.items()},
    }
    return metrics, counts, provenance


def _positive_runs(labels) -> List[Tuple[int, int]]:
    return _segments_from_binary_labels([float(value) for value in labels])



class _InMemoryRegionBGCDataset(Dataset):
    """Type-test dataset backed only by online, in-memory ESMC features."""

    def __init__(self, data_file: str, features_manager, max_orfs: int = 128):
        with open(data_file, "r", encoding="utf-8") as handle:
            self.data = json.load(handle)
        if not isinstance(self.data, list):
            raise ValueError("Type-test JSON must contain a list of samples")
        self.features_manager = features_manager
        self.embedding_dim = int(features_manager.embedding_dim)
        self.max_orfs = int(max_orfs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        genome_id = str(sample.get("original_genome_id") or sample.get("genome_id"))
        sequences = sample.get("orf_sequences", {})
        orf_ids = sample_orf_ids(sample)[: self.max_orfs]
        sequence_length = len(orf_ids)
        features = np.empty((self.embedding_dim, sequence_length), dtype=np.float32)
        missing = []
        for position, orf_id in enumerate(orf_ids):
            vector = self.features_manager.get_feature(orf_id, genome_id, copy=False)
            if vector is None:
                missing.append(f"{genome_id}_{orf_id}")
            else:
                features[:, position] = np.asarray(vector, dtype=np.float32)
        if missing:
            raise KeyError(f"Missing {len(missing)} online ORF features; examples: {missing[:10]}")

        labels = parse_binary_labels(
            sample['bgc_labels'], len(sequences), context=f"Sample {genome_id}"
        )[:sequence_length]
        intervals = []
        for start, end in _positive_runs(labels):
            x1, x2 = start / max(sequence_length, 1), end / max(sequence_length, 1)
            intervals.append([(x1 + x2) / 2.0, x2 - x1])
        interval_tensor = (
            torch.tensor(intervals, dtype=torch.float32)
            if intervals else torch.zeros((0, 2), dtype=torch.float32)
        )
        type_vector = _bgc_multi_hot_from_sample(sample)
        type_targets = type_vector.unsqueeze(0).repeat(len(intervals), 1)
        target = {"intervals": interval_tensor,
                  "labels": torch.zeros(len(intervals), dtype=torch.int64),
                  "bgc_type_vec": type_targets}
        return torch.from_numpy(features), target


def _load_model(config: str, checkpoint: str, device: torch.device):
    cfg = YAMLConfig(config, resume=None)
    model = cfg.model
    _, weight_source = load_evaluation_checkpoint(model, checkpoint)
    return model.to(device).eval(), cfg, weight_source


@torch.no_grad()
def evaluate_product_types(
    *,
    config: str,
    checkpoint: str,
    data_file: str,
    features_manager,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
    max_orfs: int = 128,
    type_threshold: float = 0.5,
) -> Tuple[Dict[str, float], Dict[str, Any], str]:
    """Classify matched pairs using the training Hungarian matcher."""
    model, cfg, weight_source = _load_model(config, checkpoint, device)
    matcher = cfg.criterion.matcher.to(device).eval()
    dataset = _InMemoryRegionBGCDataset(
        data_file=data_file, features_manager=features_manager, max_orfs=max_orfs,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda", persistent_workers=num_workers > 0,
        drop_last=False, collate_fn=bgc_collate_fn,
    )
    matched_logits: List[torch.Tensor] = []
    matched_targets: List[torch.Tensor] = []
    sample_count = total_gt = 0
    for samples, targets in loader:
        samples = samples.to(device)
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        outputs = model(samples)
        required = {"pred_bgc_types", "pred_intervals", "pred_logits"}
        if not required.issubset(outputs):
            raise KeyError(f"Model output is missing: {sorted(required - set(outputs))}")
        assignments = match_queries_by_hungarian(matcher, outputs, targets)
        sample_count += len(targets)
        for batch_index, matches in enumerate(assignments):
            total_gt += len(matches)
            for gt_index, query_index in enumerate(matches):
                if query_index is None:
                    continue
                matched_logits.append(outputs["pred_bgc_types"][batch_index, query_index].cpu())
                matched_targets.append(targets[batch_index]["bgc_type_vec"][gt_index].cpu())

    if not matched_logits:
        raise RuntimeError("Product-type evaluation found no matched BGC regions")
    metrics = compute_multilabel_micro_metrics(
        torch.stack(matched_logits), torch.stack(matched_targets), threshold=type_threshold,
    )
    matched = len(matched_logits)
    counts = {"samples": sample_count, "regions": total_gt, "matched": matched,
              "unmatched": total_gt - matched, "coverage": matched / total_gt}
    return metrics, counts, weight_source


def evaluate_region_output(*, prediction_file: str, truth_file: str):
    """Evaluate saved ORF predictions in nucleotide coordinates, aligned by ID."""
    def normalize_id(value):
        value = str(value).strip().split(":", 1)[-1].split("__")[-1]
        return value[len("genome."):] if value.startswith("genome.") else value
    expected, order_by_sample, truth, positions = {}, {}, {}, {}
    for index, sample in enumerate(_load_labeled_samples(truth_file)):
        sample_id = f"{index}:{sample.get('genome_id', '')}"
        ids = [normalize_id(value) for value in sample_orf_ids(sample)]
        labels = _sample_labels(sample, len(ids))
        if any(value not in (0, 1) for value in labels):
            raise ValueError(f"{sample_id}: truth labels must be binary")
        coordinates = sample.get('orf_positions')
        if isinstance(coordinates, str):
            coordinates = json.loads(coordinates)
        if not isinstance(coordinates, list) or len(coordinates) != len(ids):
            raise ValueError(f"{sample_id}: missing or mismatched orf_positions")
        for orf_id, label in zip(ids, labels):
            if not orf_id or orf_id in expected:
                raise ValueError(f"Duplicate or empty truth ORF ID: {orf_id}")
            expected[orf_id] = label
        order_by_sample[sample_id], truth[sample_id], positions[sample_id] = ids, labels, coordinates
    if not expected:
        raise ValueError("No ORFs found in the truth dataset")
    predicted = {}
    with open(prediction_file, encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        fields = set(reader.fieldnames or [])
        prediction_field = next((field for field in ('Pred', 'Pred01', 'Correct01') if field in fields), None)
        if 'ORF_ID' not in fields or prediction_field is None:
            raise ValueError("Prediction TSV requires ORF_ID and Pred columns")
        for row in reader:
            orf_id = normalize_id(row['ORF_ID'])
            if not orf_id or orf_id in predicted:
                raise ValueError(f"Duplicate or empty predicted ORF ID: {orf_id}")
            value = float(row[prediction_field])
            if value not in (0, 1):
                raise ValueError(f"Non-binary prediction for {orf_id}")
            if 'True' in fields and (orf_id not in expected or float(row['True']) != expected[orf_id]):
                raise ValueError(f"Prediction True label differs from JSON for {orf_id}")
            predicted[orf_id] = int(value)
    if set(predicted) != set(expected):
        raise ValueError(f"ORF coverage mismatch: missing={len(set(expected) - set(predicted))}, "
                         f"extra={len(set(predicted) - set(expected))}")
    predictions = {sample_id: [predicted[orf_id] for orf_id in ids]
                   for sample_id, ids in order_by_sample.items()}
    metrics = compute_bp_region_metrics(truth, predictions, positions)
    return metrics, {"samples": len(truth), "orfs": len(expected)}
