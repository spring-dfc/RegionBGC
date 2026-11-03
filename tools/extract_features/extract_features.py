#!/usr/bin/env python
"""Extract ESMC embeddings into a training-compatible HDF5 file."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict.feature_extractor import DEFAULT_ESMC_MODEL_PATH, load_esmc_model
from src.utils.bgc_data import sample_orf_ids


OrfRecord = Tuple[str, str]


def resolve_device(device: str) -> str:
    text = str(device).strip().lower()
    if text in ("", "auto"):
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if text == "cpu":
        return text
    if text == "cuda":
        text = "cuda:0"
    if text.startswith("cuda:"):
        index = int(text.split(":", 1)[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} is not available")
        return text
    raise ValueError(f"Unsupported device: {device}")


def load_samples(input_file: Path) -> List[Dict]:
    with input_file.open(encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list):
        raise ValueError("Input JSON must contain a list of samples")
    return samples


def iter_unique_orfs(samples: Iterable[Dict]) -> Iterator[OrfRecord]:
    """Yield unique ``(composite_id, sequence)`` records."""
    seen = set()
    for sample in samples:
        genome_id = str(sample.get("original_genome_id") or sample.get("genome_id"))
        sequences = sample.get("orf_sequences", {})

        orf_ids = sample_orf_ids(sample)
        if isinstance(sequences, dict):
            items = ((orf_id, sequences[orf_id]) for orf_id in orf_ids)
        else:
            items = zip(orf_ids, sequences)

        for orf_id, sequence in items:
            if not isinstance(sequence, str) or not sequence:
                raise ValueError(f"Genome {genome_id}, ORF {orf_id}: invalid protein sequence")
            feature_id = f"{genome_id}_{orf_id}"
            if feature_id in seen:
                continue
            seen.add(feature_id)
            yield feature_id, sequence


def iter_token_batches(
    records: Iterable[OrfRecord],
    max_tokens: int,
    truncation_length: int,
) -> Iterator[List[OrfRecord]]:
    """Group sequences by approximate token count without reordering them."""
    batch: List[OrfRecord] = []
    token_count = 0
    for feature_id, sequence in records:
        sequence = sequence[:truncation_length]
        cost = len(sequence) + 2
        if batch and token_count + cost > max_tokens:
            yield batch
            batch = []
            token_count = 0
        batch.append((feature_id, sequence))
        token_count += cost
    if batch:
        yield batch


def encode_batch(model, batch: Sequence[OrfRecord]) -> Tuple[List[str], np.ndarray]:
    ids = [feature_id for feature_id, _ in batch]
    sequences = [sequence for _, sequence in batch]
    with torch.inference_mode():
        tokens = model._tokenize(sequences)
        embeddings = model.forward(sequence_tokens=tokens).embeddings
        pooled = [
            embeddings[index, 1 : len(sequence) + 1].mean(dim=0).float().cpu().numpy()
            for index, sequence in enumerate(sequences)
        ]
    return ids, np.stack(pooled).astype(np.float32, copy=False)


def append_rows(dataset: h5py.Dataset, values: Sequence) -> None:
    start = dataset.shape[0]
    dataset.resize(start + len(values), axis=0)
    dataset[start:] = values


def extract_features(
    input_file: Path,
    output_file: Path,
    model_path: Path | None,
    device: str,
    max_tokens: int,
    truncation_length: int,
    compression: str | None,
    chunk_rows: int,
) -> int:
    samples = load_samples(input_file)
    runtime_device = resolve_device(device)
    model = load_esmc_model(device=runtime_device, model_path=str(model_path) if model_path else None)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary_file.unlink(missing_ok=True)
    count = 0

    try:
        with h5py.File(temporary_file, "w") as handle:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            ids_dataset = handle.create_dataset(
                "ids", shape=(0,), maxshape=(None,), dtype=string_dtype,
                chunks=(chunk_rows,), compression=compression,
            )
            features_dataset = None
            metadata = handle.create_group("metadata")

            batches = iter_token_batches(
                iter_unique_orfs(samples),
                max_tokens=max_tokens,
                truncation_length=truncation_length,
            )
            for batch in tqdm(batches, desc="Extracting ESMC features", unit="batch"):
                feature_ids, features = encode_batch(model, batch)
                if features_dataset is None:
                    embedding_dim = int(features.shape[1])
                    features_dataset = handle.create_dataset(
                        "features",
                        shape=(0, embedding_dim),
                        maxshape=(None, embedding_dim),
                        dtype=np.float32,
                        chunks=(chunk_rows, embedding_dim),
                        compression=compression,
                    )
                    metadata.attrs["embedding_dim"] = embedding_dim
                append_rows(ids_dataset, feature_ids)
                append_rows(features_dataset, features)
                count += len(feature_ids)

            if features_dataset is None:
                raise ValueError("No valid ORF sequences were found")
            metadata.attrs["total_features"] = count
            metadata.attrs["model"] = "esmc_300m"
            metadata.attrs["truncation_length"] = truncation_length

        os.replace(temporary_file, output_file)
    except Exception:
        temporary_file.unlink(missing_ok=True)
        raise
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return count


def build_parser() -> argparse.ArgumentParser:
    default_model = DEFAULT_ESMC_MODEL_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", "--input_file", dest="input_file", required=True)
    parser.add_argument("--output-file", "--output_file", dest="output_file", required=True)
    parser.add_argument(
        "--model-path", "--esmc_model_path", dest="model_path",
        default=str(default_model) if default_model.is_file() else None,
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--toks-per-batch", "--toks_per_batch", dest="max_tokens", type=int, default=2560)
    parser.add_argument(
        "--truncation-length", "--truncation_seq_length",
        dest="truncation_length", type=int, default=2048,
    )
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="none")
    parser.add_argument("--chunk-rows", type=int, default=1024)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_file = Path(args.input_file).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else None

    if not input_file.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_file}")
    if model_path is not None and not model_path.is_file():
        raise FileNotFoundError(f"ESMC model not found: {model_path}")
    if args.max_tokens <= 0 or args.truncation_length <= 0 or args.chunk_rows <= 0:
        raise ValueError("Batch, truncation, and chunk sizes must be positive")

    compression = None if args.compression == "none" else args.compression
    count = extract_features(
        input_file=input_file,
        output_file=output_file,
        model_path=model_path,
        device=args.device,
        max_tokens=args.max_tokens,
        truncation_length=args.truncation_length,
        compression=compression,
        chunk_rows=args.chunk_rows,
    )
    print(f"Saved {count} features to {output_file}")


if __name__ == "__main__":
    main()
