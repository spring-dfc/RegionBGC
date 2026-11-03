from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.predict.memory_features import InMemoryFeaturesManager
from src.utils.bgc_data import sample_orf_ids

DEFAULT_ESMC_MODEL_PATH = REPO_DIR / "weights" / "esmc_300m_2024_12_v0.pth"


def _resolve_device(device: Optional[str], gpu_id: Optional[int]) -> str:
    if gpu_id is not None:
        if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {gpu_id} is not available.")
        return f"cuda:{gpu_id}"
    text = str(device).strip().lower() if device is not None else "auto"
    if text in ("", "auto"):
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if text == "cpu":
        return "cpu"
    if text == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return "cuda:0"
    if text.startswith("cuda:"):
        index = int(text.split(":", 1)[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} is not available.")
        return text
    raise ValueError(f"Unsupported device: {device}")


def _load_samples(input_json: str) -> List[Dict]:
    with open(input_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list")
    return data


def _extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint
    raise TypeError(f"Unsupported ESMC checkpoint format: {type(checkpoint)}")


def _iter_unique_orfs(samples: Iterable[Dict]) -> Iterable[Tuple[str, str]]:
    seen = set()
    for sample in samples:
        genome_id = str(sample.get("original_genome_id") or sample.get("genome_id"))
        orf_sequences = sample.get("orf_sequences", {})

        orf_ids = sample_orf_ids(sample)
        if isinstance(orf_sequences, dict):
            items = ((orf_id, orf_sequences[orf_id]) for orf_id in orf_ids)
        else:
            items = zip(orf_ids, orf_sequences)

        for orf_id, sequence in items:
            if not isinstance(sequence, str) or not sequence:
                raise ValueError(f"Genome {genome_id}, ORF {orf_id}: invalid protein sequence")
            composite_id = f"{genome_id}_{orf_id}"
            if composite_id in seen:
                continue
            seen.add(composite_id)
            yield composite_id, sequence


def load_esmc_model(device: str, model_path: Optional[str] = None):
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers

    if "cuda" in device:
        gpu_idx = int(device.split(":")[-1]) if ":" in device else 0
        torch.cuda.set_device(gpu_idx)
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    resolved_model_path: Optional[Path] = None
    if model_path:
        candidate = Path(model_path).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"ESMC model not found: {candidate}")
        resolved_model_path = candidate
    elif DEFAULT_ESMC_MODEL_PATH.exists():
        resolved_model_path = DEFAULT_ESMC_MODEL_PATH

    if resolved_model_path is not None:
        with torch.device("cpu"):
            model = ESMC(
                d_model=960,
                n_heads=15,
                n_layers=30,
                tokenizer=get_esmc_model_tokenizers(),
            ).eval()
        checkpoint = torch.load(str(resolved_model_path), map_location="cpu", weights_only=True)
        state_dict = _extract_state_dict(checkpoint)
        model.load_state_dict(state_dict)
    else:
        os.environ["INFRA_PROVIDER"] = "True"
        model = ESMC.from_pretrained("esmc_300m")

    if "cuda" in device:
        # Match historical ESMC.from_pretrained() GPU loading: BF16, then FP16.
        model = model.bfloat16().half().to(device)
    else:
        model = model.to(device)

    model.eval()
    return model


def _build_token_batches(lengths: List[int], max_tokens: int) -> List[List[int]]:
    batches: List[List[int]] = []
    current_batch: List[int] = []
    current_tokens = 0

    for idx, seq_len in enumerate(lengths):
        cost = seq_len + 2
        if current_batch and current_tokens + cost > max_tokens:
            batches.append(current_batch)
            current_batch = [idx]
            current_tokens = cost
        else:
            current_batch.append(idx)
            current_tokens += cost

    if current_batch:
        batches.append(current_batch)
    return batches


def _extract_esmc_embeddings(model, sequence_tokens: torch.Tensor) -> torch.Tensor:
    """Run the ESMC embedding path without collecting unused layer outputs."""
    sequence_id = sequence_tokens != model.tokenizer.pad_token_id
    x = model.embed(sequence_tokens)
    batch_size, padded_length = x.shape[:2]
    indices = None
    if model._use_flash_attn:
        from esm.models.esmc import pad_input, unpad_input

        x, indices, *_ = unpad_input(x, sequence_id)

    # Preserve TransformerStack.forward's inputs, block order and final norm.
    chain_id = torch.ones(x.shape[:-1], dtype=torch.int64, device=x.device)
    for block in model.transformer.blocks:
        x = block(x, sequence_id, None, None, chain_id)
    x = model.transformer.norm(x)

    if model._use_flash_attn:
        x = pad_input(x, indices, batch_size, padded_length)
    return x


def extract_features_in_memory(
    input_json: str,
    device: Optional[str] = None,
    gpu_id: Optional[int] = None,
    model_path: Optional[str] = None,
    toks_per_batch: int = 2560,
    truncation_seq_length: int = 2048,
    progress_name: Optional[str] = None,
    model=None,
) -> Tuple[InMemoryFeaturesManager, Dict[str, np.ndarray]]:
    input_path = Path(input_json).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    samples = _load_samples(str(input_path))
    unique_orfs = list(_iter_unique_orfs(samples))
    if not unique_orfs:
        raise ValueError("No ORF sequences found in input JSON")

    runtime_device = _resolve_device(device=device, gpu_id=gpu_id)
    owns_model = model is None
    if owns_model:
        model = load_esmc_model(runtime_device, model_path=model_path)
    features_dict: Dict[str, np.ndarray] = {}

    ids = [item[0] for item in unique_orfs]
    sequences = [item[1][:truncation_seq_length] for item in unique_orfs]
    lengths = [len(seq) for seq in sequences]
    batches = _build_token_batches(lengths, max_tokens=toks_per_batch)
    desc = progress_name or f"ESMC {input_path.stem}"

    try:
        for batch_indices in tqdm(batches, desc=desc, unit="batch"):
            batch_sequences = [sequences[idx] for idx in batch_indices]
            batch_ids = [ids[idx] for idx in batch_indices]
            with torch.no_grad():
                token_batch = model._tokenize(batch_sequences)
                embeddings = _extract_esmc_embeddings(model, token_batch)
                pooled_features = []
                for i in range(len(batch_ids)):
                    seq_len = len(batch_sequences[i])
                    pooled = embeddings[i, 1 : 1 + seq_len, :].mean(dim=0)
                    pooled_features.append(pooled.float())
                host_features = torch.stack(pooled_features).cpu().numpy()
                features_dict.update(zip(batch_ids, host_features))
    finally:
        # Worker-owned models intentionally remain resident for the next genome.
        if owns_model:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return InMemoryFeaturesManager(features_dict), features_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract ESMC features in memory")
    parser.add_argument("--input_json", required=True, help="Prediction dataset JSON path")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--model_path", type=str, default=None, help="Optional ESMC model path")
    parser.add_argument("--toks_per_batch", type=int, default=2560)
    parser.add_argument("--truncation_seq_length", type=int, default=2048)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manager, _ = extract_features_in_memory(
        input_json=args.input_json,
        device=args.device,
        gpu_id=args.gpu_id,
        model_path=args.model_path,
        toks_per_batch=args.toks_per_batch,
        truncation_seq_length=args.truncation_seq_length,
        progress_name=f"ESMC {Path(args.input_json).stem}",
    )
    stats = manager.get_stats()
    print(
        f"In-memory features ready: total={stats['total_features']} "
        f"dim={stats['embedding_dim']} mem={stats['memory_usage_mb']:.2f}MB"
    )


if __name__ == "__main__":
    main()
