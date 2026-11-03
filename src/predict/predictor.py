from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DETECTION_THRESHOLD = 0.3
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.predict.dataset import PredictionDataset, collate_prediction_batch
from src.utils.bgc_data import sample_orf_ids
from src.utils.bgc_inference import probability_threshold
from src.core import YAMLConfig
from src.models.loading import load_evaluation_checkpoint
from src.utils.bgc_region_consensus import build_genome_consensus, update_genome_context


def _yaml_dict(cfg: YAMLConfig) -> Dict:
    value = getattr(cfg, "yaml_cfg", {})
    return value if isinstance(value, dict) else {}


def _resolve_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_DIR / path).resolve())


def _parse_device(device_text: Optional[str]) -> torch.device:
    if device_text is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = str(device_text).strip().lower()
    if text == "cpu":
        return torch.device("cpu")
    if text == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if text.startswith("cuda:"):
        idx = int(text.split(":", 1)[1])
        if torch.cuda.is_available() and 0 <= idx < torch.cuda.device_count():
            torch.cuda.set_device(idx)
            return torch.device(f"cuda:{idx}")
        raise RuntimeError(f"CUDA device {idx} is not available.")
    if text.isdigit():
        idx = int(text)
        if torch.cuda.is_available() and 0 <= idx < torch.cuda.device_count():
            torch.cuda.set_device(idx)
            return torch.device(f"cuda:{idx}")
        raise RuntimeError(f"CUDA device {idx} is not available.")
    raise ValueError(f"Unsupported device: {device_text}")


def _find_checkpoint(cfg: YAMLConfig, resume: str) -> str:
    path = _resolve_path(resume)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Specify an existing checkpoint file: {path}")
    return path


def _resolve_eval_thresholds(cfg: YAMLConfig, threshold: Optional[float], overlap_ratio: Optional[float]) -> Tuple[float, float]:
    yaml_dict = _yaml_dict(cfg)
    bgc_eval_cfg = yaml_dict.get("bgc_eval", {}) if isinstance(yaml_dict, dict) else {}

    resolved_threshold = threshold
    resolved_overlap = overlap_ratio

    if resolved_threshold is None and isinstance(bgc_eval_cfg, dict):
        resolved_threshold = bgc_eval_cfg.get("threshold")
    if resolved_overlap is None and isinstance(bgc_eval_cfg, dict):
        resolved_overlap = bgc_eval_cfg.get("overlap_ratio")

    if resolved_threshold is None:
        resolved_threshold = DEFAULT_DETECTION_THRESHOLD
    if resolved_overlap is None:
        resolved_overlap = 0.5
    resolved_threshold = probability_threshold(resolved_threshold)
    resolved_overlap = probability_threshold(resolved_overlap)
    if resolved_overlap == 1:
        raise ValueError("overlap_ratio must be less than 1")
    return resolved_threshold, resolved_overlap


@contextmanager
def _fp32_prediction():
    """Keep ESMC's acceleration settings out of RegionBGC prediction."""
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled,
            benchmark=False, deterministic=True, allow_tf32=False,
        ):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def evaluate_model(
    model,
    postprocessor,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float,
    bgc_type_threshold: float,
) -> Dict:
    model.eval()
    context_by_genome: Dict[str, Dict] = {}
    det_segments_by_genome: Dict[str, List[Dict]] = {}

    with torch.no_grad(), _fp32_prediction():
        for batch in dataloader:
            features = batch["features"].to(device)
            padding_mask = batch["padding_mask"]
            genome_ids = batch["genome_id"]
            original_ids = batch["original_genome_id"]
            seg_starts = batch["segment_start"]
            seg_ends = batch["segment_end"]

            if padding_mask.dtype == torch.bool:
                sequence_lengths = padding_mask.sum(dim=1).to(torch.int64)
            else:
                sequence_lengths = torch.full((features.shape[0],), features.shape[1], dtype=torch.int64)

            length_groups: Dict[int, List[int]] = {}
            for idx in range(int(features.shape[0])):
                sequence_length = int(sequence_lengths[idx].item())
                if sequence_length > 0:
                    length_groups.setdefault(sequence_length, []).append(idx)

            for sequence_length, indices in length_groups.items():
                index_tensor = torch.as_tensor(indices, dtype=torch.long, device=features.device)
                group_features = features.index_select(0, index_tensor)[:, :sequence_length, :]
                model_inputs = group_features.permute(0, 2, 1).contiguous()
                length_tensor = torch.full(
                    (group_features.shape[0],), sequence_length, dtype=torch.int64, device=device
                )

                outputs = model(model_inputs)
                results = postprocessor(outputs, length_tensor)

                for local_idx, result in enumerate(results):
                    batch_idx = indices[local_idx]
                    valid_length = int(sequence_length)
                    gid = str(original_ids[batch_idx])
                    update_genome_context(
                        context_by_genome,
                        gid,
                        total_length=batch["total_orfs_in_genome"][batch_idx],
                        segment_start=seg_starts[batch_idx],
                        orf_ids=batch["orf_ids"][batch_idx][:valid_length],
                    )

                    segments = result.get("segments")
                    scores = result.get("scores")
                    bgc_probs = result.get("bgc_type_probs")
                    if segments is None or scores is None:
                        continue

                    base_start = seg_starts[batch_idx] if seg_starts[batch_idx] is not None else 0
                    for j in range(int(segments.shape[0])):
                        score_value = float(scores[j].item())
                        start = int(base_start + segments[j, 0].item())
                        end = int(base_start + segments[j, 1].item())
                        if end <= start:
                            continue
                        det = {"start": start, "end": end, "score": score_value}
                        if bgc_probs is not None and bgc_probs.ndim == 2 and j < bgc_probs.shape[0]:
                            det["bgc_type_probs"] = [float(x) for x in bgc_probs[j].detach().cpu().tolist()]
                        det_segments_by_genome.setdefault(gid, []).append(det)

    return build_genome_consensus(
        det_segments_by_genome,
        context_by_genome,
        threshold=threshold,
        bgc_type_threshold=bgc_type_threshold,
    )


def build_orf_position_map(json_path: str) -> Dict[str, Dict[str, Tuple[int, int]]]:
    mapping: Dict[str, Dict[str, Tuple[int, int]]] = {}
    with open(json_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    for sample in raw:
        genome_id = str(sample.get("original_genome_id") or sample.get("genome_id"))
        positions = sample.get("orf_positions")
        if isinstance(positions, str):
            try:
                positions = json.loads(positions)
            except Exception:
                positions = []
        if not isinstance(positions, list):
            continue

        orf_ids = sample_orf_ids(sample)

        genome_map = mapping.setdefault(genome_id, {})
        for orf_id, pos in zip(orf_ids, positions):
            if not isinstance(pos, (list, tuple)) or len(pos) != 2:
                continue
            try:
                genome_map[str(orf_id)] = (int(pos[0]), int(pos[1]))
            except Exception:
                continue
        if genome_map:
            mapping[genome_id] = genome_map
    return mapping


def save_orf_predictions(fused: Dict[str, Dict], output_dir: str, threshold: float) -> str:
    path = os.path.join(output_dir, "orf_predictions.tsv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("ORF_ID\tProb\tPred\n")
        for genome_id, item in fused.items():
            probs = np.asarray(item.get("predictions", []), dtype=float)
            orf_ids = item.get("orf_ids", [f"{genome_id}:ORF_{i+1:06d}" for i in range(len(probs))])
            for orf_id, prob in zip(orf_ids, probs.tolist()):
                name = str(orf_id).split(":", 1)[-1]
                pred = 1 if float(prob) >= threshold else 0
                fh.write(f"{name}\t{float(prob):.6f}\t{pred}\n")
    return path


def save_segments_csv(
    fused: Dict[str, Dict],
    csv_path: str,
    orf_pos_map: Dict[str, Dict[str, Tuple[int, int]]],
) -> str:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("genome,class_name,confidence,start_cds,end_cds,cds_ids,bgc_range,cds_len,nt_len\n")

        for genome_id, item in fused.items():
            orf_ids = item.get("orf_ids", [])
            genome_pos_map = orf_pos_map.get(str(genome_id), {})
            final_regions = item.get("final_regions", [])

            for region in final_regions:
                start = int(region.get("start", 0))
                end_exclusive = int(region.get("end", 0))
                if end_exclusive <= start:
                    continue
                region_orf_ids = orf_ids[start:end_exclusive]
                confidence = float(region.get("confidence", 0.0))
                class_name = str(region.get("class_name", "none"))

                nt_starts = []
                nt_ends = []
                for orf_id in region_orf_ids:
                    pos = genome_pos_map.get(str(orf_id))
                    if pos:
                        nt_starts.append(int(pos[0]))
                        nt_ends.append(int(pos[1]))

                bgc_range = ""
                nt_len = ""
                if nt_starts and nt_ends:
                    nt_start = min(nt_starts)
                    nt_end = max(nt_ends)
                    bgc_range = f"{genome_id}:{nt_start}-{nt_end}"
                    nt_len = str(nt_end - nt_start + 1)

                start_cds = start + 1
                end_cds = end_exclusive
                cds_len = end_cds - start_cds + 1
                cds_ids = " ".join(str(x) for x in region_orf_ids)
                row = [
                    str(genome_id),
                    class_name,
                    f"{confidence:.6f}",
                    str(start_cds),
                    str(end_cds),
                    cds_ids,
                    bgc_range,
                    str(cds_len),
                    nt_len,
                ]
                fh.write(",".join(row) + "\n")
    return csv_path


def load_prediction_runtime(config: str, resume: str, device: Optional[str] = None) -> Dict:
    """Load RegionBGC once for reuse across genomes in one worker."""
    config = _resolve_path(config)
    cfg = YAMLConfig(config, resume=None)
    runtime_device = _parse_device(device)
    model = cfg.model
    postprocessor = cfg.postprocessor
    checkpoint_path = _find_checkpoint(cfg, resume)
    load_evaluation_checkpoint(model, checkpoint_path)
    model = model.to(device=runtime_device, dtype=torch.float32).eval()
    return {
        "cfg": cfg, "model": model, "postprocessor": postprocessor,
        "checkpoint_path": checkpoint_path, "device": runtime_device,
    }


def run_prediction(
    test_file: str,
    features_manager,
    config: str,
    resume: str,
    output_dir: str,
    max_orfs: int = 128,
    batch_size: int = 32,
    num_workers: int = 0,
    threshold: Optional[float] = None,
    overlap_ratio: Optional[float] = None,
    bgc_type_threshold: float = 0.5,
    save_orf_txt: bool = True,
    device: Optional[str] = None,
    progress_name: Optional[str] = None,
    runtime: Optional[Dict] = None,
) -> Dict[str, str]:
    test_file = _resolve_path(test_file)
    config = _resolve_path(config)
    output_dir = _resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if runtime is None:
        runtime = load_prediction_runtime(config=config, resume=resume, device=device)
    bgc_type_threshold = probability_threshold(bgc_type_threshold)
    cfg = runtime["cfg"]
    threshold, overlap_ratio = _resolve_eval_thresholds(cfg, threshold, overlap_ratio)
    runtime_device = runtime["device"]
    logging.basicConfig(level=logging.INFO, format="[RegionBGC] %(message)s")
    logging.info(
        "threshold=%.3f overlap_ratio=%.2f batch_size=%d device=%s",
        threshold,
        overlap_ratio,
        batch_size,
        runtime_device,
    )
    if progress_name:
        logging.info("prediction=%s", progress_name)

    model = runtime["model"]
    postprocessor = runtime["postprocessor"]
    checkpoint_path = runtime["checkpoint_path"]

    dataset = PredictionDataset(
        data_file=test_file,
        features_manager=features_manager,
        max_orfs=max_orfs,
        overlap_ratio=overlap_ratio,
        use_sliding_window=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_prediction_batch,
    )

    fused = evaluate_model(
        model=model,
        postprocessor=postprocessor,
        dataloader=dataloader,
        device=runtime_device,
        threshold=threshold,
        bgc_type_threshold=bgc_type_threshold,
    )

    orf_pos_map = build_orf_position_map(test_file)
    csv_path = save_segments_csv(
        fused=fused,
        csv_path=os.path.join(output_dir, "bgc_regions.csv"),
        orf_pos_map=orf_pos_map,
    )

    outputs = {
        "checkpoint_path": checkpoint_path,
        "csv_path": csv_path,
    }
    if save_orf_txt:
        outputs["orf_txt_path"] = save_orf_predictions(fused, output_dir, threshold)
    return outputs


def main() -> None:
    raise SystemExit(
        "predictor.py requires an in-memory feature manager. "
        "Please run tools/predict.py instead."
    )


if __name__ == "__main__":
    main()
