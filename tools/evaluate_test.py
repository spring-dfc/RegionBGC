"""Evaluate user-supplied labeled ORF datasets with the active interval model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from numbers import Real
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict.feature_extractor import extract_features_in_memory
from src.predict.predictor import run_prediction
from src.utils.bgc_inference import probability_threshold
from src.evaluation import evaluate_deployed_localization, evaluate_product_types, evaluate_region_output

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "regionbgc" / "regionbgc.yml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "weights" / "regionbgc.pth"
DEFAULT_ESMC = PROJECT_ROOT / "weights" / "esmc_300m_2024_12_v0.pth"


def report_metrics(metrics: dict) -> dict:
    reported = {
        key: round(float(value), 3)
        if isinstance(value, Real) and not isinstance(value, bool)
        else value
        for key, value in metrics.items()
    }
    for name in ("precision", "recall", "f1", "auroc"):
        print(f"{name}: {float(reported.get(name, 0.0)):.3f}")
    return reported


def _device(value: str) -> torch.device:
    text = str(value).strip().lower()
    if text.isdigit():
        text = f"cuda:{text}"
    device = torch.device(text)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        index = device.index or 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} is not available")
        torch.cuda.set_device(index)
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Protein-level end-to-end RegionBGC evaluation")
    parser.add_argument("task", choices=["localization", "type", "regions"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data", default=None, help="Labeled protein JSON for localization or product classification")
    parser.add_argument("--truth-data", default=None, help="Truth JSON for evaluating an existing prediction TSV")
    parser.add_argument("--predictions", help="ORF prediction TXT/TSV for nucleotide-region evaluation")
    parser.add_argument("--esmc-checkpoint", default=str(DEFAULT_ESMC))
    parser.add_argument("--toks-per-batch", type=int, default=2560)
    parser.add_argument("--truncation-seq-length", type=int, default=2048)
    parser.add_argument("--output-json")
    parser.add_argument("--work-dir", help="Directory for native deployed prediction artifacts")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-orfs", type=int, default=128)
    parser.add_argument("--threshold", type=probability_threshold, default=0.3)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--type-threshold", type=probability_threshold, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.task == "regions":
        if not args.predictions or not args.truth_data:
            raise ValueError("Region evaluation requires --predictions and --truth-data")
        metrics, counts = evaluate_region_output(
            prediction_file=args.predictions, truth_file=args.truth_data,
        )
        print(f"Counts: {counts}")
        for cutoff, values in metrics.items():
            print(f"Nucleotide regions (IoU={cutoff}): " + " | ".join(
                f"{key}={values[key]:.3f}" for key in ('precision', 'recall', 'f1', 'avg_iou')))
        if args.output_json:
            output = Path(args.output_json).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "task": "regions", "coordinate_unit": "nucleotide",
                "evaluation_protocol": "bp_iou_greedy_one_to_one",
                "prediction_file": str(Path(args.predictions).resolve()),
                "truth_data": str(Path(args.truth_data).resolve()),
                "counts": counts, "metrics": metrics,
            }, indent=2), encoding='utf-8')
        return
    for path_name in ("config", "checkpoint", "esmc_checkpoint"):
        path = Path(getattr(args, path_name)).expanduser().resolve()
        if not path.is_file():
            option = path_name.replace("_", "-")
            raise FileNotFoundError(f"--{option} does not exist: {path}")
        setattr(args, path_name, str(path))
    if not args.data:
        raise ValueError(f"{args.task} evaluation requires --data")
    data = Path(args.data).expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"--data does not exist: {data}")
    args.data = str(data)
    device = _device(args.device)
    output_path = Path(args.output_json).expanduser().resolve() if args.output_json else None
    print(f"Checkpoint: {args.checkpoint}")
    print(f"ESMC checkpoint: {args.esmc_checkpoint}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    if args.task == "localization":
        default_work = (
            output_path.parent / f"{output_path.stem}_artifacts"
            if output_path else PROJECT_ROOT / "evaluation_results" / "e2e" / "localization_artifacts"
        )
        work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else default_work
        metrics, counts, provenance = evaluate_deployed_localization(
            config=args.config,
            checkpoint=args.checkpoint,
            data_file=args.data,
            esmc_checkpoint=args.esmc_checkpoint,
            work_dir=str(work_dir),
            extract_features=extract_features_in_memory,
            run_prediction=run_prediction,
            device=str(device),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_orfs=args.max_orfs,
            toks_per_batch=args.toks_per_batch,
            truncation_seq_length=args.truncation_seq_length,
            threshold=args.threshold,
            overlap_ratio=args.overlap_ratio,
            type_threshold=args.type_threshold,
        )
        feature_stats = provenance["features"]
        source = provenance["prediction_outputs"].get("checkpoint_path", args.checkpoint)
        evaluated_input = truth_input = args.data
    else:
        features_manager, _ = extract_features_in_memory(
            input_json=args.data,
            device=str(device),
            model_path=args.esmc_checkpoint,
            toks_per_batch=args.toks_per_batch,
            truncation_seq_length=args.truncation_seq_length,
            progress_name=f"ESMC classification {Path(args.data).stem}",
        )
        feature_stats = features_manager.get_stats()
        metrics, counts, source = evaluate_product_types(
            config=args.config,
            checkpoint=args.checkpoint,
            data_file=args.data,
            features_manager=features_manager,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_orfs=args.max_orfs,
            type_threshold=args.type_threshold,
        )
        print("Matching: training Hungarian matcher; classification metrics on matched pairs")
        provenance = {}
        evaluated_input = truth_input = args.data

    print(f"Weights: {source}")
    print(f"Counts: {counts}")
    print(f"Online features: {feature_stats['total_features']} ORFs, dim={feature_stats['embedding_dim']}")
    reported_metrics = report_metrics(metrics)
    if args.task == "localization":
        for cutoff, values in provenance["region_metrics"].items():
            print(f"Nucleotide regions (IoU={cutoff}): " + " | ".join(
                f"{key}={values[key]:.3f}" for key in ('precision', 'recall', 'f1', 'avg_iou')))

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": args.task,
            "evaluation_protocol": "deployed_output" if args.task == "localization" else "gt-guided-hungarian",
            "input": evaluated_input,
            "truth_data": truth_input,
            "config": args.config,
            "checkpoint": args.checkpoint,
            "esmc_checkpoint": args.esmc_checkpoint,
            "weight_source": source,
            "features": feature_stats,
            "provenance": provenance,
            "thresholds": ({"detection": args.threshold, "overlap_ratio": args.overlap_ratio,
                            "type": args.type_threshold} if args.task == "localization"
                           else {"type": args.type_threshold}),
            "counts": counts,
            "metrics": reported_metrics,
        }
        if args.task == "type":
            payload["metric_scope"] = "matched_only"
        else:
            payload["region_metrics"] = provenance["region_metrics"]
            payload["region_counts"] = provenance["region_counts"]
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Metrics JSON: {output_path}")


if __name__ == "__main__":
    main()
