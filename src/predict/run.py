from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.predict.audit_prediction_sequences import audit
from src.predict.build_prediction_json import build_prediction_json, find_sample_fasta
from src.predict.feature_extractor import extract_features_in_memory, load_esmc_model
from src.predict.genbank_writer import write_prediction_genbanks
from src.predict.input_preparation import (
    discover_nucleotide_files,
    prepare_nucleotide_inputs,
)
from src.predict.plot_bgc_predictions import plot_bgc_predictions
from src.predict.predictor import DEFAULT_DETECTION_THRESHOLD, load_prediction_runtime, run_prediction
from src.predict.prepare_dataset import convert_fasta_file_to_dataset
from src.utils.bgc_inference import probability_threshold


USER_DEFAULTS = {
    "config": str(REPO_DIR / "configs" / "regionbgc" / "regionbgc.yml"),
    "resume": str(REPO_DIR / "weights" / "regionbgc.pth"),
    "output_dir": str(REPO_DIR / "outputs" / "prediction"),
    "max_orfs": 128,
    "feature_device": "auto",
    "predict_device": "auto",
    "gpu_id": None,
    "esmc_model_path": str(REPO_DIR / "weights" / "esmc_300m_2024_12_v0.pth"),
    "toks_per_batch": 2560,
    "truncation_seq_length": 2048,
    "predict_batch_size": 32,
    "num_workers": 4,
    "threshold": DEFAULT_DETECTION_THRESHOLD,
    "overlap_ratio": None,
    "bgc_type_threshold": 0.5,
}


MERGED_ORF_TSV_NAME = "merged_orf_predictions.tsv"
MERGED_BGC_REGIONS_CSV_NAME = "merged_bgc_regions.csv"
MERGED_BGC_REGIONS_JSON_NAME = "merged_bgc_regions.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RegionBGC: ORF-interval prediction from nucleotide genomes")
    parser.add_argument(
        "gpu_ids_text",
        nargs="?",
        default=None,
        help="Optional positional GPU ids, e.g. '0,1,2,3'. "
        "This allows commands like: python tools/predict.py 0,1,2,3",
    )
    parser.add_argument(
        "--input", "--input_dir", dest="input_path", type=str, required=True,
        help="A nucleotide FASTA file or directory",
    )
    contig_names = parser.add_mutually_exclusive_group()
    contig_names.add_argument(
        "--rename-contigs", dest="rename_contigs", action="store_true", default=True,
        help="Safely rename contigs to <sample>_<index> (default)",
    )
    contig_names.add_argument(
        "--keep-contig-names", dest="rename_contigs", action="store_false",
        help="Keep original contig names (only use when names are already unique and safe)",
    )
    parser.add_argument("--config", type=str, default=USER_DEFAULTS["config"], help="RegionBGC config path")
    parser.add_argument("--checkpoint", "--resume", dest="resume", default=USER_DEFAULTS["resume"], help="RegionBGC checkpoint file")
    parser.add_argument("--output_dir", "--output-dir", type=str, required=True, help="Writable prediction output directory")
    parser.add_argument("--max_orfs", type=int, default=USER_DEFAULTS["max_orfs"])
    parser.add_argument("--feature_device", type=str, default=USER_DEFAULTS["feature_device"])
    parser.add_argument("--predict_device", type=str, default=USER_DEFAULTS["predict_device"])
    parser.add_argument("--gpu_id", type=int, default=USER_DEFAULTS["gpu_id"])
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU ids for multi-GPU mode, e.g. '0,1,2,3'. "
        "When multiple ids are provided, genomes are distributed across GPUs in parallel.",
    )
    parser.add_argument("--esmc_model_path", type=str, default=USER_DEFAULTS["esmc_model_path"])
    parser.add_argument("--toks_per_batch", type=int, default=USER_DEFAULTS["toks_per_batch"])
    parser.add_argument("--truncation_seq_length", type=int, default=USER_DEFAULTS["truncation_seq_length"])
    parser.add_argument("--predict_batch_size", type=int, default=USER_DEFAULTS["predict_batch_size"])
    parser.add_argument("--num_workers", type=int, default=USER_DEFAULTS["num_workers"])
    parser.add_argument("--threshold", type=probability_threshold, default=USER_DEFAULTS["threshold"], help="BGC detection threshold in [0, 1] (default: 0.3)")
    parser.add_argument("--overlap_ratio", type=float, default=USER_DEFAULTS["overlap_ratio"])
    parser.add_argument("--type-threshold", "--bgc_type_threshold", "--classify_t", dest="bgc_type_threshold", type=probability_threshold, default=USER_DEFAULTS["bgc_type_threshold"], help="Product classification threshold in [0, 1] (default: 0.5)")
    parser.add_argument("--plot", action="store_true", help="Save prediction plots")
    parser.add_argument("--save-json", action="store_true", help="Keep sequence JSON outputs")
    parser.add_argument(
        "--verify-sequences", "--verify_sequences", action="store_true",
        help="Verify outputs against Prodigal and original DNA; append PASS/FAIL "
        "to the final CSV (default: NOT_RUN). No audit report files are written.",
    )
    return parser


def _validate_required_paths(args) -> None:
    required_paths = {
        "input": (args.input_path, "path"),
        "config": (args.config, "file"),
        "resume": (args.resume, "file"),
        "esmc_model_path": (args.esmc_model_path, "file"),
    }

    for field_name, (raw_path, expected_type) in required_paths.items():
        path = Path(raw_path).expanduser().resolve()
        exists = path.is_file() if expected_type == "file" else path.exists()
        if not exists:
            raise FileNotFoundError(f"--{field_name} does not exist: {path}")


def _validate_runtime(args) -> Path:
    if shutil.which("prodigal") is None:
        raise EnvironmentError("Prodigal was not found in PATH.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(prefix=".regionbgc_write_test_", dir=output_dir):
            pass
    except OSError as exc:
        raise PermissionError(
            f"Output directory is not writable: {output_dir}. "
            "Choose a directory owned by the current user."
        ) from exc
    return output_dir


def _validate_unique_input_names(source_files: List[Path]) -> None:
    stems = [path.stem for path in source_files]
    duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicates:
        raise ValueError(f"Input FASTA file names must be unique: {duplicates}")


def _parse_gpu_ids(args) -> List[int]:
    raw_value = getattr(args, "gpu_ids_text", None)
    if raw_value is None or str(raw_value).strip() == "":
        raw_value = getattr(args, "gpu_ids", None)
    if raw_value is None or str(raw_value).strip() == "":
        return [int(args.gpu_id)] if args.gpu_id is not None else []

    gpu_ids: List[int] = []
    seen = set()
    for part in str(raw_value).split(","):
        token = part.strip()
        if not token:
            continue
        gpu_id = int(token)
        if gpu_id not in seen:
            gpu_ids.append(gpu_id)
            seen.add(gpu_id)
    if not gpu_ids:
        raise ValueError("`--gpu_ids` was provided but no valid GPU ids were parsed.")
    return gpu_ids


def _resolve_device_for_gpu(device_text: Optional[str], gpu_id: Optional[int]) -> str:
    text = str(device_text).strip().lower() if device_text is not None else ""
    if text == "cpu":
        return "cpu"
    if gpu_id is not None:
        if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {gpu_id} was requested, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )
        return f"cuda:{gpu_id}"
    if text in ("", "auto"):
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if text == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if text.startswith("cuda:"):
        index = int(text.split(":", 1)[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} is not available.")
    return text


def _build_runtime_args_dict(args) -> Dict[str, object]:
    return {
        "config": str(args.config),
        "resume": str(args.resume),
        "output_dir": str(args.output_dir),
        "plot": bool(args.plot),
        "save_json": bool(args.save_json),
        "verify_sequences": bool(args.verify_sequences),
        "rename_contigs": bool(args.rename_contigs),
        "max_orfs": int(args.max_orfs),
        "feature_device": str(args.feature_device),
        "predict_device": str(args.predict_device),
        "esmc_model_path": str(args.esmc_model_path),
        "toks_per_batch": int(args.toks_per_batch),
        "truncation_seq_length": int(args.truncation_seq_length),
        "predict_batch_size": int(args.predict_batch_size),
        "num_workers": int(args.num_workers),
        "threshold": args.threshold,
        "overlap_ratio": args.overlap_ratio,
        "bgc_type_threshold": float(args.bgc_type_threshold),
    }


def _process_single_sample(
    nucleotide_file_text: str,
    sample_index: int,
    total_samples: int,
    runtime_args: Dict[str, object],
    gpu_id: Optional[int],
    esmc_model,
    prediction_runtime: Dict,
) -> Dict:
    started_at = time.monotonic()
    nucleotide_file = Path(nucleotide_file_text).expanduser().resolve()
    output_dir = Path(str(runtime_args["output_dir"])).expanduser().resolve()
    sample_name = nucleotide_file.stem
    sample_output_dir = output_dir / sample_name
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    feature_device = _resolve_device_for_gpu(str(runtime_args["feature_device"]), gpu_id)
    predict_device = _resolve_device_for_gpu(str(runtime_args["predict_device"]), gpu_id)
    gpu_label = f"GPU {gpu_id}" if gpu_id is not None else predict_device.upper()

    print(
        f"[{sample_index}/{total_samples}] [{gpu_label}] "
        f"Run Prodigal: {nucleotide_file.name}", flush=True
    )
    protein_dir, genome_fasta_dir = prepare_nucleotide_inputs(
        [nucleotide_file],
        output_dir / "prepared_inputs",
        rename_contigs=bool(runtime_args["rename_contigs"]),
    )
    fasta_file = protein_dir / f"{sample_name}.faa"

    dataset_json = sample_output_dir / "prediction_dataset.json"

    print(f"[{sample_index}/{total_samples}] [{gpu_label}] Prepare dataset: {fasta_file.name}", flush=True)
    json_path = convert_fasta_file_to_dataset(str(fasta_file), str(dataset_json))

    print(f"[{sample_index}/{total_samples}] [{gpu_label}] Extract ESMC features", flush=True)
    feature_manager, _ = extract_features_in_memory(
        input_json=json_path,
        device=feature_device,
        gpu_id=None,
        model_path=str(runtime_args["esmc_model_path"]),
        toks_per_batch=int(runtime_args["toks_per_batch"]),
        truncation_seq_length=int(runtime_args["truncation_seq_length"]),
        progress_name=f"ESMC {sample_name} ({gpu_label})",
        model=esmc_model,
    )
    feature_stats = feature_manager.get_stats()
    print(
        f"[{sample_index}/{total_samples}] [{gpu_label}] Features ready: "
        f"total={feature_stats['total_features']} dim={feature_stats['embedding_dim']} "
        f"mem={feature_stats['memory_usage_mb']:.2f}MB", flush=True
    )

    print(f"[{sample_index}/{total_samples}] [{gpu_label}] Predict BGC regions", flush=True)
    outputs = run_prediction(
        test_file=json_path, features_manager=feature_manager,
        config=str(runtime_args["config"]), resume=str(runtime_args["resume"]),
        output_dir=str(sample_output_dir), max_orfs=int(runtime_args["max_orfs"]),
        batch_size=int(runtime_args["predict_batch_size"]),
        num_workers=int(runtime_args["num_workers"]),
        threshold=(float(runtime_args["threshold"]) if runtime_args["threshold"] is not None else None),
        overlap_ratio=(float(runtime_args["overlap_ratio"]) if runtime_args["overlap_ratio"] is not None else None),
        bgc_type_threshold=float(runtime_args["bgc_type_threshold"]),
        save_orf_txt=True, device=predict_device,
        progress_name=f"Predict {sample_name} ({gpu_label})",
        runtime=prediction_runtime,
    )

    genome_fasta_path = find_sample_fasta(str(genome_fasta_dir), sample_name)
    if runtime_args["save_json"] or runtime_args["verify_sequences"]:
        outputs["bgc_regions_json"] = build_prediction_json(
            regions_csv=outputs["csv_path"], dna_fasta=str(genome_fasta_path),
            output_json=str(sample_output_dir / "bgc_regions.json"), sample_name=sample_name,
        )
    outputs.update(
        write_prediction_genbanks(
            dna_fasta=str(genome_fasta_path),
            protein_faa=str(fasta_file),
            regions_csv=outputs["csv_path"],
            output_dir=str(sample_output_dir),
            sample_name=sample_name,
        )
    )
    print(
        f"[{sample_index}/{total_samples}] [{gpu_label}] GenBank ready: "
        f"regions={len(outputs['region_gbk_paths'])}",
        flush=True,
    )
    if runtime_args["plot"]:
        outputs["bgc_regions_plot"] = plot_bgc_predictions(
            regions_csv=outputs["csv_path"], dna_fasta=str(genome_fasta_path),
            output_png=str(sample_output_dir / "bgc_regions.png"),
            title=f"RegionBGC prediction: {sample_name}",
        )
    elapsed = time.monotonic() - started_at
    print(f"[{sample_index}/{total_samples}] [{gpu_label}] Completed {sample_name} in {elapsed:.1f}s", flush=True)
    return {
        "sample_index": sample_index, "status": "success", "gpu_id": gpu_id,
        "feature_device": feature_device, "predict_device": predict_device,
        "input_fasta": str(nucleotide_file), "genome_fasta": str(genome_fasta_path),
        "sample_name": sample_name, "dataset_json": json_path,
        "feature_stats": feature_stats, "prediction_outputs": outputs,
        "uses_hdf5": False, "elapsed_seconds": elapsed,
    }


def _process_sample_batch(
    jobs: List[Tuple[int, str]],
    total_samples: int,
    runtime_args: Dict[str, object],
    gpu_id: Optional[int],
) -> List[Dict]:
    feature_device = _resolve_device_for_gpu(str(runtime_args["feature_device"]), gpu_id)
    predict_device = _resolve_device_for_gpu(str(runtime_args["predict_device"]), gpu_id)
    worker_label = f"GPU {gpu_id}" if gpu_id is not None else predict_device.upper()
    print(f"[{worker_label}] Loading ESMC model once", flush=True)
    esmc_model = load_esmc_model(feature_device, model_path=str(runtime_args["esmc_model_path"]))
    print(f"[{worker_label}] Loading RegionBGC model once", flush=True)
    prediction_runtime = load_prediction_runtime(
        config=str(runtime_args["config"]), resume=str(runtime_args["resume"]),
        device=predict_device,
    )
    print(f"[{worker_label}] Models ready; processing {len(jobs)} genome(s)", flush=True)

    summaries = []
    for sample_index, nucleotide_file_text in jobs:
        try:
            summaries.append(_process_single_sample(
                nucleotide_file_text=nucleotide_file_text, sample_index=sample_index,
                total_samples=total_samples, runtime_args=runtime_args, gpu_id=gpu_id,
                esmc_model=esmc_model, prediction_runtime=prediction_runtime,
            ))
        except Exception as exc:
            print(
                f"[{sample_index}/{total_samples}] [{worker_label}] FAILED "
                f"{Path(nucleotide_file_text).name}: {exc}",
                file=sys.stderr, flush=True,
            )
            traceback.print_exc()
            summaries.append({
                "sample_index": sample_index, "status": "failed", "gpu_id": gpu_id,
                "input_fasta": nucleotide_file_text,
                "sample_name": Path(nucleotide_file_text).stem,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return summaries

def merge_sample_prediction_outputs(output_dir: Path, sample_names: List[str], *, save_json: bool = False) -> Dict[str, str]:
    # Only merge successful samples from this run, never stale/partial files.
    sample_dirs = [output_dir / name for name in sorted(sample_names)]
    merged_outputs: Dict[str, str] = {}

    merged_orf_path = output_dir / MERGED_ORF_TSV_NAME
    merged_orf_written = False
    with open(merged_orf_path, "w", encoding="utf-8") as out_fh:
        out_fh.write("sample_name\tORF_ID\tProb\tPred\n")
        for sample_dir in sample_dirs:
            orf_path = sample_dir / "orf_predictions.tsv"
            if not orf_path.exists():
                continue

            with open(orf_path, "r", encoding="utf-8") as in_fh:
                next(in_fh, None)
                for raw_line in in_fh:
                    line = raw_line.rstrip("\n")
                    if not line:
                        continue
                    out_fh.write(f"{sample_dir.name}\t{line}\n")
                    merged_orf_written = True

    if merged_orf_written:
        merged_outputs["merged_orf_tsv"] = str(merged_orf_path)
    else:
        merged_orf_path.unlink(missing_ok=True)

    merged_bgc_regions_csv_path = output_dir / MERGED_BGC_REGIONS_CSV_NAME
    merged_bgc_regions_csv_written = False
    with open(merged_bgc_regions_csv_path, "w", encoding="utf-8", newline="") as out_fh:
        writer = None
        for sample_dir in sample_dirs:
            bgc_regions_csv_path = sample_dir / "bgc_regions.csv"
            if not bgc_regions_csv_path.exists():
                continue

            with open(bgc_regions_csv_path, "r", encoding="utf-8", newline="") as in_fh:
                reader = csv.DictReader(in_fh)
                if not reader.fieldnames:
                    continue
                if writer is None:
                    writer = csv.DictWriter(out_fh, fieldnames=["sample_name", *reader.fieldnames])
                    writer.writeheader()
                for row in reader:
                    if not row:
                        continue
                    writer.writerow({"sample_name": sample_dir.name, **row})
                    merged_bgc_regions_csv_written = True

    if merged_bgc_regions_csv_written:
        merged_outputs["merged_bgc_regions_csv"] = str(merged_bgc_regions_csv_path)
    else:
        merged_bgc_regions_csv_path.unlink(missing_ok=True)

    if not save_json:
        return merged_outputs

    merged_bgc_regions_json_path = output_dir / MERGED_BGC_REGIONS_JSON_NAME
    merged_bgc_records = []
    merged_bgc_index = 1
    for sample_dir in sample_dirs:
        json_path = sample_dir / "bgc_regions.json"
        if not json_path.exists():
            continue

        with open(json_path, "r", encoding="utf-8") as in_fh:
            try:
                records = json.load(in_fh)
            except Exception:
                continue

        if not isinstance(records, list):
            continue

        for record in records:
            if not isinstance(record, dict):
                continue
            merged_record = dict(record)
            merged_record["bgc_name"] = f"BGC{merged_bgc_index}"
            merged_bgc_index += 1
            merged_bgc_records.append(merged_record)

    if merged_bgc_records:
        with open(merged_bgc_regions_json_path, "w", encoding="utf-8") as out_fh:
            json.dump(merged_bgc_records, out_fh, indent=2, ensure_ascii=False)
        merged_outputs["merged_bgc_regions_json"] = str(merged_bgc_regions_json_path)
    else:
        merged_bgc_regions_json_path.unlink(missing_ok=True)

    return merged_outputs


def verify_sample_outputs(run_summaries: List[Dict], output_dir: Path, enabled: bool,
                          *, save_json: bool = False) -> Dict[str, str]:
    statuses = {}
    for item in run_summaries:
        status = "NOT_RUN"
        if enabled and item.get("status") == "success":
            try:
                result = audit(Path(item["input_fasta"]), output_dir, check_merged_json=save_json)
                status = result["status"]
                for error in result["errors"]:
                    print(f"[Sequence verification] {item['sample_name']}: {error}", file=sys.stderr, flush=True)
            except Exception as exc:
                status = "FAIL"
                print(f"[Sequence verification] {item['sample_name']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            print(f"[Sequence verification] {item['sample_name']}: {status}", flush=True)
        item["sequence_verification"] = status
        statuses[item["sample_name"]] = status
    return statuses


def append_verification_column(csv_path: Path, statuses: Dict[str, str]) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [name for name in reader.fieldnames if name != "sequence_verification"]
        rows = list(reader)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fields, "sequence_verification"])
        writer.writeheader()
        for row in rows:
            row["sequence_verification"] = statuses.get(row["sample_name"], "NOT_RUN")
            writer.writerow(row)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_required_paths(args)

    output_dir = _validate_runtime(args)
    source_files = discover_nucleotide_files(args.input_path)
    _validate_unique_input_names(source_files)
    runtime_args = _build_runtime_args_dict(args)
    gpu_ids = _parse_gpu_ids(args)
    use_multi_gpu = len(gpu_ids) > 1 and len(source_files) > 1

    if gpu_ids:
        print(f"Selected GPUs: {gpu_ids}")
    else:
        print("Selected device mode: CPU / explicit device strings")

    if use_multi_gpu:
        gpu_jobs: Dict[int, List[Tuple[int, str]]] = {gpu_id: [] for gpu_id in gpu_ids}
        for idx, fasta_file in enumerate(source_files, start=1):
            assigned_gpu = gpu_ids[(idx - 1) % len(gpu_ids)]
            gpu_jobs[assigned_gpu].append((idx, str(fasta_file)))

        print("Multi-GPU mode enabled. Genome batches per GPU:")
        for gpu_id in gpu_ids:
            print(f"  GPU {gpu_id}: {len(gpu_jobs[gpu_id])} genome(s)")

        run_summaries = []
        with ProcessPoolExecutor(
            max_workers=len(gpu_ids),
            mp_context=get_context("spawn"),
        ) as executor:
            future_to_gpu = {
                executor.submit(
                    _process_sample_batch,
                    gpu_jobs[gpu_id],
                    len(source_files),
                    runtime_args,
                    gpu_id,
                ): gpu_id
                for gpu_id in gpu_ids
                if gpu_jobs[gpu_id]
            }
            for future in as_completed(future_to_gpu):
                gpu_id = future_to_gpu[future]
                batch_summaries = future.result()
                print(f"GPU {gpu_id} finished {len(batch_summaries)} genome(s).")
                run_summaries.extend(batch_summaries)
        run_summaries.sort(key=lambda item: int(item.get("sample_index", 0)))
    else:
        single_gpu = gpu_ids[0] if gpu_ids else None
        run_summaries = _process_sample_batch(
            jobs=[(idx, str(fasta_file)) for idx, fasta_file in enumerate(source_files, start=1)],
            total_samples=len(source_files),
            runtime_args=runtime_args,
            gpu_id=single_gpu,
        )

    successful_samples = [
        item["sample_name"] for item in run_summaries if item.get("status") == "success"
    ]
    merged_outputs = merge_sample_prediction_outputs(output_dir, successful_samples, save_json=args.save_json)
    verification_statuses = verify_sample_outputs(run_summaries, output_dir, args.verify_sequences,
                                                 save_json=args.save_json)
    if args.verify_sequences and not args.save_json:
        for item in run_summaries:
            if item.get("sequence_verification") == "PASS":
                outputs = item.get("prediction_outputs", {})
                json_path = outputs.pop("bgc_regions_json", None)
                if json_path:
                    Path(json_path).unlink(missing_ok=True)
    if merged_outputs.get("merged_bgc_regions_csv"):
        append_verification_column(Path(merged_outputs["merged_bgc_regions_csv"]), verification_statuses)
    summary = {"runs": run_summaries, "uses_hdf5": False,
               "thresholds": {"detection": args.threshold, "classification": args.bgc_type_threshold}}
    if merged_outputs:
        summary["merged_outputs"] = merged_outputs

    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("Pipeline finished.")
    if merged_outputs.get("merged_orf_tsv"):
        print(f"Merged ORF predictions: {merged_outputs['merged_orf_tsv']}")
    if merged_outputs.get("merged_bgc_regions_csv"):
        print(f"Merged BGC regions CSV: {merged_outputs['merged_bgc_regions_csv']}")
    if merged_outputs.get("merged_bgc_regions_json"):
        print(f"Merged BGC regions JSON: {merged_outputs['merged_bgc_regions_json']}")
    print(f"Summary: {summary_path}")
    if any(item.get("status") != "success" or item.get("sequence_verification") == "FAIL"
           for item in run_summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
