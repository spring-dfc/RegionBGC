from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


HEADER_SPLIT_PATTERN = re.compile(r"\s*#\s*")
ID_PATTERN = re.compile(r"ID=([^;\s]+)")
CONTIG_PATTERN = re.compile(r"^(?P<contig>.+?)_(?P<idx>\d+)$")


def _contig_key(header_token: str) -> str:
    match = CONTIG_PATTERN.match(header_token)
    return match.group("contig") if match else header_token


def parse_fasta_grouped_by_contig(fasta_path: str) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    contig_order: List[str] = []
    contig_data: Dict[str, Dict[str, Any]] = {}
    file_stem = Path(fasta_path).stem

    def ensure_contig(contig: str) -> Dict[str, Any]:
        if contig not in contig_data:
            contig_data[contig] = {
                "orf_ids": [],
                "orf_sequences": {},
                "orf_positions": [],
            }
            contig_order.append(contig)
        return contig_data[contig]

    current_orf_id: Optional[str] = None
    current_contig: Optional[str] = None
    seq_parts: List[str] = []

    def flush_record() -> None:
        nonlocal seq_parts
        if current_orf_id is None or current_contig is None:
            return
        ensure_contig(current_contig)["orf_sequences"][current_orf_id] = "".join(seq_parts).replace(" ", "")
        seq_parts = []

    with open(fasta_path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            if not line.startswith(">"):
                seq_parts.append(line)
                continue

            flush_record()
            header = line[1:].strip()
            parts = HEADER_SPLIT_PATTERN.split(header)
            head_token = parts[0].split()[0] if parts and parts[0] else header.split()[0]
            contig = _contig_key(head_token)
            ensure_contig(contig)

            start, end = 0, 0
            if len(parts) >= 3:
                try:
                    start = int(parts[1].strip())
                    end = int(parts[2].strip())
                except Exception:
                    start, end = 0, 0

            matched = ID_PATTERN.search(header)
            local_id = matched.group(1) if matched else f"{head_token}_{start}_{end}"
            current_orf_id = f"{file_stem}_{local_id}"
            current_contig = contig

            contig_entry = ensure_contig(contig)
            contig_entry["orf_ids"].append(current_orf_id)
            contig_entry["orf_positions"].append([start, end])

    flush_record()
    return contig_order, contig_data


def build_samples_from_fasta(fasta_path: str, base_name: str, start_index: int = 1) -> List[Dict[str, Any]]:
    contig_order, contig_data = parse_fasta_grouped_by_contig(fasta_path)
    samples: List[Dict[str, Any]] = []

    for contig in contig_order:
        item = contig_data[contig]
        orf_ids = item["orf_ids"]
        samples.append(
            {
                # Contigs without ORFs are absent from the protein FASTA.
                # Keep the parsed name rather than renumbering the survivors.
                "genome_id": contig,
                "original_genome_id": contig,
                "orf_sequences": item["orf_sequences"],
                "orf_positions": json.dumps(item["orf_positions"], ensure_ascii=False),
                "bgc_labels": json.dumps([0] * len(orf_ids), ensure_ascii=False),
                "orf_ids": orf_ids,
            }
        )

    return samples


def discover_fasta_files(input_path: str) -> List[Path]:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if path.is_file():
        if path.suffix.lower() not in {".fa", ".faa", ".fasta"}:
            raise ValueError(f"Unsupported fasta file suffix: {path}")
        return [path]

    fasta_files: List[Path] = []
    for pattern in ("**/*.fa", "**/*.faa", "**/*.fasta"):
        fasta_files.extend(Path(item).resolve() for item in glob.glob(str(path / pattern), recursive=True))

    unique_files = sorted({file_path for file_path in fasta_files})
    if not unique_files:
        raise FileNotFoundError(f"No fasta files found in: {path}")
    return unique_files


def convert_fasta_file_to_dataset(fasta_file: str, output_json: str) -> str:
    fasta_path = Path(fasta_file).expanduser().resolve()
    output_path = Path(output_json).expanduser().resolve()
    if not fasta_path.exists():
        raise FileNotFoundError(f"Input fasta file not found: {fasta_path}")

    samples = build_samples_from_fasta(str(fasta_path), base_name=fasta_path.stem, start_index=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(samples, fh, indent=2, ensure_ascii=False)

    return str(output_path)


def convert_fasta_dir_to_dataset(input_dir: str, output_json: str) -> str:
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_json).expanduser().resolve()
    fasta_files = discover_fasta_files(str(input_path))

    all_samples: List[Dict[str, Any]] = []
    base_counters: Dict[str, int] = {}

    for fasta_path in fasta_files:
        base_name = fasta_path.parent.name if fasta_path.parent != input_path else fasta_path.stem
        start_index = base_counters.get(base_name, 1)
        samples = build_samples_from_fasta(str(fasta_path), base_name=base_name, start_index=start_index)
        base_counters[base_name] = start_index + len(samples)
        all_samples.extend(samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(all_samples, fh, indent=2, ensure_ascii=False)

    return str(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert protein FASTA files to prediction JSON")
    parser.add_argument("--input_dir", required=True, help="Directory containing .fa/.faa/.fasta protein files")
    parser.add_argument("--output_json", required=True, help="Output dataset JSON path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_json = convert_fasta_dir_to_dataset(args.input_dir, args.output_json)
    print(f"Dataset JSON saved to: {output_json}")


if __name__ == "__main__":
    main()
