from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def find_sample_fasta(genome_fasta_dir: str, sample_name: str) -> Path:
    base_dir = Path(genome_fasta_dir).expanduser().resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Genome FASTA directory not found: {base_dir}")

    for suffix in (".fasta", ".fa", ".fna"):
        candidate = base_dir / f"{sample_name}{suffix}"
        if candidate.exists():
            return candidate

    for suffix in (".fasta", ".fa", ".fna"):
        matches = sorted(base_dir.rglob(f"{sample_name}{suffix}"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not find original genome FASTA for sample '{sample_name}' under: {base_dir}"
    )


def parse_dna_fasta(fasta_path: str) -> Dict[str, str]:
    fasta_file = Path(fasta_path).expanduser().resolve()
    if not fasta_file.exists():
        raise FileNotFoundError(f"DNA FASTA file not found: {fasta_file}")

    contig_seqs: Dict[str, List[str]] = {}
    current_key: Optional[str] = None

    with open(fasta_file, "r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_key = line[1:].split()[0]
                contig_seqs.setdefault(current_key, [])
            elif current_key is not None:
                contig_seqs[current_key].append(line.upper())

    out = {key: "".join(parts) for key, parts in contig_seqs.items()}
    if not out:
        raise RuntimeError(f"No contigs parsed from FASTA: {fasta_file}")
    return out


def parse_bgc_range(bgc_range: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    if not bgc_range:
        return None, None, None
    try:
        contig_part, coord_part = str(bgc_range).strip().split(":", 1)
        start_s, end_s = coord_part.split("-", 1)
        start_nt = int(start_s)
        end_nt = int(end_s)
        return contig_part, start_nt, end_nt
    except Exception:
        return None, None, None


def validate_interval(start_nt: int, end_nt: int, max_len: int) -> Tuple[int, int]:
    start, end = int(start_nt), int(end_nt)
    if not 1 <= start <= end <= max_len:
        raise ValueError(
            f"Invalid BGC interval {start}-{end} for contig length {max_len}"
        )
    return start, end


def extract_subseq(seq_full: str, start_nt: int, end_nt: int) -> str:
    start_nt, end_nt = validate_interval(start_nt, end_nt, len(seq_full))
    return seq_full[start_nt - 1 : end_nt]


def _maybe_int(text: object) -> object:
    if text is None:
        return text
    value = str(text).strip()
    if not value:
        return text
    try:
        return int(value)
    except Exception:
        return text


def build_prediction_records(regions_csv: str, dna_fasta: str, sample_name: str) -> List[Dict]:
    contig_map = parse_dna_fasta(dna_fasta)
    records: List[Dict] = []

    with open(regions_csv, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader, start=1):
            contig, start_nt, end_nt = parse_bgc_range(row.get("bgc_range") or "")
            if contig is None or start_nt is None or end_nt is None:
                raise ValueError(f"Invalid BGC range in row {idx}: {row.get('bgc_range')!r}")
            if row.get("genome") != contig:
                raise ValueError(f"BGC contig mismatch in row {idx}: {row.get('genome')!r} != {contig!r}")
            if contig not in contig_map:
                raise KeyError(
                    f"Contig '{contig}' from {regions_csv} was not found in original FASTA {dna_fasta}"
                )

            seq_full = contig_map[contig]
            try:
                sequence = extract_subseq(seq_full, start_nt, end_nt)
            except ValueError as exc:
                raise ValueError(f"Region {idx} on contig {contig}: {exc}") from exc
            if len(sequence) != end_nt - start_nt + 1:
                raise ValueError(f"Sequence length mismatch in region {idx} on contig {contig}")
            region_payload = {
                "contig": contig,
                "min_nt": str(start_nt),
                "max_nt": str(end_nt),
                "length_nt": str(len(sequence)),
                "cds_ids": str(row.get("cds_ids") or "").strip(),
                "sequence": sequence,
            }
            predicted_region = {
                "segment_id": f"RegionBGC_{sample_name}_region_{idx}",
                "confidence": str(row.get("confidence") or ""),
                "class_name": str(row.get("class_name") or ""),
                "start_cds": _maybe_int(row.get("start_cds")),
                "end_cds": _maybe_int(row.get("end_cds")),
                "regions": [region_payload],
            }
            records.append(
                {
                    "bgc_name": f"BGC{idx}",
                    "genome": f"genome.protein_{sample_name}",
                    "regions": {
                        "regionbgc": [predicted_region],
                    },
                }
            )

    return records


def build_prediction_json(
    regions_csv: str,
    dna_fasta: str,
    output_json: str,
    sample_name: Optional[str] = None,
) -> str:
    sample = sample_name or Path(dna_fasta).stem
    records = build_prediction_records(
        regions_csv=regions_csv,
        dna_fasta=dna_fasta,
        sample_name=sample,
    )

    output_path = Path(output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    return str(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build RegionBGC prediction JSON from prediction CSV")
    parser.add_argument("--regions_csv", required=True, help="Path to bgc_regions.csv")
    parser.add_argument("--dna_fasta", required=True, help="Path to the original genome DNA FASTA")
    parser.add_argument("--output_json", required=True, help="Output JSON path")
    parser.add_argument("--sample_name", default=None, help="Optional sample name override")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_json = build_prediction_json(
        regions_csv=args.regions_csv,
        dna_fasta=args.dna_fasta,
        output_json=args.output_json,
        sample_name=args.sample_name,
    )
    print(f"RegionBGC prediction JSON saved to: {output_json}")


if __name__ == "__main__":
    main()
