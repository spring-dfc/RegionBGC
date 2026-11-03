"""Audit prediction IDs and sequences against Prodigal and the original DNA.

This script does not run Prodigal or either neural network. It independently
reads existing outputs and returns a result without writing any files.
Coordinates are 1-based inclusive. DNA is reverse-complemented for negative CDS.
Prodigal proteins are compared directly with dataset and GBK proteins; DNA is
not retranslated. Original nucleotide sequences, including N, are preserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq


def read_table(path: Path, delimiter: str = ",") -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def as_list(value) -> list:
    return json.loads(value) if isinstance(value, str) else value


def digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def read_prodigal_annotations(path: Path, sample: str) -> tuple[dict, dict, dict]:
    """Read Prodigal's native GenBank annotation (no nucleotide ORIGIN section)."""
    codes, lengths, genes = {}, {}, {}
    contig, pending = None, None
    with path.open() as handle:
        for line in handle:
            if line.startswith(('DEFINITION', '//')) and pending is not None:
                raise ValueError('Prodigal CDS annotation is missing its ID')
            if line.startswith('DEFINITION'):
                contig = re.search(r'seqhdr="([^"]+)"', line).group(1).split()[0]
                if contig in lengths:
                    raise ValueError(f'Duplicate Prodigal contig: {contig}')
                lengths[contig] = int(re.search(r'seqlen=(\d+)', line).group(1))
                codes[contig] = int(re.search(r'transl_table=(\d+)', line).group(1))
            elif line.lstrip().startswith('CDS '):
                if pending is not None or contig is None:
                    raise ValueError('Invalid Prodigal CDS annotation order')
                location = line.strip().split(None, 1)[1]
                strand = -1 if location.startswith('complement(') else 1
                if strand == -1:
                    if not location.endswith(')'):
                        raise ValueError(f'Invalid Prodigal CDS location: {location}')
                    location = location[len('complement('):-1]
                match = re.fullmatch(r'[<>]?(\d+)\.\.[<>]?(\d+)', location)
                if match is None:
                    raise ValueError(f'Invalid Prodigal CDS location: {location}')
                pending = (contig, int(match[1]), int(match[2]), strand)
            elif pending is not None and '/note=' in line:
                local_id = re.search(r'\bID=([^;\s"]+)', line)
                if local_id is None:
                    raise ValueError('Prodigal CDS annotation is missing its ID')
                orf_id = f'{sample}_{local_id[1]}'
                if orf_id in genes:
                    raise ValueError(f'Duplicate Prodigal ORF: {orf_id}')
                genes[orf_id] = pending
                pending = None
            elif line.startswith('//'):
                contig = None
    if pending is not None:
        raise ValueError('Incomplete Prodigal CDS annotation')
    return codes, lengths, genes


def audit(input_fasta: Path, prediction_dir: Path, *, check_merged: bool = True,
          check_merged_json: bool = True) -> dict:
    """Audit one genome; batch merge checks can be deferred until the batch ends."""
    sample = input_fasta.stem
    sample_dir = prediction_dir / sample
    faa_path = prediction_dir / "prepared_inputs" / "protein" / f"{sample}.faa"
    genes_path = faa_path.with_suffix(".genes")
    prepared_path = prediction_dir / "prepared_inputs" / "nucleotide" / f"{sample}.fna"
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    # Match working contigs to original records by order AND complete sequence.
    # All subsequent CDS extraction uses the original FASTA, not the working copy.
    source_records = list(SeqIO.parse(str(input_fasta), "fasta"))
    prepared_records = list(SeqIO.parse(str(prepared_path), "fasta"))
    check(bool(source_records), "Original FASTA is empty")
    check(len(source_records) == len(prepared_records), "Original/prepared contig count mismatch")
    check(len({r.id for r in source_records}) == len(source_records), "Duplicate original contig IDs")
    check(len({r.id for r in prepared_records}) == len(prepared_records), "Duplicate prepared contig IDs")
    dna, original_names = {}, {}
    for source, prepared in zip(source_records, prepared_records):
        sequence = str(source.seq).upper()
        check(sequence == str(prepared.seq).upper(), f"DNA copy mismatch: {source.id}")
        dna[prepared.id] = sequence
        original_names[prepared.id] = source.id

    codes, annotated_lengths, annotated_orfs = read_prodigal_annotations(genes_path, sample)
    check(annotated_lengths == {contig: len(seq) for contig, seq in dna.items()},
          'Prodigal/input contig coverage or length mismatch')

    proteins, by_contig = {}, defaultdict(list)
    orf_rows, faa_ids, counts = [], [], Counter()
    for record in SeqIO.parse(str(faa_path), "fasta"):
        parts = re.split(r"\s*#\s*", record.description)
        contig = parts[0].split()[0].rsplit("_", 1)[0]
        start, end, strand = map(int, parts[1:4])
        metadata = dict(x.split("=", 1) for x in parts[4].split(";") if "=" in x)
        orf_id = f"{sample}_{metadata['ID']}"
        faa_ids.append(orf_id)
        raw_protein = str(record.seq)
        protein = raw_protein.removesuffix("*")
        check(orf_id not in proteins, f"Duplicate ORF ID: {orf_id}")
        proteins[orf_id] = dict(contig=contig, start=start, end=end, strand=strand, protein=raw_protein)
        by_contig[contig].append(orf_id)
        check(annotated_orfs.get(orf_id) == (contig, start, end, strand),
              f'Prodigal annotation/FAA coordinate or strand mismatch: {orf_id}')

        valid = contig in dna and 1 <= start <= end <= len(dna[contig]) and strand in (-1, 1)
        nucleotides = dna[contig][start - 1:end] if valid else ""
        if strand == -1:
            nucleotides = str(Seq(nucleotides).reverse_complement())
        code = codes.get(contig)
        frame_valid = bool(nucleotides) and len(nucleotides) % 3 == 0
        check(valid, f"Invalid CDS coordinate/strand: {orf_id}")
        check(frame_valid, f"Invalid CDS reading frame: {orf_id}")
        check("*" not in protein, f"Internal stop in Prodigal protein: {orf_id}")
        counts["positive_strand" if strand == 1 else "negative_strand"] += 1
        counts["complete" if metadata.get("partial") == "00" else "partial"] += 1
        if metadata.get("start_type") not in ("ATG", "Edge"):
            counts["alternative_start"] += 1
        orf_rows.append(dict(
            orf_id=orf_id, faa_record_id=record.id, contig=contig,
            original_contig=original_names.get(contig, ""), start_nt=start, end_nt=end,
            strand=strand, partial=metadata.get("partial"), start_type=metadata.get("start_type"),
            translation_table=code, cds_nt_length=len(nucleotides), protein_aa_length=len(protein),
            coordinate_valid=valid, reading_frame_valid=frame_valid,
            cds_sha256=digest(nucleotides),
        ))
    check(bool(faa_ids), "No Prodigal ORFs to audit")
    check(list(annotated_orfs) == faa_ids, 'Prodigal annotation/FAA ORF order mismatch')
    by_id = {row["orf_id"]: row for row in orf_rows}

    dataset = json.loads((sample_dir / "prediction_dataset.json").read_text())
    check(Counter(record['original_genome_id'] for record in dataset) == Counter(by_contig.keys()),
          'Dataset/FAA contig coverage mismatch')
    dataset_ids = []
    for record in dataset:
        contig = record["original_genome_id"]
        ids, positions = as_list(record["orf_ids"]), as_list(record["orf_positions"])
        check(record["genome_id"] == contig, f"Dataset contig IDs differ: {contig}")
        check(ids == by_contig[contig], f"Dataset ORF order mismatch: {contig}")
        check(list(record['orf_sequences']) == ids,
              f'Dataset ORF ID/sequence coverage mismatch: {contig}')
        check(len(ids) == len(positions), f"Dataset coordinate count mismatch: {contig}")
        dataset_ids.extend(ids)
        for orf_id, position in zip(ids, positions):
            protein = proteins.get(orf_id)
            check(protein is not None and protein["contig"] == contig
                  and list(position) == [protein["start"], protein["end"]]
                  and record["orf_sequences"].get(orf_id) == protein["protein"],
                  f"Dataset/FAA mismatch: {orf_id}")

    predicted_ids = [r["ORF_ID"] for r in read_table(sample_dir / "orf_predictions.tsv", "\t")]
    regions = [{k: v for k, v in row.items() if k != "sequence_verification"}
               for row in read_table(sample_dir / "bgc_regions.csv")]
    json_regions = json.loads((sample_dir / "bgc_regions.json").read_text())
    check(len(regions) == len(json_regions), "CSV/JSON BGC count mismatch")
    if check_merged:
        merged_ids = [r["ORF_ID"] for r in read_table(prediction_dir / "merged_orf_predictions.tsv", "\t")
                      if r["sample_name"] == sample]
        merged_csv_path = prediction_dir / "merged_bgc_regions.csv"
        merged_csv = [{k: v for k, v in row.items() if k not in ("sample_name", "sequence_verification")}
                      for row in (read_table(merged_csv_path) if merged_csv_path.exists() else [])
                      if row["sample_name"] == sample]
        check(regions == merged_csv, "Merged/sample CSV mismatch")
    if check_merged and check_merged_json:
        merged_json_path = prediction_dir / "merged_bgc_regions.json"
        merged_json = [r for r in (json.loads(merged_json_path.read_text()) if merged_json_path.exists() else [])
                       if r["genome"] == f"genome.protein_{sample}"]
        check(len(regions) == len(merged_json), "CSV/JSON BGC count mismatch")
        # A batch-wide merge can assign different BGC names; compare the actual payload.
        check([{k: v for k, v in r.items() if k != "bgc_name"} for r in json_regions]
              == [{k: v for k, v in r.items() if k != "bgc_name"} for r in merged_json],
              "Merged/sample JSON mismatch")

    def check_gbk_cds(record, offset: int, label: str) -> list[str]:
        ids = []
        for feature in record.features:
            if feature.type != "CDS":
                continue
            orf_id = feature.qualifiers.get("locus_tag", [""])[0]
            ids.append(orf_id)
            protein = proteins.get(orf_id)
            if protein is None:
                errors.append(f"{label}: unknown CDS {orf_id}")
                continue
            check(record.id == protein["contig"]
                  and int(feature.location.start) + offset + 1 == protein["start"]
                  and int(feature.location.end) + offset == protein["end"]
                  and feature.location.strand == protein["strand"],
                  f"{label}: CDS coordinate/strand mismatch {orf_id}")
            check(feature.qualifiers.get("translation", [""])[0]
                  == protein["protein"].removesuffix("*"), f"{label}: protein mismatch {orf_id}")
            check(digest(str(feature.extract(record.seq)).upper()) == by_id[orf_id]["cds_sha256"],
                  f"{label}: source CDS DNA mismatch {orf_id}")
        return ids

    bgc_rows, region_gbk_cds = [], 0
    for index, (region, exported) in enumerate(zip(regions, json_regions), 1):
        label = f"BGC{index}"
        contig, span = region["bgc_range"].rsplit(":", 1)
        start, end = map(int, span.split("-"))
        ids = region["cds_ids"].split()
        payload = exported["regions"]["regionbgc"][0]["regions"][0]
        sequence = dna[contig][start - 1:end]
        check(region["genome"] == contig and 1 <= start <= end <= len(dna[contig]),
              f"{label}: invalid contig/range")
        check(ids == by_contig[contig][int(region["start_cds"]) - 1:int(region["end_cds"])],
              f"{label}: CDS indices/FAA order mismatch")
        check(len(ids) == int(region["cds_len"]), f"{label}: CDS count mismatch")
        check(payload["contig"] == contig and int(payload["min_nt"]) == start
              and int(payload["max_nt"]) == end and payload["sequence"] == sequence
              and payload["cds_ids"].split() == ids
              and len(sequence) == int(payload["length_nt"]) == int(region["nt_len"]) == end - start + 1,
              f"{label}: JSON/original DNA mismatch")
        positions = [proteins[o] for o in ids if o in proteins]
        check(bool(positions) and start == min(p["start"] for p in positions)
              and end == max(p["end"] for p in positions), f"{label}: range/ORF span mismatch")
        for orf_id in ids:
            protein = proteins.get(orf_id)
            if protein is None:
                errors.append(f"{label}: missing FAA ORF {orf_id}")
                continue
            contained = protein["contig"] == contig and start <= protein["start"] <= protein["end"] <= end
            check(contained, f"{label}: ORF not contained {orf_id}")
            subsequence = payload["sequence"][protein["start"] - start:protein["end"] - start + 1]
            if protein["strand"] == -1:
                subsequence = str(Seq(subsequence).reverse_complement())
            check(digest(subsequence) == by_id[orf_id]["cds_sha256"],
                  f"{label}: exported CDS DNA mismatch {orf_id}")
            bgc_rows.append(dict(bgc_name=exported["bgc_name"], bgc_start_nt=start,
                                 bgc_end_nt=end, **by_id[orf_id], inside_bgc=contained))
        path = sample_dir / "regions" / f"{contig}.region{index:03d}.gbk"
        region_records = list(SeqIO.parse(str(path), "genbank"))
        check(len(region_records) == 1, f"{label}: GBK record count mismatch")
        for record in region_records:
            check(record.id == contig and str(record.seq).upper() == sequence, f"{label}: GBK DNA mismatch")
            region_ids = check_gbk_cds(record, start - 1, label)
            region_gbk_cds += len(region_ids)
            check(Counter(region_ids) == Counter(ids), f"{label}: GBK/FAA ORF coverage mismatch")

    coverage = {}
    reference = Counter(faa_ids)
    comparisons = [("prodigal_annotation", list(annotated_orfs)),
                   ("dataset", dataset_ids), ("prediction_tsv", predicted_ids)]
    if check_merged:
        comparisons.append(("merged_prediction_tsv", merged_ids))
    for label, ids in comparisons:
        actual = Counter(ids)
        check(actual == reference, f"{label}: FAA ORF coverage mismatch")
        coverage[label] = dict(count=len(ids), missing=len(reference - actual),
                               extra=len(actual - reference), duplicates=sum(n - 1 for n in actual.values() if n > 1))
    unique_bgc_ids = {r["orf_id"] for r in bgc_rows}
    summary = dict(
        status="FAIL" if errors else "PASS", audit_script=str(Path(__file__).resolve()),
        merged_outputs_checked=check_merged,
        input_fasta=str(input_fasta), prodigal_faa=str(faa_path),
        original_contigs=len(source_records), prepared_contigs=len(prepared_records),
        prodigal_orfs=len(faa_ids), dna_translation_checked=False,
        bgc_regions=len(regions), bgc_orf_occurrences=len(bgc_rows), bgc_unique_orfs=len(unique_bgc_ids),
        region_gbk_cds_verified=region_gbk_cds, genetic_codes=sorted(set(codes.values())),
        counts=dict(counts), coverage_results=coverage, errors=errors,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Original nucleotide FASTA")
    parser.add_argument("--prediction-dir", required=True, type=Path, help="Pipeline output directory")
    args = parser.parse_args()
    prediction_dir = args.prediction_dir.expanduser().resolve()
    summary = audit(args.input.expanduser().resolve(), prediction_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
