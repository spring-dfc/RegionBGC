from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from Bio import SeqIO
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord


HEADER_SPLIT_PATTERN = re.compile(r"\s*#\s*")
ID_PATTERN = re.compile(r"ID=([^;\s]+)")
CONTIG_PATTERN = re.compile(r"^(?P<contig>.+?)_(?P<idx>\d+)$")
UNSAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ProdigalCDS:
    contig: str
    orf_id: str
    start: int
    end: int
    strand: int
    translation: str
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class PredictedRegion:
    number: int
    contig: str
    start: int
    end: int
    class_names: Tuple[str, ...]
    confidence: float
    cds_ids: Tuple[str, ...]
    start_cds: Optional[int]
    end_cds: Optional[int]


def _contig_key(header_token: str) -> str:
    match = CONTIG_PATTERN.match(header_token)
    return match.group("contig") if match else header_token


def _parse_metadata(text: str) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for item in text.strip().strip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            metadata[key.strip()] = value.strip()
        else:
            metadata[item] = ""
    return metadata


def parse_prodigal_faa(faa_path: str) -> List[ProdigalCDS]:
    """Parse Prodigal protein FASTA records and retain GenBank-relevant fields."""
    path = Path(faa_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Prodigal protein FASTA not found: {path}")

    file_stem = path.stem
    parsed: List[ProdigalCDS] = []
    seen_ids = set()
    for record in SeqIO.parse(str(path), "fasta"):
        header = record.description
        parts = HEADER_SPLIT_PATTERN.split(header)
        if len(parts) < 4:
            raise ValueError(f"Invalid Prodigal FASTA header: {header}")

        head_token = parts[0].split()[0]
        try:
            start = int(parts[1].strip())
            end = int(parts[2].strip())
            strand = int(parts[3].strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Prodigal coordinates in header: {header}") from exc
        if start < 1 or end < start or strand not in (-1, 1):
            raise ValueError(f"Invalid Prodigal location in header: {header}")

        metadata_text = parts[4] if len(parts) >= 5 else ""
        metadata = _parse_metadata(metadata_text)
        matched = ID_PATTERN.search(header)
        local_id = matched.group(1) if matched else f"{head_token}_{start}_{end}"
        orf_id = f"{file_stem}_{local_id}"
        if orf_id in seen_ids:
            raise ValueError(f"Duplicate Prodigal ORF ID: {orf_id}")
        seen_ids.add(orf_id)

        translation = str(record.seq).replace(" ", "").rstrip("*")
        if not translation:
            raise ValueError(f"Prodigal ORF has no translation: {orf_id}")
        parsed.append(
            ProdigalCDS(
                contig=_contig_key(head_token),
                orf_id=orf_id,
                start=start,
                end=end,
                strand=strand,
                translation=translation,
                metadata=metadata,
            )
        )
    if not parsed:
        raise RuntimeError(f"No Prodigal proteins parsed from: {path}")
    return parsed


def _optional_int(value: object) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None
    return int(text)


def _parse_bgc_range(value: str) -> Tuple[str, int, int]:
    try:
        contig, interval = str(value).strip().split(":", 1)
        start_text, end_text = interval.rsplit("-", 1)
        start, end = int(start_text), int(end_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid BGC nucleotide range: {value!r}") from exc
    if not contig or start < 1 or end < start:
        raise ValueError(f"Invalid BGC nucleotide range: {value!r}")
    return contig, start, end


def parse_regions_csv(regions_csv: str) -> List[PredictedRegion]:
    path = Path(regions_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"BGC regions CSV not found: {path}")

    regions: List[PredictedRegion] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for number, row in enumerate(reader, start=1):
            raw_range = str(row.get("bgc_range") or "").strip()
            if not raw_range:
                continue
            contig, start, end = _parse_bgc_range(raw_range)
            raw_classes = str(row.get("class_name") or "").strip()
            classes = tuple(item.strip() for item in raw_classes.split(";") if item.strip())
            if not classes or classes == ("none",):
                classes = ("unknown",)
            try:
                confidence = float(row.get("confidence") or 0.0)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid confidence in row {number}: {row.get('confidence')!r}") from exc
            regions.append(
                PredictedRegion(
                    number=number,
                    contig=contig,
                    start=start,
                    end=end,
                    class_names=classes,
                    confidence=confidence,
                    cds_ids=tuple(str(row.get("cds_ids") or "").split()),
                    start_cds=_optional_int(row.get("start_cds")),
                    end_cds=_optional_int(row.get("end_cds")),
                )
            )
    return regions


def _source_feature(length: int) -> SeqFeature:
    return SeqFeature(
        FeatureLocation(0, length, strand=1),
        type="source",
        qualifiers={
            "organism": ["unclassified organism"],
            "mol_type": ["genomic DNA"],
            "note": ["Input sequence analysed by RegionBGC"],
        },
    )


def _region_qualifiers(region: PredictedRegion) -> Dict[str, List[str]]:
    qualifiers = {
        "tool": ["RegionBGC"],
        "region_number": [str(region.number)],
        "product": list(region.class_names),
        "confidence": [f"{region.confidence:.6f}"],
        "note": [
            "RegionBGC predicted biosynthetic gene cluster; "
            f"original_location={region.contig}:{region.start}-{region.end}"
        ],
    }
    if region.start_cds is not None:
        qualifiers["start_cds"] = [str(region.start_cds)]
    if region.end_cds is not None:
        qualifiers["end_cds"] = [str(region.end_cds)]
    if region.cds_ids:
        qualifiers["cds_ids"] = [" ".join(region.cds_ids)]
    return qualifiers


def _cds_feature(cds: ProdigalCDS, *, offset: int = 0) -> SeqFeature:
    start = cds.start - 1 - offset
    end = cds.end - offset
    if start < 0 or end <= start:
        raise ValueError(f"Invalid shifted CDS location for {cds.orf_id}: {start}..{end}")

    notes = ["CDS predicted by Prodigal"]
    metadata_text = ";".join(
        f"{key}={value}" if value else key for key, value in cds.metadata.items()
    )
    if metadata_text:
        notes.append(metadata_text)
    return SeqFeature(
        FeatureLocation(start, end, strand=cds.strand),
        type="CDS",
        qualifiers={
            "locus_tag": [cds.orf_id],
            "protein_id": [cds.orf_id],
            "product": ["hypothetical protein"],
            "codon_start": ["1"],
            "transl_table": ["11"],
            "translation": [cds.translation],
            "note": notes,
        },
    )


def _feature_sort_key(feature: SeqFeature) -> Tuple[int, int, int]:
    priority = {"source": 0, "region": 1, "CDS": 2}.get(feature.type, 3)
    return int(feature.location.start), priority, int(feature.location.end)


def _validate_record_locations(
    records: Iterable[SeqRecord],
    cds_by_contig: Mapping[str, Sequence[ProdigalCDS]],
    regions_by_contig: Mapping[str, Sequence[PredictedRegion]],
) -> None:
    record_lengths = {record.id: len(record) for record in records}
    unknown_cds_contigs = sorted(set(cds_by_contig) - set(record_lengths))
    unknown_region_contigs = sorted(set(regions_by_contig) - set(record_lengths))
    if unknown_cds_contigs:
        raise KeyError(f"Prodigal contigs not found in DNA FASTA: {unknown_cds_contigs}")
    if unknown_region_contigs:
        raise KeyError(f"Prediction contigs not found in DNA FASTA: {unknown_region_contigs}")

    for contig, cdses in cds_by_contig.items():
        for cds in cdses:
            if cds.end > record_lengths[contig]:
                raise ValueError(
                    f"CDS {cds.orf_id} ends outside contig {contig}: "
                    f"{cds.end} > {record_lengths[contig]}"
                )
    for contig, regions in regions_by_contig.items():
        for region in regions:
            if region.end > record_lengths[contig]:
                raise ValueError(
                    f"Region {region.number} ends outside contig {contig}: "
                    f"{region.end} > {record_lengths[contig]}"
                )


def _safe_filename(value: str) -> str:
    safe = UNSAFE_FILENAME_PATTERN.sub("_", value).strip("._")
    return safe or "record"


def _write_and_validate(records: Sequence[SeqRecord], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = SeqIO.write(records, str(output_path), "genbank")
    if written != len(records):
        raise RuntimeError(f"Expected to write {len(records)} GenBank records, wrote {written}")

    parsed = list(SeqIO.parse(str(output_path), "genbank"))
    if len(parsed) != len(records):
        raise RuntimeError(f"GenBank round-trip failed for: {output_path}")
    for expected, restored in zip(records, parsed):
        if expected.id != restored.id or len(expected) != len(restored):
            raise RuntimeError(f"GenBank record changed during round-trip: {output_path}")
    return str(output_path)


def write_prediction_genbanks(
    *,
    dna_fasta: str,
    protein_faa: str,
    regions_csv: str,
    output_dir: str,
    sample_name: Optional[str] = None,
) -> Dict[str, object]:
    """Write one GenBank per predicted BGC; no whole-genome GenBank is produced."""
    dna_path = Path(dna_fasta).expanduser().resolve()
    if not dna_path.exists():
        raise FileNotFoundError(f"DNA FASTA not found: {dna_path}")
    output_path = Path(output_dir).expanduser().resolve()
    regions = parse_regions_csv(regions_csv)
    if not regions:
        return {"region_gbk_paths": []}

    with dna_path.open() as handle:
        records = list(SeqIO.parse(handle, "fasta"))
    if not records:
        raise RuntimeError(f"No DNA records parsed from: {dna_path}")

    cds_by_contig: Dict[str, List[ProdigalCDS]] = defaultdict(list)
    cds_by_id: Dict[str, ProdigalCDS] = {}
    for cds in parse_prodigal_faa(protein_faa):
        cds_by_contig[cds.contig].append(cds)
        cds_by_id[cds.orf_id] = cds

    regions_by_contig: Dict[str, List[PredictedRegion]] = defaultdict(list)
    for region in regions:
        regions_by_contig[region.contig].append(region)

    _validate_record_locations(records, cds_by_contig, regions_by_contig)

    records_by_id = {record.id: record for record in records}
    region_gbks: List[str] = []
    for region in regions:
        parent = records_by_id[region.contig]
        offset = region.start - 1
        region_length = region.end - region.start + 1
        region_record = SeqRecord(
            parent.seq[offset:region.end],
            id=parent.id,
            name=parent.name,
            description=(
                f"RegionBGC predicted region {region.number} from "
                f"{region.contig}:{region.start}-{region.end}"
            ),
            annotations={
                "molecule_type": "DNA",
                "topology": "linear",
                "data_file_division": "UNK",
                "organism": parent.annotations.get("organism", "unclassified organism"),
                "source": parent.annotations.get("source", "unclassified organism"),
                "structured_comment": {
                    "RegionBGC-Data": {
                        "NOTE": "Predicted region extracted from a larger record",
                        "Original start": str(region.start),
                        "Original end": str(region.end),
                    }
                },
            },
        )
        region_record.features = [
            _source_feature(region_length),
            SeqFeature(
                FeatureLocation(0, region_length),
                type="region",
                qualifiers=_region_qualifiers(region),
            ),
        ]

        selected_ids = set(region.cds_ids)
        if selected_ids:
            missing_ids = sorted(selected_ids - set(cds_by_id))
            if missing_ids:
                raise KeyError(
                    f"Region {region.number} references unknown CDS IDs: {missing_ids}"
                )
            selected_cds = [cds_by_id[orf_id] for orf_id in region.cds_ids]
        else:
            selected_cds = [
                cds for cds in cds_by_contig.get(region.contig, [])
                if cds.start >= region.start and cds.end <= region.end
            ]
        for cds in selected_cds:
            if cds.contig != region.contig or cds.start < region.start or cds.end > region.end:
                raise ValueError(
                    f"CDS {cds.orf_id} is not contained by region {region.number}"
                )
            region_record.features.append(_cds_feature(cds, offset=offset))
        region_record.features.sort(key=_feature_sort_key)

        filename = f"{_safe_filename(region.contig)}.region{region.number:03d}.gbk"
        region_gbks.append(
            _write_and_validate([region_record], output_path / "regions" / filename)
        )

    return {"region_gbk_paths": region_gbks}


__all__ = [
    "PredictedRegion",
    "ProdigalCDS",
    "parse_prodigal_faa",
    "parse_regions_csv",
    "write_prediction_genbanks",
]
