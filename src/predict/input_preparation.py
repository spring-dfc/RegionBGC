from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple


NUCLEOTIDE_FASTA_SUFFIXES = {".fa", ".fasta", ".fna", ".ffn", ".fas"}


def discover_nucleotide_files(input_path: str) -> List[Path]:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    if path.is_file():
        if path.suffix.lower() not in NUCLEOTIDE_FASTA_SUFFIXES:
            raise ValueError(f"Unsupported nucleotide FASTA suffix: {path.suffix}")
        return [path]

    files = sorted(
        item.resolve()
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in NUCLEOTIDE_FASTA_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No nucleotide FASTA files found under: {path}")
    return files


def _validate_unique_stems(files: Iterable[Path]) -> None:
    stems = [path.stem for path in files]
    duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicates:
        raise ValueError(f"Input file names must have unique stems: {duplicates}")


def _copy_fasta(source: Path, destination: Path, rename_contigs: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rename_contigs:
        shutil.copy2(source, destination)
        return

    record_index = 0
    with source.open("r", encoding="utf-8", errors="ignore") as src, destination.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if line.startswith(">"):
                record_index += 1
                dst.write(f">{source.stem}_{record_index}\n")
            else:
                dst.write(line)
    if record_index == 0:
        raise ValueError(f"No FASTA records found in: {source}")


def prepare_nucleotide_inputs(
    files: List[Path],
    work_dir: Path,
    *,
    rename_contigs: bool,
) -> Tuple[Path, Path]:
    """Create safe DNA working copies and translate them with Prodigal."""
    if shutil.which("prodigal") is None:
        raise EnvironmentError(
            "Prodigal was not found in PATH. Install it first, for example: "
            "conda install -c bioconda prodigal"
        )
    _validate_unique_stems(files)
    dna_dir = work_dir / "nucleotide"
    protein_dir = work_dir / "protein"
    dna_dir.mkdir(parents=True, exist_ok=True)
    protein_dir.mkdir(parents=True, exist_ok=True)

    for source in files:
        dna_path = dna_dir / f"{source.stem}.fna"
        protein_path = protein_dir / f"{source.stem}.faa"
        _copy_fasta(source, dna_path, rename_contigs=rename_contigs)
        command = [
            "prodigal", "-i", str(dna_path), "-a", str(protein_path),
            "-o", str(protein_dir / f"{source.stem}.genes"),
            "-f", "gbk", "-p", "meta", "-q",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Prodigal failed for {source}: {detail}")
    return protein_dir, dna_dir
