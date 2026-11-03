from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
    from matplotlib.transforms import blended_transform_factory
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None


def parse_fasta_contig_layout(
    fasta_path: str,
) -> Tuple[Dict[str, int], int, List[int], List[Tuple[str, int, int]]]:
    fasta_file = Path(fasta_path).expanduser().resolve()
    if not fasta_file.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")

    contig_entries: List[Tuple[str, int]] = []
    current_id = None
    seq_len = 0

    with open(fasta_file, "r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    contig_entries.append((current_id, seq_len))
                current_id = line[1:].split()[0]
                seq_len = 0
            elif current_id is not None:
                seq_len += len(line)

    if current_id is not None:
        contig_entries.append((current_id, seq_len))

    if not contig_entries:
        raise RuntimeError(f"No contigs parsed from FASTA: {fasta_file}")

    contig_offset: Dict[str, int] = {}
    boundaries: List[int] = []
    contig_spans: List[Tuple[str, int, int]] = []
    offset = 0
    for contig_id, length in contig_entries:
        start_bp = offset
        contig_offset[contig_id] = start_bp
        offset += length
        end_bp = offset
        boundaries.append(end_bp)
        contig_spans.append((contig_id, start_bp, end_bp))

    return contig_offset, offset, boundaries, contig_spans


def load_bgc_regions(regions_csv: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with open(regions_csv, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bgc_range = str(row.get("bgc_range") or "").strip()
            if not bgc_range:
                continue
            try:
                contig_part, coord_part = bgc_range.split(":", 1)
                start_s, end_s = coord_part.split("-", 1)
                start_bp = int(start_s)
                end_bp = int(end_s)
            except Exception:
                continue
            if end_bp < start_bp:
                start_bp, end_bp = end_bp, start_bp
            out.append(
                {
                    "contig": contig_part,
                    "start_bp": start_bp,
                    "end_bp": end_bp,
                    "class_name": str(row.get("class_name") or "").strip(),
                    "confidence": str(row.get("confidence") or "").strip(),
                }
            )
    return out


def _to_global_regions(
    raw_regions: List[Dict[str, object]],
    contig_offset: Dict[str, int],
) -> List[Dict[str, object]]:
    global_regions: List[Dict[str, object]] = []
    for item in raw_regions:
        contig_id = str(item["contig"])
        offset = contig_offset.get(contig_id)
        if offset is None:
            match = re.search(r"(\d+)$", contig_id)
            if match:
                suffix = match.group(1)
                for key, value in contig_offset.items():
                    if key.endswith(f"_{suffix}"):
                        offset = value
                        break
        if offset is None:
            continue
        global_regions.append(
            {
                **item,
                "global_start": int(offset) + int(item["start_bp"]),
                "global_end": int(offset) + int(item["end_bp"]),
            }
        )

    global_regions.sort(key=lambda x: int(x["global_start"]))
    return global_regions


def _kb_formatter(x, pos=None):
    return f"{x / 1000.0:.0f}kb"


def _contig_block_label(contig_id: str, order_idx_1based: int) -> str:
    match = re.search(r"(?:_|^)(\d+)$", contig_id)
    if match:
        return f"_{match.group(1)}"
    return f"_{order_idx_1based}"


def plot_bgc_predictions(
    regions_csv: str,
    dna_fasta: str,
    output_png: str,
    title: str | None = None,
    dpi: int = 250,
) -> str:
    if plt is None:
        raise RuntimeError("matplotlib is not available, cannot generate a RegionBGC plot.")

    contig_offset, total_len, boundaries, contig_spans = parse_fasta_contig_layout(dna_fasta)
    raw_regions = load_bgc_regions(regions_csv)
    regions = _to_global_regions(raw_regions, contig_offset)

    width_inch = max(12.0, min(32.0, total_len / 70000.0 if total_len > 0 else 12.0))
    fig, ax = plt.subplots(1, 1, figsize=(width_inch, 2.8), dpi=dpi)

    for boundary in boundaries:
        ax.axvline(boundary, color="#dddddd", linestyle="--", linewidth=0.6, alpha=0.9)

    baseline_y = 0.55
    ax.hlines(y=baseline_y, xmin=0, xmax=max(total_len, 1), colors="#bbbbbb", linewidth=2.0, alpha=0.8)

    if regions:
        for item in regions:
            ax.hlines(
                y=baseline_y,
                xmin=int(item["global_start"]),
                xmax=int(item["global_end"]),
                colors="#d62728",
                linewidth=6,
                alpha=0.95,
            )
    else:
        ax.text(0.02, 0.75, "No BGC regions predicted", transform=ax.transAxes, fontsize=9, color="gray")

    if contig_spans:
        x_to_axes_y = blended_transform_factory(ax.transData, ax.transAxes)
        for i, (contig_id, start_bp, end_bp) in enumerate(contig_spans, start=1):
            if end_bp <= start_bp:
                continue
            ax.text(
                (start_bp + end_bp) / 2.0,
                0.04,
                _contig_block_label(contig_id, i),
                transform=x_to_axes_y,
                ha="center",
                va="bottom",
                fontsize=6,
                color="#444444",
            )

    ax.set_xlim(0, max(total_len, 1))
    ax.set_ylim(0, 1.0)
    ax.set_yticks([baseline_y])
    ax.set_yticklabels(["RegionBGC"], fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=14))
    ax.xaxis.set_minor_locator(MultipleLocator(50_000))
    ax.xaxis.set_major_formatter(FuncFormatter(_kb_formatter))
    ax.tick_params(axis="x", which="major", labelsize=7)
    ax.tick_params(axis="x", which="minor", labelbottom=False, length=3)
    ax.set_xlabel("Genomic position (kb) across concatenated contigs", fontsize=8)
    ax.set_title(title or f"RegionBGC prediction: {Path(dna_fasta).stem}", fontsize=10, loc="left")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout(pad=0.6)
    output_path = Path(output_png).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return str(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot RegionBGC predictions on the original genome FASTA")
    parser.add_argument("--regions_csv", required=True, help="Path to bgc_regions.csv")
    parser.add_argument("--dna_fasta", required=True, help="Path to the original genome DNA FASTA")
    parser.add_argument("--output_png", required=True, help="Output PNG path")
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument("--dpi", type=int, default=250, help="Output figure DPI")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_png = plot_bgc_predictions(
        regions_csv=args.regions_csv,
        dna_fasta=args.dna_fasta,
        output_png=args.output_png,
        title=args.title,
        dpi=args.dpi,
    )
    print(f"RegionBGC plot saved to: {output_png}")


if __name__ == "__main__":
    main()
