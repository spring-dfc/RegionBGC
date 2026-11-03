# RegionBGC

RegionBGC predicts biosynthetic gene cluster (BGC) regions and product types
from nucleotide genomes using ESMC protein embeddings.
Its encoder uses Multi-Scale Transformation (MST), Intra-Scale Feature Interaction
(ISFI), and Cross-Scale Feature Fusion (CSFF); the region query decoder uses
enhanced multi-head cross-attention (EMHCA).

## Installation

```bash
git clone https://github.com/spring-dfc/RegionBGC.git
cd RegionBGC
conda create -n regionbgc python=3.11 -y
conda activate regionbgc
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
conda install -c conda-forge -c bioconda prodigal
```

The commands above use CUDA 12.4. Install the appropriate PyTorch build for your system.

## Model weights

Place the RegionBGC checkpoint downloaded from the project website at
`weights/regionbgc.pth`.
Download the [ESMC-300M weights](https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12)
and place `esmc_300m_2024_12_v0.pth` in `weights/`.

## Prediction

```bash
python tools/predict.py \
  --input /path/to/genomes \
  --output-dir outputs/predictions \
  --checkpoint weights/regionbgc.pth \
  --threshold 0.3 --type-threshold 0.5
```

Input: a nucleotide FASTA file or directory. Main outputs:
`merged_bgc_regions.csv`, `merged_orf_predictions.tsv`, and per-region GenBank files.
Prediction extracts features automatically. Adjust `--threshold` (BGC detection,
default 0.3) and `--type-threshold` (product classification, default 0.5) within [0, 1].
Use `--plot` for figures, `--save-json` for sequence JSON, and `--verify-sequences`
for sequence checks. Verification JSON is removed after passing unless `--save-json`
is set; failed verification retains it for inspection. Use `--help` to view options.

## Training

Place JSON datasets in `data/`, with `genome_id`, ordered `orf_sequences`,
per-gene `bgc_labels`, and `bgc_class` labels. Extract `.h5` features locally
before training:

```bash
for split in train validation; do
  python tools/extract_features/extract_features.py \
    --input-file data/${split}.json --output-file data/${split}.h5 \
    --model-path weights/esmc_300m_2024_12_v0.pth --device cuda:0
done
```

Use the same command with `test.json` to prepare test features when needed.

```bash
python tools/train.py \
  -c configs/regionbgc/regionbgc.yml --seed 6 \
  --train-data data/train.json --train-features data/train.h5 \
  --val-data data/validation.json --val-features data/validation.h5 \
  --output-dir outputs/training
```

Each epoch saves prediction weights; `resume_latest.pth` supports resuming training
with `--resume`. Settings are in `configs/regionbgc/regionbgc.yml`.
For ESM2-8M training, use `-c configs/regionbgc/regionbgc_esm2.yml`
and provide 320-dimensional ESM2 feature files.

## Evaluation

```bash
python tools/evaluate_test.py type --data data/test.json
python tools/evaluate_test.py regions \
  --predictions /path/to/orf_predictions.tsv --truth-data data/test_new.json
```

Product classification uses the training Hungarian matcher and reports micro-Precision,
micro-Recall, micro-F1, and micro-AUROC on matched pairs, with a classification
default threshold of 0.5 (adjustable with `--type-threshold`). Matching has no
detection-score or IoU cutoff; coverage is reported separately.


Metric functions are in `src/metrics/`: `gene.py` for gene-level localization,
`regions.py` for region-level localization, and `classification.py` for multilabel
product classification.
Independent region evaluation uses the TSV `Pred` column and JSON `orf_positions`
(inclusive nucleotide coordinates), with IoU cutoffs of 0.3 and 0.5. Avg. IoU uses
all positive-overlap one-to-one pairs. Training validation uses ORF-index intervals.

## License

[Apache-2.0](LICENSE).
