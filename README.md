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

Place `train.json`, `validation.json`, `test.json`, and `test_9genomes.json`
(nine-genome test set) in `data/`, with `genome_id`, ordered `orf_sequences`,
per-gene `bgc_labels`, and `bgc_class` labels. Extract `.h5` features locally
before training:

```bash
for split in train validation; do
  python tools/extract_features/extract_features.py \
    --input-file data/${split}.json --output-file data/${split}.h5 \
    --model-path weights/esmc_300m_2024_12_v0.pth --device cuda:0
done
```

Use the same command for `test.json` or `test_9genomes.json` when needed,
saving features as `test.h5` or `test_9genomes.h5`, respectively.

```bash
python tools/train.py \
  -c configs/regionbgc/regionbgc.yml --seed 6 \
  --train-data data/train.json --train-features data/train.h5 \
  --val-data data/validation.json --val-features data/validation.h5 \
  --output-dir outputs/training
```
`resume_latest.pth` supports resuming training
with `--resume`. 
## License

[Apache-2.0](LICENSE).
