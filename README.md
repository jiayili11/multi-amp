# MultiAMP: Multi-Stream Deep Learning for Antimicrobial Peptide Prediction

MultiAMP is a deep learning framework for antimicrobial peptide (AMP) prediction and design.  
It combines protein language models, sequence models and structure-aware graph neural networks to learn multi-scale peptide representations.

---

## Data and Models

Trained models and data are available on Hugging Face:

- https://huggingface.co/jiayi11/multi_amp

Place downloaded checkpoints (e.g. `best_model_overall.pth`) in the `checkpoints/` directory.

---

## Installation

### Option 1: Using conda (recommended)
```bash
conda env create -f environment.yml
conda activate multiamp
pip install -e .
```
### Option 2: Minimal dependencies

```
pip install torch esm biopython scipy scikit-learn pandas tqdm
pip install -e .
```
---

## Command line

### Prediction from FASTA
```
python scripts/predict.py \
  --model_path checkpoints/best_model_overall.pth \
  --fasta_path examples/example.fasta \
  --output_path results/predictions.csv
```
### De novo design
```
python scripts/design.py \
  --model_path checkpoints/best_model_overall.pth \
  --mode de_novo \
  --n_sequences 500 \
  --length 20 \
  --iterations 100 \
  --top_k 5 \
  --output_dir design_results/denovo
```
### Motif-guided design
```
python scripts/design.py \
  --model_path checkpoints/best_model_overall.pth \
  --mode motif \
  --motif KLLKLLK \
  --n_variants 100 \
  --iterations 100 \
  --top_k 5 \
  --output_dir design_results/motif
```
