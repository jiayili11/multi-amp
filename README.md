# MultiAMP: Multi-Stream Deep Learning for Antimicrobial Peptide Prediction

Antimicrobial resistance (AMR) is accelerating worldwide, undermining frontline antibiotics and making the need for novel agents more urgent than ever.
Antimicrobial peptides (AMPs) are promising therapeutics against multidrug-resistant pathogens, as they are less prone to inducing resistance.
However, current AMP prediction approaches often treat sequence and structure in isolation and at a single scale, leading to mediocre performance. Here, we propose MultiAMP, a framework that integrates multi-level information for predicting AMPs. The model captures evolutionary and contextual information from sequences alongside global and fine-grained information from structures, synergistically combining these features to enhance predictive power.
MultiAMP achieves state-of-the-art performance, outperforming existing AMP prediction methods by over 10% in MCC when identifying distant AMPs sharing less than 40% sequence identity with known AMPs.
To discover novel AMPs, we applied MultiAMP to marine organism data, discovering 484 high-confidence peptides with sequences that are highly divergent from known AMPs. Notably, MultiAMP accurately recognizes various structural types of peptides.
In addition, our approach reveals functional patterns of AMPs, providing interpretable insights into their mechanisms. Building on these findings, we further employed a gradient-based strategy and achieved the design of AMPs with specific motifs.
We believe that MultiAMP empowers both the rational discovery and mechanistic understanding of AMPs, facilitating future experimental validation and precision therapeutic design.

## Architecture

**PeptideTriStreamModel** — a three-stream architecture:

| Stream | Input | Module |
|--------|-------|--------|
| Stream 1 | Amino acid sequence | ESM-2 (650M) with multi-scale layer fusion |
| Stream 2 | Amino acid sequence | BiLSTM shallow encoder |
| Stream 3 | 3D structure (PDB) | GVP-GNN (Geometric Vector Perceptron Graph Neural Network) |

The three streams are fused via **Cross-Attention** + **Transformer Encoder** + **Gated Fusion**, producing:
- **AMP/non-AMP classification** (BCEWithLogitsLoss)
- **Contrastive learning** embeddings (SupConLoss)
- **Secondary structure reconstruction** (Focal + CRF loss)

## Setup

### Environment

```bash
conda create -n amppre python=3.9
conda activate amppre
pip install torch torchvision torchaudio
pip install fair-esm biopython torch-geometric torchcrf scikit-learn pandas tqdm
```

### Data & Checkpoints

Trained models and data are available on Hugging Face:

**https://huggingface.co/jiayi11/multi_amp**

Place downloaded checkpoints (e.g. `best_model_overall.pth`) in the `checkpoints/` directory.

See [data/README.md](data/README.md) for details on data format and directory structure.

```
data/
├── train_amp/          # Training FASTA files (11,970)
├── test_amp/           # Test FASTA files (5,355)
└── structure/          # ESMFold PDB structures
    ├── amp_train5985/
    ├── nonamp_train5985/
    ├── amp_testset/
    └── nonamp_testset/
```

## Usage

### 1. Training

```bash
python train.py --gpu 1 --epochs 30 --batch_size 16
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--gpu` | `1` | GPU device ID |
| `--epochs` | config (30) | Number of training epochs |
| `--batch_size` | config (16) | Batch size |
| `--lr` | config (5e-5) | Head learning rate |
| `--save_dir` | `./checkpoints/` | Checkpoint save directory |

The training script saves the best model (by validation AUC) to `checkpoints/best_model_overall.pth`.

### 2. Prediction

**Mode A: Predict from FASTA file** (sequence-only, no PDB needed)

```bash
python predict.py --gpu 0 \
    --model_path ./checkpoints/best_model_overall.pth \
    --fasta_path ./input/test_sequences.fasta \
    --output_path ./input/predictions.csv
```

**Mode B: Evaluate on validation dataset** (with full structural features)

```bash
python predict.py --gpu 0 --model_path ./checkpoints/best_model_overall.pth
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--gpu` | `1` | GPU device ID |
| `--model_path` | `checkpoints/best_model.pth` | Path to model checkpoint |
| `--fasta_path` | None | FASTA file for sequence-only prediction |
| `--output_path` | auto | Output CSV path |
| `--batch_size` | config (16) | Batch size |

**Output CSV columns** (FASTA mode): `id, sequence, length, probability, prediction`

### 3. De Novo AMP Design

Generate novel AMP sequences via Gumbel-Softmax optimization:

```bash
python design.py --gpu 0 \
    --mode de_novo \
    --model_path ./checkpoints/best_model_overall.pth \
    --n_sequences 500 \
    --length 25 \
    --iterations 100 \
    --top_k 10 \
    --output_dir ./design_results
```

### 4. Motif-Guided AMP Design

Design AMP sequences with a fixed functional motif:

```bash
python design.py --gpu 0 \
    --mode motif \
    --model_path ./checkpoints/best_model_overall.pth \
    --motif KLLKLLK \
    --motif_start 5 \
    --n_sequences 100 \
    --length 25 \
    --iterations 100 \
    --top_k 10 \
    --output_dir ./design_results
```

**Design Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `de_novo` | Design mode: `de_novo` or `motif` |
| `--model_path` | `checkpoints/best_model.pth` | Path to model checkpoint |
| `--n_sequences` | `10` | Number of candidates to generate |
| `--length` | `25` | Target sequence length |
| `--iterations` | `50` | Optimization iterations per sequence |
| `--top_k` | `5` | Number of top candidates to report |
| `--motif` | `KLLKLLK` | Motif sequence (motif mode only) |
| `--motif_start` | `5` | Motif starting position (motif mode only) |
| `--output_dir` | `./design_results` | Output directory |
| `--gpu` | `1` | GPU device ID |

## Project Structure

```
multiamp/
├── config.py           # Model and training configuration
├── model.py            # PeptideTriStreamModel architecture
├── dataset.py          # PeptideDataset with geometric feature extraction
├── losses.py           # SupConLoss, StructureAwareLoss (Focal + CRF)
├── train.py            # Training script
├── predict.py          # Prediction / evaluation script
├── design.py           # De novo and motif-guided AMP design
├── utils.py            # Utility functions
├── checkpoints/        # Saved model weights
├── data/               # Training and test data
├── input/              # User input files (FASTA)
└── design_results/     # Design output files
```
