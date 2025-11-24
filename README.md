# MultiAMP: Multi-Stream Deep Learning for Antimicrobial Peptide Prediction# MultiAMP



A state-of-the-art deep learning framework for antimicrobial peptide (AMP) prediction and de novo design, combining protein language models, structure-aware graph neural networks, and contrastive learning.Multi-stream deep learning for antimicrobial peptide (AMP) prediction and design.



## 🌟 Features## Overview



- **Three-Stream Architecture**: Integrates ESM-2 (650M), Bi-LSTM, and GVP-GNN for comprehensive peptide representationMultiAMP integrates ESM-2 language model, Bi-LSTM, and GVP-GNN to predict AMPs and design novel sequences with high antimicrobial activity.

- **Structure-Aware Learning**: Leverages 3D structural information through geometric vector perceptrons

- **Secondary Structure Reconstruction**: Auxiliary task with CRF-based prediction**Key Features:**

- **Contrastive Learning**: Supervised contrastive loss for better feature discrimination- ✅ High-accuracy AMP prediction (sequence-only mode supported)

- **De Novo Design**: Gumbel-Softmax optimization for novel AMP generation- ✅ De novo AMP design with gradient-based optimization

- ✅ Motif-guided design (preserve functional motifs)

## 📊 Performance- ✅ Structure-guided design (target secondary structures)

- ✅ Large-scale candidate generation (500+ candidates → select top performers)

| Metric | Score |

|--------|-------|## Installation

| AUC | 0.9660 |

| Accuracy | 0.8687 |```bash

| MCC | 0.6642 |# Clone repository

| F1 Score | 0.7372 |git clone https://github.com/jiayili11/multi-amp.git

cd multi-amp

*Results on validation set after 3 epochs*

# Create environment (or use existing amppre environment)

## 🚀 Quick Startconda env create -f environment.yml

conda activate multiamp

### Installation

# Install package

```bashpip install -e .

# Create conda environment```

conda env create -f environment.yml

conda activate amppre## Quick Start



# Or install dependencies### 1. Prediction

pip install torch esm-hub biopython scipy scikit-learn pandas tqdm

``````python

from multiamp import MultiAMPPredictor

### Data Structure

# Load model

```predictor = MultiAMPPredictor('checkpoints/best_model_overall.pth')

multiamp/

├── data/# Predict single sequence

│   ├── train_amp/          # Training FASTA files (10,000 sequences)prob = predictor.predict('KLLKLLKKLLKLLK')

│   ├── test_amp/           # Validation FASTA files (5,355 sequences)print(f'AMP Probability: {prob:.3f}')

│   └── structure/          # PDB structures

│       ├── amp_train5985/# Predict multiple sequences

│       ├── nonamp_train5985/sequences = ['KLLKLLKKLLKLLK', 'GIGKFLHSAKKFGKAFVGEIMNS']

│       ├── amp_testset/probs = predictor.predict(sequences)

│       └── nonamp_testset/

├── checkpoints/            # Model checkpoints# Predict from FASTA file

└── design_results/         # Design outputsresults = predictor.predict_fasta('examples/example.fasta')

``````



### Training### 2. Design



```bash#### De Novo Design

python train.pyGenerate 500 candidates, select top 5:

```

```bash

Training configuration:python scripts/design.py \

- Batch size: 16    --model_path checkpoints/best_model_overall.pth \

- Epochs: 30    --mode de_novo \

- ESM-2 warmup: 5 epochs    --n_sequences 500 \

- Learning rate: 1e-5 (PLM), 5e-5 (heads)    --length 20 \

- Mixed precision training (AMP)    --iterations 100 \

    --top_k 5 \

### Prediction    --output_dir ./designs/denovo

```

```bash

python predict.py#### Motif-Guided Design

```Generate 100 variants per motif, select top 5:



Outputs:```bash

- Predictions saved to `checkpoints/predictions.csv`python scripts/design.py \

- Metrics: Accuracy, AUC, F1, Precision, Recall, MCC    --model_path checkpoints/best_model_overall.pth \

    --mode motif \

### De Novo Design    --motif KLLKLLK \

    --n_variants 100 \

```bash    --flank_length 10 \

python design.py    --iterations 100 \

```    --top_k 5 \

    --output_dir ./designs/motif

Two design modes:```

1. **De Novo**: Generate novel AMPs from scratch

2. **Motif-Guided**: Design AMPs with fixed functional motifsUse predefined motif libraries:

```bash

Results saved to `design_results/`# Helix motifs: KLAKKLA, KLLKLLK, KAAKKAA, KLAKLAK, KLGKKLG

python scripts/design.py --mode motif --motif_type helix --n_variants 100 ...

## 🏗️ Architecture

# Sheet motifs: GIGKFLH, FVQWFSK, KWKSFI, RWLRWLR, VQWRAIRVRVIR

### Model Componentspython scripts/design.py --mode motif --motif_type sheet --n_variants 100 ...



1. **ESM-2 Stream** (1280-dim)# All motifs

   - Pretrained protein language model (facebook/esm2_t33_650M_UR50D)python scripts/design.py --mode motif --motif_type all --n_variants 100 ...

   - Fine-tuned with last 6 layers unfrozen```



2. **LSTM Stream** (128-dim embedding + 2-layer Bi-LSTM)#### Structure-Guided Design

   - Raw amino acid sequence encoding```bash

   - Bidirectional context modelingpython scripts/design.py \

    --model_path checkpoints/best_model_overall.pth \

3. **GVP-GNN Stream** (384-dim hidden, 512-dim output)    --mode structure \

   - Geometric Vector Perceptrons for 3D structure    --motif KLLKLLK \

   - 3-layer graph neural network    --structure HHHHHHH \

   - K-NN (k=15) + contact map edges    --n_variants 100 \

    --output_dir ./designs/structure

### Loss Functions```



- **Classification Loss**: Binary cross-entropy with logits### 3. Training

- **Contrastive Loss**: Supervised contrastive learning (τ=0.07)

- **SS Reconstruction Loss**: Focal loss + CRF + continuity regularization```bash

python scripts/train.py \

Weights: W_class=1.0, W_contrast=0.1, W_ss=0.15    --data_dir /path/to/training/data \

    --output_dir ./checkpoints \

## 📁 Core Files    --epochs 50 \

    --batch_size 16 \

| File | Description | Source |    --learning_rate 1e-4

|------|-------------|--------|```

| `model.py` | PeptideTriStreamModel implementation | Copied from amppre |

| `dataset.py` | Dataset loading and PDB parsing | Copied from amppre |### 4. Testing

| `losses.py` | Loss functions (SupConLoss, StructureAwareLoss) | Copied from amppre |

| `utils.py` | Helper functions | Copied from amppre |```bash

| `config.py` | Centralized configuration | New |python scripts/test.py \

| `train.py` | Training script | Simplified from amppre |    --model_path checkpoints/best_model_overall.pth \

| `predict.py` | Prediction/evaluation script | Simplified from amppre |    --data_dir /path/to/test/data \

| `design.py` | De novo design script | Simplified from amppre |    --output_dir ./results \

    --batch_size 32

## ⚙️ Configuration```



Edit `config.py` to customize:## Model Architecture

- Model architecture switches (ESM2, GVP, LSTM, SS reconstruction)

- Training hyperparameters (batch size, learning rate, epochs)MultiAMP uses a three-stream architecture:

- Loss weights (classification, contrastive, reconstruction)

- Data paths and checkpoints1. **ESM-2 Stream**: Pre-trained protein language model (650M params) for evolutionary context

2. **Bi-LSTM Stream**: Captures sequential patterns

## 🔬 Technical Details3. **GVP-GNN Stream**: Geometric Vector Perceptrons for 3D structure (when available)



### Data Processing**Multi-Task Learning:**

- Primary: AMP classification

- **FASTA Format**: Directory-based loading with individual `.fas` files- Auxiliary: Secondary structure prediction

- **Structure**: PDB files parsed for CA atom coordinates- Contrastive learning for enhanced discrimination

- **Secondary Structure**: DSSP-based annotation (H/C/E)

- **Padding**: Dynamic padding to max sequence length (1024)**Design Strategy (Improved):**

- Gumbel-Softmax for differentiable sequence optimization

### Training Strategy- Diversity regularization to avoid mode collapse

- Large-scale candidate generation (500+) followed by selection

1. **Warmup Phase** (0-5 epochs): ESM-2 frozen- Optimized hyperparameters: lr=0.015, temperature=0.5, 100 iterations

2. **Fine-tuning Phase** (5-30 epochs): Last 6 layers unfrozen

3. **Scheduler**: ReduceLROnPlateau (patience=6, factor=0.5)## Repository Structure

4. **Gradient Clipping**: Max norm 1.0

```

### Design Algorithmmultiamp/

├── multiamp/               # Main package

- **Gumbel-Softmax**: Differentiable discrete sampling│   ├── model.py           # Three-stream model

- **Temperature Annealing**: Initial τ=0.5│   ├── dataset.py         # Data processing

- **Iterations**: 50-100 per sequence│   ├── predictor.py       # Prediction interface

- **Regularization**: Diversity loss to prevent mode collapse│   ├── designer.py        # Design interface

│   ├── design_utils.py    # Optimization utilities

## 📝 Requirements│   ├── losses.py          # Loss functions

│   └── config.py          # Configuration

- Python 3.9+├── scripts/               # Command-line tools

- PyTorch 2.0+│   ├── design.py         # Design pipeline

- CUDA-capable GPU (recommended: 24GB+ VRAM)│   ├── train.py          # Training script

- ESM-2 checkpoint (~2.7GB)│   └── test.py           # Testing script

├── checkpoints/          # Model checkpoints

## 📄 License│   └── best_model_overall.pth  (2.7GB)

├── examples/             # Example data

This project is based on the amppre framework. Please refer to the original repository for licensing information.└── README.md

```

## 🙏 Acknowledgments

## Design Parameters

Core model implementation adapted from the amppre project. This repository provides a streamlined interface for training, prediction, and de novo design tasks.

For optimal results (based on generate_figure5_data.py):

## 📧 Contact

**De Novo Design:**

For questions or issues, please open an issue on GitHub.- `n_sequences`: 500 (generate many candidates)

- `n_iterations`: 100 (optimization steps per candidate)

---- `learning_rate`: 0.015

- `temperature`: 0.5 (Gumbel-Softmax)

**Note**: Ensure sufficient GPU memory (>= 24GB) for training with ESM-2 650M model. For systems with limited memory, consider reducing batch size or using gradient accumulation.- `top_k`: 5 (return top 5)


**Motif-Guided Design:**
- `n_variants`: 100 (variants per motif)
- `n_iterations`: 100
- `learning_rate`: 0.015
- `temperature`: 0.5
- `top_k`: 5

## Citation

If you use MultiAMP in your research, please cite:

```bibtex
@article{multiamp2024,
  title={MultiAMP: Multi-scale Sequence-Structure Modeling for Antimicrobial Peptide Prediction and Design},
  author={Your Name},
  journal={Journal},
  year={2024}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note:** The model checkpoint (`best_model_overall.pth`, 2.7GB) is required for prediction and design. Make sure it's placed in the `checkpoints/` directory.
