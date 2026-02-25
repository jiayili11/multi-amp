# MultiAMP Data

## Directory Structure

```
data/
├── train_amp/          # Training set FASTA files (11,970 files)
├── test_amp/           # Test set FASTA files (5,355 files)
└── structure/          # 3D structure data (ESMFold predictions)
    ├── amp_train5985/          # Training AMP PDB files (5,983 files)
    ├── nonamp_train5985/       # Training non-AMP PDB files (5,985 files)
    ├── amp_testset/            # Test AMP PDB files (1,234 files)
    ├── nonamp_testset/         # Test non-AMP PDB files (4,121 files)
    ├── AMPs_5985_ss_predictions.txt          # Training AMP secondary structure predictions
    ├── non_AMPs_5985_ss_predictions.txt      # Training non-AMP secondary structure predictions
    ├── test_amp_ss_predictions.txt           # Test set secondary structure predictions
    └── uniprot_marine_sim.csv                # Marine peptide sequence similarity alignment
```

## FASTA File Format

Each `.fas` file contains one peptide sequence in the following format:

```
>ID|LABEL
AMINO_ACID_SEQUENCE
SECONDARY_STRUCTURE
```

- **ID**: Sequence identifier (e.g., `8`, `CUTTED3796`)
- **LABEL**: `1` = AMP (antimicrobial peptide), `0` = non-AMP
- **AMINO_ACID_SEQUENCE**: Standard 20 amino acids + X (unknown)
- **SECONDARY_STRUCTURE**: Three-state secondary structure labels
  - `H` = alpha-helix
  - `E` = beta-strand
  - `C` = coil

Example:
```
>8|1
KVVVKWVVKVVK
CEEEEEEEECCC
```

## PDB Files

3D structures are predicted by ESMFold. Filenames correspond to the sequence IDs in FASTA files.
Used to extract geometric features for the GVP-GNN stream:
- CA/CB atom coordinates
- Backbone dihedral angles (phi, psi, omega)
- Local curvature, solvent-accessible surface area (SASA)
- Residue contact maps
- KNN graph, sequential edges, SS-motif edges

## Dataset Split

| Split    | AMP    | non-AMP | Total   |
|----------|--------|---------|---------|
| Training | ~5,985 | ~5,985  | 11,970  |
| Test     | ~1,234 | ~4,121  | 5,355   |

> Note: A small number of training PDB files are missing or fail to parse (~551 records), resulting in ~11,419 effective training samples.
