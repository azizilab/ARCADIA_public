# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ARCADIA (ARchetype-based Clustering and Alignment with Dual Integrative Autoencoders) integrates scRNA-seq and spatial proteomics (CODEX) data using archetype analysis and dual VAEs. It embeds both modalities jointly, preserving cell neighborhood structures and archetype representations.

## Environment

- Conda environment: `scvi` (always activate before running anything)
- Python 3.10, PyTorch + CUDA 12.1, scvi-tools 1.2.2.post2
- scVI requires a patch for custom training plans — run `python -m arcadia.utils.setup_scvi_patch` after install
- Install package in dev mode: `pip install -e .`

## Commands

### Run full pipeline
```bash
conda activate scvi
bash run_pipeline.sh <dataset_name>  # e.g., cite_seq, tonsil
```

### Run individual pipeline steps
```bash
python scripts/_0_preprocess_<dataset_name>.py
python scripts/_1_align_datasets.py --dataset_name <dataset_name>
python scripts/_2_spatial_integrate.py --dataset_name <dataset_name>
python scripts/_3_generate_archetypes.py --dataset_name <dataset_name>
python scripts/_4_prepare_training.py --dataset_name <dataset_name>
python scripts/_5_train_vae.py --dataset_name <dataset_name>
```

### Hyperparameter search
```bash
python scripts/hyperparameter_search.py --dataset_name <dataset_name>
```

### View training results
```bash
mlflow ui  # then open http://localhost:5000
```

### Formatting and linting
```bash
black --line-length 100 <file>
flake8 <file>
mypy <file>
```

### Fast debugging config
In `configs/config.json`, reduce cells and epochs:
```json
{"subsample": {"num_rna_cells": 500, "num_protein_cells": 500}, "plot_flag": false, "training": {"max_epochs": 10}}
```
Set `plot_x_times: 0` in training params to disable all training plots.

## Architecture

### Pipeline (6 steps, sequential)
Each step reads the latest timestamped output from the previous step. Only Step 0 is dataset-specific; Steps 1-5 are dataset-agnostic.

1. **Step 0** (`_0_preprocess_*.py`) — Dataset-specific preprocessing, QC, normalization
2. **Step 1** (`_1_align_datasets.py`) — Balance cell type proportions, HVG selection, batch correction
3. **Step 2** (`_2_spatial_integrate.py`) — Spatial neighbor features, optional COVET, CN label generation
4. **Step 3** (`_3_generate_archetypes.py`) — PCHA archetype detection, cross-modal archetype alignment via cosine distance on cell type proportions
5. **Step 4** (`_4_prepare_training.py`) — Cross-modal cell matching, data conversion to scVI format
6. **Step 5** (`_5_train_vae.py`) — Dual VAE training with custom loss functions

### Source code (`src/arcadia/`)
- `training/dual_vae_training_plan.py` — Core training plan: manages two VAEs (RNA + protein), custom losses (reconstruction, similarity, matching, cell type clustering, CN separation), adaptive similarity weight via iLISI, steady-state detection, checkpointing with MLflow
- `training/gradnorm.py`, `training/loss_scaling.py` — Gradient normalization and loss scaling utilities
- `archetypes/` — PCHA-based archetype generation (`generation.py`), cross-modal matching (`matching.py`), distance metrics (`distances.py`)
- `spatial/` — Spatial neighbor computation (`neighbors.py`), spatial analysis (`analysis.py`)
- `covet/` — COVET feature engineering (optional spatial context features)
- `data_utils/` — Data loading (`loading.py`), preprocessing (`preprocessing.py`), cleaning (`cleaning.py`)
- `plotting/` — Visualization utilities organized by pipeline stage
- `analysis/` — Post-hoc analysis and method comparison utilities

### Configuration
- `configs/config.json` — Main config controlling subsampling, plotting, spatial features, training epochs
- Training hyperparameters are defined inline in `scripts/_5_train_vae.py` and `scripts/hyperparameter_search.py`

### Data flow
All intermediate data is stored as AnnData `.h5ad` files with timestamps in `<dataset_name>/` directories. Each step auto-discovers the latest output from the previous step.

## Code Style Rules (from .cursorrules)

- **Never use try-except blocks** unless specifically told to
- Do not import packages inside functions — imports go at the top
- Write files in cells format (interactive mode `# %%`) unless it's a utility module
- Black formatting, line length 100
- Don't remove chunks of commented code unless asked to clean dead code
- If a function call references a missing function, **do not write that function** — check if it exists and can be imported first
- Use functional programming for data processing; OOP for model architectures
- Prefer vectorized operations over explicit loops
- Prioritize scanpy native plotting functions
- Keep git commit messages very short and concise
