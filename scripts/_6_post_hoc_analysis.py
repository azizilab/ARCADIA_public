#!/usr/bin/env python
# coding: utf-8

# %% Setup and Configuration

# Add src to path for arcadia package
import sys
from pathlib import Path

# Handle __file__ for both script and notebook execution
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Running as notebook - use current working directory
    # papermill sets cwd to the script directory
    ROOT = Path.cwd().resolve().parent
    if not (ROOT / "src").exists():
        # Try parent if we're in scripts/
        if (ROOT.parent / "src").exists():
            ROOT = ROOT.parent

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

"""Train VAE with archetypes vectors."""
"""Post-hoc analysis script to load latest checkpoint data and calculate CN iLISI metrics."""
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import scanpy as sc

from arcadia.analysis.post_hoc_utils import (
    assign_rna_cn_from_protein,
    load_checkpoint_data,
    load_models_from_checkpoint,
)
from arcadia.data_utils.cleaning import clean_uns_for_h5ad
from arcadia.plotting.post_hoc import plot_latent_space_mixing
from arcadia.training.utils import validate_scvi_training_mixin
from arcadia.utils.args import find_checkpoint_from_experiment_name
from arcadia.utils.logging import logger

FILENAME = "_6_post_hoc_analysis.py"

# Handle notebook execution
p = globals().get("__vsc_ipynb_file__")
if p:
    __file__ = p

# Set working directory to project root
_script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = _script_dir.parent
os.chdir(ROOT)

# Create log directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Configure plotting
mpl.rcParams.update(
    {
        "savefig.format": "pdf",
        "figure.figsize": (6, 4),
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)
sc.set_figure_params(
    scanpy=True, fontsize=10, dpi=100, dpi_save=300, vector_friendly=False, format="pdf"
)

# Validate scVI training mixin before proceeding
validate_scvi_training_mixin()

pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 10)
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=100)

config_path = Path(str(ROOT / "configs" / "config.json"))
if config_path.exists():
    with open(config_path, "r") as f:
        config_ = json.load(f)
    num_rna_cells = config_["subsample"]["num_rna_cells"]
    num_protein_cells = config_["subsample"]["num_protein_cells"]
    plot_flag = config_["plot_flag"]
else:
    num_rna_cells = num_protein_cells = 2000
    plot_flag = True

# load /home/barroz/projects/ARCADIA/CODEX_RNA_seq/data/processed_data/maxfuse_tonsil/3_rna_2025-10-22-01-33-22.h5ad


# %% Load Checkpoint Data

experiment_name = "cite_seq"  # todo: make as input arg
logger.info(f"Finding latest checkpoint from experiment: {experiment_name}")
checkpoint_path, experiment_id, run_id = find_checkpoint_from_experiment_name(experiment_name)
logger.info(f"Found checkpoint: {checkpoint_path}")
logger.info(f"Experiment ID: {experiment_id}, Run ID: {run_id}")
config_path = Path(checkpoint_path).parent.parent / "model_config.json"
checkpoint_folder = Path(checkpoint_path)
adata_rna, adata_prot = load_checkpoint_data(checkpoint_folder)
sc.pp.subsample(adata_rna, n_obs=min(len(adata_rna), num_rna_cells))


# %% Align Data Indices

if (
    adata_rna.obs_names.isin(adata_prot.obs_names).sum() > 2000
):  # if we are dealing with adata with the same index
    adata_prot = adata_prot[adata_prot.obs_names.isin(adata_rna.obs_names)]
    adata_rna = adata_rna[adata_rna.obs_names.isin(adata_prot.obs_names)]
    common_index = set(adata_prot.obs_names).intersection(set(adata_rna.obs_names))
    if len(common_index) != len(adata_rna.obs_names) or len(common_index) != len(
        adata_prot.obs_names
    ):
        raise ValueError("Mismatched indices between RNA and protein data")
    adata_rna.obs["CN"] = adata_prot.obs["CN"].loc[adata_rna.obs_names].values
    sorted_index = (
        adata_prot.obs.reset_index()
        .sort_values(["cell_types", "CN", "index"])
        .set_index("index")
        .index
    )
    adata_rna = adata_rna[sorted_index, :].copy()
    adata_prot = adata_prot[sorted_index, :].copy()
    cn_accuracy = (adata_rna.obs["CN"].astype(str) == adata_prot.obs["CN"].astype(str)).sum() / len(
        common_index
    )
    adata_rna.obs["CN_matched"] = adata_rna.obs["CN"].astype(str) == adata_prot.obs["CN"].astype(
        str
    )
    adata_prot.obs["CN_matched"] = adata_prot.obs["CN"].astype(str) == adata_rna.obs["CN"].astype(
        str
    )
    if adata_rna.obs.index.equals(adata_prot.obs.index):
        print("RNA and protein data have the same indices")
    else:
        raise ValueError("RNA and protein data have different indices")
else:
    sorted_index_prot = (
        adata_prot.obs.reset_index()
        .sort_values(["cell_types", "CN", "index"])
        .set_index("index")
        .index
    )
    sorted_index_rna = (
        adata_rna.obs.reset_index()
        .sort_values(["cell_types", "CN", "index"])
        .set_index("index")
        .index
    )
    adata_rna = adata_rna[sorted_index_rna, :].copy()
    adata_prot = adata_prot[sorted_index_prot, :].copy()
    common_index = set(adata_rna.obs_names).intersection(set(adata_prot.obs_names))


# %%

logger.info("Loading models from checkpoint...")
rna_vae, protein_vae = load_models_from_checkpoint(
    checkpoint_folder, adata_rna, adata_prot, Path(config_path)
)


# %%


# %%

adata_rna, dists = assign_rna_cn_from_protein(adata_rna, adata_prot)
combined_latent_comparison = plot_latent_space_mixing(adata_rna, adata_prot)
# save the adata files
clean_uns_for_h5ad(adata_rna)
clean_uns_for_h5ad(adata_prot)
save_path_rna = ROOT / f"adata_rna_{adata_rna.uns.get('dataset_name', 'latest')}.h5ad"
save_path_prot = ROOT / f"adata_prot_{adata_prot.uns.get('dataset_name', 'latest')}.h5ad"
adata_rna.write(save_path_rna)
adata_prot.write(save_path_prot)
logger.info(f"Saved adata files to root folder:")
logger.info(f"  - {save_path_rna}")
logger.info(f"  - {save_path_prot}")
