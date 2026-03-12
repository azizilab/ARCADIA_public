# %% This script preprocesses MaxFuse tonsil dataset:

# Key Operations:
# Download tonsil dataset from MaxFuse paper (if not already present)
# Load tonsil dataset from MaxFuse paper:
# RNA: tonsil_rna_counts.txt with metadata and gene names
# Protein: tonsil_codex.csv (spatial proteomics data)
# Filter to mutual cell types between RNA and protein datasets
# Quality control and outlier removal using MAD (Median Absolute Deviation)
# Normalize protein data using z-normalization (without log1p transformation)
# Select highly variable genes for RNA data (using knee detection)
# Normalize RNA data using normalize_total (target_sum=40000) + log1p transformation
# Perform spatial analysis on protein data
# Save processed data with timestamps

# Outputs:
# preprocessed_adata_rna_[timestamp].h5ad
# preprocessed_adata_prot_[timestamp].h5ad


# %% --- Imports and Config ---
import json
import os

# Add src to path for arcadia package
import sys
import warnings
from datetime import datetime
from pathlib import Path

import scipy.sparse as sp

# Suppress pkg_resources deprecation warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="louvain")


# Helper function to work in both scripts and notebooks
def here():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


# Determine ROOT based on whether we're running as script or notebook
try:
    # Running as script - use __file__ to find root
    _script_dir = Path(__file__).resolve().parent
    ROOT = _script_dir.parent  # scripts/ -> root
except NameError:
    # Running as notebook - use current working directory
    # papermill sets cwd to the script's directory (scripts/)
    _cwd = Path.cwd()
    if _cwd.name == "scripts":
        ROOT = _cwd.parent
    else:
        ROOT = _cwd

THIS_DIR = here()

# Add src to path for arcadia package
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Update sys.path and cwd
sys.path.append(str(ROOT))
sys.path.append(str(THIS_DIR))
os.chdir(str(ROOT))


import matplotlib as mpl
import numpy as np
import scanpy as sc
from scipy.sparse import issparse

from arcadia.data_utils import (
    download_tonsil_data,
    filter_unwanted_cell_types,
    load_tonsil_protein,
    load_tonsil_rna,
    mad_outlier_removal,
    preprocess_rna_initial_steps,
    save_processed_data,
    z_normalize_codex,
)
from arcadia.plotting import general as pf
from arcadia.plotting import preprocessing as pp_plots
from arcadia.plotting import spatial as spatial_plots
from arcadia.utils import metadata as pipeline_metadata_utils

mpl.rcParams.update(
    {
        "savefig.format": "pdf",
        "figure.figsize": (6, 4),
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": False,
    }
)
sc.set_figure_params(
    scanpy=True, fontsize=10, dpi=100, dpi_save=300, vector_friendly=False, format="pdf"
)
# Load config_ if exists
config_path = Path("configs/config.json")
if config_path.exists():
    with open(config_path, "r") as f:
        config_ = json.load(f)
    plot_flag = config_["plot_flag"]
else:
    plot_flag = True

# %% --- Timestamp (no plot directories) ---
FILENAME = "_0_preprocess_tonsil.py"
dataset_name = "tonsil"
start_time = datetime.now()
timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")

sc.settings.set_figure_params(dpi=50, facecolor="white")

# %% --- Directory Structure Definition ---
os.makedirs("processed_data", exist_ok=True)


# %% --- MaxFuse Data Download and Loading Functions ---
# %% --- Load MaxFuse Data ---
# Setup paths
data_dir = Path("raw_datasets/tonsil")
data_dir.mkdir(parents=True, exist_ok=True)

# Download and load data
download_tonsil_data(data_dir)
adata_rna = load_tonsil_rna(data_dir)
adata_prot = load_tonsil_protein(data_dir)

print(f"Loaded RNA data: {adata_rna.shape}")
print(f"Loaded protein data: {adata_prot.shape}")

# Initialize pipeline metadata with start time
adata_rna.uns["pipeline_metadata"] = pipeline_metadata_utils.initialize_pipeline_metadata(
    timestamp_str, FILENAME, dataset_name
)

# %% Convert to CSR if needed
if sp.isspmatrix_coo(adata_rna.X):
    adata_rna.X = adata_rna.X.tocsr()
    print("Converted to CSR")

# Add layers for raw data and counts (needed for downstream processing)
adata_rna.layers["raw"] = adata_rna.X.copy()
adata_prot.layers["raw"] = adata_prot.X.copy()

# Add counts layer (essential for downstream pipeline compatibility)
adata_rna.layers["counts"] = adata_rna.X.copy()
adata_prot.layers["counts"] = adata_prot.X.copy()
print("Added 'counts' layers for both RNA and protein datasets")

# Check sparsity
total_elements = adata_rna.X.shape[0] * adata_rna.X.shape[1]
zero_elements = total_elements - adata_rna.X.nnz
print(f"Number of zero elements proportion in RNA dataset: {zero_elements/total_elements}")

# Verify data is count data
if issparse(adata_rna.X):
    assert np.allclose(adata_rna.X.data, np.round(adata_rna.X.data))
else:
    assert np.allclose(adata_rna.X, np.round(adata_rna.X))

# %% --- Initial Analysis and Plots ---
pp_plots.plot_count_distribution(adata_rna, plot_flag)
pp_plots.plot_expression_heatmap(adata_rna, plot_flag)

# MaxFuse-style initial data overview plots
if plot_flag:
    # Compute PCA for both datasets for initial visualization
    sc.tl.pca(adata_rna)
    sc.tl.pca(adata_prot)

    pf.plot_data_overview(adata_rna, adata_prot, plot_flag=plot_flag)
    pf.plot_cell_type_distribution(adata_rna, adata_prot, plot_flag=plot_flag)
    spatial_plots.plot_spatial_data(adata_prot, plot_flag=plot_flag)

# Print cell types in both datasets
print(f'RNA dataset cell types: {sorted(set(adata_rna.obs["cell_types"]))}')
print(f'Protein dataset cell types: {sorted(set(adata_prot.obs["cell_types"]))}')

# Add total_counts to both datasets
print(f"\nAdding total_counts to both datasets...")
if issparse(adata_rna.X):
    adata_rna.obs["total_counts"] = np.array(adata_rna.X.sum(axis=1)).flatten()
else:
    adata_rna.obs["total_counts"] = adata_rna.X.sum(axis=1)

if issparse(adata_prot.X):
    adata_prot.obs["total_counts"] = np.array(adata_prot.X.sum(axis=1)).flatten()
else:
    adata_prot.obs["total_counts"] = adata_prot.X.sum(axis=1)

print(
    f'RNA dataset total_counts - Mean: {adata_rna.obs["total_counts"].mean():.2f}, Std: {adata_rna.obs["total_counts"].std():.2f}'
)
print(
    f'Protein dataset total_counts - Mean: {adata_prot.obs["total_counts"].mean():.2f}, Std: {adata_prot.obs["total_counts"].std():.2f}'
)

# %% --- Cell Type Filtering and Harmonization ---
# Remove problematic cell types using the utility function
print("\n=== Filtering unwanted cell types ===")
print("RNA dataset:")
adata_rna = filter_unwanted_cell_types(adata_rna, ["tumor", "dead", "nk cells"])
print("Protein dataset:")
adata_prot = filter_unwanted_cell_types(adata_prot, ["tumor", "dead", "nk cells"])

# Get common cell types
common_cell_types = set(adata_rna.obs["cell_types"]) & set(adata_prot.obs["cell_types"])
print(f"Common cell types: {sorted(common_cell_types)}")

# Filter to common cell types
adata_rna = adata_rna[adata_rna.obs["cell_types"].isin(common_cell_types)].copy()
adata_prot = adata_prot[adata_prot.obs["cell_types"].isin(common_cell_types)].copy()

print(f"After filtering to common cell types:")
print(f"RNA dataset shape: {adata_rna.shape}")
print(f"Protein dataset shape: {adata_prot.shape}")

# %% --- Apply Initial Preprocessing ---
print("\n=== Applying initial preprocessing ===")
adata_rna_processed = preprocess_rna_initial_steps(
    adata_rna.copy(),
    min_genes=200,
    min_cells=3,
    plot_flag=plot_flag,
)

print(f"Processed RNA dataset shape: {adata_rna_processed.shape}")
print(f"Cell type distribution in processed RNA dataset:")
print(adata_rna_processed.obs["cell_types"].value_counts())

# Update main RNA dataset
adata_rna = adata_rna_processed

# Plot highly variable genes (MaxFuse-style)
if plot_flag:
    sc.pl.highly_variable_genes(adata_rna)

# %% --- Apply Custom Setup for MaxFuse Dataset ---
print(f"\n=== Applying custom setup for MaxFuse dataset ===")
print(f"RNA cell types: {sorted(set(adata_rna.obs['cell_types']))}")
print(f"Protein cell types: {sorted(set(adata_prot.obs['cell_types']))}")

adata_rna.uns["pipeline_metadata"]["preprocess"]["day_of_collection"] = None
adata_rna.uns["pipeline_metadata"]["preprocess"]["patient_id"] = None
adata_rna.uns["pipeline_metadata"]["preprocess"]["zib_threshold"] = None
adata_rna.uns["pipeline_metadata"]["preprocess"]["source_files"] = {
    "rna": "raw_datasets/tonsil/tonsil_rna_counts.txt",
    "protein": "raw_datasets/tonsil/tonsil_codex.csv",
    "metadata": "raw_datasets/tonsil/tonsil_rna_meta.csv",
}
adata_prot.uns["pipeline_metadata"] = adata_rna.uns["pipeline_metadata"].copy()

# Sort by cell types (standard preprocessing step)
adata_rna = adata_rna[adata_rna.obs["cell_types"].argsort(), :].copy()
adata_prot = adata_prot[adata_prot.obs["cell_types"].argsort(), :].copy()

# Remove redundant gene columns if they exist
if "gene" in adata_rna.var.columns and np.array_equal(
    adata_rna.var["gene"].values, (adata_rna.var.index.values)
):
    adata_rna.var.drop(columns="gene", inplace=True)
if "gene" in adata_prot.var.columns and np.array_equal(
    adata_prot.var["gene"].values, (adata_prot.var.index.values)
):
    adata_prot.var.drop(columns="gene", inplace=True)

print(f"Final RNA cell types: {sorted(set(adata_rna.obs['cell_types']))}")
print(f"Final protein cell types: {sorted(set(adata_prot.obs['cell_types']))}")
print("✅ MaxFuse dataset setup completed!")

# %% --- Apply Protein Processing ---
pp_plots.plot_protein_violin(adata_prot, plot_flag)

adata_prot = mad_outlier_removal(adata_prot).copy()
adata_rna.obs["major_cell_types"] = adata_rna.obs["cell_types"]
adata_prot.obs["major_cell_types"] = adata_prot.obs["cell_types"]

# %% --- Protein Analysis ---
pp_plots.plot_protein_analysis(adata_prot, plot_flag=plot_flag)

# Take subsample for spatial analysis
adata_prot_subsampled = adata_prot[
    np.random.choice(adata_prot.n_obs, size=min(6000, adata_prot.n_obs), replace=False)
].copy()

# Apply spatial analysis
sc.pp.pca(adata_prot_subsampled, copy=False)
sc.pp.neighbors(adata_prot_subsampled, use_rep="X_pca")
sc.tl.umap(adata_prot_subsampled)

pp_plots.plot_umap_analysis(adata_prot_subsampled, "cell_types", "Cell Types", plot_flag)


# %% --- Finalize Pipeline Metadata ---
# Create sample names for metadata (tonsil doesn't have multiple samples)
sample_names = ["tonsil_sample"]

pipeline_metadata_utils.finalize_preprocess_metadata(adata_rna, adata_prot, sample_names)

# %% --- Apply Final Normalization and Create Layers ---
print("\n=== Applying final normalization and creating layers ===")

# RNA: normalize_total + log1p, save both versions
print("RNA normalization:")
print(f"  Before normalization - X range: [{adata_rna.X.min():.2f}, {adata_rna.X.max():.2f}]")

# Ensure we have raw counts in layers["counts"] (should already exist from line 288)
if "counts" not in adata_rna.layers:
    adata_rna.layers["counts"] = adata_rna.X.copy()
    print("  Saved raw counts to layers['counts']")

# Apply normalize_total(10,000)
sc.pp.normalize_total(adata_rna, target_sum=40000, inplace=True)
print(f"  After normalize_total - X range: [{adata_rna.X.min():.2f}, {adata_rna.X.max():.2f}]")

# Save normalized counts (before log1p)
adata_rna.layers["normalized"] = adata_rna.X.copy()

# Apply log1p transformation
sc.pp.log1p(adata_rna)
print(f"  After log1p - X range: [{adata_rna.X.min():.2f}, {adata_rna.X.max():.2f}]")

# Save log1p data to layer (backup)
adata_rna.layers["log1p"] = adata_rna.X.copy()
print("  Kept log1p(normalized) data in X for VAE training")

# Update metadata
adata_rna.uns["pipeline_metadata"]["normalization"] = {
    "method": "normalize_total",
    "target_sum": 40000,
    "applied": True,
    "log1p_applied": True,
    "layers_info": {
        "counts": "raw counts (backup only)",
        "normalized": "normalized counts (target_sum=40000, backup only)",
        "log1p": "log1p(normalized counts) - in X for VAE",
    },
    "note": "RNA X contains log1p(normalized) data for VAE training",
}

# Protein: z-normalization WITHOUT log1p
print("\nProtein normalization:")
print(f"  Before normalization - X range: [{adata_prot.X.min():.2f}, {adata_prot.X.max():.2f}]")

# Ensure we have raw counts in layers["counts"]
if "counts" not in adata_prot.layers:
    adata_prot.layers["counts"] = adata_prot.X.copy()
    print("  Saved raw counts to layers['counts']")

# Apply z-normalization WITHOUT log1p for protein
adata_prot = z_normalize_codex(adata_prot.copy(), apply_log1p=False)
print(f"  After z-normalization - X range: [{adata_prot.X.min():.2f}, {adata_prot.X.max():.2f}]")

# Keep z-normalized data in X for VAE training (this is what VAE needs)
# Save backup to layer as well
adata_prot.layers["z_normalized"] = adata_prot.X.copy()
print("  Kept z-normalized data in X for VAE training (protein VAE trains on normalized data)")

# Update metadata
adata_prot.uns["pipeline_metadata"]["normalization"] = {
    "method": "z_normalize",
    "applied": True,
    "log1p_applied": False,
    "layers_info": {
        "counts": "raw counts (backup only)",
        "z_normalized": "z-normalized protein expression (no log1p) - in X for VAE",
    },
    "note": "Protein X contains z-normalized data for VAE training",
}

print("\n✅ Normalization complete:")
print(f"  RNA X: log1p(normalized) (for VAE training), layers: {list(adata_rna.layers.keys())}")
print(f"  Protein X: z-normalized (for VAE training), layers: {list(adata_prot.layers.keys())}")

# %% --- Save Processed Data ---
# Set dataset_name in adata before saving (takes priority over filename extraction)
adata_rna.uns["dataset_name"] = dataset_name
adata_prot.uns["dataset_name"] = dataset_name
save_processed_data(adata_rna, adata_prot, "processed_data", caller_filename=FILENAME)
print("✅ MaxFuse dataset preprocessing completed successfully!")
print(f"Final RNA dataset shape: {adata_rna.shape}")
print(f"Final protein dataset shape: {adata_prot.shape}")
print(f"Common cell types: {sorted(set(adata_rna.obs['cell_types']))}")
print(f"Output saved to: processed_data/")
# %% --- Final Preprocessing Results Plots (MaxFuse-style) ---
if plot_flag:
    print("Generating final preprocessing results plots...")

    # For RNA visualization, use log1p layer
    adata_rna_plot = adata_rna.copy()
    adata_rna_plot.X = adata_rna_plot.layers["log1p"].copy()

    # For protein visualization, use z_normalized layer
    adata_prot_plot = adata_prot.copy()
    adata_prot_plot.X = adata_prot_plot.layers["z_normalized"].copy()

    # Subsample for plotting if datasets are large
    if adata_rna_plot.shape[0] > 5000:
        sc.pp.subsample(adata_rna_plot, n_obs=5000)
    if adata_prot_plot.shape[0] > 5000:
        sc.pp.subsample(adata_prot_plot, n_obs=5000)

    # Spatial and heatmap analysis - RNA
    if adata_rna_plot.n_obs > 6000:
        adata_rna_sub_plot = adata_rna_plot[
            np.random.choice(adata_rna_plot.n_obs, size=6000, replace=False)
        ].copy()
    else:
        adata_rna_sub_plot = adata_rna_plot.copy()

    sc.pp.pca(adata_rna_sub_plot, copy=False)
    sc.pp.neighbors(adata_rna_sub_plot, use_rep="X_pca")
    sc.tl.umap(adata_rna_sub_plot)
    pp_plots.plot_batches_and_conditions(adata_rna_sub_plot, plot_flag, modality="RNA")
    pp_plots.plot_heatmap_analysis(adata_rna_sub_plot, "rna", plot_flag)

    # Spatial and heatmap analysis - Protein
    if adata_prot_plot.n_obs > 6000:
        adata_prot_sub_plot = adata_prot_plot[
            np.random.choice(adata_prot_plot.n_obs, size=6000, replace=False)
        ].copy()
    else:
        adata_prot_sub_plot = adata_prot_plot.copy()

    sc.pp.pca(adata_prot_sub_plot, copy=False)
    sc.pp.neighbors(adata_prot_sub_plot, use_rep="X_pca")
    sc.tl.umap(adata_prot_sub_plot)
    pp_plots.plot_batches_and_conditions(adata_prot_sub_plot, plot_flag, modality="Protein")
    pp_plots.plot_heatmap_analysis(adata_prot_sub_plot, "protein", plot_flag)

    # Final UMAP plots
    sc.pp.pca(adata_rna_plot, copy=False)
    sc.pp.neighbors(adata_rna_plot)
    sc.tl.umap(adata_rna_plot)

    sc.pp.pca(adata_prot_plot, copy=False)
    sc.pp.neighbors(adata_prot_plot)
    sc.tl.umap(adata_prot_plot)

    # Plot final preprocessing results
    pf.plot_preprocessing_results(adata_rna_plot, adata_prot_plot)


# %% Processing
