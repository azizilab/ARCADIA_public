# %% This script preprocesses CITE-seq data using synthetic spatial data generation:

# Key Operations:
# Load CITE-seq spleen lymph node dataset using scVI data loader
# Extract RNA and protein data from different batches (following archetype methodology)
# Apply identical preprocessing steps as MaxFuse datasets
# Map minor cell types to major cell types for both modalities
# Filter to mutual cell types between RNA and protein datasets
# Quality control and outlier removal using MAD (Median Absolute Deviation)
# Generate synthetic spatial coordinates for protein data using minor cell type spatial segregation
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


# Helper function to work in both scripts and notebooks
def here():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


FILENAME = "_0_preprocess_cite_seq.py"
dataset_name = "cite_seq"
import scipy.sparse as sp

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="louvain")

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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

from arcadia.data_utils import (
    filter_unwanted_cell_types,
    load_cite_seq_data,
    load_cite_seq_protein,
    load_cite_seq_rna,
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
start_time = datetime.now()
timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")

sc.settings.set_figure_params(dpi=50, facecolor="white")

# %% --- Directory Structure Definition ---
# Base path relative to the project root
base_path = "raw_datasets"

# Core directories to be created under the base path
# Sub-directories are handled automatically by os.makedirs
os.makedirs("processed_data", exist_ok=True)


# %% --- Load CITE-seq Data ---
# available_batches = ["SLN111-D1",  "SLN208-D1", "SLN111-D2","SLN208-D2"]
selected_batches = ["SLN111-D1", "SLN208-D1"]
adata = load_cite_seq_data(batches=selected_batches)
# todo remove this if we want to use batches
adata.obs["batch"] = "batch_1"  # ignore bacthes for now
# temporary subsample to 10k cells
adata = adata[~adata.obs["cell_types"].isna()].copy()
# subsampel the cell tpyes Mature B  for the size fo
# sc.pp.subsample(adata, n_obs=5000)
# Initialize pipeline metadata with start time
adata.uns["pipeline_metadata"] = pipeline_metadata_utils.initialize_pipeline_metadata(
    timestamp_str, FILENAME, dataset_name
)
adata.uns["pipeline_metadata"]["applied_batch_correction"] = (
    True if all(["-D1" in batch for batch in selected_batches]) else False
)

# %% --- Cell Type Mapping and Major Cell Type Assignment ---
print("\n=== Creating cell type mappings ===")

# Create cell type mapping following CITE-seq archetype methodology
# Based on exact mapping from archetype_generation_cite_seq.py
# Use proper major cell type mapping (as you requested)
# Major cell types become the primary cell_types, minor ones used for spatial placement only

# Proper major cell type mapping (as originally intended)
# All minor cell types from the value_counts are mapped here
# cell_type_mapping = {
#     "Activated CD4 T": "CD4 T",
#     "B1 B": "B cells",
#     "CD122+ CD8 T": "CD8 T",
#     "CD4 T": "CD4 T",
#     "CD8 T": "CD8 T",
#     "Erythrocytes": "RBC",
#     "GD T": "T cells",
#     "ICOS-high Tregs": "Treg",
#     "Ifit3-high B": "B cells",
#     "Ifit3-high CD4 T": "CD4 T",
#     "Ifit3-high CD8 T": "CD8 T",
#     "Ly6-high mono": "Monocytes",
#     "Ly6-low mono": "Monocytes",
#     "MZ B": "B cells",
#     "MZ/Marco-high macrophages": "Macrophages",
#     "Mature B": "B cells",
#     "Migratory DCs": "cDCs",
#     "NK": "NK",
#     "NKT": "T cells",
#     "Neutrophils": "Neutrophils",
#     "Plasma B": "B cells",
#     "Red-pulp macrophages": "Macrophages",
#     "Transitional B": "B cells",
#     "Tregs": "Treg",
#     "cDC1s": "cDCs",
#     "cDC2s": "cDCs",
#     "pDCs": "pDCs",
#     # Major cell types mapping to themselves (in case they already exist)
#     "Macrophages": "Macrophages",
#     "B cells": "B cells",
#     "T cells": "T cells",
#     "Treg": "Treg",
#     "cDCs": "cDCs",
#     "Monocytes": "Monocytes",
#     "RBC": "RBC",
# }
cell_type_mapping = {
    "CD4 T": "CD4 T",
    "Activated CD4 T": "CD4 T",
    "Ifit3-high CD4 T": "CD4 T",
    "CD8 T": "CD8 T",
    "Ifit3-high CD8 T": "CD8 T",
    "CD122+ CD8 T": "CD8 T",
    "GD T": "T cells",
    "NKT": "T cells",
    "Mature B": "B cells",
    "Transitional B": "B cells",
    "Ifit3-high B": "B cells",
    "MZ B": "B cells",
    # "B1 B": "B cells",
    # "Plasma B": "B cells",
    "Migratory DCs": "cDCs",
    "cDC1s": "cDCs",
    "cDC2s": "cDCs",
    # "Ly6-high mono": "Monocytes",
    # "Ly6-low mono": "Monocytes",
    # "MZ/Marco-high macrophages": "Macrophages",
    # "Red-pulp macrophages": "Macrophages",
    # "pDCs": "pDCs",
    # "ICOS-high Tregs": "Treg",
    # "Erythrocytes": "RBC",
}


# Apply the mapping
print("Applying cell type mapping")

# Drop all minor cell types that are not in the mapping dictionary
print(f"Original data shape: {adata.shape}")
mapped_cell_types = set(cell_type_mapping.keys())
current_cell_types = set(adata.obs["cell_types"].unique())
unmapped_types = current_cell_types - mapped_cell_types

if unmapped_types:
    print(f"Dropping cells with unmapped cell types: {unmapped_types}")
    mask = adata.obs["cell_types"].isin(mapped_cell_types)
    adata = adata[mask].copy()
    print(f"Data shape after filtering: {adata.shape}")

# Store the original minor cell types BEFORE mapping to major cell types
adata.obs["minor_cell_types"] = adata.obs["cell_types"].copy()  # Store original fine-grained types
adata.obs["cell_types"] = pd.Categorical(
    adata.obs["cell_types"].map(cell_type_mapping)
)  # Major types become primary

# Create major_to_minor_dict for spatial function
major_to_minor_dict = {}
for minor, major in cell_type_mapping.items():
    if major not in major_to_minor_dict:
        major_to_minor_dict[major] = []
    major_to_minor_dict[major].append(minor)

# Print cell type mapping summary
print("Cell type mapping created:")
for major_type, minor_list in major_to_minor_dict.items():
    print(f"  {major_type}: {minor_list}")

# %% --- Create Color Scheme for Major and Minor Cell Types ---
print("\n=== Creating color scheme for cell types ===")

import matplotlib.colors as mcolors
from matplotlib.colors import to_hex

# Get unique major cell types
major_cell_types = list(major_to_minor_dict.keys())
print(f"Major cell types: {major_cell_types}")

# Create base colors for major cell types using scanpy default colors
major_colors = dict(
    zip(
        major_cell_types,
        [to_hex(c) for c in plt.cm.tab20(np.linspace(0, 1, len(major_cell_types)))],
    )
)

print("Major cell type colors:")
for major_type, color in major_colors.items():
    print(f"  {major_type}: {color}")

# Create minor cell type colors as shades of major cell type colors
minor_colors = {}

for major_type, minor_list in major_to_minor_dict.items():
    base_color = major_colors[major_type]

    if len(minor_list) == 1:
        # If only one minor type, use the base color
        minor_colors[minor_list[0]] = base_color
    else:
        # Create shades of the base color for multiple minor types
        # Convert hex to RGB
        rgb = mcolors.hex2color(base_color)

        # Create lighter and darker shades
        shades = []
        for i, minor_type in enumerate(minor_list):
            # Create shades by adjusting brightness
            # Scale factor ranges from 0.4 (darker) to 1.3 (lighter)
            factor = 0.4 + (0.9 / (len(minor_list) - 1)) * i if len(minor_list) > 1 else 1.0

            # Adjust RGB values
            new_rgb = tuple(min(1.0, max(0.0, c * factor)) for c in rgb)
            shade_hex = mcolors.to_hex(new_rgb)
            shades.append(shade_hex)
            minor_colors[minor_type] = shade_hex

# %% # --- Load and Process RNA and Protein Data ---
adata_rna = load_cite_seq_rna(adata)
adata_prot = load_cite_seq_protein(adata, major_to_minor_dict)

print(f"Loaded RNA data: {adata_rna.shape}")
print(f"Loaded protein data: {adata_prot.shape}")

print("\n=== Applying cell type strategy to both datasets ===")

# %% Apply mapping
print("Applying cell type mapping")

# Store the original minor cell types for both datasets (keep originals)
# Note: minor_cell_types should already be set from the initial loading
# Just verify they exist, don't overwrite them
if "minor_cell_types" not in adata_rna.obs.columns:
    adata_rna.obs["minor_cell_types"] = adata_rna.obs["cell_types"].copy()
if "minor_cell_types" not in adata_prot.obs.columns:
    adata_prot.obs["minor_cell_types"] = adata_prot.obs["cell_types"].copy()

# Check if there are minor cell types that are not in the cell type mapping
rna_unmapped_types = set(adata_rna.obs["minor_cell_types"].unique()) - set(cell_type_mapping.keys())
prot_unmapped_types = set(adata_prot.obs["minor_cell_types"].unique()) - set(
    cell_type_mapping.keys()
)

if rna_unmapped_types:
    raise ValueError(f"RNA dataset contains unmapped cell types: {rna_unmapped_types}")

if prot_unmapped_types:
    raise ValueError(f"Protein dataset contains unmapped cell types: {prot_unmapped_types}")

# Map to major cell types (these become the primary cell_types)
# Use the original minor_cell_types (which are the fine-grained types) for mapping
adata_rna.obs["cell_types"] = pd.Categorical(
    adata_rna.obs["minor_cell_types"].map(cell_type_mapping)
)
adata_prot.obs["cell_types"] = pd.Categorical(
    adata_prot.obs["minor_cell_types"].map(cell_type_mapping)
)

# Check for any unmapped values and handle NaNs
rna_unmapped = adata_rna.obs["cell_types"].isna().sum()
prot_unmapped = adata_prot.obs["cell_types"].isna().sum()
# %% Processing
if rna_unmapped > 0:
    print(f"Warning: {rna_unmapped} RNA cells with unmapped cell types")
    print(
        f"Unmapped cell types: {adata_rna.obs['cell_types'][adata_rna.obs['cell_types'].isna()].value_counts()}"
    )
    adata_rna = adata_rna[~adata_rna.obs["cell_types"].isna()].copy()

if prot_unmapped > 0:
    print(f"Warning: {prot_unmapped} protein cells with unmapped cell types")
    print(
        f"Unmapped cell types: {adata_prot.obs['cell_types'][adata_prot.obs['cell_types'].isna()].value_counts()}"
    )
    adata_prot = adata_prot[~adata_prot.obs["cell_types"].isna()].copy()
# remove any cell types with less than 100 cells
cell_type_counts_rna = adata_rna.obs["cell_types"].value_counts()
valid_cell_types_rna = cell_type_counts_rna[cell_type_counts_rna > 100].index
adata_rna = adata_rna[adata_rna.obs["cell_types"].isin(valid_cell_types_rna)].copy()

cell_type_counts_prot = adata_prot.obs["cell_types"].value_counts()
valid_cell_types_prot = cell_type_counts_prot[cell_type_counts_prot > 100].index
adata_prot = adata_prot[adata_prot.obs["cell_types"].isin(valid_cell_types_prot)].copy()

print(f"RNA dataset - Major cell types: {adata_rna.obs['cell_types'].value_counts()}")
print(f"Protein dataset - Major cell types: {adata_prot.obs['cell_types'].value_counts()}")

# %% --- Apply Color Scheme to Datasets ---
print("\n=== Applying color scheme to datasets ===")

# Apply colors to both datasets
for adata_temp, modality_name in [(adata_rna, "RNA"), (adata_prot, "Protein")]:
    # Set major cell type colors
    adata_temp.uns["cell_types_colors"] = [
        major_colors[ct] for ct in adata_temp.obs["cell_types"].cat.categories
    ]

    # Set minor cell type colors
    adata_temp.uns["minor_cell_types_colors"] = [
        minor_colors[ct] for ct in adata_temp.obs["minor_cell_types"].cat.categories
    ]

    print(f"{modality_name} dataset:")
    print(f"  Major cell types: {list(adata_temp.obs['cell_types'].cat.categories)}")
    print(f"  Minor cell types: {list(adata_temp.obs['minor_cell_types'].cat.categories)}")
    print(
        f"  Applied {len(adata_temp.uns['cell_types_colors'])} major colors and {len(adata_temp.uns['minor_cell_types_colors'])} minor colors"
    )

print("✅ Color scheme applied successfully!")

# %% %%
# Convert to CSR if needed
if sp.isspmatrix_coo(adata_rna.X):
    adata_rna.X = adata_rna.X.tocsr()
    print("Converted to CSR")

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

# CITE-seq-style initial data overview plots
if plot_flag:
    # Compute PCA for both datasets for initial visualization
    sc.tl.pca(adata_rna)
    sc.tl.pca(adata_prot)

    pf.plot_data_overview(adata_rna, adata_prot, plot_flag=plot_flag)
    pf.plot_cell_type_distribution(adata_rna, adata_prot, plot_flag=plot_flag)
    spatial_plots.plot_spatial_data(adata_prot, plot_flag=plot_flag)

    # Plot spatial distribution showing all minor cell types and their spatial segregation
    # This shows the synthetic spatial placement created by add_spatial_data_to_prot
    print("=== Plotting Minor Cell Types Spatial Distribution ===")
    spatial_plots.plot_minor_cell_types_spatial_distribution(adata_prot, plot_flag=plot_flag)

    # Also create a plot showing the major cell type mapping
    print("\n=== Cell Type Mapping Summary ===")
    mapping_summary = {}
    for major_type in adata_prot.obs["cell_types"].unique():
        major_cells = adata_prot[adata_prot.obs["cell_types"] == major_type]
        minor_types = major_cells.obs["minor_cell_types"].unique()
        mapping_summary[major_type] = list(minor_types)
        print(f"{major_type}: {list(minor_types)}")

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

# %% --- Generate UMAP with Custom Colors ---
if plot_flag:
    adata_prot_plot = adata_prot[
        np.random.choice(adata_prot.n_obs, size=min(2000, adata_prot.n_obs), replace=False)
    ].copy()
    print("\n=== Generating UMAP with custom color scheme ===")
    sc.pp.pca(adata_prot_plot)
    sc.pp.neighbors(adata_prot_plot, use_rep="X_pca")
    sc.tl.umap(adata_prot_plot)

    # Plot UMAP with custom colors - the colors are automatically used from .uns
    sc.pl.umap(
        adata_prot_plot,
        color=["cell_types", "minor_cell_types"],
        legend_fontsize=8,
        ncols=2,
        title=["Protein - Major Cell Types", "Protein - Minor Cell Types (Shaded by Major)"],
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
print("\n=== Applying initial preprocessing (MaxFuse methodology) ===")
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

# Plot highly variable genes (CITE-seq-style)
if plot_flag:
    sc.pl.highly_variable_genes(adata_rna)

# %% --- Apply Custom Setup for CITE-seq Dataset ---
print(f"\n=== Applying custom setup for CITE-seq dataset ===")
print(f"RNA cell types: {sorted(set(adata_rna.obs['cell_types']))}")
print(f"Protein cell types: {sorted(set(adata_prot.obs['cell_types']))}")

# Initialize pipeline metadata (like setup_dataset but without problematic cell type mapping)
# Preserve applied_batch_correction if it exists from earlier initialization
applied_batch_correction = adata_rna.uns.get("pipeline_metadata", {}).get(
    "applied_batch_correction", None
)
if applied_batch_correction is not None:
    adata_rna.uns["pipeline_metadata"]["applied_batch_correction"] = applied_batch_correction
adata_rna.uns["pipeline_metadata"]["preprocess"]["day_of_collection"] = None
adata_rna.uns["pipeline_metadata"]["preprocess"]["patient_id"] = None
adata_rna.uns["pipeline_metadata"]["preprocess"]["zib_threshold"] = None
adata_rna.uns["pipeline_metadata"]["preprocess"]["source_files"] = {
    "rna": "raw_datasets/cite_seq/spleen_lymph_cite_seq.h5ad",
    "protein": "raw_datasets/cite_seq/spleen_lymph_cite_seq.h5ad",
    "metadata": "raw_datasets/cite_seq/spleen_lymph_cite_seq.h5ad",
}
adata_prot.uns["pipeline_metadata"] = adata_rna.uns["pipeline_metadata"].copy()

# Sort by cell types (standard preprocessing step)
minor_cell_type_key = (
    "minor_cell_types" if "minor_cell_types" in adata_rna.obs.columns else "cell_types"
)
adata_rna = adata_rna[adata_rna.obs[minor_cell_type_key].argsort(), :].copy()
adata_prot = adata_prot[adata_prot.obs[minor_cell_type_key].argsort(), :].copy()

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
print("✅ CITE-seq dataset setup completed!")
# %% --- Apply MaxFuse-style Protein Processing ---
adata_rna = filter_unwanted_cell_types(adata_rna, ["tumor", "dead"])
adata_prot = filter_unwanted_cell_types(adata_prot, ["tumor", "dead"])
pp_plots.plot_protein_violin(adata_prot, plot_flag)

if plot_flag:
    plt.figure(figsize=(10, 6))
    plt.title("Protein expression before outlier removal")
    adata_prot_plot = adata_prot[
        np.random.choice(adata_prot.n_obs, size=2000, replace=False)
    ].copy()
    # sort by cell_types
    adata_prot_plot = adata_prot_plot[adata_prot_plot.obs["cell_types"].argsort(), :].copy()
    plt.plot(np.sort(adata_prot_plot.X[:, :50], axis=0))
    plt.show()
    plt.close()

adata_prot = mad_outlier_removal(adata_prot.copy()).copy()

if plot_flag:
    plt.figure(figsize=(10, 6))
    plt.title("Protein expression after outlier removal")
    plt.plot(np.sort(adata_prot.X[:, :50], axis=0))
    plt.show()
    plt.close()

# Add major_cell_types column (needed for downstream compatibility)
adata_rna.obs["major_cell_types"] = adata_rna.obs["cell_types"]
adata_prot.obs["major_cell_types"] = adata_prot.obs["cell_types"]

# %% --- Protein Analysis Following MaxFuse Methodology ---
pp_plots.plot_protein_analysis(adata_prot, plot_flag=plot_flag, modality="Protein")

# Take subsample for spatial analysis
adata_prot_subsampled = adata_prot[
    np.random.choice(adata_prot.n_obs, size=min(6000, adata_prot.n_obs), replace=False)
].copy()

# Apply spatial analysis (following MaxFuse methodology)
sc.pp.pca(adata_prot_subsampled, copy=False)
sc.pp.neighbors(adata_prot_subsampled, use_rep="X_pca")
sc.tl.umap(adata_prot_subsampled)

pp_plots.plot_umap_analysis(adata_prot_subsampled, "cell_types", "Cell Types", plot_flag)


# %% --- Finalize Pipeline Metadata ---
# Create sample names for metadata (cite-seq doesn't have multiple samples)
sample_names = ["cite_seq_sample"]

pipeline_metadata_utils.finalize_preprocess_metadata(adata_rna, adata_prot, sample_names)

adata_prot.obs["cell_types"].unique()
# Remove duplicates from adata_prot.obs.index and adata_prot.var.index
print(f"Before removing duplicates: {adata_prot.shape[0]} cells, {adata_prot.shape[1]} features")
print(f"Obs index is unique: {adata_prot.obs.index.is_unique}")
print(f"Var index is unique: {adata_prot.var.index.is_unique}")

# Remove duplicate obs indices
if not adata_prot.obs.index.is_unique:
    duplicate_obs = adata_prot.obs.index.duplicated(keep="first")
    print(f"Found {duplicate_obs.sum()} duplicate obs indices")
    adata_prot = adata_prot[~duplicate_obs, :].copy()
    print(f"After removing obs duplicates: {adata_prot.shape[0]} cells")
    print(f"Obs index is now unique: {adata_prot.obs.index.is_unique}")
else:
    print("No duplicates found in obs index")

# Remove duplicate var indices
if not adata_prot.var.index.is_unique:
    duplicate_var = adata_prot.var.index.duplicated(keep="first")
    print(f"Found {duplicate_var.sum()} duplicate var indices")
    adata_prot = adata_prot[:, ~duplicate_var].copy()
    print(f"After removing var duplicates: {adata_prot.shape[1]} features")
    print(f"Var index is now unique: {adata_prot.var.index.is_unique}")
else:
    print("No duplicates found in var index")
# %% --- Apply Final Normalization and Create Layers ---
print("\n=== Applying final normalization and creating layers ===")

# RNA: normalize_total + log1p, save both versions
print("RNA normalization:")
print(f"  Before normalization - X range: [{adata_rna.X.min():.2f}, {adata_rna.X.max():.2f}]")

# Ensure we have raw counts in layers["counts"] (should already exist from line 184)
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
print("✅ CITE-seq dataset preprocessing completed successfully using MaxFuse methodology!")
print(f"Final RNA dataset shape: {adata_rna.shape}")
print(f"Final protein dataset shape: {adata_prot.shape}")
print(f"Common cell types: {sorted(set(adata_rna.obs['cell_types']))}")
print(f"Output saved to: processed_data/")

# %% --- Final Preprocessing Results Plots ---
if plot_flag:
    print("Generating final preprocessing results plots...")

    # For RNA visualization, use log1p layer
    adata_rna_plot = adata_rna.copy()
    adata_rna_plot.X = adata_rna_plot.layers["log1p"].copy()

    # For protein visualization, use z_normalized layer
    adata_prot_plot = adata_prot.copy()
    adata_prot_plot.X = adata_prot_plot.layers["z_normalized"].copy()

    # Subsample for plotting if datasets are large
    if adata_rna_plot.shape[0] > 2000:
        sc.pp.subsample(adata_rna_plot, n_obs=2000)
    if adata_prot_plot.shape[0] > 2000:
        sc.pp.subsample(adata_prot_plot, n_obs=2000)

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
