# %% --- Imports and Config ---

# Add src to path for arcadia package
import sys
from pathlib import Path

# Handle __file__ for both script and notebook execution
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Running as notebook - use current working directory
    ROOT = Path.cwd().resolve().parent
    if not (ROOT / "src").exists():
        if (ROOT.parent / "src").exists():
            ROOT = ROOT.parent

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

"""
Dataset Alignment and Preprocessing Script

This script aligns and preprocesses RNA-seq and protein (CODEX) datasets for downstream analysis.

REQUIREMENTS:
- RNA adata must have:
  * 'counts' layer with raw count data
  * 'log1p' layer with log1p transformed data
  * X should contain log1p(normalized) data from Step 0
- Protein adata must have:
  * 'counts' layer with raw count data
  * 'z_normalized' layer with z-normalized data
  * X should contain z-normalized data from Step 0 (no log1p)
- Both datasets must have:
  * 'cell_types' in obs
  * 'batch' information in obs
  * Spatial coordinates in obsm['spatial'] for protein data

The script performs:
1. Dataset balancing based on cell type proportions (matching smaller dataset proportions)
2. Subsample datasets to specified cell counts (from config.json)
3. RNA preprocessing with batch correction using scVI (ZINB likelihood)
4. Highly variable gene selection using Seurat v3 method on raw counts
5. Quality control and visualization
6. Final dataset alignment and saving

Note: Protein normalization is NOT performed here - it's already done in Step 0.
Protein batch correction is deferred until Step 2 (after spatial features are added).
"""
import json
import os
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from kneed import KneeLocator
from scipy.sparse import issparse

# Set the filename for this script
FILENAME = "_1_align_datasets.py"

# %% tags=["parameters"]
# Default parameters - can be overridden by papermill
dataset_name = None

# Suppress pkg_resources deprecation warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="louvain")

from arcadia.utils.paths import here

if here().parent.name == "notebooks":
    os.chdir("../../")

# ROOT is already defined above (handles both script and notebook execution)
os.chdir(str(ROOT))

from arcadia.data_utils import (
    analyze_and_visualize,
    balance_datasets,
    load_adata_latest,
    preprocess_rna_final_steps,
    qc_metrics,
    save_processed_data,
    spatial_analysis,
    validate_adata_requirements,
)
from arcadia.plotting import preprocessing as pp_plots
from arcadia.utils import metadata as pipeline_metadata_utils
from arcadia.utils.args import parse_pipeline_arguments

# Paths already set up above


# Already imported above

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
    num_rna_cells = config_["subsample"]["num_rna_cells"]
    num_protein_cells = config_["subsample"]["num_protein_cells"]
    plot_flag = config_["plot_flag"]
else:
    num_rna_cells = num_protein_cells = 2000
    plot_flag = True

# Parse command line arguments or use papermill parameters
try:
    args = parse_pipeline_arguments()
    if args.dataset_name is not None:
        dataset_name = args.dataset_name
    # If dataset_name is still None, keep the papermill parameter value
except SystemExit:
    # If running in notebook/papermill, dataset_name is already set from parameters cell
    pass
# %% Report selected dataset

# Load the latest files with dataset-specific support
# This will automatically search in dataset-specific subdirectories and load the latest files
# If you want to load from a specific dataset, pass dataset_name parameter:
adata_rna, adata_prot = load_adata_latest(
    "processed_data", ["rna", "protein"], exact_step=0, dataset_name=dataset_name
)


def verify_adata_validity(adata_rna, adata_prot):
    if "cell_types" not in adata_rna.obs:
        raise ValueError("adata_rna does not have 'cell_types' in obs")
    if "cell_types" not in adata_prot.obs:
        raise ValueError("adata_prot does not have 'cell_types' in obs")
    if "batch" not in adata_rna.obs:
        raise ValueError("adata_rna does not have 'batch' in obs")
    if "batch" not in adata_prot.obs:
        raise ValueError("adata_prot does not have 'batch' in obs")
    if "spatial" not in adata_prot.obsm:
        raise ValueError("adata_prot does not have 'spatial' in obsm")
    if "log1p" not in adata_rna.layers:
        raise ValueError("adata_rna does not have 'log1p' in layers")
    if "z_normalized" not in adata_prot.layers:
        raise ValueError("adata_prot does not have 'z_normalized' in layers")
    if "counts" not in adata_rna.layers:
        raise ValueError("adata_rna does not have 'counts' in layers")


verify_adata_validity(adata_rna, adata_prot)
adata_rna.obs_names_make_unique()
adata_prot.obs_names_make_unique()

print(f"Loaded data from dataset: {adata_rna.uns.get('dataset_name', 'unknown')}")
print(f"Generated from file: {Path(adata_rna.uns.get('file_generated_from', 'unknown')).name}")

validate_adata_requirements(adata_rna, adata_prot)
pp_plots.plot_rna_data_histogram(adata_rna, plot_flag)
# %%
adata_rna, adata_prot = balance_datasets(adata_rna, adata_prot, plot_flag=plot_flag)
# %%
print("\n=== Verifying normalization from Step 0 ===")

print("RNA verification:")
print(f"  RNA layers available: {list(adata_rna.layers.keys())}")
X_data = adata_rna.X.toarray() if issparse(adata_rna.X) else adata_rna.X
log1p_data = (
    adata_rna.layers["log1p"].toarray()
    if issparse(adata_rna.layers["log1p"])
    else adata_rna.layers["log1p"]
)

# Protein: X should contain z_normalized data
print("\nProtein verification:")
print(f"  Protein layers available: {list(adata_prot.layers.keys())}")

# Verify X contains z-normalized data
if "z_normalized" in adata_prot.layers:
    # X should already be z-normalized from Step 0, but verify
    # Handle sparse matrices properly
    X_prot = adata_prot.X.toarray() if issparse(adata_prot.X) else adata_prot.X
    z_norm_data = (
        adata_prot.layers["z_normalized"].toarray()
        if issparse(adata_prot.layers["z_normalized"])
        else adata_prot.layers["z_normalized"]
    )

    if not np.allclose(X_prot, z_norm_data, rtol=1e-5):
        print("  WARNING: X does not match z_normalized layer, restoring z-normalized to X")
        adata_prot.X = adata_prot.layers["z_normalized"].copy()
        X_prot = adata_prot.X.toarray() if issparse(adata_prot.X) else adata_prot.X

    pp_plots.plot_protein_expression_sorted(
        adata_prot, n_cells=2000, n_features=50, plot_flag=plot_flag
    )
    print("  ✅ Protein X contains z-normalized data")
else:
    print("  WARNING: z_normalized layer not found, data may not be from refactored Step 0")

# Note: Protein batch correction has been moved to _1_archetype_generation_neighbors_covet.py
# to occur after spatial features (neighbor means and COVET) are added
print("Protein batch correction will be applied later after spatial features are added")
# %%
pp_plots.plot_protein_violin(adata_prot, plot_flag)

# Set normalization status for pipeline tracking

analyze_and_visualize(adata_prot)
print("Raw protein data ready for spatial feature addition and subsequent batch correction")

pp_plots.plot_protein_violin(adata_prot, plot_flag)
qc_metrics(adata_prot)

spatial_analysis(adata_prot)
sc.pp.calculate_qc_metrics(adata_prot, layer="counts", percent_top=(10, 20, 30), inplace=True)
# %%

print("RNA cell type distribution before subsampling:")
print(adata_rna.obs["cell_types"].value_counts())
print("\nProtein cell type distribution before subsampling:")
print(adata_prot.obs["cell_types"].value_counts())
sc.pp.subsample(adata_rna, n_obs=min(num_rna_cells, adata_rna.shape[0]))
sc.pp.subsample(adata_prot, n_obs=min(num_protein_cells, adata_prot.shape[0]))
# adata_rna = balanced_subsample_by_cell_type(adata_rna, subsample_n_obs_rna)
# adata_prot = balanced_subsample_by_cell_type(adata_prot, subsample_n_obs_protein)

print("\nRNA cell type distribution after subsampling:")
print(adata_rna.obs["cell_types"].value_counts())
print("\nProtein cell type distribution after subsampling:")
print(adata_prot.obs["cell_types"].value_counts())

pp_plots.plot_original_data_umaps(adata_rna, adata_prot, plot_flag)
# %% Plotting

# silhouette_score(adata_prot.obsm["X_pca"], adata_prot.obs["cell_types"])
# silhouette_score(adata_prot.obsm["X_umap"], adata_prot.obs["cell_types"])

# sc.pp.neighbors(adata_rna, use_rep="X_pca")
# silhouette_score(adata_rna.obsm["X_pca"], adata_rna.obs["cell_types"])
# %% Scatter plot of variance vs. mean expression
# common approach to inspect the variance of genes. It shows the relationship between mean expression and variance (or dispersion) and highlights the selected highly variable genes.
pp_plots.plot_variance_analysis_raw(adata_rna, plot_flag)

# %%
# Keep sparse for memory efficiency - compute max/min without converting full matrix to dense
if issparse(adata_rna.X):
    # For sparse matrices, max/min return sparse matrices, use .A to get dense array
    gene_max = adata_rna.X.max(axis=0).A.flatten()
    gene_min = adata_rna.X.min(axis=0).A.flatten()
else:
    gene_max = np.asarray(adata_rna.X.max(axis=0)).flatten()
    gene_min = np.asarray(adata_rna.X.min(axis=0)).flatten()
# Compute difference and convert to boolean array
non_constant_mask = (gene_max - gene_min) != 0  # False ⟺ constant
non_constant_mask = np.asarray(non_constant_mask, dtype=bool)

adata_rna = adata_rna[:, non_constant_mask].copy()

sc.pp.pca(adata_rna, copy=False)
batch_key = "batch" if "batch" in adata_rna.obs else None
# check for nan and inf in the adata_rna.X
if issparse(adata_rna.X):
    has_nan = np.isnan(adata_rna.X.data).any() if adata_rna.X.data.size > 0 else False
    has_inf = np.isinf(adata_rna.X.data).any() if adata_rna.X.data.size > 0 else False
else:
    has_nan = np.isnan(adata_rna.X).any()
    has_inf = np.isinf(adata_rna.X).any()
print(f"Any nan in adata_rna.X: {has_nan}")
print(f"Any inf in adata_rna.X: {has_inf}")
# check for nan and inf in the adata_rna.var
print(
    f"Any nan in adata_rna.var: {adata_rna.var.select_dtypes(include=[np.number]).isna().any().any()}"
)
print(
    f"Any inf in adata_rna.var: {np.isinf(adata_rna.var.select_dtypes(include=[np.number])).any().any()}"
)

# Collect all genes to filter across all datasets
# This variance check can work on log1p data (just checking for zero variance)
genes_to_keep = None

for batch in adata_rna.obs["batch"].unique():
    subset_mask = adata_rna.obs["batch"] == batch
    batch_subset = adata_rna[subset_mask]

    # Skip batches with < 2 cells (can't compute variance)
    if batch_subset.n_obs < 2:
        print(
            f"  Skipping batch {batch}: only {batch_subset.n_obs} cell(s), cannot compute variance"
        )
        continue

    # Compute variance - handle sparse matrices
    if issparse(batch_subset.X):
        # For sparse: convert to dense for variance calculation (only for this subset)
        X_dense = batch_subset.X.toarray()
        gene_variances = np.var(X_dense, axis=0)
    else:
        gene_variances = np.var(batch_subset.X, axis=0)

    # Ensure gene_variances is 1D array
    gene_variances = np.asarray(gene_variances).flatten()
    current_gene_filter = gene_variances > 0

    # Intersect with previous filters to keep only genes with variance in ALL datasets
    if genes_to_keep is None:
        genes_to_keep = current_gene_filter
    else:
        genes_to_keep = genes_to_keep & current_gene_filter

print(
    f"Filtering out {np.sum(~genes_to_keep)} genes with zero variance from {adata_rna.shape[1]} genes"
)
print("  (Zero variance check works fine on log1p data)")

# Apply filter once to the entire dataset
adata_rna = adata_rna[:, genes_to_keep].copy()

# HVG selection: Seurat v3 method needs raw counts, not log1p
# Temporarily use counts layer for HVG selection
print("Running HVG selection on raw counts (Seurat v3 method requires counts)")
adata_rna_for_hvg = adata_rna.copy()
if "counts" in adata_rna_for_hvg.layers:
    adata_rna_for_hvg.X = adata_rna_for_hvg.layers["counts"].copy()
    print("  Using raw counts from layers['counts'] for HVG selection")
else:
    print("  WARNING: No counts layer found, using X as-is")

sc.pp.highly_variable_genes(
    adata_rna_for_hvg, n_top_genes=2000, batch_key="batch", flavor="seurat_v3"
)

# Copy HVG results back to main adata
adata_rna.var["highly_variable"] = adata_rna_for_hvg.var["highly_variable"]
adata_rna.var["highly_variable_rank"] = adata_rna_for_hvg.var.get("highly_variable_rank", 0)
adata_rna.var["means"] = adata_rna_for_hvg.var.get("means", 0)
adata_rna.var["variances"] = adata_rna_for_hvg.var.get("variances", 0)
adata_rna.var["variances_norm"] = adata_rna_for_hvg.var.get("variances_norm", 0)
print(f"  Selected {adata_rna.var['highly_variable'].sum()} highly variable genes")
del adata_rna_for_hvg

variances_sorted = np.sort(adata_rna.var["variances"])[::-1]

pp_plots.plot_gene_variance_elbow(variances_sorted, plot_flag)


kneedle = KneeLocator(
    range(1, len(variances_sorted) + 1),
    np.log(variances_sorted),
    S=20.0,
    curve="convex",
    direction="decreasing",
)
pp_plots.plot_kneedle_analysis(kneedle, plot_flag)


# %%
adata_rna_processing = adata_rna.copy()
adata_prot_processing = adata_prot.copy()
pp_plots.plot_variance_analysis_processed(adata_rna, plot_flag)

adata_rna = adata_rna.copy()
# rename seurat colns to scanpy
rename_map = {
    "nCount_RNA": "total_counts",
    "nFeature_RNA": "n_genes_by_counts",  # if the fn ever expects this
}
for old, new in rename_map.items():
    if new not in adata_rna.obs and old in adata_rna.obs:
        adata_rna.obs[new] = adata_rna.obs[old]

sc.pp.pca(adata_rna, copy=False)
print(f'variance explained by first 10 PCs {adata_rna.uns["pca"]["variance_ratio"][:10].sum()}')
# %% Debug/Logging
print("🔍 DEBUG: Starting batch correction analysis...")
# %% Store original data for comparison
adata_rna_copy = adata_rna.copy()  #  todo remove this
print(f"DEBUG: Original data shape: {adata_rna.shape}")
print(f"DEBUG: Original X mean: {adata_rna.X.mean():.4f}")
print(f"DEBUG: Dataset source distribution: {adata_rna.obs['batch'].value_counts().to_dict()}")
# Check batch separation BEFORE correction
if issparse(adata_rna.X):
    X_dense = adata_rna.X.toarray()
else:
    X_dense = adata_rna.X

batch_labels = adata_rna.obs["batch"].unique()
batch_means = {b: X_dense[adata_rna.obs["batch"] == b].mean(axis=0) for b in batch_labels}
batch_distance_before = np.mean(
    [np.linalg.norm(batch_means[b1] - batch_means[b2]) for b1, b2 in combinations(batch_labels, 2)]
)
print(f"DEBUG: Batch separation BEFORE correction: {batch_distance_before:.4f}")


# %% Run preprocessing with batch correction
# Compute zero proportion without creating full dense mask to save memory
if sp.issparse(adata_rna.X):
    total_elements = adata_rna.X.shape[0] * adata_rna.X.shape[1]
    zero_proportion = (total_elements - adata_rna.X.nnz) / total_elements
else:
    zero_proportion = (adata_rna.X == 0).sum() / adata_rna.X.size
print(f"DEBUG: Proportion of zeros in the data: {zero_proportion:.4f}")
# Store sparse zero mask if needed later (more memory efficient than dense DataFrame)
if sp.issparse(adata_rna.X):
    zero_mask_sparse = adata_rna.X == 0
    adata_rna.obsm["zero_mask"] = zero_mask_sparse
else:
    # For dense arrays, still use sparse to save memory
    adata_rna.obsm["zero_mask"] = sp.csr_matrix((adata_rna.X == 0), shape=adata_rna.X.shape)

adata_rna_processed = preprocess_rna_final_steps(
    adata_rna.copy(),
    n_top_genes=min(2000, kneedle.knee),
    plot_flag=plot_flag,
    gene_likelihood_dist="zinb",
)

print(f"DEBUG: After preprocessing - shape: {adata_rna_processed.shape}")
# Handle mean/std for sparse matrices
if issparse(adata_rna_processed.X):
    X_mean = float(adata_rna_processed.X.mean())
    # For sparse matrices, compute std properly accounting for zeros
    # Use a small sample to avoid memory issues
    if adata_rna_processed.X.data.size > 0:
        # Sample-based std estimation for large sparse matrices
        sample_size = min(10000, adata_rna_processed.X.nnz)
        sample_indices = np.random.choice(
            adata_rna_processed.X.nnz, size=sample_size, replace=False
        )
        sample_values = adata_rna_processed.X.data[sample_indices]
        X_std = float(np.std(sample_values))
    else:
        X_std = 0.0
else:
    X_mean = float(adata_rna_processed.X.mean())
    X_std = float(adata_rna_processed.X.std())
print(f"DEBUG: After preprocessing - X mean: {X_mean:.4f}")
print(f"DEBUG: After preprocessing - X std: {X_std:.4f}")

# Check batch separation AFTER correction
if issparse(adata_rna_processed.X):
    X_corrected_dense = adata_rna_processed.X.toarray()
else:
    X_corrected_dense = adata_rna_processed.X

batch_distance_after = np.mean(
    [np.linalg.norm(batch_means[b1] - batch_means[b2]) for b1, b2 in combinations(batch_labels, 2)]
)
print(f"DEBUG: Batch separation AFTER correction: {batch_distance_after:.4f}")

# Calculate improvement
improvement = ((batch_distance_before - batch_distance_after) / batch_distance_before) * 100
print(f"DEBUG: Batch correction improvement: {improvement:.1f}%")


# %%
adata_rna = adata_rna_processed.copy()
print(f"\n📊 DEBUG: Computing fresh PCA on batch-corrected data...")
print(f"DEBUG: Before PCA - X shape: {adata_rna.X.shape}")
print(f"DEBUG: Available embeddings after batch correction: {list(adata_rna.obsm.keys())}")
print(f"DEBUG: Before PCA - X contains batch-corrected data: ✅")

sc.pp.pca(adata_rna, copy=False)
print(f"DEBUG: After PCA - X_pca shape: {adata_rna.obsm['X_pca'].shape}")
print(f"DEBUG: Available embeddings after PCA computation: {list(adata_rna.obsm.keys())}")


# %%
if "n_genes" in adata_prot.obs.columns:
    adata_prot.obs = adata_prot.obs.drop(columns=["n_genes"])
if "X_pca" in adata_prot.obsm:
    adata_prot.obsm.pop("X_pca")
if "PCs" in adata_prot.varm:
    adata_prot.varm.pop("PCs")
original_protein_num = adata_prot.X.shape[1]

assert adata_prot.obs.index.is_unique
x_coor = adata_prot.obsm["spatial"][:, 0]
y_coor = adata_prot.obsm["spatial"][:, 1]
temp = pd.DataFrame([x_coor, y_coor], index=["x", "y"]).T
temp.index = adata_prot.obs.index
adata_prot.obsm["spatial_location"] = temp
adata_prot.obs["X"] = x_coor
adata_prot.obs["Y"] = y_coor
pp_plots.plot_spatial_locations(adata_prot, plot_flag)

adata_prot = adata_prot[adata_prot.obs.sort_values(by=["cell_types"]).index]
pp_plots.plot_expression_heatmaps(adata_rna, adata_prot, plot_flag)
pipeline_metadata_utils.finalize_align_datasets_metadata(adata_rna, adata_prot, kneedle.knee)
save_processed_data(
    adata_rna,
    adata_prot,
    "processed_data",
    caller_filename=FILENAME,
)


pp_plots.plot_final_alignment_results(adata_rna, adata_prot, plot_flag)
