# %% Spatial Integration Script
# This script integrates spatial information and creates neighborhood-aware features for protein data.

# Key Operations:
# 1. Load preprocessed data from Step 1
# 2. Compute spatial neighbors using adaptive k-NN (k=20 for >5000 cells, k=10 otherwise)
# 3. Filter distant neighbors using 95th percentile threshold while protecting 5 closest neighbors
# 4. Calculate neighbor mean expressions as contextual features
# 5. Optionally apply COVET feature engineering (disabled by default, uses neighbor means instead)
# 6. Apply cell-type-based residualization to spatial features (optional, configurable)
# 7. Scale features globally to keep protein and spatial features on comparable ranges
# 8. Generate Cell Neighborhood (CN) labels using empirical clustering:
#    - For cite_seq: Fixed k=4 clusters (KMeans)
#    - For other datasets: Elbow method to determine optimal k (5-20 range, KMeans or Leiden)
# 9. Filter to highly variable features
# 10. Save integrated data with spatial features and CN labels

# Outputs:
# adata_rna_spatial_integrated_[timestamp].h5ad
# adata_prot_spatial_integrated_[timestamp].h5ad

import json
import os
import sys
import warnings
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

FILENAME = "_2_spatial_integrate.py"

# %% tags=["parameters"]
# Default parameters - can be overridden by papermill
dataset_name = None
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, issparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Suppress pkg_resources deprecation warnings from louvain
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="louvain")


from arcadia.utils.paths import here

if here().parent.name == "notebooks":
    os.chdir("../../")

# Handle __file__ for both script and notebook execution
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Running as notebook - use current working directory
    ROOT = Path.cwd().resolve().parent
    if not (ROOT / "src").exists():
        if (ROOT.parent / "src").exists():
            ROOT = ROOT.parent
os.chdir(str(ROOT))

config_path = Path("configs/config.json")
if config_path.exists():
    with open(config_path, "r") as f:
        config_ = json.load(f)
    num_rna_cells = config_["subsample"]["num_rna_cells"]
    num_protein_cells = config_["subsample"]["num_protein_cells"]
    plot_flag = config_["plot_flag"]
    residualize_spatial_features_flag = config_.get("residualize_spatial_features", True)
else:
    num_rna_cells = num_protein_cells = 2000
    plot_flag = True
    residualize_spatial_features_flag = True
    raise ValueError("config.json does not exist")

import matplotlib as mpl

from arcadia.data_utils import create_smart_neighbors, load_adata_latest, save_processed_data
from arcadia.plotting import preprocessing as pp_plots
from arcadia.plotting import spatial
from arcadia.plotting.spatial import (
    plot_feature_value_distributions,
    plot_protein_cn_subset_umaps,
    plot_protein_vs_cn_statistics,
    plot_residualization_comparison,
)
from arcadia.utils.args import parse_pipeline_arguments


def residualize_spatial_features_by_cell_type(
    spatial_adata: AnnData, cell_type_column: str, batch_column: str | None = None
):
    if cell_type_column not in spatial_adata.obs.columns:
        raise KeyError(
            f"{cell_type_column} missing from spatial AnnData. Provide major cell-type labels "
            "derived from intrinsic markers before residualizing spatial features."
        )
    if batch_column is not None and batch_column not in spatial_adata.obs.columns:
        batch_column = None
    cell_type_categories = spatial_adata.obs[cell_type_column].astype("category")
    spatial_matrix = (
        spatial_adata.X.toarray().astype(float)
        if issparse(spatial_adata.X)
        else np.asarray(spatial_adata.X, dtype=float)
    )
    group_cols = []
    if batch_column is not None:
        group_cols.append(batch_column)
    group_cols.append(cell_type_column)
    grouped_indices = (
        spatial_adata.obs[group_cols].groupby(group_cols, observed=True).indices
        if group_cols
        else {None: np.arange(spatial_adata.n_obs)}
    )
    for indices in grouped_indices.values():
        if len(indices) == 0:
            continue
        type_block = spatial_matrix[indices]
        type_mean = type_block.mean(axis=0, keepdims=True)
        type_std = type_block.std(axis=0, keepdims=True)
        type_std[type_std == 0] = 1.0
        spatial_matrix[indices] = (type_block - type_mean) / type_std
    residualized = spatial_adata.copy()
    residualized.X = spatial_matrix
    metadata = {
        "method": "cell_type_zscore",
        "cell_type_column": cell_type_column,
        "batch_column": batch_column,
        "n_cell_types": len(cell_type_categories.cat.categories),
        "n_batches": (
            spatial_adata.obs[batch_column].astype("category").cat.categories.size
            if batch_column is not None
            else 1
        ),
        "applied": True,
    }
    return residualized, metadata


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

# Using spatial_graph_embedding module directly for unified embedding function


# Set random seed for reproducibility
np.random.seed(55)

# Parse command line arguments or use papermill parameters
try:
    args = parse_pipeline_arguments()
    if args.dataset_name is not None:
        dataset_name = args.dataset_name
    # If dataset_name is still None, keep the papermill parameter value
except SystemExit:
    # If running in notebook/papermill, dataset_name is already set from parameters cell
    pass
# Global variables
# plot_flag = True  # Removed as it's now loaded from config_

# %% Load and Preprocess Data
# Load data

# Report selected dataset
print(f"Selected dataset: {dataset_name if dataset_name else 'default (latest)'}")

# Load the latest files with dataset-specific support
adata_rna, adata_prot = load_adata_latest(
    "processed_data", ["rna", "protein"], exact_step=1, dataset_name=dataset_name
)

print(f"Loaded data from dataset: {adata_rna.uns.get('dataset_name', 'unknown')}")
print(f"Generated from file: {Path(adata_rna.uns.get('file_generated_from', 'unknown')).name}")

# (plots directory setup removed)

# Record archetype testing range - handled by finalize function

adata_rna.X = adata_rna.X.todense() if issparse(adata_rna.X) else adata_rna.X
adata_prot.X = adata_prot.X.todense() if issparse(adata_prot.X) else adata_prot.X
# %% Subsample data
subsample_n_obs_rna = min(adata_rna.shape[0], num_rna_cells)
subsample_n_obs_protein = min(adata_prot.shape[0], num_protein_cells)
sc.pp.subsample(adata_rna, n_obs=subsample_n_obs_rna)
sc.pp.subsample(adata_prot, n_obs=subsample_n_obs_protein)
# Find archetypes using PCHA on UMAP coordinates
# %% Processing
showcase_archetype_generation = False
if showcase_archetype_generation:
    pp_plots.showcase_archetype_generation(adata_rna, adata_prot, plot_flag)

# %% Processing
original_protein_num = adata_prot.X.shape[1]
print(f"data shape: {adata_rna.shape}, {adata_prot.shape}")

# Determine which embedding to use for RNA archetype analysis
rna_archetype_embedding_name = "X_pca"
print("Using X_pca embedding for RNA archetype analysis")

# Note: This will be determined after batch correction is applied
protein_archetype_embedding_name = "X_pca"  # Default, will be updated later

# Compute PCA and UMAP for both modalities
sc.pp.pca(adata_rna)
sc.pp.neighbors(adata_rna, key_added="original_neighbors", use_rep=rna_archetype_embedding_name)
sc.tl.umap(adata_rna, neighbors_key="original_neighbors")
adata_rna.obsm["X_original_pca"] = adata_rna.obsm["X_pca"]
adata_rna.obsm["X_original_umap"] = adata_rna.obsm["X_umap"]
sc.pp.pca(adata_prot)
sc.pp.neighbors(
    adata_prot, key_added="original_neighbors", use_rep=protein_archetype_embedding_name
)
sc.tl.umap(adata_prot, neighbors_key="original_neighbors")
adata_prot.obsm["X_original_pca"] = adata_prot.obsm["X_pca"]
adata_prot.obsm["X_original_umap"] = adata_prot.obsm["X_umap"]

# %% Compute Spatial Neighbors and Means
# remove far away neighbors before setting up the neighbors means
# Calculate neighbors separately within each batch if multiple batches exist

n_neighbors = 20 if adata_prot.n_obs > 5000 else 10

# Check if batch column exists and has multiple batches
if "batch" in adata_prot.obs.columns:
    unique_batches = adata_prot.obs["batch"].unique()
    n_batches = len(unique_batches)
    print(f"Found {n_batches} batches: {unique_batches}")

    if n_batches > 1:
        # Calculate neighbors separately for each batch
        print("Calculating spatial neighbors separately within each batch...")
        n_cells = adata_prot.n_obs
        connectivities_list = []
        distances_list = []

        for batch in unique_batches:
            batch_mask = adata_prot.obs["batch"] == batch
            batch_indices = np.where(batch_mask)[0]
            adata_batch = adata_prot[batch_mask].copy()

            # Calculate neighbors for this batch
            sc.pp.neighbors(
                adata_batch,
                use_rep="spatial_location",
                key_added="spatial_neighbors",
                n_neighbors=n_neighbors,
            )

            # Get connectivities and distances for this batch
            batch_connectivities = adata_batch.obsp["spatial_neighbors_connectivities"]
            batch_distances = adata_batch.obsp["spatial_neighbors_distances"]

            # Convert batch matrices to full-size matrices
            batch_conn_rows, batch_conn_cols = batch_connectivities.nonzero()
            batch_dist_rows, batch_dist_cols = batch_distances.nonzero()

            # Map batch-local indices to global indices
            full_conn_rows = batch_indices[batch_conn_rows]
            full_conn_cols = batch_indices[batch_conn_cols]
            full_dist_rows = batch_indices[batch_dist_rows]
            full_dist_cols = batch_indices[batch_dist_cols]

            # Get values from batch matrices (convert sparse indexing result to array)
            batch_conn_data = np.array(
                batch_connectivities[batch_conn_rows, batch_conn_cols]
            ).flatten()
            batch_dist_data = np.array(batch_distances[batch_dist_rows, batch_dist_cols]).flatten()

            full_connectivities = csr_matrix(
                (batch_conn_data, (full_conn_rows, full_conn_cols)), shape=(n_cells, n_cells)
            )
            full_distances = csr_matrix(
                (batch_dist_data, (full_dist_rows, full_dist_cols)), shape=(n_cells, n_cells)
            )

            connectivities_list.append(full_connectivities)
            distances_list.append(full_distances)

        # Combine all batch matrices (they don't overlap, so we can sum them)
        combined_connectivities = sum(connectivities_list)
        combined_distances = sum(distances_list)

        # Store in adata_prot
        adata_prot.obsp["spatial_neighbors_connectivities"] = combined_connectivities
        adata_prot.obsp["spatial_neighbors_distances"] = combined_distances

        print(f"Combined spatial neighbors from {n_batches} batches")
    else:
        # Single batch, calculate normally
        print("Single batch detected, calculating neighbors normally...")
        sc.pp.neighbors(
            adata_prot,
            use_rep="spatial_location",
            key_added="spatial_neighbors",
            n_neighbors=n_neighbors,
        )
else:
    # No batch column, calculate normally
    print("No batch column found, calculating neighbors globally...")
    sc.pp.neighbors(
        adata_prot,
        use_rep="spatial_location",
        key_added="spatial_neighbors",
        n_neighbors=n_neighbors,
    )
pp_plots.plot_spatial_distance_hist(
    adata_prot,
    key="spatial_neighbors_distances",
    title="Distribution of spatial distances between protein neighbors before cutoff",
    plot_flag=plot_flag,
)
# %% Processing
adata_prot = create_smart_neighbors(adata_prot)

pp_plots.plot_spatial_distance_hist(
    adata_prot,
    key="spatial_neighbors_distances",
    title="Distribution of spatial distances between protein neighbors after cutoff",
    plot_flag=plot_flag,
)

# %% Processing

use_empirical_cn = False
use_annotated_cn = True

# CN assignment will be done after COVET features are generated (after line 424)
if use_annotated_cn and "lab_CN" in adata_prot.obs.columns:
    adata_prot.obs["CN"] = adata_prot.obs["lab_CN"]
    num_clusters = len(adata_prot.obs["CN"].unique())
    # Add CN type info to existing archetype_generation metadata

if issparse(adata_prot.X):
    adata_prot.X = adata_prot.X.toarray()

# Find most variable genes
if adata_prot.X.dtype == np.int32:
    sc.pp.highly_variable_genes(adata_prot, n_top_genes=2000, flavor="seurat_v3")
    pp_plots.plot_hvg_and_mean_variance(adata_prot, plot_flag)
else:
    adata_prot.var["highly_variable"] = True

print(f"Number of highly variable genes: {adata_prot.var['highly_variable'].sum()}")


# %% Processing
orginal_num_features = adata_prot.X.shape[1]  # Capture original protein feature count before COVET


# 1. Create protein + means + COVET features (X is not yet row-zscored here)
# Assuming prepare_data might modify adata_2_prot or its .var, so pass a copy or handle its return carefully.
# If prepare_data returns a new AnnData:
# %% Verify that var_names do not contain dashes todo remove this and just fix the names in the prepare_data function
dash_vars = [var for var in adata_prot.var_names if "-" in var]
if dash_vars:
    print(f"Warning: Found {len(dash_vars)} variable names containing dashes:")
    print(dash_vars[:10])  # Show first 10 examples
    # Replace dashes with underscores
    adata_prot.var_names = [var.replace("-", "_") for var in adata_prot.var_names]
    # raise ValueError("Replaced dashes with underscores in variable names to avoid issues with COVET features names")

if adata_prot.X.min() < 0:
    adata_prot.X = adata_prot.X - adata_prot.X.min() + 1e-6
# %% filter out CN features
use_covet = False
use_means = True
use_variances = False
valid_feature_types = ["protein"]


connectivities = adata_prot.obsp["spatial_neighbors_connectivities"].copy()
connectivities[connectivities > 0] = 1  # binarize
X_protein = adata_prot.X.copy()
if issparse(X_protein):
    X_protein = X_protein.toarray()
neighbor_sums = connectivities.dot(X_protein)
neighbor_means = np.asarray(
    neighbor_sums / connectivities.sum(1)
)  # divide by number of neighbors (k)
neighbor_means = np.nan_to_num(neighbor_means)

# Calculate neighbor variance
neighbor_squared_sums = connectivities.dot(X_protein**2)
neighbor_variances = np.asarray(neighbor_squared_sums / connectivities.sum(1) - neighbor_means**2)
neighbor_variances = np.nan_to_num(neighbor_variances)

# Add neighbor means and variances as features
if use_covet:
    raise ValueError("COVET is not supported yet")
    # adata_protein_neigh_means_and_covet = prepare_data(
    #     adata_prot.copy(), covet_k=10, covet_g=64, covet_selection_method="high_variability"
    # )
    # adata_protein_neigh_means_and_covet = adata_protein_neigh_means_and_covet[
    #     :, adata_protein_neigh_means_and_covet.var["feature_type"].isin(["CN", "protein"])
    # ]


else:
    adata_protein_neigh_means_and_covet = adata_prot.copy()

# Build list of arrays and feature info to concatenate
X_list = [adata_protein_neigh_means_and_covet.X]
var_names_list = list(adata_protein_neigh_means_and_covet.var_names)
if "feature_type" in adata_protein_neigh_means_and_covet.var.columns:
    feature_types_list = adata_protein_neigh_means_and_covet.var["feature_type"].tolist()
else:
    feature_types_list = ["protein"] * adata_protein_neigh_means_and_covet.n_vars
if use_means:
    X_list.append(neighbor_means)
    var_names_list.extend([f"{col}_neighbor_mean" for col in adata_prot.var_names])
    feature_types_list.extend(["neighbor_mean"] * adata_prot.n_vars)
    valid_feature_types.append("neighbor_mean")

if use_variances:
    X_list.append(neighbor_variances)
    var_names_list.extend([f"{col}_neighbor_variance" for col in adata_prot.var_names])
    feature_types_list.extend(["neighbor_variance"] * adata_prot.n_vars)
    valid_feature_types.append("neighbor_variance")

if use_covet:
    valid_feature_types.append("CN")

# Concatenate X matrices horizontally
X_combined = np.hstack(X_list)

# Create new var dataframe
var_df = pd.DataFrame({"feature_type": feature_types_list}, index=var_names_list)

# Create new adata with combined features, preserving obs and other fields
adata_protein_neigh_means_and_covet = ad.AnnData(
    X=X_combined,
    obs=adata_protein_neigh_means_and_covet.obs,
    var=var_df,
    obsm=adata_protein_neigh_means_and_covet.obsm,
    obsp=adata_protein_neigh_means_and_covet.obsp,
    uns=adata_protein_neigh_means_and_covet.uns,
)

# Filter by valid feature types
adata_protein_neigh_means_and_covet = adata_protein_neigh_means_and_covet[
    :, adata_protein_neigh_means_and_covet.var["feature_type"].isin(valid_feature_types)
].copy()

if "highly_variable" not in adata_protein_neigh_means_and_covet.var.columns:
    adata_protein_neigh_means_and_covet.var["highly_variable"] = True
# adata_protein_neigh_means_and_covet.X = scaler.fit_transform(adata_protein_neigh_means_and_covet.X)
# %% If prepare_data modifies inplace and returns None (less ideal):
non_protein_mask = adata_protein_neigh_means_and_covet.var["feature_type"] != "protein"
spatial_adata = adata_protein_neigh_means_and_covet[:, non_protein_mask].copy()
if spatial_adata.X.min() < 0:
    spatial_adata.X = spatial_adata.X - spatial_adata.X.min() + 1e-6
spatial_adata.X = np.nan_to_num(spatial_adata.X)
spatial_adata_raw = spatial_adata.copy()
cell_type_column = "cell_types"
spatial_residualization_metadata = {
    "method": "none",
    "cell_type_column": cell_type_column,
    "n_cell_types": None,
    "batch_column": None,
    "n_batches": None,
    "applied": False,
}
if residualize_spatial_features_flag:
    batch_column = "batch" if "batch" in spatial_adata.obs.columns else None
    (
        spatial_adata,
        spatial_residualization_metadata,
    ) = residualize_spatial_features_by_cell_type(
        spatial_adata, cell_type_column, batch_column=batch_column
    )
elif cell_type_column in spatial_adata.obs.columns:
    spatial_residualization_metadata["n_cell_types"] = (
        spatial_adata.obs[cell_type_column].astype("category").cat.categories.size
    )
    if "batch" in spatial_adata.obs.columns:
        spatial_residualization_metadata["batch_column"] = "batch"
        spatial_residualization_metadata["n_batches"] = (
            spatial_adata.obs["batch"].astype("category").cat.categories.size
        )
    else:
        spatial_residualization_metadata["batch_column"] = None
        spatial_residualization_metadata["n_batches"] = 1
if "pipeline_metadata" not in adata_protein_neigh_means_and_covet.uns:
    adata_protein_neigh_means_and_covet.uns["pipeline_metadata"] = {}
adata_protein_neigh_means_and_covet.uns["pipeline_metadata"][
    "spatial_residualization"
] = spatial_residualization_metadata

adata_protein_neigh_means_and_covet.X[:, non_protein_mask] = spatial_adata.X

if residualize_spatial_features_flag and plot_flag:
    spatial_adata_raw_plot = (
        spatial_adata_raw[:2000].copy()
        if spatial_adata_raw.n_obs > 2000
        else spatial_adata_raw.copy()
    )
    spatial_adata_resid_plot = (
        spatial_adata[:2000].copy() if spatial_adata.n_obs > 2000 else spatial_adata.copy()
    )
    for adata_temp in (spatial_adata_raw_plot, spatial_adata_resid_plot):
        sc.pp.scale(adata_temp, max_value=10)
        sc.pp.pca(adata_temp)
        sc.pp.neighbors(adata_temp)
        sc.tl.umap(adata_temp)
    pp_plots.plot_two_umaps(
        spatial_adata_raw_plot,
        "cell_types",
        "CN features before residualizing",
        spatial_adata_resid_plot,
        "cell_types",
        "CN features after residualizing",
        plot_flag=plot_flag,
    )

    # Plot residualization comparison
    plot_residualization_comparison(
        spatial_adata_raw, spatial_adata, cell_type_column=cell_type_column, plot_flag=plot_flag
    )

from arcadia.plotting.spatial import plot_spatial_data_histograms

plot_spatial_data_histograms(spatial_adata, n_cells=10, plot_flag=plot_flag)
# cala the mean of each feature and plot the histogram
# color by feature type
feature_types = spatial_adata.var["feature_type"]
mean_features = spatial_adata.X.mean(axis=0)

# Create DataFrame for easier plotting
df = pd.DataFrame({"mean_value": mean_features, "feature_type": feature_types})

# Plot feature mean distributions before scaling
from arcadia.plotting.spatial import plot_feature_mean_distributions

plot_feature_mean_distributions(spatial_adata, plot_flag=plot_flag)

if plot_flag:
    # Subsample for plotting
    spatial_adata_plot = (
        spatial_adata[:2000].copy() if spatial_adata.n_obs > 2000 else spatial_adata.copy()
    )

    sns.heatmap(spatial_adata_plot.X[:1000])
    plt.show()
    plt.close()
    sc.pp.pca(spatial_adata_plot)
    sc.pp.neighbors(spatial_adata_plot)
    sc.tl.umap(spatial_adata_plot)
    sc.pl.pca(spatial_adata_plot, color="cell_types")
    sc.pl.umap(spatial_adata_plot, color="cell_types")

    means_only_adata = spatial_adata[:, spatial_adata.var["feature_type"] == "neighbor_mean"].copy()
    # Subsample for plotting
    means_only_adata_plot = (
        means_only_adata[:2000].copy() if means_only_adata.n_obs > 2000 else means_only_adata.copy()
    )
    sc.pp.pca(means_only_adata_plot)
    sc.pp.neighbors(means_only_adata_plot)
    sc.tl.umap(means_only_adata_plot)
    sc.pl.pca(means_only_adata_plot, color="cell_types")
    sc.pl.umap(means_only_adata_plot, color="cell_types")

from arcadia.plotting.spatial import plot_feature_means_line

plot_feature_means_line(adata_protein_neigh_means_and_covet, plot_flag=plot_flag)

# Global scaling to keep protein and CN features on comparable ranges
sc.pp.scale(adata_protein_neigh_means_and_covet, max_value=10)
scale_factor = 0.5
if scale_factor != 1.0:
    adata_protein_neigh_means_and_covet.X[:, non_protein_mask] = (
        adata_protein_neigh_means_and_covet.X[:, non_protein_mask] * scale_factor
    )
adata_protein_neigh_means_and_covet.uns["pipeline_metadata"]["scale_factor"] = scale_factor
plot_feature_means_line(adata_protein_neigh_means_and_covet, plot_flag=plot_flag)
resolution = None
sc.pp.pca(adata_protein_neigh_means_and_covet)
neighbors_key = f"neighbors_{protein_archetype_embedding_name}"
sc.pp.neighbors(
    adata_protein_neigh_means_and_covet,
    use_rep=protein_archetype_embedding_name,
    key_added=neighbors_key,
)
umap_key = f"X_umap_{protein_archetype_embedding_name}"
sc.tl.umap(
    adata_protein_neigh_means_and_covet,
    neighbors_key=neighbors_key,
)
if umap_key != "X_umap":
    adata_protein_neigh_means_and_covet.obsm[umap_key] = adata_protein_neigh_means_and_covet.obsm[
        "X_umap"
    ].copy()
pp_plots.plot_original_vs_new_protein_umap(
    adata_protein_neigh_means_and_covet,
    original_key="X_original_umap",
    new_key=umap_key,
    plot_flag=plot_flag,
)
print(
    "if those two plots are similar, that means that the new CN features are not affecting the protein data"
)
# %% Generate CN labels using non-protein features (after COVET features are created)
# Extract non-protein features for CN generation
if "spatial_adata_raw" in locals():
    raw_spatial_block = spatial_adata_raw.X
    if issparse(raw_spatial_block):
        raw_spatial_block = raw_spatial_block.toarray()
    non_protein_data = np.asarray(raw_spatial_block, dtype=float)
else:
    non_protein_data = adata_protein_neigh_means_and_covet.X[:, non_protein_mask]

# Create temporary AnnData for clustering
temp_cn = AnnData(
    non_protein_data,
    obs=adata_protein_neigh_means_and_covet.obs.copy(),
    var=adata_protein_neigh_means_and_covet.var.loc[non_protein_mask].copy(),
)
sc.pp.pca(temp_cn)
sc.pp.neighbors(temp_cn)

clustering_method = "kmeans"  # "leiden" or "kmeans"

if clustering_method == "leiden":
    # Run Leiden clustering with silhouette score optimization
    best_silhouette = -1
    best_resolution = None
    best_clustering = None

    resolution_range = np.arange(0.005, 2, 0.03)  # Test resolutions from 0.1 to 2
    sli_list = []
    for resolution in resolution_range:
        sc.tl.leiden(temp_cn, resolution=resolution, key_added="leiden_temp")
        num_clusters = len(temp_cn.obs["leiden_temp"].unique())

        # Skip if too few or too many clusters
        if num_clusters < 5:
            continue

        # Calculate silhouette score
        cluster_labels = temp_cn.obs["leiden_temp"].astype(int)
        silhouette = silhouette_score(temp_cn.X, cluster_labels)
        sli_list.append(silhouette)
        print(
            f"Resolution: {resolution:.3f}, clusters: {num_clusters}, silhouette: {silhouette:.3f}"
        )

        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_resolution = resolution
            best_clustering = temp_cn.obs["leiden_temp"].copy()
        if num_clusters > 16:
            break

    # Apply best clustering to protein data
    temp_cn.obs["CN"] = best_clustering
    num_clusters = len(temp_cn.obs["CN"].unique())
    print(best_resolution, best_silhouette)
    print(f"Best resolution: {best_resolution:.3f}, silhouette score: {best_silhouette:.3f}")
    print(f"Final clusters: {num_clusters}")
    resolution = best_resolution
    adata_protein_neigh_means_and_covet.obs["CN"] = pd.Categorical(temp_cn.obs["CN"])


elif clustering_method == "kmeans":

    X = temp_cn.obsm["X_pca"]

    if dataset_name == "cite_seq":
        # Hard 4 clusters for CITE-seq
        n_cluster = 4
        print("Dataset is cite_seq → using fixed k=4 for CN clustering")
        k_values = [4]
    else:
        # Elbow search with k ≥ 5
        k_min, k_max = 5, 20
        k_values = list(range(k_min, k_max + 1))
        inertias = []

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
            inertias.append(km.inertia_)

        # Elbow: farthest point from line between first and last
        x1, y1 = k_values[0], inertias[0]
        x2, y2 = k_values[-1], inertias[-1]
        distances = []
        for k, inertia in zip(k_values, inertias):
            num = abs((y2 - y1) * k - (x2 - x1) * inertia + x2 * y1 - y2 * x1)
            den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            distances.append(num / den)

        best_idx = int(np.argmax(distances))
        n_cluster = k_values[best_idx]
        print(f"Elbow method chose k={n_cluster} for CN clustering")
        # Plot elbow curve for KMeans cluster number selection
        if plot_flag:
            plt.figure(figsize=(6, 4))
            plt.plot(k_values, inertias, marker="o", color="tab:blue")
            plt.xlabel("Number of clusters (k)")
            plt.ylabel("Inertia")
            plt.title("Elbow Method for Optimal k (CN clustering)")
            plt.axvline(n_cluster, color="tab:red", linestyle="--", label=f"Selected k={n_cluster}")
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.close()
    # Final KMeans
    kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_init="auto").fit(X)
    temp_cn.obs["CN"] = kmeans.labels_.astype(str)
    num_clusters = n_cluster
    resolution = None
    adata_protein_neigh_means_and_covet.obs["CN"] = pd.Categorical(temp_cn.obs["CN"])


# %% 2. Feature types should already be properly set by prepare_data function
# Check if feature_type was properly set by prepare_data
if use_annotated_cn and "lab_CN" in adata_protein_neigh_means_and_covet.obs.columns:
    adata_protein_neigh_means_and_covet.obs["CN"] = pd.Categorical(
        adata_protein_neigh_means_and_covet.obs["lab_CN"]
    )
    print(f"using annotated CNs {set(adata_protein_neigh_means_and_covet.obs['CN'].unique())}")
else:
    # Convert CN to categorical with CN_ prefix and assign to adata_prot
    adata_protein_neigh_means_and_covet.obs["CN"] = pd.Categorical(
        [f"CN_{cn}" for cn in temp_cn.obs["CN"]],
        categories=sorted([f"CN_{i}" for i in range(num_clusters)]),
    )
    print(f"Generated empirical CN labels using {clustering_method} on non-protein features")

spatial.plot_spatial_clusters(adata_protein_neigh_means_and_covet, plot_flag=True, max_cells=5000)
plt.close()
print("Feature types properly set by prepare_data:")
print(adata_protein_neigh_means_and_covet.var["feature_type"].value_counts())
adata_prot = adata_protein_neigh_means_and_covet.copy()
protein_feature_mask = adata_prot.var["feature_type"] == "protein"
print(f"Current highly_variable distribution by feature type:")
for feat_type in adata_prot.var["feature_type"].unique():
    mask = adata_prot.var["feature_type"] == feat_type
    if "highly_variable" in adata_prot.var.columns:
        n_hv = np.sum(adata_prot.var.loc[mask, "highly_variable"])
        n_total = np.sum(mask)
        print(f"{feat_type}: {n_hv}/{n_total} highly_variable")
    else:
        print(f"{feat_type}: no highly_variable column")

# Note: Batch correction will be applied after VAE/HV filtering (after line 1317)


# %% remove
# pp_plots.plot_umap_with_colors(adata_prot, ["cell_types", "CN", "batch"], plot_flag=plot_flag)

print("Final feature type distribution:")
print(adata_prot.var["feature_type"].value_counts())
pp_plots.plot_feature_type_heatmap(adata_prot, plot_flag)

# here we can save the adata_2_prot with the COVET features and use it as the basis for
# the VAE hyperparams search if needed, to find the best way to reduce the dim of the protein and COVET data

# %% plot one cell type CN feature and protein feature adn then combined with CN as color
most_common_cell_type = adata_prot.obs["cell_types"].value_counts().idxmax()
most_common_cell_type_mask = adata_prot.obs["cell_types"] == most_common_cell_type

plot_protein_cn_subset_umaps(adata_prot, most_common_cell_type, plot_flag)
# %% make sure prot data and cn data features are similar same scale and variance
# Compare statistical properties between protein features and cell neighborhood features

protein_mask = adata_prot.var["feature_type"] == "protein"
spatial_mask = adata_prot.var["feature_type"] != "protein"

# Extract data for each feature type
if issparse(adata_prot.X):
    protein_data = adata_prot.X[:, protein_mask].toarray()
    spatial_data = adata_prot.X[:, spatial_mask].toarray()
else:
    protein_data = adata_prot.X[:, protein_mask]
    spatial_data = adata_prot.X[:, spatial_mask]

# Calculate basic statistics
protein_stats = {
    "mean": np.mean(protein_data),
    "std": np.std(protein_data),
    "min": np.min(protein_data),
    "max": np.max(protein_data),
    "median": np.median(protein_data),
}

spatial_stats = {
    "mean": np.mean(spatial_data),
    "std": np.std(spatial_data),
    "min": np.min(spatial_data),
    "max": np.max(spatial_data),
    "median": np.median(spatial_data),
}

print("\nComparing statistical properties of protein vs CN features:")
print(f"{'Statistic':10} {'Protein':15} {'CN':15} {'Ratio (Protein/CN)':20}")
print("-" * 60)
for stat in protein_stats:
    ratio = protein_stats[stat] / spatial_stats[stat] if spatial_stats[stat] != 0 else float("inf")
    print(f"{stat:10} {protein_stats[stat]:<15.4f} {spatial_stats[stat]:<15.4f} {ratio:<20.4f}")

# Visual comparison of distributions
plot_protein_vs_cn_statistics(protein_data, spatial_data, plot_flag)

# If statistics are very different, consider scaling
scale_threshold = 10  # Define threshold for when scaling is needed
if (
    protein_stats["std"] / spatial_stats["std"] > scale_threshold
    or spatial_stats["std"] / protein_stats["std"] > scale_threshold
):
    print("\nWARNING: Large difference in variance between protein and CN features!")
    print("Consider scaling features before PCA to prevent bias.")

# Plot a sample of values from both feature sets
# First split features by type
plot_feature_value_distributions(adata_prot, num_points=1000, plot_flag=plot_flag)
# Optional: Perform scaling here if needed

# %% Check if number of protein features is larger than 200 before running VAE
n_protein_features = adata_prot.shape[1]
print(f"Number of protein features: {n_protein_features}")
adata_prot.obsm["protein+neighbors_means+covet"] = pd.DataFrame(
    adata_prot.X,
    index=adata_prot.obs_names,
    columns=adata_prot.var_names,
)
adata_2_prot_before_batch_correction = adata_prot[:, adata_prot.var["highly_variable"] == True]
print(f"Using original protein data. Shape: {adata_prot.shape}")

adata_prot = adata_2_prot_before_batch_correction
# %% <<< VAE INTEGRATION END >>>

save_processed_data(
    adata_rna,
    adata_prot,
    "processed_data",
    caller_filename=FILENAME,
)
