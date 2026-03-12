# %% Training Data Preparation Script
# This script prepares the archetype-annotated data for dual VAE training.

# Key Operations:
# 1. Load archetype-generated data from Step 3
# 2. Subsample to specified number of cells (from config.json)
# 3. Order cells by major and minor cell types for batch construction
# 4. Cross-modal cell matching:
#    - Find closest protein cell for each RNA cell using cosine distance on archetype vectors
#    - Assign cell neighborhoods (CN) from protein to RNA data
#    - Calculate initial matching distances for baseline metrics
# 5. Compute PCA and UMAP embeddings for visualization
# 6. Convert data to integer representation based on detected gene likelihood:
#    - RNA: ZINB, NB, or Normal (detected automatically)
#    - Protein: Normal or NB (detected automatically)
# 7. Normalize library sizes for both modalities (target_sum=40000)
# 8. Set up scVI-compatible batch and library size columns
# 9. Quality checks and visualizations:
#    - Cell type distributions
#    - Archetype heatmaps
#    - B cell subtypes and spatial patterns

# Outputs:
# adata_rna_subset_prepared_for_training_[timestamp].h5ad
# adata_prot_subset_prepared_for_training_[timestamp].h5ad
# Metadata tracking preprocessing steps and matching quality

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: scvi
#     language: python
#     name: python3
# ---

# %% tags=["parameters"]
# Default parameters - can be overridden by papermill

# Add src to path for arcadia package
import gc
import json
import os
import sys
import warnings
from functools import partial
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

dataset_name = None

# %% Set the filename for this script
FILENAME = "_4_prepare_training.py"

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Suppress pkg_resources deprecation warnings
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
import scanpy as sc
import scipy

from arcadia.archetypes import compute_archetype_distances
from arcadia.data_utils import (
    compute_pca_and_umap,
    load_adata_latest,
    order_cells_by_type,
    save_processed_data,
)
from arcadia.plotting import training
from arcadia.plotting.archetypes import plot_archetype_heatmaps
from arcadia.plotting.general import plot_cell_type_distribution
from arcadia.plotting.preprocessing import (
    plot_b_cells_analysis,
    plot_original_data_visualizations,
    plot_pca_and_umap,
    plot_protein_umap,
    plot_umap_visualizations_original_data,
)
from arcadia.training import (
    ensure_correct_dtype,
    is_already_integer,
    select_gene_likelihood,
    simulate_counts_zero_inflated,
    transfer_to_integer_range_nb,
    transfer_to_integer_range_normal,
)
from arcadia.utils import metadata as pipeline_metadata_utils
from arcadia.utils.args import parse_pipeline_arguments
from arcadia.utils.environment import get_umap_filtered_fucntion, setup_environment

if not hasattr(sc.tl.umap, "_is_wrapped"):
    sc.tl.umap = get_umap_filtered_fucntion()
    sc.tl.umap._is_wrapped = True

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

import matplotlib as mpl

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
device = setup_environment()

# %% Parse command line arguments
try:
    args = parse_pipeline_arguments()
    if args.dataset_name is not None:
        dataset_name = args.dataset_name
except SystemExit:
    pass
print(f"Selected dataset: {dataset_name if dataset_name else 'default (latest)'}")

# %% Load data
adata_rna, adata_prot = load_adata_latest(
    "processed_data", ["rna", "protein"], exact_step=3, dataset_name=dataset_name
)
print("Loaded RNA data:", adata_rna.shape)
print("Loaded protein data:", adata_prot.shape)

# Initialize prepare data metadata
pipeline_metadata_utils.initialize_prepare_data_metadata(adata_rna, adata_prot)
sc.pp.subsample(adata_rna, n_obs=min(num_rna_cells, adata_rna.n_obs))
sc.pp.subsample(adata_prot, n_obs=min(num_protein_cells, adata_prot.n_obs))
spatial_only = adata_prot[:, adata_prot.var["feature_type"] != "protein"]

if spatial_only.n_vars > 0 and spatial_only.n_obs > 0:
    # Subsample FIRST before any processing to prevent memory issues
    if spatial_only.n_obs > 5000:
        sc.pp.subsample(spatial_only, n_obs=5000)
    sc.pp.pca(spatial_only)
    sc.pp.neighbors(spatial_only)
    sc.tl.umap(spatial_only)
    if plot_flag:
        if "CN" in spatial_only.obs.columns:
            sc.pl.umap(spatial_only, color="CN")
        else:
            sc.pl.umap(spatial_only)
# %% Process and visualize data

sc.pp.pca(adata_rna)
sc.pp.pca(adata_prot)
sc.pp.neighbors(adata_rna)
sc.pp.neighbors(adata_prot)

adata_rna, adata_prot = order_cells_by_type(adata_rna, adata_prot)
archetype_distances = compute_archetype_distances(adata_rna, adata_prot)
matching_distance_before = np.diag(archetype_distances).mean()
# %% Plot archetype heatmaps
if plot_flag:
    plot_archetype_heatmaps(adata_rna, adata_prot)

# %% Find closest protein cells for each RNA cell
batch_size = 1000
n_rna = adata_rna.shape[0]
closest_prot_indices = np.zeros(n_rna, dtype=int)

for i in range(0, n_rna, batch_size):
    end_idx = min(i + batch_size, n_rna)
    batch_dist = scipy.spatial.distance.cdist(
        adata_rna.obsm["archetype_vec"][i:end_idx],
        adata_prot.obsm["archetype_vec"],
        metric="cosine",
    )
    closest_prot_indices[i:end_idx] = np.argmin(batch_dist, axis=1)
    print(f"Processed batch {i//batch_size + 1}/{(n_rna-1)//batch_size + 1}", end="\r")

adata_rna.obs["CN"] = adata_prot.obs["CN"].values[closest_prot_indices]
adata_rna.obs["CN"] = adata_rna.obs["CN"].cat.set_categories(adata_prot.obs["CN"].cat.categories)
print(f"Number of CN in RNA data: {len(adata_rna.obs['CN'].unique())}")
print(f"Number of CN in protein data: {len(adata_prot.obs['CN'].unique())}")
# Compute PCA and UMAP
adata_rna, adata_prot = compute_pca_and_umap(adata_rna, adata_prot)

# Additional visualizations
if plot_flag:
    plot_umap_visualizations_original_data(adata_rna, adata_prot)
    plot_pca_and_umap(adata_rna, adata_prot)
    plot_b_cells_analysis(adata_rna)
    one_cell_type = plot_protein_umap(adata_prot)
    plot_cell_type_distribution(adata_rna, adata_prot, plot_flag=plot_flag)
    plot_original_data_visualizations(adata_rna, adata_prot)
# %% Apply zero mask to the adata object
# if issparse(adata_rna.obsm["zero_mask"]):
#     adata_rna.obsm["zero_mask"] = adata_rna.obsm["zero_mask"].todense()
# else:
#     adata_rna.obsm["zero_mask"] = adata_rna.obsm["zero_mask"]
# if adata_rna.obsm["zero_mask"].sum() > 0 and not np.allclose(
#     adata_rna.X, adata_rna.X.astype(int)
# ):
#     adata_rna = apply_zero_mask(adata_rna)
# else:
#     print("zero mask has no zero locations")
# zero_proportion = adata_rna.X[adata_rna.X == 0].sum() / adata_rna.X.size
# if zero_proportion < 0.2:
#     min_non_zero_value = adata_rna.X[adata_rna.X != 0].min()
#     adata_rna.X[adata_rna.X == min_non_zero_value] = 0
# else:
#     adata_rna = apply_zero_mask(adata_rna)

# %% convert data to integer representation for training
# if we use log1p to scale the data before applying archetype analysis we need to bring back the counts by using the counts layer
if "counts" in adata_rna.layers:
    adata_rna.X = adata_rna.layers["counts"].astype(float)
    sc.pp.pca(adata_rna)
    sc.pp.neighbors(adata_rna)
    sc.tl.umap(adata_rna)
# %% Plot pre-training histograms
if plot_flag:
    training.pre_train_adata_histograms_heatmap(
        adata_rna, "rna_subset_before_convet_to_integer", "RNA"
    )
    training.pre_train_adata_histograms_heatmap(
        adata_prot, "prot_subset_before_convet_to_integer", "Protein"
    )
# if adata_rna.X.min() < 0:
#     adata_rna.X = adata_rna.X - adata_rna.X.min()
# if adata_prot.X.min() < 0:
#     adata_prot.X = adata_prot.X - adata_prot.X.min()
# TODO: this does not work well for the cite seq data MUST BE CHECKED!!!

# %% Select gene likelihood using raw counts (what VAE will train on)
# Check which layer to use for likelihood selection
rna_layer = "counts" if "counts" in adata_rna.layers else None
prot_layer = "counts" if "counts" in adata_prot.layers else None

adata_rna.uns["gene_likelihood"] = select_gene_likelihood(
    adata_rna, modality="RNA", n_sample_genes=300, use_autozi=False, layer=rna_layer
)
adata_prot.uns["gene_likelihood"] = select_gene_likelihood(
    adata_prot, modality="Protein", n_sample_genes=300, use_autozi=False, layer=prot_layer
)
likelihood_to_function = {
    "zinb": simulate_counts_zero_inflated,
    "normal": transfer_to_integer_range_normal,
    "nb": partial(transfer_to_integer_range_nb, plot_flag=plot_flag),
}
if adata_rna.uns["gene_likelihood"] not in ["zinb", "nb", "normal"]:
    raise ValueError(
        f"Gene likelihood {adata_rna.uns['gene_likelihood']} make sure there is no error in preprocessing"
    )
if adata_prot.uns["gene_likelihood"] not in ["normal", "nb"]:
    raise ValueError(
        f"Gene likelihood {adata_prot.uns['gene_likelihood']} make sure there is no error in preprocessing"
    )
if (
    adata_prot.uns["gene_likelihood"] not in likelihood_to_function
    or adata_rna.uns["gene_likelihood"] not in likelihood_to_function
):
    raise ValueError(
        f"Gene likelihood {adata_prot.uns['gene_likelihood']} or {adata_rna.uns['gene_likelihood']} not supported"
    )
# %% Convert data to integer representation
if is_already_integer(adata_rna):
    adata_rna = ensure_correct_dtype(adata_rna)
else:
    adata_rna = likelihood_to_function[adata_rna.uns["gene_likelihood"]](
        adata_rna, plot_flag=plot_flag
    )
if is_already_integer(adata_prot):
    adata_prot = ensure_correct_dtype(adata_prot)
else:
    adata_prot = likelihood_to_function[adata_prot.uns["gene_likelihood"]](
        adata_prot, plot_flag=plot_flag
    )
# %% Create spatial-only subset (1st time)
print("DEBUG: About to create spatial_only (1st time)")
# First create a temporary subsampled version of adata_prot to avoid memory spike
temp_indices = np.random.choice(adata_prot.n_obs, min(2000, adata_prot.n_obs), replace=False)
temp_indices.sort()
adata_prot_temp = adata_prot[temp_indices, :].copy()


spatial_only = adata_prot_temp[:, adata_prot_temp.var["feature_type"] != "protein"].copy()
print(f"Number of cells in spatial_only: {spatial_only.n_obs}")
del adata_prot_temp
gc.collect()
from arcadia.plotting.preprocessing import plot_spatial_only_umap

plot_spatial_only_umap(spatial_only, color_key="CN", plot_flag=plot_flag)
adata_rna_subset_copy = adata_rna.copy()
adata_prot_subset_copy = adata_prot.copy()
# %% Create spatial-only subset (2nd time)
print("DEBUG: About to create spatial_only (2nd time)")
# First create a t Creating temp subsampled adata_prot from {adata_prot.n_obs} to 2000 cells")
temp_indices = np.random.choice(adata_prot.n_obs, min(2000, adata_prot.n_obs), replace=False)
temp_indices.sort()
adata_prot_temp = adata_prot[temp_indices, :].copy()
spatial_only = adata_prot_temp[:, adata_prot_temp.var["feature_type"] != "protein"].copy()
print(f"Number of cells in spatial_only: {spatial_only.n_obs}")
del adata_prot_temp
gc.collect()

# %% Plot post-conversion histograms
if plot_flag:
    training.pre_train_adata_histograms_heatmap(
        adata_rna, "rna_subset_after_convet_to_integer", "RNA"
    )
    training.pre_train_adata_histograms_heatmap(
        adata_prot, "prot_subset_after_convet_to_integer", "Protein"
    )
# Set up index column for scVI
adata_rna.obs["index_col"] = range(len(adata_rna.obs.index))
adata_prot.obs["index_col"] = range(len(adata_prot.obs.index))

# Ensure both datasets have proper batch columns for scVI
# Check if batch column exists, if not create a default one
if "batch" not in adata_rna.obs.columns:
    print("Creating batch column for RNA data...")
    adata_rna.obs["batch"] = pd.Categorical(["batch_0"] * len(adata_rna))
else:
    print(
        f"RNA data already has batch column with {len(adata_rna.obs['batch'].unique())} unique batches"
    )
    if not isinstance(adata_rna.obs["batch"].dtype, pd.CategoricalDtype):
        adata_rna.obs["batch"] = pd.Categorical(adata_rna.obs["batch"])

if "batch" not in adata_prot.obs.columns:
    print("Creating batch column for protein data...")
    adata_prot.obs["batch"] = pd.Categorical(["batch_0"] * len(adata_prot))
else:
    print(
        f"Protein data already has batch column with {len(adata_prot.obs['batch'].unique())} unique batches"
    )
    if not isinstance(adata_prot.obs["batch"].dtype, pd.CategoricalDtype):
        adata_prot.obs["batch"] = pd.Categorical(adata_prot.obs["batch"])
print(
    f"RNA batch column: {adata_rna.obs['batch'].dtype}, unique values: {list(adata_rna.obs['batch'].unique())}"
)
print(
    f"Protein batch column: {adata_prot.obs['batch'].dtype}, unique values: {list(adata_prot.obs['batch'].unique())}"
)
# Set up scVI library size columns (required for proper scVI operation)
# This will be used by the DualVAETrainingPlan for consistent library sizes
if "_scvi_library_size" not in adata_rna.obs.columns:
    print("Setting up scVI library size for RNA data...")
    # Calculate library size as total counts per cell
    if hasattr(adata_rna.X, "toarray"):
        rna_lib_sizes = np.array(adata_rna.X.toarray().sum(axis=1)).flatten()
    else:
        rna_lib_sizes = np.array(adata_rna.X.sum(axis=1)).flatten()
    # Store log library size (scVI convention)
    adata_rna.obs["_scvi_library_size"] = np.log(rna_lib_sizes)
    print(
        f"RNA library sizes: min={rna_lib_sizes.min():.2f}, max={rna_lib_sizes.max():.2f}, mean={rna_lib_sizes.mean():.2f}"
    )
prot_lib_sizes = 40000
sc.pp.normalize_total(adata_prot, target_sum=prot_lib_sizes)
adata_rna.X = adata_rna.layers["counts"].copy()
sc.pp.normalize_total(adata_rna, target_sum=prot_lib_sizes)
adata_prot.uns["normalize_total_value"] = prot_lib_sizes
adata_rna.uns["normalize_total_value"] = prot_lib_sizes  # Store for RNA too
protein_lib_sizes = np.full(len(adata_prot), adata_prot.uns["normalize_total_value"])
adata_prot.obs["_scvi_library_size"] = np.log(protein_lib_sizes)

# Verify library sizes after normalization
print(
    f"DEBUG: RNA library sizes after normalization - mean: {np.array(adata_rna.X.sum(axis=1)).flatten().mean():.2f}, expected: {prot_lib_sizes}"
)
print(
    f"DEBUG: Protein library sizes after normalization - mean: {np.array(adata_prot.X.sum(axis=1)).flatten().mean():.2f}, expected: {prot_lib_sizes}"
)

# %% Finalize metadata and prepare for saving
pipeline_metadata_utils.finalize_prepare_data_metadata(adata_rna, adata_prot, archetype_distances)
if "log1p" not in adata_rna.layers:
    adata_rna.layers["log1p"] = adata_rna.X.copy()
if "z_normalized" not in adata_prot.layers:
    adata_prot.layers["z_normalized"] = adata_prot.X.copy()

adata_rna.X = sp.csr_matrix(adata_rna.X.astype(np.float32))
adata_prot.X = sp.csr_matrix(adata_prot.X.astype(np.float32))

# Final verification before saving
print(
    f"DEBUG: Final RNA library sizes before save - mean: {np.array(adata_rna.X.sum(axis=1)).flatten().mean():.2f}"
)
print(
    f"DEBUG: Final Protein library sizes before save - mean: {np.array(adata_prot.X.sum(axis=1)).flatten().mean():.2f}"
)

save_processed_data(adata_rna, adata_prot, "processed_data", caller_filename=FILENAME)
# %% Final plotting
if plot_flag:
    one_cell_type = adata_prot.obs["major_cell_types"][0]
    # Create subset and convert to float for PCA
    b_cell_subset = adata_rna[adata_rna.obs["major_cell_types"] == one_cell_type].copy()
    b_cell_subset.X = b_cell_subset.X.astype(float)
    plot_b_cells_analysis(b_cell_subset)
