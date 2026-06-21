# %% ---
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

# Set the filename for this script

# Add src to path for arcadia package
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

FILENAME = "_3_generate_archetypes.py"

# %% tags=["parameters"]
# Default parameters - can be overridden by papermill
dataset_name = None

# %% Archetype Generation Script
# This script identifies cell archetypes and aligns them across modalities using Principal Convex Hull Analysis (PCHA).

# Key Operations:
# 1. Load spatially integrated data from Step 2
# 2. Compute PCA and UMAP embeddings for both modalities
# 3. Apply batch correction to protein data (after spatial features are added)
# 4. Convert protein data to integer representation for batch correction compatibility
# 5. Batch-aware archetype generation:
#    - Generate archetypes separately for each batch to account for batch differences
#    - Match archetypes within modalities using cell type proportions
#    - Create unified archetype coordinate system across batches
#    - Find optimal k (number of archetypes) that works across all modalities and batches (default: 7-13)
# 6. Cross-modal archetype alignment:
#    - Match RNA and protein archetypes using cosine distance
#    - Validate alignment with extreme archetype analysis
# 7. Compute archetype vectors (cell loadings on archetypes) for each cell
# 8. Assign dominant archetype label to each cell
# 9. Generate visualizations and save results

# Outputs:
# adata_rna_archetype_generated_[timestamp].h5ad
# adata_prot_archetype_generated_[timestamp].h5ad
# Both files contain:
#   - obsm['archetype_vec']: Cell loadings on archetypes
#   - obs['archetype_label']: Dominant archetype assignment
#   - uns['archetypes']: Archetype coordinates in PCA space

# Suppress pkg_resources deprecation warnings from louvain - must be before any imports that use louvain
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="louvain")
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning
)

import numpy as np
import scanpy as sc
from scipy.sparse import issparse
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_distances

from arcadia.utils.paths import here

if here().parent.name == "notebooks":
    os.chdir("../../")

# ROOT is already defined above (handles both script and notebook execution)
os.chdir(str(ROOT))

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

from arcadia.archetypes.generation import (
    finalize_archetype_generation_with_visualizations,
    find_optimal_k_across_modalities,
    validate_batch_archetype_consistency,
)
from arcadia.archetypes.matching import validate_extreme_archetypes_matching
from arcadia.archetypes.metrics import evaluate_distance_metrics
from arcadia.data_utils import load_adata_latest, save_processed_data
from arcadia.plotting import preprocessing as pp_plots
from arcadia.plotting.archetypes import (
    plot_batch_archetype_visualization,
    plot_cn_features_correlation,
    plot_cross_modal_archetype_similarity_matrix,
)
from arcadia.plotting.general import set_consistent_cell_type_colors
from arcadia.training import select_gene_likelihood
from arcadia.utils import metadata as pipeline_metadata_utils
from arcadia.utils.args import parse_pipeline_arguments

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
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.bottom": False,
        "axes.spines.left": False,
        "axes.grid": False,
    }
)
sc.set_figure_params(
    scanpy=True, fontsize=10, dpi=100, dpi_save=300, vector_friendly=False, format="pdf"
)
# Parse command line arguments or use papermill parameters
force_k = None
try:
    import argparse as _argparse

    _parser = _argparse.ArgumentParser(description="Archetype generation with optional forced k")
    _parser.add_argument("--dataset_name", type=str, default=None)
    _parser.add_argument("--force_k", type=int, default=None, help="Force a specific number of archetypes (k)")
    args = _parser.parse_args()
    if args.dataset_name is not None:
        dataset_name = args.dataset_name
    force_k = args.force_k
except SystemExit:
    # If running in notebook/papermill, dataset_name is already set from parameters cell
    pass

# %% Load data

# Report selected dataset
print(f"Selected dataset: {dataset_name if dataset_name else 'default (latest)'}")
adata_rna, adata_prot = load_adata_latest(
    "processed_data", ["rna", "protein"], exact_step=2, dataset_name=dataset_name
)

print(f"Loaded data from dataset: {adata_rna.uns.get('dataset_name', 'unknown')}")
print(f"Generated from file: {Path(adata_rna.uns.get('file_generated_from', 'unknown')).name}")

sc.pp.subsample(adata_rna, n_obs=min(num_rna_cells, adata_rna.n_obs))
sc.pp.subsample(adata_prot, n_obs=min(num_protein_cells, adata_prot.n_obs))
# call the egde dropping them and combined grpahs which force the spatial data to be
# clustered by cell types
# from arcadia.data_utils import create_celltype_clustered_spatial_latent_space
# create_celltype_clustered_spatial_latent_space(adata_prot)
# %% Determine which embedding to use for RNA archetype analysis
# Use log1p layer from Step 0 for archetype analysis

adata_rna_viz = adata_rna.copy()
# todo: try to make the log1p before and make sure it is set as la
sc.pp.pca(adata_rna_viz)
sc.pp.neighbors(adata_rna_viz)
sc.tl.umap(adata_rna_viz)
if plot_flag:
    sc.pl.umap(adata_rna_viz, color="cell_types")
    sc.pl.pca(adata_rna_viz, color="cell_types")
adata_rna.X = adata_rna_viz.X.copy()
# Copy embedding back to original adata
adata_rna.obsm["X_pca"] = adata_rna_viz.obsm["X_pca"]
adata_rna.obsm["X_umap"] = adata_rna_viz.obsm["X_umap"]
adata_rna.uns["pca"] = adata_rna_viz.uns["pca"]
adata_rna.uns["neighbors"] = adata_rna_viz.uns["neighbors"]
adata_rna.uns["umap"] = adata_rna_viz.uns["umap"]
adata_rna.obsp["distances"] = adata_rna_viz.obsp["distances"]
adata_rna.obsp["connectivities"] = adata_rna_viz.obsp["connectivities"]

rna_archetype_embedding_name = "X_pca"
print("Using X_pca embedding for RNA archetype analysis")

# %% Data processing

# Optional debug heatmap
pp_plots.plot_protein_data_heatmap(adata_prot, n_cells=100, n_features=100, plot_flag=plot_flag)
# %% %% Apply Batch Correction to Protein Data AFTER VAE/HV Filtering
# Now that spatial features (neighbor means and COVET) have been added AND VAE/HV filtering applied
print("\\n=== Applying batch correction to protein data (after VAE/HV filtering) ===")

# Update the main adata_2_prot to include the highly variable filtered version with spatial features


# Ensure batch column exists
if "batch" not in adata_prot.obs.columns:
    if "condition" in adata_prot.obs.columns:
        adata_prot.obs["batch"] = adata_prot.obs["condition"]
    else:
        print("Warning: No batch information found, skipping batch correction")
        adata_prot.obs["batch"] = "unknown"

# Convert protein data from float to integer before batch correction
print("Converting protein data from float to integer...")

# Store a backup of the current state before batch correction
adata_2_prot_before_batch_correction = adata_prot.copy()
# # plot adata_2_prot_before_batch_correction
# sc.pl.umap(
#     adata_2_prot_before_batch_correction[
#         adata_2_prot_before_batch_correction.obs["batch"] == "CONTROL"
#     ],
#     color="cell_types",
# )

# Clean data for batch correction compatibility
print("Cleaning data for batch correction compatibility...")

# Check for NaN and infinite values
nan_count = np.isnan(adata_prot.X).sum()
inf_count = np.isinf(adata_prot.X).sum()
print(f"Found {nan_count} NaN values and {inf_count} infinite values")

# Replace NaN and infinite values with median (more robust than 0)
if nan_count > 0 or inf_count > 0:
    # Use median of finite values for replacement
    finite_mask = np.isfinite(adata_prot.X)
    if finite_mask.any():
        median_val = np.median(adata_prot.X[finite_mask])
    else:
        median_val = 1.0  # fallback to 1 if no finite values
    print(f"Replacing NaN and infinite values with median: {median_val:.4f}")
    adata_prot.X = np.nan_to_num(adata_prot.X, nan=median_val, posinf=median_val, neginf=median_val)

# Check for extreme values and apply gentle outlier handling
min_val = adata_prot.X.min()
max_val = adata_prot.X.max()
print(f"Data range before outlier handling: [{min_val:.4f}, {max_val:.4f}]")

# Use percentile-based clipping for gentle outlier removal
q01 = np.percentile(adata_prot.X, 0.1)
q99 = np.percentile(adata_prot.X, 99.9)
n_outliers = np.sum((adata_prot.X < q01) | (adata_prot.X > q99))
if n_outliers > 0:
    print(f"Clipping {n_outliers} outliers to [{q01:.4f}, {q99:.4f}] range")
    adata_prot.X = np.clip(adata_prot.X, q01, q99)

# Ensure all values are positive for batch correction compatibility
min_val = adata_prot.X.min()
if min_val <= 0:
    shift_val = abs(min_val) + 0.1  # Small positive shift
    print(f"Shifting data by {shift_val:.4f} to ensure positive values")
    adata_prot.X = adata_prot.X + shift_val

# Conservative scaling to avoid numerical issues
# Scale to a more modest range [1, 10] instead of [0, 1000]
min_val = adata_prot.X.min()
max_val = adata_prot.X.max()
print(f"Data range before scaling: [{min_val:.4f}, {max_val:.4f}]")


if max_val > min_val:
    # Conservative scaling to [1, 10] range
    target_min, target_max = 1.0, 10.0
    scale_factor = (target_max - target_min) / (max_val - min_val)
    adata_prot.X = (adata_prot.X - min_val) * scale_factor + target_min
    print(f"Scaled data to range [{target_min}, {target_max}]")

    # Re-apply scaling to non-protein features after rescaling to maintain relative weights
    num_total_features = adata_prot.X.shape[1]
    num_protein_features = adata_prot.X[:, adata_prot.var["feature_type"] == "protein"].shape[1]
    square_root_protein_features_proportion = num_protein_features / num_total_features
    if square_root_protein_features_proportion != 0:
        if "feature_type" in adata_prot.var.columns:
            adata_prot.X[:, adata_prot.var["feature_type"] != "protein"] = (
                adata_prot.X[:, adata_prot.var["feature_type"] != "protein"]
                * square_root_protein_features_proportion
            )
            print(
                f"Re-applied {square_root_protein_features_proportion} scaling to non-protein features after rescaling"
            )

    # Round to reasonable precision and convert to int
    adata_prot.X = np.round(adata_prot.X * 1000).astype(
        int
    )  # Multiply by 10 to get more granularity
    # sns.histplot(adata_prot.X[:,0])
    print(f"Final integer range: [{adata_prot.X.min()}, {adata_prot.X.max()}]")
else:
    # All values are the same, set to a small positive integer
    print("All values are identical, setting to constant value")
    adata_prot.X = np.full_like(adata_prot.X, 10, dtype=int)

print(f"Final data range: [{adata_prot.X.min()}, {adata_prot.X.max()}]")
# %% Analyze the data distribution to determine the proper conversion method

print(
    f"Protein data converted: shape={adata_prot.X.shape}, "
    f"range=[{adata_prot.X.min()}, {adata_prot.X.max()}], "
    f"dtype={adata_prot.X.dtype}"
)

# Apply batch correction if multiple batches exist
if len(adata_prot.obs["batch"].unique()) > 1:

    # Use counts layer for likelihood selection (what VAE will train on)
    adata_prot.uns["gene_likelihood"] = select_gene_likelihood(
        adata_prot.copy(), modality="Protein", layer="X"
    )
    print(f"Detected gene likelihood for protein data: {adata_prot.uns['gene_likelihood']}")

    print(f"Applying batch correction with batches: {adata_prot.obs['batch'].unique()}")

    # Batch correction is skipped - using PCA embedding instead
    print("Batch correction skipped - using PCA embedding instead")

else:
    print("Only one batch detected, skipping batch correction")
    # Still compute PCA for consistency

print(f"Protein data shape after batch correction setup: {adata_prot.shape}")
print(f"Available embeddings: {list(adata_prot.obsm.keys())}")

# Determine which embedding to use for protein archetype analysis
protein_archetype_embedding_name = "X_pca"
print("Using X_pca embedding for protein archetype analysis")

# %% Plotting

pp_plots.plot_feature_type_heatmap(adata_prot, plot_flag)
# %% Extract protein data
if issparse(adata_prot.X):
    protein_data = adata_prot.X.toarray()
else:
    protein_data = adata_prot.X.copy()

# Get neighborhood statistics
print("\nExtracting neighborhood feature statistics...")

sc.pp.pca(adata_prot)
sc.pp.neighbors(adata_prot)
sc.tl.umap(adata_prot)
print(f"New adata shape (protein features + cell neighborhood vector): {adata_prot.shape}")


# %% Compute PCA and UMAP for Both Modalities
minor_cell_types_list_prot = sorted(list(set(adata_prot.obs["cell_types"])))
major_cell_types_list_prot = sorted(list(set(adata_prot.obs["major_cell_types"])))
major_cell_types_list_rna = sorted(list(set(adata_rna.obs["major_cell_types"])))
minor_cell_types_list_rna = sorted(list(set(adata_rna.obs["cell_types"])))

# Compute PCA and UMAP for both modalities
sc.pp.pca(adata_rna)
sc.pp.neighbors(adata_rna, use_rep=rna_archetype_embedding_name)
sc.tl.umap(adata_rna)
sc.pp.pca(adata_prot)
sc.pp.neighbors(adata_prot, use_rep=protein_archetype_embedding_name)
sc.tl.umap(adata_prot)
# pp_plots.plot_modality_embeddings(adata_1_rna, adata_2_prot)
# pp_plots.plot_original_embeddings(adata_1_rna, adata_2_prot)

# %% Convert Gene Names and Compute Module Scores
# Convert gene names to uppercase
# adata_rna.var_names = adata_rna.var_names.str.upper()
# adata_prot.var_names = adata_prot.var_names.str.upper()

# Compute gene module scores
# sc.tl.score_genes(
#     adata_1_rna, gene_list=terminal_exhaustion, score_name="terminal_exhaustion_score"
# )

# %% Set Consistent Cell Type Colors Across Modalities

cell_type_colors = set_consistent_cell_type_colors(adata_rna, adata_prot)
# %% Compute PCA Dimensions
print("\nComputing PCA dimensions...")
max_possible_pca_dim_rna = min(adata_rna.X.shape[1], adata_rna.X.shape[0])
max_possible_pca_dim_rna = min(max_possible_pca_dim_rna, 80)
sc.pp.pca(adata_rna, n_comps=max_possible_pca_dim_rna - 1)
max_possible_pca_dim_prot = min(adata_prot.X.shape[1], adata_prot.X.shape[0])
if "highly_variable" in adata_prot.var.keys():
    max_possible_pca_dim_prot = min(
        max_possible_pca_dim_prot, adata_prot.var["highly_variable"].sum() - 1
    )
    max_possible_pca_dim_prot = min(max_possible_pca_dim_prot, 80)

sc.pp.pca(adata_prot, n_comps=max_possible_pca_dim_prot - 1)
# %% Select PCA components based on variance explained
print("Selecting PCA components...")
max_dim = 50
variance_ratio_selected = 0.80

cumulative_variance_ratio = np.cumsum(adata_rna.uns["pca"]["variance_ratio"])
variance_ratio_selected = min(cumulative_variance_ratio[[-1]], variance_ratio_selected)
n_comps_thresh_rna = np.argmax(cumulative_variance_ratio >= variance_ratio_selected) + 1
n_comps_thresh_rna = min(n_comps_thresh_rna, max_dim)
if n_comps_thresh_rna == 1:
    raise ValueError(
        "n_comps_thresh is 1, this is not good, try to lower the variance_ratio_selected"
    )
real_ratio = np.cumsum(adata_rna.uns["pca"]["variance_ratio"])[n_comps_thresh_rna]
sc.pp.pca(adata_rna, n_comps=n_comps_thresh_rna)
print(f"\nNumber of components explaining {real_ratio} of rna variance: {n_comps_thresh_rna}\n")

cumulative_variance_ratio = np.cumsum(adata_prot.uns["pca"]["variance_ratio"])
variance_ratio_selected = min(cumulative_variance_ratio[[-1]], variance_ratio_selected)
n_comps_thresh_prot = np.argmax(cumulative_variance_ratio >= variance_ratio_selected) + 1
n_comps_thresh_prot = min(n_comps_thresh_prot, max_dim)
real_ratio = np.cumsum(adata_prot.uns["pca"]["variance_ratio"])[n_comps_thresh_prot]
sc.pp.pca(adata_prot, n_comps=n_comps_thresh_prot)
print(f"\nNumber of components explaining {real_ratio} of protein variance: {n_comps_thresh_prot}")
if n_comps_thresh_prot == 1:
    raise ValueError(
        "n_comps_thresh is 1, this is not good, try to lower the variance_ratio_selected"
    )


# %% plot umap of original protein data and the umap to new protein data
sc.pp.neighbors(
    adata_prot,
    use_rep=protein_archetype_embedding_name,
    key_added=f"neighbors_{protein_archetype_embedding_name}",
)

sc.tl.umap(
    adata_prot,
    neighbors_key=f"neighbors_{protein_archetype_embedding_name}",
)
pp_plots.plot_original_vs_new_protein_umap(
    adata_prot,
    original_key="X_original_umap",
    new_key=f"umap_{protein_archetype_embedding_name}",
    plot_flag=plot_flag,
)
print(
    "if those two plots are similar, that means that the new CN features are not affecting the protein data"
)

# %% Plot heatmap of PCA feature contributions
pp_plots.plot_pca_feature_contributions(adata_rna, adata_prot, plot_flag)

# %% Batch-Specific Archetype Generation
# Import the new batch-specific archetype functions
min_k = 7
max_k = 13  # 17
step_size = 1
converge = 1e-5

if force_k is not None:
    print(f"\n*** Forcing k={force_k} (overriding default range [{min_k}, {max_k}]) ***")
    min_k = force_k
    max_k = force_k + 1
    step_size = 1

print("\n" + "=" * 80)
print("BATCH-AWARE ARCHETYPE GENERATION")
print("=" * 80)
print("Using batch-aware archetype generation for consistent representation.")
print("This approach:")
print("1. Generates archetypes separately for each batch to account for batch differences")
print("2. Matches archetypes within modalities using cell type proportions")
print("3. Creates unified archetype coordinate system across batches")
print("4. Finds optimal k that works across all modalities and batches")
print("5. Matches archetypes across modalities")
print("=" * 80)


print(f"\nRNA data batches: {adata_rna.obs['batch'].unique()}")
print(f"Protein data batches: {adata_prot.obs['batch'].unique()}")

# %% Step 1: Find optimal k across both modalities
print("CROSS-MODAL K OPTIMIZATION")

[adata_rna, adata_prot] = find_optimal_k_across_modalities(
    adata_list=[adata_rna, adata_prot],
    batch_key="batch",
    embedding_names=[rna_archetype_embedding_name, protein_archetype_embedding_name],
    min_k=min_k,
    max_k=max_k,
    step_size=step_size,
    converge=converge,
    modality_names=["RNA", "Protein"],
    metric="cosine",
    extreme_filtering_threshold=0.8,
    plot_flag=plot_flag,
)
# %% Validate batch archetype consistency
validate_batch_archetype_consistency(adata_rna, adata_prot, batch_key="batch")

# %% Batch-Specific Archetype Visualizations
if plot_flag:
    print("\n" + "=" * 60)
    print("BATCH-SPECIFIC ARCHETYPE VISUALIZATIONS")
    print("=" * 60)

    # Get unique batches from both modalities
    rna_batches = adata_rna.obs["batch"].unique()
    prot_batches = adata_prot.obs["batch"].unique()

    # Visualize each batch individually using batch-specific archetypes
    # This ensures archetype numbers are consistent within each batch and across modalities

    # Visualize all RNA batches separately with consistent colors
    for rna_batch in rna_batches:
        plot_batch_archetype_visualization(
            adata=adata_rna,
            batch_name=rna_batch,
            modality_name="RNA",
            archetype_embedding_name=rna_archetype_embedding_name,
            plot_flag=plot_flag,
        )

    # Visualize all Protein batches separately with consistent colors
    for prot_batch in prot_batches:
        plot_batch_archetype_visualization(
            adata=adata_prot,
            batch_name=prot_batch,
            modality_name="Protein",
            archetype_embedding_name=protein_archetype_embedding_name,
            plot_flag=plot_flag,
        )

    print("✅ Batch-specific visualizations completed")

# %% Cross-Modal Archetype Matching
print(f"Cross-modal matching completed with consistent archetype numbering.")
print(f"Each batch maintains its own archetype positions with unified numbering.")

# Get archetype vectors from both modalities (now unified across batches)
rna_archetypes = adata_rna.obsm["archetype_vec"]
prot_archetypes = adata_prot.obsm["archetype_vec"]

print(f"RNA archetype vectors shape: {rna_archetypes.shape}")
print(f"Protein archetype vectors shape: {prot_archetypes.shape}")

# Calculate cosine distance similarity matrix for cross-modal matching
similarity_matrix = cosine_distances(rna_archetypes, prot_archetypes)

# Find best matches in both directions
rna_to_prot_matches = np.argmin(similarity_matrix, axis=1)
prot_to_rna_matches = np.argmin(similarity_matrix, axis=0)

# Add matches to observation dataframes
adata_rna.obs["matching_prot_index"] = adata_prot.obs.index[rna_to_prot_matches]
adata_prot.obs["matching_rna_index"] = adata_rna.obs.index[prot_to_rna_matches]

# Update cross-modal matching metadata - handled by finalize function

print(f"Cross-modal matching completed.")
print(f"Average cross-modal distance: {similarity_matrix.min(axis=1).mean():.4f}")

# Check if we have the same cell types between modalities
common_cell_types = set(adata_rna.obs["cell_types"]) & set(adata_prot.obs["cell_types"])
print(f"Common cell types between modalities: {len(common_cell_types)}")
if len(common_cell_types) < len(set(adata_rna.obs["cell_types"])):
    print("Warning: Not all cell types are common between modalities")

# %% Extreme Archetype Identification
# Note: Extreme archetypes are already identified in the batch-specific analysis
# This section provides additional validation and cross-modal comparison

rna_extreme_mask = adata_rna.obs["is_extreme_archetype"].values
prot_extreme_mask = adata_prot.obs["is_extreme_archetype"].values

print(
    f"RNA extreme archetypes: {rna_extreme_mask.sum()} ({rna_extreme_mask.sum()/len(adata_rna)*100:.1f}%)"
)
print(
    f"Protein extreme archetypes: {prot_extreme_mask.sum()} ({prot_extreme_mask.sum()/len(adata_prot)*100:.1f}%)"
)

# Validate extreme archetypes matching across modalities
validate_extreme_archetypes_matching(adata_rna, adata_prot)

if plot_flag:
    rna_extreme_archetypes = rna_archetypes[rna_extreme_mask]
    prot_extreme_archetypes = prot_archetypes[prot_extreme_mask]
    similarity_matrix_extreme = cosine_distances(rna_extreme_archetypes, prot_extreme_archetypes)
    best_matches_extreme = np.argmin(similarity_matrix_extreme, axis=1)
    rna_extreme_cell_types = adata_rna.obs["cell_types"].values[rna_extreme_mask]
    prot_extreme_cell_types = adata_prot.obs["cell_types"].values[prot_extreme_mask]

    # Use chunked computation for full dataset to avoid memory issues
    # Compute best matches without storing full distance matrix
    n_rna_full = rna_archetypes.shape[0]
    n_prot_full = prot_archetypes.shape[0]
    # Dynamically adjust chunk size based on dataset size
    max_chunk_memory_gb = 0.5
    max_elements_per_chunk = int(max_chunk_memory_gb * 1e9 / (n_prot_full * 8))
    chunk_size = min(1000, max(100, max_elements_per_chunk))

    best_matches_full = np.zeros(n_rna_full, dtype=int)
    for i in range(0, n_rna_full, chunk_size):
        end_idx = min(i + chunk_size, n_rna_full)
        chunk_distances = cosine_distances(rna_archetypes[i:end_idx], prot_archetypes)
        best_matches_full[i:end_idx] = np.argmin(chunk_distances, axis=1)
        del chunk_distances

    y_true_full = adata_rna.obs["cell_types"].values
    y_pred_full = adata_prot.obs["cell_types"].values[best_matches_full]
    f1_full = f1_score(y_true_full, y_pred_full, average="weighted")
    print("Full dataset:")
    print(f"  F1 Score: {f1_full:.2%}")

    y_true_extreme = rna_extreme_cell_types
    y_pred_extreme = prot_extreme_cell_types[best_matches_extreme]
    f1_extreme = f1_score(y_true_extreme, y_pred_extreme, average="weighted")
    print("Extreme archetypes only (top 5%):")
    print(f"  F1 Score: {f1_extreme:.2%}")
    pp_plots.plot_extreme_archetype_confusion(adata_prot, y_true_extreme, y_pred_extreme, plot_flag)

# %% Evaluate Distance Metrics and Visualizations
# Extract archetype vectors for compatibility with existing functions
cells_archetype_vec_rna = adata_rna.obsm["archetype_vec"].values
cells_archetype_vec_prot = adata_prot.obsm["archetype_vec"].values

# Evaluate distance metrics
metrics = ["euclidean", "cityblock", "cosine", "correlation", "chebyshev"]
evaluate_distance_metrics(cells_archetype_vec_rna, cells_archetype_vec_prot, metrics)

# %% Cross-Modal Archetype Matching Visualizations
if plot_flag:
    print("\n" + "=" * 60)
    print("CROSS-MODAL ARCHETYPE MATCHING VISUALIZATIONS")
    print("=" * 60)

    # Note: Individual batch visualizations were shown earlier
    # Here we focus on cross-modal matching and overall validation

    # Plot cross-modal archetype similarity matrix

    plot_cross_modal_archetype_similarity_matrix(similarity_matrix)
    plot_cn_features_correlation(adata_prot, plot_flag=True)

# %% Finalize Archetype Generation with Visualizations
# This section integrates the archetype proportions analysis and visualizations
# Pass the consistent cell type colors to the visualization functions
finalize_archetype_generation_with_visualizations(adata_rna, adata_prot, plot_flag)

# %% Save Results
# Save results
print("\nSaving results...")

# Finalize pipeline metadata before saving
pca_params = {
    "variance_ratio_selected": 0.80,  # Update this with actual value from script
    "n_comps_rna": adata_rna.obsm["X_pca"].shape[1] if "X_pca" in adata_rna.obsm else None,
}
pipeline_metadata_utils.finalize_archetype_generation_metadata(
    adata_rna, adata_prot, similarity_matrix, rna_to_prot_matches, prot_to_rna_matches, pca_params
)

save_processed_data(
    adata_rna,
    adata_prot,
    "processed_data",
    caller_filename=FILENAME,
)

# %% Processing
