#!/usr/bin/env python
# %%
"""Calculate metrics on archetype vector embeddings instead of latent space."""

import os
import sys
from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Path setup removed - use package imports instead


# %%
# Custom function for batched cosine distance
def batched_cosine_dist(X, Y, batch_size=5000):
    """Calculate pairwise cosine distances in batches to prevent memory issues."""
    from scipy.spatial.distance import cdist

    n_x = X.shape[0]
    n_y = Y.shape[0]
    distances = np.zeros((n_x, n_y))

    for i in tqdm(range(0, n_x, batch_size), desc="Processing rows", total=n_x // batch_size + 1):
        end_i = min(i + batch_size, n_x)
        batch_X = X[i:end_i]

        for j in range(0, n_y, batch_size):
            end_j = min(j + batch_size, n_y)
            batch_Y = Y[j:end_j]

            # Use cosine distance
            batch_distances = cdist(batch_X, batch_Y, metric="cosine")
            distances[i:end_i, j:end_j] = batch_distances

        print(f"Processed {end_i}/{n_x} rows", end="\r")

    return distances


# %%
# Import custom modules
# import bar_nick_utils  # Migrated to arcadia

from arcadia.training.metrics import mixing_score

# Import log_memory_usage locally to avoid circular import
# from arcadia.training.utils import log_memory_usage


class Tee:
    """Tee class to redirect stdout to both console and log file."""

    def __init__(self, stdout, log_file):
        self.log_file = log_file
        self.stdout = stdout
        self.closed = False

    def write(self, data):
        try:
            if not self.closed and not self.log_file.closed:
                self.log_file.write(data)
                self.log_file.flush()
            self.stdout.write(data)
            self.stdout.flush()
        except Exception as e:
            self.stdout.write(f"Warning: Error writing to log file: {str(e)}\n")
            self.stdout.write(data)
            self.stdout.flush()

    def flush(self):
        try:
            if not self.closed and not self.log_file.closed:
                self.log_file.flush()
        except:
            pass
        self.stdout.flush()

    def close(self):
        try:
            if not self.closed and not self.log_file.closed:
                self.log_file.close()
                self.closed = True
        except:
            pass


# %%
# Import plotting functions from visualization.py
from arcadia.archetypes.visualization import (
    create_tsne_visualization,
    plot_archetype_heatmap,
    plot_archetype_umap,
    plot_distance_comparison,
    plot_matching_accuracy_by_cell_type,
)


# %%
def match_cells_using_archetypes(adata_rna, adata_prot):
    """Match cells between modalities using archetype vectors with cosine distance."""
    # Since we already converted the objects to have archetype vectors as X,
    # we can directly use their X matrices

    # Calculate pairwise distances using cosine distance
    print("Calculating pairwise cosine distances between archetype vectors...")
    latent_distances = batched_cosine_dist(adata_rna.X, adata_prot.X)

    # Find matches
    prot_matches_in_rna = np.argmin(latent_distances, axis=0)
    matching_distances = np.min(latent_distances, axis=0)

    # Generate random matches for comparison
    rand_indices = np.random.permutation(len(adata_rna))
    rand_latent_distances = latent_distances[rand_indices, :]
    rand_prot_matches_in_rna = np.argmin(rand_latent_distances, axis=0)
    rand_matching_distances = np.min(rand_latent_distances, axis=0)

    return {
        "prot_matches_in_rna": prot_matches_in_rna,
        "matching_distances": matching_distances,
        "rand_prot_matches_in_rna": rand_prot_matches_in_rna,
        "rand_matching_distances": rand_matching_distances,
    }


# %%
def calculate_post_training_metrics_on_archetypes(adata_rna, adata_prot, prot_matches_in_rna):
    """Calculate various metrics for model evaluation using archetype vectors."""
    # Calculate NMI scores
    nmi_cell_types_cn_rna = adjusted_mutual_info_score(
        adata_rna.obs["cell_types"], adata_rna.obs["CN"]
    )
    nmi_cell_types_cn_prot = adjusted_mutual_info_score(
        adata_prot.obs["cell_types"], adata_prot.obs["CN"]
    )
    nmi_cell_types_modalities = adjusted_mutual_info_score(
        adata_rna.obs["cell_types"].values[prot_matches_in_rna],
        adata_prot.obs["cell_types"].values,
    )

    # Calculate accuracy
    matches = (
        adata_rna.obs["cell_types"].values[prot_matches_in_rna]
        == adata_prot.obs["cell_types"].values
    )
    accuracy = matches.sum() / len(matches)

    # Calculate mixing score with cosine distance
    # Use X directly since it now contains the archetype vectors
    mixing_result = mixing_score(
        adata_rna.X,
        adata_prot.X,
        adata_rna,
        adata_prot,
        index_rna=range(len(adata_rna)),
        index_prot=range(len(adata_prot)),
        plot_flag=True,
        # metric='cosine'  # Use cosine distance for mixing score
    )

    return {
        "nmi_cell_types_cn_rna_archetypes": nmi_cell_types_cn_rna,
        "nmi_cell_types_cn_prot_archetypes": nmi_cell_types_cn_prot,
        "nmi_cell_types_modalities_archetypes": nmi_cell_types_modalities,
        "cell_type_matching_accuracy_archetypes": accuracy,
        "mixing_score_ilisi_archetypes": mixing_result["iLISI"],
        # "mixing_score_clisi_archetypes": mixing_result["cLISI"],
    }


# %%
def process_archetype_spaces(adata_rna, adata_prot):
    """Process archetype spaces from RNA and protein data."""
    print("Processing archetype spaces...")

    # Since we now have archetype vectors as X, we can use the objects directly
    rna_archetype = adata_rna.copy()
    prot_archetype = adata_prot.copy()

    # Combine for visualization
    combined_archetype = anndata.concat(
        [rna_archetype, prot_archetype],
        join="outer",
        label="modality",
        keys=["RNA", "Protein"],
    )

    print("✓ Archetype spaces processed")

    return rna_archetype, prot_archetype, combined_archetype


# %%
def calculate_metrics_for_archetypes(adata_rna, adata_prot, prefix="", subsample_size=None):
    """Calculate metrics using archetype vectors instead of latent space.

    Args:
        adata_rna: RNA AnnData object
        adata_prot: Protein AnnData object
        prefix: Prefix for metric names (e.g., "train_" or "val_")
        subsample_size: If not None, subsample the data to this size
    """
    print(f"Calculating {prefix}metrics on archetype vectors...")

    # Subsample if requested
    if subsample_size is not None:
        adata_rna = sc.pp.subsample(adata_rna, n_obs=subsample_size, copy=True)
        adata_prot = sc.pp.subsample(adata_prot, n_obs=subsample_size, copy=True)
        print(f"Subsampled to {subsample_size} cells")

    # Since we already have archetype vectors as X, we can directly use the objects
    rna_archetype_adata = adata_rna
    prot_archetype_adata = adata_prot

    # Calculate matching accuracy
    # Check if we can modify the metrics functions to use cosine
    # For now, we use the existing functions which likely use Euclidean
    from arcadia.training.metrics import compute_ari_f1, compute_silhouette_f1, matching_accuracy

    accuracy = matching_accuracy(rna_archetype_adata, prot_archetype_adata)
    print(f"✓ {prefix}matching accuracy calculated")

    # Calculate silhouette F1
    silhouette_f1 = compute_silhouette_f1(rna_archetype_adata, prot_archetype_adata)
    print(f"✓ {prefix}silhouette F1 calculated")

    # Calculate ARI F1
    combined_archetype = anndata.concat(
        [rna_archetype_adata, prot_archetype_adata],
        join="outer",
        label="modality",
        keys=["RNA", "Protein"],
    )

    # Skip PCA since archetype vectors are already low-dimensional (only ~7 dimensions)
    # sc.pp.pca(combined_archetype)

    # Use cosine distance directly on archetype vectors for neighbors
    sc.pp.neighbors(combined_archetype, n_neighbors=10, metric="cosine", use_rep="X")
    ari_f1 = compute_ari_f1(combined_archetype)
    print(f"✓ {prefix}ARI F1 calculated")

    return {
        f"{prefix}cell_type_matching_accuracy_archetypes": accuracy,
        f"{prefix}silhouette_f1_score_archetypes": silhouette_f1.mean(),
        f"{prefix}ari_f1_score_archetypes": ari_f1,
    }


# %%
# Main execution block
if __name__ == "__main__":
    # Create log directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    log_file = open(
        f"logs/archetype_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log", "w"
    )

    # Redirect stdout to both console and log file
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)

    print(f"Starting calculation of metrics on archetype vectors at {pd.Timestamp.now()}")

    # %%
    # Load data
    save_dir = Path("processed_data").absolute()
    from arcadia.training.utils import log_memory_usage

    log_memory_usage("Before loading data: ")

    # Find latest RNA and protein files
    from arcadia.data_utils.loading import get_latest_file

    rna_file = get_latest_file(save_dir, "subset_prepared_for_training_rna_")
    prot_file = get_latest_file(save_dir, "subset_prepared_for_training_prot_")
    if not rna_file or not prot_file:
        print("Error: Could not find trained data files.")
        sys.exit(1)

    print(f"Using RNA file: {os.path.basename(rna_file)}")
    print(f"Using Protein file: {os.path.basename(prot_file)}")

    # %%
    # Load data
    print("\nLoading data...")
    adata_rna = sc.read_h5ad(rna_file)
    adata_prot = sc.read_h5ad(prot_file)
    print("✓ Data loaded")
    from arcadia.training.utils import log_memory_usage

    log_memory_usage("After loading data: ")

    # Verify that archetype vectors exist
    if "archetype_vec" not in adata_rna.obsm or "archetype_vec" not in adata_prot.obsm:
        print("Error: Archetype vectors not found in data.")
        sys.exit(1)

    print(f"RNA dataset shape: {adata_rna.shape}")
    print(f"Protein dataset shape: {adata_prot.shape}")

    # %%
    # Convert to archetype-based AnnData objects
    print("\nConverting to archetype-based AnnData objects...")
    # Create new AnnData objects with archetype vectors as X
    adata_rna_arch = anndata.AnnData(X=adata_rna.obsm["archetype_vec"])
    adata_prot_arch = anndata.AnnData(X=adata_prot.obsm["archetype_vec"])

    # Copy observations and other attributes
    adata_rna_arch.obs = adata_rna.obs.copy()
    adata_prot_arch.obs = adata_prot.obs.copy()

    # Normalize RNA archetype vectors
    rna_scaler = MinMaxScaler()
    adata_rna_arch.X = rna_scaler.fit_transform(adata_rna_arch.X)

    # Normalize protein archetype vectors
    prot_scaler = MinMaxScaler()
    adata_prot_arch.X = prot_scaler.fit_transform(adata_prot_arch.X)

    # Copy uns, obsm (except archetype_vec), and obsp if they exist
    if hasattr(adata_rna, "uns"):
        adata_rna_arch.uns = adata_rna.uns.copy()
    if hasattr(adata_prot, "uns"):
        adata_prot_arch.uns = adata_prot.uns.copy()

    for key in adata_rna.obsm.keys():
        if key != "archetype_vec":
            adata_rna_arch.obsm[key] = adata_rna.obsm[key].copy()

    for key in adata_prot.obsm.keys():
        if key != "archetype_vec":
            adata_prot_arch.obsm[key] = adata_prot.obsm[key].copy()

    if hasattr(adata_rna, "obsp") and len(adata_rna.obsp) > 0:
        for key in adata_rna.obsp.keys():
            adata_rna_arch.obsp[key] = adata_rna.obsp[key].copy()

    if hasattr(adata_prot, "obsp") and len(adata_prot.obsp) > 0:
        for key in adata_prot.obsp.keys():
            adata_prot_arch.obsp[key] = adata_prot.obsp[key].copy()

    # Replace original adata with archetype-based ones
    adata_rna = adata_rna_arch
    adata_prot = adata_prot_arch

    print(f"New RNA archetype dataset shape: {adata_rna.shape}")
    print(f"New Protein archetype dataset shape: {adata_prot.shape}")
    print("✓ Converted to archetype-based AnnData objects")
    from arcadia.training.utils import log_memory_usage

    log_memory_usage("After archetype conversion: ")

    # %%
    # Normalize archetype vectors to [0,1] range
    print("\nNormalizing archetype vectors to [0,1] range...")

    # Get the dimensions of both datasets
    n_rna_dims = adata_rna.X.shape[1]
    n_prot_dims = adata_prot.X.shape[1]

    # Check if dimensions match
    if n_rna_dims != n_prot_dims:
        print(
            f"Warning: RNA and protein archetype vectors have different dimensions ({n_rna_dims} vs {n_prot_dims})"
        )

    # Verify normalization worked
    print(
        f"RNA min values: {adata_rna.X.min(axis=0).min():.4f}, max: {adata_rna.X.max(axis=0).max():.4f}"
    )
    print(
        f"Protein min values: {adata_prot.X.min(axis=0).min():.4f}, max: {adata_prot.X.max(axis=0).max():.4f}"
    )
    print("✓ Archetype vectors normalized")

    # Create a heatmap to visualize the normalized archetype vectors
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sample_size = min(50, len(adata_rna), len(adata_prot))

    # RNA heatmap
    sns.heatmap(
        adata_rna.X[:sample_size],
        ax=axes[0],
        cmap="viridis",
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
    )
    axes[0].set_title(f"Normalized RNA Archetype Vectors (n={sample_size})")

    # Protein heatmap
    sns.heatmap(
        adata_prot.X[:sample_size],
        ax=axes[1],
        cmap="viridis",
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
    )
    axes[1].set_title(f"Normalized Protein Archetype Vectors (n={sample_size})")

    plt.tight_layout()
    plt.show()

    # %%
    # Subsample for faster execution
    max_cells = 5000
    print(f"\nSubsampling data to max {max_cells} cells per modality for faster execution...")
    if len(adata_rna) > max_cells:
        adata_rna = sc.pp.subsample(adata_rna, n_obs=max_cells, copy=True)
    if len(adata_prot) > max_cells:
        adata_prot = sc.pp.subsample(adata_prot, n_obs=max_cells, copy=True)
    print(f"Subsampled RNA dataset shape: {adata_rna.shape}")
    print(f"Subsampled protein dataset shape: {adata_prot.shape}")
    from arcadia.training.utils import log_memory_usage

    log_memory_usage("After subsampling: ")

    # %%
    # Process archetype spaces
    rna_archetype, prot_archetype, combined_archetype = process_archetype_spaces(
        adata_rna, adata_prot
    )

    # %%
    # Create visualization of archetype vectors (display only)
    sc.pl.pca(combined_archetype, color="cell_types", show=False)
    plt.show()
    sc.pl.pca(rna_archetype, color="cell_types", show=False)
    plt.show()
    sc.pp.pca(prot_archetype)
    sc.pl.pca(prot_archetype, color="cell_types", show=False)
    plt.show()

    # Plot archetype heatmaps
    plot_archetype_heatmap(rna_archetype, prot_archetype, n_samples=50)

    # %%
    # Create t-SNE visualization
    create_tsne_visualization(rna_archetype, prot_archetype)

    # %%
    # Match cells and calculate distances using archetype vectors
    matching_results = match_cells_using_archetypes(adata_rna, adata_prot)

    # %%
    # Plot distance comparison
    plot_distance_comparison(
        matching_results["matching_distances"], matching_results["rand_matching_distances"]
    )

    # %%
    # Calculate metrics
    metrics = calculate_post_training_metrics_on_archetypes(
        adata_rna, adata_prot, matching_results["prot_matches_in_rna"]
    )

    # %%
    # Plot matching accuracy by cell type
    plot_matching_accuracy_by_cell_type(
        adata_rna, adata_prot, matching_results["prot_matches_in_rna"]
    )

    # %%
    # Calculate additional metrics
    additional_metrics = calculate_metrics_for_archetypes(adata_rna, adata_prot)
    metrics.update(additional_metrics)

    # %%
    # Generate UMAP visualization
    plot_archetype_umap(combined_archetype)

    # %%
    # Print results
    print("\nMetrics on Archetype Vectors:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Create a summary visualization of metrics
    plt.figure(figsize=(12, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.barh(metric_names, metric_values, color="skyblue")
    plt.xlabel("Value")
    plt.title("Archetype Vector Metrics Summary")
    plt.tight_layout()
    plt.show()

    # %%
    # Calculate metrics with MLflow if available
    try:
        mlflow.log_metrics({k: round(v, 4) for k, v in metrics.items()})
        print("✓ Metrics logged to MLflow (no plot artifacts saved)")
    except Exception as e:
        print(f"Warning: Could not log to MLflow: {e}")

    # %%
    # Clean up: restore original stdout and close log file
    print(f"\nArchetype metrics calculation completed at {pd.Timestamp.now()}")
    sys.stdout = original_stdout
    log_file.close()


# %%

from typing import Dict, List

import numpy as np
from scipy.spatial.distance import cdist


def evaluate_distance_metrics(A: np.ndarray, B: np.ndarray, metrics: List[str]) -> Dict:
    """
    Evaluates multiple distance metrics to determine which one best captures the similarity
    between matching rows in matrices A and B.

    Parameters:
    - A: np.ndarray of shape (n_samples, n_features)
    - B: np.ndarray of shape (n_samples, n_features)
    - metrics: List of distance metrics to evaluate

    Returns:
    - results: Dictionary containing evaluation metrics for each distance metric
    """
    return  # keep this here for now do not remove this function or change it in any way
    results = {}

    for metric in metrics:
        print(f"Evaluating distance metric: {metric}")

        # Compute the distance matrix between rows of A and rows of B
        distances = cdist(A, B, metric=metric)
        # For each row i, get the distances between A[i] and all rows in B
        # Then compute the rank of the matching distance
        ranks = []
        for i in range(len(A)):
            row_distances = distances[i, :]
            # Get the rank of the matching distance
            # Rank 1 means the smallest distance
            rank = np.argsort(row_distances).tolist().index(i) + 1
            ranks.append(rank)
        ranks = np.array(ranks)
        total_samples = len(A)
        # Compute evaluation metrics
        num_correct_matches = np.sum(ranks == 1)
        percentage_correct = num_correct_matches / total_samples * 100
        mean_rank = np.mean(ranks)
        mrr = np.mean(1 / ranks)
        print(f"Percentage of correct matches (rank 1): {percentage_correct:.2f}%")
        print(f"Mean rank of matching rows: {mean_rank:.2f}")
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
        print("")
        results[metric] = {
            "percentage_correct": percentage_correct,
            "mean_rank": mean_rank,
            "mrr": mrr,
            "ranks": ranks,
        }
    return results
