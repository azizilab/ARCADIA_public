# %% Metrics Functions
# This module contains functions for calculating various metrics.


import os
import sys
from datetime import datetime

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from anndata import AnnData
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from scipy.stats import chisquare
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score, silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

# Path manipulation removed - use package imports instead


def fosknn(
    adata,
    modality_key="modality",
    pair_key="pair_id",
    rep_key="X",
    pairwise_distances=None,
    k=30,
    prefix="",
    print_flag=False,
):
    """
    Calculate FOSKNN (Fraction of Samples K-nearest neighbors) for multi-omics integration.

    For ground truth pairs from different modalities, measures the fraction of cells
    whose true match is among the k-nearest neighbors in the integrated space. Higher values
    indicate better integration (ground truth pairs should appear in top-k neighbors).

    Formula: FOSKNN = (1/n) * sum(1{y_i in top-k neighbors of x_i})
    where (x_i, y_i) are ground truth pairs and top-k neighbors are found in paired modality.

    Args:
        adata (AnnData): Annotated data matrix with paired cells
        modality_key (str): Column in adata.obs containing modality labels. Default: 'modality'
        pair_key (str): Column in adata.obs containing pair IDs (matching across modalities). Default: 'pair_id'
        rep_key (str): Representation to use for distance calculation. Default: 'X' (use .X), can also be 'X_pca', etc.
        pairwise_distances (np.ndarray, optional): Precomputed pairwise distance matrix (n_cells x n_cells).
                                                   If None, computed from rep_key. Shape: (n, n)
        k (int): Number of nearest neighbors to consider. Default: 3
        prefix (str): Prefix for printed messages. Default: ''
        print_flag (bool): Whether to print detailed messages. Default: False

    Returns:
        float: FOSKNN score (0 to 1, higher is better, 1 = perfect)
               NaN if calculation fails or insufficient paired data

    Example:
        >>> import scanpy as sc
        >>> adata = sc.read_h5ad('multiomics_data.h5ad')
        >>> score = fosknn(adata, modality_key='modality', pair_key='pair_id', k=3,
        ...               print_flag=True)
        >>> print(f"FOSKNN: {score:.4f}")
    """

    if modality_key not in adata.obs.columns:
        if print_flag:
            print(f"{prefix}Error: Missing modality_key '{modality_key}' in .obs")
        return float("nan")

    if pair_key not in adata.obs.columns:
        if print_flag:
            print(f"{prefix}Error: Missing pair_key '{pair_key}' in .obs")
        return float("nan")

    # ---- Handle pairwise_distances parameter ----
    if pairwise_distances is None:
        # Get representation
        if rep_key == "X":
            X = adata.X
        elif rep_key in adata.obsm:
            X = adata.obsm[rep_key]
        else:
            if print_flag:
                print(f"{prefix}Error: Representation '{rep_key}' not found in .X or .obsm")
            return float("nan")

        # Convert to dense if sparse
        if hasattr(X, "toarray"):
            X = X.toarray()

        X = np.asarray(X, dtype=np.float32)

        # Compute pairwise distances
        pairwise_distances = cdist(X, X, metric="euclidean")
    else:
        # Verify pairwise_distances shape
        if pairwise_distances.shape[0] != adata.n_obs or pairwise_distances.shape[1] != adata.n_obs:
            if print_flag:
                print(
                    f"{prefix}Error: pairwise_distances shape {pairwise_distances.shape} doesn't match n_obs {adata.n_obs}"
                )
            return float("nan")

    modality_labels = adata.obs[modality_key].values
    pair_ids = adata.obs[pair_key].values
    modality_classes = np.unique(modality_labels)
    n_modality = len(modality_classes)

    if n_modality < 2:
        if print_flag:
            print(f"{prefix}Error: Need at least 2 modalities, found {n_modality}")
        return float("nan")

    if n_modality > 2 and print_flag:
        print(f"{prefix}Warning: Multiple modalities ({n_modality}) found.")

    # ---- Find ground truth pairs ----
    unique_pairs = np.unique(pair_ids)
    valid_pairs = []
    pair_indices = {}

    for pair_id in unique_pairs:
        pair_mask = pair_ids == pair_id
        pair_modalities = modality_labels[pair_mask]

        modality_counts = {mod: np.sum(pair_modalities == mod) for mod in modality_classes}

        # For binary case: need exactly 1 from each modality
        if n_modality == 2:
            if all(count == 1 for count in modality_counts.values()):
                valid_pairs.append(pair_id)
                pair_indices[pair_id] = np.where(pair_mask)[0]

    if len(valid_pairs) == 0:
        if print_flag:
            print(f"{prefix}Error: No valid ground truth pairs found")
        return float("nan")

    # ---- Calculate FOSKNN ----
    true_match_in_knn = 0
    total_pairs = 0

    for pair_id in valid_pairs:
        indices = pair_indices[pair_id]
        pair_modalities = modality_labels[indices]

        if n_modality == 2:
            mod1_idx = np.where(pair_modalities == modality_classes[0])[0][0]
            mod2_idx = np.where(pair_modalities == modality_classes[1])[0][0]

            idx1 = indices[mod1_idx]
            idx2 = indices[mod2_idx]

            # Get distances from idx1 to all cells in opposite modality
            opposite_modality_mask = modality_labels == modality_classes[1]
            opposite_indices = np.where(opposite_modality_mask)[0]
            distances_to_opposite = pairwise_distances[idx1, opposite_indices]

            # Find k-nearest neighbors in opposite modality
            topk_local_indices = np.argsort(distances_to_opposite)[:k]
            topk_indices = opposite_indices[topk_local_indices]

            # Check if true pair is in k-nearest neighbors
            if idx2 in topk_indices:
                true_match_in_knn += 1

            total_pairs += 1

    if total_pairs == 0:
        if print_flag:
            print(f"{prefix}Error: No valid pairs for FOSKNN calculation")
        return float("nan")

    # Final metric: fraction where true match is in k-NN
    fosknn_score = true_match_in_knn / total_pairs

    if print_flag:
        print(f"{prefix}FOSKNN (higher is better): {fosknn_score:.4f}")
        print(f"{prefix}  True matches in top-{k}: {true_match_in_knn}/{total_pairs}")
        print(f"{prefix}  Modalities: {n_modality}")
        print(f"{prefix}  Fraction: {fosknn_score:.4f}")

    return float(fosknn_score)


def foscttm(
    adata,
    modality_key="modality",
    pair_key="pair_id",
    rep_key="X",
    pairwise_distances=None,
    prefix="",
    print_flag=False,
):
    """
    Calculate FOSCTTM (Fraction of Samples Closer Than True Match) for multi-omics integration.

    For ground truth pairs from different modalities, measures the fraction of cells
    from the paired modality that are closer to a cell than its true pair. Lower values
    indicate better integration (ground truth pairs should be closest).

    Formula: FOSCTTM = (1/n) * sum(1{||x_i - z||² < ||x_i - y_i||²})
    where (x_i, y_i) are ground truth pairs, z ranges over all cells in paired modality.

    Args:
        adata (AnnData): Annotated data matrix with paired cells
        modality_key (str): Column in adata.obs containing modality labels. Default: 'modality'
        pair_key (str): Column in adata.obs containing pair IDs (matching across modalities). Default: 'pair_id'
        rep_key (str): Representation to use for distance calculation. Default: 'X' (use .X), can also be 'X_pca', etc.
        pairwise_distances (np.ndarray, optional): Precomputed pairwise distance matrix (n_cells x n_cells).
                                                   If None, computed from rep_key. Shape: (n, n)
        prefix (str): Prefix for printed messages. Default: ''
        print_flag (bool): Whether to print detailed messages. Default: False

    Returns:
        float: FOSCTTM score (lower is better, 0 = perfect, 1 = worst)
               NaN if calculation fails or insufficient paired data

    Example:
        >>> import scanpy as sc
        >>> adata = sc.read_h5ad('multiomics_data.h5ad')
        >>> score = foscttm(adata, modality_key='modality', pair_key='pair_id',
        ...                print_flag=True)
        >>> print(f"FOSCTTM: {score:.4f}")
    """

    if modality_key not in adata.obs.columns:
        if print_flag:
            print(f"{prefix}Error: Missing modality_key '{modality_key}' in .obs")
        return float("nan")

    if pair_key not in adata.obs.columns:
        if print_flag:
            print(f"{prefix}Error: Missing pair_key '{pair_key}' in .obs")
        return float("nan")

    # ---- Handle pairwise_distances parameter ----
    if pairwise_distances is None:
        # Get representation
        if rep_key == "X":
            X = adata.X
        elif rep_key in adata.obsm:
            X = adata.obsm[rep_key]
        else:
            if print_flag:
                print(f"{prefix}Error: Representation '{rep_key}' not found in .X or .obsm")
            return float("nan")

        # Convert to dense if sparse
        if hasattr(X, "toarray"):
            X = X.toarray()

        X = np.asarray(X, dtype=np.float32)

        # Compute pairwise distances
        pairwise_distances = cdist(X, X, metric="euclidean")
    else:
        # Verify pairwise_distances shape
        if pairwise_distances.shape[0] != adata.n_obs or pairwise_distances.shape[1] != adata.n_obs:
            if print_flag:
                print(
                    f"{prefix}Error: pairwise_distances shape {pairwise_distances.shape} doesn't match n_obs {adata.n_obs}"
                )
            return float("nan")

    modality_labels = adata.obs[modality_key].values
    pair_ids = adata.obs[pair_key].values
    modality_classes = np.unique(modality_labels)
    n_modality = len(modality_classes)

    if n_modality < 2:
        if print_flag:
            print(f"{prefix}Error: Need at least 2 modalities, found {n_modality}")
        return float("nan")

    if n_modality > 2 and print_flag:
        print(f"{prefix}Warning: Multiple modalities ({n_modality}) found.")

    # ---- Find ground truth pairs ----
    unique_pairs = np.unique(pair_ids)
    valid_pairs = []
    pair_indices = {}

    for pair_id in unique_pairs:
        pair_mask = pair_ids == pair_id
        pair_modalities = modality_labels[pair_mask]

        modality_counts = {mod: np.sum(pair_modalities == mod) for mod in modality_classes}

        # For binary case: need exactly 1 from each modality
        if n_modality == 2:
            if all(count == 1 for count in modality_counts.values()):
                valid_pairs.append(pair_id)
                pair_indices[pair_id] = np.where(pair_mask)[0]

    if len(valid_pairs) == 0:
        if print_flag:
            print(f"{prefix}Error: No valid ground truth pairs found")
        return float("nan")

    # ---- Calculate FOSCTTM ----
    foscttm_scores = []

    for pair_id in valid_pairs:
        indices = pair_indices[pair_id]
        pair_modalities = modality_labels[indices]

        if n_modality == 2:
            mod1_idx = np.where(pair_modalities == modality_classes[0])[0][0]
            mod2_idx = np.where(pair_modalities == modality_classes[1])[0][0]

            idx1 = indices[mod1_idx]
            idx2 = indices[mod2_idx]

            # Distance to true pair (use precomputed matrix)
            true_distance_sq = pairwise_distances[idx1, idx2] ** 2

            # Get all cells from paired modality (exclude true pair)
            paired_modality_mask = modality_labels == modality_classes[1]
            paired_modality_mask[idx2] = False

            if np.sum(paired_modality_mask) == 0:
                continue

            paired_indices = np.where(paired_modality_mask)[0]
            distances_sq = pairwise_distances[idx1, paired_indices] ** 2

            # Count how many cells are closer than true pair
            n_closer = np.sum(distances_sq < true_distance_sq)

            # Fraction of samples closer than true match
            frac_closer = n_closer / len(paired_indices)
            foscttm_scores.append(frac_closer)

    if len(foscttm_scores) == 0:
        if print_flag:
            print(f"{prefix}Error: No valid FOSCTTM scores calculated")
        return float("nan")

    # Final metric: average FOSCTTM
    foscttm_score = np.mean(foscttm_scores)

    if print_flag:
        print(f"{prefix}FOSCTTM (lower is better): {foscttm_score:.4f}")
        print(f"{prefix}  Valid pairs: {len(foscttm_scores)}/{len(unique_pairs)}")
        print(f"{prefix}  Modalities: {n_modality}")
        print(f"{prefix}  Mean FOSCTTM: {foscttm_score:.4f}")
        print(f"{prefix}  Std FOSCTTM: {np.std(foscttm_scores):.4f}")

    return float(foscttm_score)


def pair_distance(
    adata,
    modality_key="modality",
    pair_key="pair_id",
    rep_key="X",
    pairwise_distances=None,
    prefix="",
    print_flag=False,
):
    """
    Calculate pair distance metric for multi-omics data integration as percentage.

    For single-cell multi-omics datasets with ground truth pairs (e.g., RNA-protein
    measured from the same cells), this metric measures how close paired embeddings
    from different modalities are compared to the average pairwise distance.

    Formula: pair_distance_pct = (mean(||x_i - y_i||) / mean(all pairwise distances)) * 100
    where (x_i, y_i) = ground truth pairs

    Args:
        adata (AnnData): Annotated data matrix with paired cells
        modality_key (str): Column in adata.obs with modality labels. Default: 'modality' (e.g., 'RNA', 'protein')
        pair_key (str): Column in adata.obs with pair IDs (matching across modalities). Default: 'pair_id'
        rep_key (str): Representation for distance calculation. Default: 'X' (use .X), can also be 'X_pca', etc.
        pairwise_distances (np.ndarray, optional): Precomputed pairwise distance matrix (n_cells x n_cells).
                                                   If None, computed from rep_key. Shape: (n, n)
        prefix (str): Prefix for printed messages. Default: ''
        print_flag (bool): Whether to print detailed messages. Default: False

    Returns:
        float: Pair distance as percentage (lower is better, 0-100%)
               NaN if calculation fails or insufficient paired data

    Example:
        >>> import scanpy as sc
        >>> adata = sc.read_h5ad('multiomics_data.h5ad')
        >>> score = pair_distance(adata, modality_key='modality', pair_key='pair_id',
        ...                       print_flag=True)
        >>> print(f"Pair Distance: {score:.2f}%")
    """

    if modality_key not in adata.obs.columns:
        if print_flag:
            print(f"{prefix}Error: Missing modality_key '{modality_key}' in .obs")
        return float("nan")

    if pair_key not in adata.obs.columns:
        if print_flag:
            print(f"{prefix}Error: Missing pair_key '{pair_key}' in .obs")
        return float("nan")

    # ---- Handle pairwise_distances parameter ----
    if pairwise_distances is None:
        # Get representation
        if rep_key == "X":
            X = adata.X
        elif rep_key in adata.obsm:
            X = adata.obsm[rep_key]
        else:
            if print_flag:
                print(f"{prefix}Error: Representation '{rep_key}' not found in .X or .obsm")
            return float("nan")

        # Convert to dense if sparse
        if hasattr(X, "toarray"):
            X = X.toarray()

        X = np.asarray(X, dtype=np.float32)

        # Compute pairwise distances
        pairwise_distances = cdist(X, X, metric="euclidean")
    else:
        # Verify pairwise_distances shape
        if pairwise_distances.shape[0] != adata.n_obs or pairwise_distances.shape[1] != adata.n_obs:
            if print_flag:
                print(
                    f"{prefix}Error: pairwise_distances shape {pairwise_distances.shape} doesn't match n_obs {adata.n_obs}"
                )
            return float("nan")

    modality_labels = adata.obs[modality_key].values
    pair_ids = adata.obs[pair_key].values
    modality_classes = np.unique(modality_labels)
    n_modality = len(modality_classes)

    if n_modality < 2:
        if print_flag:
            print(f"{prefix}Error: Need at least 2 modalities, found {n_modality}")
        return float("nan")

    if n_modality > 2 and print_flag:
        print(f"{prefix}Warning: Multiple modalities ({n_modality}) found.")

    # ---- Find ground truth pairs ----
    unique_pairs = np.unique(pair_ids)
    valid_pairs = []
    pair_indices = {}

    for pair_id in unique_pairs:
        pair_mask = pair_ids == pair_id
        pair_modalities = modality_labels[pair_mask]

        modality_counts = {mod: np.sum(pair_modalities == mod) for mod in modality_classes}

        # For binary case: need exactly 1 from each modality
        if n_modality == 2:
            if all(count == 1 for count in modality_counts.values()):
                valid_pairs.append(pair_id)
                pair_indices[pair_id] = np.where(pair_mask)[0]

    if len(valid_pairs) == 0:
        if print_flag:
            print(f"{prefix}Error: No valid ground truth pairs found")
        return float("nan")

    # ---- Calculate pair distances ----
    true_pair_distances = []

    for pair_id in valid_pairs:
        indices = pair_indices[pair_id]
        pair_modalities = modality_labels[indices]

        if n_modality == 2:
            mod1_idx = np.where(pair_modalities == modality_classes[0])[0][0]
            mod2_idx = np.where(pair_modalities == modality_classes[1])[0][0]

            idx1 = indices[mod1_idx]
            idx2 = indices[mod2_idx]

            # Distance to true pair (use precomputed matrix)
            true_distance = pairwise_distances[idx1, idx2]
            true_pair_distances.append(true_distance)

    if len(true_pair_distances) == 0:
        if print_flag:
            print(f"{prefix}Error: No valid pair distances calculated")
        return float("nan")

    # Calculate mean of all cross-modal distances
    mod1_mask = modality_labels == modality_classes[0]
    mod2_mask = modality_labels == modality_classes[1]
    mod1_indices = np.where(mod1_mask)[0]
    mod2_indices = np.where(mod2_mask)[0]

    # Extract all cross-modal distances
    cross_modal_distances = pairwise_distances[np.ix_(mod1_indices, mod2_indices)]
    mean_cross_modal_distance = np.mean(cross_modal_distances)

    if mean_cross_modal_distance == 0:
        if print_flag:
            print(f"{prefix}Error: Mean cross-modal distance is zero")
        return float("nan")

    # Final metric: (mean true pair distance / mean cross-modal distance) * 100
    mean_true_pair_distance = np.mean(true_pair_distances)
    pair_distance_percentage = (mean_true_pair_distance / mean_cross_modal_distance) * 100

    if print_flag:
        print(f"{prefix}Pair Distance (lower is better): {pair_distance_percentage:.2f}%")
        print(f"{prefix}  Valid pairs: {len(true_pair_distances)}/{len(unique_pairs)}")
        print(f"{prefix}  Modalities: {n_modality}")
        print(f"{prefix}  Mean true pair distance: {mean_true_pair_distance:.4f}")
        print(f"{prefix}  Mean all cross-modal distance: {mean_cross_modal_distance:.4f}")
        print(f"{prefix}  Percentage: {pair_distance_percentage:.2f}%")

    return float(pair_distance_percentage)


def mixing_metric(
    adata,
    modality_key="modality",
    rep_key="X",
    k_neighborhood=300,
    neighbor_ref=5,
    prefix="",
    print_flag=False,
):
    """
    TODO: TEST THIS FUNCTION
    https://www.nature.com/articles/s41467-025-60333-z#MOESM2
    Calculate mixing metric to assess batch effect removal quality.

    Follows the definition from Nature paper: For each cell, the rank in its
    k-neighborhood corresponding to the neighbor_ref-th neighbor in each modality
    is calculated. The metric is the median of ranks over all modalities, then
    averaged over all cells.

    Paper reference: Mixing metric evaluates unwanted variation removal
    - Lower values indicate better mixing (perfect mixing = 0 would mean
      neighbors from all modalities are evenly distributed)

    Args:
        adata (AnnData): Annotated data matrix
        modality_key (str): Column in adata.obs containing modality labels.
                        Default: 'modality'
        rep_key (str): Representation to use for neighbors.
                      Default: 'X' (use .X), can also be 'X_pca', etc.
        k_neighborhood (int): Size of neighborhood for each cell.
                             Default: 300 (matches Nature paper)
        neighbor_ref (int): Reference neighbor index from each modality (1-indexed).
                           Default: 5 (5th neighbor, matches Nature paper)
        prefix (str): Prefix for printed messages. Default: ''
        print_flag (bool): Whether to print detailed messages. Default: False

    Returns:
        float: Mixing metric score (lower is better for modality mixing)
               NaN if calculation fails or insufficient data

    Raises:
        No exceptions - returns NaN on error if print_flag=False

    Example:
        >>> import scanpy as sc
        >>> adata = sc.read_h5ad('integrated_data.h5ad')
        >>> score = mixing_metric(adata, modality_key='modality', k_neighborhood=300,
        ...                       neighbor_ref=5, print_flag=True)
        >>> print(f"Mixing metric: {score:.4f}")
    """

    # ---- Input validation ----
    if modality_key not in adata.obs.columns:
        if print_flag:
            print(f"{prefix}Error: Missing modality_key '{modality_key}' in .obs")
        return float("nan")

    n_cells = adata.n_obs
    nn = min(k_neighborhood, n_cells - 1)

    if nn < neighbor_ref:
        if print_flag:
            print(f"{prefix}Error: k_neighborhood ({nn}) must be >= neighbor_ref ({neighbor_ref})")
        return float("nan")

    # ---- Calculate neighbors ----
    key_added = f'neighbors_{rep_key.replace("_", "")}' if rep_key != "X" else "neighbors"

    if key_added not in adata.uns or adata.uns[key_added].get("use_rep") != rep_key:
        sc.pp.neighbors(
            adata, use_rep=rep_key, n_neighbors=nn, key_added=key_added if rep_key != "X" else None
        )
        if print_flag:
            print(f"{prefix}Calculated neighbors using rep_key='{rep_key}'")
    else:
        if print_flag:
            print(f"{prefix}Using existing neighbors from rep_key='{rep_key}'")

    # ---- Extract batch information ----
    # Now use the correct key
    connectivity_key = f"{key_added}_connectivities" if rep_key != "X" else "connectivities"
    Csp = adata.obsp[connectivity_key].tocsr()
    modality_labels = adata.obs[modality_key].values
    modality_classes = np.unique(modality_labels)
    n_modality = len(modality_classes)

    if n_modality < 2:
        if print_flag:
            print(f"{prefix}Error: Need at least 2 modalities, found {n_modality}")
        return float("nan")

    # Create modality index mapping
    modality_to_idx = {modality: idx for idx, modality in enumerate(modality_classes)}
    modality_indices = np.array([modality_to_idx[m] for m in modality_labels])

    indptr, indices = Csp.indptr, Csp.indices

    # ---- Main computation ----
    mixing_scores = []
    skipped = 0

    for i in range(n_cells):
        start, end = indptr[i], indptr[i + 1]
        neighbor_indices = indices[start:end]

        # Remove self if present
        neighbor_indices = neighbor_indices[neighbor_indices != i]

        if len(neighbor_indices) < neighbor_ref:
            skipped += 1
            continue

        # Get modality labels of neighbors
        neighbor_modality = modality_indices[neighbor_indices]

        # Find rank of neighbor_ref-th neighbor from each modality
        ranks_per_modality = []
        for modality_idx in range(n_modality):
            modality_neighbor_mask = neighbor_modality == modality_idx
            modality_neighbor_positions = np.where(modality_neighbor_mask)[0]

            if len(modality_neighbor_positions) >= neighbor_ref:
                rank = modality_neighbor_positions[neighbor_ref - 1]
                ranks_per_modality.append(rank)

        # Calculate median rank if all modalities present
        if len(ranks_per_modality) == n_modality:
            median_rank = np.median(ranks_per_modality)
            mixing_scores.append(median_rank)

    # ---- Handle results ----
    if len(mixing_scores) == 0:
        if print_flag:
            print(f"{prefix}Error: No cells had valid ranks from all modalities")
        return float("nan")

    mixing_metric_score = np.mean(mixing_scores)

    if print_flag:
        print(f"{prefix}Mixing Metric (lower is better): {mixing_metric_score:.4f}")
        print(
            f"{prefix}  Modalities: {n_modality}, Cells evaluated: {len(mixing_scores)}/{n_cells}"
        )
        print(f"{prefix}  Parameters: k_neighborhood={nn}, neighbor_ref={neighbor_ref}")
        if skipped > 0:
            print(f"{prefix}  Skipped: {skipped} cells (insufficient neighbors)")

    return float(mixing_metric_score)


def mixing_metric_parallel(
    adata, group_key="modality", rep_key="X", dims=None, k=5, max_k=300, n_jobs=-1, verbose=True
):
    """
    Parallel version of mixing metric calculation.

    Calculates mixing metric using parallel computation for better performance on large datasets.
    For each cell, finds the rank of the k-th neighbor from each group in the neighborhood.

    Args:
        adata: AnnData object
        group_key: Column in adata.obs containing grouping variable (e.g., "modality", "batch")
        rep_key: Representation to use for neighbors (e.g., "X", "X_pca")
        dims: Optional subset of dimensions to use (e.g., range(30) for first 30 PCs)
        k: Reference neighbor index (default: 5)
        max_k: Maximum neighborhood size (default: 300)
        n_jobs: Number of parallel jobs (-1 uses all cores, default: -1)
        verbose: Whether to print progress (default: True)

    Returns:
        np.array: Mixing scores for each cell (lower is better for mixing)
    """
    # Get embeddings
    if rep_key in adata.obsm:
        embeddings = adata.obsm[rep_key]
    elif rep_key == "X":
        embeddings = adata.X
        # Convert sparse to dense if needed
        if hasattr(embeddings, "toarray"):
            embeddings = embeddings.toarray()
    else:
        raise ValueError(f"Representation {rep_key} not found in adata.obsm or adata.X")

    if dims is not None:
        embeddings = embeddings[:, dims]

    group_info = adata.obs[group_key].values
    groups = np.unique(group_info)

    # Find nearest neighbors
    nn_model = NearestNeighbors(n_neighbors=max_k, metric="euclidean")
    nn_model.fit(embeddings)
    distances, nn_indices = nn_model.kneighbors(embeddings)

    def compute_mixing_for_cell(i):
        neighbors = nn_indices[i, :]
        neighbor_groups = group_info[neighbors]

        group_ranks = []
        for group in groups:
            group_positions = np.where(neighbor_groups == group)[0]
            if len(group_positions) >= k:
                rank = group_positions[k - 1]
            else:
                rank = max_k - 1
            group_ranks.append(rank)

        return np.median(group_ranks)

    # Parallel computation
    mixing = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(compute_mixing_for_cell)(i) for i in range(adata.n_obs)
    )

    return np.array(mixing)


def modality_kbet_mixing_score(
    adata: AnnData,
    label_key="modality",
    group_key=None,
    rep_key="X",
    k=30,
    alpha=0.05,
    min_cells=10,
    min_freq=0.05,
    prefix="",
    to_print=False,
):
    kbet_score = kbet_within_cell_types(
        adata=adata,
        label_key=label_key,
        group_key=group_key,
        rep_key=rep_key,
        k=k,
        alpha=alpha,
        min_cells=min_cells,
        min_freq=min_freq,
        prefix=prefix,
        to_print=to_print,
    )
    return kbet_score


def kbet_within_cell_types(
    adata: AnnData,
    label_key="CN",
    group_key="cell_types",
    rep_key="X",
    k=30,
    alpha=0.05,
    min_cells=10,
    min_freq=0.05,  # Minimum frequency for CN categories to be included
    prefix="",
    to_print=False,
    return_neighbor_df=False,
):
    """

    k-nearest-neighbor batch-effect (kBET) like test rejection rate within each cell type:
    - For each cell, compare local kNN label counts to global counts via chi-square test.
    - Returns mean rejection rate across cell types.
    this is a great method as it can handle vastly unbalanced groups which allow us to estimate CN separation between different cn values.
    has a very different number of cells.

    For large imbalanced datasets (>10k cells with >10% size difference), automatically
    applies balanced stratified subsampling to prevent chi-square test bias.

    Args:
        adata: AnnData object
        label_key: Column in adata.obs containing labels to test (e.g., "CN", "modality")
        group_key: Column in adata.obs containing grouping (e.g., "cell_types")
        rep_key: Representation to use for neighbors (e.g., "X", "X_pca")
        k: Number of neighbors
        alpha: Significance level for chi-square test
        min_cells: Minimum number of cells required per cell type
        min_freq: Minimum frequency for CN categories to be included (default: 0.05 or 5%)
        prefix: Prefix for printed messages
        to_print: Whether to print detailed messages (default: False)
        return_neighbor_df: If True, returns (score, df) where df contains per-cell neighbor proportions

    Returns:
        float: Mean rejection rate across cell types (lower is more mixed)
        the higher the value the less mixed the data is.
        the lower the value the more mixed the data is.
        pd.DataFrame: (optional) If return_neighbor_df=True, returns DataFrame with columns:
            - cell_type: cell type group
            - cell_index: cell identifier
            - cell_label: label of the cell (e.g., modality, CN)
            - same_label_prop: proportion of neighbors with same label
            - n_neighbors: number of neighbors
    """
    if group_key is None:
        adata.obs["temp_obs_key"] = "all"
        group_key = "temp_obs_key"
    if group_key not in adata.obs.columns or label_key not in adata.obs.columns:
        if to_print:
            print(f"{prefix}Missing '{group_key}' or '{label_key}' in .obs")
        return float("nan")

    # # Check for imbalance and apply balanced subsampling if needed (similar to calculate_iLISI)
    # n_cells = adata.n_obs
    # if label_key in adata.obs:
    #     label_counts = adata.obs[label_key].value_counts()
    #     if len(label_counts) >= 2:
    #         total_cells = label_counts.sum()
    #         min_count = label_counts.min()
    #         max_count = label_counts.max()

    #         # 10% size difference threshold
    #         size_difference_pct = abs(max_count - min_count) / total_cells
    #         is_imbalanced = size_difference_pct > 0.10
    #         is_large = n_cells > 10000

    #         if is_imbalanced and is_large:
    #             # Use stratified balanced sampling - use full minority class size
    #             sample_size_per_label = min_count

    #             balanced_indices = []
    #             for label in label_counts.index:
    #                 label_mask = adata.obs[label_key] == label
    #                 label_indices = np.where(label_mask)[0]
    #                 sampled = np.random.choice(label_indices, sample_size_per_label, replace=False)
    #                 balanced_indices.extend(sampled)

    #             # Shuffle to mix labels
    #             np.random.shuffle(balanced_indices)

    #             # Create balanced subset
    #             adata = adata[balanced_indices].copy()

    #             # Log the balancing
    #             if to_print:
    #                 print(f"{prefix}kBET: Dataset imbalanced ({size_difference_pct*100:.1f}% size diff). "
    #                       f"Using balanced subsample: {sample_size_per_label} cells per {label_key} "
    #                       f"(original: {dict(label_counts)}).")
    #             n_cells = adata.n_obs
    ct_scores = {}
    all_neighbor_dfs = []
    for ct in adata.obs[group_key].unique():
        m = adata.obs[group_key] == ct
        ad = adata[m].copy()
        if ad.n_obs < min_cells:
            if to_print:
                print(f"{prefix}Skipping {ct}: insufficient cells (n={ad.n_obs}, min={min_cells})")
            continue
        if ad.obs[label_key].nunique() <= 1:
            if to_print:
                print(
                    f"{prefix}Skipping {ct}: insufficient {label_key} diversity (n_unique={ad.obs[label_key].nunique()})"
                )
            continue

        # Calculate neighbors
        nn = min(k, max(1, ad.n_obs - 1))
        sc.pp.neighbors(ad, use_rep=rep_key, n_neighbors=nn)
        Csp = ad.obsp["connectivities"].tocsr()

        # Calculate proportion of same-label neighbors for each cell
        cell_same_label_props = []
        indptr_initial, indices_initial = Csp.indptr, Csp.indices
        labels_temp = ad.obs[label_key].values

        for i in range(ad.n_obs):
            cell_label = labels_temp[i]
            neigh_idx = indices_initial[indptr_initial[i] : indptr_initial[i + 1]]
            neigh_idx = neigh_idx[neigh_idx != i]  # Exclude self

            if len(neigh_idx) > 0:
                neighbor_labels = labels_temp[neigh_idx]
                same_label_count = np.sum(neighbor_labels == cell_label)
                same_label_prop = same_label_count / len(neigh_idx)
            else:
                same_label_prop = np.nan

            cell_same_label_props.append(
                {
                    "cell_type": ct,
                    "cell_index": ad.obs.index[i],
                    "cell_label": cell_label,
                    "same_label_prop": same_label_prop,
                    "n_neighbors": len(neigh_idx),
                }
            )

        # Create DataFrame for this cell type
        ct_df = pd.DataFrame(cell_same_label_props)
        all_neighbor_dfs.append(ct_df)

        if to_print:
            print(f"\n{prefix}Same-label neighbor proportions for {ct}:")
            print(f"{prefix}  Mean: {ct_df['same_label_prop'].mean():.3f}")
            print(f"{prefix}  Median: {ct_df['same_label_prop'].median():.3f}")
            print(f"{prefix}  Std: {ct_df['same_label_prop'].std():.3f}")
            print(f"{prefix}  By label:")
            for lbl in ct_df["cell_label"].unique():
                lbl_mean = ct_df[ct_df["cell_label"] == lbl]["same_label_prop"].mean()
                lbl_count = (ct_df["cell_label"] == lbl).sum()
                print(f"{prefix}    {lbl}: {lbl_mean:.3f} (n={lbl_count})")

        # Get label distribution
        labels = ad.obs[label_key].astype("category")
        classes = labels.cat.categories
        y = labels.cat.codes.to_numpy()
        C = len(classes)

        # Global expected proportions within this cell type
        glob_counts = np.bincount(y, minlength=C).astype(float)
        glob_probs = glob_counts / glob_counts.sum()

        # Print original class distribution
        if to_print:
            for i, (cls, cnt) in enumerate(zip(classes, glob_counts)):
                # print(f"{prefix}  {cls}: {cnt} cells ({glob_probs[i]:.3f})")
                pass

        # Identify low-frequency categories
        low_freq_mask = glob_probs < min_freq
        if np.any(low_freq_mask):
            low_freq_categories = [classes[i] for i, is_low in enumerate(low_freq_mask) if is_low]
            if to_print:
                print(
                    f"{prefix}  Removing {len(low_freq_categories)} low-frequency categories: {low_freq_categories}"
                )

            # Create a mapping from old to new category codes
            # Categories with frequency < min_freq will be mapped to -1 (to be excluded)
            category_mapping = np.arange(C)
            category_mapping[low_freq_mask] = -1

            # Apply mapping to y
            y_filtered = np.array(
                [category_mapping[code] if code < len(category_mapping) else -1 for code in y]
            )

            # Create a mask for cells with valid categories
            valid_mask = y_filtered >= 0

            if valid_mask.sum() < min_cells:
                if to_print:
                    print(
                        f"{prefix}Skipping {ct}: after removing low-frequency categories, too few cells remain ({valid_mask.sum()} < {min_cells})"
                    )
                continue

            # Filter the AnnData object
            ad_filtered = ad[valid_mask].copy()

            # Recalculate neighbors on filtered data
            nn_filtered = min(k, max(1, ad_filtered.n_obs - 1))
            sc.pp.neighbors(ad_filtered, use_rep=rep_key, n_neighbors=nn_filtered)
            Csp = ad_filtered.obsp["connectivities"].tocsr()

            # Update labels and codes
            # Create new categorical with only high-frequency categories
            [cat for i, cat in enumerate(classes) if not low_freq_mask[i]]
            ad_filtered.obs[f"{label_key}_filtered"] = (
                ad_filtered.obs[label_key].astype(str).astype("category")
            )

            # Get updated codes
            labels = ad_filtered.obs[f"{label_key}_filtered"].astype("category")
            classes = labels.cat.categories
            y = labels.cat.codes.to_numpy()
            C = len(classes)

            # Recalculate global proportions
            glob_counts = np.bincount(y, minlength=C).astype(float)
            glob_probs = glob_counts / glob_counts.sum()

            # Print filtered class distribution
            if to_print:
                for i, (cls, cnt) in enumerate(zip(classes, glob_counts)):
                    pass

            # Use the filtered data for kBET
            ad = ad_filtered
        else:
            if to_print:
                print(f"{prefix}  All categories have frequency >= {min_freq}, no filtering needed")

        # Get connectivities from the (potentially filtered) data
        indptr, indices = Csp.indptr, Csp.indices
        rejects = 0
        tested = 0

        # Track rejection reasons
        rejection_reasons = {"small_neighborhood": 0, "low_expected_counts": 0}

        # Run kBET on each cell
        for i in range(ad.n_obs):
            start, end = indptr[i], indptr[i + 1]
            neigh = indices[start:end]
            neigh = neigh[neigh != i]

            # Check neighborhood size
            if neigh.size < max(5, int(0.5 * nn)):  # skip tiny neighborhoods
                rejection_reasons["small_neighborhood"] += 1
                if to_print and i < 5:  # Only print for first few cells to avoid spam
                    print(
                        f"{prefix}  Cell {i} in {ct}: Skipped - neighborhood too small ({neigh.size} < {max(5, int(0.5 * nn))})"
                    )
                continue

            # Get neighbor labels and observed counts
            neigh_labels = y[neigh]
            obs = np.bincount(neigh_labels, minlength=C).astype(float)
            exp = glob_probs * obs.sum()

            # Check if expected counts are sufficient for chi-square test
            if np.any(exp < 1):
                rejection_reasons["low_expected_counts"] += 1
                if to_print and i < 5:  # Only print for first few cells to avoid spam
                    low_exp_indices = np.where(exp < 1)[0]
                    low_exp_classes = [classes[idx] for idx in low_exp_indices]
                    print(
                        f"{prefix}  Cell {i} in {ct}: Skipped - expected counts still too low for {low_exp_classes} (min={exp.min():.3f})"
                    )
                continue

            # Perform chi-square test
            stat, p = chisquare(obs, f_exp=exp)
            rejects += int(p < alpha)
            tested += 1

        # Check if any cells were tested
        if tested == 0:
            if to_print:
                print(f"{prefix}Skipping {ct}: no valid cells passed kBET requirements (tested=0)")
                # Print rejection summary
                print(f"{prefix}  Rejection summary for {ct} ({ad.n_obs} cells):")
                print(
                    f"{prefix}    Small neighborhood: {rejection_reasons['small_neighborhood']} cells"
                )
                print(
                    f"{prefix}    Low expected counts: {rejection_reasons['low_expected_counts']} cells"
                )
                print(
                    f"{prefix}    Total rejected: {rejection_reasons['small_neighborhood'] + rejection_reasons['low_expected_counts']} cells"
                )
            continue

        # Calculate rejection rate
        rej_rate = rejects / tested
        ct_scores[ct] = rej_rate

        # Print successful test summary
        if to_print:
            print(f"{prefix}kBET rejection {ct}: {rej_rate:.3f} ({tested}/{ad.n_obs} cells tested)")

            # Print rejection summary
            if (
                rejection_reasons["small_neighborhood"] > 0
                or rejection_reasons["low_expected_counts"] > 0
            ):
                print(f"{prefix}  Rejection summary for {ct}:")
                print(
                    f"{prefix}    Small neighborhood: {rejection_reasons['small_neighborhood']} cells"
                )
                print(
                    f"{prefix}    Low expected counts: {rejection_reasons['low_expected_counts']} cells"
                )
                print(
                    f"{prefix}    Total rejected: {rejection_reasons['small_neighborhood'] + rejection_reasons['low_expected_counts']} cells"
                )
                print(f"{prefix}    Success rate: {tested/ad.n_obs:.1%} of cells tested")

    if not ct_scores:
        if to_print:
            print(f"{prefix}No valid cell types for kBET")
        if return_neighbor_df:
            return float("nan"), pd.DataFrame()
        return float("nan")

    mean_rej = float(np.mean(list(ct_scores.values())))
    if to_print:
        print(f"{prefix}Mean kBET rejection: {mean_rej:.3f}")

    if group_key == "temp_obs_key" and "temp_obs_key" in adata.obs.columns:
        adata.obs.drop(columns=["temp_obs_key"], inplace=True)

    # Combine all neighbor dataframes
    if return_neighbor_df:
        if all_neighbor_dfs:
            neighbor_df = pd.concat(all_neighbor_dfs, ignore_index=True)
        else:
            neighbor_df = pd.DataFrame()
        return mean_rej, neighbor_df

    return mean_rej


from scipy.spatial.distance import cdist
from scipy.stats import anderson_ksamp


def cms_within_cell_types(
    adata,
    label_key="CN",
    group_key="cell_types",
    rep_key="X",
    k=30,
    alpha=0.05,
    min_cells=10,
    min_per_label=5,
    prefix="",
    print_flag=False,
):
    """
    CMS-like score: for each cell, test whether neighbor distance distributions
    differ across labels using k-sample Anderson–Darling; summarize as fraction
    of significant cells per cell type.
    """
    if group_key not in adata.obs.columns or label_key not in adata.obs.columns:
        print(f"{prefix}Missing '{group_key}' or '{label_key}' in .obs")
        return float("nan")

    ct_scores = {}
    for ct in adata.obs[group_key].unique():
        m = adata.obs[group_key] == ct
        ad = adata[m].copy()
        if ad.n_obs < min_cells or ad.obs[label_key].nunique() <= 1:
            continue

        nn = min(k, max(1, ad.n_obs - 1))
        sc.pp.neighbors(ad, use_rep=rep_key, n_neighbors=nn)

        # Extract neighbor indices and pairwise distances for each focal cell
        Csp = ad.obsp["connectivities"].tocsr()
        X = ad.obsm[rep_key] if rep_key != "X" else ad.X
        if not isinstance(X, np.ndarray):
            X = X.A if hasattr(X, "A") else np.asarray(X)
        y = ad.obs[label_key].astype("category").cat.codes.to_numpy()

        indptr, indices = Csp.indptr, Csp.indices
        sig = 0
        valid = 0
        for i in range(ad.n_obs):
            start, end = indptr[i], indptr[i + 1]
            neigh = indices[start:end]
            neigh = neigh[neigh != i]
            if neigh.size < max(5, int(0.5 * nn)):
                continue
            # distances from i to neighbors
            dists = np.linalg.norm(X[neigh] - X[i], axis=1)
            labs = y[neigh]
            groups = []
            ok = True
            for lab in np.unique(labs):
                vals = dists[labs == lab]
                if vals.size < min_per_label:
                    ok = False
                    break
                groups.append(vals)
            if not ok or len(groups) < 2:
                continue
            stat, crit, p = anderson_ksamp(groups)
            sig += int(p < alpha)
            valid += 1
        if valid == 0:
            continue
        frac_sig = sig / valid
        ct_scores[ct] = frac_sig
        if print_flag:
            print(f"{prefix}CMS significant frac {ct}: {frac_sig:.3f}")

    if not ct_scores:
        print(f"{prefix}No valid cell types for CMS")
        return float("nan")
    mean_frac = float(np.mean(list(ct_scores.values())))
    if print_flag:
        print(f"{prefix}Mean CMS significant frac: {mean_frac:.3f}")
    return mean_frac


from sklearn.metrics import silhouette_score


def silhouette_by_label_within_cell_types(
    adata,
    label_key="CN",
    group_key="cell_types",
    rep_key="X",
    metric="euclidean",
    min_cells=10,
    prefix="",
    print_flag=False,
):
    """
    Mean silhouette per cell type treating 'label_key' as cluster labels.
    Higher -> better geometric separation of labels.
    """
    if group_key not in adata.obs.columns or label_key not in adata.obs.columns:
        print(f"{prefix}Missing '{group_key}' or '{label_key}' in .obs")
        return float("nan")

    ct_scores = {}
    for ct in adata.obs[group_key].unique():
        m = adata.obs[group_key] == ct
        ad = adata[m].copy()
        if ad.n_obs < min_cells or ad.obs[label_key].nunique() < 2:
            continue

        X = ad.obsm[rep_key] if rep_key != "X" else ad.X
        if not isinstance(X, np.ndarray):
            X = X.A if hasattr(X, "A") else np.asarray(X)
        labels = ad.obs[label_key].to_numpy()
        try:
            s = silhouette_score(X, labels, metric=metric)
            ct_scores[ct] = float(s)
            if print_flag:
                print(f"{prefix}Silhouette {ct}: {s:.3f}")
        except Exception:
            continue

    if not ct_scores:
        print(f"{prefix}No valid cell types for silhouette")
        return float("nan")
    mean_sil = float(np.mean(list(ct_scores.values())))
    if print_flag:
        print(f"{prefix}Mean silhouette across cell types: {mean_sil:.3f}")
    return mean_sil


def silhouette_score_calc(combined_latent, print_flag=False):
    """Calculate silhouette score with proper NaN handling.

    Args:
        combined_latent: AnnData object with latent embedding in .X

    Returns:
        float: Silhouette score or NaN if calculation fails
    """
    try:
        # Check for NaN values in the data
        if hasattr(combined_latent.X, "toarray"):
            X = combined_latent.X.toarray()  # Convert sparse matrix to dense
        else:
            X = combined_latent.X

        # Check if X contains NaN values
        if np.isnan(X).any():
            if print_flag:
                print(
                    "Warning: Input X contains NaN values. Removing NaN values for silhouette calculation."
                )

            # Create mask for rows without NaN values
            mask = ~np.isnan(X).any(axis=1)
            if not mask.any():
                print("Error: All data contains NaN values. Cannot calculate silhouette score.")
                return float("nan")

            # Filter out rows with NaN values
            X_filtered = X[mask]
            labels_filtered = combined_latent.obs["cell_types"].iloc[mask].values

            # Check if we have enough data left
            if len(X_filtered) < 2 or len(np.unique(labels_filtered)) < 2:
                print(
                    f"Error: After removing NaN values, not enough data left for silhouette calculation. "
                    f"Rows: {len(X_filtered)}, Unique labels: {len(np.unique(labels_filtered))}"
                )
                return float("nan")

            if print_flag:
                print(
                    f"Calculating silhouette score with {len(X_filtered)}/{len(X)} rows after NaN removal."
                )
            silhouette_avg = silhouette_score(X_filtered, labels_filtered)
        else:
            # No NaN values, proceed normally
            silhouette_avg = silhouette_score(X, combined_latent.obs["cell_types"])

        return silhouette_avg

    except Exception as e:
        print(f"Error calculating silhouette score: {e}")
        return float("nan")


# returns list of indices of proteins that are most aligned with adata_rna.
# for example, first item in return array (adata_prot) is closest match to adata_rna
def calc_dist(rna_latent, prot_latent, label_key="cell_types"):
    """Calculate nearest protein cell types for each RNA cell.

    Args:
        rna_latent: AnnData object with RNA latent embedding
        prot_latent: AnnData object with protein latent embedding

    Returns:
        pandas.Series: Nearest protein cell types for each RNA cell
    """
    # Handle sparse matrices
    if hasattr(rna_latent.X, "toarray"):
        rna_X = rna_latent.X.toarray()
    else:
        rna_X = rna_latent.X

    if hasattr(prot_latent.X, "toarray"):
        prot_X = prot_latent.X.toarray()
    else:
        prot_X = prot_latent.X

    # Check for NaN values
    if np.isnan(rna_X).any() or np.isnan(prot_X).any():
        print("Warning: NaN values detected in latent space. Replacing with zeros.")
        rna_X = np.nan_to_num(rna_X, nan=0.0)
        prot_X = np.nan_to_num(prot_X, nan=0.0)

    # Calculate distances and find nearest neighbors
    distances = cdist(rna_X, prot_X, metric="euclidean")
    nearest_indices = np.argmin(distances, axis=1)  # protein index
    nn_celltypes_prot = prot_latent.obs[label_key].iloc[nearest_indices]
    return nn_celltypes_prot


# F1
def f1_score_calc(adata_rna, adata_prot, label_key="cell_types"):
    """Calculate F1 score.

    Args:
        adata_rna: AnnData object with RNA data
        adata_prot: AnnData object with protein data

    Returns:
        float: F1 score
    """
    return f1_score(
        adata_rna.obs[label_key],
        calc_dist(adata_rna, adata_prot, label_key=label_key),
        average="macro",
    )


# ARI
def ari_score_calc(adata_rna, adata_prot):
    """Calculate ARI score with error handling.

    Args:
        adata_rna: AnnData object with RNA data
        adata_prot: AnnData object with protein data

    Returns:
        float: ARI score or NaN if calculation fails
    """
    try:
        return adjusted_rand_score(adata_rna.obs["cell_types"], calc_dist(adata_rna, adata_prot))
    except Exception as e:
        print(f"Error calculating ARI score: {e}")
        return float("nan")


# matching_accuracy 1-1
def matching_accuracy(rna_latent, prot_latent, global_step=None, plot_flag=False):
    """
    Calculate matching accuracy between RNA and protein cells.
    Used in MaxFuse paper.

    Args:
        rna_latent: AnnData object with RNA latent embedding
        prot_latent: AnnData object with protein latent embedding
        global_step: Current global step for plotting (optional)
        plot_flag: Whether to plot the confusion matrix

    Returns:
        float: Matching accuracy or NaN if calculation fails
    """
    if "cell_types" not in rna_latent.obs.columns or "cell_types" not in prot_latent.obs.columns:
        print("Error: 'cell_types' column missing in data")
        return float("nan")

    # Check for NaN values in latent space
    if hasattr(rna_latent.X, "toarray"):
        rna_X = rna_latent.X.toarray()
    else:
        rna_X = rna_latent.X

    if hasattr(prot_latent.X, "toarray"):
        prot_X = prot_latent.X.toarray()
    else:
        prot_X = prot_latent.X

    if np.isnan(rna_X).any() or np.isnan(prot_X).any():
        print("Warning: NaN values detected in latent space. Results may be unreliable.")

    # Calculate nearest neighbors and matching accuracy
    correct_matches = 0
    nn_celltypes_prot = calc_dist(rna_latent, prot_latent)

    for index, cell_type in enumerate(rna_latent.obs["cell_types"]):
        if cell_type == nn_celltypes_prot.iloc[index]:
            correct_matches += 1

    accuracy = correct_matches / len(nn_celltypes_prot)

    # Generate confusion matrix plot if global_step is provided
    if plot_flag:
        if global_step is None:
            global_step = 0
        pf.plot_cell_type_prediction_confusion_matrix(
            rna_latent.obs["cell_types"], nn_celltypes_prot, global_step
        )

    return accuracy


def calc_dist_cn(rna_latent, prot_latent, k=3, pairwise_distances=None):
    """Calculate nearest protein CN for each RNA cell using k-nearest neighbors.

    Args:
        rna_latent: AnnData object with RNA latent embedding
        prot_latent: AnnData object with protein latent embedding
        pairwise_distances: Precomputed distance matrix (n_rna x n_prot).
                            If None, computed from adata.X. Shape: (n_rna, n_prot)
        k: Number of nearest neighbors to consider

    Returns:
        pandas.Series: Predicted CN values for each RNA cell
    """

    # ---- Handle pairwise_distances parameter ----
    if pairwise_distances is None:
        # Handle sparse matrices
        if hasattr(rna_latent.X, "toarray"):
            rna_X = rna_latent.X.toarray()
        else:
            rna_X = rna_latent.X

        if hasattr(prot_latent.X, "toarray"):
            prot_X = prot_latent.X.toarray()
        else:
            prot_X = prot_latent.X

        # Check for NaN values
        if np.isnan(rna_X).any() or np.isnan(prot_X).any():
            print("Warning: NaN values detected in latent space. Replacing with zeros.")
            rna_X = np.nan_to_num(rna_X, nan=0.0)
            prot_X = np.nan_to_num(prot_X, nan=0.0)

        # Calculate distances
        distances = cdist(rna_X, prot_X, metric="euclidean")
    else:
        # Verify pairwise_distances shape
        if (
            pairwise_distances.shape[0] != rna_latent.n_obs
            or pairwise_distances.shape[1] != prot_latent.n_obs
        ):
            raise ValueError(
                f"pairwise_distances shape {pairwise_distances.shape} doesn't match "
                f"(n_rna={rna_latent.n_obs}, n_prot={prot_latent.n_obs})"
            )
        distances = pairwise_distances

    # Get k nearest neighbors for each RNA cell
    topk_indices = np.argsort(distances, axis=1)[:, :k]

    # For each RNA cell, find the most common CN among its k nearest protein neighbors
    predicted_cn_values = []
    for i in range(topk_indices.shape[0]):
        neighbor_indices = topk_indices[i]
        neighbor_cn_values = prot_latent.obs["CN"].iloc[neighbor_indices]
        # Find most common CN (mode)
        cn_series = pd.Series(neighbor_cn_values)
        most_common_cn = (
            cn_series.value_counts(dropna=True).index[0]
            if not cn_series.dropna().empty
            else neighbor_cn_values.iloc[0]
        )
        predicted_cn_values.append(most_common_cn)

    return pd.Series(predicted_cn_values, index=rna_latent.obs.index)


def cross_modality_cn_accuracy(
    rna_latent, prot_latent, k=3, global_step=None, distance_matrix=None
):
    """
    Calculate CN (cell neighborhood) matching accuracy between RNA and protein cells.
    Similar to cross_modality_cell_type_accuracy but for CN labels.

    Args:
        rna_latent: AnnData object with RNA latent embedding
        prot_latent: AnnData object with protein latent embedding
        k: Number of nearest neighbors to consider for CN prediction
        global_step: Current global step for plotting (optional, only plots if provided)
        distance_matrix: Precomputed distance matrix (shape: n_rna x n_prot).
                        If provided, skips distance computation. Pass None to compute on-the-fly.

    Returns:
        float: CN matching accuracy or NaN if calculation fails
    """
    # Check for required columns
    if "CN" not in rna_latent.obs.columns or "CN" not in prot_latent.obs.columns:
        print("Error: 'CN' column missing in data")
        return float("nan")

    # Check for NaN values in latent space
    if hasattr(rna_latent.X, "toarray"):
        rna_X = rna_latent.X.toarray()
    else:
        rna_X = rna_latent.X

    if hasattr(prot_latent.X, "toarray"):
        prot_X = prot_latent.X.toarray()
    else:
        prot_X = prot_latent.X

    if np.isnan(rna_X).any() or np.isnan(prot_X).any():
        print("Warning: NaN values detected in latent space. Results may be unreliable.")

    # Calculate nearest neighbors and matching accuracy
    if distance_matrix is not None:
        # Use precomputed distance matrix
        predicted_cn = calc_dist_cn(
            rna_latent, prot_latent, k=k, pairwise_distances=distance_matrix
        )
    else:
        # Compute distance matrix on-the-fly
        predicted_cn = calc_dist_cn(rna_latent, prot_latent, k=k)

    correct_matches = 0
    for index, true_cn in enumerate(rna_latent.obs["CN"]):
        if true_cn == predicted_cn.iloc[index]:
            correct_matches += 1

    accuracy = correct_matches / len(predicted_cn)

    # Generate confusion matrix plot if global_step is provided
    if global_step is not None:
        from arcadia.plotting.training import plot_cn_prediction_confusion_matrix

        cell_types = (
            rna_latent.obs["cell_types"] if "cell_types" in rna_latent.obs.columns else None
        )
        plot_cn_prediction_confusion_matrix(
            rna_latent.obs["CN"], predicted_cn, cell_types, global_step
        )

    return accuracy


def calc_dist_cn_from_matrix(rna_latent, prot_latent, distance_matrix):
    """
    Get nearest CN labels from precomputed distance matrix.

    Args:
        rna_latent: AnnData object with RNA cell data
        prot_latent: AnnData object with protein cell data
        distance_matrix: Precomputed distance matrix (shape: n_rna x n_prot)

    Returns:
        pd.Series: Nearest protein CN labels for each RNA cell
    """
    nearest_indices = np.argmin(distance_matrix, axis=1)  # protein index
    nn_cn_prot = prot_latent.obs["CN"].iloc[nearest_indices]
    return nn_cn_prot


def normalize_silhouette(silhouette_vals):
    """Normalize silhouette scores from [-1, 1] to [0, 1]."""
    return (np.mean(silhouette_vals) + 1) / 2


def compute_silhouette_f1(rna_latent, prot_latent):
    """
    Compute the Silhouette F1 score from the MaxFuse paper.

    embeddings: np.ndarray, shape (n_samples, n_features)
    celltype_labels: list or array of ground-truth biological labels
    modality_labels: list or array of modality labels (e.g., RNA, ATAC)
    """

    # protein embeddings
    prot_embeddings = prot_latent.X
    # rna embeddings
    rna_embeddings = rna_latent.X
    embeddings = np.concatenate([rna_embeddings, prot_embeddings], axis=0)
    celltype_labels = np.concatenate(
        [rna_latent.obs["cell_types"], prot_latent.obs["cell_types"]], axis=0
    )
    modality_labels = np.concatenate(
        [["rna"] * len(rna_latent.obs), ["protein"] * len(prot_latent.obs)], axis=0
    )

    le_ct = LabelEncoder()
    le_mod = LabelEncoder()
    ct = le_ct.fit_transform(celltype_labels)
    mod = le_mod.fit_transform(modality_labels)

    slt_clust = normalize_silhouette(silhouette_samples(embeddings, ct))
    slt_mod_raw = silhouette_samples(embeddings, mod)
    slt_mod = 1 - normalize_silhouette(slt_mod_raw)  # We want mixing, so invert

    slt_f1 = (
        2 * (slt_clust * slt_mod) / (slt_clust + slt_mod + 1e-8)
    )  # just so we don't divide by zero
    return slt_f1


def morans_i(
    adata,
    score_key="matched_archetype_weight",
    use_rep="X",
    n_neighbors=15,
    neighbors_key=None,
):
    """
    Compute Moran's I spatial autocorrelation statistic for score organization in an embedding.

    Moran's I measures whether similar score values cluster together spatially in the
    embedding space, without assuming any linear trajectory or pseudotime structure. Higher
    values indicate better spatial organization where similar scores are located near each
    other in the embedding.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing embeddings and scores.
    score_key : str, optional (default: 'matched_archetype_weight')
        Key in `adata.obs` containing the continuous score values to analyze.
        If the key is not found in `adata.obs`, the function returns NaN.
        Examples: 'matched_archetype_weight', 'score', 'expression_level', etc.
    use_rep : str, optional (default: 'X')
        Key in `adata.obsm` (or 'X' for adata.X) specifying which representation to use
        for computing spatial relationships. This defines the embedding space where
        neighborhoods are computed.
        Examples: 'X', 'X_pca', 'X_umap', 'embedding1', etc.
    n_neighbors : int, optional (default: 15)
        Number of nearest neighbors to consider when building the spatial connectivity graph.
        Larger values capture more global structure; smaller values focus on local neighborhoods.
        Typical range: 10-30 for single-cell data.
    neighbors_key : str, optional (default: None)
        Key name to store the neighbors graph in adata.obsp. If None, automatically generates
        a key as 'morans_{use_rep}'. Useful to avoid recomputing neighbors or to prevent
        overwriting existing neighbor graphs.

    Returns
    -------
    float
        Moran's I statistic, ranging approximately from -1 to +1:

        - **I ≈ +1**: Strong positive spatial autocorrelation. Similar score values cluster
          together (e.g., high scores in one region, low scores in another region).

        - **I ≈ 0**: No spatial autocorrelation. Scores are randomly distributed across
          the embedding with no spatial pattern.

        - **I ≈ -1**: Strong negative spatial autocorrelation. Similar scores repel each
          other (e.g., high scores surrounded by low scores in a checkerboard pattern).

        - **NaN**: Returned if score_key not found, if S0=0 (no neighbors), or if
          denominator=0 (all scores identical).

    Notes
    -----
    The function computes Global Moran's I using the formula:

    .. math::
        I = \\frac{n}{S_0} \\cdot \\frac{\\sum_{i,j} W_{ij} (x_i - \\bar{x})(x_j - \\bar{x})}{\\sum_i (x_i - \\bar{x})^2}

    where:
    - n is the number of observations (cells)
    - W_{ij} is the spatial weight matrix (connectivity between cells i and j)
    - x_i are the score values
    - \\bar{x} is the mean score
    - S_0 = \\sum_{i,j} W_{ij} is the sum of all spatial weights

    The spatial weights W are determined by k-nearest neighbors in the embedding space,
    where cells are considered "spatial neighbors" if they are close in the specified
    representation.

    **Edge cases handled:**
    - Returns NaN if `score_key` does not exist in `adata.obs`
    - Returns NaN if S0 = 0 (no spatial connections, shouldn't happen with proper neighbors)
    - Returns NaN if denominator = 0 (all scores are identical, no variance)

    Examples
    --------
    >>> # Compare score organization in two embeddings
    >>> morans_pca = morans_i(adata, score_key='my_score', use_rep='X_pca')
    >>> morans_umap = morans_i(adata, score_key='my_score', use_rep='X_umap')
    >>> print(f"PCA Moran's I: {morans_pca:.3f}")
    >>> print(f"UMAP Moran's I: {morans_umap:.3f}")
    >>>
    >>> # Higher value indicates better spatial organization
    >>> if morans_pca > morans_umap:
    >>>     print("PCA shows more clustered organization")
    >>> else:
    >>>     print("UMAP shows more clustered organization")
    >>>
    >>> # Use with default archetype weight score
    >>> moran_score = morans_i(adata, use_rep='X_pca', n_neighbors=20)
    >>>
    >>> # Handle missing scores gracefully
    >>> result = morans_i(adata, score_key='nonexistent_score')
    >>> if np.isnan(result):
    >>>     print("Score key not found")

    References
    ----------
    .. [1] Moran, P.A.P. (1950). "Notes on Continuous Stochastic Phenomena".
           Biometrika. 37 (1–2): 17–23.
    .. [2] Cliff, A.D. and Ord, J.K. (1981). Spatial Processes: Models and Applications.
           Pion, London.

    See Also
    --------
    scanpy.pp.neighbors : Compute neighborhood graph used by this function
    scipy.stats.spearmanr : Alternative rank correlation metric
    sklearn.metrics.silhouette_score : Alternative clustering quality metric
    """
    # Check if score key exists, return NaN if not
    if score_key not in adata.obs.columns:
        return float("nan")

    # Generate unique key for storing neighbors graph
    key_added = neighbors_key if neighbors_key is not None else f"morans_{use_rep}"

    # Build k-nearest neighbors graph in embedding space
    # This defines which cells are "spatial neighbors"
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors, key_added=key_added)

    # Extract score values and connectivity matrix
    scores = adata.obs[score_key].to_numpy()
    W = adata.obsp[f"{key_added}_connectivities"]  # sparse CSR matrix of neighbor weights

    # Setup: number of cells, centered scores, sum of weights
    n = scores.size
    z = scores - scores.mean()  # center scores (deviations from mean)
    S0 = W.sum()  # sum of all spatial weights

    # Vectorized numerator: z^T W z and denominator: z^T z
    numerator = float(z @ (W @ z))
    denominator = float((z**2).sum())

    # Handle edge cases: no neighbors or no variance in scores
    if S0 == 0 or denominator == 0:
        return float("nan")

    # Calculate and return Moran's I statistic
    return (n / S0) * (numerator / denominator)


def compute_ari_f1(
    adata,
    celltype_key="cell_types",
    modality_key="modality",
    n_clusters=None,
    n_runs=10,
    random_state=42,
):
    """
    https://github.com/shuxiaoc/maxfuse/blob/7ccf6b4a32e01d013265b9c72ade8878d3172aa4/Archive/tonsil/code/benchmark/metrics.R#L176
    Compute the ARI F1 score exactly as defined in MaxFuse's R analysis code.

    Parameters:
    -----------
    adata : AnnData
        Integrated embedding with .X containing the embedding coordinates
    celltype_key : str
        Key in adata.obs for ground truth cell type labels
    modality_key : str
        Key in adata.obs for modality labels (e.g., 'rna', 'protein')
    n_clusters : int, optional
        Number of true cell type clusters. If None, inferred from celltype_key
    n_runs : int
        Number of k-means runs to average over (default 10)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    dict with keys: 'ari_mix', 'ari_clust', 'ari_f1'
    """
    # Constants for ARI normalization
    ARI_MIN = -1.0
    ARI_MAX = 1.0

    # Extract labels
    celltype_labels = LabelEncoder().fit_transform(adata.obs[celltype_key].astype(str).values)
    modality_labels = LabelEncoder().fit_transform(adata.obs[modality_key].astype(str).values)

    # Infer number of clusters if not provided
    if n_clusters is None:
        n_clusters = len(np.unique(celltype_labels))

    # Get embedding
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()

    # Initialize accumulators
    ari_mix_sum = 0.0
    ari_clust_sum = 0.0
    ari_f1_sum = 0.0

    # Run k-means multiple times and average
    np.random.seed(random_state)
    for run in range(n_runs):
        # K-means with 2 clusters for mixing metric
        kmeans_2 = KMeans(n_clusters=2, random_state=random_state + run, n_init=10)
        est_modality_labels = kmeans_2.fit_predict(X)

        # K-means with true number of clusters for biology metric
        kmeans_k = KMeans(n_clusters=n_clusters, random_state=random_state + run, n_init=10)
        est_celltype_labels = kmeans_k.fit_predict(X)

        # Calculate ARI for mixing (comparing k-means 2-cluster result vs modality)
        ari_mod = adjusted_rand_score(est_modality_labels, modality_labels)
        # Normalize to [0,1]
        ari_mod_norm = (ari_mod - ARI_MIN) / (ARI_MAX - ARI_MIN)
        # Invert: 1 - normalized ARI (higher = better mixing)
        curr_ari_mix = 1.0 - ari_mod_norm

        # Calculate ARI for biology (comparing k-means K-cluster result vs cell types)
        ari_bio = adjusted_rand_score(est_celltype_labels, celltype_labels)
        # Normalize to [0,1]
        curr_ari_clust = (ari_bio - ARI_MIN) / (ARI_MAX - ARI_MIN)

        # Calculate F1 (harmonic mean)
        curr_ari_f1 = 2.0 * curr_ari_mix * curr_ari_clust / (curr_ari_mix + curr_ari_clust + 1e-10)

        # Accumulate
        ari_mix_sum += curr_ari_mix
        ari_clust_sum += curr_ari_clust
        ari_f1_sum += curr_ari_f1

    # Average over runs
    return {
        "ari_mix": ari_mix_sum / n_runs,
        "ari_clust": ari_clust_sum / n_runs,
        "ari_f1": ari_f1_sum / n_runs,
    }


# Example usage:
# result = compute_ari_f1(adata, celltype_key="cell_types", modality_key="modality")
# print(f"ARI F1: {result['ari_f1']:.4f}")
# print(f"ARI mixing: {result['ari_mix']:.4f}")
# print(f"ARI biology: {result['ari_clust']:.4f}")


def calculate_color_entropy_within_cell_types(
    combined_latent,
    color_key: str = "CN",
    rep_key: str = "X",  # use_rep argument for sc.pp.neighbors
    k: int = 15,
    prefix: str = "",
    plot_flag: bool = False,
    min_cells: int = 10,
):
    """
    Entropy-based class mixing (normalized Shannon entropy) within each cell type cluster,
    analogous to CN iLISI structure.

    Args:
        combined_latent: AnnData with latent embedding in .X or .obsm, and obs columns:
                         - 'cell_types' (grouping)
                         - color_key (the class/color label to evaluate mixing)
        color_key: obs column name containing class/color labels (e.g., 'color', 'batch', 'CN')
        rep_key: use_rep for sc.pp.neighbors (e.g., 'X', 'X_pca', 'X_umap', or a key in .obsm)
        k: number of neighbors (min(n_obs-1, k) is used per subset)
        prefix: string prefix for printed metrics
        plot_flag: kept for API compatibility; not used here
        min_cells: minimum cells per cell type to compute the metric

    Returns:
        float: mean normalized entropy across cell types
    """

    # Basic checks mirroring the iLISI-style guardrails
    if (
        "cell_types" not in combined_latent.obs.columns
        or color_key not in combined_latent.obs.columns
    ):

        print(f"{prefix}Missing 'cell_types' or '{color_key}' in .obs")
        return float("nan")

    # Helper to compute normalized entropy given neighbor label matrix
    def _normalized_entropy(counts_row):
        total = counts_row.sum()
        if total <= 0:
            return 0.0
        p = counts_row / total
        p = p[p > 0]
        H = -np.sum(p * np.log(p))
        # normalization by log(C) -> [0,1]
        C = counts_row.size
        return float(H / np.log(C)) if C > 1 else 0.0

    cell_type_scores = {}
    cell_types = combined_latent.obs["cell_types"].unique()

    for ct in cell_types:
        mask = combined_latent.obs["cell_types"] == ct
        adata_ct = combined_latent[mask].copy()

        # Skip small groups or groups with a single color
        if adata_ct.n_obs < min_cells:
            if plot_flag:
                print(f"{prefix}Skipping {ct}: insufficient cells (n={adata_ct.n_obs})")
            continue
        if adata_ct.obs[color_key].nunique() <= 1:
            if plot_flag:
                print(f"{prefix}Skipping {ct}: only one {color_key} present")
            continue

        # Build neighbors on the chosen representation
        nn = min(k, max(1, adata_ct.n_obs - 1))
        sc.pp.neighbors(adata_ct, use_rep=rep_key, n_neighbors=nn)

        # adata_ct.obsp['distances'] or 'connectivities' stores kNN graph; we’ll use indices via sc.get.neighbors
        # Extract neighbor indices from connectivities (sparse)
        Csp = adata_ct.obsp.get("connectivities", None)
        if Csp is None:
            if plot_flag:
                print(f"{prefix}No connectivities for {ct}, skipping")
            continue

        Csp = Csp.tocsr()
        labels = adata_ct.obs[color_key].astype("category")
        classes = labels.cat.categories
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = labels.cat.codes.to_numpy()

        # For each cell, gather neighbor label counts (excluding self)
        indptr, indices, data = Csp.indptr, Csp.indices, Csp.data
        n = adata_ct.n_obs
        C = len(classes)
        H_norm = np.zeros(n, dtype=float)

        for i in range(n):
            start, end = indptr[i], indptr[i + 1]
            neigh_idx = indices[start:end]
            # drop self if present
            neigh_idx = neigh_idx[neigh_idx != i]
            if neigh_idx.size == 0:
                H_norm[i] = 0.0
                continue
            neigh_labels = y_idx[neigh_idx]
            counts = np.bincount(neigh_labels, minlength=C).astype(float)
            H_norm[i] = _normalized_entropy(counts)

        # Aggregate for this cell type
        ct_score = float(np.mean(H_norm))
        cell_type_scores[ct] = ct_score
        if plot_flag:
            print(f"{prefix}Color entropy score for {ct}: {ct_score:.4f}")

    if not cell_type_scores:
        if plot_flag:
            print(f"{prefix}No valid cell type scores computed")
        return float("nan")

    mean_entropy = float(np.mean(list(cell_type_scores.values())))
    if plot_flag:
        print(f"{prefix}Mean color entropy across cell types: {mean_entropy:.4f}")
    return mean_entropy


def calculate_cn_ilisi_within_cell_types(
    combined_latent: AnnData, prefix: str = "", plot_flag: bool = False
):
    """
    Calculate CN iLISI score within each cell type cluster.

    Args:
        combined_latent: AnnData object with combined RNA and protein data latent embeddings
        prefix: Prefix for metric names (e.g., "val_", "train_")

    Returns:
        dict: Dictionary containing CN iLISI metrics
    """

    if "cell_types" in combined_latent.obs.columns and "CN" in combined_latent.obs.columns:
        cell_type_cn_ilisi_scores = {}
        cell_types = combined_latent.obs["cell_types"].unique()

        for cell_type in cell_types:
            # Filter data for this cell type
            cell_type_mask = combined_latent.obs["cell_types"] == cell_type
            cell_type_data = combined_latent[cell_type_mask].copy()

            # Only calculate if we have enough cells and multiple CN values
            if (
                cell_type_data.n_obs >= 10
                and "CN" in cell_type_data.obs.columns
                and cell_type_data.obs["CN"].nunique() > 1
            ):

                # Calculate neighbors for this cell type subset
                sc.pp.neighbors(
                    cell_type_data, use_rep="X", n_neighbors=min(15, cell_type_data.n_obs - 1)
                )

                # Calculate iLISI for CN within this cell type
                cn_ilisi_score = calculate_iLISI(cell_type_data, "CN", plot_flag=False)

                # Validate the CN iLISI score
                if np.isfinite(cn_ilisi_score) and cn_ilisi_score > 0:
                    cell_type_cn_ilisi_scores[cell_type] = float(cn_ilisi_score)
                    if plot_flag:
                        print(f"{prefix}CN iLISI score for {cell_type}: {cn_ilisi_score:.4f}")
                else:
                    if plot_flag:
                        print(f"Invalid CN iLISI score for {cell_type}: {cn_ilisi_score}")
            else:
                if plot_flag:
                    print(
                        f"Skipping CN iLISI calculation for {cell_type}: insufficient data (n_cells={cell_type_data.n_obs}, n_CN={cell_type_data.obs['CN'].nunique() if 'CN' in cell_type_data.obs.columns else 0})"
                    )

        # Calculate mean CN iLISI score across all cell types
        if cell_type_cn_ilisi_scores:
            mean_cn_ilisi = np.mean(list(cell_type_cn_ilisi_scores.values()))
            if plot_flag:
                print(f"{prefix}Mean CN iLISI across cell types: {mean_cn_ilisi:.4f}")
        else:
            if plot_flag:
                print(f"{prefix}No valid cell type CN iLISI scores calculated")
    else:
        if plot_flag:
            print(
                f"{prefix}Missing 'cell_types' or 'CN' columns in combined_latent.obs for CN iLISI calculation"
            )

    return float(mean_cn_ilisi)


def calculate_iLISI(
    adata,
    batch_key="batch",
    neighbors_key="neighbors",
    plot_flag=False,
    use_subsample=True,
    global_step=None,
):
    """
    Calculate integration Local Inverse Simpson's Index (LISI) using precomputed neighbors.

    The iLISI score measures how well different batches are mixed in the embedding space.
    Higher scores indicate better batch mixing, with a minimum value of 1
    (all neighbors same batch) and maximum of k+1 (all neighbors different batches),
    where k is the number of neighbors used.

    For large imbalanced datasets (>10k cells with >10% size difference between batches),
    this function automatically applies balanced stratified subsampling to match the gold
    standard LISI implementation approach. This prevents bias from the majority batch
    dominating the neighbor graph.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with precomputed neighbors
    batch_key : str, default='batch'
        Column in adata.obs containing batch labels
    neighbors_key : str, default='neighbors'
        Key where neighbor information is stored in adata.uns
    use_subsample: bool:
        for ploting part of the batch

    Returns
    -------
    float
        Median iLISI score across all cells. Higher values indicate better
        batch mixing in the embedding space.

    Notes
    -----
    When dataset is imbalanced (e.g., 80k protein + 12k RNA), the function uses
    balanced subsampling (minority class size from each batch) to get unbiased scores.
    """

    if neighbors_key not in adata.uns:
        raise ValueError(f"Run sc.pp.neighbors with key='{neighbors_key}' first")

    connectivities = adata.obsp[f"connectivities"]
    n_cells = adata.n_obs

    # Check for imbalance and apply balanced subsampling if needed
    if batch_key in adata.obs:
        batch_counts = adata.obs[batch_key].value_counts()
        if len(batch_counts) >= 2:
            total_cells = batch_counts.sum()
            min_count = batch_counts.min()
            max_count = batch_counts.max()

            # 10% size difference threshold
            size_difference_pct = abs(max_count - min_count) / total_cells
            is_imbalanced = size_difference_pct > 0.10
            is_large = n_cells > 10000

            if is_imbalanced and is_large:
                # Use stratified balanced sampling - use full minority class size
                sample_size_per_batch = min_count

                balanced_indices = []
                for batch in batch_counts.index:
                    batch_mask = adata.obs[batch_key] == batch
                    batch_indices = np.where(batch_mask)[0]
                    sampled = np.random.choice(batch_indices, sample_size_per_batch, replace=False)
                    balanced_indices.extend(sampled)

                # Shuffle to mix modalities
                np.random.shuffle(balanced_indices)

                # Create balanced subset
                adata_balanced = adata[balanced_indices].copy()

                # Recalculate neighbors on balanced data
                sc.pp.neighbors(adata_balanced, use_rep="X", n_neighbors=15)

                # Log the balancing
                if plot_flag:
                    print(
                        f"iLISI: Dataset imbalanced ({size_difference_pct*100:.1f}% size diff). "
                        f"Using balanced subsample: {sample_size_per_batch} cells per modality "
                        f"(original: {dict(batch_counts)})."
                    )

                # Use balanced data for iLISI
                adata = adata_balanced
                n_cells = adata.n_obs
                connectivities = adata.obsp["connectivities"]
    if False:  # ignore for now
        if use_subsample:
            sample_size = min(300, n_cells)  # Use smaller of 300 or total cells
            subset_indices = np.random.choice(n_cells, sample_size, replace=False)
            subset_connectivities = connectivities[subset_indices][:, subset_indices]
        else:
            subset_connectivities = connectivities
        plt.figure(figsize=(10, 8))
        plt.title(
            "neighbors, first half are RNA cells \nthe second half, protein cells (subset of 300)"
        )
        sns.heatmap(subset_connectivities.todense())
        mid_point = subset_connectivities.shape[1] // 2
        plt.axvline(x=mid_point, color="red", linestyle="--", linewidth=2)
        plt.axhline(y=mid_point, color="red", linestyle="--", linewidth=2)
        if global_step is not None:
            mlflow.log_figure(plt.gcf(), f"step_{global_step:05d}_neighbor_heatmap.pdf")
            plt.savefig(f"step_{global_step:05d}_neighbor_heatmap.pdf")
        else:
            mlflow.log_figure(plt.gcf(), "neighbor_heatmap.pdf")
            plt.savefig("neighbor_heatmap.pdf")
        plt.show()
        plt.close()
    lisi_scores = []
    for i in range(n_cells):
        neighbors = connectivities[i].indices
        neighbors = np.append(neighbors, i)

        batches = adata.obs[batch_key].iloc[neighbors].values
        try:
            unique_batches, counts = np.unique(batches, return_counts=True)
        except:
            print(f"batches: {batches}")
            # how many nan in batches
            x = adata.obs["CN"].iloc[neighbors][adata.obs["modality"] == "RNA"]
            y = adata.obs["CN"].iloc[neighbors][adata.obs["modality"] == "Protein"]
            print(f"nan in x: {pd.isna(x).sum()}")
            print(f"nan in y: {pd.isna(y).sum()}")
            print(f"batches: {batches}")
            print(f"batches: {batches}")
            print(f"type of batches: {type(batches)}")
        # Convert to strings to handle mixed data types (e.g., NaN floats and CN strings)

        batches = [str(batch) for batch in batches]
        unique_batches, counts = np.unique(batches, return_counts=True)

        proportions = counts / len(neighbors)
        simpson = np.sum(proportions**2)
        lisi = 1 / simpson if simpson > 0 else 0
        lisi_scores.append(lisi)

    if np.isnan(np.median(lisi_scores)):
        raise ValueError("iLISI score is NaN.")

    return np.median(lisi_scores)


def calculate_batch_ilisi_score(adata, batch_key, n_neighbors=15):
    """Calculate how well batches are mixed using iLISI (integration Local Inverse Simpson's Index)"""
    import scanpy as sc

    # Ensure neighbors are computed
    # Adjust n_neighbors for small datasets
    max_neighbors = min(n_neighbors, adata.n_obs - 1)
    if max_neighbors < 5:
        max_neighbors = min(5, adata.n_obs - 1)
    sc.pp.neighbors(adata, n_neighbors=max_neighbors)

    # Use the existing calculate_iLISI function
    return calculate_iLISI(adata, batch_key=batch_key)


def calculate_cell_type_silhouette(adata, celltype_key, use_rep="X_pca"):
    """Calculate how well cell types cluster together"""
    if celltype_key not in adata.obs.columns:
        return None

    if use_rep in adata.obsm:
        X = adata.obsm[use_rep]
    else:
        X = adata.X

    labels = adata.obs[celltype_key].values
    # Only calculate if we have multiple cell types
    if len(np.unique(labels)) < 2:
        return None

    score = silhouette_score(X, labels)
    return score


def calculate_cLISI(adata, label_key="cell_type", neighbors_key="neighbors", plot_flag=False):
    """
    Calculate cell-type Local Inverse Simpson's Index (LISI) using precomputed neighbors.

    The cLISI score measures how well cell types are separated in the embedding space.
    Higher scores indicate better cell type separation, with a minimum value of 1
    (all neighbors same cell type) and maximum of k+1 (all neighbors different cell types),
    where k is the number of neighbors used.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with precomputed neighbors
    label_key : str, default='cell_type'
        Column in adata.obs containing cell type labels
    neighbors_key : str, default='neighbors'
        Key where neighbor information is stored in adata.uns

    Returns
    -------
    float
        Median cLISI score across all cells. Higher values indicate better
        cell type separation in the embedding space.
    """

    if neighbors_key not in adata.uns:
        raise ValueError(f"Run sc.pp.neighbors with key='{neighbors_key}' first")

    connectivities = adata.obsp[f"connectivities"]
    n_cells = adata.n_obs

    lisi_scores = []
    for i in range(n_cells):
        neighbors = connectivities[i].indices
        neighbors = np.append(neighbors, i)

        labels = adata.obs[label_key].iloc[neighbors].values
        unique_labels, counts = np.unique(labels, return_counts=True)

        proportions = counts / len(neighbors)
        simpson = np.sum(proportions**2)
        lisi = 1 / simpson if simpson > 0 else 0
        lisi_scores.append(lisi)

    return np.median(lisi_scores)


def mixing_score(
    rna_inference_outputs_mean,
    protein_inference_outputs_mean,
    adata_rna_subset,
    adata_prot_subset,
    index_rna=None,
    index_prot=None,
    plot_flag=False,
):
    if index_rna is None:
        index_rna = np.arange(len(rna_inference_outputs_mean))
    if index_prot is None:
        index_prot = np.arange(len(protein_inference_outputs_mean))
    if isinstance(rna_inference_outputs_mean, torch.Tensor):
        rna_latent = rna_inference_outputs_mean.clone().detach().cpu().numpy()
        prot_latent = protein_inference_outputs_mean.clone().detach().cpu().numpy()
    else:
        rna_latent = rna_inference_outputs_mean
        prot_latent = protein_inference_outputs_mean
    combined_latent = ad.concat(
        [AnnData(rna_latent), AnnData(prot_latent)],
        join="outer",
        label="modality",
        keys=["RNA", "Protein"],
    )
    combined_major_cell_types = pd.concat(
        (
            adata_rna_subset[index_rna].obs["major_cell_types"],
            adata_prot_subset[index_prot].obs["major_cell_types"],
        ),
        join="outer",
    )
    combined_latent.obs["major_cell_types"] = combined_major_cell_types.values
    sc.pp.pca(combined_latent)
    sc.pp.neighbors(combined_latent, use_rep="X")
    iLISI = calculate_iLISI(combined_latent, "modality", plot_flag=plot_flag)
    # cLISI = calculate_cLISI(combined_latent, "major_cell_types", plot_flag=plot_flag)
    return {"iLISI": iLISI}  # , "cLISI": cLISI}


def calculate_post_training_metrics(adata_rna, adata_prot, prot_matches_in_rna):
    """Calculate various metrics for model evaluation."""
    from sklearn.metrics import adjusted_mutual_info_score

    # Ensure arrays have consistent lengths
    rna_cell_types = adata_rna.obs["cell_types"].values[prot_matches_in_rna]
    prot_cell_types = adata_prot.obs["cell_types"].values

    if len(rna_cell_types) != len(prot_cell_types):
        import warnings

        warnings.warn(
            f"Mismatched array lengths: RNA {len(rna_cell_types)} vs Protein {len(prot_cell_types)}"
        )
        # Use the minimum length to avoid errors
        min_len = min(len(rna_cell_types), len(prot_cell_types))
        rna_cell_types = rna_cell_types[:min_len]
        prot_cell_types = prot_cell_types[:min_len]

    # Calculate NMI scores
    nmi_cell_types_cn_rna = adjusted_mutual_info_score(
        adata_rna.obs["cell_types"], adata_rna.obs["CN"]
    )
    nmi_cell_types_cn_prot = adjusted_mutual_info_score(
        adata_prot.obs["cell_types"], adata_prot.obs["CN"]
    )
    nmi_cell_types_modalities = adjusted_mutual_info_score(
        rna_cell_types,
        prot_cell_types,
    )

    # Calculate accuracy
    matches = rna_cell_types == prot_cell_types
    accuracy = matches.sum() / len(matches)

    # Calculate mixing score
    mixing_result = mixing_score(
        adata_rna.obsm["X_scVI"],
        adata_prot.obsm["X_scVI"],
        adata_rna,
        adata_prot,
        index_rna=range(len(adata_rna)),
        index_prot=range(len(adata_prot)),
        plot_flag=True,
    )

    return {
        "nmi_cell_types_cn_rna": nmi_cell_types_cn_rna,
        "nmi_cell_types_cn_prot": nmi_cell_types_cn_prot,
        "nmi_cell_types_modalities": nmi_cell_types_modalities,
        "cell_type_matching_accuracy": accuracy,
        "mixing_score_ilisi": mixing_result["iLISI"],
    }


if __name__ == "__main__":
    import os
    from datetime import datetime
    from pathlib import Path

    import scanpy as sc

    # Set working directory to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Path to trained data directory
    data_dir = Path("processed_data").absolute()

    # Find latest RNA and protein files
    from arcadia.data_utils.loading import get_latest_file

    rna_file = get_latest_file(data_dir, "rna_vae_trained")
    prot_file = get_latest_file(data_dir, "protein_vae_trained")

    if not rna_file or not prot_file:
        print("Error: Could not find trained data files.")
        sys.exit(1)

    print(f"Using RNA file: {os.path.basename(rna_file)}")
    print(f"Using Protein file: {os.path.basename(prot_file)}")

    # Load data
    print("\nLoading data...")
    adata_rna = sc.read_h5ad(rna_file)
    adata_prot = sc.read_h5ad(prot_file)
    print("✓ Data loaded")

    # Combine data for silhouette score
    combined_latent = sc.concat([adata_rna, adata_prot], join="outer")

    # Calculate and print all metrics
    print("\nCalculating metrics...")
    silhouette = silhouette_score_calc(combined_latent)
    f1 = f1_score_calc(adata_rna, adata_prot)
    ari = ari_score_calc(adata_rna, adata_prot)
    accuracy = matching_accuracy(adata_rna, adata_prot)

    # Calculate advanced metrics if available
    silhouette_f1 = compute_silhouette_f1(adata_rna, adata_prot)
    ari_f1 = compute_ari_f1(adata_rna, adata_prot)
    has_advanced_metrics = True

    # Print results
    print(f"\nMetrics Results:")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"ARI Score: {ari:.3f}")
    print(f"Matching Accuracy: {accuracy:.3f}")

    if has_advanced_metrics:
        print(f"Silhouette F1 Score: {silhouette_f1.mean():.3f}")
        print(f"ARI F1 Score: {ari_f1:.3f}")

    # Save results to log file
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = Path("logs").absolute()
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / f"metrics_log_{timestamp}.txt"

    with open(log_file, "w") as f:
        f.write(f"Metrics calculated on: {timestamp}\n")
        f.write(f"RNA file: {os.path.basename(rna_file)}\n")
        f.write(f"Protein file: {os.path.basename(prot_file)}\n\n")
        f.write(f"Silhouette Score: {silhouette:.3f}\n")
        f.write(f"F1 Score: {f1:.3f}\n")
        f.write(f"ARI Score: {ari:.3f}\n")
        f.write(f"Matching Accuracy: {accuracy:.3f}\n")

        if has_advanced_metrics:
            f.write(f"Silhouette F1 Score: {silhouette_f1.mean():.3f}\n")
            f.write(f"ARI F1 Score: {ari_f1:.3f}\n")

    print(f"\nResults saved to: {log_file}")
    print("\nMetrics calculation completed!")


def is_valid_metric_value(val):
    """Check if a value is valid (not None/NaN/Inf) for MLflow logging.

    Args:
        val: Value to check (can be numpy array, float, or other numeric type)

    Returns:
        bool: True if value is valid, False otherwise
    """
    if val is None:
        return False
    if hasattr(val, "dtype"):
        return not (np.isnan(val) or np.isinf(val))
    if isinstance(val, float):
        return val == val and val != float("inf") and val != float("-inf")
    return True


def extract_training_metrics_from_history(history, metric_mapping):
    """Extract final training metrics from history dictionary.

    Args:
        history: Training history dictionary with metric names as keys
        metric_mapping: Dictionary mapping output metric names to history keys

    Returns:
        dict: Dictionary of valid metrics ready for MLflow logging
    """
    metrics_to_log = {}
    for key, hist_key in metric_mapping.items():
        if hist_key in history and history[hist_key] and len(history[hist_key]) > 0:
            last_val = history[hist_key][-1]
            if is_valid_metric_value(last_val):
                metrics_to_log[key] = float(last_val) if hasattr(last_val, "dtype") else last_val
    return metrics_to_log


def process_post_training_metrics(metrics):
    """Process post-training metrics, filtering out invalid values.

    Args:
        metrics: Dictionary of metric names to values

    Returns:
        tuple: (valid_metrics dict, nan_metrics list)
    """
    valid_metrics = {}
    nan_metrics = []
    for k, v in metrics.items():
        if is_valid_metric_value(v):
            valid_metrics[k] = round(float(v) if hasattr(v, "dtype") else v, 3)
        else:
            nan_metrics.append(k)
    return valid_metrics, nan_metrics
