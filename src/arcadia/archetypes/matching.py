"""Archetype matching utilities."""

import copy

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from arcadia.archetypes.cell_representations import identify_extreme_archetypes_percentile
from arcadia.utils.logging import setup_logger


def identify_extreme_archetypes_balanced(
    archetype_vectors, adata=None, logger_=None, percentile=95, to_print=False
):
    """
    Identify extreme archetypes using balanced selection with minimum constraints and quality filtering.

    New method:
    1. Calculate top 5% from each archetype
    2. Apply minimum of 10 cells per archetype
    3. Skip low quality archetypes
    4. Use minimum calculated count as max for all archetypes (balanced selection)

    Parameters:
    -----------
    archetype_vectors : numpy array or torch.Tensor
        The archetype vectors for cells
    adata : AnnData, optional
        AnnData object containing archetype quality in adata.uns['archetype_quality']
    percentile : float
        The percentile threshold (e.g., 95 means take top 5% from each archetype separately)

    Returns:
    --------
    extreme_mask : boolean array
        Mask indicating which cells are extreme archetypes
    dominant_dims : array
        The dominant dimension for each extreme archetype
    threshold : float
        Average threshold used across archetypes
    """
    if logger_ is None:
        logger_ = setup_logger()

    if isinstance(archetype_vectors, np.ndarray) or isinstance(archetype_vectors, pd.DataFrame):
        if isinstance(archetype_vectors, pd.DataFrame):
            archetype_vectors = archetype_vectors.values

        # Calculate proportion of each vector that is in its max dimension
        max_values = np.max(archetype_vectors, axis=1)
        total_values = np.sum(archetype_vectors, axis=1)
        proportions = max_values / total_values

        # Get the dominant dimension for each cell (for return value and labeling)
        dominant_dims = np.argmax(archetype_vectors, axis=1)
        labels = dominant_dims
        n_archetypes = archetype_vectors.shape[1]

        # Calculate cells per archetype as percentage of cells dominated by each archetype
        # This ensures each archetype contributes its top percentile regardless of dominance

        # Initialize extreme mask
        total_cells = len(archetype_vectors)
        extreme_mask = np.zeros(total_cells, dtype=bool)

        thresholds = []

        # New method: Calculate 5% from each archetype with minimum and balancing constraints
        unique_labels = np.unique(labels)
        archetype_cell_counts = []

        # Get archetype quality from adata.uns if available
        archetype_quality = None
        if adata is not None and "archetype_quality" in adata.uns:
            quality_dict = adata.uns["archetype_quality"]
            # Convert dict to list: {'0': True, '1': True, ...} -> [True, True, ...]
            archetype_quality = [quality_dict.get(str(i), True) for i in range(n_archetypes)]
            if to_print:
                logger_.info(f"Found archetype quality: {archetype_quality}")

        # First pass: Calculate how many cells each archetype should contribute
        for archetype_label in unique_labels:
            # Skip low quality archetypes if quality info is available
            if archetype_quality is not None:
                archetype_idx = int(archetype_label)
                if archetype_idx < len(archetype_quality) and not archetype_quality[archetype_idx]:
                    if to_print:
                        logger_.info(f"Skipping archetype {archetype_label} (low quality)")
                    continue
            # Find cells that have this archetype label
            archetype_cells_mask = labels == archetype_label
            archetype_cells_indices = np.where(archetype_cells_mask)[0]

            if len(archetype_cells_indices) == 0:
                continue

            # Calculate 5% of total cells, distributed among archetypes
            cells_5_percent_total = int(total_cells * (100 - percentile) / 100)
            # Distribute among valid archetypes (rough estimate)
            cells_5_percent = max(1, cells_5_percent_total // len(unique_labels))
            # Apply minimum constraint (at least 10 cells) but cap at archetype size
            cells_per_this_archetype = min(len(archetype_cells_indices), max(10, cells_5_percent))

            archetype_cell_counts.append(cells_per_this_archetype)

        # Find the minimum count across all archetypes (this becomes the max for balancing)
        if archetype_cell_counts:
            balanced_cell_count = min(archetype_cell_counts)
        else:
            balanced_cell_count = 10

        # Second pass: Select cells using the balanced count
        for archetype_label in unique_labels:
            # Skip low quality archetypes if quality info is available
            if archetype_quality is not None:
                archetype_idx = int(archetype_label)
                if archetype_idx < len(archetype_quality) and not archetype_quality[archetype_idx]:
                    continue
            # Find cells that have this archetype label
            archetype_cells_mask = labels == archetype_label
            archetype_cells_indices = np.where(archetype_cells_mask)[0]

            if len(archetype_cells_indices) == 0:
                continue

            # Get proportions for cells belonging to this archetype
            archetype_proportions = proportions[archetype_cells_indices]

            # Use the balanced cell count for all archetypes
            cells_per_this_archetype = min(balanced_cell_count, len(archetype_cells_indices))

            # Select top cells_per_this_archetype most extreme cells from this archetype
            if len(archetype_cells_indices) >= cells_per_this_archetype:
                # Get indices of top cells_per_this_archetype cells
                top_indices = np.argpartition(archetype_proportions, -cells_per_this_archetype)[
                    -cells_per_this_archetype:
                ]
                selected_cell_indices = archetype_cells_indices[top_indices]
                threshold = np.min(archetype_proportions[top_indices])
            else:
                # If we have fewer cells than desired, take all of them
                selected_cell_indices = archetype_cells_indices
                threshold = np.min(archetype_proportions) if len(archetype_proportions) > 0 else 0

            extreme_mask[selected_cell_indices] = True
            thresholds.append(threshold)

        # Average threshold across archetypes
        avg_threshold = np.mean(thresholds) if thresholds else 0

        if to_print:
            logger_.info(
                f"\nBalanced extreme selection: {np.sum(extreme_mask)} out of {len(extreme_mask)} cells are extreme archetypes ({np.sum(extreme_mask)/len(extreme_mask):.1%})"
            )
            logger_.info(
                f"Method: top {100-percentile}% per archetype, min 10 cells, balanced to {balanced_cell_count} cells per archetype"
            )
            logger_.info(f"Average threshold: {avg_threshold:.3f}")

            # Print distribution across archetypes
            for archetype_label in unique_labels:
                count = np.sum((labels == archetype_label) & extreme_mask)
                total_labeled = np.sum(labels == archetype_label)
                percentage = (count / total_labeled * 100) if total_labeled > 0 else 0
                logger_.info(
                    f"Archetype {archetype_label}: {count} extreme cells from {total_labeled} labeled ({percentage:.1f}%)"
                )

    elif isinstance(archetype_vectors, torch.Tensor):
        # Calculate proportion of each vector that is in its max dimension
        max_values = torch.max(archetype_vectors, dim=1).values
        total_values = torch.sum(archetype_vectors, dim=1)
        proportions = max_values / total_values

        # Get the dominant dimension for each cell (for return value and labeling)
        dominant_dims = torch.argmax(archetype_vectors, dim=1)
        labels = dominant_dims
        n_archetypes = archetype_vectors.shape[1]

        # Calculate cells per archetype as percentage of cells dominated by each archetype
        # This ensures each archetype contributes its top percentile regardless of dominance
        total_cells = len(archetype_vectors)

        # Initialize extreme mask
        extreme_mask = torch.zeros(total_cells, dtype=torch.bool, device=archetype_vectors.device)

        thresholds = []

        # New method: Calculate 5% from each archetype with minimum and balancing constraints
        unique_labels = torch.unique(labels)
        archetype_cell_counts = []

        # Get archetype quality from adata.uns if available
        archetype_quality = None
        if adata is not None and "archetype_quality" in adata.uns:
            quality_dict = adata.uns["archetype_quality"]
            # Convert dict to list: {'0': True, '1': True, ...} -> [True, True, ...]
            archetype_quality = [quality_dict.get(str(i), True) for i in range(n_archetypes)]
            if to_print:
                logger_.info(f"Found archetype quality: {archetype_quality}")

        # First pass: Calculate how many cells each archetype should contribute
        for archetype_label in unique_labels:
            # Skip low quality archetypes if quality info is available
            if archetype_quality is not None:
                archetype_idx = int(archetype_label.item())
                if archetype_idx < len(archetype_quality) and not archetype_quality[archetype_idx]:
                    if to_print:
                        logger_.info(f"Skipping archetype {archetype_label.item()} (low quality)")
                    continue
            # Find cells that have this archetype label
            archetype_cells_mask = labels == archetype_label
            archetype_cells_indices = torch.where(archetype_cells_mask)[0]

            if len(archetype_cells_indices) == 0:
                continue

            # Calculate 5% of total cells, distributed among archetypes
            cells_5_percent_total = int(total_cells * (100 - percentile) / 100)
            # Distribute among valid archetypes (rough estimate)
            cells_5_percent = max(1, cells_5_percent_total // len(unique_labels))
            # Apply minimum constraint (at least 10 cells) but cap at archetype size
            cells_per_this_archetype = min(len(archetype_cells_indices), max(10, cells_5_percent))

            archetype_cell_counts.append(cells_per_this_archetype)

        # Find the minimum count across all archetypes (this becomes the max for balancing)
        if archetype_cell_counts:
            balanced_cell_count = min(archetype_cell_counts)
        else:
            balanced_cell_count = 10

        # Second pass: Select cells using the balanced count
        for archetype_label in unique_labels:
            # Skip low quality archetypes if quality info is available
            if archetype_quality is not None:
                archetype_idx = int(archetype_label.item())
                if archetype_idx < len(archetype_quality) and not archetype_quality[archetype_idx]:
                    continue
            # Find cells that have this archetype label
            archetype_cells_mask = labels == archetype_label
            archetype_cells_indices = torch.where(archetype_cells_mask)[0]

            if len(archetype_cells_indices) == 0:
                continue

            # Get proportions for cells belonging to this archetype
            archetype_proportions = proportions[archetype_cells_indices]

            # Use the balanced cell count for all archetypes
            cells_per_this_archetype = min(balanced_cell_count, len(archetype_cells_indices))

            # Select top cells_per_this_archetype most extreme cells from this archetype
            if len(archetype_cells_indices) >= cells_per_this_archetype:
                # Get indices of top cells_per_this_archetype cells
                _, top_indices = torch.topk(archetype_proportions, cells_per_this_archetype)
                selected_cell_indices = archetype_cells_indices[top_indices]
                threshold = torch.min(archetype_proportions[top_indices]).item()
            else:
                # If we have fewer cells than desired, take all of them
                selected_cell_indices = archetype_cells_indices
                threshold = (
                    torch.min(archetype_proportions).item() if len(archetype_proportions) > 0 else 0
                )

            extreme_mask[selected_cell_indices] = True
            thresholds.append(threshold)

        # Average threshold across archetypes
        avg_threshold = np.mean(thresholds) if thresholds else 0

        if to_print:
            logger_.info(
                f"\nBalanced extreme selection: {torch.sum(extreme_mask)} out of {len(extreme_mask)} cells are extreme archetypes ({torch.sum(extreme_mask)/len(extreme_mask):.1%})"
            )
            logger_.info(
                f"Method: top {100-percentile}% per archetype, min 10 cells, balanced to {balanced_cell_count} cells per archetype"
            )
            logger_.info(f"Average threshold: {avg_threshold:.3f}")

            # Print distribution across archetypes
            for archetype_label in unique_labels:
                count = torch.sum((labels == archetype_label) & extreme_mask)
                total_labeled = torch.sum(labels == archetype_label)
                percentage = (count / total_labeled * 100) if total_labeled > 0 else 0
                logger_.info(
                    f"Archetype {archetype_label.item()}: {count} extreme cells from {total_labeled} labeled ({percentage:.1f}%)"
                )

    return extreme_mask, dominant_dims, avg_threshold


def reorder_rows_to_maximize_diagonal(matrix):
    """
    Reorders rows of a matrix to maximize diagonal dominance by placing the highest values
    in the closest positions to the diagonal.

    Parameters:
    -----------
    matrix : np.ndarray
        An m x n matrix.

    Returns:
    --------
    reordered_matrix : np.ndarray
        The input matrix with reordered rows.
    row_order : list
        The indices of the rows in their new order.
    """
    # Track available rows
    original = None
    if isinstance(matrix, pd.DataFrame):
        original = copy.deepcopy(matrix)
        matrix = matrix.values
    available_rows = list(range(matrix.shape[0]))
    row_order = []

    # Reorder rows iteratively
    for col in range(matrix.shape[1]):
        if not available_rows:
            break

        # Find the row with the maximum value for the current column
        best_row = max(available_rows, key=lambda r: matrix[r, col])
        row_order.append(best_row)
        available_rows.remove(best_row)

    # Handle leftover rows if there are more rows than columns
    row_order += available_rows

    # Reorder the matrix
    reordered_matrix = matrix[row_order]
    if original is not None:
        reordered_matrix = pd.DataFrame(
            reordered_matrix, index=original.index, columns=original.columns
        )
    return reordered_matrix, row_order


# plot_archetypes_matching moved to visualization.py


def match_rows(matrix1, matrix2, metric="correlation"):
    """Helper function to match rows between two matrices."""
    if metric == "correlation":
        # Compute correlation matrix
        corr_matrix = np.corrcoef(matrix1, matrix2)[: matrix1.shape[0], matrix1.shape[0] :]
        # Convert correlation to distance (1 - correlation)
        dist_matrix = 1 - corr_matrix
    else:
        # Use scipy's cdist for other metrics
        dist_matrix = cdist(matrix1, matrix2, metric=metric)

    # Use Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    total_cost = dist_matrix[row_ind, col_ind].sum()
    return row_ind, col_ind, total_cost, dist_matrix


def find_best_pair_by_row_matching(
    archetype_proportion_list_1,
    archetype_proportion_list_2,
    metric="correlation",
):
    """
    Find the best index in the list by matching rows using linear assignment.

    Parameters:
    -----------
    archetype_proportion_list : list of tuples
        List where each tuple contains (rna, protein) matrices.
    metric : str, optional
        Distance metric to use ('euclidean' or 'cosine').

    Returns:
    --------
    best_num_or_archetypes_index : int
        Index of the best matching pair in the list.
    best_total_cost : float
        Total cost of the best matching.
    best_1_archetype_order : np.ndarray
        Indices of dataset 1 rows.
    best_2_archetype_order : np.ndarray
        Indices of dataset 2 rows matched to dataset 1 rows.
    """

    best_num_or_archetypes_index = None
    best_total_cost = float("inf")
    best_1_archetype_order = None
    best_2_archetype_order = None

    for i, (archetype_proportion_1, archetype_proportion_2) in enumerate(
        zip(archetype_proportion_list_1, archetype_proportion_list_2)
    ):
        archetype_proportion_1 = (
            archetype_proportion_1.values
            if hasattr(archetype_proportion_1, "values")
            else archetype_proportion_1
        )
        archetype_proportion_2 = (
            archetype_proportion_2.values
            if hasattr(archetype_proportion_2, "values")
            else archetype_proportion_2
        )

        assert (
            archetype_proportion_1.shape[1] == archetype_proportion_2.shape[1]
        ), f"Mismatch in dimensions at index {i}."

        row_ind, col_ind, total_cost, _ = match_rows(
            archetype_proportion_1, archetype_proportion_2, metric=metric
        )
        print(f"Pair {i}: Total matching cost = {total_cost}")

        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_num_or_archetypes_index = i
            best_1_archetype_order = row_ind
            best_2_archetype_order = col_ind

    return (
        best_num_or_archetypes_index,
        best_total_cost,
        best_1_archetype_order,
        best_2_archetype_order,
    )


# archetype_vs_latent_distances_plot moved to visualization.py


def validate_extreme_archetypes_matching(
    adata_rna, adata_prot, plot_flag=False, logger_=None, to_print=True
):
    """
        Validate the extreme archetype matching between RNA and protein data.
        making sure that the extreme archetypes are preferred for over non-extreme archetypes
        as they are more likely to be correct matches of cell types across modalities.
        Here we show that when focusing on extereme archetypes only we get better matching accuracy of cell types across modalities
    this means that the extreme archetypes are more reliable and can be used as a general frame for cross modality matching
    # Show case argument for selecting extreme archetype as s general frame for corss modality matching
    """
    # Calculate pairwise distances between RNA and protein cells in chunks to avoid memory issues
    rna_vec = adata_rna.obsm["archetype_vec"]
    prot_vec = adata_prot.obsm["archetype_vec"]

    # Use chunked computation to avoid creating huge distance matrix
    # Dynamically adjust chunk size based on dataset size to avoid memory issues
    n_rna = rna_vec.shape[0]
    n_prot = prot_vec.shape[0]
    # For large datasets, use smaller chunks to avoid memory issues
    # Estimate memory: chunk_size * n_prot * 8 bytes (float64)
    # Target: keep chunks under 500MB
    max_chunk_memory_gb = 0.5
    max_elements_per_chunk = int(max_chunk_memory_gb * 1e9 / (n_prot * 8))
    chunk_size = min(1000, max(100, max_elements_per_chunk))

    closest_prot_indices = np.zeros(n_rna, dtype=int)

    for i in range(0, n_rna, chunk_size):
        end_idx = min(i + chunk_size, n_rna)
        chunk_distances = cdist(
            rna_vec[i:end_idx],
            prot_vec,
            metric="cosine",
        )
        closest_prot_indices[i:end_idx] = np.argmin(chunk_distances, axis=1)
        del chunk_distances  # Free memory immediately

    # Create match_correct column by comparing cell types
    adata_rna.obs["match_correct"] = (
        adata_rna.obs["cell_types"].values
        == adata_prot.obs["cell_types"].values[closest_prot_indices]
    )

    # Do the same for protein data (compute in chunks)
    # Adjust chunk size for protein->RNA direction (different dimensions)
    max_elements_per_chunk_prot = int(max_chunk_memory_gb * 1e9 / (n_rna * 8))
    chunk_size_prot = min(1000, max(100, max_elements_per_chunk_prot))

    closest_rna_indices = np.zeros(n_prot, dtype=int)

    for i in range(0, n_prot, chunk_size_prot):
        end_idx = min(i + chunk_size_prot, n_prot)
        chunk_distances = cdist(
            prot_vec[i:end_idx],
            rna_vec,
            metric="cosine",
        )
        closest_rna_indices[i:end_idx] = np.argmin(chunk_distances, axis=1)
        del chunk_distances  # Free memory immediately

    adata_prot.obs["match_correct"] = (
        adata_prot.obs["cell_types"].values
        == adata_rna.obs["cell_types"].values[closest_rna_indices]
    )

    # Get archetype vectors
    rna_archetypes = adata_rna.obsm["archetype_vec"]
    prot_archetypes = adata_prot.obsm["archetype_vec"]

    # Identify extreme archetypes for both modalities

    (
        rna_extreme_mask,
        rna_dominant_dim,
        rna_threshold,
    ) = identify_extreme_archetypes_percentile(rna_archetypes, logger_, 95, to_print=to_print)
    prot_extreme_mask, prot_dominant_dim, prot_threshold = identify_extreme_archetypes_percentile(
        prot_archetypes, logger_, 95, to_print=to_print
    )

    # Add the extreme archetype flag to the observation dataframes
    adata_rna.obs["is_extreme_archetype"] = rna_extreme_mask
    adata_prot.obs["is_extreme_archetype"] = prot_extreme_mask

    # Also add the proportion in dominant dimension as a quantitative measure
    rna_proportions = np.max(rna_archetypes, axis=1) / np.sum(rna_archetypes, axis=1)
    prot_proportions = np.max(prot_archetypes, axis=1) / np.sum(prot_archetypes, axis=1)

    adata_rna.obs["archetype_max_proportion"] = rna_proportions
    adata_prot.obs["archetype_max_proportion"] = prot_proportions

    # Print summary
    print("Added extreme archetype flags to both modalities")
    print(
        f"\nRNA data: {rna_extreme_mask.sum()} out of {len(rna_extreme_mask)} cells are extreme archetypes ({rna_extreme_mask.mean():.2%})"
    )
    print(f"\nRNA threshold value: {rna_threshold:.3f}")
    print(
        f"\nProtein data: {prot_extreme_mask.sum()} out of {len(prot_extreme_mask)} cells are extreme archetypes ({prot_extreme_mask.mean():.2%})"
    )
    print(f"\nProtein threshold value: {prot_threshold:.3f}")

    # Show the first few rows of the updated observation dataframes
    # Also check if extreme archetypes have better matching accuracy
    rna_extreme_accuracy = adata_rna.obs.loc[
        adata_rna.obs["is_extreme_archetype"], "match_correct"
    ].mean()
    rna_non_extreme_accuracy = adata_rna.obs.loc[
        ~adata_rna.obs["is_extreme_archetype"], "match_correct"
    ].mean()

    prot_extreme_accuracy = adata_prot.obs.loc[
        adata_prot.obs["is_extreme_archetype"], "match_correct"
    ].mean()
    prot_non_extreme_accuracy = adata_prot.obs.loc[
        ~adata_prot.obs["is_extreme_archetype"], "match_correct"
    ].mean()

    print("\nMatching accuracy comparison:")
    print(f"RNA extreme archetypes: {rna_extreme_accuracy:.2%}")
    print(f"RNA non-extreme archetypes: {rna_non_extreme_accuracy:.2%}")
    print(f"Protein extreme archetypes: {prot_extreme_accuracy:.2%}")
    print(f"Protein non-extreme archetypes: {prot_non_extreme_accuracy:.2%}")
