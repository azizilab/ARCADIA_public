"""COVET integration functions for preparing data and integrating with scVI."""

from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Literal, Optional

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from kneed import KneeLocator
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from arcadia.covet.core import compute_covet
from arcadia.covet.utils import (
    compute_mi_batch,
    extract_lower_triangle,
    generate_covet_feature_names,
)


def prepare_data(
    adata: AnnData,
    covet_k: int = 8,
    covet_g: int = 64,
    layer: str = None,
    covet_selection_method: Literal["high_variability", "low_redundancy"] = "high_variability",
    mutual_info_threshold: Optional[float] = None,
    mi_sample_size: Optional[int] = None,
    fast_mi: bool = True,
    apply_pca: bool = False,
    pca_variance_threshold: float = 0.95,
):
    """
    Prepare data by computing COVET and neighbor mean features and combining with original features.
    Adapts to existing preprocessing while following best practices for COVET computation.

    This function:
    1. Detects existing normalization in the data
    2. Applies appropriate preprocessing for COVET if needed
    3. Computes COVET on properly preprocessed data
    4. Selects COVET features using high variability (knee detection) OR low redundancy (mutual info with knee detection)
    5. Optionally applies PCA to selected COVET + neighbor mean features
    6. Combines features and standardizes for downstream analysis

    Args:
        adata: AnnData object with protein data (may be normalized or raw)
        covet_k: Number of nearest neighbors for COVET
        covet_g: Number of HVGs for COVET
        layer: Layer to use for COVET computation (default: None, uses X)
        covet_selection_method: Method to select COVET features - "high_variability" (knee detection) or "low_redundancy" (mutual info filtering) (default: "high_variability")
        mutual_info_threshold: Threshold for mutual information filtering (default: 0.3, only used if method="low_redundancy" and knee detection fails)
        mi_sample_size: Sample size for MI computation to speed up calculation (default: None, uses all cells)
        fast_mi: Use faster correlation-based approximation instead of full MI computation (default: True)
        apply_pca: Whether to apply PCA to selected COVET + neighbor features (default: False)
        pca_variance_threshold: Fraction of variance to retain in PCA (default: 0.95)

    Returns:
        adata_combined: AnnData object with combined features ready for scVI
    """
    print("Preparing data for COVET computation...")

    # Work on a copy to avoid modifying original data
    adata_work = adata.copy()
    original_obsm = adata_work.obsm.copy()
    original_obsp = adata_work.obsp.copy()
    adata_work.layers.copy()
    original_obs = adata_work.obs.copy()
    # Check for NaN/Inf in input data
    if issparse(adata_work.X):
        has_nan = np.any(~np.isfinite(adata_work.X.data))
    else:
        has_nan = np.any(~np.isfinite(adata_work.X))

    if has_nan:
        print("Warning: NaN/Inf values found in input data. Replacing with zeros.")
        if issparse(adata_work.X):
            adata_work.X.data = np.nan_to_num(adata_work.X.data)
        else:
            adata_work.X = np.nan_to_num(adata_work.X)

    # Compute COVET on the appropriately preprocessed data
    print("Computing COVET features...")
    COVET, COVET_SQRT, CovGenes = compute_covet(adata_work, k=covet_k, g=covet_g)
    COVET = COVET_SQRT  # recommended by COVET authors
    # Extract lower triangle from each COVET matrix and flatten
    n_cells = COVET.shape[0]
    n_genes = COVET.shape[1]
    n_tril_elements = (n_genes * (n_genes - 1)) // 2  # strict lower triangle only

    # Preallocate array for all flattened lower triangles
    X_covet_tril = np.zeros((n_cells, n_tril_elements))
    X_covet_diag = np.zeros((n_cells, n_genes))
    # Extract and flatten lower triangle for each cell
    for i in range(n_cells):
        X_covet_tril[i] = extract_lower_triangle(COVET[i])
        X_covet_diag[i] = np.diag(COVET[i])
    # Select COVET features based on the chosen method
    if covet_selection_method == "high_variability":
        print("Using high variability (knee detection) for COVET feature selection...")

        # Calculate variances of COVET features
        covet_tril_variances = X_covet_tril.var(0)

        # Sort variances in descending order for knee detection
        sorted_covet_tril_variances = np.sort(covet_tril_variances)[::-1]
        sorted_covet_tril_indices = np.argsort(covet_tril_variances)[::-1]
        sorted_variances = sorted_covet_tril_variances
        # Find knee point in sorted variances
        kl = KneeLocator(
            range(len(sorted_variances)),
            sorted_variances,
            curve="convex",
            direction="decreasing",
            S=20,
        )
        knee_idx = kl.elbow

        plt.figure(figsize=(12, 8))
        kl.plot_knee()
        plt.show()

        print(f"Selected {knee_idx} features out of {len(sorted_variances)} total features")
        print(f"Variance range: {sorted_variances[0]:.4f} to {sorted_variances[-1]:.4f}")
        cumulative_var = np.cumsum(sorted_variances) / np.sum(sorted_variances)
        print(f"Cumulative variance explained: {100*cumulative_var[knee_idx]:.1f}%")

        # Select indices for highly variable COVET features based on knee detection
        selected_covet_indices = sorted_covet_tril_indices[:knee_idx]
        initial_selection_count = knee_idx

    elif covet_selection_method == "low_redundancy":
        print(
            f"Using low redundancy (mutual information filtering) for COVET feature selection (threshold: {mutual_info_threshold})..."
        )

        # Get protein features for MI calculation
        X_protein_for_mi = adata_work.X.copy()
        if issparse(X_protein_for_mi):
            X_protein_for_mi = X_protein_for_mi.toarray()

        # Get neighbor means (computed later, but we need to compute them here for MI)
        if "spatial_location" not in adata.obsm:
            raise ValueError("spatial_location must exist in adata.obsm to compute CN means.")
        adata_work.obsm["spatial_location"] = adata.obsm["spatial_location"].copy()
        sc.pp.neighbors(adata_work, use_rep="spatial_location", n_neighbors=covet_k)
        connectivities = adata_work.obsp["connectivities"].copy()
        connectivities[connectivities > 0] = 1
        neighbor_sums = connectivities.dot(X_protein_for_mi)
        neighbor_means_for_mi = np.asarray(neighbor_sums / connectivities.sum(1))
        neighbor_means_for_mi = np.nan_to_num(neighbor_means_for_mi)
        neighbor_means_for_mi = StandardScaler().fit_transform(neighbor_means_for_mi)

        # Calculate mutual information for ALL COVET features with original protein features only

        # Optional sampling for very large datasets
        X_covet_mi = X_covet_tril
        X_protein_mi = X_protein_for_mi
        sample_indices = None

        if mi_sample_size is not None and mi_sample_size < X_protein_for_mi.shape[0]:
            print(
                f"Sampling {mi_sample_size} cells out of {X_protein_for_mi.shape[0]} for faster MI computation..."
            )
            np.random.seed(42)
            sample_indices = np.random.choice(
                X_protein_for_mi.shape[0], mi_sample_size, replace=False
            )
            X_covet_mi = X_covet_tril[sample_indices]
            X_protein_mi = X_protein_for_mi[sample_indices]

        computation_type = (
            "correlation-based approximation" if fast_mi else "full mutual information"
        )
        print(
            f"Computing {computation_type} for {n_tril_elements} COVET features with {X_protein_mi.shape[1]} protein features..."
        )
        print(f"Using {X_covet_mi.shape[0]} cells for computation")
        print("Using parallel batch processing for faster computation...")

        # Use multiprocessing with batching for better performance
        n_processes = min(cpu_count(), 8)  # Limit to 8 processes to avoid memory issues
        batch_size = max(50, n_tril_elements // (n_processes * 4))  # Adaptive batch size
        print(f"Using {n_processes} processes with batch size {batch_size}")

        # Create batches of COVET indices
        covet_batches = [
            list(range(i, min(i + batch_size, n_tril_elements)))
            for i in range(0, n_tril_elements, batch_size)
        ]

        # Create partial function with fixed data
        compute_mi_batch_partial = partial(
            compute_mi_batch,
            X_covet_data=X_covet_mi,
            X_protein_data=X_protein_mi,
            use_fast_mi=fast_mi,
        )

        # Parallel computation
        mi_scores = []
        with Pool(processes=n_processes) as pool:
            batch_results = list(
                tqdm(
                    pool.imap(compute_mi_batch_partial, covet_batches),
                    total=len(covet_batches),
                    desc="Computing MI scores (batched)",
                )
            )

            # Flatten results
            for batch_result in batch_results:
                mi_scores.extend(batch_result)

        # Use knee detection to select COVET features with least MI (lowest redundancy)
        print(
            "Using knee detection on mutual information scores to select least redundant features..."
        )

        # Sort MI scores in ascending order (lowest MI first)
        sorted_mi_scores = np.sort(mi_scores)
        sorted_indices = np.argsort(mi_scores)

        # Find knee point in sorted MI scores (ascending order)
        # For this type of data, we want to find where the flat region ends
        knee_idx = None

        # Method 1: Find where the curve starts rising significantly
        # Look for the transition from flat to steep
        if len(sorted_mi_scores) > 50:
            # Calculate second derivative to find inflection point
            first_diff = np.diff(sorted_mi_scores)
            second_diff = np.diff(first_diff)

            # Find where second derivative becomes large (acceleration in increase)
            threshold = np.percentile(second_diff, 85)  # Top 15% of acceleration (more sensitive)
            steep_points = np.where(second_diff > threshold)[0]

            if len(steep_points) > 0:
                # Take the first significant acceleration point
                knee_idx = steep_points[0] + 2  # +2 because we took diff twice
                print(f"Found knee using second derivative method at index {knee_idx}")

        # Method 2: Use traditional knee detection as backup
        if knee_idx is None:
            try:
                kl = KneeLocator(
                    range(len(sorted_mi_scores)),
                    sorted_mi_scores,
                    curve="convex",  # Use convex for this type of curve
                    direction="increasing",
                    S=0.1,  # Very sensitive
                )
                knee_idx = kl.elbow

                if knee_idx is not None:
                    print(f"Found knee using KneeLocator: {knee_idx}")

            except Exception as e:
                print(f"Knee detection error: {e}")
                knee_idx = None

        if knee_idx is None or knee_idx == 0:
            # Fallback: use threshold-based selection or gradient-based approach
            if mutual_info_threshold is not None:
                knee_idx = np.sum(np.array(mi_scores) < mutual_info_threshold)
                print(
                    f"Knee detection failed, using threshold-based selection: {knee_idx} features"
                )
            else:
                # Method 3: Robust gradient-based approach for this curve type
                if len(sorted_mi_scores) > 50:
                    # Find where gradient becomes much larger than the "flat" region
                    gradients = np.diff(sorted_mi_scores)

                    # Use rolling window to smooth gradients
                    window_size = min(20, len(gradients) // 10)
                    if window_size > 1:
                        smoothed_gradients = np.convolve(
                            gradients, np.ones(window_size) / window_size, mode="valid"
                        )
                        # Find where gradient exceeds 90th percentile (more sensitive)
                        grad_threshold = np.percentile(smoothed_gradients, 90)
                        steep_starts = np.where(smoothed_gradients > grad_threshold)[0]

                        if len(steep_starts) > 0:
                            knee_idx = (
                                steep_starts[0] + window_size // 2
                            )  # Adjust for window offset
                        else:
                            # Use a more conservative approach - 80th percentile of all MI scores
                            knee_idx = int(0.8 * len(mi_scores))
                    else:
                        knee_idx = int(0.8 * len(mi_scores))
                else:
                    knee_idx = int(0.8 * len(mi_scores))
                print(f"Knee detection failed, using gradient-based selection: {knee_idx} features")
        else:
            print(
                f"Knee detection successful: selected {knee_idx} features out of {len(mi_scores)} total features"
            )

        # Select indices for COVET features with lowest MI (least redundant)
        selected_covet_indices = sorted_indices[:knee_idx]

        print(
            f"Low redundancy selection: kept {len(selected_covet_indices)} out of {n_tril_elements} total COVET features"
        )
        print(f"MI scores range: {min(mi_scores):.4f} to {max(mi_scores):.4f}")
        if knee_idx > 0:
            print(
                f"Selected features have MI scores: {min(mi_scores):.4f} to {sorted_mi_scores[knee_idx-1]:.4f}"
            )
        else:
            print("No features selected!")
        initial_selection_count = n_tril_elements  # For reporting purposes

    else:
        raise ValueError(
            f"Invalid covet_selection_method: {covet_selection_method}. Must be 'high_variability' or 'low_redundancy'"
        )

    print(
        f"Selected {len(selected_covet_indices)} COVET features using {covet_selection_method} method"
    )

    # Plot selected features analysis if high variability selection was used
    if covet_selection_method == "high_variability" and len(selected_covet_indices) > 0:
        selected_variances = sorted_variances[:knee_idx]  # Variances of selected features

        # Verify we have enough features for analysis
        if len(selected_variances) < 6:
            print(f"Warning: Only {len(selected_variances)} selected features available")
            n_top = min(3, len(selected_variances))
            n_bottom = min(3, len(selected_variances) - n_top)
        else:
            n_top = 3
            n_bottom = 3

        top_indices = np.argsort(selected_variances)[-n_top:]
        bottom_indices = np.argsort(selected_variances)[:n_bottom]

        print(f"Top {n_top} variance indices in selected features: {top_indices}")
        print(f"Bottom {n_bottom} variance indices in selected features: {bottom_indices}")

        # Sort by the highest variance column and use its index to resort all others
        highest_var_idx = top_indices[-1]  # Index of highest variance column
        sort_order = np.argsort(X_covet_tril[:, highest_var_idx])

        print(f"Sorting selected features by values in column {highest_var_idx} (highest variance)")

        plt.figure(figsize=(12, 8))

        # Plot top variance columns (sorted by highest variance column)
        plt.subplot(2, 1, 1)
        for i, idx in enumerate(top_indices):
            if idx < len(selected_covet_indices):  # Check bounds
                orig_covet_idx = selected_covet_indices[idx]  # Map to original COVET feature index
                sort_order_selected = np.argsort(X_covet_tril[:, orig_covet_idx])
                plt.plot(
                    X_covet_tril[sort_order_selected, orig_covet_idx],
                    label=f"Feature {idx} (var={selected_variances[idx]:.4f})",
                )
        plt.title(f"Top {n_top} Variance COVET Features (selected)")
        plt.xlabel("Cells (sorted by feature value)")
        plt.ylabel("Feature Values")
        plt.legend()

        # Plot bottom variance columns
        plt.subplot(2, 1, 2)
        for i, idx in enumerate(bottom_indices):
            if idx < len(selected_covet_indices):  # Check bounds
                orig_covet_idx = selected_covet_indices[idx]  # Map to original COVET feature index
                sort_order_selected = np.argsort(X_covet_tril[:, orig_covet_idx])
                plt.plot(
                    X_covet_tril[sort_order_selected, orig_covet_idx],
                    label=f"Feature {idx} (var={selected_variances[idx]:.4f})",
                )
        plt.title(f"Bottom {n_bottom} Variance COVET Features (selected)")
        plt.xlabel("Cells (sorted by feature value)")
        plt.ylabel("Feature Values")
        plt.legend()

        plt.tight_layout()
        plt.show()

    elif covet_selection_method == "low_redundancy":
        print("Skipping variance-based plotting for low redundancy selection method")
        # plot the mi scores with knee detection
        plt.figure(figsize=(12, 8))

        # Plot sorted MI scores with knee point
        plt.subplot(2, 1, 1)
        plt.plot(sorted_mi_scores, "b-", label="Sorted MI Scores")
        if knee_idx is not None:
            plt.axvline(x=knee_idx, color="r", linestyle="--", label=f"Knee Point (n={knee_idx})")
        plt.title("Mutual Information Scores (Sorted)")
        plt.xlabel("COVET Feature Index (Sorted by MI)")
        plt.ylabel("Mutual Information Score")
        plt.legend()

        # Plot gradient analysis for knee detection diagnostics
        plt.subplot(2, 1, 2)
        if len(sorted_mi_scores) > 10:
            # Show first and second derivatives to explain knee detection
            first_diff = np.diff(sorted_mi_scores)
            second_diff = np.diff(first_diff)

            # Plot first derivative (gradient)
            plt.plot(
                range(1, len(sorted_mi_scores)),
                first_diff,
                "g-",
                label="First Derivative (Gradient)",
                alpha=0.7,
            )

            # Plot second derivative (acceleration)
            if len(second_diff) > 0:
                plt.plot(
                    range(2, len(sorted_mi_scores)),
                    second_diff,
                    "r-",
                    label="Second Derivative (Acceleration)",
                    alpha=0.7,
                )

                # Show the threshold used for knee detection
                if len(second_diff) > 10:
                    threshold = np.percentile(second_diff, 85)
                    plt.axhline(
                        y=threshold,
                        color="orange",
                        linestyle=":",
                        label=f"85th percentile threshold",
                    )

            # Mark the selected knee point
            if knee_idx is not None and knee_idx > 0:
                plt.axvline(
                    x=knee_idx,
                    color="r",
                    linestyle="--",
                    linewidth=2,
                    label=f"Selected Knee (n={knee_idx})",
                )

            plt.title("Gradient Analysis for Knee Detection")
            plt.xlabel("COVET Feature Index")
            plt.ylabel("Derivative Values")
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(
                0.5,
                0.5,
                "Not enough data for gradient analysis",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )

        plt.tight_layout()
        plt.show()

    # Handle any NaN/Inf that might have appeared during computation
    X_covet_tril = np.nan_to_num(X_covet_tril)

    # Step 4: Get processed protein features for combination - USE ORIGINAL DATA, NO FILTERING
    print("Step 4: Preparing protein features for combination...")
    # Use original adata with same preprocessing as applied to adata_work, but NO HVG filtering
    X_protein = adata.X.copy()  # All 110 original proteins
    if issparse(X_protein):
        X_protein = X_protein.toarray()

    print(f"X_protein shape (original, no filtering): {X_protein.shape}")
    print(
        f"Protein features: shape={X_protein.shape}, range=[{X_protein.min():.3f}, {X_protein.max():.3f}]"
    )

    # Compute CN mean features (if spatial data is available)
    if "spatial_location" not in adata.obsm:
        raise ValueError("spatial_location must exist in adata.obsm to compute CN means.")

    # Compute neighbor means (reuse if already computed for MI filtering)
    if covet_selection_method == "low_redundancy":
        # Reuse the neighbor means computed for MI filtering
        neighbor_means = neighbor_means_for_mi
    else:
        # Use original adata for neighbor computation to get all 110 proteins
        adata_original_copy = adata.copy()

        # Compute knn graph based on spatial location using original data
        sc.pp.neighbors(adata_original_copy, use_rep="spatial_location", n_neighbors=covet_k)
        connectivities = adata_original_copy.obsp["connectivities"].copy()
        connectivities[connectivities > 0] = 1  # binarize
        # Use original protein data for neighbor computation (all 110 proteins)
        X_protein_original = adata_original_copy.X.copy()
        if issparse(X_protein_original):
            X_protein_original = X_protein_original.toarray()
        neighbor_sums = connectivities.dot(X_protein_original)
        neighbor_means = np.asarray(
            neighbor_sums / connectivities.sum(1)
        )  # divide by number of neighbors (k)
        neighbor_means = np.nan_to_num(neighbor_means)
        neighbor_variances = X_covet_diag

    n_neighbor_means = neighbor_means.shape[1]

    # Extract selected COVET features
    if len(selected_covet_indices) > 0:
        X_covet_selected = X_covet_tril[:, selected_covet_indices]
    else:
        X_covet_selected = np.empty((X_covet_tril.shape[0], 0))
        print("Warning: No COVET features selected after filtering!")

    # Apply PCA if requested
    pca_applied = False
    pca_components = None
    if apply_pca and len(selected_covet_indices) > 0:
        print(
            f"Applying PCA to {X_covet_selected.shape[1]} COVET features + {n_neighbor_means} neighbor features..."
        )

        # Combine COVET and neighbor features for PCA
        features_for_pca = np.hstack([neighbor_means, neighbor_variances, X_covet_selected])

        # Apply PCA
        pca = PCA(random_state=42)
        pca_features = pca.fit_transform(features_for_pca)

        # Find number of components for desired variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= pca_variance_threshold) + 1

        print(
            f"PCA: Using {n_components} components to explain {cumulative_variance[n_components-1]:.3f} of variance"
        )
        print(f"PCA: Reduced from {features_for_pca.shape[1]} to {n_components} features")

        # Use only the required number of components
        pca_features_selected = pca_features[:, :n_components]
        pca_applied = True
        pca_components = n_components

        # Create feature names for PCA components
        pca_feature_names = [f"PCA_{i+1}" for i in range(n_components)]

        # Final combined features: protein + PCA features
        combined_X = np.hstack([X_protein, pca_features_selected])
        final_feature_names = list(adata.var_names) + pca_feature_names
        final_feature_types = ["protein"] * len(adata.var_names) + ["pca"] * n_components

    else:
        # No PCA: combine protein, neighbor means, neighbor variances, and selected COVET features
        print(
            f"Matrix shapes - X_protein: {X_protein.shape}, neighbor_means: {neighbor_means.shape}, neighbor_variances: {neighbor_variances.shape}, X_covet_selected: {X_covet_selected.shape}"
        )
        combined_X = np.hstack([X_protein, neighbor_means, neighbor_variances, X_covet_selected])
        print(f"Combined matrix shape: {combined_X.shape}")

        # Create feature names
        var_names = list(adata.var_names)  # All 110 original proteins
        cov_gene_names = list(CovGenes)  # Filtered proteins used in COVET (108)
        print(f"Original protein var_names length: {len(var_names)}")
        print(f"CovGenes (filtered proteins) length: {len(cov_gene_names)}")
        print(f"Unique protein var_names length: {len(set(var_names))}")
        if len(var_names) != len(set(var_names)):
            duplicates = [name for name in set(var_names) if var_names.count(name) > 1]
            print(f"Duplicate protein names found: {duplicates}")

        # Use all 110 original proteins for neighbor features (not filtered)
        neighbor_names = [f"CN_mean_{i}" for i in var_names]
        neighbor_variances_names = [f"CN_variance_{i}" for i in var_names]

        if len(selected_covet_indices) > 0:
            covet_names = generate_covet_feature_names(CovGenes)
            selected_covet_names = [covet_names[i] for i in selected_covet_indices]
            print(f"Selected COVET names length: {len(selected_covet_names)}")
            print(f"Unique selected COVET names length: {len(set(selected_covet_names))}")
            if len(selected_covet_names) != len(set(selected_covet_names)):
                covet_duplicates = [
                    name
                    for name in set(selected_covet_names)
                    if selected_covet_names.count(name) > 1
                ]
                print(f"Duplicate COVET names found: {covet_duplicates}")
        else:
            selected_covet_names = []

        final_feature_names = (
            var_names + neighbor_names + neighbor_variances_names + selected_covet_names
        )
        final_feature_types = (
            ["protein"] * len(var_names)  # 110 original proteins (NO FILTERING)
            + ["neighbor_mean"] * len(neighbor_names)  # 110 neighbor means
            + ["neighbor_variance"] * len(neighbor_variances_names)  # 110 neighbor variances
            + ["CN"] * len(selected_covet_names)  # 1240 COVET features
        )

    # Create AnnData object with final combined features
    print(f"Final combined_X shape: {combined_X.shape}")
    print(f"Expected feature count: {len(final_feature_names)}")
    adata_combined = ad.AnnData(combined_X)

    # Copy metadata
    adata_combined.obs = adata.obs.copy()

    # Store feature information
    n_protein = X_protein.shape[1]
    n_covet_total = X_covet_tril.shape[1]
    n_covet_selected = len(selected_covet_indices)

    adata_combined.uns["n_protein_features"] = n_protein
    adata_combined.uns["n_neighbor_mean_features"] = n_neighbor_means if not pca_applied else 0
    adata_combined.uns["n_covet_features"] = n_covet_total
    adata_combined.uns["n_covet_features_selected"] = n_covet_selected
    adata_combined.uns["covet_selection_indices"] = selected_covet_indices
    adata_combined.uns["CovGenes"] = CovGenes
    adata_combined.uns["pca_applied"] = pca_applied
    adata_combined.uns["pca_components"] = pca_components if pca_applied else 0
    adata_combined.uns["covet_selection_method"] = covet_selection_method
    adata_combined.uns["mutual_info_threshold"] = (
        mutual_info_threshold if covet_selection_method == "mutual_info" else None
    )

    # Set variable names and types
    print(f"final_feature_names length: {len(final_feature_names)}")
    print(f"Unique final_feature_names length: {len(set(final_feature_names))}")
    if len(final_feature_names) != len(set(final_feature_names)):
        duplicates = [
            name for name in set(final_feature_names) if final_feature_names.count(name) > 1
        ]
        print(f"Duplicate feature names found: {duplicates}")

    adata_combined.var_names = final_feature_names
    adata_combined.var_names_make_unique()
    adata_combined.var["feature_type"] = final_feature_types

    # Set highly variable genes based on feature types
    adata_combined.var["highly_variable"] = False

    # Set protein features as highly variable (always first n_protein features)
    adata_combined.var.iloc[:n_protein, adata_combined.var.columns.get_loc("highly_variable")] = (
        True
    )

    if pca_applied:
        # If PCA was applied, all PCA components are considered highly variable
        pca_start_idx = n_protein
        adata_combined.var.iloc[
            pca_start_idx : pca_start_idx + pca_components,
            adata_combined.var.columns.get_loc("highly_variable"),
        ] = True
    else:
        # Without PCA: neighbor means and selected COVET features are highly variable
        # Set neighbor mean features as highly variable
        adata_combined.var.iloc[
            n_protein : n_protein + n_neighbor_means,
            adata_combined.var.columns.get_loc("highly_variable"),
        ] = True

        # Set selected COVET features as highly variable
        covet_start_idx = n_protein + n_neighbor_means
        for i in range(len(selected_covet_indices)):
            adata_combined.var.iloc[
                covet_start_idx + i, adata_combined.var.columns.get_loc("highly_variable")
            ] = True

    # Print highly variable gene summary
    hvg_by_type = adata_combined.var.groupby("feature_type")["highly_variable"].sum()
    print(f"Highly variable genes by type: {dict(hvg_by_type)}")
    print(
        f"Total highly variable: {adata_combined.var['highly_variable'].sum()}/{len(adata_combined.var)}"
    )

    # Store original COVET matrices for reference
    adata_combined.uns["COVET_matrices"] = COVET

    # Store preprocessing metadata following best practices
    adata_combined.uns["prepare_data_metadata"] = {
        "covet_computed_on": "preprocessed_data",
        "preprocessing_pipeline": "adaptive_to_existing_normalization",
        "follows_covet_best_practices": True,
        "covet_parameters": {"k": covet_k, "g": covet_g},
        "feature_counts": {
            "protein": len(adata.var_names),
            "neighbor_means": neighbor_means.shape[1] if not pca_applied else 0,
            "covet_total": n_covet_total,
            "covet_selected_initial": initial_selection_count,
            "covet_selected_final": n_covet_selected,
            "pca_components": pca_components if pca_applied else 0,
        },
        "covet_selection": {
            "method": covet_selection_method,
            "mutual_info_threshold": (
                mutual_info_threshold if covet_selection_method == "mutual_info" else None
            ),
        },
        "filtering_applied": {
            "pca_applied": pca_applied,
            "pca_variance_threshold": pca_variance_threshold if pca_applied else None,
        },
    }

    print(f"Original features shape: {X_protein.shape}")
    print(f"COVET lower triangle shape (total): {X_covet_tril.shape}")
    print(
        f"COVET features selected using {covet_selection_method}: {n_covet_selected}/{n_covet_total}"
    )
    if pca_applied:
        print(
            f"PCA applied: {pca_components} components from {n_neighbor_means + n_covet_selected} features"
        )
    print(f"Final combined features shape: {combined_X.shape}")

    processing_steps = ["COVET", covet_selection_method]
    if pca_applied:
        processing_steps.append("PCA")
    print(f"Preprocessing pipeline: {' → '.join(processing_steps)}")

    # Final check for NaN or Inf values
    if np.isnan(adata_combined.X).any() or np.isinf(adata_combined.X).any():
        print("WARNING: NaN or Inf values detected in data. Replacing with zeros.")
        adata_combined.X = np.nan_to_num(adata_combined.X)

    # Ensure data is non-negative for scVI
    if np.any(adata_combined.X < 0):
        print("Ensuring final data is non-negative for scVI")
        adata_combined.X = adata_combined.X - adata_combined.X.min() + 1e-6

    # Final data statistics
    print(
        f"Data stats - min: {adata_combined.X.min():.4f}, max: {adata_combined.X.max():.4f}, mean: {adata_combined.X.mean():.4f}"
    )
    print(f"Zero rate: {np.mean(adata_combined.X == 0):.4f}")
    adata_combined.obsm.update(original_obsm)
    adata_combined.obsp.update(original_obsp)
    # adata_combined.layers.update(original_layers)
    adata_combined.obs = pd.concat(
        [
            adata_combined.obs,
            original_obs.loc[adata_combined.obs.index.difference(original_obs.index)],
        ],
        axis=0,
    ).combine_first(original_obs)
    return adata_combined
