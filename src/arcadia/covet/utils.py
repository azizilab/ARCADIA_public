"""COVET utility functions for feature extraction and selection."""

import numpy as np
from sklearn.feature_selection import mutual_info_regression


def extract_lower_triangle(matrix):
    """
    Extract lower triangle of matrix and flatten it.
    Ensures numerical stability by handling potential NaN or Inf values.
    """
    # Handle potential numerical issues
    matrix = np.nan_to_num(matrix)

    # Extract lower triangle
    n = matrix.shape[0]
    rows, cols = np.tril_indices(n, k=-1)
    values = matrix[rows, cols]

    # Ensure finite values
    if not np.all(np.isfinite(values)):
        print("WARNING: Non-finite values detected in covariance matrix. Replacing with zeros.")
        values = np.nan_to_num(values)

    return values


def generate_covet_feature_names(cov_genes):
    """
    Generate COVET feature names based on protein pairs from the covariance matrix.

    Args:
        cov_genes: List or array of gene/protein names used for COVET computation

    Returns:
        List of feature names corresponding to lower triangle elements
    """
    cov_genes = np.array(cov_genes)
    n_genes = len(cov_genes)
    feature_names = []

    # Generate names for lower triangle (including diagonal)
    for i in range(n_genes):
        for j in range(i + 1):  # j <= i for lower triangle
            if i == j:
                # Diagonal elements (variance)
                feature_names.append(f"{cov_genes[i]}-var")
            else:
                # Off-diagonal elements (covariance)
                feature_names.append(f"{cov_genes[j]}-{cov_genes[i]}")

    return feature_names


def compute_mi_batch(covet_indices, X_covet_data, X_protein_data, use_fast_mi=True):
    """Compute max MI/correlation for a batch of COVET features"""
    mi_scores_batch = []
    for covet_idx in covet_indices:
        covet_feature = X_covet_data[:, covet_idx]

        if use_fast_mi:
            # Ultra-fast correlation-based approximation
            correlations = np.abs(np.corrcoef(covet_feature, X_protein_data.T)[0, 1:])
            # Convert correlation to MI-like score: -0.5 * log(1 - r^2)
            correlations = np.clip(correlations, 0, 0.9999)  # Avoid log(0)
            mi_approx = -0.5 * np.log(1 - correlations**2)
            max_mi = max(mi_approx) if len(mi_approx) > 0 else 0.0
        else:
            # Full MI computation - slower but more accurate
            mi_scores = mutual_info_regression(
                covet_feature.reshape(-1, 1),
                X_protein_data,
                discrete_features=False,
                n_neighbors=10,  # Reduce neighbors for speed
                random_state=42,
            )
            max_mi = max(mi_scores) if len(mi_scores) > 0 else 0.0

        mi_scores_batch.append(max_mi)
    return mi_scores_batch
