"""Archetype cell representation utilities."""

import functools
import time

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.optimize import minimize, nnls
from sklearn.linear_model import OrthogonalMatchingPursuit
from tqdm import tqdm

from arcadia.utils.logging import setup_logger


def timeit(func):
    """
    Decorator to measure execution time of functions.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result

    return wrapper


def nnls_omp(basis_matrix, target_vector, tol=1e-4):
    """
    Non-negative least squares using Orthogonal Matching Pursuit.

    Parameters:
    -----------
    basis_matrix : np.ndarray
        Basis matrix of shape (n_archetypes, n_features)
    target_vector : np.ndarray
        Target vector of shape (n_features,)
    tol : float
        Tolerance for OMP

    Returns:
    --------
    weights : np.ndarray
        Non-negative weights of shape (n_archetypes,)
    """
    omp = OrthogonalMatchingPursuit(tol=tol, fit_intercept=False)
    omp.fit(basis_matrix.T, target_vector)
    weights = omp.coef_
    weights = np.maximum(0, weights)  # Enforce non-negativity
    return weights


def compute_weight_for_cell(x, A_T, n_archetypes, solver):
    """
    Compute archetype weights for a single cell using constrained optimization.

    Parameters:
    -----------
    x : np.ndarray
        Cell vector of shape (n_features,)
    A_T : np.ndarray
        Transposed archetype matrix of shape (n_features, n_archetypes)
    n_archetypes : int
        Number of archetypes
    solver : cvxpy solver
        Solver to use (e.g., cp.ECOS, cp.SCS)

    Returns:
    --------
    weights : np.ndarray
        Archetype weights of shape (n_archetypes,)
    """
    import cvxpy as cp

    w = cp.Variable(n_archetypes)
    objective = cp.Minimize(cp.sum_squares(A_T @ w - x))
    constraints = [w >= 0, cp.sum(w) == 1]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=solver)
    except cp.SolverError:
        problem.solve(solver=cp.SCS)
    return w.value


# archetype_vs_latent_distances_plot moved to visualization.py
# validate_extreme_archetypes_matching moved to matching.py


def identify_extreme_archetypes_percentile(
    archetype_vectors, logger_=None, percentile=95, to_print=False
):
    """
    Identify the top percentile of cells with highest weight concentration in a single dimension.

    Parameters:
    -----------
    archetype_vectors : numpy array
        The archetype vectors for cells
    percentile : float
        The percentile threshold (e.g., 95 for top 5%)

    Returns:
    --------
    extreme_mask : boolean array
        Mask indicating which cells are extreme archetypes
    dominant_dims : array
        The dominant dimension for each extreme archetype
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

        # Calculate the threshold value that gives the top percentile
        threshold = np.percentile(proportions, percentile)

        # Get the indices of cells with proportion > threshold
        extreme_mask = proportions >= threshold

        # Get the dominant dimension for each extreme archetype
        dominant_dims = np.argmax(archetype_vectors, axis=1)
        # Print summary statistics
        if to_print:
            logger_.info(
                f"\ndata: {np.sum(extreme_mask)} out of {len(extreme_mask)} cells are extreme archetypes ({np.sum(extreme_mask)/len(extreme_mask):.1%})"
            )
            print(f"\nthreshold value: {threshold:.3f}")
    elif isinstance(archetype_vectors, torch.Tensor):
        # Calculate proportion of each vector that is in its max dimension
        max_values = torch.max(archetype_vectors, dim=1).values
        total_values = torch.sum(archetype_vectors, dim=1)
        proportions = max_values / total_values

        # Calculate the threshold value that gives the top percentile
        threshold = torch.quantile(proportions, percentile / 100)

        # Get the indices of cells with proportion > threshold
        extreme_mask = proportions >= threshold
        # Get the dominant dimension for each extreme archetype
        dominant_dims = torch.argmax(archetype_vectors, dim=1)
        # Print summary statistics
        if to_print:
            logger_.info(
                f"\ndata: {torch.sum(extreme_mask)} out of {len(extreme_mask)} cells are extreme archetypes ({torch.sum(extreme_mask)/len(extreme_mask):.1%})"
            )
            logger_.info(f"\nthreshold value: {threshold:.3f}")
    return extreme_mask, dominant_dims, threshold


def get_cell_representations_as_archetypes_scipy(count_matrix, archetype_matrix):
    n_cells = count_matrix.shape[0]
    n_archetypes = archetype_matrix.shape[0]
    weights = np.zeros((n_cells, n_archetypes))
    A_T = archetype_matrix.T

    bounds = [(0, None)] * n_archetypes
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def objective(w, x):
        return np.sum((A_T @ w - x) ** 2)

    for i in tqdm(range(n_cells), desc="Computing archetype weights", total=n_cells):
        x = count_matrix[i]
        res = minimize(
            objective,
            np.ones(n_archetypes) / n_archetypes,
            args=(x,),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )
        weights[i] = res.x

    return weights


def get_cell_representations_as_archetypes_ols(count_matrix, archetype_matrix):
    """
    Compute archetype weights for each cell using Ordinary Least Squares (OLS).

    Parameters:
    -----------
    count_matrix : np.ndarray
        Matrix of cells in reduced-dimensional space (e.g., PCA),
        shape (n_cells, n_features).
    archetype_matrix : np.ndarray
        Matrix of archetypes,
        shape (n_archetypes, n_features).

    Returns:
    --------
    weights : np.ndarray
        Matrix of archetype weights for each cell,
        shape (n_cells, n_archetypes).
    """
    n_cells = count_matrix.shape[0]
    n_archetypes = archetype_matrix.shape[0]
    weights = np.zeros((n_cells, n_archetypes))

    # Transpose the archetype matrix
    A_T = archetype_matrix.T  # Shape: (n_features, n_archetypes)

    # For each cell, solve the least squares problem
    for i in range(n_cells):
        x = count_matrix[i]
        # Solve for w in A_T w = x
        w, residuals, rank, s = np.linalg.lstsq(A_T, x, rcond=None)
        weights[i] = w

    return weights


def get_cell_representations_as_archetypes_omp(count_matrix, archetype_matrix, tol=1e-4):
    # Preprocess archetype matrix

    n_cells = count_matrix.shape[0]
    n_archetypes = archetype_matrix.shape[0]
    weights = np.zeros((n_cells, n_archetypes))

    for i in range(n_cells):
        weights[i] = nnls_omp(archetype_matrix, count_matrix[i], tol=tol)

    row_sums = weights.sum(axis=1, keepdims=True)
    weights[row_sums == 0] = 1.0 / n_archetypes  # Assign uniform weights to zero rows

    weights /= weights.sum(axis=1, keepdims=True)

    return weights


def get_cell_representations_as_archetypes_cvxpy(
    count_matrix, archetype_matrix, solver="cp.ECOS", n_jobs=-1
):
    import cvxpy as cp

    if solver == "cp.ECOS":
        solver = cp.ECOS
    elif solver == "cp.SCS":
        solver = cp.SCS
    else:
        raise ValueError(f"Invalid solver: {solver}")
    n_cells = count_matrix.shape[0]
    n_archetypes = archetype_matrix.shape[0]  # Fix: should be [0] not entire shape
    A_T = archetype_matrix.T

    # Handle n_jobs=0 by using sequential processing
    if n_jobs == 0:
        weights = []
        for i in range(n_cells):
            weight = compute_weight_for_cell(count_matrix[i], A_T, n_archetypes, solver)
            weights.append(weight)
    else:
        weights = Parallel(n_jobs=n_jobs)(
            delayed(compute_weight_for_cell)(count_matrix[i], A_T, n_archetypes, solver)
            for i in range(n_cells)
        )
    return np.array(weights)


@timeit
def get_cell_representations_as_archetypes(count_matrix, archetype_matrix):
    """
    Compute archetype weights for each cell using cvxpy.
    """
    n_cells = count_matrix.shape[0]
    n_archetypes = archetype_matrix.shape[0]
    weights = np.zeros((n_cells, n_archetypes))
    for i in range(n_cells):
        weights[i], _ = nnls(archetype_matrix.T, count_matrix[i])
    weights /= weights.sum(axis=1, keepdims=True)  # Normalize rows
    return weights
