"""Core COVET computation functions.

COVET (Covariance Environment) implementation based on:
"The covariance environment defines cellular niches for spatial inference"
by Dana Pe'er, Doron Haviv et al.
GitHub: https://github.com/dpeerlab/ENVI
"""

import numpy as np
import scanpy as sc
import scipy
import sklearn.neighbors


def MatSqrt(Mats):
    """
    Compute matrix square root for batch of matrices
    :param Mats: 3D array of shape (n_cells, n_genes, n_genes)
    :return: Matrix square roots
    """
    e, v = np.linalg.eigh(Mats)
    e = np.where(e < 0, 0, e)  # Ensure positive eigenvalues
    e = np.sqrt(e)

    m, n = e.shape
    diag_e = np.zeros((m, n, n), dtype=e.dtype)
    diag_e.reshape(-1, n**2)[..., :: n + 1] = e
    return np.matmul(np.matmul(v, diag_e), v.transpose([0, 2, 1]))


def BatchKNN(data, batch, k):
    """
    Compute k-nearest neighbors within batches
    :param data: spatial coordinates
    :param batch: batch labels
    :param k: number of neighbors
    :return: neighbor indices
    """
    kNNGraphIndex = np.zeros(shape=(data.shape[0], k))

    for val in np.unique(batch):
        val_ind = np.where(batch == val)[0]

        batch_knn = sklearn.neighbors.kneighbors_graph(
            data[val_ind], n_neighbors=k, mode="connectivity", n_jobs=-1
        ).tocoo()
        batch_knn_ind = np.reshape(np.asarray(batch_knn.col), [data[val_ind].shape[0], k])
        kNNGraphIndex[val_ind] = val_ind[batch_knn_ind]

    return kNNGraphIndex.astype("int")


def CalcCovMats(spatial_data, kNN, genes, spatial_key="spatial", batch_key=-1):
    """
    Calculate COVET covariance matrices using shifted covariance

    :param spatial_data: AnnData object with spatial transcriptomics data
    :param kNN: number of nearest neighbors to define niche
    :param genes: list of gene names to use for covariance calculation
    :param spatial_key: key for spatial coordinates in obsm
    :param batch_key: key for batch information, or -1 for no batch
    :return: 3D array of covariance matrices (n_cells, n_genes, n_genes)
    """

    # Get expression data (log-transformed)
    ExpData = (
        np.log(spatial_data[:, genes].X.toarray() + 1)
        if scipy.sparse.issparse(spatial_data[:, genes].X)
        else np.log(spatial_data[:, genes].X + 1)
    )

    # Compute k-nearest neighbor graph
    if batch_key == -1:
        kNNGraph = sklearn.neighbors.kneighbors_graph(
            spatial_data.obsm[spatial_key],
            n_neighbors=kNN,
            mode="connectivity",
            n_jobs=-1,
        ).tocoo()
        kNNGraphIndex = np.reshape(
            np.asarray(kNNGraph.col), [spatial_data.obsm[spatial_key].shape[0], kNN]
        )
    else:
        kNNGraphIndex = BatchKNN(spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], kNN)

    # KEY INNOVATION: Use global mean instead of local mean (shifted covariance)
    # This enables direct comparison between different cellular niches
    global_mean = ExpData.mean(axis=0)  # Global mean across all cells

    # Calculate distance from global mean for each cell's neighborhood
    DistanceMatWeighted = (
        global_mean[None, None, :] - ExpData[kNNGraphIndex[np.arange(ExpData.shape[0])]]
    )

    # Compute covariance matrices: (X - μ_global)^T (X - μ_global) / (k-1)
    CovMats = np.matmul(DistanceMatWeighted.transpose([0, 2, 1]), DistanceMatWeighted) / (kNN - 1)

    # Add small regularization for numerical stability
    CovMats = CovMats + CovMats.mean() * 0.00001 * np.expand_dims(
        np.identity(CovMats.shape[-1]), axis=0
    )

    return CovMats


def compute_covet(spatial_data, k=8, g=64, genes=[], spatial_key="spatial", batch_key="batch"):
    """
    Compute COVET representation for spatial transcriptomics data

    :param spatial_data: AnnData object with spatial coordinates in obsm
    :param k: number of nearest neighbors to define cellular niche (default 8)
    :param g: number of highly variable genes to use (default 64, -1 for all genes)
    :param genes: additional genes to include in covariance calculation
    :param spatial_key: obsm key with spatial coordinates (default 'spatial')
    :param batch_key: obs key with batch information (default 'batch')

    :return: (COVET, COVET_SQRT, CovGenes)
        - COVET: covariance matrices (n_cells, n_genes, n_genes)
        - COVET_SQRT: matrix square roots for efficient distance computation
        - CovGenes: names of genes used in covariance calculation
    """

    # Check for highly variable genes
    if "highly_variable" not in spatial_data.var.columns:
        raise Exception(
            "highly_variable not in spatial_data.var.columns, you should set all genes as highly variable (or run sc.pp.highly_variable_genes(spatial_data, n_top_genes=g))"
        )

    # Select genes for COVET calculation
    if g == -1:
        CovGenes = spatial_data.var_names
    else:
        if "highly_variable" not in spatial_data.var.columns:
            if "log" in spatial_data.layers.keys():
                sc.pp.highly_variable_genes(spatial_data, n_top_genes=g, layer="log")
            elif "log1p" in spatial_data.layers.keys():
                sc.pp.highly_variable_genes(spatial_data, n_top_genes=g, layer="log1p")
            elif spatial_data.X.min() < 0:
                sc.pp.highly_variable_genes(spatial_data, n_top_genes=g)
            else:
                spatial_data.layers["log"] = (
                    np.log(spatial_data.X.toarray() + 1)
                    if scipy.sparse.issparse(spatial_data.X)
                    else np.log(spatial_data.X + 1)
                )
                sc.pp.highly_variable_genes(spatial_data, n_top_genes=g, layer="log")

        CovGenes = np.asarray(spatial_data.var_names[spatial_data.var.highly_variable])
        if len(genes) > 0:
            CovGenes = np.union1d(CovGenes, genes)

    # Handle batch information
    if batch_key not in spatial_data.obs.columns:
        batch_key = -1

    # Validate input data
    if spatial_data.X.min() < 0:
        raise ValueError("data has negative values, all values must be non-negative for COVET")

    # Calculate COVET matrices
    COVET = CalcCovMats(
        spatial_data, k, genes=CovGenes, spatial_key=spatial_key, batch_key=batch_key
    )

    # Calculate matrix square root for efficient optimal transport distance computation
    COVET_SQRT = MatSqrt(COVET)

    return (
        COVET.astype("float32"),
        COVET_SQRT.astype("float32"),
        np.asarray(CovGenes).astype("str"),
    )
