"""Environment setup utilities."""

import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import torch


def setup_environment():
    """Setup environment variables and random seeds"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.max_rows", 10)
    warnings.filterwarnings("ignore")
    pd.options.display.max_rows = 10
    pd.options.display.max_columns = 10
    np.set_printoptions(threshold=100)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return device


def get_umap_filtered_fucntion():
    """Get UMAP function that filters duplicates before running."""
    # Save original UMAP function if not already wrapped
    _original_umap = sc.tl.umap

    def umap_filtered(adata, *args, **kwargs):
        if "duplicate" in adata.obs.columns:
            # Filter duplicates and remove the triggering column
            adata_filtered = adata[~adata.obs["duplicate"]].copy()
            adata_filtered.obs["duplicate_temp"] = adata_filtered.obs["duplicate"]
            del adata_filtered.obs["duplicate"]
            # Run original UMAP on filtered data
            _original_umap(adata_filtered, *args, **kwargs)
            adata_filtered.obs["duplicate"] = adata_filtered.obs["duplicate_temp"]
            # Map results back to original adata
            import numpy as np

            umap_results = np.full((adata.n_obs, adata_filtered.obsm["X_umap"].shape[1]), np.nan)
            umap_results[~adata.obs["duplicate"].values] = adata_filtered.obsm["X_umap"]
            adata.obsm["X_umap"] = umap_results
        else:
            _original_umap(adata, *args, **kwargs)

    return umap_filtered
