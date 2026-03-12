"""Data cleaning utilities."""

import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData


def _to_serializable(x):
    # Normalize numpy scalars
    if isinstance(x, (np.generic,)):
        return x.item()
    return x


def sanitize_for_h5ad(obj):
    if isinstance(obj, dict):
        # convert all keys to str and recurse on values
        return {str(k): sanitize_for_h5ad(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_h5ad(v) for v in obj]  # HDF5 can’t store sets/tuples; list is safer
    elif isinstance(obj, pd.DataFrame):
        return obj  # DataFrames are supported in AnnData containers
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        # try to convert object arrays to strings to avoid ambiguous types
        return obj.astype(str)
    else:
        return _to_serializable(obj)


# Memory cleanup functions for AnnData objects during training


def clean_uns_for_h5ad(adata: AnnData):
    # Sanitize uns recursively
    adata.uns = sanitize_for_h5ad(dict(adata.uns))
    # Ensure container keys are strings
    adata.obsm = {str(k): v for k, v in dict(adata.obsm).items()}
    adata.varm = {str(k): v for k, v in dict(adata.varm).items()}
    adata.obsp = {str(k): v for k, v in dict(adata.obsp).items()}
    adata.varp = {str(k): v for k, v in dict(adata.varp).items()}
    # Ensure column names are strings
    adata.obs.columns = adata.obs.columns.astype(str)
    adata.var.columns = adata.var.columns.astype(str)
    if adata.raw is not None:
        adata.raw.var.columns = adata.raw.var.columns.astype(str)

    # Fix object dtype columns in obs that contain non-string objects
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            # Try to convert to string, handling tuples and other objects
            adata.obs[col] = adata.obs[col].apply(lambda x: str(x) if x is not None else "")

    keys_to_remove = []
    for key, value in adata.uns.items():
        if isinstance(value, sns.palettes._ColorPalette):
            # Convert seaborn ColorPalette to a list of colors
            adata.uns[key] = list(value)
        elif not isinstance(value, (str, int, float, list, dict, np.ndarray)):
            # Mark non-serializable keys for removal
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del adata.uns[key]
    return adata
