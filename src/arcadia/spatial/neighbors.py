"""Spatial neighbor utilities."""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import find


def create_smart_neighbors(adata_prot: sc.AnnData, percentile_threshold: int = 95) -> sc.AnnData:
    """
    Creates smart neighbors for the protein data by protecting the 5 closest neighbors for each cell
    and then filtering the neighbors based on the distance threshold
    """
    spatial_distances = adata_prot.obsp["spatial_neighbors_distances"]
    connectivities = adata_prot.obsp["spatial_neighbors_connectivities"]

    percentile_threshold = 95
    percentile_value = np.percentile(spatial_distances.data, percentile_threshold)
    # here we want to protect the 5 closest neighbors for each cell, since we can't have cells without neighbors
    # Get the mask for connections above the threshold
    above_threshold_mask = spatial_distances > percentile_value

    # For each cell, find 5 closest neighbors and protect them

    # Get non-zero values and their indices
    rows, cols, values = find(spatial_distances)
    df = pd.DataFrame({"row": rows, "col": cols, "distance": values})

    # Find 5 closest neighbors for each cell
    min_neighbors = 5
    closest_neighbors = (
        df.groupby("row")
        .apply(lambda x: x.nsmallest(min_neighbors, "distance"))
        .reset_index(drop=True)
    )

    # Create pairs of (row, col) for protected connections

    # Remove protected pairs from the above_threshold_mask
    protected_rows = closest_neighbors["row"].values
    protected_cols = closest_neighbors["col"].values

    above_threshold_mask[protected_rows, protected_cols] = False

    # Apply the modified mask to zero out connections
    connectivities[above_threshold_mask] = 0.0
    spatial_distances[above_threshold_mask] = 0.0
    connectivities[connectivities > 0] = 1
    adata_prot.obsp["spatial_neighbors_connectivities"] = connectivities
    adata_prot.obsp["spatial_neighbors_distances"] = spatial_distances
    return adata_prot
