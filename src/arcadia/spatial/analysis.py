"""Spatial analysis utilities."""

import scanpy as sc


def spatial_analysis(adata: sc.AnnData) -> None:
    """Optional spatial analysis"""
    try:
        import squidpy as sq

        print("\nPerforming spatial analysis...")
        # subsample the adata to 3000 cells
        adata_subsampled = sc.pp.subsample(adata, n_obs=3000, copy=True)
        sq.gr.spatial_neighbors(adata_subsampled, coord_type="generic")
        sq.pl.spatial_scatter(adata_subsampled, color="cell_types", shape=None)

    except ImportError:
        print("Squidpy not installed - skipping spatial analysis")
