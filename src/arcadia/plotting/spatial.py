"""Spatial analysis plotting functions."""

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.sparse import issparse

from arcadia.plotting.general import safe_mlflow_log_figure
from arcadia.utils.logging import logger


def plot_elbow_method(evs_protein, evs_rna, max_points=300, plot_flag=True):
    """Plot elbow method results."""
    if not plot_flag:
        return
    # Limit the number of points if too many
    if len(evs_protein) > max_points or len(evs_rna) > max_points:
        logger.info(f"Limiting elbow plot to first {max_points} points")
        plot_protein = evs_protein[:max_points]
        plot_rna = evs_rna[:max_points]
    else:
        plot_protein = evs_protein
        plot_rna = evs_rna

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(plot_protein)), plot_protein, marker="o", label="Protein")
    plt.plot(range(len(plot_rna)), plot_rna, marker="s", label="RNA")
    plt.xlabel("Number of Archetypes (k)")
    plt.ylabel("Explained Variance")
    plt.title("Elbow Plot: Explained Variance vs Number of Archetypes")
    plt.legend()
    plt.grid()
    plt.close()


def plot_spatial_data(adata_prot, max_cells=5000, plot_flag=True, save_flag=False):
    """Plot spatial data for protein dataset"""
    if not plot_flag:
        return
    logger.info("Plotting spatial data...")
    plt.figure(figsize=(10, 10))

    # Subsample if needed
    subsample_n_obs = min(max_cells, len(adata_prot))
    adata_prot_sub = adata_prot[np.random.choice(len(adata_prot), subsample_n_obs, replace=False)]

    # Get unique cell types and create a color map
    unique_cell_types = adata_prot_sub.obs["cell_types"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cell_types)))
    color_dict = dict(zip(unique_cell_types, colors))

    # Create scatter plot
    for cell_type in unique_cell_types:
        mask = adata_prot_sub.obs["cell_types"] == cell_type
        plt.scatter(
            adata_prot_sub.obsm["spatial"][mask, 0],
            adata_prot_sub.obsm["spatial"][mask, 1],
            c=[color_dict[cell_type]],
            label=cell_type,
            s=1.5,
            alpha=0.6,
        )

    plt.title("Protein Spatial Data")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    if mlflow.active_run() or save_flag:
        safe_mlflow_log_figure(plt.gcf(), "protein_spatial_data.pdf")
    plt.show()
    plt.close()
    plt.close()


def plot_spatial_data_comparison(adata_rna, adata_prot):
    """Plot spatial data comparison between RNA and protein datasets"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # RNA data
    axes[0].scatter(
        adata_rna.obsm["spatial"][:, 0],
        adata_rna.obsm["spatial"][:, 1],
        c=adata_rna.obs["CN"],
        cmap="tab10",
        alpha=0.6,
    )
    axes[0].set_title("RNA Spatial Data")
    axes[0].set_xlabel("X coordinate")
    axes[0].set_ylabel("Y coordinate")

    # Protein data
    axes[1].scatter(
        adata_prot.obsm["spatial"][:, 0],
        adata_prot.obsm["spatial"][:, 1],
        c=adata_prot.obs["CN"],
        cmap="tab10",
        alpha=0.6,
    )
    axes[1].set_title("Protein Spatial Data")
    axes[1].set_xlabel("X coordinate")
    axes[1].set_ylabel("Y coordinate")

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_modality_embeddings(adata_1_rna, adata_2_prot, max_cells=2000, plot_flag=True):
    """Plot PCA and UMAP embeddings for both modalities."""
    if not plot_flag:
        return
    # Subsample data if needed
    sc.pl.pca(
        adata_1_rna,
        color=["cell_types", "major_cell_types"],
        title=["RNA pca minor cell types", "RNA pca major cell types"],
    )
    sc.pl.pca(
        adata_2_prot,
        color=["cell_types", "major_cell_types"],
        title=["Protein pca minor cell types", "Protein pca major cell types"],
    )
    sc.pl.umap(
        adata_1_rna,
        color=["major_cell_types", "cell_types"],
        title=["RNA UMAP major cell types", "RNA UMAP major cell types"],
    )
    sc.pl.umap(
        adata_2_prot,
        color=["major_cell_types", "cell_types"],
        title=["Protein UMAP major cell types", "Protein UMAP major cell types"],
    )


def plot_minor_cell_types_spatial_distribution(adata_prot, plot_flag=True, max_cells=2000):
    """Plot spatial distribution of minor cell types with grid overlay.

    Creates a spatial scatter plot showing how different minor cell types
    are distributed across spatial regions, with grid lines indicating
    spatial boundaries.

    Parameters
    ----------
    adata_prot : AnnData
        Protein AnnData object with spatial coordinates and minor_cell_types
    plot_flag : bool, optional
        Whether to generate the plot, by default True
    max_cells : int, optional
        Maximum number of cells to plot, by default 2000
    """
    if not plot_flag:
        return

    # Get all unique minor cell types that have spatial coordinates
    minor_cell_types = adata_prot.obs["minor_cell_types"].unique()
    if len(minor_cell_types) == 0:
        logger.info("No minor cell types found for spatial plotting")
        return

    # Subsample for plotting
    if adata_prot.n_obs > max_cells:
        logger.info(f"Subsampling {adata_prot.n_obs} cells to {max_cells} for spatial plotting")
        plot_indices = np.random.choice(adata_prot.n_obs, max_cells, replace=False)
        adata_prot_plot = adata_prot[plot_indices].copy()
    else:
        adata_prot_plot = adata_prot.copy()

    # Take only the maximal cell type
    cell_type_counts = adata_prot_plot.obs["cell_types"].value_counts()
    if len(cell_type_counts) == 0:
        logger.info("No cell types found for spatial plotting")
        return

    maximal_cell_type = cell_type_counts.index[0]
    adata_prot_plot = adata_prot_plot[adata_prot_plot.obs["cell_types"] == maximal_cell_type].copy()

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot the entire spatial space (1000x1000) as background
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)

    # Get unique cell types from subsampled data
    plot_cell_types = sorted(set(adata_prot_plot.obs["minor_cell_types"]))
    colors = plt.cm.tab20(np.linspace(0, 1, len(plot_cell_types)))

    # Plot each minor cell type with different colors
    for i, minor_type in enumerate(plot_cell_types):
        minor_cells = adata_prot_plot[adata_prot_plot.obs["minor_cell_types"] == minor_type]
        if len(minor_cells) > 0:
            ax.scatter(
                minor_cells.obs["X"],
                minor_cells.obs["Y"],
                c=[colors[i]],
                label=f"{minor_type} ({len(minor_cells)})",
                alpha=0.8,
                s=20,
            )

    # Draw dynamic grid lines to show the spatial regions
    if "spatial_grid" in adata_prot_plot.uns:
        horizontal_splits = adata_prot_plot.uns["spatial_grid"]["horizontal_splits"]
        vertical_splits = adata_prot_plot.uns["spatial_grid"]["vertical_splits"]

        for x_split in horizontal_splits[1:-1]:  # Skip first and last (boundaries)
            ax.axvline(x=x_split, color="black", linestyle="-", alpha=0.8, linewidth=2)
        for y_split in vertical_splits[1:-1]:  # Skip first and last (boundaries)
            ax.axhline(y=y_split, color="black", linestyle="-", alpha=0.8, linewidth=2)

        # Add region labels showing what's in each grid cell
        grid_cols = adata_prot_plot.uns["spatial_grid"]["grid_cols"]
        adata_prot_plot.uns["spatial_grid"]["grid_rows"]
        max_subtypes = adata_prot_plot.uns["spatial_grid"]["max_subtypes"]
        region_colors = [
            "lightblue",
            "lightgreen",
            "lightyellow",
            "lightcoral",
            "lightpink",
            "lightgray",
            "lavender",
            "lightcyan",
            "mistyrose",
        ]

        # Label each grid cell with its subtype index
        for subtype_idx in range(max_subtypes):
            grid_row = subtype_idx // grid_cols
            grid_col = subtype_idx % grid_cols

            if grid_row < len(vertical_splits) - 1 and grid_col < len(horizontal_splits) - 1:
                x_center = (horizontal_splits[grid_col] + horizontal_splits[grid_col + 1]) / 2
                y_center = (vertical_splits[grid_row] + vertical_splits[grid_row + 1]) / 2

                color = region_colors[subtype_idx % len(region_colors)]

                ax.text(
                    x_center,
                    y_center,
                    f"Subtype #{subtype_idx + 1}\n(All Minor Types)",
                    ha="center",
                    va="center",
                    fontsize=8,
                    weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                    rotation=0,
                )

    ax.set_title(
        "CITE-seq Minor Cell Types Spatial Segregation\n"
        + "Synthetic spatial coordinates show how different cell subtypes\n"
        + "are placed in different spatial regions",
        fontsize=14,
    )
    ax.set_xlabel("X coordinate", fontsize=12)
    ax.set_ylabel("Y coordinate", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_residualization_comparison(
    spatial_adata_raw, spatial_adata_resid, cell_type_column="cell_types", plot_flag=True
):
    """Plot comparison of CN features before and after residualization.

    Creates three heatmaps showing:
    1. Cell-type mean CN features (raw)
    2. Cell-type mean CN features (residualized)
    3. Difference between residualized and raw means

    Parameters
    ----------
    spatial_adata_raw : AnnData
        Spatial AnnData before residualization
    spatial_adata_resid : AnnData
        Spatial AnnData after residualization
    cell_type_column : str, optional
        Column name for cell types, by default "cell_types"
    plot_flag : bool, optional
        Whether to generate the plot, by default True
    """
    if not plot_flag:
        return

    # Subsample for plotting if needed
    if spatial_adata_raw.n_obs > 2000:
        spatial_adata_raw_plot = spatial_adata_raw[:2000].copy()
    else:
        spatial_adata_raw_plot = spatial_adata_raw.copy()

    if spatial_adata_resid.n_obs > 2000:
        spatial_adata_resid_plot = spatial_adata_resid[:2000].copy()
    else:
        spatial_adata_resid_plot = spatial_adata_resid.copy()

    # Calculate cell-type means
    raw_means = (
        pd.DataFrame(
            (
                spatial_adata_raw_plot.X.toarray()
                if issparse(spatial_adata_raw_plot.X)
                else spatial_adata_raw_plot.X
            ),
            columns=spatial_adata_raw_plot.var_names,
        )
        .assign(cell_types=spatial_adata_raw_plot.obs[cell_type_column].values)
        .groupby("cell_types")
        .mean()
    )

    resid_means = (
        pd.DataFrame(
            (
                spatial_adata_resid_plot.X.toarray()
                if issparse(spatial_adata_resid_plot.X)
                else spatial_adata_resid_plot.X
            ),
            columns=spatial_adata_resid_plot.var_names,
        )
        .assign(cell_types=spatial_adata_resid_plot.obs[cell_type_column].values)
        .groupby("cell_types")
        .mean()
    )

    mean_diff = resid_means - raw_means

    # Plot raw means
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        raw_means,
        cmap="coolwarm",
        center=0,
        xticklabels=False,
        yticklabels=True,
    )
    plt.title("Cell-type mean CN features (raw)")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Plot residualized means
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        resid_means,
        cmap="coolwarm",
        center=0,
        xticklabels=False,
        yticklabels=True,
    )
    plt.title("Cell-type mean CN features (residualized)")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Plot difference
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        mean_diff,
        cmap="coolwarm",
        center=0,
        xticklabels=False,
        yticklabels=True,
    )
    plt.title("Difference in cell-type means (residualized - raw)")
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.close()


def plot_spatial_features(adata_prot, max_cells=2000):
    """Plot neighbor means and raw protein expression."""
    # Apply subsampling if too many cells
    spatial_features = adata_prot[adata_prot.var["feature_type"] != "protein"].X
    protein_data = adata_prot[adata_prot.var["feature_type"] == "protein"].X
    if adata_prot.shape[0] > max_cells:
        logger.info(f"Subsampling to {max_cells} cells for neighbor means plot")
        idx = np.random.choice(adata_prot.shape[0], max_cells, replace=False)
        spatial_features_plot = spatial_features[idx]
        protein_data_plot = protein_data[idx]
    else:
        spatial_features_plot = spatial_features
        protein_data_plot = protein_data

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Mean Protein Expression of Cell Neighborhoods")
    sns.heatmap(spatial_features_plot)
    plt.subplot(1, 2, 2)
    plt.title("Protein Expression per Cell")
    sns.heatmap(protein_data_plot)
    plt.show()
    plt.close()


def plot_spatial_clusters(adata_prot, max_cells=2000, plot_flag=True, save_plot=False):
    """Plot spatial clusters and related visualizations."""
    if not plot_flag:
        return
    # if the color pallete does not match the number of categories, add more colors
    if "CN_colors" in adata_prot.uns:
        if len(adata_prot.uns["CN_colors"]) < len(adata_prot.obs["CN"].cat.categories):
            new_palette = sns.color_palette("tab10", len(adata_prot.obs["CN"].cat.categories))
            adata_prot.uns["CN_colors"] = new_palette.as_hex()

    # Apply subsampling for spatial plot if too many cells
    if adata_prot.shape[0] > max_cells:
        logger.info(f"Subsampling to {max_cells} cells for spatial cluster plot")
        adata_plot = adata_prot.copy()
        sc.pp.subsample(adata_plot, n_obs=max_cells)
    else:
        adata_plot = adata_prot

    sc.pl.scatter(
        adata_plot,
        x="X",
        y="Y",
        color="CN",
        size=10,
        title="Cluster cells by their CN, can see the different CN in different regions, \nthanks to the different B cell types in each region",
        show=False,
    )
    if save_plot:
        plt.gcf().savefig(f"spatial_clusters_CN.pdf", bbox_inches="tight")
    if plot_flag:
        plt.show()
    plt.close()

    sc.pl.scatter(
        adata_plot,
        x="X",
        y="Y",
        size=10,
        color="cell_types",
        title="Cell types in spatial locations",
        show=False,
    )
    if save_plot:
        plt.gcf().savefig(f"spatial_clusters_cell_types.pdf", bbox_inches="tight")
    if plot_flag:
        plt.show()
    plt.close()
    spatial_features = adata_plot[:, adata_plot.var["feature_type"] != "protein"].copy()
    protein_features = adata_plot[:, adata_plot.var["feature_type"] == "protein"].copy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(spatial_features.X[:600, :600])
    plt.title("spatial features")
    plt.subplot(1, 2, 2)
    sns.heatmap(protein_features.X[:600, :600])
    plt.title("protein expressions of each cell")
    plt.show()
    plt.close()

    # Use subsampled data for all downstream analyses

    sc.pp.pca(spatial_features)
    sc.pp.neighbors(spatial_features)
    sc.tl.umap(spatial_features)
    sc.pl.umap(spatial_features, color=["CN", "cell_types"], title="UMAP of CN embedding")

    from anndata import concat

    protein_features_spatial_concat = concat(
        [protein_features, spatial_features],
        join="outer",
        label="modality",
        keys=["Protein", "CN"],
    )

    X = (
        protein_features_spatial_concat.X.toarray()
        if issparse(protein_features_spatial_concat.X)
        else protein_features_spatial_concat.X
    )
    X = np.nan_to_num(X)
    protein_features_spatial_concat.X = X
    sc.pp.pca(protein_features_spatial_concat)
    sc.pp.neighbors(protein_features_spatial_concat)
    sc.tl.umap(protein_features_spatial_concat)
    sc.pl.umap(
        protein_features_spatial_concat,
        color=["CN", "modality"],
        title=[
            "UMAP of CN embedding to make sure they are not mixed",
            "UMAP of CN embedding to make sure they are not mixed",
        ],
    )
    sc.pl.pca(
        protein_features_spatial_concat,
        color=["CN", "modality"],
        title=[
            "PCA of CN embedding to make sure they are not mixed",
            "PCA of CN embedding to make sure they are not mixed",
        ],
    )
    # Stacked bar plot showing cell type composition per CN
    fig, ax = plt.subplots(figsize=(14, 6))

    cell_type_proportions = (
        protein_features_spatial_concat.obs.groupby(["CN", "cell_types"])
        .size()
        .unstack(fill_value=0)
    )
    cell_type_proportions = cell_type_proportions.div(cell_type_proportions.sum(axis=1), axis=0)
    cell_type_proportions.plot(kind="bar", stacked=True, ax=ax, width=0.8)
    ax.set_xlabel("CN", fontsize=12)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title("Cell Type Composition per CN (Stacked Bar Plot)", fontsize=14)
    ax.legend(title="Cell Types", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_protein_cn_subset_umaps(adata_prot, most_common_cell_type, plot_flag=True):
    if not plot_flag:
        return
    import anndata as ad

    protein_features = adata_prot.var_names[adata_prot.var["feature_type"] == "protein"]
    spatial_features = adata_prot.var_names[adata_prot.var["feature_type"] != "protein"]
    mask = adata_prot.obs["cell_types"] == most_common_cell_type
    prot_original = adata_prot[mask, protein_features].copy()
    prot_cn = adata_prot[mask, spatial_features].copy()
    combined = ad.concat([prot_original, prot_cn], axis=1)
    combined.obs = prot_original.obs
    for adx in (prot_cn, prot_original, combined):
        sc.pp.pca(adx)
        sc.pp.neighbors(adx)
        sc.tl.umap(adx)
    sc.pl.embedding(
        prot_original,
        color="CN",
        basis="X_umap",
        title=f"Protein Feature {protein_features[0]} for {most_common_cell_type}",
    )
    sc.pl.embedding(
        prot_cn,
        color="CN",
        basis="X_umap",
        title=f"CN Feature {spatial_features[0]} for {most_common_cell_type}",
    )
    sc.pl.embedding(
        combined, color="CN", basis="X_umap", title=f"combined features for {most_common_cell_type}"
    )

    # Add spatial location plots
    subset_adata = adata_prot[mask].copy()

    # Plot spatial location colored by CN
    sc.pl.scatter(
        subset_adata,
        x="X",
        y="Y",
        color="CN",
        title=f"Spatial Location - CN clusters for {most_common_cell_type}",
    )

    # Plot spatial location colored by cell types
    sc.pl.scatter(
        subset_adata,
        x="X",
        y="Y",
        color="cell_types",
        title=f"Spatial Location - Cell types for {most_common_cell_type}",
    )


def plot_protein_vs_cn_statistics(protein_data, cn_data, plot_flag=True):
    if not plot_flag:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    protein_feature_means = np.mean(protein_data, axis=0)
    cn_feature_means = np.mean(cn_data, axis=0)
    max_num_feature_means = max(len(protein_feature_means), len(cn_feature_means))
    protein_feature_means = protein_feature_means[:max_num_feature_means]
    cn_feature_means = cn_feature_means[:max_num_feature_means]
    axes[0].hist(protein_feature_means, alpha=0.5, bins=30, label="Protein Features")
    axes[0].hist(cn_feature_means, alpha=0.5, bins=30, label="CN Features")
    axes[0].set_title("Distribution of Feature Means")
    axes[0].set_xlabel("Mean Value")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    protein_feature_vars = np.var(protein_data, axis=0)
    cn_feature_vars = np.var(cn_data, axis=0)
    axes[1].hist(protein_feature_vars, alpha=0.5, bins=30, label="Protein Features")
    axes[1].hist(cn_feature_vars, alpha=0.5, bins=30, label="CN Features")
    axes[1].set_title("Distribution of Feature Variances")
    axes[1].set_xlabel("Variance")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_modality_embeddings(adata_1_rna, adata_2_prot, max_cells=2000, plot_flag=True):
    """Plot PCA and UMAP embeddings for both modalities."""
    if not plot_flag:
        return
    # Subsample data if needed
    sc.pl.pca(
        adata_1_rna,
        color=["cell_types", "major_cell_types"],
        title=["RNA pca minor cell types", "RNA pca major cell types"],
    )
    sc.pl.pca(
        adata_2_prot,
        color=["cell_types", "major_cell_types"],
        title=["Protein pca minor cell types", "Protein pca major cell types"],
    )
    sc.pl.umap(
        adata_1_rna,
        color=["major_cell_types", "cell_types"],
        title=["RNA UMAP major cell types", "RNA UMAP major cell types"],
    )
    sc.pl.umap(
        adata_2_prot,
        color=["major_cell_types", "cell_types"],
        title=["Protein UMAP major cell types", "Protein UMAP major cell types"],
    )


def plot_minor_cell_types_spatial_distribution(adata_prot, plot_flag=True, max_cells=2000):
    """Plot spatial distribution of minor cell types with grid overlay.

    Creates a spatial scatter plot showing how different minor cell types
    are distributed across spatial regions, with grid lines indicating
    spatial boundaries.

    Parameters
    ----------
    adata_prot : AnnData
        Protein AnnData object with spatial coordinates and minor_cell_types
    plot_flag : bool, optional
        Whether to generate the plot, by default True
    max_cells : int, optional
        Maximum number of cells to plot, by default 2000
    """
    if not plot_flag:
        return

    # Get all unique minor cell types that have spatial coordinates
    minor_cell_types = adata_prot.obs["minor_cell_types"].unique()
    if len(minor_cell_types) == 0:
        logger.info("No minor cell types found for spatial plotting")
        return

    # Subsample for plotting
    if adata_prot.n_obs > max_cells:
        logger.info(f"Subsampling {adata_prot.n_obs} cells to {max_cells} for spatial plotting")
        plot_indices = np.random.choice(adata_prot.n_obs, max_cells, replace=False)
        adata_prot_plot = adata_prot[plot_indices].copy()
    else:
        adata_prot_plot = adata_prot.copy()

    # Take only the maximal cell type
    cell_type_counts = adata_prot_plot.obs["cell_types"].value_counts()
    if len(cell_type_counts) == 0:
        logger.info("No cell types found for spatial plotting")
        return

    maximal_cell_type = cell_type_counts.index[0]
    adata_prot_plot = adata_prot_plot[adata_prot_plot.obs["cell_types"] == maximal_cell_type].copy()

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot the entire spatial space (1000x1000) as background
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)

    # Get unique cell types from subsampled data
    plot_cell_types = sorted(set(adata_prot_plot.obs["minor_cell_types"]))
    colors = plt.cm.tab20(np.linspace(0, 1, len(plot_cell_types)))

    # Plot each minor cell type with different colors
    for i, minor_type in enumerate(plot_cell_types):
        minor_cells = adata_prot_plot[adata_prot_plot.obs["minor_cell_types"] == minor_type]
        if len(minor_cells) > 0:
            ax.scatter(
                minor_cells.obs["X"],
                minor_cells.obs["Y"],
                c=[colors[i]],
                label=f"{minor_type} ({len(minor_cells)})",
                alpha=0.8,
                s=20,
            )

    # Draw dynamic grid lines to show the spatial regions
    if "spatial_grid" in adata_prot_plot.uns:
        horizontal_splits = adata_prot_plot.uns["spatial_grid"]["horizontal_splits"]
        vertical_splits = adata_prot_plot.uns["spatial_grid"]["vertical_splits"]

        for x_split in horizontal_splits[1:-1]:  # Skip first and last (boundaries)
            ax.axvline(x=x_split, color="black", linestyle="-", alpha=0.8, linewidth=2)
        for y_split in vertical_splits[1:-1]:  # Skip first and last (boundaries)
            ax.axhline(y=y_split, color="black", linestyle="-", alpha=0.8, linewidth=2)

        # Add region labels showing what's in each grid cell
        grid_cols = adata_prot_plot.uns["spatial_grid"]["grid_cols"]
        adata_prot_plot.uns["spatial_grid"]["grid_rows"]
        max_subtypes = adata_prot_plot.uns["spatial_grid"]["max_subtypes"]
        region_colors = [
            "lightblue",
            "lightgreen",
            "lightyellow",
            "lightcoral",
            "lightpink",
            "lightgray",
            "lavender",
            "lightcyan",
            "mistyrose",
        ]

        # Label each grid cell with its subtype index
        for subtype_idx in range(max_subtypes):
            grid_row = subtype_idx // grid_cols
            grid_col = subtype_idx % grid_cols

            if grid_row < len(vertical_splits) - 1 and grid_col < len(horizontal_splits) - 1:
                x_center = (horizontal_splits[grid_col] + horizontal_splits[grid_col + 1]) / 2
                y_center = (vertical_splits[grid_row] + vertical_splits[grid_row + 1]) / 2

                color = region_colors[subtype_idx % len(region_colors)]

                ax.text(
                    x_center,
                    y_center,
                    f"Subtype #{subtype_idx + 1}\n(All Minor Types)",
                    ha="center",
                    va="center",
                    fontsize=8,
                    weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                    rotation=0,
                )

    ax.set_title(
        "CITE-seq Minor Cell Types Spatial Segregation\n"
        + "Synthetic spatial coordinates show how different cell subtypes\n"
        + "are placed in different spatial regions",
        fontsize=14,
    )
    ax.set_xlabel("X coordinate", fontsize=12)
    ax.set_ylabel("Y coordinate", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_residualization_comparison(
    spatial_adata_raw, spatial_adata_resid, cell_type_column="cell_types", plot_flag=True
):
    """Plot comparison of CN features before and after residualization.

    Creates three heatmaps showing:
    1. Cell-type mean CN features (raw)
    2. Cell-type mean CN features (residualized)
    3. Difference between residualized and raw means

    Parameters
    ----------
    spatial_adata_raw : AnnData
        Spatial AnnData before residualization
    spatial_adata_resid : AnnData
        Spatial AnnData after residualization
    cell_type_column : str, optional
        Column name for cell types, by default "cell_types"
    plot_flag : bool, optional
        Whether to generate the plot, by default True
    """
    if not plot_flag:
        return

    # Subsample for plotting if needed
    if spatial_adata_raw.n_obs > 2000:
        spatial_adata_raw_plot = spatial_adata_raw[:2000].copy()
    else:
        spatial_adata_raw_plot = spatial_adata_raw.copy()

    if spatial_adata_resid.n_obs > 2000:
        spatial_adata_resid_plot = spatial_adata_resid[:2000].copy()
    else:
        spatial_adata_resid_plot = spatial_adata_resid.copy()

    # Calculate cell-type means
    raw_means = (
        pd.DataFrame(
            (
                spatial_adata_raw_plot.X.toarray()
                if issparse(spatial_adata_raw_plot.X)
                else spatial_adata_raw_plot.X
            ),
            columns=spatial_adata_raw_plot.var_names,
        )
        .assign(cell_types=spatial_adata_raw_plot.obs[cell_type_column].values)
        .groupby("cell_types")
        .mean()
    )

    resid_means = (
        pd.DataFrame(
            (
                spatial_adata_resid_plot.X.toarray()
                if issparse(spatial_adata_resid_plot.X)
                else spatial_adata_resid_plot.X
            ),
            columns=spatial_adata_resid_plot.var_names,
        )
        .assign(cell_types=spatial_adata_resid_plot.obs[cell_type_column].values)
        .groupby("cell_types")
        .mean()
    )

    mean_diff = resid_means - raw_means

    # Plot raw means
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        raw_means,
        cmap="coolwarm",
        center=0,
        xticklabels=False,
        yticklabels=True,
    )
    plt.title("Cell-type mean CN features (raw)")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Plot residualized means
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        resid_means,
        cmap="coolwarm",
        center=0,
        xticklabels=False,
        yticklabels=True,
    )
    plt.title("Cell-type mean CN features (residualized)")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Plot difference
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        mean_diff,
        cmap="coolwarm",
        center=0,
        xticklabels=False,
        yticklabels=True,
    )
    plt.title("Difference in cell-type means (residualized - raw)")
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_spatial_data_histograms(spatial_adata, n_cells=10, plot_flag=True):
    """Plot histograms of spatial data."""
    if not plot_flag:
        return

    # Plot histogram of flattened spatial data
    plt.hist(spatial_adata.X.flatten()[:1000])
    plt.show()

    # Plot histograms of random cells
    for i in range(n_cells):
        plt.hist(spatial_adata.X[i, :], bins=30, alpha=0.5, label=f"Cell {i}")
    plt.legend()
    plt.show()


def plot_feature_mean_distributions(spatial_adata, plot_flag=True):
    """Plot distribution of mean feature values by feature type."""
    if not plot_flag:
        return

    feature_types = spatial_adata.var["feature_type"]
    mean_features = spatial_adata.X.mean(axis=0)

    df = pd.DataFrame({"mean_value": mean_features, "feature_type": feature_types})

    plt.figure(figsize=(10, 6))
    for feature_type in df["feature_type"].unique():
        subset = df[df["feature_type"] == feature_type]
        plt.hist(subset["mean_value"], bins=30, alpha=0.7, label=feature_type)

    plt.xlabel("Mean Feature Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Distribution of Mean Feature Values by Feature Type")
    plt.show()


def plot_feature_means_line(adata, plot_flag=True):
    """Plot line plot of mean feature values."""
    if not plot_flag:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(adata.X.mean(axis=0))
    plt.show()


def plot_feature_value_distributions(adata_prot, num_points=1000, plot_flag=True):
    if not plot_flag:
        return
    protein_mask = adata_prot.var["feature_type"] == "protein"
    spatial_mask = adata_prot.var["feature_type"] != "protein"

    if issparse(adata_prot.X):
        protein_features = adata_prot.X[:, protein_mask].toarray()
        spatial_features = adata_prot.X[:, spatial_mask].toarray()
    else:
        protein_features = adata_prot.X[:, protein_mask]
        spatial_features = adata_prot.X[:, spatial_mask]

    # Sample 2k cells
    num_samples = min(2000, protein_features.shape[0])
    sample_indices = np.random.choice(protein_features.shape[0], num_samples, replace=False)

    protein_sample = protein_features[sample_indices].flatten()
    spatial_sample = spatial_features[sample_indices].flatten()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(protein_sample, alpha=0.6, bins=50, label="Protein Values", density=True)
    plt.hist(spatial_sample, alpha=0.6, bins=50, label="Spatial Values", density=True)
    plt.xlabel("Feature Values")
    plt.ylabel("Density")
    plt.title("Distribution of Raw Values")
    plt.legend()
    plt.yscale("log")
    plt.subplot(1, 2, 2)
    sample_size = min(num_points, len(protein_sample))
    plt.scatter(protein_sample[:sample_size], spatial_sample[:sample_size], alpha=0.3, s=1)
    plt.xlabel("Protein Feature Values")
    plt.ylabel("Spatial Feature Values")
    plt.title(
        f"Protein vs Spatial Feature Values ({sample_size} points) makking sure thre is no direct correlation"
    )
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_modality_embeddings(adata_1_rna, adata_2_prot, max_cells=2000, plot_flag=True):
    """Plot PCA and UMAP embeddings for both modalities."""
    if not plot_flag:
        return
    # Subsample data if needed
    sc.pl.pca(
        adata_1_rna,
        color=["cell_types", "major_cell_types"],
        title=["RNA pca minor cell types", "RNA pca major cell types"],
    )
    sc.pl.pca(
        adata_2_prot,
        color=["cell_types", "major_cell_types"],
        title=["Protein pca minor cell types", "Protein pca major cell types"],
    )
    sc.pl.umap(
        adata_1_rna,
        color=["major_cell_types", "cell_types"],
        title=["RNA UMAP major cell types", "RNA UMAP major cell types"],
    )
    sc.pl.umap(
        adata_2_prot,
        color=["major_cell_types", "cell_types"],
        title=["Protein UMAP major cell types", "Protein UMAP major cell types"],
    )


def plot_minor_cell_types_spatial_distribution(adata_prot, plot_flag=True, max_cells=2000):
    """Plot spatial distribution of minor cell types with grid overlay.

    Creates a spatial scatter plot showing how different minor cell types
    are distributed across spatial regions, with grid lines indicating
    spatial boundaries.

    Parameters
    ----------
    adata_prot : AnnData
        Protein AnnData object with spatial coordinates and minor_cell_types
    plot_flag : bool, optional
        Whether to generate the plot, by default True
    max_cells : int, optional
        Maximum number of cells to plot, by default 2000
    """
    if not plot_flag:
        return

    # Get all unique minor cell types that have spatial coordinates
    minor_cell_types = adata_prot.obs["minor_cell_types"].unique()
    if len(minor_cell_types) == 0:
        logger.info("No minor cell types found for spatial plotting")
        return

    # Subsample for plotting
    if adata_prot.n_obs > max_cells:
        logger.info(f"Subsampling {adata_prot.n_obs} cells to {max_cells} for spatial plotting")
        plot_indices = np.random.choice(adata_prot.n_obs, max_cells, replace=False)
        adata_prot_plot = adata_prot[plot_indices].copy()
    else:
        adata_prot_plot = adata_prot.copy()

    # Take only the maximal cell type
    cell_type_counts = adata_prot_plot.obs["cell_types"].value_counts()
    if len(cell_type_counts) == 0:
        logger.info("No cell types found for spatial plotting")
        return

    maximal_cell_type = cell_type_counts.index[0]
    adata_prot_plot = adata_prot_plot[adata_prot_plot.obs["cell_types"] == maximal_cell_type].copy()

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot the entire spatial space (1000x1000) as background
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)

    # Get unique cell types from subsampled data
    plot_cell_types = sorted(set(adata_prot_plot.obs["minor_cell_types"]))
    colors = plt.cm.tab20(np.linspace(0, 1, len(plot_cell_types)))

    # Plot each minor cell type with different colors
    for i, minor_type in enumerate(plot_cell_types):
        minor_cells = adata_prot_plot[adata_prot_plot.obs["minor_cell_types"] == minor_type]
        if len(minor_cells) > 0:
            ax.scatter(
                minor_cells.obs["X"],
                minor_cells.obs["Y"],
                c=[colors[i]],
                label=f"{minor_type} ({len(minor_cells)})",
                alpha=0.8,
                s=20,
            )

    # Draw dynamic grid lines to show the spatial regions
    if "spatial_grid" in adata_prot_plot.uns:
        horizontal_splits = adata_prot_plot.uns["spatial_grid"]["horizontal_splits"]
        vertical_splits = adata_prot_plot.uns["spatial_grid"]["vertical_splits"]

        for x_split in horizontal_splits[1:-1]:  # Skip first and last (boundaries)
            ax.axvline(x=x_split, color="black", linestyle="-", alpha=0.8, linewidth=2)
        for y_split in vertical_splits[1:-1]:  # Skip first and last (boundaries)
            ax.axhline(y=y_split, color="black", linestyle="-", alpha=0.8, linewidth=2)

        # Add region labels showing what's in each grid cell
        grid_cols = adata_prot_plot.uns["spatial_grid"]["grid_cols"]
        adata_prot_plot.uns["spatial_grid"]["grid_rows"]
        max_subtypes = adata_prot_plot.uns["spatial_grid"]["max_subtypes"]
        region_colors = [
            "lightblue",
            "lightgreen",
            "lightyellow",
            "lightcoral",
            "lightpink",
            "lightgray",
            "lavender",
            "lightcyan",
            "mistyrose",
        ]

        # Label each grid cell with its subtype index
        for subtype_idx in range(max_subtypes):
            grid_row = subtype_idx // grid_cols
            grid_col = subtype_idx % grid_cols

            if grid_row < len(vertical_splits) - 1 and grid_col < len(horizontal_splits) - 1:
                x_center = (horizontal_splits[grid_col] + horizontal_splits[grid_col + 1]) / 2
                y_center = (vertical_splits[grid_row] + vertical_splits[grid_row + 1]) / 2

                color = region_colors[subtype_idx % len(region_colors)]

                ax.text(
                    x_center,
                    y_center,
                    f"Subtype #{subtype_idx + 1}\n(All Minor Types)",
                    ha="center",
                    va="center",
                    fontsize=8,
                    weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                    rotation=0,
                )

    ax.set_title(
        "CITE-seq Minor Cell Types Spatial Segregation\n"
        + "Synthetic spatial coordinates show how different cell subtypes\n"
        + "are placed in different spatial regions",
        fontsize=14,
    )
    ax.set_xlabel("X coordinate", fontsize=12)
    ax.set_ylabel("Y coordinate", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_residualization_comparison(
    spatial_adata_raw, spatial_adata_resid, cell_type_column="cell_types", plot_flag=True
):
    """Plot comparison of CN features before and after residualization.

    Creates three heatmaps showing:
    1. Cell-type mean CN features (raw)
    2. Cell-type mean CN features (residualized)
    3. Difference between residualized and raw means

    Parameters
    ----------
    spatial_adata_raw : AnnData
        Spatial AnnData before residualization
    spatial_adata_resid : AnnData
        Spatial AnnData after residualization
    cell_type_column : str, optional
        Column name for cell types, by default "cell_types"
    plot_flag : bool, optional
        Whether to generate the plot, by default True
    """
    if not plot_flag:
        return

    # Subsample for plotting if needed
    if spatial_adata_raw.n_obs > 2000:
        spatial_adata_raw_plot = spatial_adata_raw[:2000].copy()
    else:
        spatial_adata_raw_plot = spatial_adata_raw.copy()

    if spatial_adata_resid.n_obs > 2000:
        spatial_adata_resid_plot = spatial_adata_resid[:2000].copy()
    else:
        spatial_adata_resid_plot = spatial_adata_resid.copy()

    # Calculate cell-type means
    raw_means = (
        pd.DataFrame(
            (
                spatial_adata_raw_plot.X.toarray()
                if issparse(spatial_adata_raw_plot.X)
                else spatial_adata_raw_plot.X
            ),
            columns=spatial_adata_raw_plot.var_names,
        )
        .assign(cell_types=spatial_adata_raw_plot.obs[cell_type_column].values)
        .groupby("cell_types")
        .mean()
    )

    resid_means = (
        pd.DataFrame(
            (
                spatial_adata_resid_plot.X.toarray()
                if issparse(spatial_adata_resid_plot.X)
                else spatial_adata_resid_plot.X
            ),
            columns=spatial_adata_resid_plot.var_names,
        )
        .assign(cell_types=spatial_adata_resid_plot.obs[cell_type_column].values)
        .groupby("cell_types")
        .mean()
    )

    mean_diff = resid_means - raw_means

    # Plot raw means
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        raw_means,
        cmap="coolwarm",
        center=0,
        xticklabels=False,
        yticklabels=True,
    )
    plt.title("Cell-type mean CN features (raw)")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Plot residualized means
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        resid_means,
        cmap="coolwarm",
        center=0,
        xticklabels=False,
        yticklabels=True,
    )
    plt.title("Cell-type mean CN features (residualized)")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Plot difference
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        mean_diff,
        cmap="coolwarm",
        center=0,
        xticklabels=False,
        yticklabels=True,
    )
    plt.title("Difference in cell-type means (residualized - raw)")
    plt.tight_layout()
    plt.show()
    plt.close()
