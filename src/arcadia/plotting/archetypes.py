"""Archetype-related plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from anndata import AnnData
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

from arcadia.archetypes.distances import compute_archetype_distances
from arcadia.archetypes.matching import identify_extreme_archetypes_balanced
from arcadia.data_utils.preprocessing import log1p_rna
from arcadia.plotting.general import compute_opacity_per_cell_type, safe_mlflow_log_figure
from arcadia.utils.logging import logger


def plot_cross_modal_archetype_similarity_matrix(similarity_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix[:1000, :1000],
        annot=False,
        cmap="viridis",
        cbar_kws={"label": "Cosine Distance"},
    )
    plt.title("Cross-Modal Archetype Similarity Matrix")
    plt.xlabel("Protein Archetypes")
    plt.ylabel("RNA Archetypes")
    plt.tight_layout()
    plt.show()
    plt.close()

    print("✅ Cross-modal matching visualizations completed")


def plot_cn_features_correlation(
    adata_prot, features_to_plot=["CN", "neighbor_mean", "neighbor_variance"], plot_flag=True
):
    """
    Plot the correlation between CN types and spatial features.
    """
    if not plot_flag:
        return

    # Check if feature_type column exists
    if "feature_type" not in adata_prot.var.columns:
        print(
            "Warning: 'feature_type' column not found in adata_prot.var. Skipping CN features correlation plot."
        )
        return

    # Check if CN or neighbor_mean features exist
    spatial_mask = adata_prot.var["feature_type"].isin(features_to_plot)
    if not spatial_mask.any():
        print(
            f"Warning: No {features_to_plot} features found in adata_prot. Skipping CN features correlation plot."
        )
        return

    # Check if CN column exists in obs
    if "CN" not in adata_prot.obs.columns:
        print(
            "Warning: 'CN' column not found in adata_prot.obs. Skipping CN features correlation plot."
        )
        return

    adata_spatial = adata_prot[:, spatial_mask]

    # Check if we have any spatial features
    if adata_spatial.shape[1] == 0:
        print(
            f"Warning: No {features_to_plot} features found in adata_prot. Skipping CN features correlation plot."
        )
        return

    # Create a DataFrame with CN types and spatial features
    spatial_df = pd.DataFrame(
        adata_spatial.X.toarray(), index=adata_spatial.obs.index, columns=adata_spatial.var.index
    )
    spatial_df["CN"] = adata_spatial.obs["CN"].values

    # Calculate mean expression for each CN type across all spatial features
    cn_means = spatial_df.groupby("CN", observed=False).mean()

    # Check if we have enough data for clustering
    if cn_means.shape[0] < 2 or cn_means.shape[1] < 2:
        print(
            f"Warning: Insufficient data for hierarchical clustering. Skipping CN features correlation plot."
        )
        return

    # Sort CN types by the third character (convert to int) and reorder the dataframe

    # Perform hierarchical clustering on columns (spatial features) only
    col_linkage = linkage(pdist(cn_means.T, metric="euclidean"), method="ward")

    # Create clustermap with hierarchical clustering only on columns, preserving row order
    plt.figure(figsize=(20, 8))
    g = sns.clustermap(
        cn_means,
        row_cluster=True,  # Don't cluster rows, keep CN order by third character
        col_linkage=col_linkage,
        cmap="viridis",
        figsize=(20, 8),
        cbar_kws={"label": "Mean Expression"},
        dendrogram_ratio=0.15,
    )
    g.ax_heatmap.set_xlabel("Spatial Features")
    g.ax_heatmap.set_ylabel(f"{features_to_plot} Types (sorted by 3rd character)")
    plt.title(
        f"{features_to_plot} Types by Spatial Features ({features_to_plot} sorted by 3rd character)"
    )
    plt.show()
    plt.close()
    plt.close()


def plot_archetype_vs_latent_distances(archetype_dis_tensor, latent_distances, threshold):
    """Plot archetype vs latent distances"""
    logger.info("Plotting archetype vs latent distances...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot archetype distances
    axes[0].hist(archetype_dis_tensor.detach().cpu().numpy().flatten(), bins=50)
    axes[0].axvline(x=threshold, color="r", linestyle="--", label=f"Threshold: {threshold}")
    axes[0].set_title("Archetype Distances")
    axes[0].legend()

    # Plot latent distances
    axes[1].hist(latent_distances.detach().cpu().numpy().flatten(), bins=50)
    axes[1].axvline(x=threshold, color="r", linestyle="--", label=f"Threshold: {threshold}")
    axes[1].set_title("Latent Distances")
    axes[1].legend()

    plt.tight_layout()
    plt.close()
    plt.close()


def plot_archetype_embedding(adata_rna, adata_prot, use_subsample=True):
    """Plot archetype embedding"""
    # Create AnnData objects from archetype vectors
    rna_archtype = AnnData(adata_rna.obsm["archetype_vec"])
    rna_archtype.obs = adata_rna.obs.copy()

    prot_archtype = AnnData(adata_prot.obsm["archetype_vec"])
    prot_archtype.obs = adata_prot.obs.copy()
    # remove the pca from the obsm to allow for neighbor to run on the latent space directly
    rna_archtype.obsm.pop("X_pca", None)
    prot_archtype.obsm.pop("X_pca", None)

    # Apply subsampling if requested
    if use_subsample:
        # Subsample RNA data
        n_subsample_rna = min(700, rna_archtype.shape[0])
        subsample_idx_rna = np.random.choice(rna_archtype.shape[0], n_subsample_rna, replace=False)
        rna_archtype_plot = rna_archtype[subsample_idx_rna].copy()

        # Subsample protein data
        n_subsample_prot = min(700, prot_archtype.shape[0])
        subsample_idx_prot = np.random.choice(
            prot_archtype.shape[0], n_subsample_prot, replace=False
        )
        prot_archtype_plot = prot_archtype[subsample_idx_prot].copy()
    else:
        rna_archtype_plot = rna_archtype.copy()
        prot_archtype_plot = prot_archtype.copy()

    # Calculate neighbors and UMAP
    sc.pp.neighbors(rna_archtype_plot)
    sc.tl.umap(rna_archtype_plot)

    sc.pp.neighbors(prot_archtype_plot)
    sc.tl.umap(prot_archtype_plot)

    # Plot archetype vectors
    sc.pl.umap(
        rna_archtype_plot,
        color=["CN", "cell_types"],
        title=["RNA_Archetype_UMAP_CN", "RNA_Archetype_UMAP_CellTypes"],
    )
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "rna_archetype_umap.pdf")

    sc.pl.umap(
        prot_archtype_plot,
        color=["CN", "cell_types"],
        title=[
            "Protein_Archetype_UMAP_CN",
            "Protein_Archetype_UMAP_CellTypes",
        ],
    )
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "protein_archetype_umap.pdf")


# %%


def plot_archetype_proportions(
    archetype_proportion_list_rna, archetype_proportion_list_protein, max_size=20, plot_flag=True
):
    """Plot archetype proportions."""
    if not plot_flag:
        return
    # Ensure matrices aren't too large
    rna_prop = archetype_proportion_list_rna[0]
    prot_prop = archetype_proportion_list_protein[0]

    # If either has too many rows/columns, print warning and subsample
    if (
        rna_prop.shape[0] > max_size
        or rna_prop.shape[1] > max_size
        or prot_prop.shape[0] > max_size
        or prot_prop.shape[1] > max_size
    ):
        logger.warning(
            f"Warning: Large archetype proportion matrices. Limiting to {max_size} rows/columns for visualization."
        )

        # Subsample if needed
        if rna_prop.shape[0] > max_size:
            rna_prop = rna_prop.iloc[:max_size, :]
        if rna_prop.shape[1] > max_size:
            rna_prop = rna_prop.iloc[:, :max_size]
        if prot_prop.shape[0] > max_size:
            prot_prop = prot_prop.iloc[:max_size, :]
        if prot_prop.shape[1] > max_size:
            prot_prop = prot_prop.iloc[:, :max_size]

    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(rna_prop, cbar=False)
    plt.xticks()
    plt.title("RNA Archetypes")
    plt.subplot(1, 2, 2)
    plt.title("Protein Archetypes")
    sns.heatmap(prot_prop, cbar=False)
    plt.suptitle("Non-Aligned Archetypes Profiles")
    plt.yticks([])
    plt.show()
    plt.close()


def plot_archetype_weights(
    best_archetype_rna_prop, best_archetype_prot_prop, row_order, max_size=20, plot_flag=True
):
    """Plot archetype weights."""
    if not plot_flag:
        return
    # Ensure matrices aren't too large
    rna_prop = pd.DataFrame(best_archetype_rna_prop)
    prot_prop = pd.DataFrame(best_archetype_prot_prop)

    # If row_order is too large, limit it
    if len(row_order) > max_size:
        logger.warning(
            f"Warning: Limiting row_order to first {max_size} elements for visualization"
        )
        row_order = row_order[:max_size]

    # If matrices are too large, subsample them
    if (
        rna_prop.shape[0] > max_size
        or rna_prop.shape[1] > max_size
        or prot_prop.shape[0] > max_size
        or prot_prop.shape[1] > max_size
    ):
        logger.warning(
            f"Warning: Large archetype weight matrices. Limiting to {max_size} rows/columns."
        )

        # Limit rows based on row_order if possible
        if rna_prop.shape[0] > max_size:
            # If row_order is already limited, use it directly
            if len(row_order) <= max_size:
                rna_prop = rna_prop.iloc[row_order, :]
                prot_prop = prot_prop.iloc[row_order, :]
            else:
                # Otherwise use the first max_size rows
                rna_prop = rna_prop.iloc[:max_size, :]
                prot_prop = prot_prop.iloc[:max_size, :]
                row_order = row_order[:max_size]

        # Limit columns if needed
        if rna_prop.shape[1] > max_size:
            rna_prop = rna_prop.iloc[:, :max_size]
        if prot_prop.shape[1] > max_size:
            prot_prop = prot_prop.iloc[:, :max_size]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("RNA Archetype Weights vs Cell Types")
    plt.ylabel("Archetypes")
    sns.heatmap(rna_prop.iloc[row_order], cbar=False)
    plt.yticks([])
    plt.ylabel("Archetypes")
    plt.subplot(1, 2, 2)
    plt.ylabel("Archetypes")
    plt.title("Protein Archetype Weights vs Cell Types")
    sns.heatmap(prot_prop.iloc[row_order], cbar=False)
    plt.ylabel("Archetypes")
    plt.suptitle(
        "Archetype Weight Distribution Across Cell Types (Higher Similarity = Better Alignment)"
    )
    plt.yticks([])
    plt.xticks(rotation=45)
    plt.show()
    plt.close()


def plot_archetype_visualizations(
    adata_archetype_rna,
    adata_archetype_prot,
    adata_1_rna,
    adata_2_prot,
    max_cells=2000,
    plot_flag=True,
):
    """Plot archetype visualizations."""
    if not plot_flag:
        return
    # Apply subsampling if datasets are too large
    if adata_archetype_rna.shape[0] > max_cells:
        logger.info(f"Subsampling RNA archetype data to {max_cells} cells for visualization")
        rna_arch_plot = sc.pp.subsample(adata_archetype_rna, n_obs=max_cells, copy=True)
        # log1p_rna imported at top

        log1p_rna(rna_arch_plot)
    else:
        rna_arch_plot = adata_archetype_rna.copy()

    if adata_archetype_prot.shape[0] > max_cells:
        logger.info(f"Subsampling protein archetype data to {max_cells} cells for visualization")
        prot_arch_plot = sc.pp.subsample(adata_archetype_prot, n_obs=max_cells, copy=True)
    else:
        prot_arch_plot = adata_archetype_prot.copy()

    if adata_1_rna.shape[0] > max_cells:
        logger.info(f"Subsampling RNA data to {max_cells} cells for visualization")
        rna_plot = sc.pp.subsample(adata_1_rna, n_obs=max_cells, copy=True)
    else:
        rna_plot = adata_1_rna.copy()

    if adata_2_prot.shape[0] > max_cells:
        logger.info(f"Subsampling protein data to {max_cells} cells for visualization")
        prot_plot = sc.pp.subsample(adata_2_prot, n_obs=max_cells, copy=True)
    else:
        prot_plot = adata_2_prot.copy()

    # Preserve consistent color information in all plot objects
    if "cell_types_colors" in adata_1_rna.uns:
        rna_arch_plot.uns["cell_types_colors"] = adata_1_rna.uns["cell_types_colors"]
        rna_plot.uns["cell_types_colors"] = adata_1_rna.uns["cell_types_colors"]
    if "major_cell_types_colors" in adata_1_rna.uns:
        rna_arch_plot.uns["major_cell_types_colors"] = adata_1_rna.uns["major_cell_types_colors"]
        rna_plot.uns["major_cell_types_colors"] = adata_1_rna.uns["major_cell_types_colors"]

    if "cell_types_colors" in adata_2_prot.uns:
        prot_arch_plot.uns["cell_types_colors"] = adata_2_prot.uns["cell_types_colors"]
        prot_plot.uns["cell_types_colors"] = adata_2_prot.uns["cell_types_colors"]
    if "major_cell_types_colors" in adata_2_prot.uns:
        prot_arch_plot.uns["major_cell_types_colors"] = adata_2_prot.uns["major_cell_types_colors"]
        prot_plot.uns["major_cell_types_colors"] = adata_2_prot.uns["major_cell_types_colors"]

    # Calculate PCA and plot for archetype data with opacity
    from matplotlib.colors import to_rgb

    sc.pp.pca(rna_arch_plot)
    sc.pp.pca(prot_arch_plot)

    # Plot PCA for both modalities with 3 or 4 subplots each
    for adata_plot, modality in [(rna_arch_plot, "RNA"), (prot_arch_plot, "Protein")]:
        # Check if cell_types and major_cell_types are identical
        if "major_cell_types" in adata_plot.obs.columns:
            # Convert to strings to avoid categorical comparison issues
            cell_types_str = adata_plot.obs["cell_types"].astype(str)
            major_cell_types_str = adata_plot.obs["major_cell_types"].astype(str)
            are_identical = (cell_types_str == major_cell_types_str).all()
        else:
            are_identical = True

        # Determine subplot configuration
        if are_identical:
            color_by_list = ["cell_types", "archetype_label", "matched_archetype_weight"]
            title_suffix_list = ["cell types", "archetype label", "matched archetype weight"]
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        else:
            color_by_list = [
                "major_cell_types",
                "archetype_label",
                "cell_types",
                "matched_archetype_weight",
            ]
            title_suffix_list = [
                "major cell types",
                "archetype label",
                "cell types",
                "matched archetype weight",
            ]
            fig, axes = plt.subplots(1, 4, figsize=(24, 5))

        for ax, color_by, title_suffix in zip(axes, color_by_list, title_suffix_list):
            pca_coords = adata_plot.obsm["X_pca"]

            if color_by == "matched_archetype_weight":
                # Plot matched archetype weight as continuous colormap
                if "matched_archetype_weight" in adata_plot.obs.columns:
                    weights = adata_plot.obs["matched_archetype_weight"].to_numpy()
                    scatter = ax.scatter(
                        pca_coords[:, 0],
                        pca_coords[:, 1],
                        c=weights,
                        s=20,
                        cmap="viridis",
                    )
                    plt.colorbar(scatter, ax=ax, label="Matched Archetype Weight")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No matched_archetype_weight",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
            else:
                # Plot categorical with opacity
                color_cat = adata_plot.obs[color_by].astype("category")

                # Get matched archetype weight and compute alphas per cell type
                alphas = compute_opacity_per_cell_type(adata_plot)

                # Get colors
                color_key = f"{color_by}_colors"
                if color_key in adata_plot.uns:
                    colors_list = adata_plot.uns[color_key]
                    color_map = {
                        ct: to_rgb(colors_list[i]) for i, ct in enumerate(color_cat.cat.categories)
                    }
                else:
                    from matplotlib.cm import get_cmap

                    cmap = (
                        get_cmap("tab20")
                        if len(color_cat.cat.categories) > 10
                        else get_cmap("tab10")
                    )
                    color_map = {
                        ct: cmap(i % cmap.N)[:3] for i, ct in enumerate(color_cat.cat.categories)
                    }

                # Plot with opacity
                for ct in color_cat.cat.categories:
                    mask = color_cat == ct
                    if mask.sum() > 0:
                        rgb = color_map[ct]
                        rgba = np.column_stack([np.tile(rgb, (mask.sum(), 1)), alphas[mask]])
                        ax.scatter(
                            pca_coords[mask, 0],
                            pca_coords[mask, 1],
                            c=rgba,
                            s=20,
                            label=str(ct),
                        )

                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"{modality} PCA {title_suffix}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

        plt.tight_layout()
        safe_mlflow_log_figure(fig, f"archetype_{modality.lower()}_pca.pdf")
        plt.show()

    # Calculate neighbors and UMAP for archetype data
    sc.pp.neighbors(rna_arch_plot)
    sc.pp.neighbors(prot_arch_plot)
    sc.tl.umap(rna_arch_plot)
    sc.tl.umap(prot_arch_plot)

    # Plot UMAP for both modalities with 3 or 4 subplots each
    for adata_plot, modality in [(rna_arch_plot, "RNA"), (prot_arch_plot, "Protein")]:
        # Check if cell_types and major_cell_types are identical
        if "major_cell_types" in adata_plot.obs.columns:
            # Convert to strings to avoid categorical comparison issues
            cell_types_str = adata_plot.obs["cell_types"].astype(str)
            major_cell_types_str = adata_plot.obs["major_cell_types"].astype(str)
            are_identical = (cell_types_str == major_cell_types_str).all()
        else:
            are_identical = True

        # Determine subplot configuration
        if are_identical:
            color_by_list = ["cell_types", "archetype_label", "matched_archetype_weight"]
            title_suffix_list = ["cell types", "archetype label", "matched archetype weight"]
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        else:
            color_by_list = [
                "major_cell_types",
                "archetype_label",
                "cell_types",
                "matched_archetype_weight",
            ]
            title_suffix_list = [
                "major cell types",
                "archetype label",
                "cell types",
                "matched archetype weight",
            ]
            fig, axes = plt.subplots(1, 4, figsize=(24, 5))

        for ax, color_by, title_suffix in zip(axes, color_by_list, title_suffix_list):
            umap_coords = adata_plot.obsm["X_umap"]

            if color_by == "matched_archetype_weight":
                # Plot matched archetype weight as continuous colormap
                if "matched_archetype_weight" in adata_plot.obs.columns:
                    weights = adata_plot.obs["matched_archetype_weight"].to_numpy()
                    scatter = ax.scatter(
                        umap_coords[:, 0],
                        umap_coords[:, 1],
                        c=weights,
                        s=20,
                        cmap="viridis",
                    )
                    plt.colorbar(scatter, ax=ax, label="Matched Archetype Weight")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No matched_archetype_weight",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
            else:
                # Plot categorical with opacity
                color_cat = adata_plot.obs[color_by].astype("category")

                # Get matched archetype weight and compute alphas per cell type
                alphas = compute_opacity_per_cell_type(adata_plot)

                # Get colors
                color_key = f"{color_by}_colors"
                if color_key in adata_plot.uns:
                    colors_list = adata_plot.uns[color_key]
                    color_map = {
                        ct: to_rgb(colors_list[i]) for i, ct in enumerate(color_cat.cat.categories)
                    }
                else:
                    from matplotlib.cm import get_cmap

                    cmap = (
                        get_cmap("tab20")
                        if len(color_cat.cat.categories) > 10
                        else get_cmap("tab10")
                    )
                    color_map = {
                        ct: cmap(i % cmap.N)[:3] for i, ct in enumerate(color_cat.cat.categories)
                    }

                # Plot with opacity
                for ct in color_cat.cat.categories:
                    mask = color_cat == ct
                    if mask.sum() > 0:
                        rgb = color_map[ct]
                        rgba = np.column_stack([np.tile(rgb, (mask.sum(), 1)), alphas[mask]])
                        ax.scatter(
                            umap_coords[mask, 0],
                            umap_coords[mask, 1],
                            c=rgba,
                            s=20,
                            label=str(ct),
                        )

                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            ax.set_title(f"{modality} UMAP {title_suffix}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

        plt.tight_layout()
        safe_mlflow_log_figure(fig, f"archetype_{modality.lower()}_umap.pdf")
        plt.show()


def plot_archetype_heatmaps(adata_rna: AnnData, adata_prot: AnnData, subset_size=2000):
    # compute_archetype_distances imported at top

    """Plot heatmaps of archetype coordinates"""
    step_rna = max(1, int(np.ceil(adata_rna.n_obs / subset_size)))
    step_prot = max(1, int(np.ceil(adata_prot.n_obs / subset_size)))
    adata_rna_subset = adata_rna[::step_rna].copy()
    adata_prot_subset = adata_prot[::step_prot].copy()
    archetype_distances_subset = compute_archetype_distances(adata_rna_subset, adata_prot_subset)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(np.log1p(adata_rna_subset.obsm["archetype_vec"].values), cbar=False)
    plt.title("RNA Archetype Vectors")
    plt.ylabel("RNA cell index")
    plt.xlabel("Archetype Betas")
    plt.subplot(1, 2, 2)
    sns.heatmap(np.log1p(adata_prot_subset.obsm["archetype_vec"].values), cbar=False)
    plt.xlabel("Archetype Betas")
    plt.ylabel("Protein cell index")
    plt.title("Protein Archetype Vectors")
    plt.show()
    plt.close()
    # this is the heatmap of the archetype distances
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Archetype Distances")

    sns.heatmap(np.log1p(archetype_distances_subset[::5, ::5].T))
    plt.xlabel("RNA cell index")
    plt.ylabel("Protein cell index")
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    plt.title("minimum Archetype Distances between RNA and Protein cells")
    plt.scatter(
        np.arange(len(archetype_distances_subset.argmin(axis=1))),
        archetype_distances_subset.argmin(axis=1),
        s=1,
        rasterized=True,
    )
    plt.xlabel("RNA cell index")
    plt.ylabel("Protein cell index")
    plt.show()
    plt.close()
    plt.figure(figsize=(10, 5))
    # plot the same as about but only for cell with is_extreme_archetype in obs
    plt.subplot(1, 2, 1)
    sns.heatmap(
        np.log1p(
            adata_rna_subset.obsm["archetype_vec"].values[
                adata_rna_subset.obs["is_extreme_archetype"]
            ]
        ),
        cbar=False,
    )
    plt.title("Extreme RNA Archetype Vectors")
    plt.ylabel("RNA cell index")
    plt.xlabel("Archetype Betas")
    plt.subplot(1, 2, 2)
    sns.heatmap(
        np.log1p(
            adata_prot_subset.obsm["archetype_vec"].values[
                adata_prot_subset.obs["is_extreme_archetype"]
            ]
        ),
        cbar=False,
    )
    plt.xlabel("Archetype Betas")
    plt.ylabel("Protein cell index")
    plt.title("Extreme Protein Archetype Vectors")
    plt.show()
    plt.close()
    # plot the second plot just with the extreme archetypes with the scatter plot
    extreme_mask = adata_rna_subset.obs["is_extreme_archetype"].values
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Archetype Distances (Extreme Only)")

    # Only plot distances for extreme archetypes
    sns.heatmap(np.log1p(archetype_distances_subset[extreme_mask][:, ::5].T))
    plt.xlabel("RNA cell index (extreme)")
    plt.ylabel("Protein cell index")
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    plt.title("Minimum Archetype Distances (Extreme Only)")
    min_indices = archetype_distances_subset[extreme_mask].argmin(axis=1)
    plt.scatter(
        np.arange(len(min_indices)),
        min_indices,
        s=1,
        rasterized=True,
    )
    plt.xlabel("RNA cell index (extreme)")
    plt.ylabel("Protein cell index")
    plt.show()


def plot_extreme_archetypes_alignment(
    rna_latent_mean,
    protein_latent_mean,
    rna_batch,
    protein_batch,
    adata_rna,
    adata_prot,
    rna_indices,
    protein_indices,
    global_step=None,
    alpha=0.6,
    use_subsample=True,
    mode=None,
    plot_flag=True,
):
    """
    Plot extreme vs non-extreme archetypes alignment between modalities.

    Args:
        rna_latent_mean: RNA latent representations
        protein_latent_mean: Protein latent representations
        rna_batch: RNA batch data containing archetype vectors
        protein_batch: Protein batch data containing archetype vectors
        adata_rna: RNA AnnData object
        adata_prot: Protein AnnData object
        rna_indices: Indices of RNA cells in the batch
        protein_indices: Indices of protein cells in the batch
        global_step: Current training step
        alpha: Transparency level
        use_subsample: Whether to subsample for plotting
    """
    if not plot_flag:
        return

    # Get extreme archetype masks using balanced approach
    rna_extreme_mask, _, _ = identify_extreme_archetypes_balanced(
        rna_batch["archetype_vec"], adata_rna[rna_indices].copy(), 90, to_print=False
    )
    prot_extreme_mask, _, _ = identify_extreme_archetypes_balanced(
        protein_batch["archetype_vec"], adata_prot[protein_indices].copy(), 90, to_print=False
    )

    # Convert to numpy for easier manipulation
    rna_latent_np = rna_latent_mean.detach().cpu().numpy()
    protein_latent_np = protein_latent_mean.detach().cpu().numpy()
    rna_extreme_np = rna_extreme_mask.detach().cpu().numpy()
    prot_extreme_np = prot_extreme_mask.detach().cpu().numpy()

    # Subsample if requested
    if use_subsample and len(rna_latent_np) > 1000:
        n_sample = min(1000, len(rna_latent_np))
        rna_sample_idx = np.random.choice(len(rna_latent_np), n_sample, replace=False)
        protein_sample_idx = np.random.choice(len(protein_latent_np), n_sample, replace=False)

        rna_latent_np = rna_latent_np[rna_sample_idx]
        protein_latent_np = protein_latent_np[protein_sample_idx]
        rna_extreme_np = rna_extreme_np[rna_sample_idx]
        prot_extreme_np = prot_extreme_np[protein_sample_idx]
        rna_indices = np.array(rna_indices)[rna_sample_idx]
        protein_indices = np.array(protein_indices)[protein_sample_idx]

    # Combine data for PCA
    combined_latent = np.vstack([rna_latent_np, protein_latent_np])

    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined_latent)

    # Split back into RNA and protein
    n_rna = len(rna_latent_np)
    rna_pca = combined_pca[:n_rna]
    protein_pca = combined_pca[n_rna:]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Colors for modalities
    rna_color = "#1f77b4"  # Blue
    protein_color = "#ff7f0e"  # Orange

    # Plot 1: Combined modalities with connecting lines
    ax = axes[0]

    # Plot RNA cells
    rna_extreme_idx = np.where(rna_extreme_np)[0]
    np.where(~rna_extreme_np)[0]

    # # RNA non-extreme (circle markers)
    # if len(rna_non_extreme_idx) > 0:
    #     ax.scatter(
    #         rna_pca[rna_non_extreme_idx, 0],
    #         rna_pca[rna_non_extreme_idx, 1],
    #         c=rna_color,
    #         marker="o",
    #         s=30,
    #         alpha=alpha,
    #         label="RNA non-extreme",
    #     )

    # RNA extreme (x markers)
    if len(rna_extreme_idx) > 0:
        ax.scatter(
            rna_pca[rna_extreme_idx, 0],
            rna_pca[rna_extreme_idx, 1],
            c=rna_color,
            marker="x",
            s=80,
            alpha=1.0,
            label="RNA extreme",
        )

    # Plot Protein cells
    prot_extreme_idx = np.where(prot_extreme_np)[0]
    np.where(~prot_extreme_np)[0]

    # Protein non-extreme (circle markers)
    # if len(prot_non_extreme_idx) > 0:
    #     ax.scatter(
    #         protein_pca[prot_non_extreme_idx, 0],
    #         protein_pca[prot_non_extreme_idx, 1],
    #         c=protein_color,
    #         marker="o",
    #         s=30,
    #         alpha=alpha,
    #         label="Protein non-extreme",
    #     )

    # Protein extreme (x markers)
    if len(prot_extreme_idx) > 0:
        ax.scatter(
            protein_pca[prot_extreme_idx, 0],
            protein_pca[prot_extreme_idx, 1],
            c=protein_color,
            marker="x",
            s=80,
            alpha=1.0,
            label="Protein extreme",
        )

    # Draw connecting lines between extreme archetypes
    if len(rna_extreme_idx) > 0 and len(prot_extreme_idx) > 0:
        # Calculate distances between extreme cells in archetype space
        rna_extreme_archetypes = rna_batch["archetype_vec"][rna_extreme_idx]
        prot_extreme_archetypes = protein_batch["archetype_vec"][prot_extreme_idx]

        if len(rna_extreme_archetypes) > 0 and len(prot_extreme_archetypes) > 0:
            archetype_distances = torch.cdist(rna_extreme_archetypes, prot_extreme_archetypes)

            # Connect each RNA extreme to its closest protein extreme
            closest_matches = torch.argmin(archetype_distances, dim=1)

            for i, match_idx in enumerate(closest_matches):
                if i < len(rna_extreme_idx) and match_idx < len(prot_extreme_idx):
                    rna_pos = rna_pca[rna_extreme_idx[i]]
                    prot_pos = protein_pca[prot_extreme_idx[match_idx]]
                    ax.plot(
                        [rna_pos[0], prot_pos[0]],
                        [rna_pos[1], prot_pos[1]],
                        "gray",
                        alpha=0.3,
                        linewidth=0.5,
                    )

    ax.set_title(f"Combined Modalities - Extreme Archetype Alignment\nStep {global_step}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")

    # Plot 2: RNA only
    ax = axes[1]

    rna_cell_types = adata_rna.obs["cell_types"].cat.codes
    unique_cell_types = adata_rna.obs["cell_types"].cat.categories
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cell_types)))

    if len(rna_extreme_idx) > 0:
        scatter = ax.scatter(
            rna_pca[rna_extreme_idx, 0],
            rna_pca[rna_extreme_idx, 1],
            c=rna_cell_types[rna_extreme_idx],
            marker="x",
            s=80,
            alpha=1.0,
            cmap=plt.cm.tab10,
            vmin=0,
            vmax=len(unique_cell_types) - 1,
        )

    # Create legend for cell types using the same colors as the scatter plot
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="x",
            color="w",
            markeredgecolor=colors[i],
            markerfacecolor="none",
            markersize=8,
            markeredgewidth=2,
            label=cell_type,
        )
        for i, cell_type in enumerate(unique_cell_types)
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(f"RNA Modality Only\n{len(rna_extreme_idx)}/{len(rna_latent_np)} extreme")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")

    # Plot 3: Protein only
    ax = axes[2]

    prot_cell_types = adata_prot.obs["cell_types"].cat.codes
    unique_cell_types_prot = adata_prot.obs["cell_types"].cat.categories
    colors_prot = plt.cm.tab10(np.linspace(0, 1, len(unique_cell_types_prot)))

    if len(prot_extreme_idx) > 0:
        scatter = ax.scatter(
            protein_pca[prot_extreme_idx, 0],
            protein_pca[prot_extreme_idx, 1],
            c=prot_cell_types[prot_extreme_idx],
            marker="x",
            s=80,
            alpha=1.0,
            cmap=plt.cm.tab10,
            vmin=0,
            vmax=len(unique_cell_types_prot) - 1,
        )

    # Create legend for cell types using the same colors as the scatter plot
    legend_elements_prot = [
        Line2D(
            [0],
            [0],
            marker="x",
            color="w",
            markeredgecolor=colors_prot[i],
            markerfacecolor="none",
            markersize=8,
            markeredgewidth=2,
            label=cell_type,
        )
        for i, cell_type in enumerate(unique_cell_types_prot)
    ]
    ax.legend(handles=legend_elements_prot, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(f"Protein Modality Only\n{len(prot_extreme_idx)}/{len(protein_latent_np)} extreme")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")

    # Add single global legend for marker types at the bottom center
    marker_legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            linestyle="None",
            markersize=6,
            alpha=0.6,
            label="Normal cells",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="gray",
            linestyle="None",
            markersize=8,
            alpha=1.0,
            label="Extreme archetypes",
        ),
    ]
    fig.legend(
        handles=marker_legend_elements,
        bbox_to_anchor=(0.5, 0.02),
        loc="lower center",
        fontsize=10,
        title="Markers",
        ncol=2,
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])  # Leave space for bottom legend

    # Save and log to MLflow
    if global_step is not None:
        plot_name = f"extreme_archetypes_alignment_step_{global_step:05d}.pdf"
    else:
        plot_name = "extreme_archetypes_alignment.pdf"

    if mode == "training":
        plot_name = f"train_{plot_name}"
    elif mode == "validation":
        plot_name = f"val_{plot_name}"
    else:
        pass
    safe_mlflow_log_figure(fig, plot_name)


def plot_batch_archetype_visualization(
    adata,
    batch_name,
    modality_name,
    archetype_embedding_name="X_pca",
    plot_flag=True,
    max_points=2000,
):
    """
    Plot archetype visualization for a specific batch.

    Args:
        adata: AnnData object for the full dataset
        batch_name: Name of the batch to visualize
        modality_name: Name of the modality (e.g., "RNA", "Protein")
        archetype_embedding_name: Name of the embedding to use for visualization
        plot_flag: Whether to create plots
        max_points: Maximum number of points to plot for visualization
    """
    if not plot_flag:
        return

    from matplotlib.colors import to_hex

    from arcadia.archetypes import visualization as archetype_visualization

    print(f"\nVisualizing {modality_name} batch: {batch_name}")
    batch_mask = adata.obs["batch"] == batch_name
    adata_batch = adata[batch_mask].copy()

    if len(adata_batch) > 0 and f"batch_archetypes_{batch_name}" in adata.uns:
        # Use batch-specific archetype positions (already reordered for cross-modal consistency)
        batch_archetypes_full = adata.uns[f"batch_archetypes_{batch_name}"]
        batch_cell_weights = adata_batch.obsm["archetype_vec"]

        # Ensure cell weights are numpy array (not DataFrame)
        if hasattr(batch_cell_weights, "values"):
            batch_cell_weights = batch_cell_weights.values

        # Use pre-computed archetype labels
        batch_cell_types = adata_batch.obs["cell_types"].tolist()
        batch_archetype_indices = adata_batch.obs["archetype_label"].tolist()

        # Get extreme archetype mask if available
        extreme_mask = None
        if "is_extreme_archetype" in adata_batch.obs.columns:
            extreme_mask = adata_batch.obs["is_extreme_archetype"].values

        # Get cells matched arch weight if available
        matched_archetype_weight = None
        if "matched_archetype_weight" in adata_batch.obs.columns:
            matched_archetype_weight = adata_batch.obs["matched_archetype_weight"].to_numpy()

        # Get archetype quality dictionary if available
        archetype_quality_dict = adata.uns.get("archetype_quality", None)

        # Extract cell type color dictionary from adata.uns
        cell_type_colors = None
        if "cell_types_colors" in adata.uns:
            cell_types_categories = adata.obs["cell_types"].cat.categories
            colors_list = adata.uns["cell_types_colors"]
            cell_type_colors = {
                ct: (
                    to_hex(colors_list[i])
                    if not isinstance(colors_list[i], str)
                    else colors_list[i]
                )
                for i, ct in enumerate(cell_types_categories)
            }

        archetype_visualization.plot_archetypes(
            data_points=adata_batch.obsm[archetype_embedding_name],
            archetype=batch_archetypes_full,
            samples_cell_types=batch_cell_types,
            data_point_archetype_indices=batch_archetype_indices,
            modality=f"{modality_name} Batch {batch_name}",
            cell_type_colors=cell_type_colors,
            max_points=max_points,
            extreme_archetype_mask=extreme_mask,
            matched_archetype_weight=matched_archetype_weight,
            plot_pca=True,
            plot_umap=True,
            plot_flag=plot_flag,
            archetype_quality_dict=archetype_quality_dict,
        )

        print(f"✅ Successfully visualized {modality_name} batch {batch_name}")
    else:
        print(
            f"❌ Could not visualize {modality_name} batch {batch_name}: missing data or archetypes"
        )
