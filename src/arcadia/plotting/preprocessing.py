"""Preprocessing-related plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.sparse import issparse

from arcadia.data_utils.preprocessing import analyze_and_visualize, log1p_rna, qc_metrics


def plot_count_distribution(adata_rna_new, plot_flag=True):
    """Plot histogram of count distribution"""
    if issparse(adata_rna_new.X):
        data = adata_rna_new.X.data
        if data.shape[0] > 5000:
            sample = np.random.choice(data, size=5000, replace=False)
        else:
            sample = data
    else:
        flat = adata_rna_new.X.flatten()
        nonzero_flat = flat[flat > 0]
        if nonzero_flat.shape[0] > 5000:
            sample = np.random.choice(nonzero_flat, size=5000, replace=False)
        else:
            sample = nonzero_flat
    if not plot_flag:
        return

    plt.figure(figsize=(6, 4))
    sns.histplot(sample, kde=False)
    plt.yscale("log")
    plt.title("Histogram of 5k subsample of recovered counts non zero")
    plt.xlabel("Recovered count value")
    plt.ylabel("Frequency (log scale)")
    plt.tight_layout()

    plt.show()
    plt.close()


def plot_expression_heatmap(adata, plot_flag=True):
    """Plot heatmap of expression data"""
    if not plot_flag:
        return

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        adata.X.toarray()[:1000, :1000] if issparse(adata.X) else adata.X[:1000, :1000],
        cmap="viridis",
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Expression"},
    )
    plt.title("Expression Heatmap")

    plt.show()
    plt.close()


def plot_merged_dataset_analysis(adata_rna, plot_flag=True, subsample_size=2000):
    """Plot analysis of merged dataset before preprocessing"""
    if not plot_flag:
        return
    # used scanpy to subsample the data
    adata_rna_subsampled = adata_rna.copy()
    sc.pp.subsample(adata_rna_subsampled, n_obs=min(subsample_size, adata_rna.n_obs))

    plt.figure(figsize=(10, 10))
    if issparse(adata_rna_subsampled.X):
        sns.heatmap(adata_rna_subsampled.X.toarray()[:100, :100], cmap="viridis")
    else:
        sns.heatmap(adata_rna_subsampled.X[:100, :100], cmap="viridis")
    plt.show()
    plt.close()
    sc.pp.pca(adata_rna_subsampled, copy=False)
    sc.pl.pca(adata_rna_subsampled, color="batch", save="_pca_batch", show=True)
    sc.pp.neighbors(adata_rna_subsampled, use_rep="X_pca")
    sc.tl.umap(adata_rna_subsampled)
    sc.pl.umap(
        adata_rna_subsampled,
        color="batch",
        title="merged dataset before preprocessing",
        save="_umap_batch",
        show=True,
    )


def plot_protein_analysis(adata_prot, subsample_size=2000, plot_flag=True, modality=""):
    """Plot protein data analysis"""
    if not plot_flag:
        return

    adata_prot_subsampled = adata_prot.copy()
    sc.pp.subsample(adata_prot_subsampled, n_obs=min(subsample_size, adata_prot.n_obs))
    analyze_and_visualize(adata_prot_subsampled, modality=modality, plot_flag=plot_flag)
    qc_metrics(adata_prot_subsampled, plot_flag=plot_flag)

    # Use total_counts if nCount_CODEX doesn't exist
    count_col = (
        "nCount_CODEX" if "nCount_CODEX" in adata_prot_subsampled.obs.columns else "total_counts"
    )

    if count_col in adata_prot_subsampled.obs.columns:
        sc.pl.violin(
            adata_prot_subsampled,
            count_col,
            groupby="cell_types",
            rotation=90,
            save="_violin_cell_types",
            show=True,
        )
    else:
        print(
            f"Warning: Neither 'nCount_CODEX' nor 'total_counts' found in adata.obs. Skipping violin plot."
        )


def plot_protein_violin(adata_prot, plot_flag=True):
    """Plot protein violin plots"""
    if not plot_flag:
        return

    # Use total_counts if nCount_CODEX doesn't exist
    count_col = "nCount_CODEX" if "nCount_CODEX" in adata_prot.obs.columns else "total_counts"

    if count_col not in adata_prot.obs.columns:
        print(
            f"Warning: Neither 'nCount_CODEX' nor 'total_counts' found in adata.obs. Skipping violin plot."
        )
        return

    sc.pl.violin(
        adata_prot,
        count_col,
        groupby="cell_types",
        rotation=90,
        save="_protein_violin",
        show=True,
    )


def plot_original_data_umaps(adata_rna, adata_prot, plot_flag=True):
    """Plot UMAP of original RNA and protein data"""
    if not plot_flag:
        return
    # use subsample to speed up the plotting
    adata_rna_subsampled = adata_rna.copy()
    adata_prot_subsampled = adata_prot.copy()
    sc.pp.subsample(adata_rna_subsampled, n_obs=min(2000, adata_rna.n_obs))
    sc.pp.subsample(adata_prot_subsampled, n_obs=min(2000, adata_prot.n_obs))
    sc.pp.pca(adata_rna_subsampled, copy=False)
    sc.pp.neighbors(adata_rna_subsampled)
    sc.tl.umap(adata_rna_subsampled)
    sc.pl.umap(
        adata_rna_subsampled,
        color="cell_types",
        title="original RNA data",
        save="_rna_cell_types",
        show=True,
    )
    sc.pp.pca(adata_prot_subsampled, copy=False)
    sc.pp.neighbors(adata_prot_subsampled)
    sc.tl.umap(adata_prot_subsampled)
    sc.pl.umap(
        adata_prot_subsampled,
        color="cell_types",
        title="original protein data",
        save="_protein_cell_types",
        show=True,
    )

    from arcadia.plotting.general import plot_cell_type_distribution, plot_data_overview
    from arcadia.plotting.spatial import plot_spatial_data

    plot_data_overview(adata_rna_subsampled, adata_prot_subsampled, plot_flag=plot_flag)
    plot_cell_type_distribution(adata_rna_subsampled, adata_prot_subsampled, plot_flag=plot_flag)
    plot_spatial_data(adata_prot_subsampled, plot_flag=plot_flag)

    if "batch" in adata_rna.obs:
        sc.pl.umap(adata_prot_subsampled, color="batch", save="_rna_batch", show=True)


def plot_variance_analysis_raw(adata_rna, plot_flag=True):
    """Plot variance vs mean expression for raw data"""
    if not plot_flag:
        return

    adata_rna_temp = adata_rna.copy()
    batch_key = "batch" if "batch" in adata_rna_temp.obs else None
    sc.pp.highly_variable_genes(
        adata_rna_temp, n_top_genes=2000, batch_key=batch_key, flavor="seurat"
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(
        adata_rna_temp.var["means"],
        adata_rna_temp.var["variances"],
        alpha=0.3,
        label="All genes",
    )
    plt.scatter(
        adata_rna_temp.var["means"][adata_rna_temp.var["highly_variable"]],
        adata_rna_temp.var["variances"][adata_rna_temp.var["highly_variable"]],
        color="red",
        label="Highly variable genes",
    )
    plt.xlabel("Mean expression")
    plt.ylabel("Variance")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title("Raw data - Variance vs. Mean Expression of Genes")

    plt.show()
    plt.close()


def plot_gene_variance_elbow(variances_sorted, plot_flag=True):
    """Plot elbow plot of gene variances"""
    if not plot_flag:
        return

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(variances_sorted) + 1), variances_sorted)
    plt.xlabel("Gene rank")
    plt.ylabel("Variance")
    plt.yscale("log")
    plt.title("Elbow plot of Gene Variances")
    plt.axvline(x=1000, color="red", linestyle="dashed", label="n_top_genes=1000")
    plt.legend()

    plt.show()
    plt.close()


def plot_kneedle_analysis(kneedle, plot_flag=True):
    """Plot kneedle analysis"""
    if not plot_flag:
        return

    kneedle.plot_knee()


def plot_variance_analysis_processed(adata_rna, plot_flag=True):
    """Plot variance vs mean expression for processed data"""
    if not plot_flag:
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(adata_rna.var["means"], adata_rna.var["variances"], alpha=0.3, label="All genes")
    plt.scatter(
        adata_rna.var["means"][adata_rna.var["highly_variable"]],
        adata_rna.var["variances"][adata_rna.var["highly_variable"]],
        color="red",
        label="Highly variable genes",
    )
    plt.xlabel("Mean expression")
    plt.ylabel("Variance")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title("Processed data - Variance vs. Mean Expression of Genes")

    plt.show()
    plt.close()


def plot_batch_correction_before(adata_rna, plot_flag=True):
    """Plot batch correction analysis before correction"""
    if not plot_flag:
        return
    if "batch" not in adata_rna.obs.columns:
        print("Warning: No batch information found, skipping batch correction plots")
        return
    if adata_rna.obs["batch"].nunique() == 1:
        print("Warning: there is only one batch, skipping batch correction plots")
        return

    sc.pp.pca(adata_rna, copy=False)
    sc.pp.neighbors(adata_rna, copy=False)
    sc.tl.umap(adata_rna)
    sc.pl.umap(
        adata_rna,
        color="batch",
        title="RNA: Dataset Source before batch correction",
        save="_batch_before",
        show=True,
    )
    sc.pl.umap(
        adata_rna,
        color="cell_types",
        title="RNA: Cell Types before batch correction",
        save="_cell_types_before",
        show=True,
    )


def plot_batch_correction_after(
    adata_rna,
    adata_prot,
    batch_distance_before,
    batch_distance_after,
    pca_distance,
    improvement,
    plot_flag=True,
):
    """Plot batch correction analysis after correction"""
    if not plot_flag:
        return

    # Check batch separation in UMAP space
    # Use X_umap if available, otherwise X_scANVI_umap, otherwise skip
    umap_key = None
    if "X_umap" in adata_rna.obsm:
        umap_key = "X_umap"
    elif "X_scANVI_umap" in adata_rna.obsm:
        umap_key = "X_scANVI_umap"

    if umap_key is not None:
        batch_mask_old = adata_rna.obs["batch"] == "old"
        batch_mask_new = adata_rna.obs["batch"] == "new"
        if batch_mask_old.any() and batch_mask_new.any():
            umap_batch1 = adata_rna.obsm[umap_key][batch_mask_old]
            umap_batch2 = adata_rna.obsm[umap_key][batch_mask_new]
            umap_distance = np.linalg.norm(umap_batch1.mean(axis=0) - umap_batch2.mean(axis=0))
            print(f"DEBUG: Batch separation in UMAP space: {umap_distance:.4f}")
        else:
            umap_distance = np.nan
            print(
                "DEBUG: Only one batch found or batches not named 'old'/'new', skipping UMAP batch separation calculation"
            )
    else:
        umap_distance = np.nan
        print("DEBUG: No UMAP embedding found, skipping UMAP batch separation calculation")

    print("📊 DEBUG: Final batch correction summary:")
    print(f"  - Original batch separation: {batch_distance_before:.4f}")
    print(f"  - After correction: {batch_distance_after:.4f}")
    print(f"  - PCA space separation: {pca_distance:.4f}")
    if not np.isnan(umap_distance):
        print(f"  - UMAP space separation: {umap_distance:.4f}")
    print(f"  - Overall improvement: {improvement:.1f}%")

    # Use available UMAP embedding or compute if needed
    if "X_umap" not in adata_rna.obsm:
        sc.pp.neighbors(adata_rna, copy=False)
        sc.tl.umap(adata_rna)

    sc.pl.umap(
        adata_rna,
        color="cell_types",
        title="RNA: Cell Types (Batch Corrected)",
        save="_cell_types_after",
        show=True,
    )
    # same for protein
    sc.pp.neighbors(adata_prot, copy=False)  # Compute the neighbors needed for UMAP
    sc.tl.umap(adata_prot)  # Calculate UMAP coordinates
    sc.pl.umap(
        adata_prot,
        color="cell_types",
        title="Protein: Cell Types",
        save="_protein_cell_types_after",
        show=True,
    )

    # Only plot batch if there are multiple batches
    if "batch" in adata_rna.obs and adata_rna.obs["batch"].nunique() > 1:
        sc.pl.umap(
            adata_rna,
            color="batch",
            title="RNA: Dataset Source (Should Be Mixed!)",
            save="_batch_after",
            show=True,
        )


def plot_spatial_locations(adata_prot, plot_flag=True):
    """Plot spatial locations of cells"""
    if not plot_flag:
        return

    sc.pl.scatter(
        adata_prot, x="X", y="Y", color="cell_types", title="cell types with spatial locations"
    )


def plot_expression_heatmaps(adata_rna, adata_prot, plot_flag=True):
    """Plot expression heatmaps for RNA and protein data"""
    if not plot_flag:
        return

    num_cells = min(1000, adata_rna.n_obs, adata_prot.n_obs)
    random_indices_protein = np.random.choice(adata_prot.n_obs, num_cells, replace=False)
    random_indices_rna = np.random.choice(adata_rna.n_obs, num_cells, replace=False)
    protein_data = adata_prot.X[random_indices_protein, :]
    sns.heatmap(
        protein_data.todense() if issparse(protein_data) else protein_data,
        xticklabels=False,
        yticklabels=False,
    )
    plt.title("Protein Expression Heatmap (Random 100 Cells)")

    plt.show()
    plt.close()
    rna_data = (
        adata_rna.X[random_indices_rna, :].todense()
        if issparse(adata_rna.X)
        else adata_rna.X[random_indices_rna, :]
    )
    sns.heatmap(rna_data, xticklabels=False, yticklabels=False)
    plt.title("RNA Expression Heatmap (Random 100 Cells)")

    plt.show()
    plt.close()


def plot_final_alignment_results(adata_rna, adata_prot, plot_flag=True):
    """Plot final preprocessing results"""
    if not plot_flag:
        return

    # Compute UMAP for both datasets
    adata_rna_to_plot = adata_rna.copy()
    adata_prot_to_plot = adata_prot.copy()
    sc.pp.subsample(adata_rna_to_plot, n_obs=min(2000, adata_rna.n_obs))
    sc.pp.subsample(adata_prot_to_plot, n_obs=min(2000, adata_prot.n_obs))

    log1p_rna(adata_rna_to_plot)
    sc.pp.pca(adata_rna_to_plot, copy=False)
    sc.pp.neighbors(adata_rna_to_plot, copy=False)
    sc.tl.umap(adata_rna_to_plot)
    sc.pp.pca(adata_prot_to_plot, copy=False)
    sc.pp.neighbors(adata_prot_to_plot, copy=False)
    sc.tl.umap(adata_prot_to_plot)
    from arcadia.plotting.general import plot_preprocessing_results

    plot_preprocessing_results(adata_rna_to_plot, adata_prot_to_plot, plot_flag=plot_flag)
    if "batch" in adata_rna.obs and adata_rna.obs["batch"].nunique() > 1:
        pass  # Batch visualization handled by other plots


def plot_spatial_distance_hist(
    adata_prot, key="spatial_neighbors_distances", title=None, plot_flag=True
):
    if not plot_flag:
        return
    data = adata_prot.obsp[key].data
    # subsample the data randomly
    subsample_size = min(len(data), 2000)
    data = data[np.random.choice(len(data), subsample_size, replace=False)]
    sns.histplot(data)
    if title is not None:
        plt.title(title)

    plt.show()
    plt.close()


def plot_empirical_vs_annotated_cns(adata_prot, resolution=None, plot_flag=True):
    if (not plot_flag) or "lab_CN" not in adata_prot.obs or resolution is None:
        return
    temp_empirical = adata_prot.copy()
    # Ensure neighbor graph exists for Leiden
    if "neighbors" not in temp_empirical.obsp:
        sc.pp.neighbors(temp_empirical)
    sc.tl.leiden(temp_empirical, resolution=resolution, key_added="CN")
    temp_empirical.obs["CN"] = pd.Categorical(
        [f"CN_{cn}" for cn in temp_empirical.obs["CN"]],
        categories=sorted([f"CN_{i}" for i in range(len(temp_empirical.obs["CN"].unique()))]),
    )

    temp_annotated = adata_prot.copy()
    temp_annotated.obs["CN"] = pd.Categorical(temp_annotated.obs["lab_CN"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    spatial_coords = temp_empirical.obsm["spatial_location"].to_numpy()
    scatter1 = ax1.scatter(
        spatial_coords[:, 0],
        spatial_coords[:, 1],
        c=pd.Categorical(temp_empirical.obs["CN"]).codes,
        cmap="tab20",
        alpha=0.6,
    )
    ax1.set_title("Empirical CNs")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    plt.colorbar(scatter1, ax=ax1, label="Empirical CN")

    spatial_coords = temp_annotated.obsm["spatial_location"].to_numpy()
    unique_ann_cns = temp_annotated.obs["CN"].cat.categories
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ann_cns)))
    cn_to_color = dict(zip(unique_ann_cns, colors))
    point_colors = [cn_to_color[cn] for cn in temp_annotated.obs["CN"]]
    ax2.scatter(spatial_coords[:, 0], spatial_coords[:, 1], c=point_colors, alpha=0.6)
    ax2.set_title("Annotated CNs")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    legend_elements = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=cn_to_color[cn], markersize=8, label=cn
        )
        for cn in unique_ann_cns
    ]
    ax2.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()

    plt.show()
    plt.close()


def plot_feature_type_heatmap(adata, plot_flag=True):
    if not plot_flag:
        return
    plt.figure(figsize=(10, 8))
    sns.heatmap(adata.X[:1000, :1000] if not issparse(adata.X) else adata.X[:1000, :1000].toarray())
    plt.title("Protein data with all features types")
    plt.xlabel("Features")
    plt.ylabel("Cells")
    feature_types = adata.var["feature_type"]
    unique_types = feature_types.unique()
    tick_positions = []
    tick_labels = []
    for feature_type in unique_types:
        type_mask = feature_types == feature_type
        type_indices = np.where(type_mask)[0]
        if len(type_indices) > 0:
            start_idx = type_indices[0]
            tick_positions.append(start_idx)
            tick_labels.append(f"{adata.var_names[start_idx]} ({feature_type})")
    plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=8)

    plt.show()
    plt.close()


def plot_original_embeddings(adata_rna, adata_prot, plot_flag=True):
    if not plot_flag:
        return
    sc.pl.embedding(
        adata_rna,
        basis="X_original_pca",
        color=["cell_types", "major_cell_types"],
        title=["Original RNA pca minor cell types", "Original RNA pca major cell types"],
    )
    sc.pl.embedding(
        adata_prot,
        basis="X_original_pca",
        color=["cell_types", "major_cell_types"],
        title=["Original Protein pca minor cell types", "Original Protein pca major cell types"],
    )
    sc.pl.embedding(
        adata_rna,
        basis="X_original_umap",
        color=["major_cell_types", "cell_types"],
        title=["Original RNA UMAP major cell types", "Original RNA UMAP minor cell types"],
    )
    sc.pl.embedding(
        adata_prot,
        basis="X_original_umap",
        color=["major_cell_types", "cell_types"],
        title=["Original Protein UMAP major cell types", "Original Protein UMAP minor cell types"],
    )


def plot_original_vs_new_protein_umap(
    adata_prot, original_key="X_original_umap", new_key="X_umap", plot_flag=True, save_flag=False
):
    if not plot_flag:
        return
    import time

    # use a subsample to speed up the plotting
    adata_prot_subsampled = sc.pp.subsample(
        adata_prot, n_obs=min(2000, adata_prot.n_obs), copy=True
    )
    scale_factor = adata_prot.uns["pipeline_metadata"].get("scale_factor")
    colors = ["cell_types"]
    if "batch" in adata_prot.obs:
        colors.append("batch")
    if "CN" in adata_prot.obs:
        colors.append("CN")
    sc.pl.embedding(
        adata_prot_subsampled,
        basis="X_original_umap",
        color=colors,
        title=["Original Protein UMAP {}".format(c) for c in colors],
        show=False,
    )
    if save_flag:
        plt.savefig(
            f"original_protein_umap_{original_key}_scale_factor_{scale_factor}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        )
    plt.show()

    sc.pl.embedding(
        adata_prot_subsampled,
        basis="X_umap",
        color=colors,
        title=["New Protein UMAP {}".format(c) for c in colors],
        show=False,
    )
    if save_flag:
        plt.savefig(
            f"new_protein_umap_{new_key}_scale_factor_{scale_factor}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        )
    plt.show()


def plot_pca_feature_contributions(adata_rna, adata_prot, plot_flag=True):
    if not plot_flag:
        return
    rna_pca_components = adata_rna.varm["PCs"]
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        rna_pca_components,
        cmap="viridis",
        center=0,
        xticklabels=range(1, rna_pca_components.shape[1] + 1),
        yticklabels=False,
    )
    plt.title("RNA: Feature Contributions to PCA Dimensions")
    plt.xlabel("PCA Dimensions")
    plt.ylabel("Original Features")

    plt.show()
    plt.close()

    feature_total_contribution = np.abs(rna_pca_components).sum(axis=1)
    half_point = len(feature_total_contribution) // 2
    first_half_contrib = feature_total_contribution[:half_point].sum()
    second_half_contrib = feature_total_contribution[half_point:].sum()
    print("RNA PCA feature contribution balance:")
    print(f"First half contribution: {first_half_contrib:.2f}")
    print(f"Second half contribution: {second_half_contrib:.2f}")
    print(f"Ratio (first:second): {first_half_contrib/second_half_contrib:.2f}")

    prot_pca_components = adata_prot.varm["PCs"]
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        prot_pca_components,
        cmap="viridis",
        center=0,
        xticklabels=range(1, prot_pca_components.shape[1] + 1),
        yticklabels=False,
    )
    plt.title("Protein: Feature Contributions to PCA Dimensions")
    plt.xlabel("PCA Dimensions")
    plt.ylabel("Original Features")

    plt.show()
    plt.close()

    feature_total_contribution = np.abs(prot_pca_components).sum(axis=1)
    half_point = len(feature_total_contribution) // 2
    first_half_contrib = feature_total_contribution[:half_point].sum()
    second_half_contrib = feature_total_contribution[half_point:].sum()
    print("Protein PCA feature contribution balance:")
    print(f"First half contribution: {first_half_contrib:.2f}")
    print(f"Second half contribution: {second_half_contrib:.2f}")
    print(f"Ratio (first:second): {first_half_contrib/second_half_contrib:.2f}")

    if "feature_type" in adata_prot.var:
        feature_types = adata_prot.var["feature_type"].unique()
        for ft in feature_types:
            mask = adata_prot.var["feature_type"] == ft
            ft_contribution = np.abs(prot_pca_components[mask]).sum()
            print(
                f"Contribution from {ft} features: {ft_contribution:.2f} "
                + f"({ft_contribution/np.abs(prot_pca_components).sum()*100:.2f}%)"
            )


def plot_hvg_and_mean_variance(adata_prot, plot_flag=True):
    if not plot_flag:
        return
    plt.figure(figsize=(10, 6))
    sc.pl.highly_variable_genes(adata_prot, show=True)
    plt.title("Highly Variable Genes")

    plt.show()
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.scatter(
        np.mean(adata_prot.X, axis=0),
        np.var(adata_prot.X, axis=0),
        c=adata_prot.var["highly_variable"],
        cmap="viridis",
        alpha=0.6,
    )
    plt.xlabel("Mean Expression")
    plt.ylabel("Variance")
    plt.title("Mean-Variance Relationship")
    plt.colorbar(label="Highly Variable")

    plt.show()
    plt.close()


def plot_two_umaps(
    adata_left, color_left, title_left, adata_right, color_right, title_right, plot_flag=True
):
    if not plot_flag:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sc.pl.umap(adata_left, color=color_left, ax=axes[0], show=False, title=title_left)
    sc.pl.umap(adata_right, color=color_right, ax=axes[1], show=False, title=title_right)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_extreme_archetype_confusion(
    adata_prot, y_true_extreme, y_pred_extreme, plot_flag=True, save_flag=False
):
    import time

    from sklearn.metrics import f1_score

    if not plot_flag:
        return
    f1_extreme = f1_score(y_true_extreme, y_pred_extreme, average="weighted")

    scale_factor = adata_prot.uns["pipeline_metadata"].get("scale_factor")
    plt.figure(figsize=(12, 10))
    confusion_matrix_extreme = pd.crosstab(
        y_true_extreme,
        y_pred_extreme,
        rownames=["RNA Cell Types"],
        colnames=["Matched Protein Cell Types"],
        margins=False,
    )
    ax = sns.heatmap(
        confusion_matrix_extreme,
        annot=True,
        fmt="d",
        cmap="viridis",
        cbar=False,
        annot_kws={"fontsize": 14, "fontweight": "bold"},
    )
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12, rotation=45)
    plt.tight_layout(pad=1.5)
    plt.title(
        f"Cell Type Matching Using Top 5% Most Extreme Archetypes (Cosine Distance)\nF1 Score: {f1_extreme:.2%}"
    )
    if save_flag:
        plt.savefig(
            f"extreme_archetype_confusion_scale_factor_{scale_factor}_f1_score_{f1_extreme:.2%}_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
        )
    plt.show()
    plt.close()


def plot_umap_analysis(adata_temp, color_key, title, plot_flag=True):
    """Plot UMAP analysis for protein data"""
    if not plot_flag:
        return

    sc.pl.umap(adata_temp, color=color_key, title=title, show=True)
    plt.show()


def plot_batches_and_conditions(adata_subsampled, plot_flag=True, modality=""):
    """Plot batches and conditions analysis for protein data"""
    if not plot_flag:
        return

    modality_prefix = f"{modality} - " if modality else ""
    sc.pl.umap(
        adata_subsampled,
        color="cell_types",
        title=f"{modality_prefix}Cell Types",
        save="cell_types_umap",
        show=True,
    )

    # Only plot condition if it exists in the dataset
    if (
        "condition" in adata_subsampled.obs.columns
        and adata_subsampled.obs["condition"].nunique() > 1
    ):
        sc.pl.umap(
            adata_subsampled,
            color="condition",
            title=f"{modality_prefix}Condition",
            save="condition_umap",
            show=True,
        )
        # Plot by condition
        for condition in adata_subsampled.obs["condition"].unique():
            sc.pl.umap(
                adata_subsampled[adata_subsampled.obs["condition"] == condition],
                color="cell_types",
                title=f"{modality_prefix}{condition} - Cell Types",
                save=f"{condition}_cell_types",
                show=True,
            )

    # Only plot Image if it exists in the dataset
    if "Image" in adata_subsampled.obs.columns and adata_subsampled.obs["Image"].nunique() > 1:
        sc.pl.umap(
            adata_subsampled,
            color="Image",
            title=f"{modality_prefix}Image",
            save="sample_umap",
            show=True,
        )
    if "batch" in adata_subsampled.obs.columns and adata_subsampled.obs["batch"].nunique() > 1:
        sc.pl.umap(
            adata_subsampled,
            color="batch",
            title=f"{modality_prefix}Batch",
            save="batch_umap",
            show=True,
        )
    if "sample" in adata_subsampled.obs.columns and adata_subsampled.obs["sample"].nunique() > 1:
        sc.pl.umap(
            adata_subsampled,
            color="sample",
            title=f"{modality_prefix}Sample",
            save="sample_umap",
            show=True,
        )


def plot_heatmap_analysis(adata_subsampled, modality, plot_flag=True):
    """Plot heatmap analysis for adata"""
    if not plot_flag:
        return

    # Get the actual data dimensions
    n_cells, n_features = adata_subsampled.X.shape

    # Limit the heatmap to a reasonable size (max 1000x1000)
    max_cells = min(1000, n_cells)
    max_features = min(1000, n_features)

    # Extract data for heatmap
    if hasattr(adata_subsampled.X, "toarray"):
        data_for_heatmap = adata_subsampled.X[:max_cells, :max_features].toarray()
    else:
        data_for_heatmap = adata_subsampled.X[:max_cells, :max_features]

    # Only plot if we have valid 2D data
    if data_for_heatmap.ndim == 2 and data_for_heatmap.size > 0:
        sns.heatmap(
            data_for_heatmap,
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
            cbar_kws={"label": "Expression"},
        )
        plt.title(f"{modality} Expression Heatmap ({max_cells} cells × {max_features} features)")

        plt.show()
        plt.close()
    else:
        print(f"Warning: Cannot plot heatmap - data shape: {data_for_heatmap.shape}")


def plot_umap_visualizations_original_data(adata_rna, adata_prot, subset_size=2000):
    """Generate UMAP visualizations for original RNA and protein data"""
    adata_rna_subset = sc.pp.subsample(
        adata_rna, n_obs=min(subset_size, adata_rna.n_obs), copy=True
    ).copy()
    adata_prot_subset = sc.pp.subsample(
        adata_prot, n_obs=min(subset_size, adata_prot.n_obs), copy=True
    ).copy()
    if "connectivities" not in adata_rna_subset.obsm:
        sc.pp.neighbors(adata_rna_subset)
    if "connectivities" not in adata_prot_subset.obsm:
        sc.pp.neighbors(adata_prot_subset)
    sc.tl.umap(adata_rna_subset)
    sc.tl.umap(adata_prot_subset)
    sc.pl.umap(
        adata_rna_subset,
        color=["CN", "cell_types", "archetype_label"],
        title=[
            "RNA exp UMAP, CN",
            "RNA exp UMAP, cell types",
            "RNA exp UMAP, archetype label",
        ],
        show=False,
    )

    sc.pl.umap(
        adata_prot_subset,
        color=["CN", "cell_types", "archetype_label"],
        title=[
            "Protein exp UMAP, CN",
            "Protein exp UMAP, cell types",
            "Protein exp UMAP, archetype label",
        ],
        show=False,
    )
    plt.tight_layout()
    if "batch" in adata_rna_subset.obs:
        sc.pl.umap(
            adata_rna_subset,
            color="batch",
            title="RNA UMAP, batch",
        )
    plt.show()
    plt.close()


def plot_rna_data_histogram(adata_rna, plot_flag=True):
    """Plot histogram of RNA data values."""
    if not plot_flag:
        return
    from scipy.sparse import issparse

    plt.figure()
    hist_rna = (
        adata_rna.X.data[:10000].flatten()
        if issparse(adata_rna.X)
        else adata_rna.X[:, :100].flatten()
    )
    sns.histplot(hist_rna, bins=200)
    plt.yscale("log")
    plt.title("Histogram of RNA data")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    plt.close()


def plot_protein_expression_sorted(adata_prot, n_cells=2000, n_features=50, plot_flag=True):
    """Plot sorted protein expression values."""
    if not plot_flag:
        return
    from scipy.sparse import issparse

    X_prot = adata_prot.X.toarray() if issparse(adata_prot.X) else adata_prot.X
    plt.figure(figsize=(10, 6))
    plt.title("Protein expression (z-normalized from Step 0)")
    plt.plot(np.sort(X_prot[:n_cells, :n_features], axis=0))
    plt.show()


def plot_spatial_only_umap(adata_spatial_only, color_key="CN", plot_flag=True):
    """Plot UMAP of spatial-only data."""
    if not plot_flag:
        return
    sc.pp.pca(adata_spatial_only)
    sc.pp.neighbors(adata_spatial_only)
    sc.tl.umap(adata_spatial_only)
    sc.pl.umap(adata_spatial_only, color=color_key)


def plot_b_cells_analysis(adata_rna, subset_size=5000):
    """Plot analysis for B cells"""
    adata_rna_subset = sc.pp.subsample(
        adata_rna, n_obs=min(subset_size, adata_rna.n_obs), copy=True
    )
    adata_B_cells = adata_rna_subset[
        adata_rna_subset.obs["major_cell_types"] == adata_rna_subset.obs["major_cell_types"][0]
    ].copy()

    # Ensure data is float for PCA
    if not np.issubdtype(adata_B_cells.X.dtype, np.floating):
        adata_B_cells.X = adata_B_cells.X.astype(float)

    sc.pp.pca(adata_B_cells)
    sc.pp.neighbors(adata_B_cells, use_rep="X_pca")
    sc.tl.umap(adata_B_cells)
    if "tissue" in adata_B_cells.obs:
        sc.pl.umap(
            adata_B_cells,
            color=["tissue"],
            title="RNA: verifying tissue does not give a major effect",
        )
    else:
        sc.pl.umap(
            adata_B_cells,
            color=["cell_types"],
            title="RNA: verifying cell types are well separated",
        )


def plot_pca_and_umap(adata_rna, adata_prot, subset_size=2000):
    """Plot PCA and UMAP visualizations"""
    adata_rna_subset = sc.pp.subsample(
        adata_rna, n_obs=min(subset_size, adata_rna.n_obs), copy=True
    ).copy()
    adata_prot_subset = sc.pp.subsample(
        adata_prot, n_obs=min(subset_size, adata_prot.n_obs), copy=True
    ).copy()
    sc.pl.pca(
        adata_rna_subset,
        color=["cell_types", "major_cell_types"],
        title=["RNA PCA: cell types", "RNA PCA: major cell types"],
    )
    sc.pl.pca(
        adata_prot_subset,
        color=["cell_types", "major_cell_types"],
        title=["Protein PCA: cell types", "Protein PCA: major cell types"],
    )
    sc.pl.embedding(
        adata_rna_subset,
        basis="X_original_umap",
        color=["cell_types", "major_cell_types"],
        title=["RNA UMAP (orig): cell types", "RNA UMAP (orig): major cell types"],
    )
    sc.pl.embedding(
        adata_prot_subset,
        basis="X_original_umap",
        color=["cell_types", "major_cell_types"],
        title=[
            "Protein UMAP (orig): cell types",
            "Protein UMAP (orig): major cell types",
        ],
    )


def plot_protein_umap(adata_prot, subset_size=1000):
    """Plot protein UMAP visualizations"""
    adata_prot_subset = sc.pp.subsample(
        adata_prot, n_obs=min(subset_size, adata_prot.n_obs), copy=True
    )
    sc.pp.neighbors(adata_prot_subset, use_rep="X_pca", key_added="X_neighborhood")
    sc.tl.umap(adata_prot_subset, neighbors_key="X_neighborhood")
    adata_prot_subset.obsm["X_original_umap"] = adata_prot_subset.obsm["X_umap"]
    sc.pl.umap(
        adata_prot_subset,
        color="CN",
        title="Protein UMAP of CN vectors colored by CN label",
        neighbors_key="original_neighbors",
    )
    one_cell_type = adata_prot_subset.obs["major_cell_types"][0]
    sc.pl.umap(
        adata_prot_subset[adata_prot_subset.obs["major_cell_types"] == one_cell_type],
        color="cell_types",
        title="Protein UMAP of CN vectors colored by minor cell type label",
    )
    return one_cell_type


def plot_original_data_visualizations(adata_rna, adata_prot, subset_size=1000):
    """Plot original data visualizations"""
    adata_rna_subset = sc.pp.subsample(
        adata_rna, n_obs=min(subset_size, adata_rna.n_obs), copy=True
    )
    adata_prot_subset = sc.pp.subsample(
        adata_prot, n_obs=min(subset_size, adata_prot.n_obs), copy=True
    )
    sc.pl.embedding(
        adata_rna_subset,
        color=["CN", "cell_types", "archetype_label"],
        basis="X_original_umap",
        title=[
            "Original rna data CN",
            "Original rna data cell types",
            "Original rna data archetype label",
        ],
    )

    sc.pl.embedding(
        adata_prot_subset,
        color=[
            "CN",
            "cell_types",
            "archetype_label",
        ],
        basis="X_original_umap",
        title=[
            "Original protein data CN",
            "Original protein data cell types",
            "Original protein data archetype label",
        ],
    )


def plot_protein_feature_distributions(
    adata_prot, layer=None, n_features=30, n_cells=1000, plot_flag=True
):
    """Plot KDE distributions of protein features in subplots.

    Creates a grid of KDE plots showing the distribution of protein features.
    Useful for visualizing feature distributions before and after normalization.

    Parameters
    ----------
    adata_prot : AnnData
        Protein AnnData object
    layer : str, optional
        Layer to plot from (e.g., 'z_normalized'), if None uses X, by default None
    n_features : int, optional
        Number of features to plot, by default 30
    n_cells : int, optional
        Number of cells to subsample for plotting, by default 1000
    plot_flag : bool, optional
        Whether to generate the plot, by default True
    """
    if not plot_flag:
        return

    # Subsample cells
    adata_prot_subsampled = sc.pp.subsample(
        adata_prot, n_obs=min(n_cells, adata_prot.n_obs), copy=True
    )

    # Get data from specified layer or X
    if layer is not None and layer in adata_prot_subsampled.layers:
        data_df = adata_prot_subsampled.to_df(layer=layer)
    else:
        data_df = adata_prot_subsampled.to_df()

    # Limit number of features
    n_features = min(n_features, len(data_df.columns))

    # Create subplot grid (5 rows x 6 columns = 30 features)
    n_rows = 5
    n_cols = 6
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 6))
    plt.subplots_adjust(hspace=1.0)  # Increase hspace for more vertical space

    for i in range(n_features):
        row = i // n_cols
        col = i % n_cols
        if row < n_rows and col < n_cols:
            sns.kdeplot(
                data_df.iloc[:, i],
                legend=False,
                ax=axs[row, col],
            )
            axs[row, col].set_title(adata_prot_subsampled.var_names[i])
            axs[row, col].set_xlabel(None)
            axs[row, col].set_ylabel(None)
            axs[row, col].set_yticks([])

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_protein_data_heatmap(adata_prot, n_cells=100, n_features=100, plot_flag=True):
    """Plot heatmap of protein data (debug/exploratory visualization).

    Creates a simple heatmap showing a subset of protein expression data.
    Useful for quick visual inspection of data structure.

    Parameters
    ----------
    adata_prot : AnnData
        Protein AnnData object
    n_cells : int, optional
        Number of cells to plot, by default 100
    n_features : int, optional
        Number of features to plot, by default 100
    plot_flag : bool, optional
        Whether to generate the plot, by default True
    """
    if not plot_flag:
        return

    plt.figure(figsize=(6, 6))
    sns.heatmap(adata_prot.X[:n_cells, :n_features], linewidths=0)
    plt.title(f"Protein data heatmap (first {n_cells}x{n_features})")
    plt.show()
    plt.close()
