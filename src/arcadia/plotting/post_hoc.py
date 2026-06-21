"""Post-hoc plotting utilities for counterfactual analysis and metrics visualization."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData

from arcadia.utils.logging import logger


def plot_umap_latent_arcadia(latent_arcadia):
    cmap = mpl.colormaps.get_cmap("tab20")

    X = latent_arcadia.obsm["X_umap"]  # (n_obs, 2)
    ct = latent_arcadia.obs["cell_types"].astype("category")
    archetype_label = latent_arcadia.obs["archetype_label"].astype("category")
    matched_archetype_weight = latent_arcadia.obs["matched_archetype_weight"].to_numpy()
    is_extreme = latent_arcadia.obs["is_extreme_archetype"].values

    groups = list(ct.cat.categories)
    arch_groups = list(archetype_label.cat.categories)

    # assign colors per cell type and per archetype
    group_colors = {g: cmap(i % cmap.N) for i, g in enumerate(groups)}
    arch_colors = {g: cmap(i % cmap.N) for i, g in enumerate(arch_groups)}

    # normalize power for alpha mapping
    alpha_min = 0.15
    pmin, pmax = np.nanmin(matched_archetype_weight), np.nanmax(matched_archetype_weight)
    gamma = 4
    norm = np.clip((matched_archetype_weight - pmin) / (pmax - pmin), 0, 1)
    alphas = alpha_min + (1 - alpha_min) * (norm**gamma)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- Subplot 1: Cell type color ---
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=2, c="#e5e5e5", linewidths=0, alpha=0.3)

    for i, g in enumerate(groups):
        m = ct.to_numpy() == g
        if not m.any():
            continue
        rgb = group_colors[g][:3]
        rgba = np.column_stack([np.tile(rgb, (m.sum(), 1)), alphas[m]])
        # plot "regular" cells
        m_not_extreme = m & (~is_extreme)
        if m_not_extreme.any():
            ax.scatter(
                X[m_not_extreme, 0],
                X[m_not_extreme, 1],
                s=8,
                c=rgba[~is_extreme[m]],
                linewidths=0,
                label=str(g),
            )
        # plot "extreme" cells with 'x' marker and larger marker size, but do not include in legend
        m_extreme = m & is_extreme
        if m_extreme.any():
            ax.scatter(
                X[m_extreme, 0],
                X[m_extreme, 1],
                s=15,
                c="black",
                linewidths=1.5,
                marker="x",
                label=None,
                zorder=3,
            )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP: color = cell type, opacity = matched_archetype_weight")
    ax.legend(markerscale=2, frameon=False, fontsize=8, loc="best", title="Cell Types")
    ax.grid(False)

    # --- Subplot 2: Archetype label color ---
    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], s=2, c="#e5e5e5", linewidths=0, alpha=0.3)

    for i, g in enumerate(arch_groups):
        m = archetype_label.to_numpy() == g
        if not m.any():
            continue
        rgb = arch_colors[g][:3]
        rgba = np.column_stack([np.tile(rgb, (m.sum(), 1)), alphas[m]])
        # plot "regular" cells
        m_not_extreme = m & (~is_extreme)
        if m_not_extreme.any():
            ax.scatter(
                X[m_not_extreme, 0],
                X[m_not_extreme, 1],
                s=8,
                c=rgba[~is_extreme[m]],
                linewidths=0,
                label=str(g),
            )
        # plot "extreme" cells with 'x' marker and larger marker size, but do not include in legend
        m_extreme = m & is_extreme
        if m_extreme.any():
            ax.scatter(
                X[m_extreme, 0],
                X[m_extreme, 1],
                s=15,
                # c=rgba[is_extreme[m]],
                c="black",
                linewidths=1.5,
                marker="x",
                label=None,
                zorder=3,
            )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP: color = archetype_label, opacity = matched_archetype_weight")
    ax.legend(markerscale=2, frameon=False, fontsize=8, loc="best", title="Archetype")
    ax.grid(False)

    plt.suptitle("UMAP: color = cell type, opacity = matched_archetype_weight\nX are extreme cells")
    fig.tight_layout()
    plt.show()


def plot_morans_i(adata, score_key, emb1_key, emb2_key, n_neighbors=15):
    from arcadia.training.metrics import morans_i as morans_i_func

    key1 = f"morans_{emb1_key}"
    key2 = f"morans_{emb2_key}"
    sc.pp.neighbors(adata, use_rep=emb1_key, n_neighbors=n_neighbors, key_added=key1)
    sc.pp.neighbors(adata, use_rep=emb2_key, n_neighbors=n_neighbors, key_added=key2)

    # Visualization for Moran's I: Moran scatter plots and bar comparison
    def _compute_spatial_lag(adata, score_key, neighbors_key):
        z = adata.obs[score_key].to_numpy()
        z_std = (z - z.mean()) / z.std()
        W = adata.obsp[f"{neighbors_key}_connectivities"]
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        row_sums[row_sums == 0.0] = 1.0
        lag = (W @ z_std) / row_sums
        return z_std, np.asarray(lag).ravel()

    z1, lag1 = _compute_spatial_lag(adata, score_key, key1)
    z2, lag2 = _compute_spatial_lag(adata, score_key, key2)

    I1 = morans_i_func(
        adata, score_key=score_key, use_rep=emb1_key, n_neighbors=n_neighbors, neighbors_key=key1
    )
    I2 = morans_i_func(
        adata, score_key=score_key, use_rep=emb2_key, n_neighbors=n_neighbors, neighbors_key=key2
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Moran scatter for embedding 1
    ax = axes[0]
    ax.scatter(z1, lag1, s=8, alpha=0.5)
    # regression line
    m, b = np.polyfit(z1, lag1, 1)
    xs = np.linspace(z1.min(), z1.max(), 100)
    ax.plot(xs, m * xs + b, color="crimson", linewidth=2)
    ax.axhline(0, color="#999", linewidth=1)
    ax.axvline(0, color="#999", linewidth=1)
    ax.set_title(f"Moran scatter — {emb1_key} (I={I1:.3f})")
    ax.set_xlabel("Standardized score (z)")
    ax.set_ylabel("Spatial lag Wz")

    # Moran scatter for embedding 2
    ax = axes[1]
    ax.scatter(z2, lag2, s=8, alpha=0.5)
    m, b = np.polyfit(z2, lag2, 1)
    xs = np.linspace(z2.min(), z2.max(), 100)
    ax.plot(xs, m * xs + b, color="crimson", linewidth=2)
    ax.axhline(0, color="#999", linewidth=1)
    ax.axvline(0, color="#999", linewidth=1)
    ax.set_title(f"Moran scatter — {emb2_key} (I={I2:.3f})")
    ax.set_xlabel("Standardized score (z)")
    ax.set_ylabel("Spatial lag Wz")

    # Bar comparison
    ax = axes[2]
    ax.bar(["arcadia", emb2_key], [I1, I2], color=["#4C72B0", "#55A868"], alpha=0.8)
    ax.set_ylabel("Moran's I (higher better)")
    ax.set_title("Moran's I comparison")
    for i, val in enumerate([I1, I2]):
        ax.text(i, val, f"{val:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.show()


def plot_umap_per_cell_type(adata_latent, modality, model_name):
    """Plot UMAP per cell type with opacity based on matched_archetype_weight."""
    import matplotlib.pyplot as plt
    import numpy as np
    import scanpy as sc

    cell_types = adata_latent.obs["cell_types"].astype("category").cat.categories  # ordered groups

    layer = None  # or e.g., "log1p" if you keep normalized counts in a layer

    # figure layout
    ncols = 3
    nrows = int(np.ceil(len(cell_types) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows))
    axs = np.ravel(axs)

    for ax, g in zip(axs, cell_types):
        # subset to the current type; copy to avoid side effects
        ad = adata_latent[adata_latent.obs["cell_types"] == g].copy()
        if ad.n_obs < 10:
            ax.set_axis_off()
            ax.set_title(f"{g} — too few cells")
            continue

        if layer is not None:
            ad.X = ad.layers[layer]

        # standard Scanpy pipeline for embedding
        sc.pp.scale(ad, max_value=10)
        sc.pp.neighbors(ad, n_neighbors=15)  # graph
        sc.tl.umap(ad)  # UMAP

        # plot UMAP with opacity = archetype_power (reuse your contrast mapping)
        power = np.exp(ad.obs["matched_archetype_weight"].to_numpy())  # ensure present in subset
        lo, hi = np.nanpercentile(power, 5), np.nanpercentile(power, 95)  # robust limits
        x = np.clip((power - lo) / (hi - lo + 1e-12), 0, 1)  # normalize
        alpha_min, alpha_max = 0.1, 1.0
        alphas = alpha_min + (alpha_max - alpha_min) * (x**5)  # gamma to boost contrast

        # fixed color per type (tab10 cycle)
        idx = list(cell_types).index(g) % 10
        color = plt.cm.tab10(idx / 9.0)[:3]  # convert idx to 0..1 and get RGB
        xy = ad.obsm["X_umap"]  # UMAP coords generated above
        rgba = np.column_stack([np.tile(color, (ad.n_obs, 1)), alphas])

        ax.scatter(xy[:, 0], xy[:, 1], s=10, c=rgba, linewidths=0)
        ax.set_title(f"{g} — UMAP (alpha=matched_archetype_weight)")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

    # clean unused axes
    for ax in axs[len(cell_types) :]:
        fig.delaxes(ax)
    fig.suptitle(
        f"UMAP: color = cell type, opacity = matched_archetype_weight\nX are extreme cells\n{model_name}"
    )
    fig.tight_layout()
    plt.show()


def plot_individual_and_combined_umaps(adata_rna, adata_prot, model_name, plot_flag=True):
    if not plot_flag:
        return None
    dataset_name = adata_rna.uns["dataset_name"]
    sc.pp.neighbors(adata_rna, use_rep="latent", n_neighbors=15, key_added="neighbors_scvi")
    sc.tl.umap(adata_rna, neighbors_key="neighbors_scvi")
    sc.pl.embedding(
        adata_rna,
        basis="X_umap",
        color=["cell_types", "CN", "matched_archetype_weight"],
        title=f"{model_name} rna",
    )
    sc.pp.neighbors(adata_prot, use_rep="latent", n_neighbors=15, key_added="neighbors_scvi")
    sc.tl.umap(adata_prot, neighbors_key="neighbors_scvi")
    sc.pl.embedding(
        adata_prot,
        basis="X_umap",
        color=["cell_types", "CN", "matched_archetype_weight"],
        title=f"{model_name} prot",
    )
    combined_latent_comparison = AnnData(
        np.vstack([adata_rna.obsm["latent"], adata_prot.obsm["latent"]])
    )
    combined_latent_comparison.obs = pd.concat([adata_rna.obs, adata_prot.obs])
    combined_latent_comparison.obs["modality"] = ["RNA"] * len(adata_rna) + ["Protein"] * len(
        adata_prot
    )
    sc.pp.neighbors(combined_latent_comparison, use_rep="X", n_neighbors=15)
    sc.tl.umap(combined_latent_comparison)
    from datetime import datetime

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(
        combined_latent_comparison.obsm["X_umap"],
        index=combined_latent_comparison.obs.index,
        columns=["UMAP1", "UMAP2"],
    )
    df["modality"] = combined_latent_comparison.obs["modality"]
    df["cell_types"] = combined_latent_comparison.obs["cell_types"]
    df["CN"] = combined_latent_comparison.obs["CN"]
    df["matched_archetype_weight"] = combined_latent_comparison.obs["matched_archetype_weight"]
    df.to_csv(
        f"/home/barroz/projects/ARCADIA/zzz_umap_{model_name}_{dataset_name}_{timestamp_str}.csv"
    )
    print(
        f"Saved dataframe to /home/barroz/projects/ARCADIA/zzz_umap_{model_name}_{dataset_name}_{timestamp_str}.csv"
    )
    sc.pl.embedding(
        combined_latent_comparison,
        basis="X_umap",
        color=["cell_types", "CN", "matched_archetype_weight", "modality"],
        title=f"{model_name} combined",
    )
    # # save fig as pdf
    # plt.savefig(f"/home/barroz/projects/ARCADIA/umap_{model_name}_combined_{timestamp_str}.pdf")
    # print(
    #     f"Saved figure to /home/barroz/projects/ARCADIA/umap_{model_name}_combined_{timestamp_str}.pdf"
    # )
    # print(
    #     f"Saved dataframe to /home/barroz/projects/ARCADIA/zzz_umap_{model_name}_{timestamp_str}.csv"
    # )
    return combined_latent_comparison


def plot_latent_space_mixing(adata_rna, adata_prot, subsample_size=8000):
    """Plot latent space mixing between RNA and protein data."""
    combined_latent_comparison = AnnData(
        np.vstack([adata_rna.obsm["X_scVI"], adata_prot.obsm["X_scVI"]])
    )
    combined_latent_comparison.obs = pd.concat([adata_rna.obs, adata_prot.obs])
    combined_latent_comparison.obs["modality"] = ["RNA"] * len(adata_rna) + ["Protein"] * len(
        adata_prot
    )
    if combined_latent_comparison.n_obs > subsample_size:
        logger.info(
            f"Subsampling from {combined_latent_comparison.n_obs} to {subsample_size} cells..."
        )
        sc.pp.subsample(combined_latent_comparison, n_obs=subsample_size, random_state=42)
        logger.info(f"✓ Subsampled to {combined_latent_comparison.n_obs} cells")
    sc.pp.pca(combined_latent_comparison)
    sc.pp.neighbors(combined_latent_comparison, use_rep="X")
    sc.tl.umap(combined_latent_comparison)
    obs_to_plot = ["cell_types", "modality", "CN"]
    (
        obs_to_plot.append("CN_matched")
        if "CN_matched" in combined_latent_comparison.obs.columns
        else None
    )

    sc.pl.umap(
        combined_latent_comparison,
        color=obs_to_plot,
        title="Latent Space Mixing (should be good)",
    )
    rna_data = combined_latent_comparison[combined_latent_comparison.obs["modality"] == "RNA"]
    protein_data = combined_latent_comparison[
        combined_latent_comparison.obs["modality"] == "Protein"
    ]
    valid_rna_obs_to_plot = ["CN"]
    valid_protein_obs_to_plot = ["CN"]
    if "batch" in rna_data.obs.columns and len(set(rna_data.obs["batch"].unique())) > 1:
        valid_rna_obs_to_plot.append("batch")
    if "batch" in protein_data.obs.columns and len(set(protein_data.obs["batch"].unique())) > 1:
        valid_protein_obs_to_plot.append("batch")
    sc.pl.umap(rna_data, color=valid_rna_obs_to_plot, title=f"RNA {valid_rna_obs_to_plot}")
    sc.pl.umap(
        protein_data, color=valid_protein_obs_to_plot, title=f"Protein {valid_protein_obs_to_plot}"
    )
    (
        sc.pl.umap(combined_latent_comparison, color=["CN_matched"])
        if "CN_matched" in combined_latent_comparison.obs.columns
        else None
    )
    # add umap with color of archetype_max_proportion
    sc.pl.umap(
        combined_latent_comparison,
        color="archetype_max_proportion",
        title="Archetype Max Proportion",
    )
    # also separate by modality
    sc.pl.umap(rna_data, color="archetype_max_proportion", title="RNA Archetype Max Proportion")
    sc.pl.umap(
        protein_data, color="archetype_max_proportion", title="Protein Archetype Max Proportion"
    )
    return combined_latent_comparison


# %%
def generate_cn_umap_plots(combined_latent, metrics_dict, subsample_size=2000):
    """Generate UMAP plots for each cell type with CN as colors using scanpy.

    Args:
        combined_latent: Combined latent space AnnData object
        metrics_dict: Dictionary containing all calculated metrics with keys:
            - 'overall': dict with overall scores (cn_ilisi_score, color_entropy_score, etc.)
            - 'per_cell_type': dict with per-cell-type metrics
        subsample_size: Maximum number of cells to use for plotting (default: 2000)
    """
    logger.info(
        f"Generating UMAP plots for each cell type with CN colors (subsample size: {subsample_size})..."
    )

    # Extract overall scores from metrics dict
    overall_metrics = metrics_dict.get("overall", {})
    cn_ilisi_score = overall_metrics.get("cn_ilisi_score", 0.0)
    color_entropy_score = overall_metrics.get("color_entropy_score", 0.0)

    # Extract per-cell-type metrics
    per_cell_type_metrics = metrics_dict.get("per_cell_type", {})

    # Check if required columns exist
    if "cell_types" not in combined_latent.obs.columns:
        logger.error("Missing 'cell_types' column for plotting")
        return

    if "CN" not in combined_latent.obs.columns:
        logger.error("Missing 'CN' column for plotting")
        return

    # Subsample the data if it's larger than subsample_size
    if combined_latent.n_obs > subsample_size:
        logger.info(f"Subsampling from {combined_latent.n_obs} to {subsample_size} cells...")
        sc.pp.subsample(combined_latent, n_obs=subsample_size, random_state=42)
        logger.info(f"✓ Subsampled to {combined_latent.n_obs} cells")
    else:
        logger.info(f"Using all {combined_latent.n_obs} cells (below subsample threshold)")

    # Calculate UMAP if not already present
    if "X_umap" not in combined_latent.obsm:
        logger.info("Computing UMAP embedding...")
        sc.tl.umap(combined_latent, random_state=42)

    # Get unique cell types and filter valid ones
    all_cell_types = combined_latent.obs["cell_types"].unique()
    valid_cell_types = []

    for cell_type in all_cell_types:
        cell_type_mask = combined_latent.obs["cell_types"] == cell_type
        cell_type_data = combined_latent[cell_type_mask]

        # Skip if too few cells or only one CN value
        if cell_type_data.n_obs < 10:
            continue

        cn_values = cell_type_data.obs["CN"].dropna().unique()
        if len(cn_values) <= 1:
            continue

        valid_cell_types.append(cell_type)

    logger.info(
        f"Found {len(valid_cell_types)} valid cell types for plotting: {list(valid_cell_types)}"
    )

    if len(valid_cell_types) == 0:
        logger.warning("No valid cell types found for plotting")
        return

    # Use provided per-cell-type metrics or set defaults for missing cell types
    cell_type_metrics = {}
    for cell_type in valid_cell_types:
        if cell_type in per_cell_type_metrics:
            # Use provided metrics
            cell_type_metrics[cell_type] = per_cell_type_metrics[cell_type]
            logger.info(f"Using provided metrics for {cell_type}")
        else:
            # Set defaults if metrics not provided
            logger.warning(f"No metrics provided for {cell_type}, using defaults")
            cell_type_metrics[cell_type] = {"cn_ilisi": 0.0, "color_entropy": 0.0}

    # Create subplot figure using scanpy
    logger.info("Creating combined subplot figure with all cell types...")

    # Create a list of cell type data with comprehensive titles for scanpy
    cell_type_titles = []
    for cell_type in valid_cell_types:
        ct_metrics = cell_type_metrics[cell_type]

        # Get additional metrics if available
        kbet_value = ct_metrics.get("kbet", 0.0)
        cms_value = ct_metrics.get("cms", 0.0)

        # Create comprehensive title with all metrics
        title = (
            f"{cell_type}\n"
            f"CN iLISI: {ct_metrics['cn_ilisi']:.3f} | "
            f"Color Entropy: {ct_metrics['color_entropy']:.3f}\n"
            f"kBET: {kbet_value:.3f} | "
            f"CMS: {cms_value:.3f}"
        )
        cell_type_titles.append(title)

    # Calculate subplot layout
    n_plots = len(valid_cell_types)
    n_cols = min(3, n_plots)  # Maximum 3 columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

    # Create a single figure with subplots for all cell types
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))

    # Handle case where there's only one subplot
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Flatten axes array for easier indexing
    axes_flat = axes.flatten() if n_plots > 1 else [axes[0]]

    logger.info(f"Creating combined figure with {n_plots} subplots ({n_rows}x{n_cols})")

    # Create subplots for each cell type
    for i, (cell_type, title) in enumerate(zip(valid_cell_types, cell_type_titles)):
        cell_type_mask = combined_latent.obs["cell_types"] == cell_type
        cell_type_data = combined_latent[cell_type_mask].copy()

        # Subsample if needed for performance
        if cell_type_data.n_obs > subsample_size:
            sc.pp.subsample(cell_type_data, n_obs=subsample_size, random_state=42)
            logger.info(f"Subsampled {cell_type} to {subsample_size} cells")

        logger.info(f"Creating subplot {i+1}/{n_plots} for cell type: {cell_type}")

        # Plot this cell type on the specific subplot
        sc.pl.umap(
            cell_type_data,
            color="CN",
            title=title,
            legend_loc="right margin",
            size=30,
            alpha=0.7,
            ax=axes_flat[i],
            show=False,  # Don't show individual plots
            frameon=False,
        )

    # Hide unused subplots
    for j in range(len(valid_cell_types), len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Adjust layout and show the combined figure
    plt.tight_layout()
    plt.show()

    logger.info("✓ Combined cell type UMAP subplots completed")

    # Create overall plots using scanpy
    logger.info("Creating overall UMAP plots...")

    # Plot by cell type
    sc.pl.umap(
        combined_latent,
        color="cell_types",
        title="All Cell Types",
        legend_loc="right margin",
        size=30,
        alpha=0.7,
        show=True,
    )

    # Plot by CN with all metrics in title
    metrics_title = (
        f"All Cells - CN Distribution\n"
        f"CN iLISI: {cn_ilisi_score:.4f} | Color Entropy: {color_entropy_score:.4f}\n"
        f"CMS: {overall_metrics.get('cms_score', 0.0):.4f} | "
        f"kBET: {overall_metrics.get('kbet_score', 0.0):.4f}"
    )

    sc.pl.umap(
        combined_latent,
        color="CN",
        title=metrics_title,
        legend_loc="right margin",
        size=30,
        alpha=0.7,
        show=True,
    )

    # Plot by modality
    sc.pl.umap(
        combined_latent,
        color="modality",
        title="All Cells - Modality Distribution",
        legend_loc="right margin",
        size=30,
        alpha=0.7,
        show=True,
    )

    logger.info("✓ UMAP plotting completed")


# %%
def plot_metrics_by_cell_type(metrics_dict, save_path=None):
    """
    Create bar plots of metrics clustered by cell types.

    Args:
        metrics_dict: Dictionary containing metrics with 'overall' and 'per_cell_type' keys
        save_path: Path to save the plots (optional)
    """
    logger.info("Creating bar plots of metrics by cell type...")

    # Extract per-cell-type metrics
    cell_type_metrics = metrics_dict["per_cell_type"]
    if not cell_type_metrics:
        logger.warning("No cell type metrics available for plotting")
        return

    # Get all metric names and cell types
    metric_names = list(next(iter(cell_type_metrics.values())).keys())
    cell_types = list(cell_type_metrics.keys())

    # Create figure with subplots for each metric - increased height and spacing
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(14, 5 * len(metric_names)))
    if len(metric_names) == 1:
        axes = [axes]  # Make axes iterable if there's only one subplot

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4)

    # Color palette for bars
    colors = plt.cm.tab10(np.linspace(0, 1, len(cell_types)))

    # Plot each metric as a separate subplot
    for i, metric_name in enumerate(metric_names):
        # Extract values for this metric across all cell types
        values = [cell_type_metrics[ct][metric_name] for ct in cell_types]

        # Sort cell types by metric value
        sorted_indices = np.argsort(values)
        sorted_cell_types = [cell_types[idx] for idx in sorted_indices]
        sorted_values = [values[idx] for idx in sorted_indices]
        sorted_colors = [colors[idx] for idx in sorted_indices]

        # Create bar plot
        bars = axes[i].bar(sorted_cell_types, sorted_values, color=sorted_colors, alpha=0.7)

        # Add overall metric value as a horizontal line if available
        if (
            metric_name in metrics_dict["overall"]
            or f"{metric_name}_score" in metrics_dict["overall"]
        ):
            # Handle different naming conventions
            overall_key = (
                metric_name if metric_name in metrics_dict["overall"] else f"{metric_name}_score"
            )
            overall_value = metrics_dict["overall"].get(overall_key, None)

            if overall_value is not None and np.isfinite(overall_value):
                axes[i].axhline(
                    y=overall_value,
                    color="red",
                    linestyle="--",
                    label=f"Overall: {overall_value:.4f}",
                )
                axes[i].legend()

        # Add value labels on top of bars with better positioning
        max_value = max(sorted_values) if sorted_values else 1
        for bar, value in zip(bars, sorted_values):
            height = bar.get_height()
            # Adjust label position based on bar height
            label_y = height + max_value * 0.02  # 2% of max value above bar
            axes[i].text(
                bar.get_x() + bar.get_width() / 2.0,
                label_y,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                rotation=0,
                fontsize=10,
            )

        # Set title and labels with better spacing
        axes[i].set_title(
            f'{metric_name.replace("_", " ").title()} by Cell Type', fontsize=12, pad=20
        )
        axes[i].set_ylabel(metric_name, fontsize=11)
        axes[i].set_ylim(0, max(sorted_values) * 1.2)  # Add more space for labels

        # Rotate x-axis labels for better readability with more space
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha="right", fontsize=10)

        # Add more space at bottom for rotated labels
        axes[i].tick_params(axis="x", pad=10)

    # Adjust layout
    plt.tight_layout()

    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Metrics bar plot saved to: {save_path}")

    # Show plot
    plt.show()


# Add a grouped bar chart for easier comparison
def plot_grouped_metrics(metrics_dict, save_path=None):
    """
    Create a grouped bar chart to compare all metrics across cell types.

    Args:
        metrics_dict: Dictionary containing metrics with 'overall' and 'per_cell_type' keys
        save_path: Path to save the plot (optional)
    """
    logger.info("Creating grouped bar chart of all metrics...")

    # Extract per-cell-type metrics
    cell_type_metrics = metrics_dict["per_cell_type"]
    if not cell_type_metrics:
        logger.warning("No cell type metrics available for plotting")
        return

    # Get all metric names and cell types
    metric_names = list(next(iter(cell_type_metrics.values())).keys())
    cell_types = list(cell_type_metrics.keys())

    # Define which metrics should be inverted (lower is better)
    invert_metrics = {"kbet"}  # kBET rejection rate

    # Normalize all metrics to 0-1 range for comparison
    normalized_metrics = {}
    for metric in metric_names:
        # Collect all values for this metric
        all_values = [cell_type_metrics[ct][metric] for ct in cell_types]

        # Handle edge cases
        if not all_values or all(v == 0 for v in all_values):
            normalized_metrics[metric] = [0.0] * len(cell_types)
            continue

        min_val = min(all_values)
        max_val = max(all_values)

        # Normalize to 0-1
        if max_val == min_val:
            normalized_values = [0.5] * len(cell_types)  # All same value = middle
        else:
            normalized_values = [(v - min_val) / (max_val - min_val) for v in all_values]

        # Invert if lower is better (so higher normalized value = better performance)
        if metric in invert_metrics:
            normalized_values = [1 - v for v in normalized_values]

        normalized_metrics[metric] = normalized_values

        logger.info(
            f"Normalized {metric}: min={min_val:.3f}, max={max_val:.3f}, inverted={metric in invert_metrics}"
        )

    # Create a figure with better proportions
    fig, ax = plt.subplots(figsize=(16, 10))

    # Set width of bars
    bar_width = 0.15

    # Set position of bars on x axis
    indices = np.arange(len(cell_types))

    # Plot bars for each metric using normalized values
    for i, metric in enumerate(metric_names):
        # Extract normalized values for this metric
        values = normalized_metrics[metric]
        original_values = [cell_type_metrics[ct][metric] for ct in cell_types]

        # Plot bars
        pos = indices + i * bar_width
        bars = ax.bar(
            pos,
            values,
            bar_width,
            alpha=0.7,
            label=f"{metric.replace('_', ' ').title()}{' (inv)' if metric in invert_metrics else ''}",
        )

        # Add value labels showing both normalized and original values
        for bar, norm_value, orig_value in zip(bars, values, original_values):
            height = bar.get_height()
            # Better label positioning
            label_y = height + 0.02  # Fixed spacing since all values are 0-1
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                label_y,
                f"{norm_value:.2f}\n({orig_value:.2f})",
                ha="center",
                va="bottom",
                rotation=0,
                fontsize=8,
            )

    # Add overall values as horizontal lines (also normalized)
    for i, metric in enumerate(metric_names):
        # Handle different naming conventions
        overall_key = metric if metric in metrics_dict["overall"] else f"{metric}_score"
        overall_value = metrics_dict["overall"].get(overall_key, None)

        if overall_value is not None and np.isfinite(overall_value):
            # Normalize the overall value using the same scale as individual values
            all_values = [cell_type_metrics[ct][metric] for ct in cell_types]
            if all_values:
                min_val = min(all_values)
                max_val = max(all_values)

                if max_val != min_val:
                    normalized_overall = (overall_value - min_val) / (max_val - min_val)
                    if metric in invert_metrics:
                        normalized_overall = 1 - normalized_overall

                    ax.axhline(
                        y=normalized_overall,
                        color=plt.cm.tab10(i / 10),
                        linestyle="--",
                        alpha=0.5,
                        label=f"Overall {metric}: {normalized_overall:.2f} ({overall_value:.3f})",
                    )

    # Add labels and title with better spacing
    ax.set_xlabel("Cell Type", fontsize=12, labelpad=15)
    ax.set_ylabel("Normalized Metric Value (0-1, Higher = Better)", fontsize=12, labelpad=15)
    ax.set_title(
        "Normalized Comparison of All Metrics Across Cell Types\n(Higher values = better performance)",
        fontsize=14,
        pad=20,
    )
    ax.set_xticks(indices + bar_width * (len(metric_names) - 1) / 2)
    ax.set_xticklabels(cell_types, rotation=45, ha="right", fontsize=11)

    # Set y-axis to 0-1 range with some padding for labels
    ax.set_ylim(0, 1.2)

    # Add legend with better positioning
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)

    # Add padding for x-axis labels
    ax.tick_params(axis="x", pad=10)

    # Adjust layout
    plt.tight_layout()

    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Grouped metrics plot saved to: {save_path}")

    # Show plot
    plt.show()


def plot_counterfactual_distribution_histograms(
    samples_dict, random_seed=42, bins=30, figsize=(18, 12)
):
    """
    Plot distribution histograms for randomly selected cells from counterfactual samples.

    Args:
        samples_dict: Output from sample_from_counterfactual_distributions()
        random_seed: Random seed for reproducible cell selection
        bins: Number of histogram bins
        figsize: Figure size tuple
    """
    logger.info("Creating distribution histograms for randomly selected cells...")

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Get sample data
    rna_same_samples = samples_dict[
        "rna_same_modal_samples"
    ]  # (n_samples, n_rna_cells, n_rna_genes)
    rna_cf_samples = samples_dict[
        "rna_counterfactual_samples"
    ]  # (n_samples, n_protein_cells, n_rna_genes)
    protein_same_samples = samples_dict[
        "protein_same_modal_samples"
    ]  # (n_samples, n_protein_cells, n_proteins)
    protein_cf_samples = samples_dict[
        "protein_counterfactual_samples"
    ]  # (n_samples, n_rna_cells, n_proteins)

    logger.info(f"Sample shapes:")
    logger.info(f"  RNA same-modal: {rna_same_samples.shape}")
    logger.info(f"  RNA counterfactual: {rna_cf_samples.shape}")
    logger.info(f"  Protein same-modal: {protein_same_samples.shape}")
    logger.info(f"  Protein counterfactual: {protein_cf_samples.shape}")

    # Define colors for different sample types
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # === RNA DISTRIBUTIONS FIGURE ===
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # === RNA SAME-MODAL (3 random cells) ===
    for i in range(3):
        ax = axes[i]

        # Randomly select a cell and gene
        random_cell = np.random.randint(0, rna_same_samples.shape[1])
        random_gene = np.random.randint(0, rna_same_samples.shape[2])

        # Get all samples for this cell-gene combination
        cell_gene_samples = rna_same_samples[:, random_cell, random_gene]  # Shape: (n_samples,)

        # Create histogram with KDE
        sns.histplot(
            data=cell_gene_samples,
            bins=bins,
            kde=True,
            alpha=0.7,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
            ax=ax,
        )

        # Get cell and gene names if available
        cell_name = f"Cell_{random_cell}"
        if "adata_rna_normalized" in samples_dict and hasattr(
            samples_dict["adata_rna_normalized"], "obs"
        ):
            if len(samples_dict["adata_rna_normalized"].obs) > random_cell:
                if "cell_types" in samples_dict["adata_rna_normalized"].obs.columns:
                    cell_type = (
                        samples_dict["adata_rna_normalized"].obs["cell_types"].iloc[random_cell]
                    )
                    cell_name = f"{cell_type}_{random_cell}"

        gene_name = f"Gene_{random_gene}"
        if "adata_rna_normalized" in samples_dict and hasattr(
            samples_dict["adata_rna_normalized"], "var_names"
        ):
            if len(samples_dict["adata_rna_normalized"].var_names) > random_gene:
                gene_name = samples_dict["adata_rna_normalized"].var_names[random_gene]

        ax.set_title(f"RNA Same-Modal\n{cell_name} - {gene_name}", fontsize=10)
        ax.set_xlabel("Expression Count")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        # Add statistics text
        mean_val = np.mean(cell_gene_samples)
        std_val = np.std(cell_gene_samples)
        ax.text(
            0.02,
            0.98,
            f"μ={mean_val:.2f}\nσ={std_val:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # === RNA COUNTERFACTUAL (3 random cells) ===
    for i in range(3):
        ax = axes[i + 3]

        # Randomly select a cell and gene
        random_cell = np.random.randint(0, rna_cf_samples.shape[1])
        random_gene = np.random.randint(0, rna_cf_samples.shape[2])

        # Get all samples for this cell-gene combination
        cell_gene_samples = rna_cf_samples[:, random_cell, random_gene]  # Shape: (n_samples,)

        # Create histogram with KDE
        sns.histplot(
            data=cell_gene_samples,
            bins=bins,
            kde=True,
            alpha=0.7,
            color=colors[i + 3],
            edgecolor="black",
            linewidth=0.5,
            ax=ax,
        )

        # Get cell and gene names if available
        cell_name = f"Cell_{random_cell}"
        if "adata_prot_normalized" in samples_dict and hasattr(
            samples_dict["adata_prot_normalized"], "obs"
        ):
            if len(samples_dict["adata_prot_normalized"].obs) > random_cell:
                if "cell_types" in samples_dict["adata_prot_normalized"].obs.columns:
                    cell_type = (
                        samples_dict["adata_prot_normalized"].obs["cell_types"].iloc[random_cell]
                    )
                    cell_name = f"{cell_type}_{random_cell}"

        gene_name = f"Gene_{random_gene}"
        if "adata_rna_normalized" in samples_dict and hasattr(
            samples_dict["adata_rna_normalized"], "var_names"
        ):
            if len(samples_dict["adata_rna_normalized"].var_names) > random_gene:
                gene_name = samples_dict["adata_rna_normalized"].var_names[random_gene]

        ax.set_title(f"RNA Counterfactual\n{cell_name} - {gene_name}", fontsize=10)
        ax.set_xlabel("Expression Count")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        # Add statistics text
        mean_val = np.mean(cell_gene_samples)
        std_val = np.std(cell_gene_samples)
        ax.text(
            0.02,
            0.98,
            f"μ={mean_val:.2f}\nσ={std_val:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.suptitle(
        "Distribution of Expression Counts Across Samples\n(RNA Same-Modal vs RNA Counterfactual)",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout()
    plt.show()

    # === PROTEIN DISTRIBUTIONS FIGURE ===
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # === PROTEIN SAME-MODAL (3 random cells) ===
    for i in range(3):
        ax = axes[i]

        # Randomly select a cell and protein
        random_cell = np.random.randint(0, protein_same_samples.shape[1])
        random_protein = np.random.randint(0, protein_same_samples.shape[2])

        # Get all samples for this cell-protein combination
        cell_protein_samples = protein_same_samples[
            :, random_cell, random_protein
        ]  # Shape: (n_samples,)

        # Create histogram with KDE
        sns.histplot(
            data=cell_protein_samples,
            bins=bins,
            kde=True,
            alpha=0.7,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
            ax=ax,
        )

        # Get cell and protein names if available
        cell_name = f"Cell_{random_cell}"
        if "adata_prot_normalized" in samples_dict and hasattr(
            samples_dict["adata_prot_normalized"], "obs"
        ):
            if len(samples_dict["adata_prot_normalized"].obs) > random_cell:
                if "cell_types" in samples_dict["adata_prot_normalized"].obs.columns:
                    cell_type = (
                        samples_dict["adata_prot_normalized"].obs["cell_types"].iloc[random_cell]
                    )
                    cell_name = f"{cell_type}_{random_cell}"

        protein_name = f"Protein_{random_protein}"
        if "adata_prot_normalized" in samples_dict and hasattr(
            samples_dict["adata_prot_normalized"], "var_names"
        ):
            if len(samples_dict["adata_prot_normalized"].var_names) > random_protein:
                protein_name = samples_dict["adata_prot_normalized"].var_names[random_protein]

        ax.set_title(f"Protein Same-Modal\n{cell_name} - {protein_name}", fontsize=10)
        ax.set_xlabel("Expression Count")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        # Add statistics text
        mean_val = np.mean(cell_protein_samples)
        std_val = np.std(cell_protein_samples)
        ax.text(
            0.02,
            0.98,
            f"μ={mean_val:.2f}\nσ={std_val:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # === PROTEIN COUNTERFACTUAL (3 random cells) ===
    for i in range(3):
        ax = axes[i + 3]

        # Randomly select a cell and protein
        random_cell = np.random.randint(0, protein_cf_samples.shape[1])
        random_protein = np.random.randint(0, protein_cf_samples.shape[2])

        # Get all samples for this cell-protein combination
        cell_protein_samples = protein_cf_samples[
            :, random_cell, random_protein
        ]  # Shape: (n_samples,)

        # Create histogram with KDE
        sns.histplot(
            data=cell_protein_samples,
            bins=bins,
            kde=True,
            alpha=0.7,
            color=colors[i + 3],
            edgecolor="black",
            linewidth=0.5,
            ax=ax,
        )

        # Get cell and protein names if available
        cell_name = f"Cell_{random_cell}"
        if "adata_rna_normalized" in samples_dict and hasattr(
            samples_dict["adata_rna_normalized"], "obs"
        ):
            if len(samples_dict["adata_rna_normalized"].obs) > random_cell:
                if "cell_types" in samples_dict["adata_rna_normalized"].obs.columns:
                    cell_type = (
                        samples_dict["adata_rna_normalized"].obs["cell_types"].iloc[random_cell]
                    )
                    cell_name = f"{cell_type}_{random_cell}"

        protein_name = f"Protein_{random_protein}"
        if "adata_prot_normalized" in samples_dict and hasattr(
            samples_dict["adata_prot_normalized"], "var_names"
        ):
            if len(samples_dict["adata_prot_normalized"].var_names) > random_protein:
                protein_name = samples_dict["adata_prot_normalized"].var_names[random_protein]

        ax.set_title(f"Protein Counterfactual\n{cell_name} - {protein_name}", fontsize=10)
        ax.set_xlabel("Expression Count")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        # Add statistics text
        mean_val = np.mean(cell_protein_samples)
        std_val = np.std(cell_protein_samples)
        ax.text(
            0.02,
            0.98,
            f"μ={mean_val:.2f}\nσ={std_val:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.suptitle(
        "Distribution of Expression Counts Across Samples\n(Protein Same-Modal vs Protein Counterfactual)",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout()
    plt.show()

    logger.info("✓ Distribution histograms completed")


def plot_same_modal_vs_counterfactual_rna_differential_expression_and_umap(
    adata_rna, same_modal_adata_rna, counterfactual_adata_rna
):
    for cell_type in adata_rna.obs["cell_types"].unique():
        adata_rna_current_cell = adata_rna[adata_rna.obs["cell_types"] == cell_type].copy()

        # Filter out CN groups with fewer than 2 samples
        cn_counts = adata_rna_current_cell.obs["CN"].value_counts()
        valid_cns = cn_counts[cn_counts >= 2].index
        logger.info(f"Cell type: {cell_type}, CN counts: {cn_counts.to_dict()}")

        if len(valid_cns) < 2:
            logger.warning(f"Skipping {cell_type}: insufficient CN groups with >= 2 samples")
            continue

        adata_rna_current_cell = adata_rna_current_cell[
            adata_rna_current_cell.obs["CN"].isin(valid_cns)
        ].copy()

        sc.pp.log1p(adata_rna_current_cell)
        sc.tl.rank_genes_groups(
            adata_rna_current_cell,
            groupby="CN",
            method="wilcoxon",
            pts=True,  # fraction expressed in/out
            tie_correct=True,
        )
        top_degs_per_cluster = {}
        de_results = adata_rna_current_cell.uns["rank_genes_groups"]
        for group in adata_rna_current_cell.obs["CN"].unique():
            group_padj = de_results["pvals_adj"][group]
            group_genes = de_results["names"][group]
            group_logfc = de_results["logfoldchanges"][group]
            sig_mask = group_padj < 0.005
            sig_genes = group_genes[sig_mask]
            sig_logfc = np.abs(group_logfc[sig_mask])
            if len(sig_genes) > 0:
                top_idx = np.argsort(sig_logfc)[-3:][::-1]
                top_degs_per_cluster[group] = sig_genes[top_idx].tolist()

        # Plot UMAPs colored by top DEG expression
        all_top_degs = []
        for genes in top_degs_per_cluster.values():
            all_top_degs.extend(genes)
        all_top_degs = list(set(all_top_degs))[:12]  # limit to 12 genes for visualization
        sc.pp.pca(adata_rna_current_cell)
        sc.pp.neighbors(adata_rna_current_cell)
        sc.tl.umap(adata_rna_current_cell)
        items_to_plot = all_top_degs + ["CN", "cell_types"]

        sc.pl.umap(
            adata_rna_current_cell,
            color=items_to_plot,
            ncols=4,
            # vmin=0,
            # vmax="p99",
            cmap="viridis",
            frameon=False,
            show=True,
        )
    same_counterfactual_adata_rna_combined = sc.concat(
        [same_modal_adata_rna, counterfactual_adata_rna],
        join="outer",
        label="data_type",
        keys=["Original", "Counterfactual"],
    )
    sc.pp.pca(same_counterfactual_adata_rna_combined)
    sc.pp.neighbors(same_counterfactual_adata_rna_combined)
    sc.tl.umap(same_counterfactual_adata_rna_combined)
    sc.pl.umap(same_counterfactual_adata_rna_combined, color=["cell_types", "CN", "data_type"])
    sc.pl.umap(
        same_counterfactual_adata_rna_combined[
            same_counterfactual_adata_rna_combined.obs["data_type"] == "Original"
        ],
        color=["CN"],
        title="Original RNA",
    )
    sc.pl.umap(
        same_counterfactual_adata_rna_combined[
            same_counterfactual_adata_rna_combined.obs["data_type"] == "Counterfactual"
        ],
        color=["CN"],
        title="Counterfactual RNA",
    )


# %%
