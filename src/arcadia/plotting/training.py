"""Training-related plotting functions."""

import math
from typing import Iterable

import anndata
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from anndata import AnnData
from matplotlib.colors import Normalize
from scipy.sparse import issparse
from scipy.stats import nbinom, norm, poisson
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from umap import UMAP

from arcadia.plotting.general import safe_mlflow_log_figure
from arcadia.utils.logging import logger


def plot_all_fits_for_gene(gene_data, fit_results, gene_index, modality=""):
    """
    Visualizes the fits of all four candidate distributions for a single gene.

    Creates a 2x2 grid showing the observed data histogram against each of the
    four fitted probability distributions (Normal, Poisson, NB, ZINB).

    Parameters:
    - gene_data: 1D numpy array of the gene's expression data.
    - fit_results: A dict containing the calculated AICs and fitted parameters.
    - gene_index: The index of the gene in the original AnnData object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Distribution Fits for Gene Index {gene_index}", fontsize=16)
    axes = axes.flatten()

    distributions = ["normal", "poisson", "nb", "zinb"]
    colors = {"normal": "orange", "poisson": "red", "nb": "green", "zinb": "purple"}

    for i, dist_type in enumerate(distributions):
        ax = axes[i]
        params = fit_results["params"].get(dist_type)
        aic = fit_results["aics"].get(dist_type, float("inf"))

        # Plot the real data histogram on each subplot
        sns.histplot(
            gene_data, bins=30, stat="density", color="skyblue", label="Observed Data", ax=ax
        )

        if params is None or aic == float("inf"):
            ax.text(
                0.5,
                0.5,
                "Fit Failed or Inapplicable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            x_range = np.linspace(gene_data.min(), gene_data.max(), 300)

            # Plot the fitted distribution curve
            if dist_type == "normal":
                pdf = norm.pdf(x_range, loc=params["mu"], scale=params["std"])
                ax.plot(x_range, pdf, color=colors[dist_type], label=f"Normal Fit")
            elif dist_type == "poisson":
                pmf = poisson.pmf(np.round(x_range), mu=params["mu"])
                ax.plot(x_range, pmf, color=colors[dist_type], label=f"Poisson Fit")
            elif dist_type == "nb":
                pmf = nbinom.pmf(np.round(x_range), n=params["size"], p=params["prob"])
                ax.plot(x_range, pmf, color=colors[dist_type], label=f"NB Fit")
            elif dist_type == "zinb":
                size, prob = 1 / params["alpha"], (1 / params["alpha"]) / (
                    (1 / params["alpha"]) + params["mu"]
                )
                pmf_nb = nbinom.pmf(np.round(x_range), n=size, p=prob)
                pi = params["pi"]
                pmf = (1 - pi) * pmf_nb
                # Add the zero-inflation spike separately for visibility
                pmf[np.round(x_range) == 0] += pi
                ax.plot(x_range, pmf, color=colors[dist_type], label=f"ZINB Fit")

        ax.set_title(f"{dist_type.capitalize()} (AIC: {aic:.2f})")
        ax.legend()
        ax.set_xlabel("Expression Value")
        ax.set_ylabel("Density")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.close()


def pre_train_adata_histograms_heatmap(adata, plot_name, feature_type="protein"):
    n_cells = min(1000, adata.n_obs)
    n_features = min(200, adata.n_vars)

    random_cells = np.random.choice(adata.obs_names, n_cells, replace=False)
    random_features = np.random.choice(adata.var_names, n_features, replace=False)

    subset = adata[random_cells, random_features]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

    # Convert to dense if sparse
    X_dense = subset.X.toarray() if hasattr(subset.X, "toarray") else subset.X
    sns.heatmap(X_dense, cmap="viridis", cbar_kws={"label": "Expression"}, ax=ax1, linewidths=0)
    ax1.set_title(f"Random Subset of {feature_type} Expression")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.set_xlabel(feature_type)
    ax1.set_ylabel("Cell")
    # if the proportion of zeros is greater than 0.9, plot the distribution of the non-zero values
    if X_dense.mean() < 0.1:
        sns.histplot(data=np.log1p(X_dense.flatten()), bins=100, ax=ax2)
    else:
        sns.histplot(data=(X_dense.flatten()), bins=100, ax=ax2)
    ax2.set_title(f"Distribution of {feature_type} Expression Values")
    ax2.set_xlabel("Expression Value")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), f"{plot_name}_heatmap.pdf")
    plt.show()
    plt.close()
    plt.close()

    plt.figure(figsize=(15, 10))
    plt.subplot(1, 1, 1)
    plt.plot(adata.X.mean(axis=0), "b-", alpha=0.7)
    plt.title(f"Mean {feature_type} Expression Across Cells")
    plt.xlabel(f"{feature_type} Index")
    plt.ylabel("Mean Expression")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.close()
    safe_mlflow_log_figure(plt.gcf(), f"{plot_name}_mean.pdf")


def plot_warmup_loss_distributions(
    warmup_raw_losses: dict, precentile: tuple[float, float] = (5, 95), plot_flag=True
):
    """
    Plots histograms of raw loss distributions from the warmup phase,
    showing original.

    Args:
        warmup_raw_losses: A dictionary where keys are loss names and
                           values are lists of raw loss values.
        precentile: Tuple of percentiles (lower, upper) for clipping.
                   Default is (5, 95) [not used].
    """
    if not plot_flag:
        return
    if not warmup_raw_losses:
        print("No warmup losses to plot.")
        return

    loss_names = list(warmup_raw_losses.keys())
    num_losses = len(loss_names)

    if num_losses == 0:
        return

    # Calculate layout once
    ncols = 2
    nrows = math.ceil(num_losses / ncols)

    # Create figure and axes only once
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 5))
    axes = axes.flatten() if num_losses > 1 else [axes]  # Handle single subplot case

    # Pre-compute colors for reuse
    colors = {
        "original": "skyblue",
        "clipped": "orangered",
        "median_orig": "blue",
        "median_clip": "red",
    }

    # Process all loss types at once
    for i, loss_name in enumerate(loss_names):
        ax = axes[i]
        values = warmup_raw_losses[loss_name]

        if not values:
            ax.set_title(f"{loss_name}\n(No data)")
            ax.text(0.5, 0.5, "No data collected", ha="center", va="center")
            continue

        # Convert to numpy array once for faster operations
        values_array = np.array(values)

        # Calculate clipping bounds once
        lower, upper = np.percentile(values_array, precentile)

        # Calculate median values
        median_original = np.median(values_array)

        # Plot both histograms efficiently
        ax.scatter(
            range(len(values_array)),
            values_array,
            color=colors["original"],
            label="Original",
            alpha=0.6,
        )

        # Add median lines
        ax.axhline(
            median_original,
            color=colors["median_orig"],
            linestyle="--",
            linewidth=1.5,
            label=f"Original median: {median_original:.2f}",
        )

        ax.set_title(f"Distribution for {loss_name}")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Raw Loss Value")
        ax.legend()

    # Hide unused subplots all at once
    for j in range(num_losses, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "warmup_loss_distributions.pdf")
    plt.close(fig)


def plot_data_overview(adata_1, adata_2, max_cells=5000, plot_flag=True):
    """Plot overview of RNA and protein data"""
    if not plot_flag:
        return
    logger.info("Plotting data overview...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Subsample if needed
    subsample_n_obs_1 = min(max_cells, len(adata_1))
    adata_1_sub = adata_1[np.random.choice(len(adata_1), subsample_n_obs_1, replace=False)]

    subsample_n_obs_2 = min(max_cells, len(adata_2))
    adata_2_sub = adata_2[np.random.choice(len(adata_2), subsample_n_obs_2, replace=False)]

    # RNA data
    sc.pl.pca(adata_1_sub, color="cell_types", show=False, ax=axes[0])
    axes[0].set_title("RNA PCA")

    # Protein data
    sc.pl.pca(adata_2_sub, color="cell_types", show=False, ax=axes[1])
    axes[1].set_title("Protein PCA")

    plt.tight_layout()
    plt.show()
    plt.close()
    plt.close()


def plot_cell_type_distribution(adata_1, adata_2, max_cells=5000, plot_flag=True):
    """Plot cell type distribution for both datasets"""
    if not plot_flag:
        return
    logger.info("Plotting cell type distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # RNA data
    subsample_n_obs = min(max_cells, len(adata_1))
    adata_1_sub = adata_1[np.random.choice(len(adata_1), subsample_n_obs, replace=False)]
    sns.countplot(data=adata_1_sub.obs, x="cell_types", ax=axes[0])
    axes[0].set_title("RNA Cell Types")
    axes[0].tick_params(axis="x", rotation=45)

    # Protein data
    subsample_n_obs = min(max_cells, len(adata_2))
    adata_2_sub = adata_2[np.random.choice(len(adata_2), subsample_n_obs, replace=False)]
    sns.countplot(data=adata_2_sub.obs, x="cell_types", ax=axes[1])
    axes[1].set_title("Protein Cell Types")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()
    plt.close()
    plt.close()


def plot_spatial_data(adata_prot, max_cells=5000, plot_flag=True):
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
    if mlflow.active_run():
        safe_mlflow_log_figure(plt.gcf(), "protein_spatial_data.pdf")
    plt.show()
    plt.close()
    plt.close()


def plot_preprocessing_results(adata_1, adata_2, plot_flag=True):
    """Plot results after preprocessing"""
    if not plot_flag:
        return
    logger.info("Plotting preprocessing results...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # RNA data (log-transformed)
    sc.pl.pca(adata_1, color="cell_types", show=False, ax=axes[0, 0])
    axes[0, 0].set_title("RNA PCA (Log-transformed)")

    sc.pl.umap(adata_1, color="cell_types", show=False, ax=axes[0, 1])
    axes[0, 1].set_title("RNA UMAP (Log-transformed)")

    # Protein data
    sc.pl.pca(adata_2, color="cell_types", show=False, ax=axes[1, 0])
    axes[1, 0].set_title("Protein PCA")

    sc.pl.umap(adata_2, color="cell_types", show=False, ax=axes[1, 1])
    axes[1, 1].set_title("Protein UMAP")

    plt.tight_layout()
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
    plt.close()


# %% VAE Plotting Functions
# This module contains functions for plotting VAE-specific visualizations.


def plot_train_val_normalized_losses(history_, plot_flag=True):
    """Plot training and validation losses normalized in the same figure.

    Args:
        history_: Dictionary containing training and validation loss histories

    This function:
    1. Uses epoch-wise means of the losses
    2. Normalizes the losses for better visualization (0-1 range)
    3. Plots training and validation losses in separate subplots for easier comparison
    4. Uses consistent colors for the same loss type across both plots
    5. Properly handles validation data collected at intervals (check_val_every_n_epoch)
    """
    if not plot_flag:
        return
    # Get all loss keys from history_
    loss_keys = [k for k in history_.keys() if "loss" in k.lower() and len(history_[k]) > 0]

    # Split into train and validation losses
    train_loss_keys = [k for k in loss_keys if k.startswith("train_")]
    val_loss_keys = [k for k in loss_keys if k.startswith("val_") and k != "val_epochs"]

    # Skip if we don't have training losses
    if not train_loss_keys:
        logger.info("No training loss data available")
        return

    # Create figure with two vertically stacked subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)

    # Dictionary to hold normalized losses
    train_normalized_losses = {}
    val_normalized_losses = {}

    # Get the validation epochs information
    val_epochs = history_.get("val_epochs", [])

    # Create epochs array for training data
    train_data_lengths = [len(history_.get(k, [])) for k in train_loss_keys]
    max_train_length = max(train_data_lengths) if train_data_lengths else 0
    train_epochs = list(range(max_train_length))
    logger.info(f"Using {len(train_epochs)} training epochs")

    # Format value helper function
    def format_value(value):
        """Format numeric values - round to integer if >= 10"""
        if abs(value) >= 10:
            return f"{int(round(value))}"
        else:
            return f"{value:.2f}"

    # Get all unique loss types (without train/val prefix)
    all_loss_types = set()
    for key in train_loss_keys + val_loss_keys:
        loss_type = key.replace("train_", "").replace("val_", "")
        all_loss_types.add(loss_type)

    # Create a fixed color mapping
    color_map = {}
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    for i, loss_type in enumerate(sorted(all_loss_types)):
        color_idx = i % len(colors)
        color_map[loss_type] = colors[color_idx]

    # Process training losses
    for train_key in train_loss_keys:
        if train_key in history_ and len(history_[train_key]) > 0:
            train_values = np.array(history_[train_key], dtype=np.float64)
            # Remove inf and nan
            train_values = train_values[~np.isinf(train_values) & ~np.isnan(train_values)]

            if len(train_values) > 0:
                loss_type = train_key.replace("train_", "")
                # Normalize to 0-1 range
                min_val = np.min(train_values)
                max_val = np.max(train_values)
                if max_val > min_val:  # Avoid division by zero
                    train_normalized_losses[loss_type] = {
                        "values": (train_values - min_val) / (max_val - min_val),
                        "min": min_val,
                        "max": max_val,
                    }

    # Plot training losses
    for loss_type, data in train_normalized_losses.items():
        label = f"{loss_type.replace('_', ' ').title()} (min:{format_value(data['min'])}, max:{format_value(data['max'])})"
        axes[0].plot(
            train_epochs[: len(data["values"])],
            data["values"],
            label=label,
            alpha=0.8,
            color=color_map.get(loss_type),
            linewidth=2,
        )

    # Process validation losses - only if we have validation data
    has_val_data = False
    for val_key in val_loss_keys:
        if val_key in history_ and len(history_[val_key]) > 0:
            val_values = np.array(history_[val_key], dtype=np.float64)
            # Remove inf and nan
            val_values = val_values[~np.isinf(val_values) & ~np.isnan(val_values)]

            if len(val_values) > 0:
                has_val_data = True
                loss_type = val_key.replace("val_", "")
                # Normalize to 0-1 range
                min_val = np.min(val_values)
                max_val = np.max(val_values)
                if max_val > min_val:  # Avoid division by zero
                    val_normalized_losses[loss_type] = {
                        "values": (val_values - min_val) / (max_val - min_val),
                        "min": min_val,
                        "max": max_val,
                    }
                else:
                    # If min == max, just plot a constant line at 0.5
                    val_normalized_losses[loss_type] = {
                        "values": np.ones_like(val_values) * 0.5,
                        "min": min_val,
                        "max": max_val,
                    }

    # Plot validation losses - even if we have just one data point
    if has_val_data:
        for loss_type, data in val_normalized_losses.items():
            # Make sure val_epochs has correct length, even if we have a single data point
            if len(val_epochs) == 0 and len(data["values"]) > 0:
                # If we have no validation epochs but we have validation data,
                # create epochs based on validation data length
                val_epochs = list(range(len(data["values"])))

            # Use the actual validation epochs from history_
            val_epochs_plot = val_epochs
            if len(val_epochs_plot) > len(data["values"]):
                val_epochs_plot = val_epochs_plot[: len(data["values"])]
            elif len(val_epochs_plot) < len(data["values"]):
                # Extend val_epochs if necessary
                val_epochs_plot = list(range(len(data["values"])))

            label = f"{loss_type.replace('_', ' ').title()} (min:{format_value(data['min'])}, max:{format_value(data['max'])})"

            # For single validation point, use a larger marker
            if len(data["values"]) == 1:
                axes[1].plot(
                    val_epochs_plot,
                    data["values"],
                    label=label,
                    alpha=0.8,
                    marker="o",
                    markersize=10,
                    color=color_map.get(loss_type),
                    linewidth=2,
                )
                # Add a text annotation showing the value
                axes[1].annotate(
                    f"{data['min']:.4f}",
                    (val_epochs_plot[0], data["values"][0]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=10,
                )
            else:
                axes[1].plot(
                    val_epochs_plot,
                    data["values"],
                    label=label,
                    alpha=0.8,
                    marker="o",
                    markersize=5,
                    color=color_map.get(loss_type),
                    linewidth=2,
                )
    else:
        axes[1].text(
            0.5,
            0.5,
            "No validation data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[1].transAxes,
            fontsize=14,
        )

    # Set titles and labels
    axes[0].set_title("Normalized Training Losses (0-1 scale)")
    axes[1].set_title("Normalized Validation Losses (0-1 scale)")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel("Normalized Loss")
    axes[1].set_ylabel("Normalized Loss")

    # Add legends outside the plots
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    if has_val_data:
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add grid
    axes[0].grid(True)
    axes[1].grid(True)

    # Add a shared y-axis limit from 0 to 1
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.05)

    # Set a fixed x-axis limit to match training epochs
    max_epoch = max(len(train_epochs) - 1, max(val_epochs) if val_epochs else 0)
    axes[0].set_xlim(-0.1, max_epoch + 0.1)

    # Adjust layout to fit the legend outside
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make space for legend

    safe_mlflow_log_figure(plt.gcf(), "train_val_normalized_losses.pdf")
    plt.close()


def plot_latent_pca_both_modalities_cn(
    rna_mean,
    protein_mean,
    adata_rna_subset,
    adata_prot_subset,
    index_rna,
    index_prot,
    global_step=None,
    use_subsample=True,
):
    # Subsample if requested - use separate sampling for RNA and protein
    if use_subsample:
        # Sample RNA data
        n_subsample_rna = min(700, len(index_rna))
        rna_subsample_idx = np.random.choice(len(index_rna), n_subsample_rna, replace=False)
        index_rna = np.array(index_rna)[rna_subsample_idx]
        rna_mean = rna_mean[rna_subsample_idx]

        # Sample protein data (separately)
        n_subsample_prot = min(700, len(index_prot))
        prot_subsample_idx = np.random.choice(len(index_prot), n_subsample_prot, replace=False)
        index_prot = np.array(index_prot)[prot_subsample_idx]
        protein_mean = protein_mean[prot_subsample_idx]

    plt.figure(figsize=(10, 5))
    pca = PCA(n_components=3)
    # concatenate the means
    combined_mean = np.concatenate([rna_mean, protein_mean], axis=0)
    pca.fit(combined_mean)
    combined_pca = pca.transform(combined_mean)
    num_rna = len(rna_mean)
    plt.subplot(1, 3, 1)
    sns.scatterplot(
        x=combined_pca[:num_rna, 0],
        y=combined_pca[:num_rna, 1],
        hue=adata_rna_subset[index_rna].obs["CN"],
    )
    plt.title("RNA")

    plt.subplot(1, 3, 2)
    sns.scatterplot(
        x=combined_pca[num_rna:, 0],
        y=combined_pca[num_rna:, 1],
        hue=adata_prot_subset[index_prot].obs["CN"],
    )
    plt.title("protein")
    plt.suptitle("PCA of latent space during training\nColor by CN label")

    ax = plt.subplot(1, 3, 3, projection="3d")
    ax.scatter(
        combined_pca[:num_rna, 0],
        combined_pca[:num_rna, 1],
        combined_pca[:num_rna, 2],
        c="red",
        label="RNA",
    )
    ax.scatter(
        combined_pca[num_rna:, 0],
        combined_pca[num_rna:, 1],
        combined_pca[num_rna:, 2],
        c="blue",
        label="protein",
        alpha=0.5,
    )

    # Only draw lines if we have equal numbers of points
    if len(rna_mean) == len(protein_mean):
        for i, (rna_point, prot_point) in enumerate(
            zip(combined_pca[:num_rna], combined_pca[num_rna:])
        ):
            if i < min(num_rna, len(combined_pca) - num_rna):  # Ensure we don't go out of bounds
                ax.plot(
                    [rna_point[0], prot_point[0]],
                    [rna_point[1], prot_point[1]],
                    [rna_point[2], prot_point[2]],
                    "k--",
                    alpha=0.6,
                    lw=0.5,
                )

    ax.set_title("merged RNA and protein")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.tight_layout()

    if global_step is not None:
        safe_mlflow_log_figure(plt.gcf(), f"latent_pca_both_modalities_step_{global_step:05d}.pdf")
    else:
        safe_mlflow_log_figure(plt.gcf(), "latent_pca_both_modalities.pdf")
    plt.show()
    plt.close()
    plt.close()


def plot_counterfactual_comparison(
    adata_rna,
    adata_prot,
    counterfactual_adata_rna,
    counterfactual_protein_adata,
    epoch="",
    use_subsample=True,
    rna_similarity_score=None,
    protein_similarity_score=None,
    rna_ilisi_score=None,
    protein_ilisi_score=None,
    plot_flag=True,
):
    """
    Plot a comparison of real RNA and counterfactual RNA data.

    Args:
        adata_rna: AnnData object for real RNA data
        adata_prot: AnnData object for counterfactual RNA data
    """
    if not plot_flag:
        return
    # Create AnnData objects for real RNA and counterfactual RNA (from protein)
    # Add a column to identify the modality
    # Subsample if requested - use separate sampling for RNA and protein
    if use_subsample:

        # Find the minimum size across all datasets to ensure valid indexing
        min_rna_size = min(len(adata_rna), len(counterfactual_adata_rna))
        min_prot_size = min(len(adata_prot), len(counterfactual_protein_adata))

        # Sample RNA data using integer indices
        n_subsample_rna = min(2000, min_rna_size)
        if n_subsample_rna > 0:
            rna_subsample_idx = np.random.choice(min_rna_size, n_subsample_rna, replace=False)
        else:
            print("Warning: No cells available for RNA subsampling")
            return

        # Sample protein data using integer indices
        n_subsample_prot = min(2000, min_prot_size)
        if n_subsample_prot > 0:
            prot_subsample_idx = np.random.choice(min_prot_size, n_subsample_prot, replace=False)
        else:
            print("Warning: No cells available for protein subsampling")
            return

        # Use integer indexing for all datasets
        counterfactual_adata_rna = counterfactual_adata_rna[rna_subsample_idx].copy()
        counterfactual_protein_adata = counterfactual_protein_adata[prot_subsample_idx].copy()
        adata_rna = adata_rna[rna_subsample_idx].copy()
        adata_prot = adata_prot[prot_subsample_idx].copy()

    rna_real_adata = adata_rna
    rna_real_adata.obs["data_type"] = "Real RNA"
    counterfactual_adata_rna.obs["data_type"] = "Counterfactual RNA"

    # Combine both datasets
    adata_combined = sc.concat([rna_real_adata, counterfactual_adata_rna], join="outer")

    # Apply log transformation to the combined RNA data before processing
    print("DEBUG: About to apply log1p to RNA combined data (line 917)")
    print(
        f"DEBUG: RNA data shape: {adata_combined.shape}, X range: [{adata_combined.X.min():.4f}, {adata_combined.X.max():.4f}]"
    )
    sc.pp.log1p(adata_combined)
    print("DEBUG: Applied log1p to RNA combined data (line 917)")

    # Process the combined data
    sc.pp.pca(adata_combined)
    sc.pp.neighbors(adata_combined)
    sc.tl.umap(adata_combined)

    # Create the 3 subplots for UMAP
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)

    # Plot 1: Color by modality (real vs counterfactual)
    sc.pl.umap(
        adata_combined,
        color="data_type",
        title="Real RNA vs Counterfactual RNA"
        + (
            f"\nSim: {rna_similarity_score}, iLISI: {rna_ilisi_score}"
            if rna_similarity_score is not None and rna_ilisi_score is not None
            else ""
        ),
        palette={"Real RNA": "#1f77b4", "Counterfactual RNA": "#ff7f0e"},
        ax=axes[0],
        show=False,
    )

    # Plot 2: Color by cell neighborhood (CN)
    sc.pl.umap(adata_combined, color="CN", title="Cell Neighborhoods", ax=axes[1], show=False)

    # Plot 3: Color by cell types
    sc.pl.umap(adata_combined, color="cell_types", title="Cell Types", ax=axes[2], show=False)

    # Also save as MLflow artifact if MLflow is active
    safe_mlflow_log_figure(
        plt.gcf(), f"counterfactual/rna_counterfactual_comparison_umap_epoch_{epoch:04d}.pdf"
    )

    # Create the 3 subplots for PCA
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)

    # Plot 1: Color by modality (real vs counterfactual)
    sc.pl.pca(
        adata_combined,
        color="data_type",
        title="Real RNA vs Counterfactual RNA"
        + (
            f"\nSim: {rna_similarity_score}, iLISI: {rna_ilisi_score}"
            if rna_similarity_score is not None and rna_ilisi_score is not None
            else ""
        ),
        palette={"Real RNA": "#1f77b4", "Counterfactual RNA": "#ff7f0e"},
        ax=axes[0],
        show=False,
    )

    # Plot 2: Color by cell neighborhood (CN)
    sc.pl.pca(adata_combined, color="CN", title="Cell Neighborhoods", ax=axes[1], show=False)

    # Plot 3: Color by cell types
    sc.pl.pca(adata_combined, color="cell_types", title="Cell Types", ax=axes[2], show=False)

    # Also save as MLflow artifact if MLflow is active
    safe_mlflow_log_figure(
        plt.gcf(), f"counterfactual/rna_counterfactual_comparison_pca_epoch_{epoch:04d}.pdf"
    )

    # do the same for the protein
    protein_real_adata = adata_prot
    protein_real_adata.obs["data_type"] = "Real Protein"
    counterfactual_protein_adata.obs["data_type"] = "Counterfactual Protein"

    # Combine both datasets
    adata_combined = sc.concat([protein_real_adata, counterfactual_protein_adata], join="outer")

    # Apply log transformation to the combined protein data before processing
    print("DEBUG: About to apply log1p to Protein combined data (line 991)")
    print(
        f"DEBUG: Protein data shape: {adata_combined.shape}, X range: [{adata_combined.X.min():.4f}, {adata_combined.X.max():.4f}]"
    )
    sc.pp.log1p(adata_combined)
    print("DEBUG: Applied log1p to Protein combined data (line 991)")

    sc.pp.pca(adata_combined)
    sc.pp.neighbors(adata_combined)
    sc.tl.umap(adata_combined)

    # Create the 3 subplots for UMAP
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)

    # Plot 1: Color by modality (real vs counterfactual)
    sc.pl.umap(
        adata_combined,
        color="data_type",
        title="Real Protein vs Counterfactual Protein"
        + (
            f"\nSim: {protein_similarity_score}, iLISI: {protein_ilisi_score}"
            if protein_similarity_score is not None and protein_ilisi_score is not None
            else ""
        ),
        palette={"Real Protein": "#1f77b4", "Counterfactual Protein": "#ff7f0e"},
        ax=axes[0],
        show=False,
    )

    # Plot 2: Color by cell neighborhood (CN)
    sc.pl.umap(adata_combined, color="CN", title="Cell Neighborhoods", ax=axes[1], show=False)

    # Plot 3: Color by cell types
    sc.pl.umap(adata_combined, color="cell_types", title="Cell Types", ax=axes[2], show=False)

    # Also save as MLflow artifact if MLflow is active
    safe_mlflow_log_figure(
        plt.gcf(), f"counterfactual/protein_counterfactual_comparison_umap_epoch_{epoch:04d}.pdf"
    )

    # Create the 3 subplots for PCA
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)

    # Plot 1: Color by modality (real vs counterfactual)
    sc.pl.pca(
        adata_combined,
        color="data_type",
        title="Real Protein vs Counterfactual Protein"
        + (
            f"\nSim: {protein_similarity_score}, iLISI: {protein_ilisi_score}"
            if protein_similarity_score is not None and protein_ilisi_score is not None
            else ""
        ),
        palette={"Real Protein": "#1f77b4", "Counterfactual Protein": "#ff7f0e"},
        ax=axes[0],
        show=False,
    )

    # Plot 2: Color by cell neighborhood (CN)
    sc.pl.pca(adata_combined, color="CN", title="Cell Neighborhoods", ax=axes[1], show=False)

    # Plot 3: Color by cell types
    sc.pl.pca(adata_combined, color="cell_types", title="Cell Types", ax=axes[2], show=False)

    # Also save as MLflow artifact if MLflow is active
    safe_mlflow_log_figure(
        plt.gcf(), f"counterfactual/protein_counterfactual_comparison_pca_epoch_{epoch:04d}.pdf"
    )


def plot_latent_pca_both_modalities_by_celltype(
    adata_rna_subset,
    adata_prot_subset,
    rna_latent,
    prot_latent,
    index_rna=None,
    index_prot=None,
    global_step=None,
    use_subsample=True,
):
    """Plot PCA of latent space colored by cell type."""
    if index_rna is None:
        index_rna = range(len(rna_latent))
    if index_prot is None:
        index_prot = range(len(prot_latent))

    # Ensure indices are within bounds

    # Subsample if requested - use separate sampling for RNA and protein
    if use_subsample:
        # Sample RNA data
        n_subsample_rna = min(1000, len(index_rna))
        rna_subsample_idx = np.random.choice(len(index_rna), n_subsample_rna, replace=False)
        index_rna = np.array(index_rna)[rna_subsample_idx]
        rna_latent = rna_latent[rna_subsample_idx]
        # Sample protein data (separately)
        n_subsample_prot = min(1000, len(index_prot))
        prot_subsample_idx = np.random.choice(len(index_prot), n_subsample_prot, replace=False)
        index_prot = np.array(index_prot)[prot_subsample_idx]
        prot_latent = prot_latent[prot_subsample_idx]

    # Ensure all indices are valid
    if len(index_rna) == 0 or len(index_prot) == 0:
        logger.warning("Warning: No valid indices for plotting. Skipping plot.")
        return

    num_rna = len(index_rna)

    combined_latent = np.vstack([rna_latent, prot_latent])
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined_latent)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    if "cell_type" in adata_rna_subset.obs:
        hue_col = "cell_type"
    elif "cell_types" in adata_rna_subset.obs:
        hue_col = "cell_types"
    else:
        hue_col = "major_cell_types"

    sns.scatterplot(
        x=combined_pca[:num_rna, 0],
        y=combined_pca[:num_rna, 1],
        hue=adata_rna_subset[index_rna].obs[hue_col],
    )
    plt.title("RNA")

    plt.subplot(1, 2, 2)
    if "cell_type" in adata_prot_subset.obs:
        hue_col = "cell_type"
    elif "cell_types" in adata_prot_subset.obs:
        hue_col = "cell_types"
    else:
        hue_col = "major_cell_types"

    sns.scatterplot(
        x=combined_pca[num_rna:, 0],
        y=combined_pca[num_rna:, 1],
        hue=adata_prot_subset[index_prot].obs[hue_col],
    )
    plt.title("protein")
    plt.suptitle("PCA of latent space during training\nColor by cell type")
    plt.tight_layout()

    if global_step is not None:
        safe_mlflow_log_figure(plt.gcf(), f"latent_pca_celltype_step_{global_step:05d}.pdf")
    else:
        safe_mlflow_log_figure(plt.gcf(), "latent_pca_celltype.pdf")
    plt.show()
    plt.close()
    plt.close()


def plot_latent_mean_std_legacy(
    rna_inference_outputs,
    protein_inference_outputs,
    adata_rna,
    adata_prot,
    index_rna=None,
    index_prot=None,
    use_subsample=True,
):
    """Plot latent space visualization combining heatmaps and PCA plots.

    Args:
        rna_inference_outputs: RNA inference outputs containing qz means and scales
        protein_inference_outputs: Protein inference outputs containing qz means and scales
        adata_rna: RNA AnnData object
        adata_prot: Protein AnnData object
        index_rna: Indices for RNA data (optional)
        index_prot: Indices for protein data (optional)
        use_subsample: Whether to subsample to 700 points (default: True)
    """
    if index_rna is None:
        index_rna = range(len(adata_rna.obs.index))
    if index_prot is None:
        index_prot = range(len(adata_prot.obs.index))

    # Convert tensors to numpy if needed
    rna_mean = rna_inference_outputs["qz"].mean.detach().cpu().numpy()
    protein_mean = protein_inference_outputs["qz"].mean.detach().cpu().numpy()
    rna_std = rna_inference_outputs["qz"].scale.detach().cpu().numpy()
    protein_std = protein_inference_outputs["qz"].scale.detach().cpu().numpy()

    # Subsample if requested - use separate sampling for RNA and protein
    if use_subsample:
        # Sample RNA data
        n_subsample_rna = min(700, len(index_rna))
        rna_subsample_idx = np.random.choice(len(index_rna), n_subsample_rna, replace=False)
        index_rna = np.array(index_rna)[rna_subsample_idx]
        rna_mean = rna_mean[rna_subsample_idx]
        rna_std = rna_std[rna_subsample_idx]

        # Sample protein data (separately)
        n_subsample_prot = min(700, len(index_prot))
        prot_subsample_idx = np.random.choice(len(index_prot), n_subsample_prot, replace=False)
        index_prot = np.array(index_prot)[prot_subsample_idx]
        protein_mean = protein_mean[prot_subsample_idx]
        protein_std = protein_std[prot_subsample_idx]

    # Plot heatmaps
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    sns.heatmap(rna_mean)
    plt.title("RNA Mean Latent Space")

    plt.subplot(122)
    sns.heatmap(protein_mean)
    plt.title("Protein Mean Latent Space")
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    sns.heatmap(rna_std)
    plt.title("RNA Std Latent Space")

    plt.subplot(122)
    sns.heatmap(protein_std)
    plt.title("Protein Std Latent Space")
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.close()

    # Create AnnData objects for PCA visualization
    rna_ann = AnnData(X=rna_mean, obs=adata_rna.obs.iloc[index_rna].copy())
    protein_ann = AnnData(X=protein_mean, obs=adata_prot.obs.iloc[index_prot].copy())

    # Plot PCA and distributions
    plt.figure(figsize=(15, 5))

    # RNA PCA
    plt.subplot(131)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(rna_ann.X)

    df = pd.DataFrame(
        {
            "PC1": pca_result[:, 0],
            "PC2": pca_result[:, 1],
            "CN": rna_ann.obs["CN"],  # Add the CN column
        }
    )
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="CN")
    plt.title("RNA Latent Space PCA")

    # Protein PCA
    plt.subplot(132)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(protein_ann.X)
    df = pd.DataFrame(
        {
            "PC1": pca_result[:, 0],
            "PC2": pca_result[:, 1],
            "CN": protein_ann.obs["CN"],  # Add the CN column
        }
    )
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="CN")
    plt.title("Protein Latent Space PCA")

    # Standard deviation distributions
    plt.subplot(133)
    plt.hist(rna_std.flatten(), bins=50, alpha=0.5, label="RNA", density=True)
    plt.hist(protein_std.flatten(), bins=50, alpha=0.5, label="Protein", density=True)
    plt.title("Latent Space Standard Deviations")
    plt.xlabel("Standard Deviation")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()
    plt.close()


def plot_rna_protein_matching_means_and_scale(
    rna_latent_mean,
    protein_latent_mean,
    rna_latent_std,
    protein_latent_std,
    archetype_dis_mat,
    use_subsample=True,
    global_step=None,
    plot_flag=True,
):
    """
    Plot the means and scales as halo  and lines between the best matches
    of the RNA and protein
    Args:
        rna_inference_outputs: the output of the RNA inference
        protein_inference_outputs: the output of the protein inference
        archetype_dis_mat: the archetype distance matrix
        use_subsample: whether to use subsampling
        global_step: the current training step, if None then not during training
    """
    if not plot_flag:
        return
    if use_subsample:
        rna_subsample_idx = np.random.choice(
            rna_latent_mean.shape[0], min(700, rna_latent_mean.shape[0]), replace=False
        )
        protein_subsample_idx = np.random.choice(
            protein_latent_mean.shape[0],
            min(700, protein_latent_mean.shape[0]),
            replace=False,
        )
    else:
        rna_subsample_idx = np.arange(rna_latent_mean.shape[0])
        protein_subsample_idx = np.arange(protein_latent_mean.shape[0])
    prot_new_order = archetype_dis_mat.argmin(axis=0).detach().cpu().numpy()

    rna_means = rna_latent_mean[rna_subsample_idx]
    rna_scales = rna_latent_std[rna_subsample_idx]
    protein_means = protein_latent_mean[prot_new_order][protein_subsample_idx]
    protein_scales = protein_latent_std[prot_new_order][protein_subsample_idx]
    # match the order of the means to the archetype_dis
    # Combine means for PCA
    combined_means = np.concatenate([rna_means, protein_means], axis=0)

    # Fit PCA on means
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_means)

    # Transform scales using the same PCA transformation
    combined_scales = np.concatenate([rna_scales, protein_scales], axis=0)
    scales_transformed = pca.transform(combined_scales)

    # Plot with halos
    fig = plt.figure(figsize=(8, 6))

    # Plot RNA points and halos
    for i in range(rna_means.shape[0]):
        # Add halo using scale information
        circle = plt.Circle(
            (pca_result[i, 0], pca_result[i, 1]),
            radius=np.linalg.norm(scales_transformed[i]) * 0.05,
            color="blue",
            alpha=0.1,
        )
        plt.gca().add_patch(circle)
    # Plot Protein points and halos
    for i in range(protein_means.shape[0]):
        # Add halo using scale information
        circle = plt.Circle(
            (
                pca_result[rna_means.shape[0] + i, 0],
                pca_result[rna_means.shape[0] + i, 1],
            ),
            radius=np.linalg.norm(scales_transformed[rna_means.shape[0] + i]) * 0.05,
            color="orange",
            alpha=0.1,
        )
        plt.gca().add_patch(circle)

    # Add connecting lines
    for i in range(rna_means.shape[0]):
        color = "red" if (i % 2 == 0) else "green"
        plt.plot(
            [pca_result[i, 0], pca_result[rna_means.shape[0] + i, 0]],
            [pca_result[i, 1], pca_result[rna_means.shape[0] + i, 1]],
            "k-",
            alpha=0.2,
            color=color,
        )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA of RNA and Protein with Scale Halos")
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.tight_layout()

    if global_step is not None:
        safe_mlflow_log_figure(
            plt.gcf(), f"rna_protein_matching_means_and_scale_step_{global_step:05d}.pdf"
        )
    else:
        safe_mlflow_log_figure(plt.gcf(), "rna_protein_matching_means_and_scale.pdf")
    plt.show()
    plt.close()
    plt.close(fig)


def plot_inference_outputs(
    rna_inference_outputs,
    protein_inference_outputs,
    latent_distances,
    rna_distances,
    prot_distances,
):
    """Plot inference outputs"""
    logger.info("Plotting inference outputs...")
    fig, axes = plt.subplots(2, 3)

    # Plot latent distances
    axes[0, 0].hist(latent_distances.detach().cpu().numpy().flatten(), bins=50)
    axes[0, 0].set_title("Latent Distances")

    # Plot RNA distances
    axes[0, 1].hist(rna_distances.detach().cpu().numpy().flatten(), bins=50)
    axes[0, 1].set_title("RNA Distances")

    # Plot protein distances
    axes[0, 2].hist(prot_distances.detach().cpu().numpy().flatten(), bins=50)
    axes[0, 2].set_title("Protein Distances")

    # Plot latent vs RNA distances
    axes[1, 0].scatter(
        rna_distances.detach().cpu().numpy().flatten(),
        latent_distances.detach().cpu().numpy().flatten(),
        alpha=0.1,
    )
    axes[1, 0].set_title("Latent vs RNA Distances")

    # Plot latent vs protein distances
    axes[1, 1].scatter(
        prot_distances.detach().cpu().numpy().flatten(),
        latent_distances.detach().cpu().numpy().flatten(),
        alpha=0.1,
    )
    axes[1, 1].set_title("Latent vs Protein Distances")

    # Plot RNA vs protein distances
    axes[1, 2].scatter(
        rna_distances.detach().cpu().numpy().flatten(),
        prot_distances.detach().cpu().numpy().flatten(),
        alpha=0.1,
    )
    axes[1, 2].set_title("RNA vs Protein Distances")

    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "inference_outputs.pdf")
    plt.show()
    plt.close()
    plt.close()


def plot_similarity_loss_history(
    similarity_loss_all_history,
    active_similarity_loss_active_history,
    global_step,
    similarity_weight=None,
    similarity_dynamic=None,
    plot_flag=True,
):
    """
    Plot the similarity loss history_ and highlight active steps
    """
    if not plot_flag:
        return
    # Skip plotting if similarity weight is zero or similarity dynamic is false
    if similarity_weight is not None and similarity_weight == 0:
        return
    if similarity_dynamic is not None and not similarity_dynamic:
        return

    if len(similarity_loss_all_history) < 10:
        return
    plt.figure()
    colors = [
        "red" if active else "blue" for active in active_similarity_loss_active_history[-1000:]
    ]
    num_samples = len(similarity_loss_all_history[-1000:])
    dot_size = max(1, 1000 // num_samples)  # Adjust dot size based on the number of samples
    plt.scatter(
        np.arange(num_samples),
        similarity_loss_all_history[-1000:],
        c=colors,
        s=dot_size,
    )
    plt.title(f"step_{global_step:05d} Similarity loss history_ (last 1000 steps)")
    plt.xlabel("Step")
    plt.ylabel("Similarity Loss")
    plt.xticks(
        np.arange(0, num_samples, step=max(1, num_samples // 10)),
        np.arange(
            max(0, len(similarity_loss_all_history) - 1000),
            len(similarity_loss_all_history),
            step=max(1, num_samples // 10),
        ),
    )
    red_patch = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="red",
        markersize=10,
        label="Active",
        alpha=0.5,
    )
    blue_patch = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="blue",
        markersize=10,
        label="Inactive",
        alpha=0.5,
    )
    plt.legend(handles=[red_patch, blue_patch])
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), f"similarity_loss_history_step_{global_step:05d}.pdf")
    plt.close()


def plot_losses(keys, title, history_=None):
    """Plot normalized losses for given keys."""
    plt.figure(figsize=(10, 5))
    normalized_losses = {}
    labels = {}

    for key in keys:
        if history_ is None:
            # Assume keys are already values
            values = key if isinstance(key, (list, np.ndarray)) else [key]
        else:
            values = history_[key]
        if len(values) > 1:  # Only process if we have more than 1 value
            values = np.array(values[1:])  # Skip first step
            # Remove inf and nan
            values = values[~np.isinf(values) & ~np.isnan(values)]
            if len(values) > 0:  # Check again after filtering
                min_val = np.min(values)
                max_val = np.max(values)
                label = f"{key} min: {min_val:.0f} max: {max_val:.0f}"
                labels[key] = label
                if max_val > min_val:  # Avoid division by zero
                    normalized_losses[key] = (values - min_val) / (max_val - min_val)

    # Plot each normalized loss
    for key, values in normalized_losses.items():
        if "total" in key.lower():
            plt.plot(values, label=labels[key], alpha=0.7, linestyle="--")
        else:
            plt.plot(values, label=labels[key], alpha=0.7)

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Normalized Loss")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), f"{title.lower().replace(' ', '_')}.pdf")
    plt.show()
    plt.close()


def plot_normalized_losses(history_):
    """Plot normalized training and validation losses in separate figures."""
    # Get all loss keys from history_
    loss_keys = [k for k in history_.keys() if "loss" in k.lower() and len(history_[k]) > 0]

    # Split into train and validation losses
    train_loss_keys = [k for k in loss_keys if k.startswith("train_") and "adv" not in k.lower()]
    val_loss_keys = [
        k
        for k in loss_keys
        if (k.startswith("val_") or k.startswith("validation_")) and "adv" not in k.lower()
    ]

    # Function to normalize and plot losses
    def _plot_losses(keys, title):
        plt.figure(figsize=(10, 5))
        normalized_losses = {}
        labels = {}

        for key in keys:
            values = history_[key]
            if len(values) > 1:  # Only process if we have more than 1 value
                values = np.array(values[1:])  # Skip first step
                # Remove inf and nan
                values = values[~np.isinf(values) & ~np.isnan(values)]
                if len(values) > 0:  # Check again after filtering
                    min_val = np.min(values)
                    max_val = np.max(values)
                    label = f"{key} min: {min_val:.0f} max: {max_val:.0f}"
                    labels[key] = label
                    if max_val > min_val:  # Avoid division by zero
                        normalized_losses[key] = (values - min_val) / (max_val - min_val)

        # Plot each normalized loss
        for key, values in normalized_losses.items():
            if "total" in key.lower():
                plt.plot(values, label=labels[key], alpha=0.7, linestyle="--")
            else:
                plt.plot(values, label=labels[key], alpha=0.7)

        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel("Normalized Loss")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        safe_mlflow_log_figure(plt.gcf(), f"{title.lower().replace(' ', '_')}.pdf")
        plt.show()
        plt.close()

    # Plot training losses
    if train_loss_keys:
        _plot_losses(train_loss_keys, "Normalized Training Losses")

    # Plot validation losses
    if val_loss_keys:
        _plot_losses(val_loss_keys, "Normalized Validation Losses")


def plot_cell_type_prediction_confusion_matrix(
    true_cell_types=None, predicted_cell_types=None, global_step=None
):
    if true_cell_types is None or predicted_cell_types is None:  # show default confusion matrix
        confusion_matrix_values = np.array(
            [
                [91, 4, 3, 0, 0, 0],
                [22, 77, 0, 0, 0, 0],
                [14, 1, 79, 4, 0, 0],
                [13, 1, 3, 70, 0, 0],
                [11, 1, 11, 0, 65, 2],
                [10, 2, 0, 0, 2, 70],
                [52, 17, 16, 16, 10, 0],
            ]
        )
        x_labels = ["B-CD22-CD40", "B-Ki67", "CD4 T", "CD8 T", "DC", "Plasma"]
        y_labels = [
            "B-CD22-CD40",
            "B-Ki67",
            "CD4 T",
            "CD8 T",
            "DC",
            "Plasma",
            "Unknown",
        ]
        confusion_matrix_df = pd.DataFrame(
            confusion_matrix_values, index=y_labels, columns=x_labels
        )
    else:
        # use pandas to get the confusion matrix instead of sklearn
        confusion_matrix_df = pd.crosstab(
            true_cell_types.values,
            predicted_cell_types.values,
            rownames=["True"],
            colnames=["Predicted"],
            margins=False,
        )
    if len(set(true_cell_types)) > 8:
        fontsize_ticks = 16
        fontsize_percentage = 17
    else:
        fontsize_ticks = 30
        fontsize_percentage = 25
    percentages = (confusion_matrix_df / confusion_matrix_df.sum(axis=0)) * 100
    df_percentages = percentages.copy()
    # todo temp save percentages to csv
    path = "zzzz_cell_type_accuracy_percentages.csv"
    df_percentages.to_csv(path)
    from pathlib import Path

    print(f"Saved percentages to csv in {Path(path).absolute()}")
    # Create the figure with size ratio matching the image
    plt.figure(figsize=(12, 10))

    # Create the heatmap with percentage values and no color bar
    ax = sns.heatmap(
        percentages,
        annot=True,
        fmt=".0f",
        cmap="viridis",
        cbar=False,
        annot_kws={"fontsize": fontsize_percentage, "fontweight": "bold"},
    )

    # Add '%' sign to annotations
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")

    # Style adjustments with x-ticks rotated at 45 degrees
    plt.xticks(fontsize=fontsize_ticks, rotation=45)  # 45 degree rotation on x-axis labels
    plt.yticks(fontsize=fontsize_ticks, rotation=45)

    # Add some padding at the bottom to accommodate rotated labels
    plt.tight_layout(pad=1.5)

    # Display the plot
    # Optional: To save the figure
    if global_step is not None:
        safe_mlflow_log_figure(plt.gcf(), f"metrics/step_{global_step:05d}_cell_type_accuracy.pdf")
    else:
        safe_mlflow_log_figure(plt.gcf(), "metrics/cell_type_accuracy.pdf")
    plt.show()
    plt.close()
    plt.close()


def plot_cn_prediction_confusion_matrix(
    true_cn=None, predicted_cn=None, cell_types=None, global_step=None
):
    """
    Plot confusion matrix for CN (cell neighborhood) prediction accuracy.
    Similar to plot_cell_type_prediction_confusion_matrix but for CN labels.

    Args:
        true_cn: True CN labels (pandas Series or array-like)
        predicted_cn: Predicted CN labels (pandas Series or array-like)
        cell_types: Cell type labels for per-cell-type analysis (optional)
        global_step: Current global step for saving (optional, only plots if provided)
    """
    if true_cn is None or predicted_cn is None:
        print("Error: Both true_cn and predicted_cn must be provided")
        return

    # Create confusion matrix using pandas crosstab
    confusion_matrix_df = pd.crosstab(
        true_cn.values if hasattr(true_cn, "values") else true_cn,
        predicted_cn.values if hasattr(predicted_cn, "values") else predicted_cn,
        rownames=["True"],
        colnames=["Predicted"],
        margins=False,
    )

    # Adjust font sizes based on number of CN categories
    if len(set(true_cn)) > 8:
        fontsize_ticks = 16
        fontsize_percentage = 17
    else:
        fontsize_ticks = 30
        fontsize_percentage = 25

    # Calculate percentages (normalize by column - predicted)
    percentages = (confusion_matrix_df / confusion_matrix_df.sum(axis=0)) * 100

    # Create the figure
    plt.figure(figsize=(12, 10))

    # Create the heatmap with percentage values and no color bar
    ax = sns.heatmap(
        percentages,
        annot=True,
        fmt=".0f",
        cmap="viridis",
        cbar=False,
        annot_kws={"fontsize": fontsize_percentage, "fontweight": "bold"},
    )

    # Add '%' sign to annotations
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")

    # Style adjustments with x-ticks rotated at 45 degrees
    plt.xticks(fontsize=fontsize_ticks, rotation=45)
    plt.yticks(fontsize=fontsize_ticks, rotation=45)

    # Add some padding at the bottom to accommodate rotated labels
    plt.tight_layout(pad=1.5)

    # Save the figure
    if global_step is not None:
        safe_mlflow_log_figure(plt.gcf(), f"metrics/step_{global_step:05d}_cn_accuracy.pdf")
    else:
        safe_mlflow_log_figure(plt.gcf(), "metrics/cn_accuracy.pdf")
    plt.show()

    # If cell types are provided, create per-cell-type confusion matrices
    if cell_types is not None:
        plot_cn_prediction_per_cell_type(true_cn, predicted_cn, cell_types, global_step)


def plot_cn_prediction_per_cell_type(true_cn, predicted_cn, cell_types, global_step=None):
    """
    Plot CN prediction confusion matrices for each cell type in separate subplots.

    Args:
        true_cn: True CN labels (pandas Series or array-like)
        predicted_cn: Predicted CN labels (pandas Series or array-like)
        cell_types: Cell type labels (pandas Series or array-like)
        global_step: Current global step for saving (optional)
    """
    # Convert to pandas Series if needed
    if not isinstance(true_cn, pd.Series):
        true_cn = pd.Series(true_cn)
    if not isinstance(predicted_cn, pd.Series):
        predicted_cn = pd.Series(predicted_cn)
    if not isinstance(cell_types, pd.Series):
        cell_types = pd.Series(cell_types)

    # Get unique cell types
    unique_cell_types = sorted(cell_types.unique())
    n_cell_types = len(unique_cell_types)

    # Calculate grid dimensions
    n_cols = min(3, n_cell_types)
    n_rows = int(np.ceil(n_cell_types / n_cols))

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows))
    if n_cell_types == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Get all unique CN labels for consistent axes
    all_cn_labels = sorted(set(true_cn.unique()) | set(predicted_cn.unique()))

    # Plot confusion matrix for each cell type
    for idx, cell_type in enumerate(unique_cell_types):
        ax = axes[idx]

        # Filter data for this cell type
        mask = cell_types == cell_type
        true_cn_ct = true_cn[mask]
        predicted_cn_ct = predicted_cn[mask]

        # Skip if no data
        if len(true_cn_ct) == 0:
            ax.axis("off")
            continue

        # Create confusion matrix
        confusion_matrix_df = pd.crosstab(
            true_cn_ct,
            predicted_cn_ct,
            rownames=["True"],
            colnames=["Predicted"],
            margins=False,
        )

        # Reindex to include all CN labels (fill missing with 0)
        confusion_matrix_df = confusion_matrix_df.reindex(
            index=all_cn_labels, columns=all_cn_labels, fill_value=0
        )

        # Calculate percentages (normalize by column)
        col_sums = confusion_matrix_df.sum(axis=0)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        percentages = (confusion_matrix_df / col_sums) * 100

        # Calculate accuracy for this cell type
        correct = (true_cn_ct == predicted_cn_ct).sum()
        accuracy = correct / len(true_cn_ct) * 100

        # Adjust font sizes based on number of CN categories
        if len(all_cn_labels) > 8:
            fontsize_ticks = 8
            fontsize_percentage = 9
        else:
            fontsize_ticks = 10
            fontsize_percentage = 11

        # Create heatmap
        sns.heatmap(
            percentages,
            annot=True,
            fmt=".0f",
            cmap="viridis",
            cbar=False,
            ax=ax,
            annot_kws={"fontsize": fontsize_percentage, "fontweight": "bold"},
        )

        # Add '%' sign to annotations
        for text in ax.texts:
            text.set_text(f"{text.get_text()}%")

        # Set title with cell type and accuracy
        ax.set_title(
            f"{cell_type}\n(n={len(true_cn_ct)}, acc={accuracy:.1f}%)",
            fontsize=fontsize_ticks + 2,
            fontweight="bold",
        )

        # Style adjustments
        ax.set_xlabel("Predicted", fontsize=fontsize_ticks)
        ax.set_ylabel("True", fontsize=fontsize_ticks)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fontsize_ticks)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=fontsize_ticks)

    # Hide unused subplots
    for idx in range(n_cell_types, len(axes)):
        axes[idx].axis("off")

    # Add overall title
    fig.suptitle(
        "CN Prediction Accuracy per Cell Type",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()

    # Save the figure
    if global_step is not None:
        safe_mlflow_log_figure(fig, f"metrics/step_{global_step:05d}_cn_accuracy_per_cell_type.pdf")
    else:
        safe_mlflow_log_figure(fig, "metrics/cn_accuracy_per_cell_type.pdf")

    plt.show()
    plt.close()


def plot_pca_umap_latent_space_during_train(
    prefix, combined_latent, epoch, global_step="", plot_flag=True
):
    if not plot_flag:
        return
    sc.pp.pca(combined_latent)

    # Helper function to create scatter plot with different markers for extreme archetypes
    def plot_with_extreme_markers(ax, combined_latent, color_col, title, basis="X_pca"):
        # Check if extreme archetype information exists
        if "is_extreme_archetype" in combined_latent.obs.columns:
            extreme_mask = combined_latent.obs["is_extreme_archetype"].astype(bool)
            non_extreme_mask = ~extreme_mask

            # Get coordinates
            coords = combined_latent.obsm[basis]

            # Get unique categories for color mapping
            categories = (
                combined_latent.obs[color_col].cat.categories
                if pd.api.types.is_categorical_dtype(combined_latent.obs[color_col])
                else combined_latent.obs[color_col].unique()
            )

            # Use specific colors for modalities, default tab10 for others
            if color_col == "modality":
                # Use blue and orange for clear distinction
                color_map = {"RNA": "#1f77b4", "Protein": "#ff7f0e"}  # blue and orange
            else:
                colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
                color_map = dict(zip(categories, colors))

            # Plot non-extreme cells with 'o' marker
            legend_handles = []
            legend_labels = []

            if non_extreme_mask.sum() > 0:
                for category in categories:
                    cat_mask = (combined_latent.obs[color_col] == category) & non_extreme_mask
                    if cat_mask.sum() > 0:
                        scatter = ax.scatter(
                            coords[cat_mask, 0],
                            coords[cat_mask, 1],
                            c=[color_map[category]],
                            marker="o",
                            s=20,
                            alpha=0.6,
                        )
                        # Add to legend (only non-extreme, as if only using circles)
                        legend_handles.append(scatter)
                        legend_labels.append(str(category))

            # Plot extreme cells with 'x' marker (no legend entries)
            if extreme_mask.sum() > 0:
                for category in categories:
                    cat_mask = (combined_latent.obs[color_col] == category) & extreme_mask
                    if cat_mask.sum() > 0:
                        ax.scatter(
                            coords[cat_mask, 0],
                            coords[cat_mask, 1],
                            c=[color_map[category]],
                            marker="x",
                            s=80,
                            alpha=1.0,
                        )

            # Add main legend for categories
            if len(categories) <= 10 and legend_handles:
                main_legend = ax.legend(
                    legend_handles,
                    legend_labels,
                    fontsize=8,
                    title="Categories",
                    loc="upper right",
                    bbox_to_anchor=(1.0, 1.0),
                )
                ax.add_artist(main_legend)  # Keep this legend when adding the second one

            # Note: Marker legend is now created globally for the entire figure

        else:
            # Fallback to scanpy plotting if extreme archetype info not available
            if basis == "X_pca":
                sc.pl.pca(
                    combined_latent,
                    color=color_col,
                    ax=ax,
                    show=False,
                    title=title,
                    legend_loc=None,
                )
            elif basis == "X_umap":
                sc.pl.umap(
                    combined_latent,
                    color=color_col,
                    ax=ax,
                    show=False,
                    title=title,
                    legend_loc=None,
                )
            return

        ax.set_title(title)
        # Set appropriate axis labels based on basis
        if basis == "X_pca":
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
        elif basis == "X_umap":
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")

    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    plot_with_extreme_markers(ax1, combined_latent, "modality", "Combined Latent PCA by Modality")

    ax2 = fig.add_subplot(1, 3, 2)
    plot_with_extreme_markers(
        ax2, combined_latent, "cell_types", "Combined Latent PCA by Cell Type"
    )

    ax3 = fig.add_subplot(1, 3, 3)
    plot_with_extreme_markers(ax3, combined_latent, "CN", "Combined Latent PCA by CN")

    # Add single global legend for marker types (extreme vs normal)
    if "is_extreme_archetype" in combined_latent.obs.columns:
        from matplotlib.lines import Line2D

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
        # Add single shared legend at the bottom of the figure
        fig.legend(
            handles=marker_legend_elements,
            fontsize=10,
            title="Markers",
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
        )

    plt.tight_layout(rect=[0, 0.12, 1, 1])  # Leave space for bottom legend
    if global_step is not None:
        pca_file = f"{prefix}_combined_latent_pca_step_{global_step:05d}.pdf"
    else:
        pca_file = f"{prefix}_combined_latent_pca.pdf"
    safe_mlflow_log_figure(plt.gcf(), f"train/{pca_file}")
    combined_latent.obsm.pop("X_pca", None)

    plt.close()
    plt.close()

    sc.pp.neighbors(combined_latent, use_rep="X")
    sc.tl.umap(combined_latent)
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    plot_with_extreme_markers(
        ax1,
        combined_latent,
        "modality",
        f"{prefix}_Combined Latent UMAP by Modality",
        basis="X_umap",
    )

    ax2 = fig.add_subplot(1, 3, 2)
    plot_with_extreme_markers(
        ax2,
        combined_latent,
        "cell_types",
        f"{prefix}_Combined Latent UMAP by Cell Type",
        basis="X_umap",
    )

    ax3 = fig.add_subplot(1, 3, 3)
    plot_with_extreme_markers(
        ax3, combined_latent, "CN", f"{prefix}_Combined Latent UMAP by CN", basis="X_umap"
    )

    # Add single global legend for marker types (extreme vs normal)
    if "is_extreme_archetype" in combined_latent.obs.columns:
        from matplotlib.lines import Line2D

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
        # Add single shared legend at the bottom of the figure
        fig.legend(
            handles=marker_legend_elements,
            fontsize=10,
            title="Markers",
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
        )

    plt.tight_layout(rect=[0, 0.12, 1, 1])  # Leave space for bottom legend
    if global_step is not None:
        umap_file = f"{prefix}_combined_latent_umap_step_{global_step:05d}.pdf"
    else:
        umap_file = f"{prefix}_combined_latent_umap.pdf"
    safe_mlflow_log_figure(plt.gcf(), f"train/{umap_file}")
    logger.info(f"{prefix} combined latent UMAP visualized and saved")

    # Create folder structure for cell type specific plots
    if global_step is not None:
        epoch_folder = f"train/cells_cn/step_{global_step:05d}"
    else:
        epoch_folder = "train/cells_cn/step_00000"

    # Get unique cell types
    unique_cell_types = combined_latent.obs["cell_types"].unique()

    # Plot UMAP for each cell type
    for cell_type in unique_cell_types:
        cell_type_locs = combined_latent.obs["cell_types"] == cell_type
        if cell_type_locs.sum() < 2:
            continue
        rna_locs = combined_latent[cell_type_locs].obs["modality"] == "RNA"
        if rna_locs.sum() < 2:
            continue
        prot_locs = combined_latent[cell_type_locs].obs["modality"] == "Protein"
        if prot_locs.sum() < 2:
            continue
        cell_type_data = combined_latent[cell_type_locs].copy()

        # Create figure for this cell type
        fig = plt.figure(figsize=(18, 6))

        # Plot RNA only UMAP colored by CN
        ax1 = fig.add_subplot(1, 3, 1)
        rna_cells = cell_type_data[cell_type_data.obs["modality"] == "RNA"].copy()
        n_labels_rna = len(rna_cells.obs["CN"].unique())
        n_samples_rna = len(rna_cells)
        if n_labels_rna >= 2 and n_samples_rna > n_labels_rna:
            rna_sil_score = silhouette_score(rna_cells.X, rna_cells.obs["CN"])
        else:
            rna_sil_score = 0

        plot_with_extreme_markers(
            ax1,
            rna_cells,
            "CN",
            f"RNA UMAP by CN - {cell_type}\nSil Score: {rna_sil_score:.3f}",
            basis="X_umap",
        )

        # Plot Protein only UMAP colored by CN
        ax2 = fig.add_subplot(1, 3, 2)
        prot_cells = cell_type_data[cell_type_data.obs["modality"] == "Protein"].copy()
        prot_sil_score = 0
        n_labels_prot = len(prot_cells.obs["CN"].unique())
        n_samples_prot = len(prot_cells)
        if n_labels_prot >= 2 and n_samples_prot > n_labels_prot:
            # Check for NaN values before calculating silhouette score
            if np.isnan(prot_cells.X).any():
                logger.warning(
                    "NaN values detected in Protein latent space, skipping silhouette score"
                )
                prot_sil_score = 0
            else:
                prot_sil_score = silhouette_score(prot_cells.X, prot_cells.obs["CN"])

        plot_with_extreme_markers(
            ax2,
            prot_cells,
            "CN",
            f"Protein UMAP by CN - {cell_type}\nSil Score: {prot_sil_score:.3f}",
            basis="X_umap",
        )

        # Plot combined UMAP colored by modality
        ax3 = fig.add_subplot(1, 3, 3)
        # Calculate combined silhouette score with NaN checking
        if len(cell_type_data) > 1:
            if np.isnan(cell_type_data.X).any():
                logger.warning(
                    "NaN values detected in combined latent space, skipping silhouette score"
                )
                combined_sil_score = 0
            else:
                combined_sil_score = silhouette_score(
                    cell_type_data.X, cell_type_data.obs["modality"]
                )
        else:
            combined_sil_score = 0

        plot_with_extreme_markers(
            ax3,
            cell_type_data,
            "modality",
            f"Combined UMAP by Modality - {cell_type}\nSil Score: {combined_sil_score:.3f}",
            basis="X_umap",
        )

        # Add single global legend for marker types (extreme vs normal)
        if "is_extreme_archetype" in cell_type_data.obs.columns:
            from matplotlib.lines import Line2D

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
            for ax in [ax1, ax2, ax3]:
                ax.legend(
                    handles=marker_legend_elements,
                    fontsize=10,
                    title="Markers",
                    ncol=2,
                )

        plt.tight_layout()

        # Save the figure with cell type in filename
        cell_type_safe = cell_type.replace(" ", "_").replace("/", "_")
        if global_step is not None:
            cell_type_file = f"{prefix}_{cell_type_safe}_umap_step_{global_step:05d}.pdf"
        else:
            cell_type_file = f"{prefix}_{cell_type_safe}_umap_step_00000.pdf"

        safe_mlflow_log_figure(plt.gcf(), f"{epoch_folder}/{cell_type_file}")
        logger.debug(f"{prefix}{cell_type} UMAP visualized and saved")
        plt.close()

    # Plot UMAP for each modality separately colored by CN
    fig = plt.figure(figsize=(18, 6))

    # RNA UMAP
    ax1 = fig.add_subplot(1, 2, 1)
    rna_latent = combined_latent[combined_latent.obs["modality"] == "RNA"]
    plot_with_extreme_markers(
        ax1, rna_latent, "CN", f"{prefix}_RNA Latent UMAP by CN", basis="X_umap"
    )

    # Protein UMAP
    ax2 = fig.add_subplot(1, 2, 2)
    prot_latent = combined_latent[combined_latent.obs["modality"] == "Protein"]
    plot_with_extreme_markers(
        ax2, prot_latent, "CN", f"{prefix}_Protein Latent UMAP by CN", basis="X_umap"
    )

    # Add single global legend for marker types (extreme vs normal)
    if "is_extreme_archetype" in combined_latent.obs.columns:
        from matplotlib.lines import Line2D

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
        # Add single shared legend at the bottom of the figure
        fig.legend(
            handles=marker_legend_elements,
            fontsize=10,
            title="Markers",
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
        )

    plt.tight_layout(rect=[0, 0.12, 1, 1])  # Leave space for bottom legend

    if global_step is not None:
        modality_cn_file = f"{prefix}_modality_cn_umap_step_{global_step:05d}.pdf"
    else:
        modality_cn_file = f"{prefix}_modality_cn_umap_step_00000.pdf"
    safe_mlflow_log_figure(plt.gcf(), f"train/{modality_cn_file}")
    logger.info(f"{prefix}Modality-specific CN UMAPs visualized and saved")

    # Plot UMAP for each modality separately colored by cell_types
    fig = plt.figure(figsize=(18, 6))

    # RNA UMAP
    ax1 = fig.add_subplot(1, 2, 1)
    rna_latent = combined_latent[combined_latent.obs["modality"] == "RNA"]
    plot_with_extreme_markers(
        ax1, rna_latent, "cell_types", f"{prefix}_RNA Latent UMAP by Cell Types", basis="X_umap"
    )

    # Protein UMAP
    ax2 = fig.add_subplot(1, 2, 2)
    prot_latent = combined_latent[combined_latent.obs["modality"] == "Protein"]
    plot_with_extreme_markers(
        ax2,
        prot_latent,
        "cell_types",
        f"{prefix}_Protein Latent UMAP by Cell Types",
        basis="X_umap",
    )

    # Add single global legend for marker types (extreme vs normal)
    if "is_extreme_archetype" in combined_latent.obs.columns:
        from matplotlib.lines import Line2D

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
        # Add single shared legend at the bottom of the figure
        fig.legend(
            handles=marker_legend_elements,
            fontsize=10,
            title="Markers",
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
        )

    plt.tight_layout(rect=[0, 0.12, 1, 1])  # Leave space for bottom legend

    if global_step is not None:
        modality_celltype_file = f"{prefix}_modality_celltypes_umap_step_{global_step:05d}.pdf"
    else:
        modality_celltype_file = f"{prefix}_modality_celltypes_umap_step_00000.pdf"
    safe_mlflow_log_figure(plt.gcf(), f"train/{modality_celltype_file}")
    logger.info(f"{prefix}Modality-specific Cell Type UMAPs visualized and saved")

    # Plot UMAP for each modality colored by matched_archetype_weight (if available)
    if "matched_archetype_weight" in combined_latent.obs.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # RNA UMAP with matched_archetype_weight
        rna_latent = combined_latent[combined_latent.obs["modality"] == "RNA"]
        coords_rna = rna_latent.obsm["X_umap"]
        weights_rna = rna_latent.obs["matched_archetype_weight"].to_numpy()
        scatter1 = ax1.scatter(
            coords_rna[:, 0], coords_rna[:, 1], c=weights_rna, cmap="viridis", s=20, alpha=0.8
        )
        plt.colorbar(scatter1, ax=ax1, label="Matched Archetype Weight")
        ax1.set_xlabel("UMAP1")
        ax1.set_ylabel("UMAP2")
        ax1.set_title(f"{prefix}_RNA Latent UMAP by Matched Archetype Weight")

        # Protein UMAP with matched_archetype_weight
        prot_latent = combined_latent[combined_latent.obs["modality"] == "Protein"]
        coords_prot = prot_latent.obsm["X_umap"]
        weights_prot = prot_latent.obs["matched_archetype_weight"].to_numpy()
        scatter2 = ax2.scatter(
            coords_prot[:, 0], coords_prot[:, 1], c=weights_prot, cmap="viridis", s=20, alpha=0.8
        )
        plt.colorbar(scatter2, ax=ax2, label="Matched Archetype Weight")
        ax2.set_xlabel("UMAP1")
        ax2.set_ylabel("UMAP2")
        ax2.set_title(f"{prefix}_Protein Latent UMAP by Matched Archetype Weight")

        plt.tight_layout()

        if global_step is not None:
            weight_file = f"{prefix}_modality_archetype_weight_umap_step_{global_step:05d}.pdf"
        else:
            weight_file = f"{prefix}_modality_archetype_weight_umap_step_00000.pdf"
        safe_mlflow_log_figure(plt.gcf(), f"train/{weight_file}")
        logger.info(
            f"{prefix}Modality-specific Matched Archetype Weight UMAPs visualized and saved"
        )
        plt.close()


# %%
def reconstruction_comparison_plot(
    rna_vae,
    protein_vae,
    rna_batch,
    protein_batch,
    rna_inference_outputs,
    protein_inference_outputs,
    global_step=None,
):
    """
    Plot comparison of original vs reconstructed data for both modalities.

    Samples actual counts from the learned distributions to compare with input counts.

    Creates PCA and UMAP visualizations comparing original data with VAE reconstructions,
    similar to plot_pca_umap_latent_space_during_train but showing original vs reconstructed
    instead of RNA vs Protein modalities.

    Args:
        rna_vae: RNA VAE model
        protein_vae: Protein VAE model
        rna_batch: RNA batch dictionary with 'X', 'labels', etc.
        protein_batch: Protein batch dictionary with 'X', 'labels', etc.
        rna_inference_outputs: RNA inference outputs from VAE forward pass
        protein_inference_outputs: Protein inference outputs from VAE forward pass
        global_step: Current training step for file naming
    """

    logger.info("Creating reconstruction comparison plots...")

    # Helper function to create scatter plot with extreme archetype markers (reuse from existing function)
    def plot_with_extreme_markers(ax, adata, color_col, title, basis="X_pca"):
        """Create scatter plot with different markers for extreme archetypes."""
        # Check if extreme archetype information exists
        if "is_extreme_archetype" in adata.obs.columns:
            extreme_mask = adata.obs["is_extreme_archetype"].astype(bool)
            non_extreme_mask = ~extreme_mask
            coords = adata.obsm[basis]

            # Get unique categories for color mapping
            categories = (
                adata.obs[color_col].cat.categories
                if pd.api.types.is_categorical_dtype(adata.obs[color_col])
                else adata.obs[color_col].unique()
            )

            # Use specific colors for data_type (original vs reconstructed)
            if color_col == "data_type":
                color_map = {"Original": "#2ca02c", "Reconstructed": "#d62728"}  # green and red
            else:
                colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
                color_map = dict(zip(categories, colors))

            # Plot non-extreme cells
            legend_handles = []
            legend_labels = []

            if non_extreme_mask.sum() > 0:
                for category in categories:
                    mask = (adata.obs[color_col] == category) & non_extreme_mask
                    if mask.sum() > 0:
                        scatter = ax.scatter(
                            coords[mask, 0],
                            coords[mask, 1],
                            c=[color_map[category]],
                            marker="o",
                            s=20,
                            alpha=0.6,
                            label=category,
                        )
                        legend_handles.append(scatter)
                        legend_labels.append(str(category))

            # Plot extreme archetype cells with 'x' markers
            if extreme_mask.sum() > 0:
                for category in categories:
                    mask = (adata.obs[color_col] == category) & extreme_mask
                    if mask.sum() > 0:
                        ax.scatter(
                            coords[mask, 0],
                            coords[mask, 1],
                            c=[color_map[category]],
                            marker="x",
                            s=60,
                            alpha=1.0,
                            linewidths=2,
                        )

            ax.legend(legend_handles, legend_labels, fontsize=8, markerscale=1.5)
        else:
            # No extreme archetype info - plot normally
            sc.pl.embedding(
                adata, basis=basis.replace("X_", ""), color=color_col, ax=ax, show=False
            )

        ax.set_title(title)
        ax.set_xlabel(f"{basis.split('_')[1].upper()}1" if "_" in basis else "Dim1")
        ax.set_ylabel(f"{basis.split('_')[1].upper()}2" if "_" in basis else "Dim2")

    # CRITICAL: Must normalize data and re-encode with constant library size
    # The inference_outputs were computed with per-cell library sizes during training,
    # but we need to use constant library size for both encoding and decoding to match
    # create_counterfactual_adata approach and avoid shifts
    if "normalize_total_value" not in protein_vae.adata.uns:
        raise ValueError(
            "protein_vae.adata.uns must contain 'normalize_total_value' "
            "indicating the constant library size used for normalization"
        )

    constant_lib_size = protein_vae.adata.uns["normalize_total_value"]
    constant_lib_size_log = np.log(constant_lib_size)
    logger.info(
        f"Using constant library size: {constant_lib_size} (log: {constant_lib_size_log:.4f})"
    )

    # Get original data and normalize to constant library size
    rna_original = rna_batch["X"].detach().cpu().numpy()
    protein_original = protein_batch["X"].detach().cpu().numpy()

    # Normalize using scanpy (same as create_counterfactual_adata) to ensure exact library size
    # This avoids rounding errors from manual normalization
    rna_original_adata_temp = anndata.AnnData(rna_original)
    sc.pp.normalize_total(rna_original_adata_temp, target_sum=constant_lib_size)
    rna_original_normalized = (
        rna_original_adata_temp.X.toarray()
        if issparse(rna_original_adata_temp.X)
        else rna_original_adata_temp.X
    ).astype(np.int32)

    protein_original_adata_temp = anndata.AnnData(protein_original)
    sc.pp.normalize_total(protein_original_adata_temp, target_sum=constant_lib_size)
    protein_original_normalized = (
        protein_original_adata_temp.X.toarray()
        if issparse(protein_original_adata_temp.X)
        else protein_original_adata_temp.X
    ).astype(np.int32)

    # Re-encode normalized data with constant library size (critical to avoid shift!)
    # Set models to train mode (eval mode can give wrong results due to dropout/batchnorm)
    rna_vae.module.eval()
    protein_vae.module.eval()

    with torch.no_grad():
        # Convert normalized data to tensors
        rna_X_normalized = torch.tensor(
            rna_original_normalized, dtype=torch.float32, device=rna_batch["X"].device
        )
        protein_X_normalized = torch.tensor(
            protein_original_normalized, dtype=torch.float32, device=protein_batch["X"].device
        )

        # Create constant library size tensors
        constant_lib_tensor_rna = torch.full(
            (rna_X_normalized.shape[0], 1),
            constant_lib_size_log,
            dtype=torch.float32,
            device=rna_X_normalized.device,
        )
        constant_lib_tensor_protein = torch.full(
            (protein_X_normalized.shape[0], 1),
            constant_lib_size_log,
            dtype=torch.float32,
            device=protein_X_normalized.device,
        )

        # Re-encode with constant library size using full forward pass (not direct inference call)
        rna_batch_normalized = {
            "X": rna_X_normalized,
            "batch": rna_batch["batch"],
            "library": constant_lib_tensor_rna,
            "labels": rna_batch.get(
                "labels", torch.arange(rna_X_normalized.shape[0], device=rna_X_normalized.device)
            ),
        }
        protein_batch_normalized = {
            "X": protein_X_normalized,
            "batch": protein_batch["batch"],
            "library": constant_lib_tensor_protein,
            "labels": protein_batch.get(
                "labels",
                torch.arange(protein_X_normalized.shape[0], device=protein_X_normalized.device),
            ),
        }

        # Get new latent representations with constant library size
        rna_inference_normalized = rna_vae.module.inference(
            rna_batch_normalized["X"],
            rna_batch_normalized["batch"],
            rna_batch_normalized["library"],
        )
        protein_inference_normalized = protein_vae.module.inference(
            protein_batch_normalized["X"],
            protein_batch_normalized["batch"],
            protein_batch_normalized["library"],
        )

        rna_latent_mean = rna_inference_normalized["qz"].mean
        protein_latent_mean = protein_inference_normalized["qz"].mean

        # Decode with constant library size
        rna_generative_outputs = rna_vae.module.generative(
            rna_latent_mean,
            constant_lib_tensor_rna,
            rna_batch["batch"],
        )
        protein_generative_outputs = protein_vae.module.generative(
            protein_latent_mean,
            constant_lib_tensor_protein,
            protein_batch["batch"],
        )

        # Get gene likelihoods to handle different distributions appropriately
        rna_likelihood = rna_vae.adata.uns.get("gene_likelihood", "zinb").lower()
        protein_likelihood = protein_vae.adata.uns.get("gene_likelihood", "zinb").lower()

        # Sample actual counts from the distributions
        # Process RNA: sample from distribution
        if "px" in rna_generative_outputs:
            rna_px_dist = rna_generative_outputs["px"]
            rna_reconstructed = rna_px_dist.sample().detach().cpu().numpy()
        elif "px_rate" in rna_generative_outputs:
            if rna_likelihood == "zinb":
                rna_dist = ZeroInflatedNegativeBinomial(
                    mu=rna_generative_outputs["px_rate"],
                    theta=rna_generative_outputs["px_r"],
                    zi_logits=rna_generative_outputs["px_dropout"],
                )
                rna_reconstructed = rna_dist.sample().detach().cpu().numpy()
            elif rna_likelihood == "nb":
                rna_dist = NegativeBinomial(
                    mu=rna_generative_outputs["px_rate"], theta=rna_generative_outputs["px_r"]
                )
                rna_reconstructed = rna_dist.sample().detach().cpu().numpy()
            elif rna_likelihood == "normal":
                rna_reconstructed = rna_generative_outputs["px_scale"].detach().cpu().numpy()
            else:
                raise ValueError(f"Unknown RNA gene_likelihood: {rna_likelihood}")
        else:
            raise KeyError("Expected 'px' distribution or 'px_rate' in RNA generative outputs")

        # Process Protein: sample from distribution
        if "px" in protein_generative_outputs:
            protein_px_dist = protein_generative_outputs["px"]
            protein_reconstructed = protein_px_dist.sample().detach().cpu().numpy()
        elif "px_rate" in protein_generative_outputs:
            if protein_likelihood == "zinb":
                protein_dist = ZeroInflatedNegativeBinomial(
                    mu=protein_generative_outputs["px_rate"],
                    theta=protein_generative_outputs["px_r"],
                    zi_logits=protein_generative_outputs["px_dropout"],
                )
                protein_reconstructed = protein_dist.sample().detach().cpu().numpy()
            elif protein_likelihood == "nb":
                protein_dist = NegativeBinomial(
                    mu=protein_generative_outputs["px_rate"],
                    theta=protein_generative_outputs["px_r"],
                )
                protein_reconstructed = protein_dist.sample().detach().cpu().numpy()
            elif protein_likelihood == "normal":
                protein_reconstructed = (
                    protein_generative_outputs["px_scale"].detach().cpu().numpy()
                )
            else:
                raise ValueError(f"Unknown protein gene_likelihood: {protein_likelihood}")
        else:
            raise KeyError("Expected 'px' distribution or 'px_rate' in protein generative outputs")

    # Get observation metadata
    rna_obs = rna_vae.adata[rna_batch["labels"]].obs.copy()
    protein_obs = protein_vae.adata[protein_batch["labels"]].obs.copy()

    # Create combined AnnData for RNA (normalized original + reconstructed)
    rna_original_adata = anndata.AnnData(rna_original_normalized, obs=rna_obs.copy())
    rna_original_adata.obs["data_type"] = "Original"

    rna_reconstructed_adata = anndata.AnnData(rna_reconstructed, obs=rna_obs.copy())
    rna_reconstructed_adata.obs["data_type"] = "Reconstructed"

    rna_combined = anndata.concat(
        [rna_original_adata, rna_reconstructed_adata],
        join="outer",
        label="data_source",
        keys=["Original", "Reconstructed"],
    )

    # Create combined AnnData for Protein (normalized original + reconstructed)
    protein_original_adata = anndata.AnnData(protein_original_normalized, obs=protein_obs.copy())
    protein_original_adata.obs["data_type"] = "Original"

    protein_reconstructed_adata = anndata.AnnData(protein_reconstructed, obs=protein_obs.copy())
    protein_reconstructed_adata.obs["data_type"] = "Reconstructed"

    protein_combined = anndata.concat(
        [protein_original_adata, protein_reconstructed_adata],
        join="outer",
        label="data_source",
        keys=["Original", "Reconstructed"],
    )

    # === RNA PCA PLOTS ===
    logger.info("Creating RNA reconstruction PCA plots...")
    # Log-transform for PCA (scanpy PCA expects log-normalized data)
    sc.pp.pca(rna_combined)

    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    plot_with_extreme_markers(ax1, rna_combined, "data_type", "RNA PCA: Original vs Reconstructed")

    ax2 = fig.add_subplot(1, 3, 2)
    plot_with_extreme_markers(ax2, rna_combined, "cell_types", "RNA PCA by Cell Type")

    ax3 = fig.add_subplot(1, 3, 3)
    plot_with_extreme_markers(ax3, rna_combined, "CN", "RNA PCA by CN")

    plt.tight_layout()
    if global_step is not None:
        pca_file = f"rna_reconstruction_pca_step_{global_step:05d}.pdf"
    else:
        pca_file = "rna_reconstruction_pca.pdf"

    # aaaa = protein_original_adata.X - protein_reconstructed_adata.X
    # plt.figure(figsize=(10, 10))
    # # sns.heatmap(np.log1p(aaaa), cmap="viridis")
    # # plt.plot(aaaa.mean(axis=1))
    # plt.plot(aaaa.mean(axis=1))
    # plt.legend()
    # safe_mlflow_log_figure(plt.gcf(), f"reconstruction_plots/heatmap.pdf")

    safe_mlflow_log_figure(plt.gcf(), f"reconstruction_plots/{pca_file}")
    plt.close()

    # === RNA UMAP PLOTS ===
    logger.info("Creating RNA reconstruction UMAP plots...")
    # Use X_pca for neighbors (default), NOT raw X - this prevents shift
    sc.pp.neighbors(rna_combined)
    sc.tl.umap(rna_combined)

    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    plot_with_extreme_markers(
        ax1, rna_combined, "data_type", "RNA UMAP: Original vs Reconstructed", basis="X_umap"
    )

    ax2 = fig.add_subplot(1, 3, 2)
    plot_with_extreme_markers(
        ax2, rna_combined, "cell_types", "RNA UMAP by Cell Type", basis="X_umap"
    )

    ax3 = fig.add_subplot(1, 3, 3)
    plot_with_extreme_markers(ax3, rna_combined, "CN", "RNA UMAP by CN", basis="X_umap")

    plt.tight_layout()
    if global_step is not None:
        umap_file = f"rna_reconstruction_umap_step_{global_step:05d}.pdf"
    else:
        umap_file = "rna_reconstruction_umap.pdf"
    safe_mlflow_log_figure(plt.gcf(), f"reconstruction_plots/{umap_file}")
    plt.close()

    # === PROTEIN PCA PLOTS ===
    logger.info("Creating Protein reconstruction PCA plots...")
    # Log-transform for PCA (scanpy PCA expects log-normalized data)
    protein_combined.X = np.log1p(protein_combined.X.astype(np.float32))

    # Handle NaN values before PCA
    if np.any(np.isnan(protein_combined.X)):
        logger.warning("Found NaN values in protein data, replacing with zeros")
        protein_combined.X = np.nan_to_num(protein_combined.X, nan=0.0, posinf=0.0, neginf=0.0)

    sc.pp.pca(protein_combined)

    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    plot_with_extreme_markers(
        ax1, protein_combined, "data_type", "Protein PCA: Original vs Reconstructed"
    )

    ax2 = fig.add_subplot(1, 3, 2)
    plot_with_extreme_markers(ax2, protein_combined, "cell_types", "Protein PCA by Cell Type")

    ax3 = fig.add_subplot(1, 3, 3)
    plot_with_extreme_markers(ax3, protein_combined, "CN", "Protein PCA by CN")

    plt.tight_layout()
    if global_step is not None:
        pca_file = f"protein_reconstruction_pca_step_{global_step:05d}.pdf"
    else:
        pca_file = "protein_reconstruction_pca.pdf"
    safe_mlflow_log_figure(plt.gcf(), f"reconstruction_plots/{pca_file}")
    plt.close()

    # === PROTEIN UMAP PLOTS ===
    logger.info("Creating Protein reconstruction UMAP plots...")
    # Use X_pca for neighbors (default), NOT raw X - this prevents shift
    sc.pp.neighbors(protein_combined)
    sc.tl.umap(protein_combined)

    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    plot_with_extreme_markers(
        ax1,
        protein_combined,
        "data_type",
        "Protein UMAP: Original vs Reconstructed",
        basis="X_umap",
    )

    ax2 = fig.add_subplot(1, 3, 2)
    plot_with_extreme_markers(
        ax2, protein_combined, "cell_types", "Protein UMAP by Cell Type", basis="X_umap"
    )

    ax3 = fig.add_subplot(1, 3, 3)
    plot_with_extreme_markers(ax3, protein_combined, "CN", "Protein UMAP by CN", basis="X_umap")

    plt.tight_layout()
    if global_step is not None:
        umap_file = f"protein_reconstruction_umap_step_{global_step:05d}.pdf"
    else:
        umap_file = "protein_reconstruction_umap.pdf"
    safe_mlflow_log_figure(plt.gcf(), f"reconstruction_plots/{umap_file}")
    plt.close()

    logger.info("Reconstruction comparison plots completed successfully!")


def plot_cosine_distance(rna_batch, protein_batch):
    umap_model = UMAP(n_components=2, random_state=42).fit(rna_batch["archetype_vec"], min_dist=5)
    # Transform both modalities using the same UMAP model
    rna_archetype_2pc = umap_model.transform(rna_batch["archetype_vec"])
    prot_archetype_2pc = umap_model.transform(protein_batch["archetype_vec"])

    rna_norm = rna_archetype_2pc / np.linalg.norm(rna_archetype_2pc, axis=1)[:, None]
    scale = 1.2
    prot_norm = scale * prot_archetype_2pc / np.linalg.norm(prot_archetype_2pc, axis=1)[:, None]
    plt.scatter(rna_norm[:, 0], rna_norm[:, 1], label="RNA", alpha=0.7)
    plt.scatter(prot_norm[:, 0], prot_norm[:, 1], label="Protein", alpha=0.7)

    for rna, prot in zip(rna_norm, prot_norm):
        plt.plot([rna[0], prot[0]], [rna[1], prot[1]], "k--", alpha=0.6, lw=0.5)

    # Add unit circle for reference
    theta = np.linspace(0, 2 * np.pi, 100)

    plt.plot(np.cos(theta), np.sin(theta), "grey", linestyle="--", alpha=0.3)
    plt.axis("equal")
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(scale * np.cos(theta), scale * np.sin(theta), "grey", linestyle="--", alpha=0.3)
    plt.axis("equal")

    plt.title("Normalized Vector Alignment\n(Euclidean Distance ∝ Cosine Distance)")
    plt.xlabel("PC1 (Normalized)")
    plt.ylabel("PC2 (Normalized)")
    plt.legend()
    plt.tight_layout()

    plt.show()
    plt.close()
    plt.close()


def plot_pre_training_visualizations(adata_rna_subset, adata_prot_subset, plot_flag=True):
    """Plot PCA and UMAP visualizations before training."""
    if not plot_flag:
        return

    sc.pl.pca(adata_rna_subset, color=["CN", "cell_types", "batch"])
    sc.pl.pca(adata_prot_subset, color=["CN", "cell_types", "batch"])
    sc.pl.umap(adata_rna_subset, color=["CN", "cell_types", "batch"])
    sc.pl.umap(adata_prot_subset, color=["CN", "cell_types", "batch"])


def plot_batch_specific_umaps(adata_rna_subset, plot_flag=True):
    """Plot batch-specific UMAP visualizations."""
    if not plot_flag:
        return

    from arcadia.data_utils import log1p_rna

    for batch in adata_rna_subset.obs["batch"].unique():
        adata = adata_rna_subset[adata_rna_subset.obs["batch"] == batch].copy()
        adata_2 = adata_rna_subset[adata_rna_subset.obs["batch"] == batch].copy()

        if not adata.uns["pipeline_metadata"].get("log1p", False):
            log1p_rna(adata)

        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.pp.pca(adata_2)
        sc.pp.neighbors(adata_2)
        sc.tl.umap(adata_2)

        plot_umap_with_extremes(adata, color_key=["CN", "cell_types"], title=f"{batch} \n log umap")
        plot_umap_with_extremes(
            adata, color_key=["CN", "cell_types"], basis="pca", title=f"{batch} \n log pca"
        )
        plot_umap_with_extremes(
            adata_2, color_key=["CN", "cell_types"], basis="pca", title=f"{batch} \n pca"
        )
        plot_umap_with_extremes(adata_2, color_key=["CN", "cell_types"], title=f"{batch} \n umap")


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


def plot_latent_distances(latent_distances, threshold):
    """Plot latent distances and threshold"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot latent distances heatmap
    sns.heatmap(latent_distances.detach().cpu().numpy(), ax=axes[0])
    axes[0].set_title("Latent Distances")
    axes[0].legend()

    # Plot latent distances
    axes[1].hist(latent_distances.detach().cpu().numpy().flatten(), bins=50)
    axes[1].axvline(x=threshold, color="r", linestyle="--", label=f"Threshold: {threshold}")
    axes[1].set_title("Latent Distances")
    axes[1].legend()

    plt.tight_layout()

    plt.show()
    plt.close()
    plt.close()


def plot_combined_latent_space(combined_latent, use_subsample=True):
    """Plot combined latent space visualizations"""
    # Subsample if requested
    if use_subsample:
        subsample_n_obs = min(1000, combined_latent.shape[0])
        subsample_idx = np.random.choice(combined_latent.shape[0], subsample_n_obs, replace=False)
        combined_latent_plot = combined_latent[subsample_idx].copy()
    else:
        combined_latent_plot = combined_latent.copy()

    # Plot UMAP
    sc.tl.umap(combined_latent_plot)
    sc.pl.umap(
        combined_latent_plot,
        color=["CN", "modality", "cell_types"],
        title=[
            "UMAP Combined Latent space CN",
            "UMAP Combined Latent space modality",
            "UMAP Combined Latent space cell types",
        ],
        alpha=0.5,
    )
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "combined_latent_space_umap.pdf")

    # Plot PCA
    if "X_pca" not in combined_latent_plot.obsm:
        sc.pp.pca(combined_latent_plot)
    sc.pl.pca(
        combined_latent_plot,
        color=["CN", "modality"],
        title=["PCA Combined Latent space CN", "PCA Combined Latent space modality"],
        alpha=0.5,
    )
    # remove the pca from the obsm to allow for neighbor to run on the latent space directly
    combined_latent_plot.obsm.pop("X_pca", None)
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "combined_latent_space_pca.pdf")


def plot_cell_type_distributions(combined_latent, top_n=3, use_subsample=True):
    """Plot UMAP for top N most common cell types"""
    top_cell_types = combined_latent.obs["cell_types"].value_counts().index[:top_n]

    for cell_type in top_cell_types:
        cell_type_data = combined_latent[combined_latent.obs["cell_types"] == cell_type]

        # Subsample if requested
        if use_subsample and cell_type_data.shape[0] > 700:
            n_subsample = min(700, cell_type_data.shape[0])
            subsample_idx = np.random.choice(cell_type_data.shape[0], n_subsample, replace=False)
            cell_type_data_plot = cell_type_data[subsample_idx].copy()
        else:
            cell_type_data_plot = cell_type_data.copy()

        sc.pl.umap(
            cell_type_data_plot,
            color=["CN", "modality", "cell_types"],
            title=[
                f"Combined latent space UMAP {cell_type}, CN",
                f"Combined latent space UMAP {cell_type}, modality",
                f"Combined latent space UMAP {cell_type}, cell types",
            ],
            alpha=0.5,
        )
        plt.tight_layout()
        safe_mlflow_log_figure(
            plt.gcf(), f"cell_type_distribution/cell_type_distribution_{cell_type}.pdf"
        )


def plot_rna_protein_latent_cn_cell_type_umap(adata_rna, adata_prot, use_subsample=True):
    """Plot RNA and protein embeddings"""
    # Create copies to avoid modifying the original data
    if use_subsample:
        # Subsample RNA data
        n_subsample_rna = min(700, adata_rna.shape[0])
        subsample_idx_rna = np.random.choice(adata_rna.shape[0], n_subsample_rna, replace=False)
        adata_rna_plot = adata_rna[subsample_idx_rna].copy()

        # Subsample protein data
        n_subsample_prot = min(700, adata_prot.shape[0])
        subsample_idx_prot = np.random.choice(adata_prot.shape[0], n_subsample_prot, replace=False)
        adata_prot_plot = adata_prot[subsample_idx_prot].copy()
    else:
        adata_rna_plot = adata_rna.copy()
        adata_prot_plot = adata_prot.copy()

    sc.pl.embedding(
        adata_rna_plot,
        color=["CN", "cell_types"],
        basis="X_scVI",
        title=["RNA_latent_CN", "RNA_Latent_CellTypes"],
    )
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "rna_latent_embeddings.pdf")

    sc.pl.embedding(
        adata_prot_plot,
        color=["CN", "cell_types"],
        basis="X_scVI",
        title=["Protein_latent_CN", "Protein_Laten_CellTypes"],
    )
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "protein_latent_embeddings.pdf")


def plot_latent_single(means, adata, index, color_label="CN", title="", subset_size=1000):
    adata_subset = sc.pp.subsample(adata, n_obs=min(subset_size, adata.n_obs), copy=True)
    plt.figure()
    pca = PCA(n_components=3)
    means_cpu = means.detach().cpu().numpy()
    index_cpu = index.detach().cpu().numpy().flatten()
    pca.fit(means_cpu)
    rna_pca = pca.transform(means_cpu)
    plt.subplot(1, 1, 1)
    plt.scatter(
        rna_pca[:, 0],
        rna_pca[:, 1],
        c=pd.Categorical(adata_subset[index_cpu].obs[color_label].values).codes,
        cmap="jet",
    )
    plt.title(title)
    plt.show()
    plt.close()


# %%


def plot_umap_with_extremes(
    adata,
    color_key=None,
    extreme_key="is_extreme_archetype",
    basis="umap",
    base_size=20,
    extreme_size=40,
    extreme_color=None,
    alpha=0.8,
    title=None,
    subset_size=1000,
):
    """Plot UMAP colored by an obs key and overlay extremes (marker 'x').

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with `obsm['X_umap']`/`obsm['X_pca']` or compute as needed.
    color_key : str | None
        `adata.obs` key to color points by (categorical or numeric). If None, uses light gray.
    extreme_key : str
        `adata.obs` boolean key indicating which points to overlay with an 'x' marker.
    basis : {"umap", "pca"}
        Coordinate basis to use. Defaults to "umap".
    base_size : float
        Marker size for non-extreme points.
    extreme_size : float
        Marker size for extreme points.
    extreme_color : str | None
        Color for extreme points. If None, reuse the base color of each point.
    alpha : float
        Alpha for base points.
    title : str | None
        Optional plot title.
    """

    basis = str(basis).lower()
    if basis not in {"umap", "pca"}:
        raise ValueError("basis must be 'umap' or 'pca'")

    if basis == "umap":
        if "X_umap" not in adata.obsm:
            if "neighbors" not in adata.uns:
                sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        coords = adata.obsm["X_umap"]
    else:
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata)
        coords = adata.obsm["X_pca"][:, :2]

    if extreme_key not in adata.obs:
        raise KeyError(f"Expected '{extreme_key}' in adata.obs")

    extreme_mask = adata.obs[extreme_key].astype(bool).to_numpy()

    # Accept a single key or a list of keys
    if color_key is None:
        keys = [None]
    elif isinstance(color_key, (list, tuple)):
        keys = list(color_key)
    else:
        keys = [color_key]

    n_panels = len(keys)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7), squeeze=False)
    axes = axes.ravel()

    for idx, key in enumerate(keys):
        ax = axes[idx]

        # Build colors for this panel
        if key is None:
            base_colors = np.full((adata.n_obs,), "lightgray", dtype=object)
            colorbar_info = None
        else:
            values = adata.obs[key]
            # Check if numeric (handle categorical dtype properly)
            if (
                hasattr(values, "dtype")
                and hasattr(values.dtype, "name")
                and values.dtype.name == "category"
            ):
                is_numeric = False
            else:
                try:
                    is_numeric = np.issubdtype(values.dtype, np.number)
                except TypeError:
                    is_numeric = False

            if is_numeric:
                norm = Normalize(vmin=np.nanmin(values.values), vmax=np.nanmax(values.values))
                cmap = plt.get_cmap("viridis")
                base_colors = cmap(norm(values.values))
                colorbar_info = (norm, cmap, f"{key}")
            else:
                cats = values.astype("category")
                palette = sc.pl.palettes.vega_20_scanpy
                if len(cats.cat.categories) > len(palette):
                    palette = sc.pl.palettes.default_102
                color_map = {cat: palette[i] for i, cat in enumerate(cats.cat.categories)}
                base_colors = cats.map(color_map).to_numpy()
                colorbar_info = None

        # Base scatter
        ax.scatter(
            coords[~extreme_mask, 0],
            coords[~extreme_mask, 1],
            s=base_size,
            c=base_colors[~extreme_mask],
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

        # Extreme overlay
        overlay_colors = base_colors[extreme_mask] if extreme_color is None else extreme_color
        ax.scatter(
            coords[extreme_mask, 0],
            coords[extreme_mask, 1],
            s=extreme_size,
            c=overlay_colors,
            marker="x",
            linewidths=1.0,
            rasterized=True,
        )

        if basis == "umap":
            ax.set(xticks=[], yticks=[], xlabel="UMAP1", ylabel="UMAP2")
        else:
            ax.set(xticks=[], yticks=[], xlabel="PC1", ylabel="PC2")

        # Title per panel (without the main title)
        if basis == "umap":
            panel_title = "UMAP" if key is None else f"{key}"
        else:
            panel_title = "PCA" if key is None else f"{key}"
        ax.set_title(panel_title)

        # Colorbar only for continuous keys
        if colorbar_info is not None:
            norm, cmap, label = colorbar_info
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
            cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(label)

    # Add suptitle if provided
    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.98)

    plt.tight_layout()
    plt.show()


def plot_encoded_latent_space_comparison(
    adata_rna_subset,
    adata_prot_subset,
    rna_latent,
    prot_latent,
    index_rna=None,
    index_prot=None,
    global_step=None,
    use_subsample=True,
    n_subsample=2000,
    suffix="",
    modality_name="Protein",
    plot_flag=True,
):
    """
    Plot encoded latent space of both RNA and protein modalities with PCA,
    colored by modality, cell types, and CN like in the reference image.
    """
    if not plot_flag:
        return
    if index_rna is None:
        index_rna = range(len(rna_latent))
    if index_prot is None:
        index_prot = range(len(prot_latent))

    # Subsample if requested - use separate sampling for RNA and protein
    if use_subsample:
        # Sample RNA data
        n_subsample_rna = min(n_subsample, len(index_rna))
        rna_subsample_idx = np.random.choice(len(index_rna), n_subsample_rna, replace=False)
        index_rna = np.array(index_rna)[rna_subsample_idx]
        rna_latent_sub = rna_latent[rna_subsample_idx]

        # Sample protein data (separately)
        n_subsample_prot = min(n_subsample, len(index_prot))
        prot_subsample_idx = np.random.choice(len(index_prot), n_subsample_prot, replace=False)
        index_prot = np.array(index_prot)[prot_subsample_idx]
        prot_latent_sub = prot_latent[prot_subsample_idx]
    else:
        rna_latent_sub = rna_latent
        prot_latent_sub = prot_latent

    # Ensure all indices are valid
    if len(index_rna) == 0 or len(index_prot) == 0:
        logger.warning("Warning: No valid indices for plotting. Skipping plot.")
        return

    num_rna = len(index_rna)
    len(index_prot)

    # Create combined latent space for PCA fitting
    combined_latent = np.vstack([rna_latent_sub, prot_latent_sub])

    # Fit PCA on combined data
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined_latent)

    # Split back into RNA and protein
    rna_pca = combined_pca[:num_rna]
    prot_pca = combined_pca[num_rna:]

    # Create the three-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Real vs Counterfactual (Modality comparison)
    ax1 = axes[0]

    # Plot RNA points (blue)
    ax1.scatter(
        rna_pca[:, 0], rna_pca[:, 1], c="#1f77b4", s=20, alpha=0.6, label=f"Real {modality_name}"
    )

    # Plot Protein points (orange)
    ax1.scatter(
        prot_pca[:, 0],
        prot_pca[:, 1],
        c="#ff7f0e",
        s=20,
        alpha=0.6,
        label=f"Counterfactual {modality_name}",
    )

    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title(
        f"Real {modality_name} vs Counterfactual {modality_name}\nSim: 0.1935, iLISI: 1.0"
    )
    ax1.legend()

    # Panel 2: Cell Neighborhoods (CN)
    ax2 = axes[1]

    # Get CN information
    if "CN" in adata_rna_subset.obs.columns:
        rna_cn = adata_rna_subset[index_rna].obs["CN"]
        prot_cn = adata_prot_subset[index_prot].obs["CN"]

        # Create combined CN labels for consistent coloring
        all_cn = list(rna_cn) + list(prot_cn)
        unique_cn = sorted(set(all_cn))

        # Plot RNA points
        for cn in unique_cn:
            rna_mask = rna_cn == cn
            if rna_mask.sum() > 0:
                ax2.scatter(
                    rna_pca[rna_mask, 0], rna_pca[rna_mask, 1], s=20, alpha=0.6, label=f"{cn}"
                )

        # Plot protein points with same color scheme
        for cn in unique_cn:
            prot_mask = prot_cn == cn
            if prot_mask.sum() > 0:
                ax2.scatter(prot_pca[prot_mask, 0], prot_pca[prot_mask, 1], s=20, alpha=0.6)
    else:
        # Fallback if CN not available
        ax2.scatter(rna_pca[:, 0], rna_pca[:, 1], s=20, alpha=0.6, label="RNA")
        ax2.scatter(prot_pca[:, 0], prot_pca[:, 1], s=20, alpha=0.6, label="Protein")

    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("Cell Neighborhoods")
    ax2.legend()

    # Panel 3: Cell Types
    ax3 = axes[2]

    # Determine cell type column name
    if "cell_type" in adata_rna_subset.obs:
        cell_type_col = "cell_type"
    elif "cell_types" in adata_rna_subset.obs:
        cell_type_col = "cell_types"
    else:
        cell_type_col = "major_cell_types"

    if cell_type_col in adata_rna_subset.obs.columns:
        rna_cell_types = adata_rna_subset[index_rna].obs[cell_type_col]
        prot_cell_types = adata_prot_subset[index_prot].obs[cell_type_col]

        # Create combined cell type labels for consistent coloring
        all_cell_types = list(rna_cell_types) + list(prot_cell_types)
        unique_cell_types = sorted(set(all_cell_types))

        # Use different colors for cell types
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cell_types)))
        color_map = dict(zip(unique_cell_types, colors))

        # Plot RNA points
        for i, cell_type in enumerate(unique_cell_types):
            rna_mask = rna_cell_types == cell_type
            if rna_mask.sum() > 0:
                ax3.scatter(
                    rna_pca[rna_mask, 0],
                    rna_pca[rna_mask, 1],
                    c=[color_map[cell_type]],
                    s=20,
                    alpha=0.6,
                    label=cell_type,
                )

        # Plot protein points with same color scheme
        for cell_type in unique_cell_types:
            prot_mask = prot_cell_types == cell_type
            if prot_mask.sum() > 0:
                ax3.scatter(
                    prot_pca[prot_mask, 0],
                    prot_pca[prot_mask, 1],
                    c=[color_map[cell_type]],
                    s=20,
                    alpha=0.6,
                )
    else:
        # Fallback if cell types not available
        ax3.scatter(rna_pca[:, 0], rna_pca[:, 1], s=20, alpha=0.6, label="RNA")
        ax3.scatter(prot_pca[:, 0], prot_pca[:, 1], s=20, alpha=0.6, label="Protein")

    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_title("Cell Types")
    ax3.legend()

    plt.tight_layout()

    # Save the figure
    modality_suffix = modality_name.lower()
    if global_step is not None:
        filename = (
            f"encoded_latent_space_comparison_{modality_suffix}_{suffix}_step_{global_step:05d}.pdf"
        )
    else:
        filename = f"encoded_latent_space_comparison_{modality_suffix}_{suffix}.pdf"

    safe_mlflow_log_figure(plt.gcf(), filename)
    plt.show()
    plt.close()


def test_plot_latent_pca_both_modalities_by_celltype():
    """Test function for plot_latent_pca_both_modalities_by_celltype with synthetic data."""
    import numpy as np
    from anndata import AnnData

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic data
    n_cells = 2000
    n_cell_types = 5
    latent_dim = 20

    # Create latent representations for RNA and protein
    rna_latent = np.random.normal(0, 1, size=(n_cells, latent_dim))
    prot_latent = np.random.normal(0, 1, size=(n_cells, latent_dim))

    # Create cell type labels
    cell_type_names = [f"CellType_{i}" for i in range(n_cell_types)]
    cell_types = np.random.choice(cell_type_names, size=n_cells)

    # Create CN (neighborhood) labels
    cn_names = [f"CN_{i}" for i in range(3)]
    cn_labels = np.random.choice(cn_names, size=n_cells)

    # Create AnnData objects
    adata_rna = AnnData(X=np.random.lognormal(0, 1, size=(n_cells, 100)))
    adata_prot = AnnData(X=np.random.lognormal(0, 1, size=(n_cells, 50)))

    # Add cell type and CN information to obs
    adata_rna.obs["cell_types"] = cell_types
    adata_rna.obs["CN"] = cn_labels
    adata_prot.obs["cell_types"] = cell_types
    adata_prot.obs["CN"] = cn_labels

    # Plot with default settings
    logger.info("Testing plot_latent_pca_both_modalities_by_celltype with default settings...")
    plot_latent_pca_both_modalities_by_celltype(
        adata_rna_subset=adata_rna,
        adata_prot_subset=adata_prot,
        rna_latent=rna_latent,
        prot_latent=prot_latent,
        use_subsample=True,
    )

    # Plot without subsampling
    logger.info("Testing plot_latent_pca_both_modalities_by_celltype without subsampling...")
    plot_latent_pca_both_modalities_by_celltype(
        adata_rna_subset=adata_rna,
        adata_prot_subset=adata_prot,
        rna_latent=rna_latent,
        prot_latent=prot_latent,
        use_subsample=False,
    )

    # Plot with specified indices
    logger.info("Testing plot_latent_pca_both_modalities_by_celltype with specific indices...")
    index_rna = np.random.choice(n_cells, size=500, replace=False)
    index_prot = np.random.choice(n_cells, size=500, replace=False)
    plot_latent_pca_both_modalities_by_celltype(
        adata_rna_subset=adata_rna,
        adata_prot_subset=adata_prot,
        rna_latent=rna_latent,
        prot_latent=prot_latent,
        index_rna=index_rna,
        index_prot=index_prot,
        use_subsample=True,
    )

    logger.info("All tests completed!")


def plot_training_metrics_history(metrics_history):
    """Plot training metrics history_ over epochs.

    Args:
        metrics_history (list): List of dictionaries containing metrics for each epoch
        save_path (str): Path to save the plot

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    try:
        # Convert metrics history_ to DataFrame
        metrics_df = pd.DataFrame(metrics_history)
        metrics_df["epoch"] = range(len(metrics_df))

        # Create subplots based on number of metrics
        n_metrics = len(metrics_df.columns) - 1  # Subtract 1 for epoch column
        n_rows = (n_metrics + 2) // 3  # 3 plots per row, round up

        fig = plt.figure(figsize=(15, 5 * n_rows))
        for i, metric in enumerate(metrics_df.columns.drop("epoch")):
            plt.subplot(n_rows, 3, i + 1)
            sns.lineplot(data=metrics_df, x="epoch", y=metric)
            plt.title(metric)

        plt.tight_layout()
        safe_mlflow_log_figure(fig, "training_metrics.pdf")
        logger.info("Training metrics plotted and saved")
    except Exception as e:
        logger.error(f"Error plotting training metrics: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
    return fig


def visualize_gmm_threshold(X, dynamic_threshold, gmm, noise_idx):
    nonzero_vals = X[X > 0].reshape(-1, 1)
    x = np.linspace(nonzero_vals.min(), nonzero_vals.max(), 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x)
    pdf = np.exp(logprob)
    responsibilities = gmm.predict_proba(x)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    plt.figure(figsize=(10, 6))
    plt.hist(
        nonzero_vals,
        bins=50,
        density=True,
        alpha=0.6,
        color="gray",
        label="Nonzero values histogram",
    )
    plt.plot(x, pdf, "-k", label="GMM total density")
    plt.plot(x, pdf_individual[:, noise_idx], "--r", label="Noise component")
    plt.plot(x, pdf_individual[:, 1 - noise_idx], "--b", label="Signal component")
    plt.axvline(
        dynamic_threshold,
        color="green",
        linestyle="--",
        label=f"Threshold = {dynamic_threshold:.3f}",
    )
    plt.title("Gaussian Mixture Model Fit and Dynamic Threshold")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    safe_mlflow_log_figure(plt.gcf(), "gmm_threshold.pdf")
    plt.close()


def visualize_integer_conversion(X, X_int):
    plt.figure(figsize=(10, 6))
    plt.hist(X.flatten(), bins=50, alpha=0.5, label="Original values")
    plt.hist(X_int.flatten(), bins=50, alpha=0.5, label="Integer-converted")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Original vs. Integer-Converted Data")
    plt.legend()
    plt.show()
    safe_mlflow_log_figure(plt.gcf(), "integer_conversion.pdf")
    plt.close()


def visualize_integer_conversion_subplots(X, X_int, bins=50):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Handle sparse matrices - convert to dense for plotting
    X_flat = X.toarray().flatten() if hasattr(X, "toarray") else X.flatten()
    X_int_flat = X_int.toarray().flatten() if hasattr(X_int, "toarray") else X_int.flatten()

    # Original values
    axes[0].hist(X_flat, bins=bins, color="skyblue", alpha=0.7)
    axes[0].set_title("Original Values")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")

    # Integer-converted values
    axes[1].hist(X_int_flat, bins=bins, color="orange", alpha=0.7)
    axes[1].set_title("Integer-Converted Values")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()
    plt.close()
    plt.close()


def plot_rna_protein_matching_means_and_scale(
    rna_latent_mean,
    protein_latent_mean,
    rna_latent_std,
    protein_latent_std,
    archetype_dis_mat,
    use_subsample=True,
    global_step=None,
):
    """
    Plot the means and scales as halo  and lines between the best matches
    of the RNA and protein
    Args:
        rna_inference_outputs: the output of the RNA inference
        protein_inference_outputs: the output of the protein inference
        archetype_dis_mat: the archetype distance matrix
        use_subsample: whether to use subsampling
        global_step: the current training step, if None then not during training
    """
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax_scatter = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    if use_subsample:
        rna_subsample_idx = np.random.choice(
            rna_latent_mean.shape[0], min(700, rna_latent_mean.shape[0]), replace=False
        )
        protein_subsample_idx = np.random.choice(
            protein_latent_mean.shape[0],
            min(700, protein_latent_mean.shape[0]),
            replace=False,
        )
    else:
        rna_subsample_idx = np.arange(rna_latent_mean.shape[0])
        protein_subsample_idx = np.arange(protein_latent_mean.shape[0])
    prot_new_order = archetype_dis_mat.argmin(axis=0).detach().cpu().numpy()

    rna_means = rna_latent_mean[rna_subsample_idx]
    rna_latent_std[rna_subsample_idx]
    protein_means = protein_latent_mean[prot_new_order][protein_subsample_idx]
    protein_latent_std[prot_new_order][protein_subsample_idx]
    # match the order of the means to the archetype_dis
    # Combine means for PCA
    combined_means = np.concatenate([rna_means, protein_means], axis=0)

    # Fit PCA on means
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_means)

    # Plot RNA points
    ax_scatter.scatter(
        pca_result[: rna_means.shape[0], 0],
        pca_result[: rna_means.shape[0], 1],
        c="blue",
        label="RNA",
        alpha=0.6,
    )
    # Plot Protein points
    ax_scatter.scatter(
        pca_result[rna_means.shape[0] :, 0],
        pca_result[rna_means.shape[0] :, 1],
        c="orange",
        label="Protein",
        alpha=0.6,
    )

    # Add connecting lines
    for i in range(rna_means.shape[0]):
        color = "red" if (i % 2 == 0) else "green"
        ax_scatter.plot(
            [pca_result[i, 0], pca_result[rna_means.shape[0] + i, 0]],
            [pca_result[i, 1], pca_result[rna_means.shape[0] + i, 1]],
            "-",
            alpha=0.2,
            color=color,
        )

    ax_scatter.set_xlabel("PCA Component 1")
    ax_scatter.set_ylabel("PCA Component 2")
    ax_scatter.set_title("PCA of RNA and Protein Latent Means")
    ax_scatter.legend()
    ax_scatter.set_aspect("equal")

    # Plot histogram of latent distances
    latent_distances = np.linalg.norm(rna_means - protein_means, axis=1)
    sns.histplot(latent_distances, bins=30, ax=ax_hist, kde=True)
    ax_hist.set_xlabel("Latent Distance")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("Distribution of Latent Distances")

    # Plot lines for mean latent distances
    plt.plot(
        [rna_latent_mean.mean(), protein_latent_mean.mean()],
        [0, 0],
        "-",
        color="k",
        lw=2,
    )
    plt.plot(
        [rna_latent_mean.mean(), rna_latent_mean.mean()],
        [-1, 1],
        "-",
        color="b",
        alpha=0.6,
        label="RNA",
    )
    plt.plot(
        [protein_latent_mean.mean(), protein_latent_mean.mean()],
        [-1, 1],
        "-",
        color="r",
        alpha=0.6,
        label="Protein",
    )
    plt.xlabel("Latent Distance")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    padded_step = f"{global_step:05d}"
    safe_mlflow_log_figure(
        fig, f"train/rna_protein_matching_means_and_scale_step_{padded_step}_.pdf"
    )
    plt.show()
    plt.close()
    plt.close(fig)


def plot_counterfactual_comparison_with_suffix(
    adata_rna,
    adata_prot,
    counterfactual_adata_rna,
    counterfactual_protein_adata,
    epoch="",
    suffix="",
    use_subsample=True,
    rna_similarity_score=None,
    protein_similarity_score=None,
    rna_ilisi_score=None,
    protein_ilisi_score=None,
    plot_flag=True,
):
    """
    Plot a comparison of same-modal reconstruction (ground truth) vs counterfactual data.

    Args:
        adata_rna: AnnData object for same-modal RNA reconstruction (RNA→RNA encoder→RNA decoder)
        adata_prot: AnnData object for same-modal protein reconstruction (Protein→Protein encoder→Protein decoder)
        counterfactual_adata_rna: AnnData object for counterfactual RNA (Protein→Protein encoder→RNA decoder)
        counterfactual_protein_adata: AnnData object for counterfactual protein (RNA→RNA encoder→Protein decoder)
        epoch: Epoch number for filename
        suffix: Suffix to add to filename (e.g., "train", "val")
        use_subsample: Whether to subsample for plotting
        rna_similarity_score: RNA similarity metric for title
        protein_similarity_score: Protein similarity metric for title
        rna_ilisi_score: RNA iLISI score for title
        protein_ilisi_score: Protein iLISI score for title
    """
    if not plot_flag:
        return
    # Create AnnData objects for real RNA and counterfactual RNA (from protein)
    # Add a column to identify the modality
    # Subsample if requested - use separate sampling for RNA and protein
    if use_subsample:
        # Sample RNA data
        n_subsample_rna = min(2000, len(adata_rna.obs_names))
        rna_subsample_idx = np.random.choice(
            len(adata_rna.obs_names), n_subsample_rna, replace=False
        )
        adata_rna = adata_rna[rna_subsample_idx].copy()

        # Sample protein data (separately)
        n_subsample_prot = min(2000, len(adata_prot.obs_names))
        prot_subsample_idx = np.random.choice(
            len(adata_prot.obs_names), n_subsample_prot, replace=False
        )
        adata_prot = adata_prot[prot_subsample_idx].copy()

        # Sample counterfactual RNA data (separately)
        n_subsample_rna = min(2000, len(counterfactual_adata_rna.obs_names))
        rna_subsample_idx = np.random.choice(
            len(counterfactual_adata_rna.obs_names), n_subsample_rna, replace=False
        )
        counterfactual_adata_rna = counterfactual_adata_rna[rna_subsample_idx].copy()

        # Sample counterfactual protein data (separately)
        n_subsample_prot = min(2000, len(counterfactual_protein_adata.obs_names))
        prot_subsample_idx = np.random.choice(
            len(counterfactual_protein_adata.obs_names), n_subsample_prot, replace=False
        )
        counterfactual_protein_adata = counterfactual_protein_adata[prot_subsample_idx].copy()

    # Combine both datasets using concatenate with proper batch keys
    adata_combined = adata_rna.concatenate(
        counterfactual_adata_rna,
        batch_categories=["Real RNA", "Counterfactual RNA"],
        batch_key="data_type",
    )

    # Process the combined data
    sc.pp.neighbors(adata_combined, use_rep="X")
    sc.tl.umap(adata_combined)

    # Create the 3 subplots for UMAP
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)

    # Plot 1: Color by modality (same-modal reconstruction vs counterfactual)
    sc.pl.umap(
        adata_combined,
        color="data_type",
        title="Same-modal RNA Recon vs Counterfactual RNA"
        + (
            f"\nSim: {rna_similarity_score}, iLISI: {rna_ilisi_score}"
            if rna_similarity_score is not None and rna_ilisi_score is not None
            else ""
        ),
        palette={"Real RNA": "#1f77b4", "Counterfactual RNA": "#ff7f0e"},
        ax=axes[0],
        show=False,
    )

    # Plot 2: Color by cell neighborhood (CN)
    sc.pl.umap(adata_combined, color="CN", title="Cell Neighborhoods", ax=axes[1], show=False)

    # Plot 3: Color by cell types
    sc.pl.umap(adata_combined, color="cell_types", title="Cell Types", ax=axes[2], show=False)

    # Also save as MLflow artifact if MLflow is active
    suffix_str = f"_{suffix}" if suffix else ""
    safe_mlflow_log_figure(
        plt.gcf(),
        f"counterfactual/rna_counterfactual_comparison_umap{suffix_str}_epoch_{epoch:04d}.pdf",
    )

    # Create the 3 subplots for PCA
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)

    # Compute PCA for the combined data
    sc.pp.pca(adata_combined)

    # Plot 1: Color by modality (same-modal reconstruction vs counterfactual)
    sc.pl.pca(
        adata_combined,
        color="data_type",
        title="Same-modal RNA Recon vs Counterfactual RNA"
        + (
            f"\nSim: {rna_similarity_score}, iLISI: {rna_ilisi_score}"
            if rna_similarity_score is not None and rna_ilisi_score is not None
            else ""
        ),
        palette={"Real RNA": "#1f77b4", "Counterfactual RNA": "#ff7f0e"},
        ax=axes[0],
        show=False,
    )

    # Plot 2: Color by cell neighborhood (CN)
    sc.pl.pca(adata_combined, color="CN", title="Cell Neighborhoods", ax=axes[1], show=False)

    # Plot 3: Color by cell types
    sc.pl.pca(adata_combined, color="cell_types", title="Cell Types", ax=axes[2], show=False)

    # Also save as MLflow artifact if MLflow is active
    safe_mlflow_log_figure(
        plt.gcf(),
        f"counterfactual/rna_counterfactual_comparison_pca{suffix_str}_epoch_{epoch:04d}.pdf",
    )

    # do the same for the protein
    # Combine both protein datasets using concatenate with proper batch keys
    adata_combined_protein = adata_prot.concatenate(
        counterfactual_protein_adata,
        batch_categories=["Real Protein", "Counterfactual Protein"],
        batch_key="data_type",
    )

    # Process the combined protein data
    sc.pp.neighbors(adata_combined_protein, use_rep="X")
    sc.tl.umap(adata_combined_protein)

    # Create the 3 subplots for UMAP
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)

    # Plot 1: Color by modality (same-modal reconstruction vs counterfactual)
    sc.pl.umap(
        adata_combined_protein,
        color="data_type",
        title="Same-modal Protein Recon vs Counterfactual Protein"
        + (
            f"\nSim: {protein_similarity_score}, iLISI: {protein_ilisi_score}"
            if protein_similarity_score is not None and protein_ilisi_score is not None
            else ""
        ),
        palette={"Real Protein": "#1f77b4", "Counterfactual Protein": "#ff7f0e"},
        ax=axes[0],
        show=False,
    )

    # Plot 2: Color by cell neighborhood (CN)
    sc.pl.umap(
        adata_combined_protein, color="CN", title="Cell Neighborhoods", ax=axes[1], show=False
    )

    # Plot 3: Color by cell types
    sc.pl.umap(
        adata_combined_protein, color="cell_types", title="Cell Types", ax=axes[2], show=False
    )

    # Also save as MLflow artifact if MLflow is active
    safe_mlflow_log_figure(
        plt.gcf(),
        f"counterfactual/protein_counterfactual_comparison_umap{suffix_str}_epoch_{epoch:04d}.pdf",
    )

    # Create the 3 subplots for PCA
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)

    # Compute PCA for the combined protein data
    sc.pp.pca(adata_combined_protein)

    # Plot 1: Color by modality (same-modal reconstruction vs counterfactual)
    sc.pl.pca(
        adata_combined_protein,
        color="data_type",
        title="Same-modal Protein Recon vs Counterfactual Protein"
        + (
            f"\nSim: {protein_similarity_score}, iLISI: {protein_ilisi_score}"
            if protein_similarity_score is not None and protein_ilisi_score is not None
            else ""
        ),
        palette={"Real Protein": "#1f77b4", "Counterfactual Protein": "#ff7f0e"},
        ax=axes[0],
        show=False,
    )

    # Plot 2: Color by cell neighborhood (CN)
    sc.pl.pca(
        adata_combined_protein, color="CN", title="Cell Neighborhoods", ax=axes[1], show=False
    )

    # Plot 3: Color by cell types
    sc.pl.pca(
        adata_combined_protein, color="cell_types", title="Cell Types", ax=axes[2], show=False
    )

    # Also save as MLflow artifact if MLflow is active
    safe_mlflow_log_figure(
        plt.gcf(),
        f"counterfactual/protein_counterfactual_comparison_pca{suffix_str}_epoch_{epoch:04d}.pdf",
    )


def set_consistent_cell_type_colors(adata_rna, adata_prot):
    """
    Set consistent cell type colors across both RNA and protein modalities.
    Uses custom palettes from color_palletes.py if dataset_name matches, otherwise falls back to matplotlib colormaps.

    Args:
        adata_rna: RNA AnnData object
        adata_prot: Protein AnnData object

    Returns:
        dict: Dictionary mapping cell types to hex colors
    """
    print("\n=== Setting consistent cell type colors across modalities ===")

    # Replace "T cells" with "GD/NK T" in both datasets
    if "T cells" in adata_rna.obs["cell_types"].cat.categories:
        adata_rna.obs["cell_types"] = adata_rna.obs["cell_types"].cat.rename_categories(
            {"T cells": "GD/NK T"}
        )
        print("Renamed 'T cells' to 'GD/NK T' in RNA data")
    else:
        print("No 'T cells' in RNA data")
    if "T cells" in adata_prot.obs["cell_types"].cat.categories:
        adata_prot.obs["cell_types"] = adata_prot.obs["cell_types"].cat.rename_categories(
            {"T cells": "GD/NK T"}
        )
        print("Renamed 'T cells' to 'GD/NK T' in Protein data")

    # Get all unique cell types from both modalities
    rna_cell_types = set(adata_rna.obs["cell_types"].unique())
    prot_cell_types = set(adata_prot.obs["cell_types"].unique())
    all_cell_types = sorted(list(rna_cell_types.union(prot_cell_types)))

    print(f"RNA cell types: {sorted(rna_cell_types)}")
    print(f"Protein cell types: {sorted(prot_cell_types)}")
    print(f"Union of all cell types: {all_cell_types}")

    # Create consistent color mapping for all cell types
    from matplotlib.colors import to_hex

    # Try to use custom palette from color_palletes.py if available
    dataset_name = adata_rna.uns.get("dataset_name", "").lower()
    cell_type_colors = {}

    # Import color palettes directly from arcadia.assets
    from arcadia.assets.color_palletes import synthetic_palette, tonsil_palette

    print(f"Dataset name from adata: '{dataset_name}'")

    # Select appropriate palette based on dataset name or cell types
    # Check dataset name first
    if "tonsil" in dataset_name or "maxfuse_tonsil" in dataset_name:
        custom_palette = tonsil_palette
        print(f"Using tonsil_palette for dataset: {dataset_name}")
    elif (
        "cite_seq" in dataset_name
        or "cite" in dataset_name
        or "synthetic" in dataset_name
        or "spleen" in dataset_name
    ):
        custom_palette = synthetic_palette
        print(f"Using synthetic_palette for dataset: {dataset_name}")
    else:
        # If dataset name doesn't match, try to infer from cell types
        # Check if cell types match synthetic palette better
        synthetic_matches = sum(1 for ct in all_cell_types if ct in synthetic_palette)
        tonsil_matches = sum(1 for ct in all_cell_types if ct in tonsil_palette)

        if synthetic_matches > tonsil_matches and synthetic_matches > 0:
            custom_palette = synthetic_palette
            print(
                f"Inferred synthetic_palette from cell types (matched {synthetic_matches}/{len(all_cell_types)})"
            )
        elif tonsil_matches > 0:
            custom_palette = tonsil_palette
            print(
                f"Inferred tonsil_palette from cell types (matched {tonsil_matches}/{len(all_cell_types)})"
            )
        else:
            custom_palette = None
            print(f"No matching palette found for dataset: {dataset_name}")

    # Use custom palette colors if cell types match
    if custom_palette is not None:
        for cell_type in all_cell_types:
            if cell_type in custom_palette:
                cell_type_colors[cell_type] = custom_palette[cell_type]

        print(
            f"Applied custom palette colors for {len(cell_type_colors)}/{len(all_cell_types)} cell types"
        )
        if len(cell_type_colors) < len(all_cell_types):
            missing = [ct for ct in all_cell_types if ct not in cell_type_colors]
            print(f"Missing cell types in palette: {missing}")

    # Fill in missing colors with matplotlib colormaps
    remaining_cell_types = [ct for ct in all_cell_types if ct not in cell_type_colors]
    if remaining_cell_types:
        n_colors = len(remaining_cell_types)
        if n_colors <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_colors))
        elif n_colors <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
        else:
            colors = plt.cm.Set3(np.linspace(0, 1, n_colors))

        for cell_type, color in zip(remaining_cell_types, colors):
            cell_type_colors[cell_type] = to_hex(color)

        print(f"Generated matplotlib colors for remaining {len(remaining_cell_types)} cell types")

    # Apply color mapping to both AnnData objects
    adata_rna.uns["cell_types_colors"] = [
        cell_type_colors[ct] for ct in adata_rna.obs["cell_types"].cat.categories
    ]
    adata_prot.uns["cell_types_colors"] = [
        cell_type_colors[ct] for ct in adata_prot.obs["cell_types"].cat.categories
    ]

    # Also set colors for major_cell_types if they exist
    if "major_cell_types" in adata_rna.obs.columns:
        major_cell_types = sorted(
            list(
                set(adata_rna.obs["major_cell_types"].unique()).union(
                    set(adata_prot.obs["major_cell_types"].unique())
                )
            )
        )
        n_major = len(major_cell_types)
        if n_major <= 10:
            major_colors = plt.cm.tab10(np.linspace(0, 1, n_major))
        elif n_major <= 20:
            major_colors = plt.cm.tab20(np.linspace(0, 1, n_major))
        else:
            major_colors = plt.cm.Set3(np.linspace(0, 1, n_major))

        major_cell_type_colors = {
            cell_type: to_hex(color) for cell_type, color in zip(major_cell_types, major_colors)
        }
        adata_rna.uns["major_cell_types_colors"] = [
            major_cell_type_colors[ct] for ct in adata_rna.obs["major_cell_types"].cat.categories
        ]
        adata_prot.uns["major_cell_types_colors"] = [
            major_cell_type_colors[ct] for ct in adata_prot.obs["major_cell_types"].cat.categories
        ]

    print("✅ Consistent cell type colors set for both modalities")

    return cell_type_colors


def plot_first_batch_umaps(
    rna_batch,
    protein_batch,
    rna_vae,
    protein_vae,
    adata_rna,
    adata_prot,
    colors=None,
    title=None,
    use_subsample=True,
    mlflow_name=None,
    plot_flag=True,
):
    """Plot UMAPs for RNA and protein data."""
    if not plot_flag:
        return
    #  run pca and umap on both batches
    cell_type_indexes = rna_batch["labels"]
    cell_type_labels = rna_vae.adata.obs["cell_types"][cell_type_indexes]
    batch_adata_rna = AnnData(
        X=rna_batch["X"].detach().cpu().numpy(),
        obs={"cell_types": cell_type_labels.values},
    )
    cell_type_indexes = protein_batch["labels"]
    cell_type_labels = protein_vae.adata.obs["cell_types"][cell_type_indexes]
    batch_adata_prot = AnnData(
        X=protein_batch["X"].detach().cpu().numpy(),
        obs={"cell_types": cell_type_labels.values},
    )
    # Create copies to avoid modifying the original data
    if use_subsample:
        # Subsample RNA data
        n_subsample_rna = min(700, adata_rna.shape[0], adata_prot.shape[0])
        subsample_idx_rna = np.random.choice(adata_rna.shape[0], n_subsample_rna, replace=False)
        adata_rna_plot = adata_rna[subsample_idx_rna].copy()

        # Subsample protein data
        n_subsample_prot = min(700, adata_prot.shape[0])
        subsample_idx_prot = np.random.choice(adata_prot.shape[0], n_subsample_prot, replace=False)
        protein_adata_plot = adata_prot[subsample_idx_prot].copy()
    else:
        adata_rna_plot = adata_rna.copy()
        protein_adata_plot = adata_prot.copy()
    if not adata_rna_plot.uns["pipeline_metadata"]["normalization"].get("log1p_applied", False):
        sc.pp.log1p(adata_rna_plot)
    sc.pp.pca(adata_rna_plot)
    sc.pp.neighbors(adata_rna_plot, key_added="original_neighbors", use_rep="X_pca")
    sc.tl.umap(adata_rna_plot, neighbors_key="original_neighbors")
    sc.pp.pca(protein_adata_plot)
    sc.pp.neighbors(protein_adata_plot, key_added="original_neighbors", use_rep="X_pca")
    sc.tl.umap(protein_adata_plot, neighbors_key="original_neighbors")
    # to mlflow
    if isinstance(mlflow_name, str):
        mlflow_name = [f"{mlflow_name}_rna_umap.pdf", f"{mlflow_name}_protein_umap.pdf"]
    elif isinstance(mlflow_name, list):
        pass
    else:
        raise ValueError(
            f"mlflow_name must be a string or a list of strings, got {type(mlflow_name)}"
        )
    if title is None:
        plot_title_rna = None
        plot_title_protein = None
    elif isinstance(title, str):
        plot_title_rna = [f"{title} RNA"]
        plot_title_protein = [f"{title} Protein"]
    elif isinstance(title, Iterable):
        plot_title_rna = [f"{t} RNA" for t in title]
        plot_title_protein = [f"{t} Protein" for t in title]
    else:
        raise ValueError(f"title must be a string or None, got {type(title)}")
    sc.pl.umap(adata_rna_plot, color=colors, title=plot_title_rna, legend_loc="on data")

    safe_mlflow_log_figure(plt.gcf(), mlflow_name[0])
    sc.pl.umap(protein_adata_plot, color=colors, title=plot_title_protein, legend_loc="on data")
    safe_mlflow_log_figure(plt.gcf(), mlflow_name[1])

    plt.figure()
    plt.subplot(2, 2, 1)
    x = (
        adata_rna_plot.X.toarray()
        if hasattr(adata_rna_plot.X, "toarray")
        else np.asarray(adata_rna_plot.X)
    )
    sns.histplot(x.flatten(), bins=100)
    x_2 = (
        protein_adata_plot.X.toarray()
        if hasattr(protein_adata_plot.X, "toarray")
        else np.asarray(protein_adata_plot.X)
    )
    plt.subplot(2, 2, 2)
    sns.histplot(x_2.flatten(), bins=100)
    plt.title("RNA VAE Adata X")
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "rna_vae_adata_X_hist.pdf")

    plt.figure()
    plt.subplot(2, 2, 1)
    sns.heatmap(x, cmap="viridis")
    plt.subplot(2, 2, 2)
    sns.heatmap(x_2, cmap="viridis")
    plt.title("Protein VAE Adata X")
    plt.tight_layout()
    safe_mlflow_log_figure(plt.gcf(), "rna_vae_adata_X_heatmap.pdf")
    plt.close()
