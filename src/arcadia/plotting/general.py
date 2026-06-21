"""General plotting utilities and helper functions."""

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import scanpy as sc
import seaborn as sns

from arcadia.utils.logging import logger


def safe_mlflow_log_figure(fig, file_path):
    """Safely log a figure to MLflow if an experiment is active."""
    try:
        # If file_path starts with step_, save to train folder in MLflow artifacts
        if file_path.startswith("step_"):
            # Extract step number and pad with leading zeros
            step_num = file_path.split("_")[1].split(".")[0]
            padded_step = f"{int(step_num):05d}"
            new_filename = f"step_{padded_step}_{'_'.join(file_path.split('_')[2:])}"
            # Log to MLflow in train folder with padded step number
            mlflow.log_figure(fig, f"train/{new_filename}")
            artifact_path = f"train/{new_filename}"

        else:
            # Regular logging for non-step files
            mlflow.log_figure(fig, file_path)
            artifact_path = file_path

        run_info = mlflow.active_run().info
        full_artifact_uri = f"{run_info.artifact_uri}/{artifact_path}"
        logger.info(
            f"Successfully logged figure to MLflow artifact URI: {full_artifact_uri.replace('file://', '')}"
        )
    except Exception as e:
        logger.warning(f"Could not log figure to MLflow: {str(e)}")
        logger.info("Continuing without MLflow logging...")


def compute_opacity_per_cell_type(
    adata,
    weight_col="matched_archetype_weight",
    cell_type_col="cell_types",
    alpha_min=0.15,
    gamma=4,
):
    """
    Compute opacity values (alphas) for each cell based on matched archetype weight,
    normalized per cell type to ensure each cell type uses the full opacity range.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the data
    weight_col : str
        Column name in adata.obs containing the archetype weight values
    cell_type_col : str
        Column name in adata.obs containing the cell type labels
    alpha_min : float
        Minimum alpha value (default: 0.15)
    gamma : float
        Gamma value for power transformation (default: 4)

    Returns
    -------
    alphas : np.ndarray
        Array of alpha values with same length as adata.n_obs
    """
    if weight_col not in adata.obs.columns:
        return np.ones(adata.n_obs)

    weights = adata.obs[weight_col].to_numpy()
    cell_types = adata.obs[cell_type_col].astype("category")
    alphas = np.zeros(adata.n_obs)

    # Normalize per cell type
    for ct in cell_types.cat.categories:
        mask = cell_types == ct
        ct_weights = weights[mask]

        if len(ct_weights) > 0:
            pmin, pmax = np.nanmin(ct_weights), np.nanmax(ct_weights)
            if pmax > pmin:
                norm = np.clip((ct_weights - pmin) / (pmax - pmin), 0, 1)
                ct_alphas = alpha_min + (1 - alpha_min) * (norm**gamma)
            else:
                ct_alphas = np.ones(len(ct_weights))
            alphas[mask] = ct_alphas

    return alphas


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
