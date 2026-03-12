import os
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import umap
from anndata import AnnData
from matplotlib.patches import FancyArrowPatch
from py_pcha import PCHA
from sklearn.manifold import TSNE

# Path manipulation removed - use package imports instead

# Import helper functions needed by plotting functions (avoid circular import)
# These are imported locally within functions that need them to avoid circular dependencies

# STANDARD FORMATTING CONSTANTS FOR CONSISTENCY ACROSS ALL PLOTTING FUNCTIONS
STANDARD_FIGURE_SIZE = (12, 8)
STANDARD_CELL_SIZE = 30
STANDARD_EXTREME_CELL_SIZE = 100
STANDARD_ARCHETYPE_SIZE = 500
STANDARD_ORIGIN_SIZE = 300
STANDARD_TARGET_SIZE = 200
STANDARD_CELL_ALPHA = 0.6
STANDARD_EXTREME_ALPHA = 1.0
STANDARD_GRID_ALPHA = 0.3
STANDARD_TEXT_SIZE = 14
STANDARD_LEGEND_SIZE = 10
STANDARD_ARROW_HEAD_WIDTH = 0.8
STANDARD_ARROW_HEAD_LENGTH = 1.0
STANDARD_ARROW_LINE_WIDTH = 3


# Global color scheme for consistent archetype coloring across all functions
def get_archetype_colors(n_archetypes):
    """Get consistent colors for archetypes across all visualization functions."""
    colors = plt.cm.tab10(np.linspace(0, 1, n_archetypes))
    return {i: colors[i] for i in range(n_archetypes)}


def shift_data_for_origin_placement(umap_coords, archetypes, margin_factor=0.1):
    """
    Shift UMAP coordinates and archetypes to position origin (0,0) down-left from data.

    Parameters:
    -----------
    umap_coords : np.ndarray
        UMAP coordinates of shape (n_cells, 2)
    archetypes : np.ndarray
        Archetype coordinates of shape (n_archetypes, 2)
    margin_factor : float
        Fraction of data range to use as margin from origin

    Returns:
    --------
    shifted_umap_coords : np.ndarray
        Shifted UMAP coordinates
    shifted_archetypes : np.ndarray
        Shifted archetype coordinates
    """
    # Combine all coordinates to get overall data range
    all_coords = np.vstack([umap_coords, archetypes])

    # Calculate data ranges
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Calculate margins
    x_range * margin_factor
    y_margin = y_range * margin_factor

    # Calculate shift to center x-axis around 0 and move y to positive values
    x_center = (x_min + x_max) / 2
    x_shift = -x_center  # Center x around 0
    y_shift = -y_min + y_margin  # Move y to positive with margin

    # Apply shifts - ensure float type for operations
    shifted_umap_coords = umap_coords.astype(float)
    shifted_umap_coords[:, 0] += x_shift
    shifted_umap_coords[:, 1] += y_shift

    shifted_archetypes = archetypes.astype(float)
    shifted_archetypes[:, 0] += x_shift
    shifted_archetypes[:, 1] += y_shift

    return shifted_umap_coords, shifted_archetypes


def get_archetypes(umap_coords, k=6):
    """Get archetypes from UMAP coordinates."""
    archetypes, archetype_activities, _, _, _ = PCHA(umap_coords.T, noc=k)
    # Use precomputed cell weights from PCHA archetype_activities (transposed to match expected shape)
    cell_weights = np.array(archetype_activities).T
    return archetypes, cell_weights


def plot_umap_with_archetypes(
    adata,
    archetypes,
    cell_weights,
    title="RNA UMAP with Archetypes",
    use_extreme_markers=False,
    plot_flag=True,
):
    """Plot UMAP with archetypes and cell assignments."""
    if not plot_flag:
        return
    umap_coords = adata.obsm["X_umap"]

    # Shift coordinates to position origin down-left from data
    shifted_umap_coords, shifted_archetypes = shift_data_for_origin_placement(
        umap_coords, archetypes
    )

    plt.figure(figsize=STANDARD_FIGURE_SIZE)

    if use_extreme_markers and "is_extreme_archetype" in adata.obs.columns:
        # Use different markers for extreme vs non-extreme archetypes
        extreme_mask = adata.obs["is_extreme_archetype"].astype(bool)
        non_extreme_mask = ~extreme_mask

        # Get archetype assignments for color mapping
        archetype_assignments = np.argmax(cell_weights, axis=1)
        n_archetypes = cell_weights.shape[1]

        # Create consistent color map for archetypes
        archetype_color_map = get_archetype_colors(n_archetypes)

        # Plot non-extreme cells with circle markers
        if non_extreme_mask.sum() > 0:
            for archetype_idx in range(n_archetypes):
                arch_mask = (archetype_assignments == archetype_idx) & non_extreme_mask
                if arch_mask.sum() > 0:
                    plt.scatter(
                        shifted_umap_coords[arch_mask, 0],
                        shifted_umap_coords[arch_mask, 1],
                        c=[archetype_color_map[archetype_idx]],
                        alpha=STANDARD_CELL_ALPHA,
                        marker="o",
                        s=STANDARD_CELL_SIZE,
                    )

        # Plot extreme cells with x markers
        if extreme_mask.sum() > 0:
            for archetype_idx in range(n_archetypes):
                arch_mask = (archetype_assignments == archetype_idx) & extreme_mask
                if arch_mask.sum() > 0:
                    plt.scatter(
                        shifted_umap_coords[arch_mask, 0],
                        shifted_umap_coords[arch_mask, 1],
                        c=[archetype_color_map[archetype_idx]],
                        alpha=STANDARD_EXTREME_ALPHA,
                        marker="x",
                        s=STANDARD_EXTREME_CELL_SIZE,
                    )

        # Plot archetypes
        archetype_scatter = plt.scatter(
            shifted_archetypes[:, 0],
            shifted_archetypes[:, 1],
            c="red",
            marker="*",
            s=STANDARD_ARCHETYPE_SIZE,
            edgecolors="black",
            linewidth=1,
        )

        # Create main legend for archetype categories
        from matplotlib.lines import Line2D

        main_legend_elements = []
        for archetype_idx in range(n_archetypes):
            main_legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=archetype_color_map[archetype_idx],
                    markersize=8,
                    label=f"Archetype {archetype_idx + 1}",
                )
            )
        main_legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="red",
                linestyle="None",
                markersize=12,
                label="Archetypes",
            )
        )

        main_legend = plt.legend(
            handles=main_legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=STANDARD_LEGEND_SIZE,
            title="Categories",
        )
        plt.gca().add_artist(main_legend)

        # Add second legend for marker types
        marker_legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="gray",
                linestyle="None",
                markersize=6,
                alpha=STANDARD_CELL_ALPHA,
                label="Normal cells",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                color="gray",
                linestyle="None",
                markersize=8,
                alpha=STANDARD_EXTREME_ALPHA,
                label="Extreme archetypes",
            ),
        ]
        plt.legend(
            handles=marker_legend_elements,
            bbox_to_anchor=(1.05, 0.3),
            loc="upper left",
            fontsize=STANDARD_LEGEND_SIZE,
            title="Markers",
        )
    else:
        # Default plotting without extreme markers
        scatter = plt.scatter(
            shifted_umap_coords[:, 0],
            shifted_umap_coords[:, 1],
            c=np.argmax(cell_weights, axis=1).flatten(),
            cmap="tab10",
            alpha=STANDARD_CELL_ALPHA,
        )

        # Plot archetypes
        plt.scatter(
            shifted_archetypes[:, 0],
            shifted_archetypes[:, 1],
            c="red",
            marker="*",
            s=STANDARD_ARCHETYPE_SIZE,
            label="Archetypes",
            edgecolors="black",
            linewidth=1,
        )

        # Add lines between archetypes to form a proper polygon
        if len(shifted_archetypes) > 2:
            # Calculate center point of archetypes
            center = np.mean(shifted_archetypes, axis=0)

            # Calculate angles from center to each archetype
            angles = np.arctan2(
                shifted_archetypes[:, 1] - center[1], shifted_archetypes[:, 0] - center[0]
            )

            # Sort archetypes by angle to create proper polygon order
            sorted_indices = np.argsort(angles)

            # Draw lines between consecutive archetypes in sorted order
            for i in range(len(sorted_indices)):
                current_idx = sorted_indices[i]
                next_idx = sorted_indices[
                    (i + 1) % len(sorted_indices)
                ]  # Wrap around to close polygon

                plt.plot(
                    [shifted_archetypes[current_idx, 0], shifted_archetypes[next_idx, 0]],
                    [shifted_archetypes[current_idx, 1], shifted_archetypes[next_idx, 1]],
                    "k-",
                    alpha=0.7,
                    linewidth=2,
                )

        # plt.colorbar(label="Archetype Assignment (largest weight)")
        plt.legend()

    # Add lines between archetypes to form a proper polygon (for extreme marker case too)
    if (
        use_extreme_markers
        and "is_extreme_archetype" in adata.obs.columns
        and len(shifted_archetypes) > 2
    ):
        # Calculate center point of archetypes
        center = np.mean(shifted_archetypes, axis=0)

        # Calculate angles from center to each archetype
        angles = np.arctan2(
            shifted_archetypes[:, 1] - center[1], shifted_archetypes[:, 0] - center[0]
        )

        # Sort archetypes by angle to create proper polygon order
        sorted_indices = np.argsort(angles)

        # Draw lines between consecutive archetypes in sorted order
        for i in range(len(sorted_indices)):
            current_idx = sorted_indices[i]
            next_idx = sorted_indices[(i + 1) % len(sorted_indices)]  # Wrap around to close polygon

            plt.plot(
                [shifted_archetypes[current_idx, 0], shifted_archetypes[next_idx, 0]],
                [shifted_archetypes[current_idx, 1], shifted_archetypes[next_idx, 1]],
                "k-",
                alpha=0.7,
                linewidth=2,
            )

    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.tight_layout()
    from arcadia.plotting.general import safe_mlflow_log_figure

    fig = plt.gcf()
    safe_mlflow_log_figure(fig, f"showcase_{title.lower().replace(' ', '_')}.pdf")
    plt.show()


def plot_umap_with_celltypes(
    adata,
    archetypes=None,
    title="RNA UMAP with Cell Types",
    use_extreme_markers=False,
    plot_flag=True,
):
    """Plot UMAP with cell types and archetypes (optional)."""
    if not plot_flag:
        return
    umap_coords = adata.obsm["X_umap"]

    # Shift coordinates if archetypes are provided
    if archetypes is not None:
        shifted_umap_coords, shifted_archetypes = shift_data_for_origin_placement(
            umap_coords, archetypes
        )
    else:
        shifted_umap_coords = umap_coords
        shifted_archetypes = None

    plt.figure(figsize=STANDARD_FIGURE_SIZE)

    if use_extreme_markers and "is_extreme_archetype" in adata.obs.columns:
        extreme_mask = adata.obs["is_extreme_archetype"].astype(bool)
        non_extreme_mask = ~extreme_mask
        cell_type_categories = adata.obs["cell_types"].cat.categories
        n_categories = len(cell_type_categories)
        colors = plt.cm.tab10(np.linspace(0, 1, n_categories))
        cell_type_color_map = {cat: colors[i] for i, cat in enumerate(cell_type_categories)}
        if non_extreme_mask.sum() > 0:
            for cell_type in cell_type_categories:
                ct_mask = (adata.obs["cell_types"] == cell_type) & non_extreme_mask
                if ct_mask.sum() > 0:
                    plt.scatter(
                        shifted_umap_coords[ct_mask, 0],
                        shifted_umap_coords[ct_mask, 1],
                        c=[cell_type_color_map[cell_type]],
                        alpha=STANDARD_CELL_ALPHA,
                        marker="o",
                        s=STANDARD_CELL_SIZE,
                    )
        if extreme_mask.sum() > 0:
            for cell_type in cell_type_categories:
                ct_mask = (adata.obs["cell_types"] == cell_type) & extreme_mask
                if ct_mask.sum() > 0:
                    plt.scatter(
                        shifted_umap_coords[ct_mask, 0],
                        shifted_umap_coords[ct_mask, 1],
                        c=[cell_type_color_map[cell_type]],
                        alpha=1,
                        marker="x",
                        s=STANDARD_EXTREME_CELL_SIZE * 3,
                    )
        if shifted_archetypes is not None:
            plt.scatter(
                shifted_archetypes[:, 0],
                shifted_archetypes[:, 1],
                c="red",
                marker="*",
                s=STANDARD_ARCHETYPE_SIZE,
                edgecolors="black",
                linewidth=1,
            )
        from matplotlib.lines import Line2D

        main_legend_elements = []
        for cell_type in cell_type_categories:
            main_legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=cell_type_color_map[cell_type],
                    markersize=8,
                    label=str(cell_type),
                )
            )
        if archetypes is not None:
            main_legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="red",
                    linestyle="None",
                    markersize=12,
                    label="Archetypes",
                )
            )
        main_legend = plt.legend(
            handles=main_legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=STANDARD_LEGEND_SIZE,
            title="Cell Types",
        )
        plt.gca().add_artist(main_legend)
        marker_legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="gray",
                linestyle="None",
                markersize=6,
                alpha=STANDARD_CELL_ALPHA,
                label="Normal cells",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                color="gray",
                linestyle="None",
                markersize=8,
                alpha=STANDARD_EXTREME_ALPHA,
                label="Extreme archetypes",
            ),
        ]
        plt.legend(
            handles=marker_legend_elements,
            bbox_to_anchor=(1.05, 0.3),
            loc="upper left",
            fontsize=STANDARD_LEGEND_SIZE,
            title="Markers",
        )
    else:
        # Plot cells colored by cell type, with string labels on colorbar
        cell_types_cat = adata.obs["cell_types"].astype("category")
        codes = cell_types_cat.cat.codes
        list(cell_types_cat.cat.categories)
        scatter = plt.scatter(
            shifted_umap_coords[:, 0],
            shifted_umap_coords[:, 1],
            c=codes,
            cmap="tab10",
            alpha=STANDARD_CELL_ALPHA,
        )
        if shifted_archetypes is not None:
            plt.scatter(
                shifted_archetypes[:, 0],
                shifted_archetypes[:, 1],
                c="red",
                marker="*",
                s=STANDARD_ARCHETYPE_SIZE,
                edgecolors="black",
                linewidth=1,
            )
        # cbar = plt.colorbar(scatter, ticks=np.arange(len(categories)))
        # cbar.ax.set_yticklabels(categories)
        # cbar.set_label("Cell Types")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.tight_layout()
    from arcadia.plotting.general import safe_mlflow_log_figure

    fig = plt.gcf()
    safe_mlflow_log_figure(fig, f"showcase_{title.lower().replace(' ', '_')}.pdf")
    plt.show()


def plot_cell_archetype_combination(
    umap_coords,
    archetypes,
    cell_weights,
    n_cells=1,
    seed=56,
    use_extreme_markers=False,
    plot_flag=True,
):
    """Plot cells as combinations of archetypes."""
    if not plot_flag:
        return
    np.random.seed(seed)

    # Shift coordinates to position origin down-left from data
    shifted_umap_coords, shifted_archetypes = shift_data_for_origin_placement(
        umap_coords, archetypes
    )

    # Calculate max archetype assignment for color mapping
    archetype_assignments = np.argmax(cell_weights, axis=1)
    n_archetypes = cell_weights.shape[1]

    if n_cells > len(shifted_umap_coords):
        n_cells = len(shifted_umap_coords)

    cell_indices = np.random.choice(len(shifted_umap_coords), size=n_cells, replace=False)

    plt.figure(figsize=STANDARD_FIGURE_SIZE)

    if use_extreme_markers:
        # Use consistent color map for archetypes
        archetype_color_map = get_archetype_colors(n_archetypes)

        # Plot all cells by archetype assignment
        for archetype_idx in range(n_archetypes):
            arch_mask = archetype_assignments == archetype_idx
            if arch_mask.sum() > 0:
                plt.scatter(
                    shifted_umap_coords[arch_mask, 0],
                    shifted_umap_coords[arch_mask, 1],
                    c=[archetype_color_map[archetype_idx]],
                    alpha=0.3,
                    marker="o",
                    s=50,
                )
    else:
        # Default plotting without extreme markers
        scatter = plt.scatter(
            shifted_umap_coords[:, 0],
            shifted_umap_coords[:, 1],
            c=np.argmax(cell_weights, axis=1),
            cmap="tab10",
            alpha=0.3,
            label="Cells",
        )

    plt.scatter(
        shifted_archetypes[:, 0],
        shifted_archetypes[:, 1],
        c="red",
        marker="*",
        s=500,
        label="Archetypes",
    )

    for idx in cell_indices:
        plt.scatter(
            shifted_umap_coords[idx, 0],
            shifted_umap_coords[idx, 1],
            c="black",
            s=150,
            edgecolor="yellow",
            zorder=5,
        )

        for j, (ax, ay) in enumerate(shifted_archetypes):
            weight = cell_weights[idx, j]
            if weight > 0.05:
                plt.plot(
                    [shifted_umap_coords[idx, 0], ax],
                    [shifted_umap_coords[idx, 1], ay],
                    color="gray",
                    alpha=weight,
                    linewidth=20 * weight + 0.5,
                )

        # plt.text(
        #     umap_coords[idx, 0],
        #     umap_coords[idx, 1],
        #     f"{idx}",
        #     color="yellow",
        #     fontsize=10,
        #     weight="bold",
        # )

    if not use_extreme_markers:
        # plt.colorbar(label="Archetype Assignment")
        plt.legend()
    else:
        # Create main legend for archetype categories
        from matplotlib.lines import Line2D

        main_legend_elements = []
        archetype_color_map = get_archetype_colors(n_archetypes)
        for archetype_idx in range(n_archetypes):
            main_legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=archetype_color_map[archetype_idx],
                    markersize=8,
                    label=f"Archetype {archetype_idx + 1}",
                )
            )
        main_legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="red",
                linestyle="None",
                markersize=12,
                label="Archetypes",
            )
        )
        main_legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markeredgecolor="yellow",
                linestyle="None",
                markersize=8,
                label="Selected Cells",
            )
        )

        main_legend = plt.legend(
            handles=main_legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=10,
            title="Categories",
        )
        plt.gca().add_artist(main_legend)

    plt.title("Cell Combinations with Archetypes")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.tight_layout()
    from arcadia.plotting.general import safe_mlflow_log_figure

    fig = plt.gcf()
    safe_mlflow_log_figure(fig, "showcase_cell_combinations_with_archetypes.pdf")
    plt.show()


def plot_archetype_vectors(
    umap_coords, archetypes, cell_weights=None, use_extreme_markers=False, plot_flag=True
):
    """Plot cells colored by archetype assignment with Archetypes."""
    if not plot_flag:
        return

    # Shift coordinates to position origin down-left from data
    shifted_umap_coords, shifted_archetypes = shift_data_for_origin_placement(
        umap_coords, archetypes
    )

    plt.figure(figsize=(12, 8))

    n_archetypes = len(shifted_archetypes)

    # Get consistent colors for archetypes
    archetype_color_map = get_archetype_colors(n_archetypes)

    if cell_weights is not None:
        # Plot cells colored by their archetype assignment
        archetype_assignments = np.argmax(cell_weights, axis=1)

        for archetype_idx in range(n_archetypes):
            arch_mask = archetype_assignments == archetype_idx
            if arch_mask.sum() > 0:
                plt.scatter(
                    shifted_umap_coords[arch_mask, 0],
                    shifted_umap_coords[arch_mask, 1],
                    c=[archetype_color_map[archetype_idx]],
                    alpha=0.6,
                    marker="o",
                    s=30,
                    label=f"Archetype {archetype_idx + 1}",
                )
    else:
        # Fallback: plot all cells in gray if no cell_weights provided
        plt.scatter(
            shifted_umap_coords[:, 0],
            shifted_umap_coords[:, 1],
            alpha=0.5,
            s=50,
            c="lightgray",
            label="Cells",
        )

    # Plot Archetypes as red stars
    plt.scatter(
        shifted_archetypes[:, 0],
        shifted_archetypes[:, 1],
        c="red",
        marker="*",
        s=500,
        label="Archetypes",
        edgecolors="black",
        linewidth=1,
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.title("Cells Colored by Archetype Assignment")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.tight_layout()
    from arcadia.plotting.general import safe_mlflow_log_figure

    fig = plt.gcf()
    safe_mlflow_log_figure(fig, "showcase_cells_colored_by_archetype_assignment.pdf")
    plt.show()


def plot_archetype_vectors_with_arrows(
    umap_coords,
    archetypes,
    cell_weights,
    n_cells=2,
    seed=42,
    threshold=0.05,
    use_extreme_markers=False,
    plot_flag=True,
):
    """Plot archetype vectors as arrows from origin to archetype centers with cells colored by archetype assignment."""
    if not plot_flag:
        return
    np.random.seed(seed)

    # Shift coordinates to position origin down-left from data
    shifted_umap_coords, shifted_archetypes = shift_data_for_origin_placement(
        umap_coords, archetypes
    )

    plt.figure(figsize=(12, 8))

    n_archetypes = len(shifted_archetypes)

    # Get consistent colors for archetypes
    archetype_color_map = get_archetype_colors(n_archetypes)

    # Plot cells colored by their archetype assignment
    if cell_weights is not None:
        archetype_assignments = np.argmax(cell_weights, axis=1)

        for archetype_idx in range(n_archetypes):
            arch_mask = archetype_assignments == archetype_idx
            if arch_mask.sum() > 0:
                plt.scatter(
                    shifted_umap_coords[arch_mask, 0],
                    shifted_umap_coords[arch_mask, 1],
                    c=[archetype_color_map[archetype_idx]],
                    alpha=0.5,
                    marker="o",
                    s=50,
                    label=f"Archetype {archetype_idx + 1} cells",
                )
    else:
        plt.scatter(
            shifted_umap_coords[:, 0],
            shifted_umap_coords[:, 1],
            alpha=0.5,
            s=50,
            c="lightgray",
            label="Cells",
        )

    # Plot archetypes as red stars
    plt.scatter(
        shifted_archetypes[:, 0],
        shifted_archetypes[:, 1],
        c="red",
        marker="*",
        s=500,
        label="Archetype Centers",
        edgecolors="black",
        linewidth=1,
    )

    # Plot archetype vectors from origin (0,0) with colored body and black arrowhead
    origin = np.array([0.0, 0.0])
    for i, archetype in enumerate(shifted_archetypes):
        # Draw black outline arrow (thinner, underneath)
        arrow_outline = FancyArrowPatch(
            posA=origin,
            posB=archetype,
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=3,
            color="black",
            alpha=1.0,
            zorder=2,
            shrinkA=0,
            shrinkB=0,
            linestyle="-",
            fill=False,
        )
        plt.gca().add_patch(arrow_outline)
        # Draw colored arrow body (thicker, on top)
        arrow = FancyArrowPatch(
            posA=origin,
            posB=archetype,
            arrowstyle="-|>",
            mutation_scale=28,
            linewidth=6,
            color=archetype_color_map[i],
            alpha=0.85,
            zorder=3,
            shrinkA=0,
            shrinkB=0.2,
            linestyle="-",
        )
        plt.gca().add_patch(arrow)

    plt.scatter(origin[0], origin[1], c="black", marker="x", s=300, label="Origin (0,0)")
    plt.text(
        origin[0] + 0.5,
        origin[1] + 0.5,
        "Origin",
        fontsize=14,
        fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8),
    )

    # Create legend for archetype vectors
    from matplotlib.lines import Line2D

    main_legend_elements = []

    # Add archetype cell colors to legend
    for archetype_idx in range(n_archetypes):
        main_legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=archetype_color_map[archetype_idx],
                markersize=6,
                label=f"Archetype {archetype_idx + 1} cells",
            )
        )

    # Add other elements
    main_legend_elements.extend(
        [
            Line2D(
                [0],
                [0],
                marker="*",
                color="red",
                linestyle="None",
                markersize=12,
                markeredgecolor="black",
                label="Archetype Centers",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                color="black",
                linestyle="None",
                markersize=12,
                label="Origin (0,0)",
            ),
        ]
    )

    # Add vector elements
    for i in range(len(archetypes)):
        main_legend_elements.append(
            Line2D(
                [0],
                [0],
                color=archetype_color_map[i],
                linewidth=3,
                label=f"Vector to Archetype {i+1}",
            )
        )

    plt.legend(
        handles=main_legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=10,
        title="Elements",
    )

    plt.title(f"Archetype Vectors from Origin (0,0)\nThreshold: {threshold}")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.axis("equal")
    plt.tight_layout()
    from arcadia.plotting.general import safe_mlflow_log_figure

    fig = plt.gcf()
    safe_mlflow_log_figure(fig, "showcase_archetype_vectors_with_arrows.pdf")
    plt.show()


def plot_archetypes_with_cell_arch_label(
    umap_coords,
    archetypes,
    cell_weights,
    n_cells=2,
    seed=871,
    threshold=0.5,
    use_extreme_markers=False,
    plot_flag=True,
):
    """Plot detailed cell combinations with archetype vectors."""
    if not plot_flag:
        return
    np.random.seed(seed)

    if n_cells > len(umap_coords):
        n_cells = len(umap_coords)

    cell_indices = np.random.choice(len(umap_coords), size=n_cells, replace=False)

    # Shift coordinates to position origin down-left from data
    shifted_umap_coords, shifted_archetypes = shift_data_for_origin_placement(
        umap_coords, archetypes
    )

    n_archetypes = len(shifted_archetypes)

    # Create consistent color map
    archetype_color_map = get_archetype_colors(n_archetypes)

    plt.figure(figsize=(16, 12))

    if use_extreme_markers:
        # Plot all cells by archetype assignment
        archetype_assignments = np.argmax(cell_weights, axis=1)

        # Plot all cells by archetype assignment with consistent colors
        for archetype_idx in range(n_archetypes):
            arch_mask = archetype_assignments == archetype_idx
            if arch_mask.sum() > 0:
                plt.scatter(
                    shifted_umap_coords[arch_mask, 0],
                    shifted_umap_coords[arch_mask, 1],
                    c=[archetype_color_map[archetype_idx]],
                    alpha=0.3,
                    marker="o",
                    s=50,
                )
    else:
        # Default plotting without extreme markers
        scatter = plt.scatter(
            shifted_umap_coords[:, 0],
            shifted_umap_coords[:, 1],
            c=np.argmax(cell_weights, axis=1),
            cmap="tab10",
            vmin=0,
            vmax=n_archetypes - 1,
            alpha=0.3,
            label="Cells",
        )

    for i, (x, y) in enumerate(shifted_archetypes):
        plt.scatter(
            x,
            y,
            c=[archetype_color_map[i]],  # Use consistent colors
            marker="*",
            s=700,
            edgecolor="black",
            linewidths=2,
            label=f"Archetype {i+1}",
        )

    target_counter = 1
    for idx in cell_indices:
        steps = [np.zeros(2)]  # Start from origin (0,0) - FIXED!
        archetype_indices = []
        for j in range(n_archetypes):
            weight = cell_weights[idx, j]
            if weight > threshold:
                next_step = steps[-1] + weight * shifted_archetypes[j]
                steps.append(next_step)
                archetype_indices.append(j)
        steps = np.array(steps)

        # Plot the selected cell at its actual UMAP position with a special marker
        plt.scatter(
            shifted_umap_coords[idx, 0],
            shifted_umap_coords[idx, 1],
            c="black",
            s=200,
            edgecolor="red",
            linewidth=3,
            marker="o",
            zorder=10,
            label=f"Target Cell {target_counter}" if target_counter == 1 else "",
        )

        # Add text label next to the target point
        plt.text(
            shifted_umap_coords[idx, 0] + 0.5,  # Offset to the right
            shifted_umap_coords[idx, 1] + 0.5,  # Offset upward
            f"Target {target_counter}",
            fontsize=14,
            fontweight="bold",
            color="red",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8),
        )

        # Draw arrows showing step-by-step archetype composition from origin
        for k in range(1, len(steps)):
            # Black outline arrow (thinner, underneath)
            arrow_outline = FancyArrowPatch(
                posA=steps[k - 1],
                posB=steps[k],
                arrowstyle="-|>",
                mutation_scale=17,
                linewidth=3,
                color="black",
                alpha=1.0,
                zorder=2,
                shrinkA=0,
                shrinkB=0,
                linestyle="-",
                fill=False,
            )
            plt.gca().add_patch(arrow_outline)
            # Colored arrow body (thicker, on top)
            arrow = FancyArrowPatch(
                posA=steps[k - 1],
                posB=steps[k],
                arrowstyle="-|>",
                mutation_scale=22,
                linewidth=6,
                color=archetype_color_map[archetype_indices[k - 1]],
                alpha=0.85,
                zorder=3,
                shrinkA=0,
                shrinkB=0.2,
                linestyle="-",
            )
            plt.gca().add_patch(arrow)

        # Mark the origin (0,0)
        plt.scatter(
            0,
            0,
            c="black",
            marker="x",
            s=300,
            linewidth=3,
            label="Origin (0,0)" if target_counter == 1 else "",
        )

        # Add "Origin" text next to the origin point (only for the first target to avoid repetition)
        if target_counter == 1:
            plt.text(
                0.5,  # Offset to the right
                0.5,  # Offset upward
                "Origin",
                fontsize=14,
                fontweight="bold",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8
                ),
            )

        # Note: Composed position marker removed as requested

        target_counter += 1

    if not use_extreme_markers:
        # plt.colorbar(label="Archetype Assignment")
        plt.legend(bbox_to_anchor=(1.15, 1), loc="upper left", borderaxespad=0, labelspacing=1.2)
    else:
        # Create main legend for archetype categories
        from matplotlib.lines import Line2D

        main_legend_elements = []
        for archetype_idx in range(n_archetypes):
            main_legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="w",
                    markerfacecolor=archetype_color_map[archetype_idx],
                    markeredgecolor="black",
                    markersize=12,
                    label=f"Archetype {archetype_idx + 1}",
                )
            )
        main_legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markeredgecolor="red",
                linestyle="None",
                markersize=10,
                label="Selected Cells",
            )
        )
        main_legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markeredgecolor="red",
                linestyle="None",
                markersize=10,
                label="Origin (0,0)",  # Updated label
            )
        )

        main_legend = plt.legend(
            handles=main_legend_elements,
            bbox_to_anchor=(1.15, 1),
            loc="upper left",
            fontsize=10,
            title="Categories",
        )
        plt.gca().add_artist(main_legend)

    plt.title("Detailed Cell Combination from Origin (0,0)")  # Updated title
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.tight_layout()
    from arcadia.plotting.general import safe_mlflow_log_figure

    fig = plt.gcf()
    safe_mlflow_log_figure(fig, "showcase_detailed_cell_combination.pdf")
    plt.show()


def plot_archetypes(
    data_points,
    archetype,
    samples_cell_types: List[str],
    data_point_archetype_indices: List[int],
    modality="",
    cell_type_colors: Dict[str, Any] = None,
    max_points=2000,
    plot_flag=True,
    plot_pca=True,
    plot_umap=True,
    extreme_archetype_mask=None,
    matched_archetype_weight=None,
    archetype_quality_dict=None,
    lines_attached="closest",
):
    """Plot archetypes with subsampling for large datasets.

    Parameters:
    -----------
    data_points : array
        Data points matrix
    archetype : array
        Archetype matrix
    samples_cell_types : List[str]
        Cell type for each data point
    data_point_archetype_indices : List[int]
        Archetype index for each data point
    modality : str
        Modality name for plotting
    cell_type_colors : Dict[str, Any]
        Cell type color mapping
    max_points : int
        Maximum number of points to plot
    plot_pca : bool
        Whether to plot PCA visualization
    plot_umap : bool
        Whether to plot UMAP visualization
    extreme_archetype_mask : array or None
        Boolean mask indicating which data points are extreme archetypes
    matched_archetype_weight : array or None
        Array of cells matched archetype weight values for opacity mapping
    archetype_quality_dict : dict or None
        Dictionary mapping archetype index to quality (True=good, False=poor)
    lines_attached : str
        Method for selecting cells to attach lines to archetypes.
        'random': randomly select cells (default)
        'closest': select the closest cells to each archetype
    """
    if not plot_flag:
        return
    if not isinstance(samples_cell_types, List):
        raise TypeError("samples_cell_types should be a list of strings.")
    if not isinstance(data_point_archetype_indices, List):
        raise TypeError("data_point_archetype_indices should be a list of integers.")
    if len(data_points) != len(samples_cell_types) or len(data_points) != len(
        data_point_archetype_indices
    ):
        raise ValueError(
            "Length of data_points, samples_cell_types, and data_point_archetype_indices must be equal."
        )

    # Check the shapes of data_points and archetype
    print("Shape of data_points:", data_points.shape)
    print("Shape of archetype before any adjustment:", archetype.shape)

    # Ensure archetype has the same number of features as data_points
    if archetype.shape[1] != data_points.shape[1]:
        # Check if transposing helps
        if archetype.T.shape[1] == data_points.shape[1]:
            print("Transposing archetype array to match dimensions.")
            archetype = archetype.T
        else:
            raise ValueError("archetype array cannot be reshaped to match data_points dimensions.")

    print("Shape of archetype after adjustment:", archetype.shape)

    # Apply subsampling if data_points is too large
    if len(data_points) > max_points:
        print(f"Subsampling data to {max_points} points for visualization")
        # Create random indices for subsampling
        subsample_indices = np.random.choice(len(data_points), max_points, replace=False)
        data_points_subset = data_points[subsample_indices]
        samples_cell_types_subset = [samples_cell_types[i] for i in subsample_indices]
        data_point_archetype_indices_subset = [
            data_point_archetype_indices[i] for i in subsample_indices
        ]
        # Handle extreme archetype mask if provided
        if extreme_archetype_mask is not None:
            extreme_archetype_mask_subset = extreme_archetype_mask[subsample_indices]
        else:
            extreme_archetype_mask_subset = None
        # Handle matched arch weight if provided
        if matched_archetype_weight is not None:
            matched_archetype_weight_subset = matched_archetype_weight[subsample_indices]
        else:
            matched_archetype_weight_subset = None
    else:
        data_points_subset = data_points
        samples_cell_types_subset = samples_cell_types
        data_point_archetype_indices_subset = data_point_archetype_indices
        extreme_archetype_mask_subset = extreme_archetype_mask
        matched_archetype_weight_subset = matched_archetype_weight

    # Combine data points and archetypes
    num_archetypes = archetype.shape[0]
    data = np.concatenate((data_points_subset, archetype), axis=0)
    labels = ["data"] * len(data_points_subset) + ["archetype"] * num_archetypes
    cell_types = samples_cell_types_subset + ["archetype"] * num_archetypes

    # Compute alphas from matched archetype weight if provided, otherwise use full opacity
    if matched_archetype_weight_subset is not None:
        alpha_min = 0.15
        pmin, pmax = np.nanmin(matched_archetype_weight_subset), np.nanmax(
            matched_archetype_weight_subset
        )
        gamma = 4
        norm = np.clip((matched_archetype_weight_subset - pmin) / (pmax - pmin), 0, 1)
        alphas = alpha_min + (1 - alpha_min) * (norm**gamma)
    else:
        alphas = np.ones(len(data_points_subset))

    # Perform PCA and UMAP with limited dimensions for efficiency
    data_pca = data[:, : min(50, data.shape[1])]

    # Run UMAP with subsampling if needed
    umap_max_points = min(2000, data.shape[0])  # UMAP is more scalable than t-SNE
    if data.shape[0] > umap_max_points:
        print(f"Further subsampling to {umap_max_points} points for UMAP")
        # Always include archetype points in UMAP
        archetype_start_idx = len(data_points_subset)
        archetype_indices = list(range(archetype_start_idx, archetype_start_idx + num_archetypes))

        # Sample data points (excluding archetypes)
        data_indices = list(range(len(data_points_subset)))
        num_data_to_sample = min(umap_max_points - num_archetypes, len(data_indices))
        sampled_data_indices = np.random.choice(data_indices, num_data_to_sample, replace=False)

        # Combine sampled data indices with archetype indices
        umap_indices = list(sampled_data_indices) + archetype_indices

        umap_data = data_pca[umap_indices]
        data_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(umap_data)

        # Map back to original indices
        umap_labels = [labels[i] for i in umap_indices]
        umap_cell_types = [cell_types[i] for i in umap_indices]
        umap_data_point_arch_indices = []
        umap_archetype_numbers = []

        for idx, i in enumerate(umap_indices):
            if i < len(data_point_archetype_indices_subset):
                # This is a data point
                umap_data_point_arch_indices.append(data_point_archetype_indices_subset[i])
                umap_archetype_numbers.append(np.nan)
            else:
                # This is an archetype
                archetype_num = i - len(data_point_archetype_indices_subset)
                umap_data_point_arch_indices.append(np.nan)
                umap_archetype_numbers.append(archetype_num)
    else:
        data_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(data_pca)
        umap_labels = labels
        umap_cell_types = cell_types
        umap_data_point_arch_indices = (
            data_point_archetype_indices_subset + [np.nan] * num_archetypes
        )
        umap_archetype_numbers = [np.nan] * len(data_points_subset) + list(range(num_archetypes))

    # Create a numbering for archetypes
    archetype_numbers = [np.nan] * len(data_points_subset) + list(range(num_archetypes))

    # Create DataFrames for plotting
    df_pca = pd.DataFrame(
        {
            "PCA1": data_pca[:, 0],
            "PCA2": data_pca[:, 1],
            "type": labels,
            "cell_type": cell_types,
            "archetype_number": archetype_numbers,
            "data_point_archetype_index": data_point_archetype_indices_subset
            + [np.nan] * num_archetypes,
        }
    )

    df_umap = pd.DataFrame(
        {
            "UMAP1": data_umap[:, 0],
            "UMAP2": data_umap[:, 1],
            "type": umap_labels,
            "cell_type": umap_cell_types,
            "archetype_number": umap_archetype_numbers,
            "data_point_archetype_index": umap_data_point_arch_indices,
        }
    )

    # Use the provided color mapping or generate a new one
    if cell_type_colors is not None:
        palette_dict = cell_type_colors
    else:
        # Define color palette based on unique cell types
        unique_cell_types = list(pd.unique(samples_cell_types))
        palette = sns.color_palette("tab20", len(unique_cell_types))
        palette_dict = {cell_type: color for cell_type, color in zip(unique_cell_types, palette)}
        palette_dict["archetype"] = "black"  # Assign black to archetype

    # Ensure 'archetype' color is set
    if "archetype" not in palette_dict:
        palette_dict["archetype"] = "black"

    # Normalize all colors to RGB tuples for consistent handling
    from matplotlib.colors import to_rgb

    normalized_palette = {}
    for cell_type, color in palette_dict.items():
        if isinstance(color, str):
            # Convert hex string or named color to RGB tuple
            normalized_palette[cell_type] = to_rgb(color)
        else:
            # Already a tuple/array, ensure it's a tuple
            normalized_palette[cell_type] = tuple(color[:3]) if len(color) >= 3 else tuple(color)
    palette_dict = normalized_palette

    # Plot PCA
    if plot_pca:
        plt.figure(figsize=(10, 10))
        df_pca = df_pca.sort_values(by="cell_type")

        # Plot data points first (lowest z-order)
        df_pca_data_only = df_pca[df_pca["type"] == "data"]
        df_pca_archetypes_only = df_pca[df_pca["type"] == "archetype"]

        # Handle extreme archetypes if mask is provided
        if extreme_archetype_mask_subset is not None:
            # Split data points into extreme and non-extreme
            extreme_mask = extreme_archetype_mask_subset
            # Use boolean indexing with numpy arrays
            extreme_indices = np.where(extreme_mask)[0]
            normal_indices = np.where(~extreme_mask)[0]
            df_pca_data_extreme = df_pca_data_only.iloc[extreme_indices]
            df_pca_data_normal = df_pca_data_only.iloc[normal_indices]

            # Plot normal data points with circles
            if len(df_pca_data_normal) > 0:
                normal_indices = np.where(~extreme_mask)[0]
                for cell_type in df_pca_data_normal["cell_type"].unique():
                    if cell_type == "archetype":
                        continue
                    ct_mask = df_pca_data_normal["cell_type"] == cell_type
                    ct_data = df_pca_data_normal[ct_mask]
                    ct_indices_in_normal = np.where(ct_mask)[0]
                    ct_alphas = alphas[normal_indices[ct_indices_in_normal]]
                    rgb = palette_dict[cell_type]
                    rgba = np.column_stack([np.tile(rgb, (len(ct_data), 1)), ct_alphas])
                    plt.scatter(
                        ct_data["PCA1"],
                        ct_data["PCA2"],
                        c=rgba,
                        s=50,
                        zorder=1,
                        marker="o",
                    )

            # Plot extreme data points with x markers
            if len(df_pca_data_extreme) > 0:
                for cell_type in df_pca_data_extreme["cell_type"].unique():
                    if cell_type == "archetype":
                        continue
                    ct_data = df_pca_data_extreme[df_pca_data_extreme["cell_type"] == cell_type]
                    plt.scatter(
                        ct_data["PCA1"],
                        ct_data["PCA2"],
                        c=[palette_dict[cell_type]],
                        marker="x",
                        s=150,
                        alpha=1.0,
                        zorder=3,
                    )
        else:
            # Plot data points normally
            for cell_type in df_pca_data_only["cell_type"].unique():
                if cell_type == "archetype":
                    continue
                ct_mask = df_pca_data_only["cell_type"] == cell_type
                ct_data = df_pca_data_only[ct_mask]
                ct_indices = np.where(ct_mask)[0]
                ct_alphas = alphas[ct_indices]
                rgb = palette_dict[cell_type]
                rgba = np.column_stack([np.tile(rgb, (len(ct_data), 1)), ct_alphas])
                plt.scatter(
                    ct_data["PCA1"],
                    ct_data["PCA2"],
                    c=rgba,
                    s=50,
                    zorder=1,
                    marker="o",
                )

        # Plot archetype points on top with X markers, colored by quality
        if archetype_quality_dict is not None:
            # Split archetypes by quality
            good_quality_archetypes = []
            poor_quality_archetypes = []

            for idx, row in df_pca_archetypes_only.iterrows():
                archetype_num = (
                    int(row["archetype_number"]) if not pd.isna(row["archetype_number"]) else None
                )
                if archetype_num is not None and archetype_num in archetype_quality_dict:
                    if archetype_quality_dict[archetype_num]:
                        good_quality_archetypes.append(idx)
                    else:
                        poor_quality_archetypes.append(idx)
                else:
                    # Default to poor quality if not found
                    poor_quality_archetypes.append(idx)

            # Plot good quality archetypes in black
            if good_quality_archetypes:
                good_df = df_pca_archetypes_only.iloc[
                    [
                        idx
                        for idx in range(len(df_pca_archetypes_only))
                        if df_pca_archetypes_only.index[idx] in good_quality_archetypes
                    ]
                ]
                plt.scatter(
                    good_df["PCA1"],
                    good_df["PCA2"],
                    marker="^",
                    s=800,
                    c="black",
                    edgecolors="white",
                    linewidth=0.5,
                    zorder=5,
                    label="Good Quality Archetypes",
                )

            # Plot poor quality archetypes in red
            if poor_quality_archetypes:
                poor_df = df_pca_archetypes_only.iloc[
                    [
                        idx
                        for idx in range(len(df_pca_archetypes_only))
                        if df_pca_archetypes_only.index[idx] in poor_quality_archetypes
                    ]
                ]
                plt.scatter(
                    poor_df["PCA1"],
                    poor_df["PCA2"],
                    marker="^",
                    s=800,
                    c="red",
                    edgecolors="white",
                    linewidth=0.5,
                    zorder=5,
                    label="Poor Quality Archetypes",
                )
        else:
            # Default behavior - all archetypes in black
            plt.scatter(
                df_pca_archetypes_only["PCA1"],
                df_pca_archetypes_only["PCA2"],
                marker="^",
                s=500,
                c="black",
                edgecolors="white",
                linewidth=0.5,
                zorder=5,
                label="Archetypes",
            )

        # Create custom legend with manual entries for cell types
        from matplotlib.lines import Line2D

        legend_elements = []
        all_labels = []

        # Create manual legend entries for cell types
        unique_cell_types = sorted(
            [ct for ct in df_pca_data_only["cell_type"].unique() if ct != "archetype"]
        )
        for cell_type in unique_cell_types:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=palette_dict[cell_type],
                    markersize=8,
                    label=cell_type,
                )
            )
            all_labels.append(cell_type)

        # Add quality-based archetype legends or single archetype legend
        if archetype_quality_dict is not None:
            # Add good quality archetype legend
            good_archetype_legend = Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="black",
                markeredgecolor="white",
                markersize=12,
                linewidth=2,
                label="Good Quality Archetypes",
            )
            legend_elements.append(good_archetype_legend)
            all_labels.append("Good Quality Archetypes")

            # Add poor quality archetype legend
            poor_archetype_legend = Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="red",
                markeredgecolor="white",
                markersize=12,
                linewidth=2,
                label="Poor Quality Archetypes",
            )
            legend_elements.append(poor_archetype_legend)
            all_labels.append("Poor Quality Archetypes")
        else:
            # Default single archetype legend
            archetype_legend = Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="black",
                markeredgecolor="white",
                markersize=12,
                linewidth=2,
                label="Archetypes",
            )
            legend_elements.append(archetype_legend)
            all_labels.append("Archetypes")

        if extreme_archetype_mask_subset is not None:
            extreme_legend = Line2D(
                [0],
                [0],
                marker="x",
                color="gray",
                linestyle="None",
                markersize=8,
                label="Extreme Archetypes",
            )
            legend_elements.append(extreme_legend)
            all_labels.append("Extreme Archetypes")

        plt.legend(
            legend_elements,
            all_labels,
            title="Cell Types",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        # Annotate archetype points with numbers
        archetype_points = df_pca[df_pca["type"] == "archetype"]
        for _, row in archetype_points.iterrows():
            if not pd.isna(row["archetype_number"]):
                plt.text(
                    row["PCA1"],
                    row["PCA2"],
                    str(int(row["archetype_number"])),
                    fontsize=12,
                    fontweight="bold",
                    color="red",
                    zorder=10,
                    ha="center",
                    va="center",
                )
        # Add lines from each data point to its matching archetype
        df_pca_data = df_pca[df_pca["type"] == "data"].copy()
        df_pca_archetypes = df_pca[df_pca["type"] == "archetype"].copy()

        # Clean and prepare archetype coordinates mapping
        df_pca_archetypes_clean = df_pca_archetypes.dropna(subset=["archetype_number"]).copy()
        df_pca_archetypes_clean["archetype_number"] = df_pca_archetypes_clean[
            "archetype_number"
        ].astype(int)

        # Create a mapping from archetype_number to its PCA coordinates
        archetype_coords = df_pca_archetypes_clean.set_index("archetype_number")[["PCA1", "PCA2"]]

        # Now for each data point, draw a line to its corresponding archetype, limiting to max_lines
        max_lines = min(1000, len(df_pca_data))  # Limit number of lines to prevent clutter

        if lines_attached == "closest":
            # For each archetype, find the closest cells
            lines_per_archetype = max_lines // num_archetypes
            selected_indices = []

            for arch_num in archetype_coords.index:
                arch_coord = archetype_coords.loc[arch_num]
                # Filter data points that belong to this archetype
                arch_data = df_pca_data[
                    df_pca_data["data_point_archetype_index"] == arch_num
                ].copy()

                if len(arch_data) > 0:
                    # Calculate distances to this archetype
                    arch_data["distance"] = np.sqrt(
                        (arch_data["PCA1"] - arch_coord["PCA1"]) ** 2
                        + (arch_data["PCA2"] - arch_coord["PCA2"]) ** 2
                    )
                    # Select closest cells
                    n_select = min(lines_per_archetype, len(arch_data))
                    closest = arch_data.nsmallest(n_select, "distance")
                    selected_indices.extend(closest.index.tolist())

            df_pca_data_subset = df_pca_data.loc[selected_indices]
            print(
                f"Attaching lines to {len(df_pca_data_subset)} closest cells "
                f"({lines_per_archetype} per archetype)"
            )
        else:
            # Random selection (default behavior)
            if len(df_pca_data) > max_lines:
                print(f"Limiting connection lines to {max_lines} for visualization clarity")
                line_indices = np.random.choice(len(df_pca_data), max_lines, replace=False)
                df_pca_data_subset = df_pca_data.iloc[line_indices]
            else:
                df_pca_data_subset = df_pca_data

        for idx, row in df_pca_data_subset.iterrows():
            if pd.isna(row["data_point_archetype_index"]):
                continue
            archetype_index = int(row["data_point_archetype_index"])
            data_point_coords = (row["PCA1"], row["PCA2"])
            archetype_point_coords = archetype_coords.loc[archetype_index]
            plt.plot(
                [data_point_coords[0], archetype_point_coords["PCA1"]],
                [data_point_coords[1], archetype_point_coords["PCA2"]],
                color="black",
                linewidth=1.5,
                alpha=0.3,
            )

        plt.legend(
            legend_elements,
            all_labels,
            title="Cell Types",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        plt.title(f"{modality} PCA: Data Points and Archetypes\nColored by Cell Types")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")

        # Set axis limits dynamically based on data range with 5% margin
        pca1_min, pca1_max = df_pca["PCA1"].min(), df_pca["PCA1"].max()
        pca2_min, pca2_max = df_pca["PCA2"].min(), df_pca["PCA2"].max()
        pca1_range = pca1_max - pca1_min
        pca2_range = pca2_max - pca2_min
        pca1_margin = pca1_range * 0.05
        pca2_margin = pca2_range * 0.05
        plt.xlim(pca1_min - pca1_margin, pca1_max + pca1_margin)
        plt.ylim(pca2_min - pca2_margin, pca2_max + pca2_margin)

        # Remove grid lines, tick marks, and tick labels
        plt.grid(False)
        ax = plt.gca()
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        plt.tight_layout()

        # Save to MLflow
        modality_safe = modality.replace(" ", "_").replace("/", "_")
        from arcadia.plotting.general import safe_mlflow_log_figure

        safe_mlflow_log_figure(plt.gcf(), f"archetypes/{modality_safe}_PCA.pdf")

        plt.show()

    # Plot UMAP
    if plot_umap:
        plt.figure(figsize=(10, 10))
        df_umap = df_umap.sort_values(by="cell_type")

        # Plot data points first (lowest z-order)
        df_umap_data_only = df_umap[df_umap["type"] == "data"]
        df_umap_archetypes_only = df_umap[df_umap["type"] == "archetype"]

        # Handle extreme archetypes if mask is provided
        if extreme_archetype_mask_subset is not None:
            # For UMAP, we need to map back to the subsampled indices
            # Create a mapping from original to UMAP indices
            umap_extreme_mask = np.zeros(len(df_umap_data_only), dtype=bool)
            if (
                hasattr(extreme_archetype_mask_subset, "__len__")
                and len(extreme_archetype_mask_subset) > 0
            ):
                # Map extreme mask to UMAP subset
                for i, is_extreme in enumerate(extreme_archetype_mask_subset):
                    if i < len(umap_extreme_mask) and is_extreme:
                        umap_extreme_mask[i] = True

            df_umap_data_extreme = df_umap_data_only[umap_extreme_mask]
            df_umap_data_normal = df_umap_data_only[~umap_extreme_mask]

            # Plot normal data points with circles
            if len(df_umap_data_normal) > 0:
                for cell_type in df_umap_data_normal["cell_type"].unique():
                    if cell_type == "archetype":
                        continue
                    ct_mask = df_umap_data_normal["cell_type"] == cell_type
                    ct_data = df_umap_data_normal[ct_mask]
                    ct_indices_in_normal = np.where(ct_mask)[0]
                    # Map to original indices in umap_extreme_mask
                    normal_mask_indices = np.where(~umap_extreme_mask)[0]
                    ct_alphas = alphas[normal_mask_indices[ct_indices_in_normal]]
                    rgb = palette_dict[cell_type]
                    rgba = np.column_stack([np.tile(rgb, (len(ct_data), 1)), ct_alphas])
                    plt.scatter(
                        ct_data["UMAP1"],
                        ct_data["UMAP2"],
                        c=rgba,
                        s=50,
                        zorder=1,
                        marker="o",
                    )

            # Plot extreme data points with x markers
            if len(df_umap_data_extreme) > 0:
                for cell_type in df_umap_data_extreme["cell_type"].unique():
                    if cell_type == "archetype":
                        continue
                    ct_data = df_umap_data_extreme[df_umap_data_extreme["cell_type"] == cell_type]
                    plt.scatter(
                        ct_data["UMAP1"],
                        ct_data["UMAP2"],
                        c=[palette_dict[cell_type]],
                        marker="x",
                        s=150,
                        alpha=1.0,
                        zorder=3,
                    )
        else:
            # Plot data points normally
            for cell_type in df_umap_data_only["cell_type"].unique():
                if cell_type == "archetype":
                    continue
                ct_mask = df_umap_data_only["cell_type"] == cell_type
                ct_data = df_umap_data_only[ct_mask]
                ct_indices = np.where(ct_mask)[0]
                ct_alphas = alphas[ct_indices]
                rgb = palette_dict[cell_type]
                rgba = np.column_stack([np.tile(rgb, (len(ct_data), 1)), ct_alphas])
                plt.scatter(
                    ct_data["UMAP1"],
                    ct_data["UMAP2"],
                    c=rgba,
                    s=50,
                    zorder=1,
                    marker="o",
                )

        # Plot archetype points on top with X markers, colored by quality
        if archetype_quality_dict is not None:
            # Split archetypes by quality
            good_quality_archetypes = []
            poor_quality_archetypes = []

            for idx, row in df_umap_archetypes_only.iterrows():
                archetype_num = (
                    int(row["archetype_number"]) if not pd.isna(row["archetype_number"]) else None
                )
                if archetype_num is not None and archetype_num in archetype_quality_dict:
                    if archetype_quality_dict[archetype_num]:
                        good_quality_archetypes.append(idx)
                    else:
                        poor_quality_archetypes.append(idx)
                else:
                    # Default to poor quality if not found
                    poor_quality_archetypes.append(idx)

            # Plot good quality archetypes in black
            if good_quality_archetypes:
                good_df = df_umap_archetypes_only.iloc[
                    [
                        idx
                        for idx in range(len(df_umap_archetypes_only))
                        if df_umap_archetypes_only.index[idx] in good_quality_archetypes
                    ]
                ]
                plt.scatter(
                    good_df["UMAP1"],
                    good_df["UMAP2"],
                    marker="^",
                    s=800,
                    c="black",
                    edgecolors="white",
                    linewidth=0.5,
                    zorder=5,
                    label="Good Quality Archetypes",
                )

            # Plot poor quality archetypes in red
            if poor_quality_archetypes:
                poor_df = df_umap_archetypes_only.iloc[
                    [
                        idx
                        for idx in range(len(df_umap_archetypes_only))
                        if df_umap_archetypes_only.index[idx] in poor_quality_archetypes
                    ]
                ]
                plt.scatter(
                    poor_df["UMAP1"],
                    poor_df["UMAP2"],
                    marker="^",
                    s=800,
                    c="red",
                    edgecolors="white",
                    linewidth=0.5,
                    zorder=5,
                    label="Poor Quality Archetypes",
                )
        else:
            # Default behavior - all archetypes in black
            plt.scatter(
                df_umap_archetypes_only["UMAP1"],
                df_umap_archetypes_only["UMAP2"],
                marker="^",
                s=500,
                c="black",
                edgecolors="white",
                linewidth=0.5,
                zorder=5,
                label="Archetypes",
            )

        # Create custom legend with manual entries for cell types
        from matplotlib.lines import Line2D

        legend_elements = []
        all_labels = []

        # Create manual legend entries for cell types
        unique_cell_types = sorted(
            [ct for ct in df_umap_data_only["cell_type"].unique() if ct != "archetype"]
        )
        for cell_type in unique_cell_types:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=palette_dict[cell_type],
                    markersize=8,
                    label=cell_type,
                )
            )
            all_labels.append(cell_type)

        # Add quality-based archetype legends or single archetype legend
        if archetype_quality_dict is not None:
            # Add good quality archetype legend
            good_archetype_legend = Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="black",
                markeredgecolor="white",
                markersize=12,
                linewidth=2,
                label="Good Quality Archetypes",
            )
            legend_elements.append(good_archetype_legend)
            all_labels.append("Good Quality Archetypes")

            # Add poor quality archetype legend
            poor_archetype_legend = Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="red",
                markeredgecolor="white",
                markersize=12,
                linewidth=2,
                label="Poor Quality Archetypes",
            )
            legend_elements.append(poor_archetype_legend)
            all_labels.append("Poor Quality Archetypes")
        else:
            # Default single archetype legend
            archetype_legend = Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="black",
                markeredgecolor="white",
                markersize=12,
                linewidth=2,
                label="Archetypes",
            )
            legend_elements.append(archetype_legend)
            all_labels.append("Archetypes")

        if extreme_archetype_mask_subset is not None:
            extreme_legend = Line2D(
                [0],
                [0],
                marker="x",
                color="gray",
                linestyle="None",
                markersize=8,
                label="Extreme Archetypes",
            )
            legend_elements.append(extreme_legend)
            all_labels.append("Extreme Archetypes")

        plt.legend(
            legend_elements,
            all_labels,
            title="Cell Types",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        # Annotate archetype points with numbers
        archetype_points_umap = df_umap[df_umap["type"] == "archetype"]
        for _, row in archetype_points_umap.iterrows():
            if not pd.isna(row["archetype_number"]):
                plt.text(
                    row["UMAP1"],
                    row["UMAP2"],
                    str(int(row["archetype_number"])),
                    fontsize=12,
                    fontweight="bold",
                    color="red",
                    zorder=10,
                    ha="center",
                    va="center",
                )

        # Add lines from each data point to its matching archetype in UMAP plot
        df_umap_data = df_umap[df_umap["type"] == "data"].copy()
        df_umap_archetypes = df_umap[df_umap["type"] == "archetype"].copy()

        # Create a mapping from archetype_number to its UMAP coordinates
        df_umap_archetypes_clean = df_umap_archetypes.dropna(subset=["archetype_number"]).copy()
        df_umap_archetypes_clean["archetype_number"] = df_umap_archetypes_clean[
            "archetype_number"
        ].astype(int)
        archetype_coords_umap = df_umap_archetypes_clean.set_index("archetype_number")[
            ["UMAP1", "UMAP2"]
        ]

        # Now for each data point, draw a line to its corresponding archetype, limiting to max_lines
        max_lines = min(1000, len(df_umap_data))  # Limit number of lines to prevent clutter

        if lines_attached == "closest":
            # For each archetype, find the closest cells
            lines_per_archetype = max_lines // num_archetypes
            selected_indices = []

            for arch_num in archetype_coords_umap.index:
                arch_coord = archetype_coords_umap.loc[arch_num]
                # Filter data points that belong to this archetype
                arch_data = df_umap_data[
                    df_umap_data["data_point_archetype_index"] == arch_num
                ].copy()

                if len(arch_data) > 0:
                    # Calculate distances to this archetype
                    arch_data["distance"] = np.sqrt(
                        (arch_data["UMAP1"] - arch_coord["UMAP1"]) ** 2
                        + (arch_data["UMAP2"] - arch_coord["UMAP2"]) ** 2
                    )
                    # Select closest cells
                    n_select = min(lines_per_archetype, len(arch_data))
                    closest = arch_data.nsmallest(n_select, "distance")
                    selected_indices.extend(closest.index.tolist())

            df_umap_data_subset = df_umap_data.loc[selected_indices]
            print(
                f"Attaching lines to {len(df_umap_data_subset)} closest cells "
                f"({lines_per_archetype} per archetype)"
            )
        else:
            # Random selection (default behavior)
            if len(df_umap_data) > max_lines:
                line_indices = np.random.choice(len(df_umap_data), max_lines, replace=False)
                df_umap_data_subset = df_umap_data.iloc[line_indices]
            else:
                df_umap_data_subset = df_umap_data

        for idx, row in df_umap_data_subset.iterrows():
            if pd.isna(row["data_point_archetype_index"]):
                continue
            archetype_index = int(row["data_point_archetype_index"])
            data_point_coords = (row["UMAP1"], row["UMAP2"])
            archetype_point_coords = archetype_coords_umap.loc[archetype_index]
            plt.plot(
                [data_point_coords[0], archetype_point_coords["UMAP1"]],
                [data_point_coords[1], archetype_point_coords["UMAP2"]],
                color="black",
                linewidth=1.5,
                alpha=0.3,
            )
        plt.title(f"{modality} UMAP Scatter Plot with Archetypes Numbered")
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")

        # Set axis limits dynamically based on data range with 5% margin
        umap1_min, umap1_max = df_umap["UMAP1"].min(), df_umap["UMAP1"].max()
        umap2_min, umap2_max = df_umap["UMAP2"].min(), df_umap["UMAP2"].max()
        umap1_range = umap1_max - umap1_min
        umap2_range = umap2_max - umap2_min
        umap1_margin = umap1_range * 0.05
        umap2_margin = umap2_range * 0.05
        plt.xlim(umap1_min - umap1_margin, umap1_max + umap1_margin)
        plt.ylim(umap2_min - umap2_margin, umap2_max + umap2_margin)

        # Remove grid lines, tick marks, and tick labels
        plt.grid(False)
        ax = plt.gca()
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        plt.tight_layout()

        # Save to MLflow
        modality_safe = modality.replace(" ", "_").replace("/", "_")
        from arcadia.plotting.general import safe_mlflow_log_figure

        safe_mlflow_log_figure(plt.gcf(), f"archetypes/{modality_safe}_UMAP.pdf")

        plt.show()


# Main execution
if __name__ == "__main__":
    file_prefixes = ["preprocessed_adata_rna_", "preprocessed_adata_prot_"]
    folder = "processed_data"

    # Load the latest files
    latest_files = {prefix: get_latest_file(folder, prefix) for prefix in file_prefixes}
    adata_1_rna = sc.read(latest_files["preprocessed_adata_rna_"])
    adata_2_prot = sc.read(latest_files["preprocessed_adata_prot_"])

    # Plot initial UMAPs
    sc.pl.umap(adata_1_rna, color="cell_types", title="RNA UMAP with Cell Types")
    sc.pl.umap(
        adata_2_prot, color="cell_types", title="Protein UMAP with Cell Types", frameon=False
    )
    sc.pl.embedding(
        adata_2_prot,
        basis="spatial",
        color="cell_types",
        title="Spatial Distribution of Cell Types",
        frameon=False,
        use_raw=False,
        size=10,
    )
    # Get archetypes
    umap_coords = adata_1_rna.obsm["X_umap"]
    archetypes, cell_weights = get_archetypes(umap_coords)
    archetypes = archetypes.T
    # Generate all visualizations
    plot_umap_with_archetypes(adata_1_rna, archetypes, cell_weights)
    plot_umap_with_celltypes(adata_1_rna, archetypes)
    plot_cell_archetype_combination(umap_coords, archetypes, cell_weights)
    plot_archetype_vectors(umap_coords, archetypes, cell_weights)
    plot_archetype_vectors_with_arrows(umap_coords, archetypes, cell_weights, threshold=0.05)
    plot_archetypes_with_cell_arch_label(umap_coords, archetypes, cell_weights)


# ============================================================================
# Plotting functions moved from other modules
# ============================================================================


# Moved from cell_representations.py
def archetype_vs_latent_distances_plot(
    archetype_dis_tensor, latent_distances, threshold, use_subsample=True
):
    if use_subsample:
        subsample_indexes = torch.tensor(np.arange(min(300, archetype_dis_tensor.shape[0])))
    else:
        subsample_indexes = torch.tensor(np.arange(archetype_dis_tensor.shape[0]))
    archetype_dis_tensor_ = archetype_dis_tensor.detach().cpu()
    archetype_dis_tensor_ = torch.index_select(
        archetype_dis_tensor_, 0, subsample_indexes
    )  # Select rows
    archetype_dis_tensor_ = torch.index_select(
        archetype_dis_tensor_, 1, subsample_indexes
    )  # Select columns
    latent_distances_ = latent_distances.detach().cpu()
    latent_distances_ = torch.index_select(latent_distances_, 0, subsample_indexes)
    latent_distances_ = torch.index_select(latent_distances_, 1, subsample_indexes)
    latent_distances_ = latent_distances_.numpy()
    archetype_dis = archetype_dis_tensor_.numpy()
    all_distances = np.sort(archetype_dis.flatten())
    below_threshold_distances = np.sort(archetype_dis)[latent_distances_ < threshold].flatten()
    fig, ax1 = plt.subplots()
    counts_all, bins_all, _ = ax1.hist(
        all_distances, bins=100, alpha=0.5, label="All Distances", color="blue"
    )
    ax1.set_ylabel("Count (All Distances)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2 = ax1.twinx()
    counts_below, bins_below, _ = ax2.hist(
        below_threshold_distances,
        bins=bins_all,
        alpha=0.5,
        label="Below Threshold",
        color="green",
    )
    ax2.set_ylabel("Count (Below Threshold)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    plt.title(
        f"Number Below Threshold: {np.sum(latent_distances.detach().cpu().numpy() < threshold)}"
    )
    plt.savefig("archetype_vs_latent_distances_threshold.pdf")
    plt.show()
    plt.close()

    plt.figure()
    plt.subplot(1, 2, 1)
    sns.heatmap(latent_distances_)
    plt.title("latent_distances")
    plt.subplot(1, 2, 2)
    sns.heatmap(archetype_dis)
    plt.title("archetype_distances")
    plt.savefig("archetype_vs_latent_distances_heatmap.pdf")
    plt.show()
    plt.close()


# Moved from matching.py
def plot_archetypes_matching(data1, data2, rows=5, max_cols=20, plot_flag=True):
    """Plot archetype matching between two modalities.

    Parameters:
    -----------
    data1: DataFrame
        First modality archetype data
    data2: DataFrame
        Second modality archetype data
    rows: int
        Number of rows to plot
    max_cols: int
        Maximum number of columns to plot
    plot_flag: bool
        If False, return immediately without plotting
    """
    if not plot_flag:
        return
    # Ensure matrices aren't too wide for visualization
    if data1.shape[1] > max_cols or data2.shape[1] > max_cols:
        print(
            f"Warning: Large archetype matrices. Limiting to {max_cols} columns for visualization."
        )
        data1_plot = data1.iloc[:, :max_cols]
        data2_plot = data2.iloc[:, :max_cols]
    else:
        data1_plot = data1
        data2_plot = data2

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("RNA Archetype Weights\nAcross Cell Types")
    offset = 1
    rows = min(rows, len(data1_plot), len(data2_plot))
    for i in range(rows):
        y1 = data1_plot.iloc[i] + i * offset
        y2 = data2_plot.iloc[i] + i * offset
        plt.plot(y1, label=f"modality 1 archetype {i + 1}")
        plt.plot(y2, linestyle="--", label=f"modality 2 archetype {i + 1}")
    plt.xlabel("Columns")
    plt.ylabel("proportion of cell types accounted for an archetype")
    plt.title("Show that the archetypes are aligned by using")
    # rotate x labels
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title("Protein Archetype Weights\nAcross Cell Types")
    plt.suptitle("Cross-Modal Archetype Weight Distribution Analysis")
    offset = 1
    rows = min(rows, len(data1_plot), len(data2_plot))
    for i in range(rows):
        y1 = data1_plot.iloc[i] + i * offset
        y2 = data2_plot.iloc[i] + i * offset
        plt.plot(y1, label=f"modality 1 archetype {i + 1}")
        plt.plot(y2, linestyle="--", label=f"modality 2 archetype {i + 1}")
    plt.xlabel("Columns")
    plt.ylabel("proportion of cell types accounted for an archetype")
    plt.title("Show that the archetypes are aligned by using")
    # rotate x labels
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig("cross_modal_archetype_weights.pdf")
    plt.show()
    plt.close()


# Moved from metrics.py
def plot_archetype_umap(adata_combined, save_path=None):
    """Generate UMAP visualization of archetype vectors."""
    print("Generating UMAP visualization...")

    # Compute UMAP with cosine distance
    sc.pp.neighbors(adata_combined, use_rep="X", metric="cosine", n_neighbors=15)
    sc.tl.umap(adata_combined)

    # Plot by modality
    plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(2, 2, 1)
    sc.pl.umap(adata_combined, color="modality", ax=ax1, show=False, title="Modality")

    # Plot by cell type
    ax2 = plt.subplot(2, 2, 2)
    sc.pl.umap(adata_combined, color="cell_types", ax=ax2, show=False, title="Cell Types")

    # Plot by neighborhood
    ax3 = plt.subplot(2, 2, 3)
    sc.pl.umap(adata_combined, color="CN", ax=ax3, show=False, title="Cell Neighborhood")

    # Plot by major cell type if available
    if "major_cell_types" in adata_combined.obs:
        ax4 = plt.subplot(2, 2, 4)
        sc.pl.umap(
            adata_combined, color="major_cell_types", ax=ax4, show=False, title="Major Cell Types"
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("✓ UMAP visualization completed")


# Moved from metrics.py
def plot_archetype_heatmap(adata_rna, adata_prot, n_samples=50, save_path=None):
    """Plot heatmap of archetype vectors for a sample of cells."""
    print("Generating archetype vector heatmaps...")

    # Subsample for visualization
    if len(adata_rna) > n_samples:
        rna_sample = sc.pp.subsample(adata_rna, n_obs=n_samples, copy=True)
    else:
        rna_sample = adata_rna.copy()

    if len(adata_prot) > n_samples:
        prot_sample = sc.pp.subsample(adata_prot, n_obs=n_samples, copy=True)
    else:
        prot_sample = adata_prot.copy()

    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # RNA heatmap
    sns.heatmap(rna_sample.X, ax=axes[0], cmap="viridis", xticklabels=False, yticklabels=False)
    axes[0].set_title(f"RNA Archetype Vectors (n={len(rna_sample)})")

    # Protein heatmap
    sns.heatmap(prot_sample.X, ax=axes[1], cmap="viridis", xticklabels=False, yticklabels=False)
    axes[1].set_title(f"Protein Archetype Vectors (n={len(prot_sample)})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("✓ Archetype heatmaps generated")


# Moved from metrics.py
def plot_matching_accuracy_by_cell_type(adata_rna, adata_prot, prot_matches_in_rna, save_path=None):
    """Plot matching accuracy by cell type."""
    print("Generating cell type matching accuracy plot...")

    # Calculate matching by cell type
    matches = (
        adata_rna.obs["cell_types"].values[prot_matches_in_rna]
        == adata_prot.obs["cell_types"].values
    )

    # Group by cell type
    prot_cell_types = adata_prot.obs["cell_types"].values
    unique_cell_types = np.unique(prot_cell_types)

    accuracies = []
    for cell_type in unique_cell_types:
        cell_type_indices = prot_cell_types == cell_type
        type_matches = matches[cell_type_indices]
        accuracy = type_matches.sum() / len(type_matches) if len(type_matches) > 0 else 0
        accuracies.append((cell_type, accuracy, np.sum(cell_type_indices)))

    # Sort by accuracy
    accuracies.sort(key=lambda x: x[1], reverse=True)

    # Create DataFrame for plotting
    df = pd.DataFrame(accuracies, columns=["Cell Type", "Accuracy", "Count"])

    # Plot
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x="Cell Type", y="Accuracy", data=df, palette="viridis")

    # Add count labels
    for i, row in enumerate(df.itertuples()):
        ax.text(
            i, 0.05, f"n={row.Count}", ha="center", rotation=90, color="white", fontweight="bold"
        )

    plt.xticks(rotation=90)
    plt.title("Matching Accuracy by Cell Type")
    plt.ylim(0, 1.0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("✓ Cell type matching accuracy plot generated")


# Moved from metrics.py
def plot_distance_comparison(matching_distances, rand_matching_distances, save_path=None):
    """Plot comparison of actual vs random matching distances."""
    print("Generating distance comparison plot...")

    plt.figure(figsize=(12, 6))

    # Plot distance distributions
    plt.subplot(1, 2, 1)
    sns.histplot(matching_distances, kde=True, color="blue", label="Actual matches")
    sns.histplot(rand_matching_distances, kde=True, color="red", label="Random matches")
    plt.title("Distribution of Matching Distances")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.legend()

    # Plot cumulative distributions
    plt.subplot(1, 2, 2)
    sns.ecdfplot(matching_distances, label="Actual matches")
    sns.ecdfplot(rand_matching_distances, label="Random matches")
    plt.title("Cumulative Distribution of Distances")
    plt.xlabel("Distance")
    plt.ylabel("Proportion")
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("✓ Distance comparison plot generated")


# Moved from metrics.py
def create_tsne_visualization(rna_archetype, prot_archetype, save_path=None):
    """Create t-SNE visualization of RNA and protein archetype vectors."""
    print("Generating t-SNE visualization...")

    # Subsample if very large
    max_cells = 2000
    if len(rna_archetype) > max_cells:
        rna_sample = sc.pp.subsample(rna_archetype, n_obs=max_cells, copy=True)
    else:
        rna_sample = rna_archetype.copy()

    if len(prot_archetype) > max_cells:
        prot_sample = sc.pp.subsample(prot_archetype, n_obs=max_cells, copy=True)
    else:
        prot_sample = prot_archetype.copy()

    # Combine data
    combined_data = np.vstack([rna_sample.X, prot_sample.X])

    # Create labels for modality and cell type
    modality_labels = np.array(["RNA"] * len(rna_sample) + ["Protein"] * len(prot_sample))
    cell_type_labels = np.concatenate(
        [rna_sample.obs["cell_types"].values, prot_sample.obs["cell_types"].values]
    )

    # Run t-SNE with cosine distance
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, metric="cosine")
    embedding = tsne.fit_transform(combined_data)

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # By modality
    scatter1 = axes[0].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[0 if m == "RNA" else 1 for m in modality_labels],
        cmap="viridis",
        alpha=0.7,
        s=5,
    )
    axes[0].set_title("t-SNE by Modality (Cosine Distance)")
    legend1 = axes[0].legend(
        handles=scatter1.legend_elements()[0], labels=["RNA", "Protein"], loc="upper right"
    )
    axes[0].add_artist(legend1)

    # By cell type
    unique_types = np.unique(cell_type_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
    cell_type_colors = {t: colors[i] for i, t in enumerate(unique_types)}

    for cell_type in unique_types:
        mask = cell_type_labels == cell_type
        axes[1].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            color=cell_type_colors[cell_type],
            label=cell_type,
            alpha=0.7,
            s=5,
        )

    axes[1].set_title("t-SNE by Cell Type (Cosine Distance)")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("✓ t-SNE visualization completed")


# %%
def match_cells_using_archetypes(adata_rna, adata_prot):
    """Match cells between modalities using archetype vectors with cosine distance."""
    # Since we already converted the objects to have archetype vectors as X,
    # we can directly use their X matrices

    # Calculate pairwise distances using cosine distance
    print("Calculating pairwise cosine distances between archetype vectors...")
    latent_distances = batched_cosine_dist(adata_rna.X, adata_prot.X)

    # Find matches
    prot_matches_in_rna = np.argmin(latent_distances, axis=0)
    matching_distances = np.min(latent_distances, axis=0)

    # Generate random matches for comparison
    rand_indices = np.random.permutation(len(adata_rna))
    rand_latent_distances = latent_distances[rand_indices, :]
    rand_prot_matches_in_rna = np.argmin(rand_latent_distances, axis=0)
    rand_matching_distances = np.min(rand_latent_distances, axis=0)

    return {
        "prot_matches_in_rna": prot_matches_in_rna,
        "matching_distances": matching_distances,
        "rand_prot_matches_in_rna": rand_prot_matches_in_rna,
        "rand_matching_distances": rand_matching_distances,
    }


# %%
def calculate_post_training_metrics_on_archetypes(adata_rna, adata_prot, prot_matches_in_rna):
    """Calculate various metrics for model evaluation using archetype vectors."""
    # Calculate NMI scores
    nmi_cell_types_cn_rna = adjusted_mutual_info_score(
        adata_rna.obs["cell_types"], adata_rna.obs["CN"]
    )
    nmi_cell_types_cn_prot = adjusted_mutual_info_score(
        adata_prot.obs["cell_types"], adata_prot.obs["CN"]
    )
    nmi_cell_types_modalities = adjusted_mutual_info_score(
        adata_rna.obs["cell_types"].values[prot_matches_in_rna],
        adata_prot.obs["cell_types"].values,
    )

    # Calculate accuracy
    matches = (
        adata_rna.obs["cell_types"].values[prot_matches_in_rna]
        == adata_prot.obs["cell_types"].values
    )
    accuracy = matches.sum() / len(matches)

    # Calculate mixing score with cosine distance
    # Use X directly since it now contains the archetype vectors
    mixing_result = mixing_score(
        adata_rna.X,
        adata_prot.X,
        adata_rna,
        adata_prot,
        index_rna=range(len(adata_rna)),
        index_prot=range(len(adata_prot)),
        plot_flag=True,
        # metric='cosine'  # Use cosine distance for mixing score
    )

    return {
        "nmi_cell_types_cn_rna_archetypes": nmi_cell_types_cn_rna,
        "nmi_cell_types_cn_prot_archetypes": nmi_cell_types_cn_prot,
        "nmi_cell_types_modalities_archetypes": nmi_cell_types_modalities,
        "cell_type_matching_accuracy_archetypes": accuracy,
        "mixing_score_ilisi_archetypes": mixing_result["iLISI"],
        # "mixing_score_clisi_archetypes": mixing_result["cLISI"],
    }


# %%
def process_archetype_spaces(adata_rna, adata_prot):
    """Process archetype spaces from RNA and protein data."""
    print("Processing archetype spaces...")

    # Since we now have archetype vectors as X, we can use the objects directly
    rna_archetype = adata_rna.copy()
    prot_archetype = adata_prot.copy()

    # Combine for visualization
    combined_archetype = anndata.concat(
        [rna_archetype, prot_archetype],
        join="outer",
        label="modality",
        keys=["RNA", "Protein"],
    )

    print("✓ Archetype spaces processed")

    return rna_archetype, prot_archetype, combined_archetype


# %%
def calculate_metrics_for_archetypes(adata_rna, adata_prot, prefix="", subsample_size=None):
    """Calculate metrics using archetype vectors instead of latent space.

    Args:
        adata_rna: RNA AnnData object
        adata_prot: Protein AnnData object
        prefix: Prefix for metric names (e.g., "train_" or "val_")
        subsample_size: If not None, subsample the data to this size
    """
    print(f"Calculating {prefix}metrics on archetype vectors...")

    # Subsample if requested
    if subsample_size is not None:
        adata_rna = sc.pp.subsample(adata_rna, n_obs=subsample_size, copy=True)
        adata_prot = sc.pp.subsample(adata_prot, n_obs=subsample_size, copy=True)
        print(f"Subsampled to {subsample_size} cells")

    # Since we already have archetype vectors as X, we can directly use the objects
    rna_archetype_adata = adata_rna
    prot_archetype_adata = adata_prot

    # Calculate matching accuracy
    # Check if we can modify the metrics functions to use cosine
    # For now, we use the existing functions which likely use Euclidean
    from arcadia.training.metrics import compute_ari_f1, compute_silhouette_f1, matching_accuracy

    accuracy = matching_accuracy(rna_archetype_adata, prot_archetype_adata)
    print(f"✓ {prefix}matching accuracy calculated")

    # Calculate silhouette F1
    silhouette_f1 = compute_silhouette_f1(rna_archetype_adata, prot_archetype_adata)
    print(f"✓ {prefix}silhouette F1 calculated")

    # Calculate ARI F1
    combined_archetype = anndata.concat(
        [rna_archetype_adata, prot_archetype_adata],
        join="outer",
        label="modality",
        keys=["RNA", "Protein"],
    )

    # Skip PCA since archetype vectors are already low-dimensional (only ~7 dimensions)
    # sc.pp.pca(combined_archetype)

    # Use cosine distance directly on archetype vectors for neighbors
    sc.pp.neighbors(combined_archetype, n_neighbors=10, metric="cosine", use_rep="X")
    ari_f1 = compute_ari_f1(combined_archetype)
    print(f"✓ {prefix}ARI F1 calculated")

    return {
        f"{prefix}cell_type_matching_accuracy_archetypes": accuracy,
        f"{prefix}silhouette_f1_score_archetypes": silhouette_f1.mean(),
        f"{prefix}ari_f1_score_archetypes": ari_f1,
    }


# %%
# Main execution block
if __name__ == "__main__":
    # Create log directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    log_file = open(
        f"logs/archetype_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log", "w"
    )

    # Redirect stdout to both console and log file
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)

    print(f"Starting calculation of metrics on archetype vectors at {pd.Timestamp.now()}")

    # %%
    # Load data
    save_dir = Path("processed_data").absolute()
    from arcadia.training.utils import log_memory_usage

    log_memory_usage("Before loading data: ")

    # Find latest RNA and protein files
    from arcadia.data_utils.loading import get_latest_file

    rna_file = get_latest_file(save_dir, "subset_prepared_for_training_rna_")
    prot_file = get_latest_file(save_dir, "subset_prepared_for_training_prot_")
    if not rna_file or not prot_file:
        print("Error: Could not find trained data files.")
        sys.exit(1)

    print(f"Using RNA file: {os.path.basename(rna_file)}")
    print(f"Using Protein file: {os.path.basename(prot_file)}")

    # %%
    # Load data
    print("\nLoading data...")
    adata_rna = sc.read_h5ad(rna_file)
    adata_prot = sc.read_h5ad(prot_file)
    print("✓ Data loaded")
    from arcadia.training.utils import log_memory_usage

    log_memory_usage("After loading data: ")

    # Verify that archetype vectors exist
    if "archetype_vec" not in adata_rna.obsm or "archetype_vec" not in adata_prot.obsm:
        print("Error: Archetype vectors not found in data.")
        sys.exit(1)

    print(f"RNA dataset shape: {adata_rna.shape}")
    print(f"Protein dataset shape: {adata_prot.shape}")

    # %%
    # Convert to archetype-based AnnData objects
    print("\nConverting to archetype-based AnnData objects...")
    # Create new AnnData objects with archetype vectors as X
    adata_rna_arch = anndata.AnnData(X=adata_rna.obsm["archetype_vec"])
    adata_prot_arch = anndata.AnnData(X=adata_prot.obsm["archetype_vec"])

    # Copy observations and other attributes
    adata_rna_arch.obs = adata_rna.obs.copy()
    adata_prot_arch.obs = adata_prot.obs.copy()

    # Normalize RNA archetype vectors
    rna_scaler = MinMaxScaler()
    adata_rna_arch.X = rna_scaler.fit_transform(adata_rna_arch.X)

    # Normalize protein archetype vectors
    prot_scaler = MinMaxScaler()
    adata_prot_arch.X = prot_scaler.fit_transform(adata_prot_arch.X)

    # Copy uns, obsm (except archetype_vec), and obsp if they exist
    if hasattr(adata_rna, "uns"):
        adata_rna_arch.uns = adata_rna.uns.copy()
    if hasattr(adata_prot, "uns"):
        adata_prot_arch.uns = adata_prot.uns.copy()

    for key in adata_rna.obsm.keys():
        if key != "archetype_vec":
            adata_rna_arch.obsm[key] = adata_rna.obsm[key].copy()

    for key in adata_prot.obsm.keys():
        if key != "archetype_vec":
            adata_prot_arch.obsm[key] = adata_prot.obsm[key].copy()

    if hasattr(adata_rna, "obsp") and len(adata_rna.obsp) > 0:
        for key in adata_rna.obsp.keys():
            adata_rna_arch.obsp[key] = adata_rna.obsp[key].copy()

    if hasattr(adata_prot, "obsp") and len(adata_prot.obsp) > 0:
        for key in adata_prot.obsp.keys():
            adata_prot_arch.obsp[key] = adata_prot.obsp[key].copy()

    # Replace original adata with archetype-based ones
    adata_rna = adata_rna_arch
    adata_prot = adata_prot_arch

    print(f"New RNA archetype dataset shape: {adata_rna.shape}")
    print(f"New Protein archetype dataset shape: {adata_prot.shape}")
    print("✓ Converted to archetype-based AnnData objects")
    from arcadia.training.utils import log_memory_usage

    log_memory_usage("After archetype conversion: ")

    # %%
    # Normalize archetype vectors to [0,1] range
    print("\nNormalizing archetype vectors to [0,1] range...")

    # Get the dimensions of both datasets
    n_rna_dims = adata_rna.X.shape[1]
    n_prot_dims = adata_prot.X.shape[1]

    # Check if dimensions match
    if n_rna_dims != n_prot_dims:
        print(
            f"Warning: RNA and protein archetype vectors have different dimensions ({n_rna_dims} vs {n_prot_dims})"
        )

    # Verify normalization worked
    print(
        f"RNA min values: {adata_rna.X.min(axis=0).min():.4f}, max: {adata_rna.X.max(axis=0).max():.4f}"
    )
    print(
        f"Protein min values: {adata_prot.X.min(axis=0).min():.4f}, max: {adata_prot.X.max(axis=0).max():.4f}"
    )
    print("✓ Archetype vectors normalized")

    # Create a heatmap to visualize the normalized archetype vectors
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sample_size = min(50, len(adata_rna), len(adata_prot))

    # RNA heatmap
    sns.heatmap(
        adata_rna.X[:sample_size],
        ax=axes[0],
        cmap="viridis",
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
    )
    axes[0].set_title(f"Normalized RNA Archetype Vectors (n={sample_size})")

    # Protein heatmap
    sns.heatmap(
        adata_prot.X[:sample_size],
        ax=axes[1],
        cmap="viridis",
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
    )
    axes[1].set_title(f"Normalized Protein Archetype Vectors (n={sample_size})")

    plt.tight_layout()
    plt.show()

    # %%
    # Subsample for faster execution
    max_cells = 5000
    print(f"\nSubsampling data to max {max_cells} cells per modality for faster execution...")
    if len(adata_rna) > max_cells:
        adata_rna = sc.pp.subsample(adata_rna, n_obs=max_cells, copy=True)
    if len(adata_prot) > max_cells:
        adata_prot = sc.pp.subsample(adata_prot, n_obs=max_cells, copy=True)
    print(f"Subsampled RNA dataset shape: {adata_rna.shape}")
    print(f"Subsampled protein dataset shape: {adata_prot.shape}")
    from arcadia.training.utils import log_memory_usage

    log_memory_usage("After subsampling: ")

    # %%
    # Process archetype spaces
    rna_archetype, prot_archetype, combined_archetype = process_archetype_spaces(
        adata_rna, adata_prot
    )

    # %%
    # Create visualization of archetype vectors (display only)
    sc.pl.pca(combined_archetype, color="cell_types", show=False)
    plt.show()
    sc.pl.pca(rna_archetype, color="cell_types", show=False)
    plt.show()
    sc.pp.pca(prot_archetype)
    sc.pl.pca(prot_archetype, color="cell_types", show=False)
    plt.show()

    # Plot archetype heatmaps
    plot_archetype_heatmap(rna_archetype, prot_archetype, n_samples=50)

    # %%
    # Create t-SNE visualization
    create_tsne_visualization(rna_archetype, prot_archetype)

    # %%
    # Match cells and calculate distances using archetype vectors
    matching_results = match_cells_using_archetypes(adata_rna, adata_prot)

    # %%
    # Plot distance comparison
    plot_distance_comparison(
        matching_results["matching_distances"], matching_results["rand_matching_distances"]
    )

    # %%
    # Calculate metrics
    metrics = calculate_post_training_metrics_on_archetypes(
        adata_rna, adata_prot, matching_results["prot_matches_in_rna"]
    )

    # %%
    # Plot matching accuracy by cell type
    plot_matching_accuracy_by_cell_type(
        adata_rna, adata_prot, matching_results["prot_matches_in_rna"]
    )

    # %%
    # Calculate additional metrics
    additional_metrics = calculate_metrics_for_archetypes(adata_rna, adata_prot)
    metrics.update(additional_metrics)

    # %%
    # Generate UMAP visualization
    plot_archetype_umap(combined_archetype)

    # %%
    # Print results
    print("\nMetrics on Archetype Vectors:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Create a summary visualization of metrics
    plt.figure(figsize=(12, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.barh(metric_names, metric_values, color="skyblue")
    plt.xlabel("Value")
    plt.title("Archetype Vector Metrics Summary")
    plt.tight_layout()
    plt.show()

    # %%
    # Calculate metrics with MLflow if available
    try:
        mlflow.log_metrics({k: round(v, 4) for k, v in metrics.items()})
        print("✓ Metrics logged to MLflow (no plot artifacts saved)")
    except Exception as e:
        print(f"Warning: Could not log to MLflow: {e}")

    # %%
    # Clean up: restore original stdout and close log file
    print(f"\nArchetype metrics calculation completed at {pd.Timestamp.now()}")
    sys.stdout = original_stdout
    log_file.close()


# %%

# evaluate_distance_metrics is imported from arcadia.archetypes.metrics via __init__.py
# No need for duplicate import here


# Moved from generation.py - plotting functions for archetype proportions
def plot_archetype_proportions_before_after_matching(
    batch_proportions_before: Dict[str, List[pd.DataFrame]],
    batch_proportions_after: Dict[str, List[pd.DataFrame]] = None,
    k_matched_results: Dict[int, Dict] = None,
    batch_archetypes: Dict[str, Dict] = None,
    optimal_k: int = None,
    modality_names: List[str] = None,
    cell_type_colors: Dict[str, str] = None,
    plot_flag: bool = True,
    save_plots: bool = False,
    output_dir: str = "plots/",
    before_label: str = "Before Matching",
    after_label: str = "After Matching",
    title_prefix: str = "Archetype Cell Type Proportions",
) -> None:
    """
    Create bar graphs showing archetype proportions before and after matching.

    Shows subplots for each batch and cross-modality comparison with:
    - Each bar group represents one archetype
    - Bars within group show proportion of each cell type
    - Colors represent different cell types
    - Before/after matching comparison

    Args:
        batch_proportions_before: Dictionary of batch proportions before matching
        batch_proportions_after: Dictionary of batch proportions after matching (optional)
        k_matched_results: Results from archetype matching (optional)
        optimal_k: The optimal k value to plot (if None, uses first available)
        modality_names: Names of modalities for titles
        cell_type_colors: Dictionary mapping cell types to colors
        plot_flag: Whether to create plots
        save_plots: Whether to save plots to files
        output_dir: Directory to save plots
    """
    if not plot_flag:
        return

    # Import locally to avoid circular dependency
    from arcadia.archetypes.generation import get_cell_type_colors

    print("\n=== Creating archetype proportion bar plots ===")

    # Determine which proportions to use
    plot_both = batch_proportions_after is not None and batch_proportions_before is not None

    # Get batches and determine k value to plot
    if plot_both:
        batches = list(batch_proportions_before.keys())
        # Ensure after has same batches
        batches_after = list(batch_proportions_after.keys())
        batches = sorted(list(set(batches).intersection(set(batches_after))))
    elif batch_proportions_after is not None:
        batches = list(batch_proportions_after.keys())
    else:
        batches = list(batch_proportions_before.keys())

    if optimal_k is None:
        # Use first k value available
        first_batch = batches[0]
        if k_matched_results is not None:
            available_k = list(k_matched_results.keys())
            if available_k:
                optimal_k = available_k[0]
        elif batch_archetypes is not None and first_batch in batch_archetypes:
            if len(batch_archetypes[first_batch]["k_values"]) > 0:
                optimal_k = batch_archetypes[first_batch]["k_values"][0]
        else:
            # Fallback: assume k values start from 1
            optimal_k = 1

    print(f"Plotting proportions for k={optimal_k}")

    # Create subplot layout
    n_batches = len(batches)
    if modality_names is None:
        modality_names = batches
    elif len(modality_names) != len(batches):
        modality_names = batches

    # Determine subplot grid - if plotting both, need 2 columns
    # Use consistent subplot dimensions so bars have same width
    # Calculate single-subplot width first, then scale for combined plot
    if n_batches <= 2:
        single_subplot_width = 6
        single_fig_height = 6
    elif n_batches <= 4:
        single_subplot_width = 6  # 12 / 2 = 6 per subplot
        single_fig_height = 5  # 10 / 2 = 5 per subplot
    else:
        single_subplot_width = 5  # 15 / 3 = 5 per subplot
        single_fig_height = 5

    if plot_both:
        ncols = 2
        nrows = int(np.ceil(n_batches / 1))
        # Double the width so each subplot has same width as single plot
        figsize = (single_subplot_width * 2, single_fig_height * nrows)
    else:
        if n_batches <= 2:
            nrows, ncols = 1, n_batches
            figsize = (single_subplot_width * n_batches, single_fig_height)
        elif n_batches <= 4:
            nrows, ncols = 2, 2
            figsize = (single_subplot_width * 2, single_fig_height * 2)
        else:
            nrows = int(np.ceil(n_batches / 3))
            ncols = 3
            figsize = (single_subplot_width * 3, single_fig_height * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Get all cell types for consistent coloring
    all_cell_types = set()
    for batch in batches:
        # Find k_idx for this batch
        k_idx = None
        if optimal_k is None:
            # Auto-select first available k
            if batch_archetypes is not None and batch in batch_archetypes:
                if len(batch_archetypes[batch]["k_values"]) > 0:
                    k_idx = 0
            elif batch in batch_proportions_before and len(batch_proportions_before[batch]) > 0:
                k_idx = 0
        elif batch_archetypes is not None and batch in batch_archetypes:
            if optimal_k in batch_archetypes[batch]["k_values"]:
                k_idx = batch_archetypes[batch]["k_values"].index(optimal_k)
        else:
            # Fallback: assume 0-based indexing
            k_idx = optimal_k - 1 if optimal_k > 0 else 0

        if (
            k_idx is not None
            and batch in batch_proportions_before
            and k_idx < len(batch_proportions_before[batch])
        ):
            prop_df = batch_proportions_before[batch][k_idx]
            if isinstance(prop_df, pd.DataFrame):
                all_cell_types.update(prop_df.columns)
        if (
            plot_both
            and batch in batch_proportions_after
            and k_idx is not None
            and k_idx < len(batch_proportions_after[batch])
        ):
            prop_df = batch_proportions_after[batch][k_idx]
            if isinstance(prop_df, pd.DataFrame):
                all_cell_types.update(prop_df.columns)

    all_cell_types = sorted(list(all_cell_types))

    # Generate colors if not provided
    if cell_type_colors is None:
        cell_type_colors = get_cell_type_colors(all_cell_types)

    def plot_proportions(ax, prop_df, title_suffix, modality_name):
        """Helper function to plot normalized stacked bars"""
        if prop_df is None or prop_df.empty:
            ax.text(
                0.5,
                0.5,
                f"k={optimal_k} not available\nfor {modality_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{modality_name} - {title_suffix}")
            return

        n_archetypes = prop_df.shape[0]

        # Normalize proportions so each archetype sums to 1 (all bars same height)
        prop_df_normalized = prop_df.div(prop_df.sum(axis=1), axis=0)

        # Prepare data for stacked bar plot
        x_pos = np.arange(n_archetypes)
        bar_width = 0.8

        # Prepare data for stacking
        bottom = np.zeros(n_archetypes)

        # Plot stacked bars for each cell type
        for j, cell_type in enumerate(all_cell_types):
            if cell_type in prop_df_normalized.columns:
                values = prop_df_normalized[cell_type].values
                color = cell_type_colors.get(cell_type, f"C{j}")

                ax.bar(
                    x_pos,
                    values,
                    bar_width,
                    bottom=bottom,
                    label=cell_type,
                    color=color,
                    alpha=0.8,
                )
                bottom += values

        # Customize plot
        ax.set_xlabel("Archetype Index")
        ax.set_ylabel("Cell Type Proportion (Normalized)")
        ax.set_title(f"{modality_name} - {title_suffix}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"A{i}" for i in range(n_archetypes)])
        ax.set_ylim(0, 1)
        # Set consistent x-axis limits to ensure same bar spacing
        ax.set_xlim(-0.5, n_archetypes - 0.5)

    # Plot each batch
    for i, batch in enumerate(batches):
        modality_name = modality_names[i] if i < len(modality_names) else batch

        # Find k_idx for this batch
        k_idx = None
        if optimal_k is None:
            # Auto-select first available k
            if batch_archetypes is not None and batch in batch_archetypes:
                if len(batch_archetypes[batch]["k_values"]) > 0:
                    k_idx = 0
            elif batch in batch_proportions_before and len(batch_proportions_before[batch]) > 0:
                k_idx = 0
        elif batch_archetypes is not None and batch in batch_archetypes:
            if optimal_k in batch_archetypes[batch]["k_values"]:
                k_idx = batch_archetypes[batch]["k_values"].index(optimal_k)
        else:
            # Fallback: assume 0-based indexing
            k_idx = optimal_k - 1 if optimal_k > 0 else 0

        if plot_both:
            # Plot before and after side by side
            ax_before = axes[i * 2]
            ax_after = axes[i * 2 + 1]

            # Before plot
            prop_df_before = None
            if (
                k_idx is not None
                and batch in batch_proportions_before
                and k_idx < len(batch_proportions_before[batch])
            ):
                prop_df_before = batch_proportions_before[batch][k_idx]
            plot_proportions(ax_before, prop_df_before, before_label, modality_name)

            # After plot
            prop_df_after = None
            if batch in batch_proportions_after and len(batch_proportions_after[batch]) > 0:
                if len(batch_proportions_after[batch]) == 1:
                    prop_df_after = batch_proportions_after[batch][0]
                elif (
                    k_idx is not None
                    and k_idx < len(batch_proportions_after[batch])
                    and batch_proportions_after[batch][k_idx] is not None
                ):
                    prop_df_after = batch_proportions_after[batch][k_idx]
            plot_proportions(ax_after, prop_df_after, after_label, modality_name)

            # Add legend only to first subplot
            if i == 0:
                ax_before.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            # Plot only one (before or after)
            ax = axes[i]
            if batch_proportions_after is not None:
                prop_df = None
                if batch in batch_proportions_after and len(batch_proportions_after[batch]) > 0:
                    if len(batch_proportions_after[batch]) == 1:
                        prop_df = batch_proportions_after[batch][0]
                    elif k_idx is not None and k_idx < len(batch_proportions_after[batch]):
                        prop_df = batch_proportions_after[batch][k_idx]
                plot_proportions(ax, prop_df, after_label, modality_name)
            else:
                prop_df = None
                if (
                    k_idx is not None
                    and batch in batch_proportions_before
                    and k_idx < len(batch_proportions_before[batch])
                ):
                    prop_df = batch_proportions_before[batch][k_idx]
                plot_proportions(ax, prop_df, before_label, modality_name)

            # Add legend only to first subplot
            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Hide extra subplots
    if plot_both:
        n_used = n_batches * 2
    else:
        n_used = n_batches
    for i in range(n_used, len(axes)):
        axes[i].set_visible(False)

    # Determine k value for title
    k_for_title = optimal_k
    if k_for_title is None and batch_archetypes is not None:
        # Get k from first batch
        first_batch = batches[0]
        if first_batch in batch_archetypes and len(batch_archetypes[first_batch]["k_values"]) > 0:
            k_for_title = batch_archetypes[first_batch]["k_values"][0]

    if plot_both:
        title_suffix = f"{before_label} vs {after_label}"
    else:
        title_suffix = after_label if batch_proportions_after is not None else before_label
    plt.suptitle(f"{title_prefix} {title_suffix} (k={k_for_title})", fontsize=14, y=0.98)
    plt.tight_layout()

    if save_plots:
        print("Saving plots is disabled in the pipeline; displaying only.")

    plt.show()


def plot_cross_modal_archetype_comparison(
    global_proportions_mod1: Dict[int, pd.DataFrame],
    global_proportions_mod2: Dict[int, pd.DataFrame],
    optimal_k: int,
    modality_names: List[str] = ["RNA", "Protein"],
    cell_type_colors: Dict[str, str] = None,
    cross_modal_orders: Dict[int, np.ndarray] = None,
    archetype_quality_dict: Dict[int, bool] = None,
    similarity_scores: np.ndarray = None,
    quality_threshold: float = None,
    plot_flag: bool = True,
    save_plots: bool = False,
    output_dir: str = "plots/",
) -> None:
    """
    Create bar plots comparing archetype proportions across modalities.

    Args:
        global_proportions_mod1: Global proportions for first modality
        global_proportions_mod2: Global proportions for second modality
        optimal_k: Optimal k value to plot
        modality_names: Names of the two modalities
        cell_type_colors: Dictionary mapping cell types to colors
        cross_modal_orders: Cross-modal ordering information
        archetype_quality_dict: Dictionary mapping archetype indices to quality (True=good, False=poor)
        similarity_scores: Array of quality scores (Jensen-Shannon distances between cell type proportions)
        quality_threshold: Threshold for determining poor quality (scores above this are poor quality)
        plot_flag: Whether to create plots
        save_plots: Whether to save plots
        output_dir: Directory to save plots
    """
    if not plot_flag:
        return

    # Import locally to avoid circular dependency
    from arcadia.archetypes.generation import get_cell_type_colors

    if optimal_k not in global_proportions_mod1 or optimal_k not in global_proportions_mod2:
        print(f"Warning: k={optimal_k} not available in both modalities for cross-modal comparison")
        return

    print(f"\n=== Creating cross-modal archetype comparison for k={optimal_k} ===")

    # Get proportions
    prop1 = global_proportions_mod1[optimal_k]
    prop2 = global_proportions_mod2[optimal_k]

    # Apply cross-modal ordering if available
    if cross_modal_orders is not None and optimal_k in cross_modal_orders:
        order = cross_modal_orders[optimal_k]
        if 1 in order:  # Check if we have ordering for modality 2
            reorder_indices = order[1]
            prop2 = prop2.iloc[reorder_indices]
            print(f"Applied cross-modal reordering {reorder_indices} to {modality_names[1]}")

    # Get all cell types
    all_cell_types = sorted(list(set(prop1.columns) | set(prop2.columns)))

    # Generate colors if not provided
    if cell_type_colors is None:
        cell_type_colors = get_cell_type_colors(all_cell_types)

    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Plot function for each modality
    def plot_modality_proportions(ax, prop_df, title, modality_idx, scores=None):
        n_archetypes = prop_df.shape[0]
        x_pos = np.arange(n_archetypes)
        bar_width = 0.8

        # Normalize proportions to ensure they sum to 1 for each archetype
        prop_df_normalized = prop_df.div(prop_df.sum(axis=1), axis=0)

        # Create stacked bars
        bottom = np.zeros(n_archetypes)

        # Determine which archetypes are poor quality (for entire bar hatching)
        poor_quality_mask = np.zeros(n_archetypes, dtype=bool)
        if archetype_quality_dict is not None:
            for i in range(n_archetypes):
                if i in archetype_quality_dict and not archetype_quality_dict[i]:
                    poor_quality_mask[i] = True

        for j, cell_type in enumerate(all_cell_types):
            if cell_type in prop_df_normalized.columns:
                values = prop_df_normalized[cell_type].values
                color = cell_type_colors.get(cell_type, f"C{j}")

                # Only add label for first modality
                label = cell_type if modality_idx == 0 else ""

                # Plot bars - apply hatching per archetype if poor quality
                for i in range(n_archetypes):
                    hatch_pattern = "///" if poor_quality_mask[i] else None
                    edgecolor = "black" if poor_quality_mask[i] else None
                    linewidth = 1 if poor_quality_mask[i] else 0

                    # Only add label for first archetype
                    current_label = label if i == 0 else ""

                    ax.bar(
                        x_pos[i],
                        values[i],
                        bar_width,
                        bottom=bottom[i],
                        label=current_label,
                        color=color,
                        alpha=0.8,
                        hatch=hatch_pattern,
                        edgecolor=edgecolor,
                        linewidth=linewidth,
                    )

                # Update bottom for next cell type
                bottom += values

        ax.set_xlabel("Archetype Index")
        if modality_idx == 0:
            ax.set_ylabel("Cell Type Proportion")
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_ylim(0, 1)

        # Create x-axis labels with similarity scores if available
        if scores is not None and len(scores) >= n_archetypes:
            # FIXED: Ensure scores is a numpy array before checking ndim
            if not isinstance(scores, np.ndarray):
                scores = np.array(scores)

            # Use diagonal values (self-similarity) or minimum distances for each archetype
            if scores.ndim == 2:
                # If it's a similarity matrix, get the minimum distance for each archetype
                archetype_similarities = np.min(scores[:n_archetypes], axis=1)
            else:
                # If it's already a 1D array, use it directly
                archetype_similarities = scores[:n_archetypes]

            labels = [f"A{i}\n({sim:.3f})" for i, sim in enumerate(archetype_similarities)]
        else:
            labels = [f"A{i}" for i in range(n_archetypes)]

        ax.set_xticklabels(labels)

    # Plot both modalities
    plot_modality_proportions(ax1, prop1, f"{modality_names[0]} Archetypes", 0, similarity_scores)
    plot_modality_proportions(ax2, prop2, f"{modality_names[1]} Archetypes", 1, similarity_scores)

    # Add shared legend
    legend_handles, legend_labels = ax1.get_legend_handles_labels()

    # Add legend entry for poor quality indicator if we have quality information
    if archetype_quality_dict is not None:
        # Check if we actually have any poor quality archetypes
        has_poor_quality = any(not quality for quality in archetype_quality_dict.values())
        if has_poor_quality:
            import matplotlib.patches as mpatches

            poor_quality_patch = mpatches.Patch(
                facecolor="gray",
                hatch="///",
                edgecolor="black",
                alpha=0.8,
                label="Poor Quality Archetype",
            )
            legend_handles.append(poor_quality_patch)
            legend_labels.append("Poor Quality Archetype")

    ax1.legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), loc="upper left")

    # Create informative title explaining the scores
    title = f"Cross-Modal Archetype Comparison (k={optimal_k})"
    if quality_threshold is not None:
        subtitle = f"Values in parentheses: Jensen-Shannon distances between cell type proportions\n(Lower is better; threshold={quality_threshold:.1f} for quality assessment)"
    else:
        subtitle = "Values in parentheses: Cross-modal similarity scores (lower is better)"

    plt.suptitle(title, fontsize=14, y=0.98)
    plt.figtext(0.5, 0.02, subtitle, ha="center", fontsize=10, style="italic")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for subtitle

    if save_plots:
        print("Saving plots is disabled in the pipeline; displaying only.")

    plt.show()


def visualize_archetype_proportions_analysis(
    adata_rna: AnnData,
    adata_prot: AnnData,
    plot_flag: bool = True,
) -> None:
    """
    Visualize archetype proportions analysis including the archetype matching plot.

    This function creates the visualization that shows pairs of lines (one from each modality)
    where each pair represents the proportion of cell types for a given archetype.

    Args:
        adata_rna: RNA AnnData object with archetype information
        adata_prot: Protein AnnData object with archetype information
        plot_flag: Whether to generate plots
    """
    if not plot_flag:
        return

    # Import locally to avoid circular dependency
    from arcadia.archetypes.matching import reorder_rows_to_maximize_diagonal

    print("\n=== Visualizing Archetype Proportions Analysis ===")

    # Check if we have the necessary archetype proportion data
    rna_prop = adata_rna.uns.get("best_archetype_rna_prop", None)
    prot_prop = adata_prot.uns.get("best_archetype_prot_prop", None)

    if rna_prop is not None and prot_prop is not None:
        print("Using pre-computed archetype proportions")

        # Plot archetype matching (the main plot requested)
        plot_archetypes_matching(rna_prop, prot_prop, rows=8)

        # Plot archetype weights with diagonal maximization
        _, row_order = reorder_rows_to_maximize_diagonal(rna_prop)
        from arcadia.plotting.archetypes import plot_archetype_weights

        plot_archetype_weights(rna_prop, prot_prop, row_order)

        # Show overlap of cell types proportions in archetypes

        plt.figure(figsize=(10, 6))
        prot_prop.idxmax(axis=0).plot(kind="bar", color="red", hatch="\\", label="Protein")
        rna_prop.idxmax(axis=0).plot(kind="bar", alpha=0.5, hatch="/", label="RNA")
        plt.title("Overlap of cell types proportions in archetypes")
        plt.legend()
        plt.xlabel("Major Cell Types")
        plt.ylabel("Proportion")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("archetype_cell_type_overlap.pdf")
        plt.show()
        plt.close()

    else:
        print("Computing archetype proportions from current archetype vectors...")

        # Compute proportions from current archetype vectors
        # This is a simplified version - the full implementation would compute
        # proportions properly from the batch-aware archetype structure

        # Get unified archetype vectors
        rna_archetypes = adata_rna.obsm.get("archetype_vec", None)
        prot_archetypes = adata_prot.obsm.get("archetype_vec", None)

        if rna_archetypes is None or prot_archetypes is None:
            print("Warning: No archetype vectors found, skipping proportion analysis")
            return

        # Simple proportion computation based on dominant archetype per cell type
        rna_cell_types = (
            adata_rna.obs["cell_types"]
            if "cell_types" in adata_rna.obs
            else adata_rna.obs.get("major_cell_types", None)
        )
        prot_cell_types = (
            adata_prot.obs["cell_types"]
            if "cell_types" in adata_prot.obs
            else adata_prot.obs.get("major_cell_types", None)
        )

        if rna_cell_types is None or prot_cell_types is None:
            print("Warning: No cell type information found, skipping proportion analysis")
            return

        print(
            "Simplified proportion analysis completed (full analysis requires batch-specific proportions)"
        )
