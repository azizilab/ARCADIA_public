"""Data preprocessing utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
from anndata import AnnData
from pandas import CategoricalDtype
from scipy.sparse import find
from scipy.stats import median_abs_deviation


def balance_datasets(
    adata_rna: AnnData, adata_prot: AnnData, plot_flag: bool = False, flexibility: float = 0.1
):
    """
    Balance the datasets based on the cell type proportions per batch.

    Each batch (regardless of modality) is treated independently.
    Finds the smallest/limiting batch and balances all batches to match its proportions.

    Args:
        adata_prot (AnnData): Protein assay AnnData object containing cell type annotations
            in .obs["cell_types"] and batch information in .obs["batch"].
        adata_rna (AnnData): RNA assay AnnData object containing cell type annotations
            in .obs["cell_types"] and batch information in .obs["batch"].
        plot_flag (bool): If True, displays stacked bar plots showing normalized cell type
            proportions for each batch (all bars have same height).
        flexibility (float): Flexibility percentage (0.0 to 1.0). If limiting batch has X cells
            of a type, reference batch can have X * (1 + flexibility). Default 0.1 (10%).
    """
    if "batch" not in adata_rna.obs.columns or "batch" not in adata_prot.obs.columns:
        raise ValueError("Both adata objects must have 'batch' column in obs")

    all_batches_rna = sorted(adata_rna.obs["batch"].unique())
    all_batches_prot = sorted(adata_prot.obs["batch"].unique())
    all_batches = sorted(set(all_batches_rna).union(set(all_batches_prot)))

    print(f"Found batches in RNA: {all_batches_rna}")
    print(f"Found batches in Protein: {all_batches_prot}")
    print(
        f"Processing {len(all_batches_rna)} RNA batches and {len(all_batches_prot)} Protein batches independently"
    )

    # Collect all batches with their data - process RNA and Protein separately
    batch_data = {}

    # Process RNA batches separately
    for batch in all_batches_rna:
        batch_rna = adata_rna[adata_rna.obs["batch"] == batch]
        if batch_rna.n_obs == 0:
            print(f"Warning: RNA batch {batch} has no cells, skipping")
            continue

        counts = batch_rna.obs["cell_types"].value_counts()
        batch_key = f"rna_{batch}"  # Unique key for RNA batch
        batch_data[batch_key] = {
            "modality": "rna",
            "original_batch": batch,
            "adata": batch_rna,
            "counts": pd.Series(counts),
            "total": counts.sum(),
        }

    # Process Protein batches separately
    for batch in all_batches_prot:
        batch_prot = adata_prot[adata_prot.obs["batch"] == batch]
        if batch_prot.n_obs == 0:
            print(f"Warning: Protein batch {batch} has no cells, skipping")
            continue

        counts = batch_prot.obs["cell_types"].value_counts()
        batch_key = f"prot_{batch}"  # Unique key for Protein batch
        batch_data[batch_key] = {
            "modality": "protein",
            "original_batch": batch,
            "adata": batch_prot,
            "counts": pd.Series(counts),
            "total": counts.sum(),
        }

    if not batch_data:
        raise ValueError("No valid batches found for balancing")

    # Find smallest batch (reference) and calculate its proportions
    smallest_batch = min(batch_data.keys(), key=lambda x: batch_data[x]["total"])
    ref_data = batch_data[smallest_batch]
    ref_props = ref_data["counts"] / ref_data["total"]

    print(f"\n{'='*60}")
    print(f"Reference batch: {ref_data['original_batch']} ({ref_data['modality']})")
    print(f"Total cells: {ref_data['total']}")
    print(f"Proportions:\n{ref_props}")
    print(f"{'='*60}")

    # Calculate limiting cell type counts across all batches (excluding reference)
    # Then allow reference batch to have flexibility% more than the limiting count
    limiting_counts_per_type = {}
    for cell_type in ref_props.index:
        counts_per_batch = []
        for batch, data in batch_data.items():
            if batch == smallest_batch:
                continue
            batch_counts = data["counts"]
            count = batch_counts.get(cell_type, 0) if cell_type in batch_counts.index else 0
            counts_per_batch.append(count)

        if counts_per_batch:
            limiting_counts_per_type[cell_type] = min(counts_per_batch)
        else:
            limiting_counts_per_type[cell_type] = 0

    # Calculate achievable proportions for non-reference batches
    # Then allow reference batch to have flexibility% more than those proportions
    non_ref_achievable_props = []

    for batch, data in batch_data.items():
        if batch == smallest_batch:
            continue

        batch_counts = data["counts"]
        batch_total = data["total"]

        achievable_props = {}
        for cell_type in ref_props.index:
            available_count = (
                batch_counts.get(cell_type, 0) if cell_type in batch_counts.index else 0
            )
            max_achievable_prop = available_count / batch_total if batch_total > 0 else 0
            achievable_props[cell_type] = max_achievable_prop

        non_ref_achievable_props.append(pd.Series(achievable_props))

    # Find minimum proportions across non-reference batches
    if non_ref_achievable_props:
        min_non_ref_props = pd.Series(
            {
                cell_type: min(
                    prop_series.get(cell_type, 0) for prop_series in non_ref_achievable_props
                )
                for cell_type in ref_props.index
            }
        )
    else:
        min_non_ref_props = ref_props

    # Reference batch can have limiting_count * (1 + flexibility) cells of each type
    # This allows reference to keep more cells and achieve higher proportions
    ref_data = batch_data[smallest_batch]
    ref_batch_total = ref_data["total"]
    ref_batch_counts = ref_data["counts"]
    ref_achievable_props = {}

    for cell_type in ref_props.index:
        # Limiting count (minimum across non-reference batches)
        limiting_count = limiting_counts_per_type.get(cell_type, 0)
        # Reference can have flexibility% more cells than limiting batch
        flexible_count = int(limiting_count * (1 + flexibility))
        # But can't exceed what reference batch actually has
        usable_count = min(flexible_count, ref_batch_counts.get(cell_type, 0))
        # Calculate what proportion this represents for reference batch
        ref_achievable_props[cell_type] = (
            usable_count / ref_batch_total if ref_batch_total > 0 else 0
        )

    ref_achievable_props = pd.Series(ref_achievable_props)

    # Final target proportions:
    # Use min_non_ref_props as base (what non-reference batches can achieve)
    # Reference batch will sample to match these proportions but can use flexible counts
    target_props = min_non_ref_props.copy()
    # Renormalize to sum to 1
    target_props = target_props / target_props.sum()

    print(f"\nFlexibility: {flexibility * 100:.1f}%")
    print(
        f"Limiting cell type counts (min across non-reference batches): {limiting_counts_per_type}"
    )
    if non_ref_achievable_props:
        print(f"\nMinimum achievable proportions (non-reference batches):")
        print(min_non_ref_props)
        print(f"\nReference batch flexible cell counts (limiting_count * {1 + flexibility:.2f}):")
        ref_flexible_counts = {
            cell_type: int(limiting_counts_per_type.get(cell_type, 0) * (1 + flexibility))
            for cell_type in ref_props.index
        }
        print(ref_flexible_counts)
        print(f"\nReference batch flexible proportions (based on flexible counts):")
        print(ref_achievable_props)
    print(f"\nOriginal reference proportions:")
    print(ref_props)
    print(
        f"\nFinal target proportions (based on non-reference minimum, reference can use flexible counts):"
    )
    print(target_props)

    # Sample cells from each batch to match achievable proportions
    selected_indices_rna = []
    selected_indices_prot = []

    for batch_key, data in batch_data.items():
        original_batch = data["original_batch"]
        modality_label = data["modality"].upper()
        print(f"\n{'='*60}")
        print(f"Processing batch: {original_batch} ({modality_label})")
        print(f"{'='*60}")

        batch_adata = data["adata"]
        batch_total = data["total"]

        # Sample each cell type to match target proportions
        indices = []
        for cell_type, target_prop in target_props.items():
            if batch_key == smallest_batch:
                # Reference batch: can use up to flexible_count (limiting_count * (1 + flexibility))
                limiting_count = limiting_counts_per_type.get(cell_type, 0)
                flexible_count = int(limiting_count * (1 + flexibility))
                # Calculate target based on proportion
                target_count_by_prop = int(target_prop * batch_total)
                # Reference batch can use the maximum of (target_by_prop, flexible_count)
                # This allows flexibility to keep more cells
                max_allowed = max(target_count_by_prop, flexible_count)
            else:
                # Other batches: use target proportion (limited by available)
                max_allowed = int(target_prop * batch_total)

            cell_mask = batch_adata.obs["cell_types"] == cell_type
            available_indices = np.where(cell_mask)[0]

            if len(available_indices) == 0:
                continue

            # Sample up to max_allowed (or all available if less)
            n_sample = min(max_allowed, len(available_indices))
            sampled = (
                np.random.choice(available_indices, n_sample, replace=False)
                if n_sample < len(available_indices)
                else available_indices
            )
            indices.extend(batch_adata.obs.index[sampled].tolist())

        if batch_key == smallest_batch:
            print(
                f"Batch {original_batch} ({modality_label}): Selected {len(indices)}/{data['total']} cells (reference batch)"
            )
        else:
            print(
                f"Batch {original_batch} ({modality_label}): Selected {len(indices)}/{data['total']} cells"
            )
        print(f"  Target proportions: {target_props.to_dict()}")

        # Add to appropriate modality list
        if data["modality"] == "rna":
            selected_indices_rna.extend(indices)
        else:
            selected_indices_prot.extend(indices)

    # Create balanced datasets
    adata_rna_balanced = (
        adata_rna[selected_indices_rna].copy() if selected_indices_rna else adata_rna[[]].copy()
    )
    adata_prot_balanced = (
        adata_prot[selected_indices_prot].copy() if selected_indices_prot else adata_prot[[]].copy()
    )

    # Print summary
    print(f"\n{'='*60}")
    print("Final balancing summary:")
    print(f"{'='*60}")
    print(f"Final RNA dataset size: {adata_rna_balanced.n_obs}")
    print(f"Final Protein dataset size: {adata_prot_balanced.n_obs}")

    print("\nFinal proportions:")
    print(f"RNA proportions:\n{adata_rna_balanced.obs['cell_types'].value_counts(normalize=True)}")
    print(
        f"Protein proportions:\n{adata_prot_balanced.obs['cell_types'].value_counts(normalize=True)}"
    )

    print("\nFinal proportions per batch:")
    for batch in sorted(all_batches):
        batch_rna = adata_rna_balanced[adata_rna_balanced.obs["batch"] == batch]
        batch_prot = adata_prot_balanced[adata_prot_balanced.obs["batch"] == batch]
        if batch_rna.n_obs > 0:
            print(f"\nBatch {batch} (RNA):")
            print(f"  RNA proportions:\n{batch_rna.obs['cell_types'].value_counts(normalize=True)}")
        if batch_prot.n_obs > 0:
            print(f"\nBatch {batch} (Protein):")
            print(
                f"  Protein proportions:\n{batch_prot.obs['cell_types'].value_counts(normalize=True)}"
            )

    # Plot stacked bar charts for each batch if requested
    if plot_flag:
        _plot_batch_proportions(adata_rna_balanced, adata_prot_balanced, all_batches)

    return adata_rna_balanced, adata_prot_balanced


def _plot_batch_proportions(adata_rna: AnnData, adata_prot: AnnData, all_batches: list):
    """
    Plot normalized stacked bar charts showing cell type proportions for each batch.

    All bars have the same height (normalized to 1.0) to show relative proportions.
    """
    from arcadia.archetypes.generation import get_cell_type_colors

    # Collect all cell types across all batches
    all_cell_types = set()
    for batch in all_batches:
        batch_rna = (
            adata_rna[adata_rna.obs["batch"] == batch]
            if batch in adata_rna.obs["batch"].values
            else None
        )
        batch_prot = (
            adata_prot[adata_prot.obs["batch"] == batch]
            if batch in adata_prot.obs["batch"].values
            else None
        )
        if batch_rna is not None and batch_rna.n_obs > 0:
            all_cell_types.update(batch_rna.obs["cell_types"].unique())
        if batch_prot is not None and batch_prot.n_obs > 0:
            all_cell_types.update(batch_prot.obs["cell_types"].unique())

    all_cell_types = sorted(list(all_cell_types))
    cell_type_colors = get_cell_type_colors(all_cell_types)

    # Prepare data for plotting
    n_batches = len(
        [
            b
            for b in all_batches
            if b in adata_rna.obs["batch"].values or b in adata_prot.obs["batch"].values
        ]
    )
    if n_batches == 0:
        return

    # Determine subplot layout
    ncols = min(3, n_batches)
    nrows = int(np.ceil(n_batches / ncols))
    figsize = (6 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    plot_idx = 0
    for batch in sorted(all_batches):
        batch_rna = (
            adata_rna[adata_rna.obs["batch"] == batch]
            if batch in adata_rna.obs["batch"].values
            else None
        )
        batch_prot = (
            adata_prot[adata_prot.obs["batch"] == batch]
            if batch in adata_prot.obs["batch"].values
            else None
        )

        # Get proportions for this batch (combine RNA and Protein if both exist)
        total_cells = 0
        if batch_rna is not None and batch_rna.n_obs > 0:
            rna_props = batch_rna.obs["cell_types"].value_counts(normalize=True)
            batch_props = rna_props
            modality_label = "RNA"
            total_cells = batch_rna.n_obs
        elif batch_prot is not None and batch_prot.n_obs > 0:
            prot_props = batch_prot.obs["cell_types"].value_counts(normalize=True)
            batch_props = prot_props
            modality_label = "Protein"
            total_cells = batch_prot.n_obs
        else:
            continue

        # Normalize to ensure sum is 1.0
        batch_props = batch_props / batch_props.sum()

        # Create stacked bar plot
        ax = axes[plot_idx]
        x_pos = np.arange(1)  # Single bar per batch
        bar_width = 0.8
        bottom = np.zeros(1)

        for cell_type in all_cell_types:
            prop_value = batch_props.get(cell_type, 0.0)
            color = cell_type_colors.get(cell_type, f"C{all_cell_types.index(cell_type)}")

            ax.bar(
                x_pos,
                prop_value,
                bar_width,
                bottom=bottom,
                label=cell_type,
                color=color,
                alpha=0.8,
            )
            bottom += prop_value

        ax.set_xlabel("")
        ax.set_ylabel("Cell Type Proportion (Normalized)")
        ax.set_title(f"{batch} ({modality_label}) - {total_cells} cells")
        ax.set_xticks([])
        ax.set_ylim(0, 1)

        # Add legend only to first subplot
        if plot_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Cell Type Proportions per Batch (Normalized)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()


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


def filter_unwanted_cell_types(adata: sc.AnnData, unwanted_types: list = None) -> sc.AnnData:
    """Filter out specified unwanted cell types from dataset

    Parameters:
    -----------
    adata : sc.AnnData
        AnnData object to filter
    unwanted_types : list, optional
        List of cell types to remove. Default is ["tumor", "dead", "nk cells"]

    Returns:
    --------
    sc.AnnData
        Filtered AnnData object
    """
    if unwanted_types is None:
        unwanted_types = ["tumor", "dead", "nk cells"]

    if "cell_types" not in adata.obs.columns:
        print("Warning: 'cell_types' column not found in adata.obs")
        return adata

    initial_cells = adata.shape[0]

    # Filter out unwanted cell types that exist in the dataset
    existing_unwanted = [ct for ct in unwanted_types if ct in adata.obs["cell_types"].values]

    if existing_unwanted:
        print(f"Filtering out {existing_unwanted} cell types...")
        adata = adata[~adata.obs["cell_types"].isin(existing_unwanted)].copy()
        print(f"Cells: {initial_cells} -> {adata.shape[0]} after filtering")
    else:
        print("No unwanted cell types found to filter")

    return adata


def qc_metrics(adata: sc.AnnData, plot_flag=True) -> None:
    """Compute and visualize QC metrics"""
    # use a random subset of the data
    adata = adata[np.random.choice(adata.n_obs, size=min(adata.n_obs, 2000), replace=False), :]

    # Use 'counts' layer if available, otherwise use current X
    layer_to_use = "counts" if "counts" in adata.layers.keys() else None
    sc.pp.calculate_qc_metrics(adata, layer=layer_to_use, percent_top=(10, 20, 30), inplace=True)
    if not plot_flag:
        return

    # Visualize initial QC
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Use appropriate count columns based on what's available
    count_col = "nCount_CODEX" if "nCount_CODEX" in adata.obs.columns else "total_counts"
    feature_col = "nFeature_CODEX" if "nFeature_CODEX" in adata.obs.columns else "n_genes_by_counts"

    if count_col in adata.obs.columns:
        sc.pl.violin(adata, count_col, show=False, ax=axs[0])
        axs[0].set_title("Total Counts")
    else:
        axs[0].text(0.5, 0.5, "Count data not available", ha="center", va="center")
        axs[0].set_title("Total Counts (N/A)")

    if feature_col in adata.obs.columns:
        sc.pl.violin(adata, feature_col, show=False, ax=axs[1])
        axs[1].set_title("Features Detected")
    else:
        axs[1].text(0.5, 0.5, "Feature data not available", ha="center", va="center")
        axs[1].set_title("Features Detected (N/A)")

    plt.tight_layout()
    plt.show()


def balanced_subsample_by_cell_type(adata, n_obs, cell_type_col="cell_types"):
    """
    Subsample cells ensuring good representation of each cell type.

    For each cell type:
    - If cell type has less than half of expected proportion, take all available cells
    - Otherwise, sample proportionally
    - Ensure total selection equals target n_obs

    Args:
        adata: AnnData object
        n_obs: Target number of cells to sample
        cell_type_col: Column name containing cell type information

    Returns:
        Subsampled AnnData object
    """
    if n_obs >= adata.shape[0]:
        return adata.copy()

    cell_counts = adata.obs[cell_type_col].value_counts()
    n_cell_types = len(cell_counts)
    expected_per_type = n_obs / n_cell_types

    # First pass: identify underrepresented cell types and take all their cells
    selected_positions = []
    remaining_cells_needed = n_obs
    remaining_cell_types = []

    for cell_type, count in cell_counts.items():
        if count < expected_per_type * 0.5:  # Less than half of expected proportion
            # Take all cells of this type
            cell_mask = adata.obs[cell_type_col] == cell_type
            cell_positions = np.where(cell_mask)[0]
            selected_positions.extend(cell_positions.tolist())
            remaining_cells_needed -= count
            print(f"Taking all {count} cells of underrepresented type: {cell_type}")
        else:
            remaining_cell_types.append((cell_type, count))

    # Second pass: sample proportionally from remaining cell types
    if remaining_cell_types and remaining_cells_needed > 0:
        total_remaining_cells = sum(count for _, count in remaining_cell_types)

        for cell_type, count in remaining_cell_types:
            # Calculate proportional sample size for this cell type
            proportion = count / total_remaining_cells
            n_to_sample = int(remaining_cells_needed * proportion)

            # Ensure we don't sample more than available
            n_to_sample = min(n_to_sample, count)

            if n_to_sample > 0:
                cell_mask = adata.obs[cell_type_col] == cell_type
                cell_positions = np.where(cell_mask)[0]
                sampled_positions = np.random.choice(
                    cell_positions, size=n_to_sample, replace=False
                )
                selected_positions.extend(sampled_positions.tolist())
                print(f"Sampling {n_to_sample}/{count} cells of type: {cell_type}")

    # Adjust if we have slight over/under sampling due to rounding
    if len(selected_positions) != n_obs:
        diff = n_obs - len(selected_positions)
        if diff > 0:
            # Need more cells - randomly sample from remaining
            all_positions = set(range(adata.shape[0]))
            available_positions = list(all_positions - set(selected_positions))
            if available_positions:
                additional = np.random.choice(
                    available_positions, size=min(diff, len(available_positions)), replace=False
                )
                selected_positions.extend(additional.tolist())
        elif diff < 0:
            # Have too many cells - randomly remove some
            selected_positions = list(
                np.random.choice(selected_positions, size=n_obs, replace=False)
            )

    print(f"Final selection: {len(selected_positions)} cells")
    return adata[selected_positions].copy()


def mad_outlier_removal(adata: sc.AnnData) -> sc.AnnData:
    """Cell type-aware outlier detection"""
    print("\nIdentifying outliers...")
    adata.obs["outlier"] = False

    # Convert cell_types to categorical if it's not already
    if not isinstance(adata.obs["cell_types"].dtype, CategoricalDtype):
        adata.obs["cell_types"] = adata.obs["cell_types"].astype("category")

    # Get unique cell types (works for both categorical and non-categorical)
    cell_types = (
        adata.obs["cell_types"].cat.categories
        if hasattr(adata.obs["cell_types"], "cat")
        else adata.obs["cell_types"].unique()
    )

    for cell_type in cell_types:
        ct_mask = adata.obs["cell_types"] == cell_type
        if ct_mask.sum() == 0:  # Skip if no cells of this type
            continue

        # Use total_counts if nCount_CODEX doesn't exist
        count_col = "total_counts"
        adata.obs[count_col] = adata.X.sum(axis=1)

        counts = adata.obs.loc[ct_mask, count_col]

        if len(counts) < 3:  # Skip if too few cells for meaningful statistics
            continue

        med = np.median(counts)
        mad = median_abs_deviation(counts, scale="normal")
        MAD_THRESHOLD = 2
        lower = med - MAD_THRESHOLD * mad
        upper = med + MAD_THRESHOLD * mad

        adata.obs.loc[ct_mask, "outlier"] = (counts < lower) | (counts > upper)
        print(f"Removing {adata.obs.loc[ct_mask, 'outlier'].sum()} outlier cells for {cell_type}")

    print(f"Removing {adata.obs.outlier.sum()} outlier cells in total")
    return adata[~adata.obs.outlier].copy()


def log1p_rna(adata):
    """
    Apply log1p transformation to RNA data and track in metadata.

    Parameters:
    -----------
    adata : AnnData
        AnnData object containing RNA data

    Returns:
    --------
    adata : AnnData
        AnnData object with log1p transformed data in .X
    """
    # Initialize pipeline_metadata if it doesn't exist
    if "pipeline_metadata" not in adata.uns:
        adata.uns["pipeline_metadata"] = {}
    if "log1p" not in adata.uns["pipeline_metadata"]:
        adata.uns["pipeline_metadata"]["log1p"] = True
    if adata.uns["pipeline_metadata"]["log1p"] is False:
        print("Log1p transformation already applied")
        return adata
    # Apply log1p transformation
    sc.pp.log1p(adata)

    # Track in metadata
    adata.uns["pipeline_metadata"]["log1p"] = True
    print("Applied log1p transformation to RNA data")

    return adata


def z_normalize_codex(adata, apply_log1p=True):
    """
    https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2021.727626/full

    Apply Z normalization to CODEX protein data.

    This implements the recommended Z normalization approach for CODEX data:
    - Normalizes each marker intensity separately across all cells
    - Most consistent performance across different noise types
    - Handles both low signal intensity and high background effectively
    - Best for both rare and common cell types

    Based on comprehensive benchmarking, Z normalization is the most effective
    approach for CODEX data, outperforming log-double-z, min-max, and arcsinh methods.

    Parameters:
    -----------
    adata : AnnData
        AnnData object containing CODEX protein data

    Returns:
    --------
    adata : AnnData
        AnnData object with Z-normalized data in .X and .layers['z_normalized']
    """
    # Store raw data
    adata.raw = adata.copy()

    # Get data from layers['counts'] or use X directly

    X = adata.X.copy()

    if hasattr(X, "toarray"):
        X = X.toarray()  # Convert sparse to dense if needed
    # Apply log1p transformation first (if requested)
    if apply_log1p:
        X = np.log1p(X)
        adata.layers["log1p"] = X.copy()
        print("Applied log1p transformation")

    # Z normalize each marker separately across all cells
    # This is the key step: normalize each marker intensity separately
    means = np.mean(X, axis=0)  # Mean for each marker across all cells
    stds = np.std(X, axis=0)  # Std for each marker across all cells
    stds[stds == 0] = 1  # Avoid division by zero for constant markers

    X_z = (X - means) / stds

    # Store the normalized data
    adata.X = X_z
    adata.layers["z_normalized"] = X_z.copy()
    if apply_log1p:
        adata.uns["pipeline_metadata"]["log1p"] = True
    else:
        adata.uns["pipeline_metadata"]["log1p"] = False
    print("Z normalization complete")
    return adata


def analyze_and_visualize(
    input_adata: sc.AnnData, modality: str = "", plot_flag=True, save_flag=False
) -> None:
    """Dimensionality reduction and visualization"""

    print("\nRunning dimensionality reduction...")
    # take a random subset of the data
    adata = input_adata[
        np.random.choice(input_adata.n_obs, size=min(input_adata.n_obs, 2000), replace=False), :
    ]
    sc.pp.pca(adata, n_comps=15)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=10)
    sc.tl.umap(adata)
    if not plot_flag:
        return
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sc.pl.umap(
        adata, color="cell_types", title=f"{modality} Cell Types", ax=ax1, show=False, frameon=False
    )

    # Use appropriate count column for second plot
    count_col = "nCount_CODEX" if "nCount_CODEX" in adata.obs.columns else "total_counts"
    if count_col in adata.obs.columns:
        sc.pl.umap(
            adata, color=count_col, title=f"{modality} Counts", ax=ax2, show=False, frameon=False
        )
    else:
        ax2.text(
            0.5, 0.5, "Count data not available", ha="center", va="center", transform=ax2.transAxes
        )
        ax2.set_title("Counts (N/A)")
    plt.tight_layout()
    if save_flag:
        fig.savefig(f"umap_{modality}.pdf", dpi=300, bbox_inches="tight")
    elif plot_flag:
        plt.show()
    else:
        None


def spatial_analysis(adata: sc.AnnData) -> None:
    """Optional spatial analysis"""
    try:
        import squidpy as sq

        print("\nPerforming spatial analysis...")

        sq.gr.spatial_neighbors(adata, coord_type="generic")
        sq.pl.spatial_scatter(
            sc.pp.subsample(adata.copy(), n_obs=min(3000, adata.n_obs), copy=True),
            color="cell_types",
            shape=None,
        )

    except ImportError:
        print("Squidpy not installed - skipping spatial analysis")


def preprocess_rna_initial_steps(adata_rna, min_genes=200, min_cells=3, plot_flag=False):
    """
    Preprocess RNA data initial steps: QC and filtering ONLY (NO normalization/log transform)

    This function should be applied to each batch separately before merging.
    Data remains as counts/pseudo-counts for proper batch correction.
    """
    print(f"Starting initial preprocessing for batch with {adata_rna.n_obs} cells...")

    # Step 1: Basic filtering (without QC metrics)
    print("Step 1: Basic filtering...")
    sc.pp.filter_cells(adata_rna, min_genes=min_genes)
    sc.pp.filter_genes(adata_rna, min_cells=min_cells)

    # Step 2-3: Store raw counts and calculate QC metrics
    adata_rna = store_raw_counts_and_qc_metrics(adata_rna)

    # Step 3b: Filter low-quality cells (now that QC metrics are available)
    adata_rna = filter_low_quality_cells(
        adata_rna, max_pct_mt=20, min_genes=200, min_counts=1000, plot_flag=plot_flag
    )
    sc.pp.highly_variable_genes(
        adata_rna,
        n_top_genes=min(8000, adata_rna.shape[1]),
        flavor="seurat_v3",
        batch_key=(
            "batch"
            if "batch" in adata_rna.obs and len(adata_rna.obs["batch"].unique()) > 1
            else None
        ),
    )

    print(
        f"Initial preprocessing completed for batch with {adata_rna.n_obs} cells (data kept as counts)"
    )
    return adata_rna


def preprocess_rna_final_steps(
    adata_rna,
    n_top_genes=2000,
    plot_flag=False,
    batch_correction_method="scANVI",
    gene_likelihood_dist=None,
):
    """
    Complete RNA preprocessing after merging: HVG → batch_correction
    NO normalization, NO log transformation, NO scaling - data kept as counts!

    This function should be applied to the merged dataset.
    """
    print("Starting final preprocessing steps on merged dataset...")

    # Step 5: Feature selection (counts only)
    adata_rna = feature_selection_counts_only(adata_rna, n_top_genes, plot_flag=plot_flag)

    # Step 6: Batch correction on raw counts
    if adata_rna.uns["pipeline_metadata"].get("applied_batch_correction", False):
        adata_rna = apply_batch_correction_pipeline(
            adata_rna,
            plot_flag=plot_flag,
            batch_correction_method=batch_correction_method,
            gene_likelihood_dist=gene_likelihood_dist,
        )
        adata_rna.uns["pipeline_metadata"]["applied_batch_correction"] = True

    print("Final preprocessing completed successfully (data kept as counts)!")
    return adata_rna


def order_cells_by_type(adata_rna, adata_prot):
    """Order cells by major and minor cell types"""
    print("\nOrdering cells by major and minor cell types...")
    if "minor_cell_types" in adata_rna.obs.columns:
        cell_type_key = "minor_cell_types"
        new_order_rna = adata_rna.obs.sort_values(by=["major_cell_types", cell_type_key]).index
        new_order_prot = adata_prot.obs.sort_values(by=["major_cell_types", cell_type_key]).index

    else:
        cell_type_key = "cell_types"
        new_order_rna = adata_rna.obs.sort_values(by=[cell_type_key]).index
        new_order_prot = adata_prot.obs.sort_values(by=[cell_type_key]).index

    adata_rna = adata_rna[new_order_rna]
    adata_prot = adata_prot[new_order_prot]
    return adata_rna, adata_prot


def compute_pca_and_umap(adata_rna, adata_prot):
    """Compute PCA and UMAP for visualization"""
    sc.pp.pca(adata_rna)
    sc.pp.pca(adata_prot)
    sc.pp.neighbors(adata_rna, key_added="original_neighbors")
    sc.tl.umap(adata_rna, neighbors_key="original_neighbors")
    adata_rna.obsm["X_original_umap"] = adata_rna.obsm["X_umap"]
    sc.pp.neighbors(adata_prot, key_added="original_neighbors")
    sc.tl.umap(adata_prot, neighbors_key="original_neighbors")
    adata_prot.obsm["X_original_umap"] = adata_prot.obsm["X_umap"]
    return adata_rna, adata_prot


def apply_zero_mask(adata_rna: AnnData) -> AnnData:
    # if there is a mismatch between index of adata_rna.obs and adata_rna.var, we need to reindex the zero_mask
    zero_mask = adata_rna.obsm["zero_mask"]
    # Calculate statistics before applying zero mask
    currently_nonzero = adata_rna.X != 0
    currently_zero = adata_rna.X == 0
    zero_mask = adata_rna.obsm["zero_mask"]
    # make col names upper case
    zero_mask.columns = zero_mask.columns.str.upper()
    zero_mask = zero_mask.loc[:, ~zero_mask.columns.duplicated()]
    zero_mask = zero_mask.loc[adata_rna.obs.index, adata_rna.var.index.str.upper()]
    # Values that will be set to zero but are currently not zero
    will_become_zero = currently_nonzero & zero_mask.values
    will_become_zero_count = will_become_zero.sum()
    will_become_zero_prop = will_become_zero_count / adata_rna.X.size

    # Values that are already zero and will stay zero (thanks to mask)
    already_zero_in_mask = currently_zero & zero_mask.values
    already_zero_in_mask_count = already_zero_in_mask.sum()
    already_zero_in_mask_prop = already_zero_in_mask_count / adata_rna.X.size

    # Zero values that are NOT in the zero mask
    zero_not_in_mask = currently_zero & (~zero_mask.values)
    zero_not_in_mask_count = zero_not_in_mask.sum()
    zero_not_in_mask_prop = zero_not_in_mask_count / adata_rna.X.size

    print(
        f"Values that will become zero (currently non-zero): {will_become_zero_count} ({float(will_become_zero_prop):.4f})"
    )
    print(
        f"Values already zero and in mask: {already_zero_in_mask_count} ({already_zero_in_mask_prop:.4f})"
    )
    print(f"Zero values NOT in zero mask: {zero_not_in_mask_count} ({zero_not_in_mask_prop:.4f})")

    # Check if all obs indices are in zero_mask obs index, if not apply str.upper() to obs
    if not adata_rna.obs.index.isin(zero_mask.index).all():
        adata_rna.obs.index = adata_rna.obs.index.str.upper()
        zero_mask.index = zero_mask.index.str.upper()

    # Check if all var indices are in zero_mask columns, if not apply str.upper() to var
    if not adata_rna.var.index.isin(zero_mask.columns).all():
        adata_rna.var.index = adata_rna.var.index.str.upper()
        zero_mask.columns = zero_mask.columns.str.upper()

    # drop duplicate columns
    zero_mask = zero_mask.loc[:, zero_mask.columns]

    zero_mask = zero_mask.loc[adata_rna.obs.index, adata_rna.var.index]
    adata_rna.obsm["zero_mask"] = zero_mask
    # all values below the max mock value of the cells and genes we know to be zeros should be set to zero
    max_mock_zero_value = np.max(adata_rna.X[zero_mask.values])
    adata_rna.X[adata_rna.X <= max_mock_zero_value] = 0
    assert adata_rna.X.max() != 0
    return adata_rna


def store_raw_counts_and_qc_metrics(adata_rna):
    """
    Step 2-3: Store raw counts and calculate QC metrics
    """
    print("Step 2: Storing raw counts...")
    if "counts" not in adata_rna.layers:
        adata_rna.layers["counts"] = adata_rna.X.copy()

    print("Step 3: Calculating QC metrics...")
    # Auto-detect species for mitochondrial genes
    if adata_rna.var_names.str.startswith("MT-").any():  # Human
        adata_rna.var["mt"] = adata_rna.var_names.str.startswith("MT-")
        mt_prefix = "MT-"
    else:  # Mouse
        adata_rna.var["mt"] = adata_rna.var_names.str.startswith("mt-")
        mt_prefix = "mt-"

    # Add other gene types
    adata_rna.var["ribo"] = adata_rna.var_names.str.startswith(("RPS", "RPL"))
    adata_rna.var["hb"] = adata_rna.var_names.str.contains("^HB[^(P)]")

    # Calculate basic QC metrics manually to avoid issues
    # Calculate total counts and n_genes_by_counts
    if sp.issparse(adata_rna.X):
        adata_rna.obs["total_counts"] = np.array(adata_rna.X.sum(axis=1)).flatten()
        adata_rna.obs["n_genes_by_counts"] = np.array((adata_rna.X > 0).sum(axis=1)).flatten()
    else:
        adata_rna.obs["total_counts"] = adata_rna.X.sum(axis=1)
        adata_rna.obs["n_genes_by_counts"] = (adata_rna.X > 0).sum(axis=1)

    # Calculate gene-level QC
    if sp.issparse(adata_rna.X):
        adata_rna.var["total_counts"] = np.array(adata_rna.X.sum(axis=0)).flatten()
        adata_rna.var["n_cells_by_counts"] = np.array((adata_rna.X > 0).sum(axis=0)).flatten()
    else:
        adata_rna.var["total_counts"] = adata_rna.X.sum(axis=0)
        adata_rna.var["n_cells_by_counts"] = (adata_rna.X > 0).sum(axis=0)

    # Calculate pct_counts_mt using raw counts if available
    if "counts" in adata_rna.layers:
        mt_mask = adata_rna.var["mt"].values  # Convert to numpy array
        mt_counts = adata_rna.layers["counts"][:, mt_mask].sum(axis=1)
        if hasattr(mt_counts, "A1"):  # sparse matrix
            mt_counts = mt_counts.A1
        total_counts = adata_rna.layers["counts"].sum(axis=1)
        if hasattr(total_counts, "A1"):  # sparse matrix
            total_counts = total_counts.A1
        adata_rna.obs["pct_counts_mt"] = (mt_counts / total_counts) * 100

    print(f"Using {mt_prefix} as mitochondrial gene prefix")
    print(f"Found {adata_rna.var['mt'].sum()} mitochondrial genes")
    print(f"Found {adata_rna.var['ribo'].sum()} ribosomal genes")

    return adata_rna


def plot_rna_during_preprocessing(adata_rna, plot_flag=False, title=""):
    """Plot RNA data during preprocessing steps"""
    if plot_flag:
        # Use a subsample (max 2k cells) for visualization to avoid plotting the whole dataset
        n_plot = min(2000, adata_rna.n_obs)
        adata_plot = sc.pp.subsample(adata_rna.copy(), n_obs=n_plot, copy=True).copy()
        sc.pp.log1p(adata_plot)
        sc.pp.pca(adata_plot, copy=False)
        sc.pp.neighbors(adata_plot, copy=False)
        sc.tl.umap(adata_plot)
        if "batch" in adata_plot.obs.columns:
            sc.pl.umap(adata_plot, color="batch", title=f"RNA: Dataset Source {title} log1p")
        if "cell_types" in adata_plot.obs.columns:
            sc.pl.umap(adata_plot, color="cell_types", title=f"RNA: Cell Types {title} log1p")
        sample_data = adata_rna.X.copy()
        if sp.issparse(sample_data):
            sample_data = sample_data.toarray()

        # Sample 2k values total from all X
        flat_data = sample_data.flatten()
        n_samples = min(3000, len(flat_data))
        sample_indices = np.random.choice(len(flat_data), n_samples, replace=False)
        sampled_values = flat_data[sample_indices]

        plt.figure(figsize=(8, 6))
        sns.histplot(sampled_values, kde=False)
        plt.xlabel("Expression values")
        plt.ylabel("Count")
        plt.title(f"Expression distribution ({n_samples} random samples), \n{title}")
        plt.show()
        plt.close()


def filter_low_quality_cells(
    adata_rna, max_pct_mt=20, min_genes=200, min_counts=1000, plot_flag=False
):
    """
    Filter low-quality cells based on QC metrics
    """
    print("Filtering low-quality cells...")
    print(f"Before filtering: {adata_rna.n_obs} cells")

    # Create boolean masks for filtering
    cell_filter = (
        (adata_rna.obs["pct_counts_mt"] < max_pct_mt)
        & (adata_rna.obs["n_genes_by_counts"] >= min_genes)
        & (adata_rna.obs["total_counts"] >= min_counts)
    )

    print(f"Cells passing QC: {cell_filter.sum()}/{len(cell_filter)} ({cell_filter.mean():.1%})")
    print(f"- High MT%: {(adata_rna.obs['pct_counts_mt'] >= max_pct_mt).sum()} cells removed")
    print(
        f"- Low gene count: {(adata_rna.obs['n_genes_by_counts'] < min_genes).sum()} cells removed"
    )
    print(f"- Low total counts: {(adata_rna.obs['total_counts'] < min_counts).sum()} cells removed")
    if plot_flag:
        plot_rna_during_preprocessing(
            adata_rna[cell_filter].copy(), plot_flag, title="after low quality cell filtering"
        )
    return adata_rna[cell_filter].copy()


def feature_selection_counts_only(adata_rna, n_top_genes=2000, plot_flag=False):
    """
    Feature selection without any normalization, log transformation, or scaling.
    Data remains as counts/pseudo-counts throughout.
    """
    print("Step 5: Feature selection (counts only)...")
    batch_key = "batch" if "batch" in adata_rna.obs else None
    adata_rna = adata_rna[adata_rna.X.sum(axis=1) > 0].copy()

    # HVG selection on raw counts data
    sc.pp.highly_variable_genes(
        adata_rna, n_top_genes=n_top_genes, flavor="seurat_v3", batch_key=batch_key
    )

    # Filter to HVGs - counts layer will be filtered along with X
    # DON'T overwrite counts layer here - it should already exist from Step 0
    adata_rna = adata_rna[:, adata_rna.var["highly_variable"]]

    print(
        f"Selected {adata_rna.n_vars} highly variable genes (data kept as log1p, counts layer preserved)"
    )

    # NO normalization, NO log transformation, NO scaling!

    if plot_flag:
        plot_rna_during_preprocessing(
            adata_rna, plot_flag, title="after feature selection (counts only)"
        )
    return adata_rna


def evaluate_batch_correction_methods(methods, batch_key, celltype_key=None):
    """
    Evaluate batch correction methods quantitatively

    Parameters:
    methods: list of tuples [(adata, method_name, use_rep)]
    batch_key: column name for batch information
    celltype_key: column name for cell type information (optional)

    Returns:
    best_method_name, best_adata, evaluation_results
    """
    from arcadia.training.metrics import calculate_batch_ilisi_score, calculate_cell_type_silhouette

    results = []

    print("Evaluating batch correction methods...")
    print("=" * 50)

    for adata_method, method_name, use_rep in methods:
        print(f"Evaluating {method_name}...")
        adata_method_subsampled = adata_method.copy()
        sc.pp.subsample(adata_method_subsampled, n_obs=min(adata_method_subsampled.shape[0], 8000))
        # 1. Batch mixing iLISI (higher is better)
        batch_ilisi = calculate_batch_ilisi_score(adata_method_subsampled, batch_key)

        # 2. Cell type clustering silhouette (higher is better)
        silhouette = None
        if celltype_key:
            silhouette = calculate_cell_type_silhouette(
                adata_method_subsampled, celltype_key, use_rep
            )

        results.append(
            {
                "method": method_name,
                "adata": adata_method,
                "batch_ilisi": batch_ilisi,
                "silhouette": silhouette,
                "use_rep": use_rep,
            }
        )

        print(
            f"  Batch mixing iLISI: {batch_ilisi:.4f}"
            if batch_ilisi
            else "  Batch mixing iLISI: Failed"
        )
        print(
            f"  Cell type silhouette: {silhouette:.4f}"
            if silhouette
            else "  Cell type silhouette: N/A"
        )

    # Select best method
    valid_results = [r for r in results if r["batch_ilisi"] is not None]

    if not valid_results:
        print("No valid results found. Using first method as fallback.")
        return methods[0][1], methods[0][0], results

    if celltype_key and any(r["silhouette"] is not None for r in valid_results):
        # Combined scoring: normalize both metrics and combine
        print("Using combined scoring (batch mixing + cell type clustering)")

        valid_with_silhouette = [r for r in valid_results if r["silhouette"] is not None]
        if valid_with_silhouette:
            ilisi_scores = np.array([r["batch_ilisi"] for r in valid_with_silhouette])
            silhouettes = np.array([r["silhouette"] for r in valid_with_silhouette])

            # Normalize scores to 0-1 range
            ilisi_norm = (ilisi_scores - ilisi_scores.min()) / (
                ilisi_scores.max() - ilisi_scores.min() + 1e-10
            )
            silhouette_norm = (silhouettes - silhouettes.min()) / (
                silhouettes.max() - silhouettes.min() + 1e-10
            )

            # Combined score (equal weighting)
            combined_scores = ilisi_norm + silhouette_norm

            best_idx = np.argmax(combined_scores)
            best_method = valid_with_silhouette[best_idx]["method"]
            best_adata = valid_with_silhouette[best_idx]["adata"]
        else:
            # Fallback to iLISI only
            ilisi_scores = [r["batch_ilisi"] for r in valid_results]
            best_idx = np.argmax(ilisi_scores)
            best_method = valid_results[best_idx]["method"]
            best_adata = valid_results[best_idx]["adata"]
    else:
        # Use only batch mixing iLISI
        print("Using batch mixing iLISI only")
        ilisi_scores = [r["batch_ilisi"] for r in valid_results]
        best_idx = np.argmax(ilisi_scores)
        best_method = valid_results[best_idx]["method"]
        best_adata = valid_results[best_idx]["adata"]

    print("=" * 50)
    print(f"BEST METHOD: {best_method}")
    print("=" * 50)

    return best_method, best_adata, results


def generate_batch_correction_plots(methods, best_method_name, batch_key, celltype_key):
    """
    Generate comparison plots for batch correction methods
    """
    print("\nGenerating UMAP visualizations...")

    n_methods = len(methods)
    n_cols = 2
    n_rows = (n_methods + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (adata_method, method_name, use_rep) in enumerate(methods):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        adata_method_subsampled = adata_method.copy()
        sc.pp.subsample(adata_method_subsampled, n_obs=min(adata_method_subsampled.shape[0], 2000))
        # Always recompute UMAP to ensure we use the correct representation

        if use_rep in adata_method_subsampled.obsm:
            sc.pp.neighbors(adata_method_subsampled, use_rep=use_rep)
        else:
            print(f"Warning: {use_rep} not found for {method_name}, using X_pca")
            sc.pp.neighbors(adata_method_subsampled, use_rep="X_pca")
        sc.tl.umap(adata_method_subsampled)

        # Color by batch
        sc.pl.umap(adata_method_subsampled, color=batch_key, ax=ax, show=False, frameon=False)

        # Mark best method
        title = f"{method_name} Batch Correction"
        if method_name == best_method_name:
            title += " ⭐ BEST"
        ax.set_title(title)

    # Hide extra subplots
    for i in range(len(methods), n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        "/home/barroz/projects/ARCADIA/.vscode/_batch_correction_comparison.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Second figure with cell types if available
    if celltype_key:
        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
        if n_rows == 1:
            axes2 = axes2.reshape(1, -1)

        for i, (adata_method, method_name, use_rep) in enumerate(methods):
            row, col = i // n_cols, i % n_cols
            ax = axes2[row, col]
            adata_method_subsampled = adata_method.copy()
            sc.pp.subsample(
                adata_method_subsampled, n_obs=min(adata_method_subsampled.shape[0], 2000)
            )

            # Recompute neighbors and UMAP using the correct representation for cell type visualization
            if use_rep in adata_method_subsampled.obsm:
                sc.pp.neighbors(adata_method_subsampled, use_rep=use_rep)
            else:
                print(f"Warning: {use_rep} not found for {method_name}, using X_pca")
                sc.pp.neighbors(adata_method_subsampled, use_rep="X_pca")
            sc.tl.umap(adata_method_subsampled)

            sc.pl.umap(
                adata_method_subsampled, color=celltype_key, ax=ax, show=False, frameon=False
            )
            title = f"{method_name} - Cell Types"
            if method_name == best_method_name:
                title += " ⭐ BEST"
            ax.set_title(title)

        # Hide extra subplots
        for i in range(len(methods), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes2[row, col].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            "/home/barroz/projects/ARCADIA/.vscode/_batch_correction_celltypes.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()


def apply_batch_correction_pipeline(
    adata: AnnData,
    plot_flag: bool = False,
    batch_correction_method: str | list[str] = "ComBat",
    gene_likelihood_dist=None,
) -> AnnData:
    """
    Step 7: Comprehensive batch correction pipeline
    """
    import scvi

    batch_key = "batch" if "batch" in adata.obs else None

    if not batch_key or len(set(adata.obs[batch_key])) <= 1:
        print("Single dataset detected, no batch correction needed")
        return adata

    print("Step 7: Batch correction...")
    print("Multiple datasets detected. Performing batch correction comparison...")

    print("Applying different batch correction methods...")
    if isinstance(batch_correction_method, str):
        batch_correction_method = [batch_correction_method]
    methods = []
    if "ComBat" in batch_correction_method:
        # 1. ComBat correction
        adata_combat = adata.copy()
        sc.pp.combat(adata_combat, key=batch_key)
        sc.pp.pca(adata_combat)
        adata_combat.obsm["X_ComBat"] = adata_combat.obsm["X_pca"].copy()
        methods.append((adata_combat, "ComBat", "X_ComBat"))

    if "Scanorama" in batch_correction_method:
        import scanorama

        # 2. Scanorama correction
        adata_scanorama_input = adata.copy()
        sc.pp.pca(adata_scanorama_input)
        batch_values = adata_scanorama_input.obs[batch_key].unique()
        adata_list = [
            adata_scanorama_input[adata_scanorama_input.obs[batch_key] == batch].copy()
            for batch in batch_values
        ]
        adatas_corrected = scanorama.correct_scanpy(adata_list, return_dimred=True)
        adata_scanorama = adatas_corrected[0].concatenate(*adatas_corrected[1:])
        adata_scanorama.obsm["X_Scanorama"] = adata_scanorama.obsm["X_pca"].copy()
        methods.append((adata_scanorama, "Scanorama", "X_Scanorama"))
    if "Harmony" in batch_correction_method:
        import scanpy.external as sce

        # 3. Harmony correction
        adata_harmony = adata.copy()
        sc.pp.pca(adata_harmony)
        sce.pp.harmony_integrate(adata_harmony, key=batch_key)
        adata_harmony.obsm["X_Harmony"] = adata_harmony.obsm["X_pca_harmony"].copy()
        methods.append((adata_harmony, "Harmony", "X_Harmony"))

    # scANVI if cell types are available

    celltype_key = None
    if "cell_types" in adata.obs.columns and "scANVI" in batch_correction_method:
        celltype_key = "cell_types"
        adata_scanvi = adata.copy()
        if sp.issparse(adata_scanvi.X):
            adata_scanvi_values = adata_scanvi.X.toarray()
        else:
            adata_scanvi_values = adata_scanvi.X

        # Clean data for scANVI - it needs clean, finite values
        if np.isnan(adata_scanvi_values).any() or np.isinf(adata_scanvi_values).any():
            print("WARNING: Found NaN/inf values, cleaning for scANVI...")
            if sp.issparse(adata_scanvi_values):
                adata_scanvi_values = np.nan_to_num(
                    adata_scanvi_values, nan=0.0, posinf=0.0, neginf=0.0
                )
            else:
                adata_scanvi.X = np.nan_to_num(adata_scanvi.X, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"After cleaning - X has NaN: {np.isnan(adata_scanvi.X).any()}")

        # Ensure scANVI gets proper count data

        # Fix negative values (shouldn't happen with raw counts, but safety check)
        if adata_scanvi_values.min() < 0:
            print("WARNING: Found negative values, clipping to 0")
            adata_scanvi_values = np.clip(adata_scanvi_values, 0, None)

        # Ensure non-negative integers for scANVI
        adata_scanvi_values = np.round(np.abs(adata_scanvi_values)).astype(np.float32)
        adata_scanvi.X = adata_scanvi_values
        scvi.model.SCANVI.setup_anndata(
            adata_scanvi,
            batch_key=batch_key,
            labels_key=celltype_key,
            unlabeled_category="Unknown",
        )
        early_stopping_kwargs = {
            "early_stopping": True,  # activate callback
            "early_stopping_monitor": "elbo_validation",  # what to watch
            "early_stopping_patience": 200,  # # of checks with no improvement
            "early_stopping_min_delta": 0.001,  # required improvement size
            "check_val_every_n_epoch": 15,  # validation frequency
        }

        if gene_likelihood_dist is None:
            raise ValueError("gene_likelihood_dist is required for scANVI")
        model = scvi.model.SCANVI(
            adata_scanvi, n_latent=50, n_hidden=256, gene_likelihood=gene_likelihood_dist
        )
        print(f"Dataset size is [samples, dims], {adata_scanvi.shape[0]}, {adata_scanvi.shape[1]}")
        if adata_scanvi.shape[0] < 10000:
            batch_size = 512
        else:
            batch_size = 4096
        model.train(
            max_epochs=500,  # todo back to 500
            batch_size=batch_size,
            **early_stopping_kwargs,
        )
        if plot_flag:
            history = model.history
            # The ELBO is stored under these two keys
            train = history["elbo_train"]["elbo_train"]
            val = history["elbo_validation"]["elbo_validation"]

            plt.figure(figsize=(5, 4))
            plt.plot(train, label="train", color="steelblue")
            plt.plot(val, label="validation", color="darkorange")
            plt.xlabel("Epoch")
            plt.ylabel("Negative ELBO")
            plt.title("scANVI loss curves")
            plt.legend()
            plt.tight_layout()
            plt.savefig("batch_correction_scANVI_training_loss.pdf")
            plt.show()
            plt.close()

        adata_scanvi.obsm["X_scANVI"] = model.get_latent_representation()

        # get the normalized expression (not counts) and choose a random batch as the transform_batch (crucial)
        adata_scanvi.obsm["scANVI_normalized"] = model.get_normalized_expression(
            transform_batch=adata_scanvi.obs[batch_key][0]
        )
        # adata_scanvi.X = adata_scanvi.obsm["scANVI_normalized"]
        sc.pp.pca(adata_scanvi)

        adata_scanvi.uns["scANVI_model"] = model

        # sc.pp.pca(adata_scanvi)
        # sc.pp.neighbors(adata_scanvi)
        # sc.tl.umap(adata_scanvi)
        # sc.pl.umap(adata_scanvi, color="batch")
        # sc.pl.umap(adata_scanvi, color="cell_types"

        # import scanpy as sc
        # import matplotlib.pyplot as plt

        # # Create AnnData from normalized expression
        # adata_norm = sc.AnnData(X=adata_scanvi.layers["scANVI_normalized"],obs=adata_scanvi.obs,var=adata_scanvi.var)
        # # Add cell types

        # # Run PCA
        # sc.pp.pca(adata_norm, n_comps=50)
        # # Compute neighborhood graph
        # sc.pp.neighbors(adata_norm, n_pcs=50)
        # # Calculate UMAP
        # sc.tl.umap(adata_norm)

        # # Plot UMAP colored by cell_types
        # sc.pl.umap(adata_norm, color="cell_types", title="UMAP of scANVI normalized expression colored by cell_types")

        # plt.savefig(
        #     "/home/barroz/projects/ARCADIA/.vscode/_batch_correction_scANVI.pdf",
        #     dpi=300,
        #     bbox_inches="tight",
        # )
        # sc.pl.umap(adata_norm, color="batch", title="UMAP of scANVI normalized expression colored by batch")

        # plt.savefig(
        #     "/home/barroz/projects/ARCADIA/.vscode/_batch_correction_scANVI2.pdf",
        #     dpi=300,
        #     bbox_inches="tight",
        # )
        # sc.pl.embedding(
        #     adata_scanvi,
        #     basis="X_scANVI",
        #     color="cell_types",
        #     title="UMAP (scANVI latent space) colored by cell_types",
        # )
        # plt.savefig(
        #     "/home/barroz/projects/ARCADIA/.vscode/_batch_correction_scANVI3.pdf",
        #     dpi=300,
        #     bbox_inches="tight",
        # )
        # sc.pl.embedding(
        #     adata_scanvi,
        #     basis="X_scANVI",
        #     color="batch",
        #     title="UMAP (scANVI latent space) colored by batch",
        # )
        # plt.savefig(
        #     "/home/barroz/projects/ARCADIA/.vscode/_batch_correction_scANVI4.pdf",
        #     dpi=300,
        #     bbox_inches="tight",
        # )
        # plt.close()

        methods.append((adata_scanvi, "scANVI", "X_scANVI"))
        print("scANVI completed successfully!")

    # Evaluate methods and select best
    best_method_name, best_adata, evaluation_results = evaluate_batch_correction_methods(
        methods, batch_key, celltype_key
    )

    # Generate visualization comparison
    if plot_flag:
        generate_batch_correction_plots(methods, best_method_name, batch_key, celltype_key)

    # Apply the best batch correction to the data
    print(f"\nApplying {best_method_name} batch correction to the data...")

    # Copy the batch-corrected data to the main adata object
    result_adata = adata.copy()

    if best_method_name == "scANVI":
        # For scANVI, ONLY copy the embeddings, NOT the X data
        # X was modified by scANVI (rounded for integer input), so we keep the original
        result_adata.obsm["X_scANVI"] = best_adata.obsm["X_scANVI"].copy()
        result_adata.obsm["X_scANVI_pca"] = best_adata.obsm["X_pca"].copy()
        # result_adata.obsm["X_scANVI_neighbors"] = best_adata.obsm["neighbors"].copy()
        result_adata.obsm["X_scANVI_umap"] = best_adata.obsm["X_umap"].copy()
        print(f"Preserved X_scANVI embedding, kept original X data (log1p normalized)")

    elif best_method_name == "ComBat":
        # For ComBat, the counts are already batch-corrected
        result_adata.X = best_adata.X.copy()
        result_adata.obsm["X_pca"] = best_adata.obsm["X_pca"].copy()
        print(f"Applied ComBat corrected data to X and X_pca")

    else:
        # For PCA-based methods (Harmony, Scanorama)
        # Use the corrected representation for PCA and scale the original for X
        use_rep = next(rep for _, method, rep in methods if method == best_method_name)
        result_adata.obsm["X_pca"] = best_adata.obsm[use_rep].copy()
        result_adata.obsm[f"X_{best_method_name}"] = best_adata.obsm[use_rep].copy()

        # For visualization purposes, we need to update X with batch-corrected scaled data
        # The best approach is to use the corrected data from the best method
        if hasattr(best_adata, "X") and best_adata.X is not None:
            result_adata.X = best_adata.X.copy()

        print(f"Applied {best_method_name} corrected representation to X_pca and scaled data to X")

    # Ensure we have the corrected PCA representation
    if "X_pca" not in result_adata.obsm:
        print("Warning: No X_pca found, computing PCA on corrected data...")
        sc.pp.pca(result_adata)

    print(f"Final result - X shape: {result_adata.X.shape}")
    print(f"Final result - X_pca shape: {result_adata.obsm['X_pca'].shape}")
    print(f"Final result - Available embeddings: {list(result_adata.obsm.keys())}")
    print(f"Batch correction successfully applied to both X and X_pca representations")

    print(f"\nUsing {best_method_name} for downstream analysis")
    if plot_flag:
        plot_rna_during_preprocessing(
            result_adata, plot_flag, title=f"after {best_method_name} batch correction"
        )
    return result_adata


# Batch correction evaluation functions are defined above (evaluate_batch_correction_methods, generate_batch_correction_plots)


def harmonize_cell_types_names(
    adata_rna: AnnData, adata_prot: AnnData, harmonization_mapping
) -> tuple[AnnData, AnnData]:
    """harmonize_cell_types_names harmonizes the cell types names between rna and protein anndatas
    Input: adata_rna, adata_prot
    1. Initialize uns["pipeline_metadata"]
    2. Writes obs["cell_types"]
    3. Removes redundant gene cols
    4. ZIB Thresholding for RNA
    Output: cleaned rna_adta and adata_prot
    """
    print("[DEBUG] harmonize_cell_types_names: Starting function")
    print(f"[DEBUG] RNA shape: {adata_rna.shape}, Protein shape: {adata_prot.shape}")

    # Initialize metadata tracking with sets of unique values
    print("[DEBUG] Copying pipeline metadata...")
    adata_prot.uns["pipeline_metadata"] = adata_rna.uns["pipeline_metadata"].copy()
    print("[DEBUG] Pipeline metadata copied")

    print("[DEBUG] Sorting RNA data by cell_types...")
    print("[DEBUG] RNA data sorted")

    print("[DEBUG] Sorting Protein data by cell_types...")
    print("[DEBUG] Protein data sorted")

    # make sure we dont have gene column in var if it is equal to the index
    print("[DEBUG] Checking for redundant gene columns...")
    if "gene" in adata_rna.var.columns and np.array_equal(
        adata_rna.var["gene"].values, (adata_rna.var.index.values)
    ):
        adata_rna.var.drop(columns="gene", inplace=True)
    if "gene" in adata_prot.var.columns and np.array_equal(
        adata_prot.var["gene"].values, (adata_prot.var.index.values)
    ):
        adata_prot.var.drop(columns="gene", inplace=True)
    print("[DEBUG] Gene column check complete")

    print("[DEBUG] Loading harmonization mappings...")
    rna_map = harmonization_mapping["rna_to_codex_mapping"]
    expected_codex_annotations_common_set = harmonization_mapping[
        "expected_codex_annotations_common_set"
    ]
    print(f"[DEBUG] RNA map has {len(rna_map)} entries")
    print(f"[DEBUG] Expected common set has {len(expected_codex_annotations_common_set)} entries")

    def apply_map(adata, mapping):
        print(f"[DEBUG] apply_map: Starting mapping on dataset with {adata.shape[0]} cells")
        # convert to series for easy mapping, keep original index order
        s = adata.obs["cell_types"]
        print(f"[DEBUG] apply_map: Unique cell types before mapping: {len(s.unique())}")
        # unmapped labels become NaN → later dropped
        s_mapped = s.map(mapping)
        print(f"[DEBUG] apply_map: Mapped, {s_mapped.isna().sum()} cells will be dropped")
        adata = adata[s_mapped.notna()].copy()  # drop rows with NaN
        print(f"[DEBUG] apply_map: After dropping, {adata.shape[0]} cells remain")
        adata.obs["cell_types"] = s_mapped.loc[adata.obs_names]
        print(f"[DEBUG] apply_map: Mapping complete")
        return adata

    print("[DEBUG] Applying mapping to RNA data...")
    adata_rna = apply_map(adata_rna, rna_map)
    print("[DEBUG] RNA mapping complete")

    print("[DEBUG] Computing cell type intersections and differences...")
    set(adata_rna.obs["cell_types"]).intersection(set(adata_prot.obs["cell_types"]))
    # difference between the two sets
    rna_unmapped_types = set(adata_rna.obs["cell_types"]).difference(
        set(adata_prot.obs["cell_types"])
    )
    prot_unmapped_types = set(adata_prot.obs["cell_types"]).difference(
        set(adata_rna.obs["cell_types"])
    )
    shared_cell_types = set(adata_rna.obs["cell_types"]).intersection(
        set(adata_prot.obs["cell_types"])
    )
    print(f"[DEBUG] Shared cell types: {len(shared_cell_types)}")
    print(f"[DEBUG] RNA unmapped types: {len(rna_unmapped_types)}")
    print(f"[DEBUG] Protein unmapped types: {len(prot_unmapped_types)}")

    print("[DEBUG] Validating shared cell types against expected set...")
    if set(shared_cell_types) != set(expected_codex_annotations_common_set):
        print(
            f"cell types expected but missing: {set(expected_codex_annotations_common_set)- set(shared_cell_types)}"
        )
        print(
            f"cell types in common but not expected: {set(shared_cell_types)- set(expected_codex_annotations_common_set)}"
        )
        if len(set(shared_cell_types) - set(expected_codex_annotations_common_set)) > 0:
            raise ValueError(
                f"cell types expected but missing: {set(expected_codex_annotations_common_set)- set(shared_cell_types)}"
            )
    final_labels = sorted(list(shared_cell_types))
    print("[DEBUG] Filtering unwanted cell types from protein data...")
    adata_prot = filter_unwanted_cell_types(
        adata_prot, ["dead", "tumor", "endothelial cells", "exclude"]
    )
    print(f"[DEBUG] Protein data after filtering: {adata_prot.shape}")

    print("[DEBUG] Filtering unwanted cell types from RNA data...")
    adata_rna = filter_unwanted_cell_types(
        adata_rna, ["dead", "tumor", "endothelial cells", "exclude"]
    )
    print(f"[DEBUG] RNA data after filtering: {adata_rna.shape}")
    print("[DEBUG] Recomputing shared cell types after filtering...")
    shared_cell_types = set(adata_rna.obs["cell_types"]).intersection(
        set(adata_prot.obs["cell_types"])
    )
    print(f"[DEBUG] Final shared cell types: {len(shared_cell_types)}")

    final_labels = sorted(list(shared_cell_types))
    # only keep the final labels
    final_labels = set(final_labels)

    print("[DEBUG] Filtering RNA data to final labels...")
    adata_rna = adata_rna[adata_rna.obs["cell_types"].isin(final_labels)].copy()
    print(f"[DEBUG] RNA data after label filtering: {adata_rna.shape}")

    print("[DEBUG] Filtering Protein data to final labels...")
    adata_prot = adata_prot[adata_prot.obs["cell_types"].isin(final_labels)].copy()
    print(f"[DEBUG] Protein data after label filtering: {adata_prot.shape}")

    print("[DEBUG] Setting categorical categories...")
    for ad in (adata_prot, adata_rna):
        ad.obs["cell_types"] = (
            ad.obs["cell_types"].astype("category").cat.set_categories(final_labels)
        )
    print("[DEBUG] Categorical categories set")

    print("Final shared classes:", final_labels)
    print("\nProtein counts:\n", adata_prot.obs["cell_types"].value_counts())
    print("\nRNA counts:\n", adata_rna.obs["cell_types"].value_counts())
    assert not adata_rna.obs["cell_types"].isna().any()
    assert not adata_prot.obs["cell_types"].isna().any()
    adata_rna.obs["major_cell_types"] = adata_rna.obs["cell_types"]
    adata_prot.obs["major_cell_types"] = adata_prot.obs["cell_types"]

    return adata_rna, adata_prot


def convert_to_sparse_csr(adata_rna, adata_prot, logger=None):
    """Convert data matrices to sparse CSR format for memory efficiency.

    Args:
        adata_rna: RNA AnnData object
        adata_prot: Protein AnnData object
        logger: Optional logger instance

    Returns:
        tuple: (adata_rna, adata_prot) with converted matrices
    """
    if logger:
        logger.info("Converting data matrices to sparse format for memory efficiency...")

    for adata, name in [(adata_rna, "RNA"), (adata_prot, "Protein")]:
        if not sp.issparse(adata.X):
            if logger:
                logger.info(f"Converting {name} X from {type(adata.X)} to sparse CSR matrix")
            adata.X = sp.csr_matrix(adata.X)
        else:
            if logger:
                logger.info(f"{name} X already sparse: {type(adata.X)}")

        if adata.X.dtype != np.float32:
            adata.X = adata.X.astype(np.float32)

    return adata_rna, adata_prot
