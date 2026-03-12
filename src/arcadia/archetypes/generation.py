# %%
"""
Archetype generation and cross-modal matching module.

This module implements the complete archetype analysis pipeline:
1. Per-batch archetype generation within each modality
2. Within-modality batch matching for quality control
3. Cross-modal archetype matching to find optimal k
4. Unified archetype representation creation
5. Extreme archetype identification and filtering

Key functions:
- generate_archetypes_per_batch: Generate archetypes separately for each batch
- match_archetypes_across_batches_for_all_k: Match archetypes across batches for all k values
- find_optimal_k_across_modalities: Complete cross-modal archetype analysis (main entry point)
- create_unified_archetype_representation: Create unified archetype vectors
- filter_extreme_archetypes_by_cross_modal_quality: Filter extreme archetypes by cross-modal quality
"""

import warnings
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.colors import to_hex

# Suppress pkg_resources deprecation warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="louvain")

from py_pcha import PCHA
from tqdm import tqdm

# Import necessary functions from arcadia.archetypes.matching
from arcadia.archetypes.matching import (
    find_best_pair_by_row_matching,
    identify_extreme_archetypes_balanced,
)

# Import plotting functions from visualization.py
from arcadia.archetypes.visualization import (
    plot_archetype_proportions_before_after_matching,
    plot_cross_modal_archetype_comparison,
    visualize_archetype_proportions_analysis,
)


def update_archetype_labels(adata: AnnData, archetype_vectors_key: str = "archetype_vec") -> None:
    """
    Update archetype labels based on current archetype vectors.

    This utility ensures that archetype labels are always in sync with archetype vectors
    by finding the dominant archetype for each cell.

    Args:
        adata: AnnData object to update
        archetype_vectors_key: Key in adata.obsm containing archetype vectors
    """
    if archetype_vectors_key not in adata.obsm:
        print(f"Warning: {archetype_vectors_key} not found in adata.obsm")
        return

    archetype_vectors = adata.obsm[archetype_vectors_key]
    if hasattr(archetype_vectors, "values"):
        archetype_vectors = archetype_vectors.values

    # Update labels to match current vectors
    adata.obs["archetype_label"] = pd.Categorical(np.argmax(archetype_vectors, axis=1))
    print(f"Updated archetype labels: {len(adata.obs['archetype_label'].unique())} unique labels")


def get_cell_type_colors(cell_types: List[str], palette_name: str = "tab20") -> Dict[str, str]:
    """
    Generate consistent cell type colors.

    Args:
        cell_types: List of unique cell types
        palette_name: Name of matplotlib colormap to use

    Returns:
        Dictionary mapping cell types to hex colors
    """
    n_colors = len(cell_types)
    if n_colors <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, n_colors))

    return {cell_type: to_hex(color) for cell_type, color in zip(cell_types, colors)}


def filter_extreme_archetypes_by_cross_modal_quality(
    adata_1: AnnData,
    adata_2: AnnData,
    cell_type_key: str = "cell_types",
    proportion_threshold: float = 0.5,
    verbose: bool = True,
) -> None:
    """
    Filter extreme archetypes to keep only those with good cross-modal matching quality.

    This function compares the cell type proportions of each archetype between modalities
    and removes the extreme archetype flag from cells belonging to archetypes with poor
    cross-modal matching (high dissimilarity in cell type proportions).

    Args:
        adata_1: First modality AnnData object (e.g., RNA)
        adata_2: Second modality AnnData object (e.g., Protein)
        cell_type_key: Key in obs containing cell type information
        proportion_threshold: Maximum allowed dissimilarity in proportions (0-1) lower is stricter
        verbose: Whether to print detailed information
    """
    print(f"\n=== Filtering extreme archetypes by cross-modal matching quality ===")

    # Get number of archetypes (should be same for both modalities)
    n_archetypes_1 = len(adata_1.obs["archetype_label"].unique())
    n_archetypes_2 = len(adata_2.obs["archetype_label"].unique())

    if n_archetypes_1 != n_archetypes_2:
        print(f"Warning: Different number of archetypes: {n_archetypes_1} vs {n_archetypes_2}")
        n_archetypes = min(n_archetypes_1, n_archetypes_2)
    else:
        n_archetypes = n_archetypes_1

    print(f"Comparing {n_archetypes} archetype pairs")

    # Calculate cell type proportions for each archetype in both modalities
    archetype_quality_scores = []
    good_quality_archetypes = []

    for arch_idx in range(n_archetypes):
        # Get cells belonging to this archetype in both modalities
        mask_1 = adata_1.obs["archetype_label"] == arch_idx
        mask_2 = adata_2.obs["archetype_label"] == arch_idx

        if not mask_1.any() or not mask_2.any():
            if verbose:
                print(f"Archetype {arch_idx}: No cells found in one or both modalities")
            archetype_quality_scores.append(1.0)  # Mark as poor quality
            continue

        # Get cell types for cells in this archetype
        cell_types_1 = adata_1.obs[cell_type_key][mask_1]
        cell_types_2 = adata_2.obs[cell_type_key][mask_2]

        # Calculate proportions
        props_1 = cell_types_1.value_counts(normalize=True).sort_index()
        props_2 = cell_types_2.value_counts(normalize=True).sort_index()

        # Align proportions (handle different cell types)
        all_cell_types = sorted(list(set(props_1.index) | set(props_2.index)))
        props_1_aligned = props_1.reindex(all_cell_types, fill_value=0.0)
        props_2_aligned = props_2.reindex(all_cell_types, fill_value=0.0)

        # Calculate dissimilarity (using Jensen-Shannon divergence for robustness)
        # JS divergence is symmetric and bounded [0,1]
        from scipy.spatial.distance import jensenshannon

        js_distance = jensenshannon(props_1_aligned.values, props_2_aligned.values)

        archetype_quality_scores.append(js_distance)

        if js_distance <= proportion_threshold:
            good_quality_archetypes.append(arch_idx)

        if verbose:
            n_cells_1 = mask_1.sum()
            n_cells_2 = mask_2.sum()
            print(
                f"Archetype {arch_idx}: {n_cells_1} cells (mod1), {n_cells_2} cells (mod2), "
                f"JS distance: {js_distance:.3f}, Quality: {'GOOD' if js_distance <= proportion_threshold else 'POOR'}"
            )

    print(
        f"\nFound {len(good_quality_archetypes)} good quality archetype pairs out of {n_archetypes}"
    )
    print(f"Good quality archetypes: {good_quality_archetypes}")

    # Count extreme archetypes before filtering
    extreme_before_1 = adata_1.obs["is_extreme_archetype"].sum()
    extreme_before_2 = adata_2.obs["is_extreme_archetype"].sum()

    # Filter extreme archetypes - keep only those belonging to good quality archetypes
    for adata, modality_name in [(adata_1, "Modality 1"), (adata_2, "Modality 2")]:
        # Create mask for cells belonging to good quality archetypes
        good_quality_mask = adata.obs["archetype_label"].isin(good_quality_archetypes)

        # Update extreme archetype flag: keep extreme flag only for good quality archetypes
        adata.obs["is_extreme_archetype"] = adata.obs["is_extreme_archetype"] & good_quality_mask

        extreme_after = adata.obs["is_extreme_archetype"].sum()
        extreme_removed = (
            extreme_before_1 if modality_name == "Modality 1" else extreme_before_2
        ) - extreme_after

        print(
            f"{modality_name}: Removed {extreme_removed} extreme archetype flags, "
            f"{extreme_after} remain ({extreme_after/len(adata)*100:.1f}%)"
        )

    # Create archetype quality dictionary (archetype_index -> is_good_quality)
    archetype_quality_dict = {}
    for arch_idx in range(n_archetypes):
        archetype_quality_dict[arch_idx] = arch_idx in good_quality_archetypes

    # Store quality scores in uns for potential analysis
    quality_metadata = {
        "archetype_quality_scores": archetype_quality_scores,
        "good_quality_archetypes": good_quality_archetypes,
        "archetype_quality_dict": archetype_quality_dict,
        "proportion_threshold": proportion_threshold,
        "n_good_quality": len(good_quality_archetypes),
        "n_total_archetypes": n_archetypes,
    }

    adata_1.uns["cross_modal_quality"] = quality_metadata
    adata_2.uns["cross_modal_quality"] = quality_metadata.copy()

    # Also store the dictionary directly for easy access
    adata_1.uns["archetype_quality"] = archetype_quality_dict
    adata_2.uns["archetype_quality"] = archetype_quality_dict.copy()

    print(f"✅ Extreme archetype filtering completed")


def generate_archetypes_per_batch(
    adata: AnnData,
    batch_key: str = "batch",
    embedding_name: str = "X_pca",
    min_k: int = 8,
    max_k: int = 9,
    step_size: int = 1,
    converge: float = 1e-5,
    modality_name: str = "RNA",
    plot_flag: bool = True,
) -> Dict[str, Dict]:
    """
    Generate archetypes separately for each batch within a modality.

    Args:
        adata: AnnData object containing the data
        batch_key: Key in adata.obs containing batch information
        embedding_name: Name of embedding to use for archetype generation
        min_k: Minimum number of archetypes to test
        max_k: Maximum number of archetypes to test
        step_size: Step size for archetype number testing
        converge: Convergence threshold for early stopping
        modality_name: Name of modality for logging

    Returns:
        batch_archetypes: Dictionary containing archetype information for each batch
    """
    print(f"\n=== Generating {modality_name} archetypes per batch ===")

    # Get unique batches
    batches = adata.obs[batch_key].unique()
    print(f"Found {len(batches)} batches: {batches}")

    batch_archetypes = {}

    for batch in batches:
        print(f"\nProcessing batch: {batch}")

        # Get batch-specific data
        batch_mask = adata.obs[batch_key] == batch
        batch_adata = adata[batch_mask].copy()

        if batch_adata.shape[0] < min_k * 2:
            print(f"Warning: Batch {batch} has only {batch_adata.shape[0]} cells, skipping")
            continue

        print(f"Batch {batch} has {batch_adata.shape[0]} cells")

        X_batch = batch_adata.obsm[embedding_name].T

        archetype_list = []
        archetype_activities_list = []
        evs = []

        total_tests = (max_k - min_k) // step_size
        for i, k in tqdm(
            enumerate(range(min_k, max_k, step_size)),
            total=total_tests,
            desc=f"Batch {batch} {modality_name} Archetypes",
        ):
            archetype, archetype_activities, _, _, ev = PCHA(X_batch, noc=k)
            evs.append(ev)
            archetype_list.append(np.array(archetype).T)
            # Store archetype_activities (transposed to match expected shape)
            archetype_activities_list.append(np.array(archetype_activities).T)

            # Early stopping if convergence achieved
            if i > 0 and evs[i] - evs[i - 1] < converge:
                print(f"Early stopping for batch {batch} at k={k}")
                break

        batch_archetypes[batch] = {
            "archetype_list": archetype_list,
            "archetype_activities_list": archetype_activities_list,
            "explained_variances": evs,
            "k_values": list(range(min_k, min_k + len(archetype_list) * step_size, step_size)),
            "batch_indices": np.where(batch_mask)[0],
            "embedding_name": embedding_name,
        }

        print(f"Generated {len(archetype_list)} archetype sets for batch {batch}")

    return batch_archetypes


def compute_batch_archetype_proportions(
    batch_archetypes: Dict[str, Dict],
    adata: AnnData,
    batch_key: str = "batch",
    modality_name: str = "RNA",
    plot_flag: bool = True,
) -> Dict[str, List[pd.DataFrame]]:
    """
    Compute cell type proportions for archetypes in each batch.

    Args:
        batch_archetypes: Dictionary containing batch archetype information
        adata: Original AnnData object to extract batch data from
        batch_key: Key in adata.obs containing batch information
        modality_name: Name of modality for logging

    Returns:
        batch_proportions: Dictionary containing proportion DataFrames for each batch
    """
    print(f"\n=== Computing {modality_name} archetype proportions per batch ===")

    batch_proportions = {}

    for batch, batch_info in batch_archetypes.items():
        print(f"Computing proportions for batch: {batch}")

        # Reconstruct batch_adata from original adata (avoids storing full copy)
        batch_mask = adata.obs[batch_key] == batch
        batch_adata = adata[batch_mask].copy()
        archetype_list = batch_info["archetype_list"]
        archetype_activities_list = batch_info.get("archetype_activities_list", None)
        batch_info["embedding_name"]

        # Ensure we have major_cell_types
        if "major_cell_types" not in batch_adata.obs.columns:
            batch_adata.obs["major_cell_types"] = batch_adata.obs["cell_types"]

        major_cell_types_list = sorted(list(set(batch_adata.obs["major_cell_types"])))
        major_cell_types_amount = [
            batch_adata.obs["major_cell_types"].value_counts()[cell_type]
            for cell_type in major_cell_types_list
        ]

        proportion_list = []
        # iterate over the archetype list which is a list of archetype vectors of each batch
        for i, archetypes in enumerate(tqdm(archetype_list, desc=f"Batch {batch} proportions")):
            # Use precomputed archetype_activities if available, otherwise compute weights
            if archetype_activities_list is not None and i < len(archetype_activities_list):
                weights = archetype_activities_list[i]
            else:
                raise ValueError("Archetype activities list is not available")

            archetype_num = archetypes.shape[0]
            arch_prop = pd.DataFrame(
                np.zeros((archetype_num, len(major_cell_types_list))),
                columns=major_cell_types_list,
            )

            # Calculate proportions for each archetype
            for curr_archetype in range(archetype_num):
                # Get weights for current archetype - weights should be (n_cells, n_archetypes)
                weights_col = weights[:, curr_archetype]

                cell_types_col = batch_adata.obs["major_cell_types"].values

                df = pd.DataFrame({"weight": weights_col, "major_cell_types": cell_types_col})

                grouped = df.groupby("major_cell_types", observed=False)["weight"].sum()[
                    major_cell_types_list
                ]
                arch_prop.loc[curr_archetype, :] = grouped.values / major_cell_types_amount

            # Normalize proportions
            arch_prop = (arch_prop.T / arch_prop.sum(1)).T
            arch_prop = arch_prop / arch_prop.sum(0)
            proportion_list.append(arch_prop.copy())

        batch_proportions[batch] = proportion_list
        print(f"Computed proportions for {len(proportion_list)} archetype sets in batch {batch}")

    return batch_proportions


def create_unified_archetype_representation(
    adata: AnnData,
    matched_archetypes: Dict[str, Dict],
    batch_archetypes: Dict[str, Dict],
    batch_key: str = "batch",
    modality_name: str = "RNA",
    optimal_k: int = None,
) -> AnnData:
    """
    Create unified archetype vectors for all cells using matched archetypes.

    Args:
        adata: Original AnnData object
        matched_archetypes: Dictionary containing matched archetype information
        batch_archetypes: Dictionary containing batch archetype information
        batch_key: Key in adata.obs containing batch information
        modality_name: Name of modality for logging
        optimal_k: The optimal k value selected (to find correct archetype_activities)

    Returns:
        adata: AnnData object with updated archetype vectors
    """
    print(f"\n=== Creating unified {modality_name} archetype representation ===")

    # Initialize archetype vectors array
    n_cells = adata.shape[0]
    reference_batch = matched_archetypes["reference_batch"]
    n_archetypes = len(matched_archetypes[reference_batch]["proportions"])

    unified_archetype_vectors = np.zeros((n_cells, n_archetypes))

    # Process each batch
    for batch in adata.obs[batch_key].unique():
        if batch not in matched_archetypes:
            print(f"Warning: Batch {batch} not found in matched_archetypes, skipping")
            continue

        print(f"Processing batch: {batch}")

        # Get batch-specific data
        batch_mask = adata.obs[batch_key] == batch
        batch_indices = np.where(batch_mask)[0]

        # Get archetype order for this batch
        archetype_order = matched_archetypes[batch]["archetype_order"]

        # Get pre-computed archetype_activities directly from batch_archetypes (already available!)
        # Find the k_idx for optimal_k
        k_idx = batch_archetypes[batch]["k_values"].index(optimal_k)
        original_batch_archetype_vectors = batch_archetypes[batch]["archetype_activities_list"][
            k_idx
        ]

        # Reorder the COLUMNS of the archetype vectors according to the matching results
        # This ensures that column 0 corresponds to the same biological archetype across modalities
        batch_archetype_vectors = original_batch_archetype_vectors[:, archetype_order]

        # Assign to unified array
        unified_archetype_vectors[batch_indices] = batch_archetype_vectors

        print(f"Processed {len(batch_indices)} cells from batch {batch}")

    # Add unified archetype vectors to adata
    adata.obsm["archetype_vec"] = pd.DataFrame(
        unified_archetype_vectors,
        index=adata.obs.index,
        columns=[str(i) for i in range(n_archetypes)],
    )

    # Add archetype labels (dominant archetype for each cell)
    # After reordering, archetype index i corresponds to the same biological archetype across modalities
    adata.obs["archetype_label"] = pd.Categorical(np.argmax(unified_archetype_vectors, axis=1))

    # NOTE: We don't store unified archetypes anymore - only batch-specific ones
    # Each batch has its own archetype positions but with consistent numbering
    # reference_archetypes = matched_archetypes[reference_batch]["archetypes"]
    # reference_order = matched_archetypes[reference_batch]["archetype_order"]
    # adata.uns["archetypes"] = reference_archetypes[reference_order]

    # Store individual batch archetypes for visualization (reordered for consistency)
    for batch in matched_archetypes.keys():
        if batch != "reference_batch":  # Skip the metadata key
            batch_archetypes_original = matched_archetypes[batch]["archetypes"]
            batch_order = matched_archetypes[batch]["archetype_order"]
            adata.uns[f"batch_archetypes_{batch}"] = batch_archetypes_original[batch_order]

    # Add metadata about the batch-specific archetype generation
    if "pipeline_metadata" not in adata.uns:
        adata.uns["pipeline_metadata"] = {}
    if "archetype_generation" not in adata.uns["pipeline_metadata"]:
        adata.uns["pipeline_metadata"]["archetype_generation"] = {}

    # Create minimal metadata (avoid storing full matched_archetypes which contains large matrices)
    minimal_matched_info = {
        batch: {
            "archetype_order": (
                config["archetype_order"].tolist()
                if hasattr(config["archetype_order"], "tolist")
                else list(config["archetype_order"])
            ),
            "matching_cost": float(config.get("matching_cost", 0.0)),
            # Removed: "proportions", "archetypes" - these are large DataFrames/arrays causing bloat
        }
        for batch, config in matched_archetypes.items()
        if batch != "reference_batch"
    }
    minimal_matched_info["reference_batch"] = reference_batch

    adata.uns["pipeline_metadata"]["archetype_generation"].update(
        {
            "batch_specific_generation": True,
            "reference_batch": reference_batch,
            "n_archetypes": n_archetypes,
            "batches_processed": list(matched_archetypes.keys())[
                1:
            ],  # Exclude 'reference_batch' key
            "modality": modality_name,
            "matched_archetypes": minimal_matched_info,  # Store only minimal matching info (no large matrices)
        }
    )

    print(f"Created unified archetype representation with {n_archetypes} archetypes")
    print(f"Archetype vectors shape: {unified_archetype_vectors.shape}")

    return adata


def match_archetypes_across_batches_for_all_k(
    batch_archetypes: Dict[str, Dict],
    batch_proportions: Dict[str, List[pd.DataFrame]],
    modality_name: str = "RNA",
    metric: str = "cosine",
    plot_flag: bool = True,
) -> Dict[int, Dict]:
    """
    Match archetypes across batches for all available k values within a modality.

    This creates batch-matched archetype sets for each k value, keeping all options
    open for later cross-modal comparison.

    Args:
        batch_archetypes: Dictionary containing batch archetype information
        batch_proportions: Dictionary containing proportion DataFrames for each batch
        modality_name: Name of modality for logging
        metric: Distance metric for matching

    Returns:
        Dict[int, Dict]: For each k, contains matched archetype information across batches
    """
    print(f"\n=== Matching {modality_name} archetypes across batches for all k values ===")

    batches = list(batch_archetypes.keys())
    if not batches:
        raise ValueError("No batches found in batch_archetypes")

    reference_batch = batches[0]
    print(f"Using {reference_batch} as reference batch")

    # Get all k values available across all batches
    all_k_values = set()
    for batch in batches:
        all_k_values.update(batch_archetypes[batch]["k_values"])
    all_k_values = sorted(list(all_k_values))

    # For each k, match archetypes across batches
    k_matched_results = {}

    for k in all_k_values:
        print(f"\nMatching archetypes for k={k}...")

        # Check if all batches have this k value
        valid_batches = []
        k_proportions = {}
        k_archetypes = {}

        for batch in batches:
            if k in batch_archetypes[batch]["k_values"]:
                k_idx = batch_archetypes[batch]["k_values"].index(k)
                k_proportions[batch] = batch_proportions[batch][k_idx]
                k_archetypes[batch] = batch_archetypes[batch]["archetype_list"][k_idx]
                valid_batches.append(batch)

        if len(valid_batches) < len(batches):
            print(
                f"  k={k} not available for all batches (only {len(valid_batches)}/{len(batches)}), skipping"
            )
            continue

        # Match archetypes across batches for this k

        best_configs = {}
        total_matching_cost = 0.0

        for batch in valid_batches:
            if batch == reference_batch:
                # Reference batch uses identity order
                best_batch_order = np.arange(len(k_proportions[batch]))
                best_ref_order = np.arange(len(k_proportions[batch]))
                best_total_cost = 0.0
            else:
                # Find best matching order against reference batch
                reference_proportions = k_proportions[reference_batch]
                batch_proportions_df = k_proportions[batch]

                (
                    _,
                    best_total_cost,
                    best_ref_order,
                    best_batch_order,
                ) = find_best_pair_by_row_matching(
                    [reference_proportions],
                    [batch_proportions_df],
                    metric=metric,
                )

                total_matching_cost += best_total_cost

            # Store matched configuration for this batch
            best_configs[batch] = {
                "proportions": k_proportions[batch],
                "archetypes": k_archetypes[batch],
                "archetype_order": best_batch_order,
                "reference_order": best_ref_order,
                "matching_cost": best_total_cost,
            }

        # Store results for this k
        k_matched_results[k] = {
            "matched_configs": best_configs,
            "total_within_modality_cost": total_matching_cost,
            "reference_batch": reference_batch,
            "valid_batches": valid_batches,
        }

        avg_cost = total_matching_cost / max(1, len(valid_batches) - 1)
        print(f"  k={k}: average within-modality matching cost = {avg_cost:.4f}")

    print(f"Completed batch matching for {len(k_matched_results)} k values in {modality_name}")
    return k_matched_results


def compute_global_modality_proportions(
    batch_proportions: Dict[str, List[pd.DataFrame]],
    modality_name: str = "RNA",
    batch_archetypes: Dict[str, Dict] = None,
) -> Dict[int, pd.DataFrame]:
    """
    Compute global (aggregated across batches) cell type proportions for each k value.

    Args:
        batch_proportions: Dictionary containing proportion DataFrames for each batch
        modality_name: Name of modality for logging
        batch_archetypes: Dictionary containing batch archetype information (used to get actual k values)

    Returns:
        Dict[int, pd.DataFrame]: For each k, contains global proportions across all batches
    """
    print(f"\n=== Computing global proportions for {modality_name} ===")

    # Get actual k values from batch_archetypes if available, otherwise use old method
    if batch_archetypes is not None:
        # Get all k values available across all batches
        all_k_values = set()
        for batch in batch_archetypes:
            all_k_values.update(batch_archetypes[batch]["k_values"])
        all_k_values = sorted(list(all_k_values))
    else:
        # Fallback to old method (0-based indexing converted to 1-based)
        all_k_values = set()
        for batch in batch_proportions:
            all_k_values.update(range(len(batch_proportions[batch])))
        all_k_values = [k_idx + 1 for k_idx in sorted(list(all_k_values))]

    global_proportions = {}

    for k in all_k_values:
        print(f"Computing global proportions for k={k}...")

        # Collect proportions from all batches for this k
        batch_props_for_k = []
        batch_weights = []

        for batch in batch_proportions:
            # Find the correct index for this k value
            if batch_archetypes is not None:
                # Use actual k values from batch_archetypes
                if k in batch_archetypes[batch]["k_values"]:
                    k_idx = batch_archetypes[batch]["k_values"].index(k)
                    if k_idx < len(batch_proportions[batch]):
                        batch_prop = batch_proportions[batch][k_idx]
                        batch_props_for_k.append(batch_prop)
                        # Weight by number of cells (sum of proportions gives total cells)
                        batch_weights.append(batch_prop.sum().sum())
            else:
                # Fallback to old method (k is 1-based, convert to 0-based index)
                k_idx = k - 1
                if k_idx < len(batch_proportions[batch]):
                    batch_prop = batch_proportions[batch][k_idx]
                    batch_props_for_k.append(batch_prop)
                    # Weight by number of cells (sum of proportions gives total cells)
                    batch_weights.append(batch_prop.sum().sum())

        if not batch_props_for_k:
            print(f"  No batches have k={k}, skipping")
            continue

        # Combine all batch proportions into a single global proportion matrix
        # Average the proportions weighted by batch size
        total_weight = sum(batch_weights)
        global_prop = None

        for batch_prop, weight in zip(batch_props_for_k, batch_weights):
            weighted_prop = batch_prop * (weight / total_weight)
            if global_prop is None:
                global_prop = weighted_prop.copy()
            else:
                global_prop += weighted_prop

        global_proportions[k] = global_prop
        print(f"  Global proportions for k={k}: {global_prop.shape}")

    print(f"Computed global proportions for k values: {list(global_proportions.keys())}")
    return global_proportions


def compute_global_modality_proportions_from_matched(
    k_matched_results: Dict[int, Dict],
    modality_name: str = "RNA",
) -> Dict[int, pd.DataFrame]:
    """
    Compute global (aggregated across batches) cell type proportions using ALIGNED batch proportions.

    This function uses the proportions from within-modality matching results, where each batch's
    archetypes have been reordered to match the reference batch. This ensures the global proportions
    reflect the aligned archetype ordering for cross-modal comparison.

    Args:
        k_matched_results: Dictionary containing matched archetype configurations for each k value
        modality_name: Name of modality for logging

    Returns:
        Dict[int, pd.DataFrame]: For each k, contains global proportions across all aligned batches
    """
    print(f"\n=== Computing global proportions for {modality_name} ===")

    global_proportions = {}

    for k, k_results in k_matched_results.items():
        print(f"Computing global proportions for k={k}...")

        matched_configs = k_results["matched_configs"]
        k_results["reference_batch"]

        # Collect aligned proportions from all batches
        batch_props_for_k = []
        batch_weights = []

        for batch, config in matched_configs.items():
            # Get the proportions DataFrame (already aligned via archetype_order)
            batch_prop = config["proportions"]
            archetype_order = config["archetype_order"]

            # Reorder the proportions to match the reference batch ordering
            # The archetype_order tells us how to reorder this batch to match reference
            aligned_prop = batch_prop.iloc[archetype_order].reset_index(drop=True)

            batch_props_for_k.append(aligned_prop)
            # Weight by total cell count (sum of all proportions)
            batch_weights.append(aligned_prop.sum().sum())

        if not batch_props_for_k:
            print(f"  No batches have k={k}, skipping")
            continue

        # Combine all aligned batch proportions into a single global proportion matrix
        # Average the proportions weighted by batch size
        total_weight = sum(batch_weights)
        global_prop = None

        for batch_prop, weight in zip(batch_props_for_k, batch_weights):
            weighted_prop = batch_prop * (weight / total_weight)
            if global_prop is None:
                global_prop = weighted_prop.copy()
            else:
                # Ensure columns align (fill missing cell types with 0)
                for col in weighted_prop.columns:
                    if col not in global_prop.columns:
                        global_prop[col] = 0.0
                for col in global_prop.columns:
                    if col not in weighted_prop.columns:
                        weighted_prop[col] = 0.0
                global_prop = global_prop.add(weighted_prop, fill_value=0)

        global_proportions[k] = global_prop
        print(f"  Global proportions for k={k}: {global_prop.shape}")

    print(f"Computed global proportions for k values: {list(global_proportions.keys())}")
    return global_proportions


def find_optimal_k_across_modalities(
    adata_list: list,
    batch_key: str = "batch",
    embedding_names: list = None,
    min_k: int = 8,
    max_k: int = 12,
    step_size: int = 1,
    converge: float = 1e-5,
    modality_names: list = None,
    metric: str = "cosine",
    extreme_filtering_threshold: float = 0.5,
    plot_flag: bool = True,
) -> tuple:
    """
    Find the optimal k value and create unified archetype representations for all modalities.

    This function combines cross-modal k optimization with final archetype representation
    creation, eliminating the need for a separate processing step.

    Process:
    1. Generate archetypes for each batch in each modality (for all k values)
    2. Match archetypes within each modality across batches - QUALITY CONTROL PHASE
    3. Compute global proportions for each modality across all batches
    4. Cross-modal comparison using global proportions to find optimal k
    5. Create unified archetype representations using optimal k
    6. Identify extreme archetypes for each modality
    7. Filter extreme archetypes based on cross-modal matching quality

    Args:
        adata_list: List of AnnData objects for different modalities (will be modified in-place)
        batch_key: Key in adata.obs containing batch information
        embedding_names: List of embedding names to use for each modality
        min_k: Minimum number of archetypes to test
        max_k: Maximum number of archetypes to test
        step_size: Step size for archetype number testing
        converge: Convergence threshold for early stopping (used for per-batch generation)
        modality_names: List of modality names for logging
        metric: Distance metric for matching
        extreme_filtering_threshold: Maximum allowed JS divergence for keeping extreme archetypes (0-1)
        plot_flag: Whether to generate archetype proportion plots

    Returns:
        tuple: (optimal_k, processed_adata_list) - AnnData objects are modified in-place and returned
    """
    print(f"\n=== Finding optimal k across {len(adata_list)} modalities ===")

    if embedding_names is None:
        embedding_names = ["X_pca"] * len(adata_list)
    if modality_names is None:
        modality_names = [f"Modality{i}" for i in range(len(adata_list))]

    # Phase 1: Generate archetypes for all modalities
    print("\n" + "=" * 60)
    print("PHASE 1: GENERATING ARCHETYPES FOR ALL MODALITIES")
    print("=" * 60)

    all_batch_archetypes = []
    all_batch_proportions = []

    for i, (adata, embedding_name, modality_name) in enumerate(
        zip(adata_list, embedding_names, modality_names)
    ):
        print(f"\nGenerating archetypes for {modality_name}...")

        # Generate archetypes per batch for this modality
        batch_archetypes = generate_archetypes_per_batch(
            adata=adata,
            batch_key=batch_key,
            embedding_name=embedding_name,
            min_k=min_k,
            max_k=max_k,
            step_size=step_size,
            converge=converge,
            modality_name=modality_name,
            plot_flag=plot_flag,
        )

        # Compute proportions
        batch_proportions = compute_batch_archetype_proportions(
            batch_archetypes=batch_archetypes,
            adata=adata,
            batch_key=batch_key,
            modality_name=modality_name,
            plot_flag=plot_flag,
        )

        all_batch_archetypes.append(batch_archetypes)
        all_batch_proportions.append(batch_proportions)

    # Phase 2: Match archetypes within each modality for all k values - QUALITY CONTROL
    print("\n" + "=" * 60)
    print("PHASE 2: WITHIN-MODALITY BATCH MATCHING - QUALITY CONTROL")
    print("=" * 60)

    all_k_matched_results = []
    within_modality_quality_scores = {}

    # Collect data for combined plotting
    combined_batch_proportions = {}
    combined_batch_archetypes = {}
    all_cell_types = set()

    for modality_idx, (batch_archetypes, batch_proportions, modality_name) in enumerate(
        zip(all_batch_archetypes, all_batch_proportions, modality_names)
    ):
        print(f"\nQuality control for {modality_name} within-modality batch matching...")
        k_matched_results = match_archetypes_across_batches_for_all_k(
            batch_archetypes=batch_archetypes,
            batch_proportions=batch_proportions,
            modality_name=modality_name,
            metric=metric,
            plot_flag=plot_flag,
        )

        # Collect data for combined plotting instead of plotting separately
        for batch, proportions in batch_proportions.items():
            combined_key = f"{modality_name}_batch_{batch}"
            combined_batch_proportions[combined_key] = proportions
            combined_batch_archetypes[combined_key] = batch_archetypes[batch]

        # Collect cell types from original adata (not from stored batch_adata)
        cell_types = set(adata.obs.get("cell_types", adata.obs.get("major_cell_types", [])))
        all_cell_types.update(cell_types)

        all_k_matched_results.append(k_matched_results)

        # Store quality scores for each k value
        quality_scores = {}
        for k, results in k_matched_results.items():
            quality_scores[k] = {
                "total_within_modality_cost": results["total_within_modality_cost"],
                "num_valid_batches": len(results["valid_batches"]),
                "avg_cost": results["total_within_modality_cost"]
                / max(1, len(results["valid_batches"]) - 1),
            }

        within_modality_quality_scores[modality_name] = quality_scores
        print(f"✅ {modality_name} quality control completed for {len(k_matched_results)} k values")

    # Create combined plot with all modalities and batches
    if plot_flag and combined_batch_proportions:
        print("\n=== Creating combined before matching proportions plot ===")
        cell_type_colors = get_cell_type_colors(sorted(list(all_cell_types)))

        plot_archetype_proportions_before_after_matching(
            batch_proportions_before=combined_batch_proportions,
            batch_proportions_after=None,  # Before matching
            k_matched_results=None,  # Will auto-select first available k
            batch_archetypes=combined_batch_archetypes,
            optimal_k=None,
            modality_names=list(combined_batch_proportions.keys()),
            cell_type_colors=cell_type_colors,
            plot_flag=plot_flag,
            save_plots=False,
        )

    # Phase 3: Compute global proportions for cross-modal matching
    # IMPORTANT: Use ALIGNED proportions from within-modality matching, not original proportions
    print("\n" + "=" * 60)
    print("PHASE 3: COMPUTING GLOBAL PROPORTIONS FOR CROSS-MODAL MATCHING")
    print("(Using aligned proportions from within-modality matching)")
    print("=" * 60)

    all_global_proportions = []
    for modality_idx, (k_matched_results, batch_archetypes, modality_name) in enumerate(
        zip(all_k_matched_results, all_batch_archetypes, modality_names)
    ):
        global_proportions = compute_global_modality_proportions_from_matched(
            k_matched_results=k_matched_results,
            modality_name=modality_name,
        )
        all_global_proportions.append(global_proportions)

    # Phase 4: Find common k values across all modalities
    print("\n" + "=" * 60)
    print("PHASE 4: FINDING COMMON K VALUES FOR CROSS-MODAL COMPARISON")
    print("=" * 60)

    # Find common k values from global proportions (these already passed quality control)
    common_k_values = None
    for modality_idx, (global_proportions, modality_name) in enumerate(
        zip(all_global_proportions, modality_names)
    ):
        modality_k_values = set(global_proportions.keys())
        print(
            f"{modality_name} available k values from global proportions: {sorted(modality_k_values)}"
        )

        if common_k_values is None:
            common_k_values = modality_k_values
        else:
            common_k_values = common_k_values.intersection(modality_k_values)

    common_k_values = sorted(list(common_k_values))
    print(f"K values available for cross-modal comparison: {common_k_values}")

    if not common_k_values:
        raise ValueError("No common k values found across all modalities!")

    # Phase 5: Cross-modal comparison using global proportions
    print("\n" + "=" * 60)
    print("PHASE 5: CROSS-MODAL ARCHETYPE COMPARISON USING GLOBAL PROPORTIONS")
    print("=" * 60)

    best_k = None
    best_cross_modal_cost = float("inf")
    k_comparison_results = {}
    cross_modal_orderings = {}  # Store cross-modal ordering information

    for k in common_k_values:
        print(f"\nTesting k={k} for cross-modal compatibility using global proportions...")

        # Get the global proportions for this k from each modality
        modality_global_proportions = []

        for modality_idx, (global_proportions, modality_name) in enumerate(
            zip(all_global_proportions, modality_names)
        ):
            if k in global_proportions:
                modality_global_proportions.append(global_proportions[k])
                print(
                    f"  {modality_name} global proportions for k={k}: {global_proportions[k].shape}"
                )
            else:
                print(f"  k={k} not available in {modality_name}, skipping")
                break

        if len(modality_global_proportions) == len(modality_names):
            # Compare archetype sets across modalities using global proportions
            cross_modal_cost = 0.0
            comparison_count = 0
            k_cross_modal_orders = {}

            # Compare each modality against the first (reference) modality for cross-modal ordering
            reference_modality_idx = 0
            for j in range(1, len(modality_global_proportions)):
                mod1_proportions = modality_global_proportions[reference_modality_idx]
                mod2_proportions = modality_global_proportions[j]

                _, cost, ref_order, target_order = find_best_pair_by_row_matching(
                    [mod1_proportions],
                    [mod2_proportions],
                    metric=metric,
                )

                # Store the cross-modal ordering (how to reorder modality j to match modality 0)
                k_cross_modal_orders[j] = target_order

                cross_modal_cost += cost
                comparison_count += 1
                print(
                    f"  {modality_names[reference_modality_idx]} vs {modality_names[j]}: cost = {cost:.4f}"
                )
                print(f"    Cross-modal order for {modality_names[j]}: {target_order}")

            if comparison_count > 0 and cross_modal_cost != float("inf"):
                avg_cross_modal_cost = cross_modal_cost / comparison_count

                # Normalize by number of archetypes to prevent bias toward smaller k values
                # This ensures we select based on average matching quality per archetype pair
                normalized_cross_modal_cost = avg_cross_modal_cost / k

                print(f"  k={k}: average cross-modal cost = {avg_cross_modal_cost:.4f}")
                print(
                    f"  k={k}: normalized cost (per archetype) = {normalized_cross_modal_cost:.4f}"
                )

                k_comparison_results[k] = {
                    "cross_modal_cost": avg_cross_modal_cost,
                    "normalized_cross_modal_cost": normalized_cross_modal_cost,
                    "comparison_count": comparison_count,
                }

                # Store cross-modal orderings for this k
                cross_modal_orderings[k] = k_cross_modal_orders

                # Use normalized cost for selection to avoid bias toward smaller k values
                if normalized_cross_modal_cost < best_cross_modal_cost:
                    best_cross_modal_cost = normalized_cross_modal_cost
                    best_k = k
                    print(
                        f"  -> New best k={k} with normalized cost {normalized_cross_modal_cost:.4f}"
                    )
            else:
                print(f"  k={k}: failed cross-modal comparison")
        else:
            print(f"  k={k}: not available in all modalities")

    if best_k is None:
        best_k = common_k_values[0]
        print(f"Fallback: using first common k={best_k}")

    print(
        f"\n🎯 SELECTED OPTIMAL K={best_k} with normalized cross-modal cost {best_cross_modal_cost:.4f}"
    )

    # Get the cross-modal orderings for the selected k
    selected_cross_modal_orders = cross_modal_orderings.get(best_k, {})
    if selected_cross_modal_orders:
        print("Cross-modal archetype orderings:")
        for modality_idx, order in selected_cross_modal_orders.items():
            print(f"  {modality_names[modality_idx]} -> {order}")

    # Store cell type colors for later plotting (after quality filtering)
    # Use the first modality's adata (already in adata_list[0])
    cell_types = sorted(
        list(
            set(adata_list[0].obs.get("cell_types", adata_list[0].obs.get("major_cell_types", [])))
        )
    )
    cell_type_colors = get_cell_type_colors(cell_types)

    # Phase 6: Create unified archetype representations for each modality
    print("\n" + "=" * 60)
    print("PHASE 6: CREATING UNIFIED ARCHETYPE REPRESENTATIONS")
    print("=" * 60)

    for modality_idx, (adata, modality_name) in enumerate(zip(adata_list, modality_names)):
        print(f"\nProcessing {modality_name} unified representation...")

        # Get the optimal k configuration for this modality
        k_matched_results = all_k_matched_results[modality_idx]
        batch_archetypes = all_batch_archetypes[modality_idx]

        if best_k not in k_matched_results:
            raise ValueError(f"Optimal k={best_k} not found in {modality_name} results")

        optimal_k_results = k_matched_results[best_k]
        matched_configs = optimal_k_results["matched_configs"]
        reference_batch = optimal_k_results["reference_batch"]

        # Create matched_archetypes structure
        matched_archetypes = {"reference_batch": reference_batch}
        for batch, config_ in matched_configs.items():
            matched_archetypes[batch] = {
                "archetype_order": config_["archetype_order"],
                "proportions": config_["proportions"],
                "archetypes": config_["archetypes"],
                # archetype_activities available in batch_archetypes, accessed directly in create_unified_archetype_representation
                "matching_cost": config_["matching_cost"],
            }

        # Create unified archetype representation
        adata = create_unified_archetype_representation(
            adata=adata,
            matched_archetypes=matched_archetypes,
            batch_archetypes=batch_archetypes,
            batch_key=batch_key,
            modality_name=modality_name,
            optimal_k=best_k,
        )

        # Update the adata in the list
        adata_list[modality_idx] = adata

        # Store quality control information and cross-modal results
        if "pipeline_metadata" not in adata.uns:
            adata.uns["pipeline_metadata"] = {}
        if "archetype_generation" not in adata.uns["pipeline_metadata"]:
            adata.uns["pipeline_metadata"]["archetype_generation"] = {}

        # Store within-modality quality control results
        adata.uns["pipeline_metadata"]["archetype_generation"]["within_modality_quality"] = (
            within_modality_quality_scores[modality_name]
        )

        # Store cross-modal comparison results
        adata.uns["pipeline_metadata"]["archetype_generation"][
            "cross_modal_comparison"
        ] = k_comparison_results

        # Store the selected cross-modal ordering for this modality
        if modality_idx in selected_cross_modal_orders:
            adata.uns["pipeline_metadata"]["archetype_generation"]["cross_modal_ordering"] = (
                selected_cross_modal_orders[modality_idx]
            )

        # Store global proportions for reference
        adata.uns["global_archetype_proportions"] = all_global_proportions[modality_idx]

        # Apply cross-modal archetype ordering if needed (for non-reference modalities)
        if modality_idx in selected_cross_modal_orders:
            cross_modal_order = selected_cross_modal_orders[modality_idx]
            print(f"Applying cross-modal ordering {cross_modal_order} to {modality_name}")

            # Reorder the archetype vectors columns according to cross-modal matching
            original_vectors = adata.obsm["archetype_vec"].values
            reordered_vectors = original_vectors[:, cross_modal_order]

            # Update the archetype vectors
            adata.obsm["archetype_vec"] = pd.DataFrame(
                reordered_vectors,
                index=adata.obs.index,
                columns=[str(i) for i in range(reordered_vectors.shape[1])],
            )

            # Recalculate archetype labels based on reordered vectors
            adata.obs["archetype_label"] = pd.Categorical(np.argmax(reordered_vectors, axis=1))

            # Update batch-specific archetype positions to match cross-modal reordering
            for batch in adata.obs[batch_key].unique():
                batch_key_name = f"batch_archetypes_{batch}"
                if batch_key_name in adata.uns:
                    original_batch_positions = adata.uns[batch_key_name]
                    adata.uns[batch_key_name] = original_batch_positions[cross_modal_order]
                    print(f"  Reordered {batch_key_name} with order {cross_modal_order}")

            print(
                f"Updated archetype positions in uns to match cross-modal reordering {cross_modal_order}"
            )

        # Ensure archetype labels are properly updated for cases without cross-modal reordering
        if modality_idx not in selected_cross_modal_orders:
            update_archetype_labels(adata)
            print(f"Updated archetype labels for {modality_name} (no cross-modal reordering)")

        # Identify extreme archetypes
        print(f"Identifying extreme {modality_name} archetypes...")
        archetype_vectors = adata.obsm["archetype_vec"]
        extreme_mask, _, _ = identify_extreme_archetypes_balanced(
            archetype_vectors, adata=adata, percentile=90
        )
        adata.obs["is_extreme_archetype"] = extreme_mask

        n_extreme = extreme_mask.sum()
        print(f"Identified {n_extreme} extreme archetype cells ({n_extreme/len(adata)*100:.1f}%)")

        print(f"✅ {modality_name} unified archetype representation complete")

        # Ensure the updated adata is in the list (redundant but safe)
        adata_list[modality_idx] = adata
        adata_list[modality_idx] = add_matched_archetype_weight(adata_list[modality_idx])

    # Collect after-matching proportions per batch for plotting
    combined_batch_proportions_after = {}
    if plot_flag:
        for modality_idx, (modality_name, k_matched_results) in enumerate(
            zip(modality_names, all_k_matched_results)
        ):
            if best_k not in k_matched_results:
                continue

            optimal_k_results = k_matched_results[best_k]
            matched_configs = optimal_k_results["matched_configs"]

            # Extract proportions for each batch after WITHIN-MODALITY matching
            # Apply archetype_order to align batches within the modality (NOT cross-modal order)
            for batch, config_ in matched_configs.items():
                combined_key = f"{modality_name}_batch_{batch}"
                if combined_key not in combined_batch_archetypes:
                    continue
                batch_k_values = combined_batch_archetypes[combined_key]["k_values"]
                if best_k not in batch_k_values:
                    continue
                k_idx = batch_k_values.index(best_k)
                list_len = len(batch_k_values)
                if combined_key not in combined_batch_proportions_after:
                    combined_batch_proportions_after[combined_key] = [None] * list_len

                # Get original proportions and the matching order
                proportions_original = config_["proportions"].copy()
                within_modality_order = config_["archetype_order"]
                # Convert numpy array to list for reliable pandas indexing
                order_list = (
                    within_modality_order.tolist()
                    if hasattr(within_modality_order, "tolist")
                    else list(within_modality_order)
                )

                # Apply reordering - iloc selects rows by integer position
                proportions_after_df = proportions_original.iloc[order_list].reset_index(drop=True)
                combined_batch_proportions_after[combined_key][k_idx] = proportions_after_df

        # Create combined before/after plot
        if combined_batch_proportions_after and combined_batch_proportions:
            print("\n=== Creating combined before/after matching proportions plot ===")
            cell_type_colors = get_cell_type_colors(sorted(list(all_cell_types)))

            plot_archetype_proportions_before_after_matching(
                batch_proportions_before=combined_batch_proportions,
                batch_proportions_after=combined_batch_proportions_after,
                k_matched_results=None,
                batch_archetypes=combined_batch_archetypes,
                optimal_k=best_k,
                modality_names=list(combined_batch_proportions.keys()),
                cell_type_colors=cell_type_colors,
                plot_flag=plot_flag,
                save_plots=False,
                before_label="Before Matching",
                after_label="After Within-Modality Matching",
                title_prefix="Archetype Cell Type Proportions",
            )

            # Create additional plot showing cross-modal aligned proportions
            # For consistent comparison, we use the SAME proportions as within-modality
            # but apply cross-modal ordering for non-reference modalities
            combined_batch_proportions_final = {}
            for modality_idx, (modality_name, k_matched_results) in enumerate(
                zip(modality_names, all_k_matched_results)
            ):
                if best_k not in k_matched_results:
                    continue

                optimal_k_results = k_matched_results[best_k]
                matched_configs = optimal_k_results["matched_configs"]
                cross_modal_order = selected_cross_modal_orders.get(modality_idx)

                for batch, config_ in matched_configs.items():
                    combined_key = f"{modality_name}_batch_{batch}"
                    if combined_key not in combined_batch_archetypes:
                        continue
                    batch_k_values = combined_batch_archetypes[combined_key]["k_values"]
                    if best_k not in batch_k_values:
                        continue
                    k_idx = batch_k_values.index(best_k)
                    list_len = len(batch_k_values)
                    if combined_key not in combined_batch_proportions_final:
                        combined_batch_proportions_final[combined_key] = [None] * list_len

                    # Start with original proportions
                    proportions_df = config_["proportions"].copy()
                    # Apply within-modality ordering first
                    within_modality_order = config_["archetype_order"]
                    order_list = (
                        within_modality_order.tolist()
                        if hasattr(within_modality_order, "tolist")
                        else list(within_modality_order)
                    )
                    proportions_df = proportions_df.iloc[order_list].reset_index(drop=True)
                    # Then apply cross-modal ordering for non-reference modalities
                    if cross_modal_order is not None:
                        cross_order_list = (
                            cross_modal_order.tolist()
                            if hasattr(cross_modal_order, "tolist")
                            else list(cross_modal_order)
                        )
                        proportions_df = proportions_df.iloc[cross_order_list].reset_index(
                            drop=True
                        )
                    combined_batch_proportions_final[combined_key][k_idx] = proportions_df

            if combined_batch_proportions_final:
                print("\n=== Creating within- vs cross-modal matching proportions plot ===")
                plot_archetype_proportions_before_after_matching(
                    batch_proportions_before=combined_batch_proportions_after,
                    batch_proportions_after=combined_batch_proportions_final,
                    batch_archetypes=combined_batch_archetypes,
                    optimal_k=best_k,
                    modality_names=list(combined_batch_proportions_after.keys()),
                    cell_type_colors=cell_type_colors,
                    plot_flag=plot_flag,
                    save_plots=False,
                    before_label="After Within-Modality Matching",
                    after_label="After Cross-Modal Matching",
                    title_prefix="Archetype Cell Type Proportions",
                )

    # Phase 7: Filter extreme archetypes based on cross-modal matching quality
    print(f"\n{'='*60}")
    print("PHASE 7: FILTERING EXTREME ARCHETYPES BY CROSS-MODAL QUALITY")
    print(f"{'='*60}")

    if len(adata_list) >= 2:
        # Apply filtering between first two modalities (typically RNA and Protein)
        filter_extreme_archetypes_by_cross_modal_quality(
            adata_1=adata_list[0],
            adata_2=adata_list[1],
            cell_type_key="cell_types",
            proportion_threshold=extreme_filtering_threshold,
            verbose=True,
        )
        print(
            f"✅ Extreme archetype filtering completed between {modality_names[0]} and {modality_names[1]}"
        )

        # Create cross-modal comparison plot with quality information
        if plot_flag and len(all_global_proportions) >= 2:
            print("\n=== Creating cross-modal comparison with poor quality archetype marking ===")
            archetype_quality_dict = adata_list[0].uns.get("archetype_quality", None)

            # Get quality scores that determine the poor/good quality assessment
            quality_scores = None
            quality_metadata = adata_list[0].uns.get("cross_modal_quality", None)
            if quality_metadata is not None and "archetype_quality_scores" in quality_metadata:
                quality_scores = quality_metadata["archetype_quality_scores"]
                # FIXED: Ensure quality_scores is a numpy array
                if not isinstance(quality_scores, np.ndarray):
                    quality_scores = np.array(quality_scores)
                threshold = quality_metadata.get("proportion_threshold", 0.3)
                print(f"Using Jensen-Shannon distances (quality scores) with threshold {threshold}")
                print(f"Quality scores: {[f'{score:.3f}' for score in quality_scores[:8]]}")
            else:
                print("No quality scores found, computing cosine distances as fallback")
                # Fallback to cosine distances if quality scores not available
                if "archetype_vec" in adata_list[0].obsm and "archetype_vec" in adata_list[1].obsm:
                    from sklearn.metrics.pairwise import cosine_distances

                    archetype_vec_1 = adata_list[0].obsm["archetype_vec"]
                    archetype_vec_2 = adata_list[1].obsm["archetype_vec"]

                    # Compute cosine distance matrix between archetypes
                    similarity_matrix = cosine_distances(archetype_vec_1, archetype_vec_2)

                    # Get minimum distance for each archetype (best cross-modal match)
                    quality_scores = np.min(similarity_matrix, axis=1)
                    print(f"Computed cosine similarity scores for {len(quality_scores)} archetypes")

            plot_cross_modal_archetype_comparison(
                global_proportions_mod1=all_global_proportions[0],
                global_proportions_mod2=all_global_proportions[1],
                optimal_k=best_k,
                modality_names=modality_names[:2],
                cell_type_colors=cell_type_colors,
                cross_modal_orders=cross_modal_orderings,
                archetype_quality_dict=archetype_quality_dict,
                similarity_scores=quality_scores,
                quality_threshold=(
                    quality_metadata.get("proportion_threshold", 0.3) if quality_metadata else None
                ),
                plot_flag=plot_flag,
                save_plots=False,
            )
    else:
        print("Skipping extreme archetype filtering - need at least 2 modalities")

    print(f"\n{'='*60}")
    print(f"🎯 CROSS-MODAL ARCHETYPE ANALYSIS COMPLETE WITH K={best_k}")
    print(f"{'='*60}")

    # Clean up temporary metadata used during archetype generation to reduce file size
    print("\nCleaning up temporary archetype generation metadata...")
    for adata in adata_list:
        if "pipeline_metadata" in adata.uns:
            if "archetype_generation" in adata.uns["pipeline_metadata"]:
                # Remove the archetype_generation metadata (contains matched_archetypes, proportions, etc.)
                del adata.uns["pipeline_metadata"]["archetype_generation"]
                print(f"  Removed pipeline_metadata.archetype_generation from adata")

    print("✅ Temporary metadata cleaned up - file size optimized")

    return adata_list


def validate_batch_archetype_consistency(
    adata_rna: AnnData,
    adata_prot: AnnData,
    batch_key: str = "batch",
) -> None:
    """
    Validate that batch-specific archetype generation produces consistent results.

    Args:
        adata_rna: RNA AnnData object with archetype vectors
        adata_prot: Protein AnnData object with archetype vectors
        batch_key: Key in obs containing batch information
    """
    print("\n=== Validating batch archetype consistency ===")

    for modality, adata in [("RNA", adata_rna), ("Protein", adata_prot)]:
        print(f"\n{modality} validation:")

        if "archetype_vec" not in adata.obsm:
            print(f"Warning: No archetype_vec found in {modality} data")
            continue

        archetype_vectors = adata.obsm["archetype_vec"]

        # Check archetype vector properties per batch
        for batch in adata.obs[batch_key].unique():
            batch_mask = adata.obs[batch_key] == batch
            batch_vectors = archetype_vectors[batch_mask]

            # Check that vectors sum to approximately 1 (within tolerance)
            vector_sums = batch_vectors.sum(axis=1)
            sum_check = np.allclose(vector_sums, 1.0, atol=1e-2)

            # Check for non-negative values
            non_negative_check = (batch_vectors >= -1e-6).all().all()

            # Check for reasonable distribution of dominant archetypes
            dominant_archetypes = np.argmax(batch_vectors.values, axis=1)
            unique_dominants = len(np.unique(dominant_archetypes))

            print(f"  Batch {batch}:")
            print(f"    - Cells: {batch_mask.sum()}")
            print(f"    - Vector sums ≈ 1.0: {sum_check}")
            print(f"    - All non-negative: {non_negative_check}")
            print(
                f"    - Unique dominant archetypes: {unique_dominants}/{archetype_vectors.shape[1]}"
            )

            if not sum_check:
                print(
                    f"    - Warning: Vector sum range: [{vector_sums.min():.3f}, {vector_sums.max():.3f}]"
                )

            if not non_negative_check:
                min_val = batch_vectors.values.min()
                print(f"    - Warning: Minimum value: {min_val:.6f}")

    # Cross-modal validation if both modalities have archetype vectors
    if "archetype_vec" in adata_rna.obsm and "archetype_vec" in adata_prot.obsm:
        print(f"\nCross-modal validation:")

        # Check that both modalities have same number of archetypes
        rna_n_archetypes = adata_rna.obsm["archetype_vec"].shape[1]
        prot_n_archetypes = adata_prot.obsm["archetype_vec"].shape[1]

        print(f"  - RNA archetypes: {rna_n_archetypes}")
        print(f"  - Protein archetypes: {prot_n_archetypes}")
        print(f"  - Archetype count match: {rna_n_archetypes == prot_n_archetypes}")

        if rna_n_archetypes != prot_n_archetypes:
            print("  - Warning: Archetype count mismatch between modalities!")

    print("\nValidation complete.")


def add_matched_archetype_weight(adata: sc.AnnData) -> sc.AnnData:
    """
    Adds the archetype significance to the adata object
    which is the weight of the most significant archetype for each cell, how much a cell is "extreme"
    in regards of how much of the weight of its most significant archetype

    Input:
    adata: sc.AnnData
        AnnData object with archetype_label in obs and archetype_vec in obsm
    Output:
    adata: sc.AnnData
        AnnData object with matched_archetype_weight in obs
    """
    if "archetype_label" not in adata.obs.columns:
        raise ValueError("archetype_label column not found in adata.obs")
    if "archetype_vec" not in adata.obsm.keys():
        raise ValueError("archetype_vec not found in adata.obsm")
    archetype_vec = adata.obsm["archetype_vec"]
    archetype_label = adata.obs["archetype_label"]
    matched_archetype_weight = np.take_along_axis(
        archetype_vec.values, np.array(archetype_label.values).astype(int)[:, None], axis=1
    )
    adata.obs["matched_archetype_weight"] = matched_archetype_weight
    return adata


def finalize_archetype_generation_with_visualizations(
    adata_rna: AnnData,
    adata_prot: AnnData,
    plot_flag: bool = True,
) -> None:
    """
    Finalize archetype generation process and create comprehensive visualizations.

    This function should be called after find_optimal_k_across_modalities to add
    the archetype proportion analysis and visualizations that were in the original code.

    Args:
        adata_rna: RNA AnnData object with archetype vectors
        adata_prot: Protein AnnData object with archetype vectors
        plot_flag: Whether to generate plots
    """
    print("\n" + "=" * 60)
    print("FINALIZING ARCHETYPE GENERATION WITH VISUALIZATIONS")
    print("=" * 60)

    # Visualize archetype proportions analysis
    visualize_archetype_proportions_analysis(adata_rna, adata_prot, plot_flag)

    # Create archetype AnnData objects for visualization
    if not plot_flag:
        return
    adata_archetype_rna = AnnData(adata_rna.obsm["archetype_vec"])
    adata_archetype_prot = AnnData(adata_prot.obsm["archetype_vec"])
    adata_archetype_rna.obs = adata_rna.obs.copy()
    adata_archetype_prot.obs = adata_prot.obs.copy()
    adata_archetype_rna.obs_names = adata_rna.obs_names
    adata_archetype_prot.obs_names = adata_prot.obs_names

    # Plot archetype visualizations
    from arcadia.plotting.archetypes import plot_archetype_visualizations
    from arcadia.plotting.general import safe_mlflow_log_figure

    # First do the standard archetype visualizations
    plot_archetype_visualizations(
        adata_archetype_rna, adata_archetype_prot, adata_rna, adata_prot, max_cells=2000
    )

    # Visualize cells with extreme archetype assignment using the boolean columns in obs
    # Visualize all cells in UMAP, highlight extreme archetypes with 'x' marker, color by cell_types

    print("Creating extreme archetype visualizations with consistent colors...")
    adata_rna_to_plot = adata_rna.copy()
    adata_prot_to_plot = adata_prot.copy()
    sc.pp.subsample(adata_rna_to_plot, n_obs=min(2000, adata_rna.n_obs))
    sc.pp.subsample(adata_prot_to_plot, n_obs=min(2000, adata_prot.n_obs))

    # Determine and set correct color palette
    from arcadia.assets.color_palletes import synthetic_palette, tonsil_palette

    dataset_name = adata_rna.uns.get("dataset_name", "").lower()
    all_cell_types = set(adata_rna.obs["cell_types"].unique()) | set(
        adata_prot.obs["cell_types"].unique()
    )

    if "tonsil" in dataset_name or "maxfuse_tonsil" in dataset_name:
        palette = tonsil_palette
        print(f"Using tonsil_palette for dataset: {dataset_name}")
    elif (
        "cite_seq" in dataset_name
        or "cite" in dataset_name
        or "synthetic" in dataset_name
        or "spleen" in dataset_name
    ):
        palette = synthetic_palette
        print(f"Using synthetic_palette for dataset: {dataset_name}")
    else:
        synthetic_matches = sum(1 for ct in all_cell_types if ct in synthetic_palette)
        tonsil_matches = sum(1 for ct in all_cell_types if ct in tonsil_palette)

        if synthetic_matches > tonsil_matches and synthetic_matches > 0:
            palette = synthetic_palette
            print(
                f"Inferred synthetic_palette from cell types (matched {synthetic_matches}/{len(all_cell_types)})"
            )
        elif tonsil_matches > 0:
            palette = tonsil_palette
            print(
                f"Inferred tonsil_palette from cell types (matched {tonsil_matches}/{len(all_cell_types)})"
            )
        else:
            palette = None
            print(f"No matching palette found, using default colors")

    # Apply palette to both adata objects
    if palette is not None:
        cell_types_list = sorted(list(all_cell_types))
        colors = [palette.get(ct, "#808080") for ct in cell_types_list]
        adata_rna_to_plot.uns["cell_types_colors"] = colors
        adata_prot_to_plot.uns["cell_types_colors"] = colors

    # Create combined figure with 2 subplots - extreme archetypes by cell type
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

    # RNA plot
    sc.pl.umap(
        adata_rna_to_plot,
        color="cell_types",
        ax=axes1[0],
        show=False,
        size=20,
        legend_loc="on data",
        title="RNA extreme archetypes highlighted",
    )
    # Overlay extreme archetypes as 'x'
    extreme_mask_rna = adata_rna_to_plot.obs["is_extreme_archetype"].values.astype(bool)
    umap_rna = adata_rna_to_plot.obsm["X_umap"]
    axes1[0].scatter(
        umap_rna[extreme_mask_rna, 0],
        umap_rna[extreme_mask_rna, 1],
        marker="x",
        s=100,
        c="black",
        edgecolor="black",
        linewidth=1.5,
        label="Extreme archetype",
    )
    axes1[0].legend()
    axes1[0].set_xticks([])
    axes1[0].set_yticks([])
    axes1[0].grid(False)

    # Protein plot
    sc.pl.umap(
        adata_prot_to_plot,
        color="cell_types",
        ax=axes1[1],
        show=False,
        size=20,
        legend_loc="on data",
        title="Protein extreme archetypes highlighted",
    )
    # Overlay extreme archetypes as 'x'
    extreme_mask_prot = adata_prot_to_plot.obs["is_extreme_archetype"].values.astype(bool)
    umap_prot = adata_prot_to_plot.obsm["X_umap"]
    axes1[1].scatter(
        umap_prot[extreme_mask_prot, 0],
        umap_prot[extreme_mask_prot, 1],
        marker="x",
        s=100,
        c="black",
        edgecolor="black",
        linewidth=1.5,
        label="Extreme archetype",
    )
    axes1[1].legend()
    axes1[1].set_xticks([])
    axes1[1].set_yticks([])
    axes1[1].grid(False)

    plt.tight_layout()
    safe_mlflow_log_figure(fig1, "archetype_extreme_cells_by_type.pdf")
    plt.show()

    # Create matched archetype weight figure
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    sc.pl.umap(
        adata_rna_to_plot,
        color="matched_archetype_weight",
        size=40,
        ax=axes2[0],
        show=False,
        cmap="viridis",
        title="RNA matched archetype weight",
    )
    axes2[0].set_xticks([])
    axes2[0].set_yticks([])
    axes2[0].grid(False)

    sc.pl.umap(
        adata_prot_to_plot,
        color="matched_archetype_weight",
        size=40,
        ax=axes2[1],
        show=False,
        cmap="viridis",
        title="Protein matched archetype weight",
    )
    axes2[1].set_xticks([])
    axes2[1].set_yticks([])
    axes2[1].grid(False)

    plt.tight_layout()
    safe_mlflow_log_figure(fig2, "archetype_matched_weights.pdf")
    plt.show()

    print("✅ Archetype visualizations completed")
