"""Utilities for comparing ARCADIA model performance against baseline models."""

import os
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

from arcadia.training import metrics as mtrc


def align_data(adata_arcadia, adata_other, modality, other_model_name):
    """
    Align two AnnData objects by finding common indices and features.

    Args:
        adata_arcadia: ARCADIA AnnData object
        adata_other: Other model AnnData object
        modality: "rna" or "protein"
        other_model_name: Name of the other model (for logging)

    Returns:
        tuple: (aligned_adata_arcadia, aligned_adata_other, common_indices)
    """
    obs_type = "gene_ids" if modality == "rna" else "protein_ids"
    # find common indices between adata_arcadia and adata_other
    common_indices = np.intersect1d(adata_arcadia.obs_names, adata_other.obs_names)
    # subset both adata_arcadia and adata_other to the common indices
    print(
        "num common indices",
        len(common_indices),
        "num in other but not in arcadia",
        len(set(adata_other.obs_names) - set(common_indices)),
        "num in arcadia but not in other",
        len(set(adata_arcadia.obs_names) - set(common_indices)),
    )
    if len(common_indices) == 0:
        # Fallback: force match by taking first N cells from other model
        adata_other = adata_other[: len(adata_arcadia)]
        adata_other.obs.index = adata_arcadia.obs.index
        common_indices = adata_arcadia.obs_names
        print("Warning: no common indices, forcing match by taking first N cells from other model")
    adata_other = adata_other[common_indices]
    adata_arcadia = adata_arcadia[common_indices]

    # check for mutual features
    mutual_obs = np.intersect1d(adata_arcadia.var_names, adata_other.var_names)
    print(
        f"num mutual {obs_type}",
        len(mutual_obs),
        f"num in {other_model_name} but not in arcadia",
        len(set(adata_other.var_names) - set(mutual_obs)),
        f"num in arcadia but not in {other_model_name}",
        len(set(adata_arcadia.var_names) - set(mutual_obs)),
    )
    if len(mutual_obs) == 0:
        raise ValueError(f"no mutual {obs_type}")
    # print difference in features (max 10)
    print(
        f"{obs_type} in {other_model_name} but not in arcadia",
        tuple(set(adata_other.var_names) - set(mutual_obs))[:10],
        "...",
    )
    print(
        f"{obs_type} in arcadia but not in {other_model_name}",
        tuple(set(adata_arcadia.var_names) - set(mutual_obs))[:10],
        "...",
    )

    # Copy annotations from ARCADIA to other model
    adata_other.obs["CN"] = adata_arcadia.obs["CN"] if "CN" in adata_arcadia.obs.columns else None
    adata_other.obs["matched_archetype_weight"] = (
        adata_arcadia.obs["matched_archetype_weight"]
        if "matched_archetype_weight" in adata_arcadia.obs.columns
        else None
    )
    adata_other.obs["is_extreme_archetype"] = (
        adata_arcadia.obs["is_extreme_archetype"]
        if "is_extreme_archetype" in adata_arcadia.obs.columns
        else None
    )
    adata_other.obs["archetype_label"] = (
        adata_arcadia.obs["archetype_label"]
        if "archetype_label" in adata_arcadia.obs.columns
        else None
    )
    adata_other = adata_other[:, mutual_obs]

    return adata_arcadia, adata_other, common_indices


def get_metrics_funcs(plot_flag=True):
    """Get metric functions and their kwargs for model comparison."""
    # Define metrics (using original names from compare_results.py)
    metrics_funcs_two_modalities = {
        "combined_latent_silhouette_f1": mtrc.compute_silhouette_f1,
        "cross_modality_cell_type_accuracy": mtrc.matching_accuracy,
        "cross_modality_cn_accuracy": mtrc.cross_modality_cn_accuracy,
        "f1_score": mtrc.f1_score_calc,
        "cn_f1_score": mtrc.f1_score_calc,
        "ari_score": mtrc.ari_score_calc,
    }

    metric_funcs_one_modality = {
        "cn_kbet_within_cell_types": mtrc.kbet_within_cell_types,
        "calculate_cell_type_silhouette": mtrc.calculate_cell_type_silhouette,
        "morans_i": mtrc.morans_i,
    }

    metric_funcs_combined_modalities = {
        "ari_f1": mtrc.compute_ari_f1,
        "cn_ilisi_within_cell_types": mtrc.calculate_cn_ilisi_within_cell_types,
        "silhouette_score": mtrc.silhouette_score_calc,
        "calculate_iLISI": mtrc.calculate_iLISI,
        "cn_kbet_within_cell_types": mtrc.kbet_within_cell_types,
        "modality_kbet_mixing_score": mtrc.modality_kbet_mixing_score,
        "pair_distance": mtrc.pair_distance,
        "mixing_metric": mtrc.mixing_metric,
        "morans_i": mtrc.morans_i,  # Moran's I on combined data
    }

    # Metric parameters (base kwargs without distance matrices)
    metrics_kwargs = {
        "cn_kbet_within_cell_types": {"label_key": "CN", "group_key": "cell_types", "rep_key": "X"},
        "modality_kbet_mixing_score": {"label_key": "modality", "group_key": None, "rep_key": "X"},
        "calculate_cell_type_silhouette": {"celltype_key": "cell_types", "use_rep": "X"},
        "calculate_iLISI": {"batch_key": "modality", "plot_flag": plot_flag},
        "cn_f1_score": {"label_key": "CN"},
        "cross_modality_cn_accuracy": {"k": 3, "global_step": 0 if plot_flag else None},
        "morans_i": {"score_key": "matched_archetype_weight", "use_rep": "X", "n_neighbors": 15},
        "pair_distance": {
            "modality_key": "modality",
            "pair_key": "pair_id",
            "rep_key": "X",
            "print_flag": True,
        },
        "foscttm": {
            "modality_key": "modality",
            "pair_key": "pair_id",
            "rep_key": "X",
            "print_flag": True,
        },
        "fosknn": {"modality_key": "modality", "pair_key": "pair_id", "rep_key": "X", "k": 3},
        "mixing_metric": {
            "modality_key": "modality",
            "rep_key": "X",
            "k_neighborhood": 300,
            "neighbor_ref": 5,
        },
        "cross_modality_cell_type_accuracy": {"plot_flag": False},
    }
    return (
        metrics_funcs_two_modalities,
        metric_funcs_one_modality,
        metric_funcs_combined_modalities,
        metrics_kwargs,
    )


def calculate_single_model_metrics(
    adata_latent_rna,
    adata_latent_prot,
    combined_latent,
    model_name="model",
    combined_distances=None,
    cross_modal_distances=None,
    metrics_funcs_two_modalities=None,
    metric_funcs_one_modality=None,
    metric_funcs_combined_modalities=None,
    metrics_kwargs=None,
    plot_flag=True,
):
    """
    Calculate metrics for a single model and return a DataFrame.

    This function calculates all metrics for one model and returns a DataFrame
    with model_name column for easy merging and comparison.
    """
    if metrics_funcs_two_modalities is None:
        (
            metrics_funcs_two_modalities,
            metric_funcs_one_modality,
            metric_funcs_combined_modalities,
            metrics_kwargs,
        ) = get_metrics_funcs(plot_flag=plot_flag)

    results = {}

    # Two modality metrics
    print(f"\nTwo-modality metrics (RNA ↔ Protein) for {model_name}:")
    for metric, func in metrics_funcs_two_modalities.items():
        print(f"Calculating {metric} for {model_name}")
        current_time = time()
        kwargs = metrics_kwargs.get(metric, {}).copy()

        if metric == "cross_modality_cn_accuracy" and cross_modal_distances is not None:
            kwargs["distance_matrix"] = cross_modal_distances

        results[metric] = func(adata_latent_rna, adata_latent_prot, **kwargs)
        print(f"Time taken for {metric}: {time() - current_time:.2f} seconds")

    # One modality metrics
    print(f"\nSingle-modality metrics for {model_name}:")
    for metric, func in metric_funcs_one_modality.items():
        print(f"Calculating {metric} for {model_name}")
        current_time = time()
        kwargs = metrics_kwargs.get(metric, {})
        results[f"{metric}_rna"] = func(adata_latent_rna, **kwargs)
        results[f"{metric}_prot"] = func(adata_latent_prot, **kwargs)
        print(f"Time taken for {metric}: {time() - current_time:.2f} seconds")

    # Combined modality metrics
    print(f"\nCombined-modality metrics for {model_name}:")
    for metric, func in metric_funcs_combined_modalities.items():
        print(f"Calculating {metric} for {model_name}")
        current_time = time()
        kwargs = metrics_kwargs.get(metric, {}).copy()

        if metric in ["pair_distance", "foscttm", "fosknn"] and combined_distances is not None:
            kwargs["pairwise_distances"] = combined_distances

        val = func(combined_latent, **kwargs)

        # Handle dict results by flattening them
        if isinstance(val, dict):
            for subkey, subval in val.items():
                results[f"{metric}_{subkey}"] = subval
        else:
            results[metric] = val

        print(f"Time taken for {metric}: {time() - current_time:.2f} seconds")

    # Convert to DataFrame
    rows = []
    for key, value in results.items():
        rows.append({"metric": key, "value": value})

    results_df = pd.DataFrame(rows)
    results_df["model_name"] = model_name

    return results_df


def merge_model_results(results_df_arcadia, results_df_other=None, other_model_name="other"):
    """
    Merge two model results DataFrames and create comparison columns.

    Parameters:
    -----------
    results_df_arcadia : pd.DataFrame
        Results DataFrame for ARCADIA model with 'metric' and 'value' columns
    results_df_other : pd.DataFrame or None
        Results DataFrame for other model with 'metric' and 'value' columns.
        If None, creates a results table with just ARCADIA and NaN for other model.
    other_model_name : str
        Name of the other model

    Returns:
    --------
    pd.DataFrame
        Pivoted DataFrame with comparison columns (diff, diff_%, arcadia_better)
    """
    # Handle case when results_df_other is None
    if results_df_other is None:
        # Create results pivot with just ARCADIA data
        results_pivot = results_df_arcadia.copy()
        results_pivot["model_name"] = other_model_name
        results_pivot = results_pivot.rename(columns={"metric": "metric_name", "value": "arcadia"})
        results_pivot = results_pivot[["metric_name", "arcadia"]]
        # Add NaN column for other model
        results_pivot[other_model_name] = np.nan
    else:
        # Combine both DataFrames
        combined_df = pd.concat([results_df_arcadia, results_df_other], ignore_index=True)

        # Pivot to have models as columns
        results_pivot = combined_df.pivot(
            index="metric", columns="model_name", values="value"
        ).reset_index()

        # Rename columns to match expected format
        if "arcadia" in results_pivot.columns:
            pass  # Keep as is
        elif (
            "model_name" in results_df_arcadia.columns
            and results_df_arcadia["model_name"].iloc[0] in results_pivot.columns
        ):
            # Rename the model column to "arcadia"
            arcadia_col = results_df_arcadia["model_name"].iloc[0]
            results_pivot = results_pivot.rename(columns={arcadia_col: "arcadia"})

        if (
            other_model_name not in results_pivot.columns
            and "model_name" in results_df_other.columns
            and results_df_other["model_name"].iloc[0] in results_pivot.columns
        ):
            # Rename the other model column
            other_col = results_df_other["model_name"].iloc[0]
            results_pivot = results_pivot.rename(columns={other_col: other_model_name})

        # Rename metric column to metric_name for consistency
        if "metric" in results_pivot.columns:
            results_pivot = results_pivot.rename(columns={"metric": "metric_name"})

    # Define which metrics are better when higher vs lower
    metric_direction = {
        "ari_score": "higher",
        "ari_f1_ari_clust": "higher",
        "ari_f1_ari_f1": "higher",
        "ari_f1_ari_mix": "higher",
        "calculate_cell_type_silhouette_prot": "higher",
        "calculate_cell_type_silhouette_rna": "higher",
        "calculate_iLISI": "higher",
        "cell_type_accuracy": "higher",
        "cms_within_cell_types_prot": "higher",
        "cms_within_cell_types_rna": "higher",
        "cn_entropy_within_cell_types": "lower",
        "cn_ilisi_within_cell_types": "lower",
        "cn_kbet_within_cell_types_prot": "higher",
        "cn_kbet_within_cell_types_rna": "higher",
        "combined_latent_silhouette_f1": "higher",
        "cross_modality_cell_type_accuracy": "higher",
        "cross_modality_cn_accuracy": "higher",
        "f1_score": "higher",
        "foscttm": "lower",
        "fosknn": "higher",
        "kbet_within_cell_types": "higher",
        "mixing_metric": "lower",
        "morans_i": "higher",
        "morans_i_prot": "higher",
        "morans_i_rna": "higher",
        "pair_distance": "lower",
        "silhouette_f1": "higher",
        "silhouette_score": "higher",
    }

    # Calculate difference (arcadia - other_model)
    if "arcadia" in results_pivot.columns and other_model_name in results_pivot.columns:
        results_pivot["diff"] = results_pivot["arcadia"] - results_pivot[other_model_name]

        # Calculate percent change: (arcadia - other) / |other| * 100
        # Handle division by zero and NaN values
        other_abs = results_pivot[other_model_name].abs()
        diff_percent = (
            (results_pivot["arcadia"] - results_pivot[other_model_name]) / other_abs * 100
        )

        # Replace inf and -inf with NaN, then round and convert to int
        diff_percent = diff_percent.replace([np.inf, -np.inf], np.nan)
        results_pivot["diff_%"] = diff_percent.round().astype("Int64")  # Use nullable integer type

        # Determine if arcadia is better
        # Use metric_name column (already renamed above if needed)
        metric_col = "metric_name" if "metric_name" in results_pivot.columns else "metric"
        results_pivot["arcadia_better"] = results_pivot.apply(
            lambda row: (
                (
                    row["arcadia"] > row[other_model_name]
                    if metric_direction.get(row[metric_col], "higher") == "higher"
                    else row["arcadia"] < row[other_model_name]
                )
                if pd.notna(row["arcadia"]) and pd.notna(row[other_model_name])
                else None
            ),
            axis=1,
        )

    # Rename metric column to metric_name for consistency (if not already done)
    if "metric" in results_pivot.columns:
        results_pivot = results_pivot.rename(columns={"metric": "metric_name"})

    # Sort by metric name
    results_pivot = results_pivot.sort_values("metric_name")

    return results_pivot


def save_comparison_results(results_pivot, output_file="results_comparison_accumulated.csv"):
    """Save comparison results to CSV file."""
    output_path = Path(output_file).resolve()

    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            # Check if existing file is empty or has no valid data
            if existing_df.empty or len(existing_df.columns) == 0:
                print(f"\nWarning: Existing file is empty, overwriting...")
                results_pivot.to_csv(output_file, index=False)
                print(f"✓ Created new results file: {output_path}")
            else:
                results_pivot = pd.concat([existing_df, results_pivot], ignore_index=True)
                results_pivot.to_csv(output_file, index=False)
                print(f"✓ Appended {len(results_pivot) - len(existing_df)} rows to: {output_path}")
                print(f"  Total rows in file: {len(results_pivot)}")
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"\nWarning: Could not read existing file ({e}), overwriting...")
            results_pivot.to_csv(output_file, index=False)
            print(f"✓ Created new results file: {output_path}")
    else:
        results_pivot.to_csv(output_file, index=False)
        print(f"✓ Created new results file: {output_path}")
        print(f"  Total rows: {len(results_pivot)}")
