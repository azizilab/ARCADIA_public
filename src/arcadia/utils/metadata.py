# %%
"""
Pipeline Metadata Utilities

This module contains all functions for managing pipeline metadata across the CODEX RNA-seq preprocessing pipeline.
Each preprocessing script should call the appropriate function to update metadata before saving data.
"""

import inspect
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def get_caller_file():
    """Get the filename of the calling script."""
    # two frames up
    caller_frame = inspect.stack()[2]
    filename = caller_frame.filename  # Full path to the caller's file

    # Handle notebook execution - look for the original script name
    if "ipykernel" in filename or "/tmp/" in filename or "/var/tmp/" in filename:
        # Try to find a more meaningful filename from the stack
        for frame_info in inspect.stack():
            frame_filename = frame_info.filename
            if (
                frame_filename.endswith(".py")
                and "ipykernel" not in frame_filename
                and "/tmp/" not in frame_filename
                and "/var/tmp/" not in frame_filename
                and "_" in Path(frame_filename).name
            ):
                filename = frame_filename
                break

        # If still a temp file, try to get from globals or environment
        if "ipykernel" in filename or "/tmp/" in filename or "/var/tmp/" in filename:
            # Check if there's a way to get the original script name
            # This is a fallback for notebook execution
            # Look for environment variables or other clues
            if "PAPERMILL_INPUT_PATH" in os.environ:
                filename = os.environ["PAPERMILL_INPUT_PATH"]

    return filename


def _add_source_file_to_metadata(adata_rna, adata_prot, rna_path, prot_path, step_name):
    """
    Helper function to add source file information to pipeline metadata.

    Args:
        adata_rna: RNA AnnData object
        adata_prot: Protein AnnData object
        rna_path: Path to RNA source file
        prot_path: Path to protein source file
        step_name: Name of the pipeline step (e.g., 'align_datasets')
    """
    if "pipeline_metadata" not in adata_rna.uns:
        adata_rna.uns["pipeline_metadata"] = {}
    if "pipeline_metadata" not in adata_prot.uns:
        adata_prot.uns["pipeline_metadata"] = {}

    if step_name not in adata_rna.uns["pipeline_metadata"]:
        adata_rna.uns["pipeline_metadata"][step_name] = {}
        adata_prot.uns["pipeline_metadata"][step_name] = {}

    adata_rna.uns["pipeline_metadata"][step_name]["rna_source_file"] = str(rna_path)
    adata_prot.uns["pipeline_metadata"][step_name]["protein_source_file"] = str(prot_path)


def initialize_pipeline_metadata(
    timestamp_str: str,
    FILENAME: str,
    dataset_name: str,
) -> Dict[str, Any]:
    """
    Initialize pipeline metadata structure for the first preprocessing step.
    Creates empty dicts for all pipeline steps to avoid existence checks downstream.

    Args:
        timestamp_str: Timestamp string for the pipeline run
        FILENAME: Name of the preprocessing file
        dataset_name: Name of the dataset being processed

    Returns:
        Dictionary containing initial pipeline metadata structure with all step placeholders
    """
    return {
        "start_time": timestamp_str,
        "filename": FILENAME,
        "dataset_name": dataset_name,
        "preprocess": {
            "filtering_steps": [],
        },
        "align_datasets": {},
        "spatial_info_integrate": {},
        "archetype_generation": {},
        "prepare_data": {},
        "train_vae": {},
    }


def update_preprocess_metadata(
    adata,
    filtering_steps: List[str],
    normalization: Optional[str] = None,
    n_highly_variable_genes_rna: Optional[int] = None,
) -> None:
    """
    Update preprocessing metadata for the dataset preprocessing steps.

    Args:
        adata: AnnData object to update
        filtering_steps: List of filtering steps to add
        normalization: Normalization method used (optional)
        n_highly_variable_genes_rna: Number of highly variable genes selected (optional)
    """
    if "pipeline_metadata" not in adata.uns:
        adata.uns["pipeline_metadata"] = {}
    if "preprocess" not in adata.uns["pipeline_metadata"]:
        adata.uns["pipeline_metadata"]["preprocess"] = {"filtering_steps": []}

    # Add filtering steps
    for step in filtering_steps:
        adata.uns["pipeline_metadata"]["preprocess"]["filtering_steps"].append(step)

    # Add normalization info if provided
    if normalization is not None:
        adata.uns["pipeline_metadata"]["preprocess"]["normalization"] = normalization

    # Add HVG count if provided
    if n_highly_variable_genes_rna is not None:
        adata.uns["pipeline_metadata"]["preprocess"][
            "n_highly_variable_genes_rna"
        ] = n_highly_variable_genes_rna


# Removed: setup_plots_directory_metadata (plots saving removed from pipeline)


def initialize_archetype_generation_metadata(
    adata_rna,
    adata_prot,
    rna_file: str,
    prot_file: str,
    cn_type: str,
    num_cn_clusters: int,
    leiden_resolution: Optional[float] = None,
    vae_hyperparams: Optional[Dict[str, Any]] = None,
    pca_params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Initialize archetype generation metadata.

    Args:
        adata_rna: RNA AnnData object
        adata_prot: Protein AnnData object
        rna_file: Path to RNA source file
        prot_file: Path to protein source file
        cn_type: Type of cellular neighborhoods ("empirical" or "annotated")
        num_cn_clusters: Number of CN clusters
        leiden_resolution: Leiden clustering resolution (for empirical CN)
        vae_hyperparams: VAE hyperparameters dictionary
        pca_params: PCA parameters dictionary
    """
    # Add archetype generation metadata
    archetype_metadata = {
        "rna_source_file_path": rna_file,
        "protein_source_file_path": prot_file,
        "cn_type": cn_type,
        "num_cn_clusters": num_cn_clusters,
    }

    if leiden_resolution is not None:
        archetype_metadata["leiden_resolution"] = leiden_resolution

    if vae_hyperparams is not None:
        archetype_metadata["vae_hyperparams"] = vae_hyperparams

    if pca_params is not None:
        archetype_metadata["pca"] = pca_params

    adata_rna.uns["pipeline_metadata"]["archetype_generation"] = archetype_metadata
    adata_prot.uns["pipeline_metadata"]["archetype_generation"] = archetype_metadata.copy()


def update_archetype_generation_metadata(
    adata_rna,
    adata_prot,
    cross_modal_distance_matrix: Optional[np.ndarray] = None,
    rna_to_prot_matches: Optional[np.ndarray] = None,
    prot_to_rna_matches: Optional[np.ndarray] = None,
) -> None:
    """
    Update archetype generation metadata with cross-modal matching results.

    Args:
        adata_rna: RNA AnnData object
        adata_prot: Protein AnnData object
        cross_modal_distance_matrix: Distance matrix between RNA and protein archetypes
        rna_to_prot_matches: RNA to protein archetype matches
        prot_to_rna_matches: Protein to RNA archetype matches
    """
    if "archetype_generation" not in adata_rna.uns["pipeline_metadata"]:
        print("Warning: archetype_generation metadata not found. Initialize first.")
        return

    update_dict = {}

    if cross_modal_distance_matrix is not None:
        update_dict["cross_modal_distance_matrix"] = cross_modal_distance_matrix

    if rna_to_prot_matches is not None:
        update_dict["rna_to_prot_matches"] = rna_to_prot_matches

    if prot_to_rna_matches is not None:
        update_dict["prot_to_rna_matches"] = prot_to_rna_matches

    adata_rna.uns["pipeline_metadata"]["archetype_generation"].update(update_dict)
    adata_prot.uns["pipeline_metadata"]["archetype_generation"] = adata_rna.uns[
        "pipeline_metadata"
    ]["archetype_generation"].copy()


def initialize_prepare_data_metadata(adata_rna, adata_prot) -> None:
    """
    Initialize prepare data metadata.

    Args:
        adata_rna: RNA AnnData object
        adata_prot: Protein AnnData object
    """
    adata_rna.uns["pipeline_metadata"]["prepare_data"].update(
        {
            "original_shape": list(adata_rna.shape),
            "archetype_matching": {
                "distance_matrix": None,  # Not stored to reduce file size
                "initial_distance": None,
                "distance_matrix_shape": None,
                "mean_distance": None,
                "median_distance": None,
            },
        }
    )

    # Copy metadata to protein adata
    adata_prot.uns["pipeline_metadata"]["prepare_data"] = adata_rna.uns["pipeline_metadata"][
        "prepare_data"
    ].copy()
    adata_prot.uns["pipeline_metadata"]["prepare_data"]["original_shape"] = list(adata_prot.shape)


def update_prepare_data_metadata(adata_rna, adata_prot, archetype_distances: np.ndarray) -> None:
    """
    Update prepare data metadata with archetype matching results.

    Args:
        adata_rna: RNA AnnData object
        adata_prot: Protein AnnData object
        archetype_distances: Distance matrix between RNA and protein archetypes
    """
    if "prepare_data" not in adata_rna.uns["pipeline_metadata"]:
        initialize_prepare_data_metadata(adata_rna, adata_prot)

    # Store only the summary statistics, NOT the full 8GB distance matrix (not used in training)
    adata_rna.uns["pipeline_metadata"]["prepare_data"]["archetype_matching"] = {
        "distance_matrix": None,  # REMOVED: 8GB matrix causes file bloat, not used in training
        "initial_distance": float(np.diag(archetype_distances).mean()),
        "distance_matrix_shape": list(
            archetype_distances.shape
        ),  # Keep shape for reference (as list for h5ad)
        "mean_distance": float(np.mean(archetype_distances)),
        "median_distance": float(np.median(archetype_distances)),
    }

    # Copy the same metadata to protein adata
    adata_prot.uns["pipeline_metadata"]["prepare_data"]["archetype_matching"] = adata_rna.uns[
        "pipeline_metadata"
    ]["prepare_data"]["archetype_matching"].copy()


def initialize_train_vae_metadata(
    adata_rna, adata_prot, training_parameters: Dict[str, Any]
) -> None:
    """
    Initialize VAE training metadata.

    Args:
        adata_rna: RNA AnnData object
        adata_prot: Protein AnnData object
        training_parameters: Dictionary of training parameters
    """
    training_metadata = {
        "train_vae": {
            "training_parameters": training_parameters,
            "training_results": {
                "final_losses": None,
                "best_epoch": None,
                "training_time": None,
                "validation_metrics": None,
            },
        }
    }

    # Add training metadata to adata objects
    adata_rna.uns["pipeline_metadata"]["train_vae"].update(training_metadata["train_vae"])
    adata_prot.uns["pipeline_metadata"]["train_vae"] = adata_rna.uns["pipeline_metadata"][
        "train_vae"
    ].copy()


def update_train_vae_metadata(adata_rna, adata_prot, training_results: Dict[str, Any]) -> None:
    """
    Update VAE training metadata with results.

    Args:
        adata_rna: RNA AnnData object
        adata_prot: Protein AnnData object
        training_results: Dictionary of training results
    """
    if "train_vae" not in adata_rna.uns["pipeline_metadata"]:
        print("Warning: train_vae metadata not found. Initialize first.")
        return

    if "training_results" not in adata_rna.uns["pipeline_metadata"]["train_vae"]:
        adata_rna.uns["pipeline_metadata"]["train_vae"]["training_results"] = {}

    adata_rna.uns["pipeline_metadata"]["train_vae"]["training_results"].update(training_results)
    adata_prot.uns["pipeline_metadata"]["train_vae"] = adata_rna.uns["pipeline_metadata"][
        "train_vae"
    ].copy()


def copy_pipeline_metadata(source_adata, target_adata) -> None:
    """
    Copy pipeline metadata from source to target AnnData object.

    Args:
        source_adata: Source AnnData object
        target_adata: Target AnnData object
    """
    if "pipeline_metadata" in source_adata.uns:
        target_adata.uns["pipeline_metadata"] = source_adata.uns["pipeline_metadata"].copy()


def get_pipeline_summary(adata) -> Dict[str, Any]:
    """
    Get a summary of the pipeline metadata.

    Args:
        adata: AnnData object

    Returns:
        Dictionary containing pipeline summary
    """
    if "pipeline_metadata" not in adata.uns:
        return {"error": "No pipeline metadata found"}

    metadata = adata.uns["pipeline_metadata"]
    summary = {
        "stages_completed": list(metadata.keys()),
        "start_time": metadata.get("start_time", "Unknown"),
    }

    # Add stage-specific summaries
    if "preprocess" in metadata:
        summary["preprocessing"] = {
            "filtering_steps": len(metadata["preprocess"].get("filtering_steps", [])),
            "normalization": metadata["preprocess"].get("normalization", "Not specified"),
        }

    if "archetype_generation" in metadata:
        summary["archetype_generation"] = {
            "cn_type": metadata["archetype_generation"].get("cn_type", "Unknown"),
            "num_cn_clusters": metadata["archetype_generation"].get("num_cn_clusters", "Unknown"),
        }

    if "prepare_data" in metadata:
        summary["prepare_data"] = {
            "original_shape": metadata["prepare_data"].get("original_shape", "Unknown"),
        }

    if "train_vae" in metadata:
        summary["train_vae"] = {
            "max_epochs": metadata["train_vae"]["training_parameters"].get("max_epochs", "Unknown"),
            "latent_dim": metadata["train_vae"]["training_parameters"].get("latent_dim", "Unknown"),
        }

    return summary


# %%
# Functions to be called at the end of each preprocessing script:


def finalize_preprocess_metadata(
    adata_rna, adata_prot, selected_batches_or_samples: List[str]
) -> None:
    """
    Finalize metadata for preprocessing scripts.

    Args:
        adata_rna: RNA AnnData object
        adata_prot: Protein AnnData object
        selected_batches_or_samples: List of batch names or sample names used in preprocessing
    """
    filtering_steps = [
        f"selected_batch_rna_{selected_batches_or_samples}",
        "filtered_cell_types_prot",
        "mad_outlier_removal",
    ]

    for adata in [adata_rna, adata_prot]:
        update_preprocess_metadata(adata, filtering_steps)


def finalize_align_datasets_metadata(adata_rna, adata_prot, n_highly_variable_genes: int) -> None:
    """
    Finalize metadata for _1_align_datasets.py
    """
    update_preprocess_metadata(
        adata_rna,
        [],
        normalization="raw_data_before_spatial_features",
        n_highly_variable_genes_rna=n_highly_variable_genes,
    )

    # Copy metadata to protein adata
    copy_pipeline_metadata(adata_rna, adata_prot)


def finalize_spatial_info_integrate_metadata(
    adata_rna,
    adata_prot,
    rna_file: str,
    prot_file: str,
    vae_hyperparams: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Finalize metadata for _2_spatial_info_integrate.py
    """
    initialize_archetype_generation_metadata(
        adata_rna,
        adata_prot,
        rna_file,
        prot_file,
        cn_type="annotated",  # or "empirical" based on your settings
        num_cn_clusters=7,  # Update based on actual number
        vae_hyperparams=vae_hyperparams,
    )


def finalize_archetype_generation_metadata(
    adata_rna,
    adata_prot,
    similarity_matrix: np.ndarray,
    rna_to_prot_matches: np.ndarray,
    prot_to_rna_matches: np.ndarray,
    pca_params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Finalize metadata for _3_archetype_generation_neighbors_covet.py
    """
    # Add PCA parameters if provided
    if pca_params is not None and "archetype_generation" in adata_rna.uns["pipeline_metadata"]:
        adata_rna.uns["pipeline_metadata"]["archetype_generation"]["pca"] = pca_params
        adata_prot.uns["pipeline_metadata"]["archetype_generation"]["pca"] = pca_params

    update_archetype_generation_metadata(
        adata_rna,
        adata_prot,
        cross_modal_distance_matrix=similarity_matrix,
        rna_to_prot_matches=rna_to_prot_matches,
        prot_to_rna_matches=prot_to_rna_matches,
    )


def finalize_prepare_data_metadata(adata_rna, adata_prot, archetype_distances: np.ndarray) -> None:
    """
    Finalize metadata for _4_prepare_data_for_training.py
    """
    initialize_prepare_data_metadata(adata_rna, adata_prot)
    update_prepare_data_metadata(adata_rna, adata_prot, archetype_distances)


def finalize_train_vae_metadata(
    adata_rna,
    adata_prot,
    training_params: Dict[str, Any],
    training_results: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Finalize metadata for _5_train_vae_with_archetypes_vectors.py
    """
    initialize_train_vae_metadata(adata_rna, adata_prot, training_params)

    if training_results is not None:
        update_train_vae_metadata(adata_rna, adata_prot, training_results)
