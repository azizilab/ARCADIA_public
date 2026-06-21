"""Data loading and saving utilities."""

import io
import math
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import requests
import scanpy as sc
import scipy.sparse as sp
import scvi
from scipy.io import mmread
from scipy.sparse import csr_matrix, issparse

from arcadia.data_utils.cleaning import clean_uns_for_h5ad
from arcadia.utils.metadata import _add_source_file_to_metadata, get_caller_file


def get_latest_file(
    folder, prefix, index_from_end=0, dataset_name=None, exact_step=None
) -> Optional[str]:
    """
    Get the latest file with given prefix from folder.
    With new naming convention (e.g., 0_rna_2025-01-01-12-00-00.h5ad), finds files from exact step.

    Args:
        folder: Base folder to search in
        prefix: File prefix to match (e.g., "rna", "protein")
        index_from_end: Index from end of sorted files (0 = latest/highest number)
        dataset_name: Optional dataset name to search in specific subdirectory
        exact_step: Exact step number to load (e.g., if exact_step=2, will ONLY load step 2 files)

    Returns:
        Path to the latest matching file, or None if not found
    """
    base_folder = Path(folder)

    # If dataset_name is provided, search in dataset-specific subdirectory first
    if dataset_name:
        dataset_folder = base_folder / dataset_name
        if dataset_folder.exists():
            files = [f for f in os.listdir(dataset_folder) if f.endswith(".h5ad")]
            # Filter files that match the pattern: step_prefix_timestamp.h5ad or step_prefix.h5ad
            pattern_files = []
            for f in files:
                # Check if file matches NEW pattern: number_prefix_timestamp.h5ad or number_prefix.h5ad
                # Must start with a digit followed by underscore
                if f[0].isdigit() and "_" in f:
                    parts = f.split("_")
                    if len(parts) >= 2:  # At least step_prefix
                        step_part = parts[0]
                        prefix_part = parts[1]
                        if step_part.isdigit() and prefix_part == prefix:
                            pattern_files.append(f)
            if pattern_files:
                return _get_latest_from_numbered_files(
                    dataset_folder, pattern_files, index_from_end, exact_step
                )

    # Search in all subdirectories and main folder
    all_files = []

    # Check main folder
    if base_folder.exists():
        main_files = [f for f in os.listdir(base_folder) if f.endswith(".h5ad")]
        for f in main_files:
            # Check if file matches NEW pattern: number_prefix_timestamp.h5ad or number_prefix.h5ad
            # Must start with a digit followed by underscore
            if f[0].isdigit() and "_" in f:
                parts = f.split("_")
                if len(parts) >= 2:
                    step_part = parts[0]
                    prefix_part = parts[1]
                    if step_part.isdigit() and prefix_part == prefix:
                        all_files.append((base_folder, f))

    # Check subdirectories (dataset-specific folders)
    if base_folder.exists():
        for subdir in base_folder.iterdir():
            if subdir.is_dir():
                sub_files = [f for f in os.listdir(subdir) if f.endswith(".h5ad")]
                for f in sub_files:
                    # Check if file matches NEW pattern: number_prefix_timestamp.h5ad or number_prefix.h5ad
                    # Must start with a digit followed by underscore
                    if f[0].isdigit() and "_" in f:
                        parts = f.split("_")
                        if len(parts) >= 2:
                            step_part = parts[0]
                            prefix_part = parts[1]
                            if step_part.isdigit() and prefix_part == prefix:
                                all_files.append((subdir, f))

    if not all_files:
        return None

    # Filter by exact step if specified
    if exact_step is not None:
        filtered_files = []
        for folder_path, filename in all_files:
            step_part = filename.split("_")[0]
            try:
                step_num = int(step_part)
                if step_num == exact_step:
                    filtered_files.append((folder_path, filename))
            except ValueError:
                # Skip files that don't follow the pattern when exact_step is specified
                pass
        all_files = filtered_files

    if not all_files:
        return None

    # Sort all files by timestamp within the same step (latest first)
    def get_timestamp_from_filename(folder_file_tuple):
        folder_path, filename = folder_file_tuple
        # Extract timestamp from filename like "3_rna_2025-01-01-12-00-00.h5ad"
        parts = filename.split("_")
        if len(parts) >= 3:
            # Join timestamp parts: "2025-01-01-12-00-00"
            timestamp_str = "_".join(parts[2:]).replace(".h5ad", "")
            return timestamp_str
        else:
            # Fallback to modification time
            mtime = os.path.getmtime(folder_path / filename)
            return str(int(mtime))

    all_files.sort(key=get_timestamp_from_filename, reverse=True)

    # Get the requested file
    folder_path, latest_file = all_files[index_from_end]
    full_path = folder_path / latest_file

    # Print file information
    file_time = datetime.fromtimestamp(os.path.getmtime(full_path))
    time_diff = datetime.now() - file_time

    if time_diff.days > 0:
        print(
            f"{full_path} was created {time_diff.days} days, {time_diff.seconds//3600} hours, {(time_diff.seconds%3600)//60} minutes ago"
        )
    elif time_diff.seconds > 3600:
        print(
            f"{full_path} was created {time_diff.seconds//3600} hours, {(time_diff.seconds%3600)//60} minutes ago"
        )
    else:
        print(f"{full_path} was created {time_diff.seconds} seconds ago")

    return str(full_path)


def _get_latest_from_numbered_files(folder, files, index_from_end=0, exact_step=None):
    """Helper function to get latest file from numbered files (e.g., 0_rna_2025-01-01-12-00-00.h5ad, 1_rna.h5ad)."""

    def get_step_number(filename):
        # Extract step number from filename like "3_rna_2025-01-01-12-00-00.h5ad" or "3_rna.h5ad"
        step_part = filename.split("_")[0]
        try:
            return int(step_part)
        except ValueError:
            # Fallback to modification time for files that don't follow the pattern
            mtime = os.path.getmtime(os.path.join(folder, filename))
            return int(mtime)

    # Filter by exact step if specified
    if exact_step is not None:
        filtered_files = []
        for f in files:
            step_num = get_step_number(f)
            if isinstance(step_num, int) and step_num == exact_step:
                filtered_files.append(f)
        files = filtered_files

    if not files:
        return None

    # Sort by timestamp within the same step (latest first)
    def get_timestamp_from_filename_helper(filename):
        parts = filename.split("_")
        if len(parts) >= 3:
            # Join timestamp parts: "2025-01-01-12-00-00"
            timestamp_str = "_".join(parts[2:]).replace(".h5ad", "")
            return timestamp_str
        else:
            # Fallback to modification time
            mtime = os.path.getmtime(os.path.join(folder, filename))
            return str(int(mtime))

    files.sort(key=get_timestamp_from_filename_helper, reverse=True)
    latest_file = files[index_from_end]
    return os.path.join(folder, latest_file)


def load_adata_latest(
    folder, file_prefixes, dataset_name=None, exact_step=None, index_from_end=0, return_path=False
):
    """
    Load the latest AnnData files with dataset-specific support.
    With new naming convention, looks for files like #_rna_timestamp.h5ad, #_protein_timestamp.h5ad.
    Automatically adds source file information to pipeline metadata for the calling step.

    Args:
        folder: Base folder containing processed data
        file_prefixes: Dict, list, or string of modality names to load (e.g., "rna", "protein")
        dataset_name: Optional dataset name to load from specific subdirectory
        exact_step: Exact step number to load (e.g., if exact_step=2, will ONLY load step 2 files)
        index_from_end: Index from end of sorted file list (0 = latest, 1 = second latest, etc.)
        return_path: Deprecated, kept for compatibility. Paths are now stored in pipeline_metadata.

    Returns:
        - If file_prefixes is a dict:
            return loaded_data_dict
        - If file_prefixes is a list:
            return tuple(adata_list)
        - If file_prefixes is a string:
            return adata
    """

    # Auto-detect the calling step if exact_step is not provided
    calling_step = None
    step_name_mapping = {
        1: "align_datasets",
        2: "spatial_info_integrate",
        3: "archetype_generation",
        4: "prepare_data",
        5: "train_vae",
    }

    caller_file = get_caller_file()
    caller_filename = Path(caller_file).name

    if exact_step is None:
        if caller_filename.startswith("_"):
            parts = caller_filename.split("_")
            if len(parts) > 1 and parts[1].isdigit():
                # For step N, we want to load files from EXACTLY step N-1
                calling_step = int(parts[1])
                exact_step = calling_step - 1
                print(
                    f"Auto-detected calling step {calling_step}, will load files from EXACTLY step {exact_step}"
                )
    else:
        # If exact_step is provided, infer calling_step from caller filename
        if caller_filename.startswith("_"):
            parts = caller_filename.split("_")
            if len(parts) > 1 and parts[1].isdigit():
                calling_step = int(parts[1])

    if isinstance(file_prefixes, dict):
        # Load multiple files and return as dict
        loaded_data = {}
        dataset_names = []
        file_paths = {}

        for key, modality in file_prefixes.items():
            file_path = get_latest_file(
                folder,
                modality,
                dataset_name=dataset_name,
                exact_step=exact_step,
                index_from_end=index_from_end,
            )
            if file_path:
                loaded_data[key] = sc.read(file_path)
                file_paths[key] = file_path
                print(f"Loaded {key}: {loaded_data[key].shape}")
                print(f"File path: {Path(file_path).resolve()}")

                # Extract dataset name from file path to validate consistency
                file_path_obj = Path(file_path)
                dataset_from_path = file_path_obj.parent.name
                dataset_names.append(dataset_from_path)
            else:
                raise FileNotFoundError(
                    f"No files found for modality {modality} from step {exact_step} in {folder}"
                )

        # Validate that all files come from the same dataset folder
        if len(set(dataset_names)) > 1:
            dataset_mapping = {
                key: Path(file_paths[key]).parent.name for key in file_prefixes.keys()
            }
            raise ValueError(
                f"Loaded files are from different datasets: {dataset_mapping}. "
                f"File paths: {file_paths}. "
                f"All modalities must be from the same dataset folder."
            )

        # Validate dataset_name parameter matches loaded files
        if dataset_name is not None:

            def normalize_name(name):
                return name.lower().replace("_", "").replace("-", "")

            normalized_dataset_name = normalize_name(dataset_name)

            for key, adata in loaded_data.items():
                adata_dataset_name = adata.uns.get("dataset_name")
                dataset_from_path = Path(file_paths[key]).parent.name

                normalized_adata_name = (
                    normalize_name(adata_dataset_name) if adata_dataset_name else ""
                )
                normalized_folder_name = normalize_name(dataset_from_path)

                # Check if folder name matches file's dataset_name (consistency check)
                if normalized_adata_name and normalized_adata_name != normalized_folder_name:
                    raise ValueError(
                        f"Dataset name inconsistency for {key} modality: folder name '{dataset_from_path}' "
                        f"does not match dataset_name in file '{adata_dataset_name}'. "
                        f"File path: {file_paths[key]}."
                    )

                # Check if the provided dataset_name matches (either exact match or folder contains it)
                folder_contains_dataset = (
                    normalized_folder_name.endswith(normalized_dataset_name)
                    or normalized_dataset_name in normalized_folder_name
                )
                exact_match = normalized_dataset_name == normalized_adata_name

                if not exact_match and not folder_contains_dataset:
                    raise ValueError(
                        f"Dataset name mismatch for {key} modality. "
                        f"Expected dataset_name: '{dataset_name}', "
                        f"but loaded file has dataset_name in uns: '{adata_dataset_name}'. "
                        f"File path: {file_paths[key]}. "
                        f"Folder name: '{dataset_from_path}'. "
                        f"Please ensure the dataset_name parameter matches the dataset "
                        f"or the folder name contains the expected dataset name."
                    )

        return loaded_data

    elif isinstance(file_prefixes, list):
        # Load multiple files and return as tuple
        loaded_data = []
        dataset_names = []
        file_paths = []

        for modality in file_prefixes:
            file_path = get_latest_file(
                folder,
                modality,
                dataset_name=dataset_name,
                exact_step=exact_step,
                index_from_end=index_from_end,
            )
            if file_path:
                adata = sc.read(file_path)
                loaded_data.append(adata)
                file_paths.append(file_path)
                print(f"Loaded {modality}: {adata.shape}")
                print(f"File path: {Path(file_path).resolve()}")

                # Extract dataset name from file path to validate consistency
                file_path_obj = Path(file_path)
                dataset_from_path = file_path_obj.parent.name
                dataset_names.append(dataset_from_path)
            else:
                raise FileNotFoundError(
                    f"No files found for modality {modality} from step {exact_step} in {folder}"
                )

        # Validate that all files come from the same dataset folder
        if len(set(dataset_names)) > 1:
            raise ValueError(
                f"Loaded files are from different datasets: {dict(zip(file_prefixes, dataset_names))}. "
                f"File paths: {dict(zip(file_prefixes, file_paths))}. "
                f"All modalities must be from the same dataset folder."
            )

        # Validate dataset_name parameter matches loaded files
        if dataset_name is not None:

            def normalize_name(name):
                return name.lower().replace("_", "").replace("-", "")

            normalized_dataset_name = normalize_name(dataset_name)

            for i, (modality, adata) in enumerate(zip(file_prefixes, loaded_data)):
                adata_dataset_name = adata.uns.get("dataset_name")
                dataset_from_path = Path(file_paths[i]).parent.name

                normalized_adata_name = (
                    normalize_name(adata_dataset_name) if adata_dataset_name else ""
                )
                normalized_folder_name = normalize_name(dataset_from_path)

                # Check if folder name matches file's dataset_name (consistency check)
                if normalized_adata_name and normalized_adata_name != normalized_folder_name:
                    raise ValueError(
                        f"Dataset name inconsistency for {modality} modality: folder name '{dataset_from_path}' "
                        f"does not match dataset_name in file '{adata_dataset_name}'. "
                        f"File path: {file_paths[i]}."
                    )

                # Check if the provided dataset_name matches (either exact match or folder contains it)
                folder_contains_dataset = (
                    normalized_folder_name.endswith(normalized_dataset_name)
                    or normalized_dataset_name in normalized_folder_name
                )
                exact_match = normalized_dataset_name == normalized_adata_name

                if not exact_match and not folder_contains_dataset:
                    raise ValueError(
                        f"Dataset name mismatch for {modality} modality. "
                        f"Expected dataset_name: '{dataset_name}', "
                        f"but loaded file has dataset_name in uns: '{adata_dataset_name}'. "
                        f"File path: {file_paths[i]}. "
                        f"Folder name: '{dataset_from_path}'. "
                        f"Please ensure the dataset_name parameter matches the dataset "
                        f"or the folder name contains the expected dataset name."
                    )

        # Add source file information to pipeline metadata if loading rna and protein
        if (
            calling_step is not None
            and calling_step in step_name_mapping
            and len(file_prefixes) == 2
            and "rna" in [m.lower() for m in file_prefixes]
            and "protein" in [m.lower() for m in file_prefixes]
        ):
            step_name = step_name_mapping[calling_step]
            # Find rna and protein indices
            rna_idx = next(i for i, m in enumerate(file_prefixes) if "rna" in m.lower())
            prot_idx = next(i for i, m in enumerate(file_prefixes) if "protein" in m.lower())

            adata_rna = loaded_data[rna_idx]
            adata_prot = loaded_data[prot_idx]
            rna_path = file_paths[rna_idx]
            prot_path = file_paths[prot_idx]

            _add_source_file_to_metadata(adata_rna, adata_prot, rna_path, prot_path, step_name)
            print(f"Added source file info to pipeline_metadata['{step_name}']")

        return tuple(loaded_data)

    else:
        # Load single file
        file_path = get_latest_file(
            folder,
            file_prefixes,
            dataset_name=dataset_name,
            exact_step=exact_step,
            index_from_end=index_from_end,
        )
        if file_path:
            adata = sc.read(file_path)
            print(f"Loaded {file_prefixes}: {adata.shape}")
            print(f"File path: {Path(file_path).resolve()}")

            # Validate dataset_name parameter matches loaded file
            if dataset_name is not None:
                adata_dataset_name = adata.uns.get("dataset_name")
                dataset_from_path = Path(file_path).parent.name

                # Normalize names for comparison (handle underscore/case variations)
                def normalize_name(name):
                    return name.lower().replace("_", "").replace("-", "")

                normalized_dataset_name = normalize_name(dataset_name)
                normalized_adata_name = (
                    normalize_name(adata_dataset_name) if adata_dataset_name else ""
                )
                normalized_folder_name = normalize_name(dataset_from_path)

                # Check if folder name matches file's dataset_name (consistency check)
                # Allow if folder contains dataset_name or vice versa (handles path construction issues)
                folder_matches_file = (
                    normalized_adata_name == normalized_folder_name
                    or normalized_folder_name.endswith(normalized_adata_name)
                    or normalized_adata_name in normalized_folder_name
                    or normalized_folder_name in normalized_adata_name
                )
                if normalized_adata_name and not folder_matches_file:
                    raise ValueError(
                        f"Dataset name inconsistency: folder name '{dataset_from_path}' "
                        f"does not match dataset_name in file '{adata_dataset_name}'. "
                        f"File path: {file_path}."
                    )

                # Check if the provided dataset_name matches (either exact match or folder contains it)
                folder_contains_dataset = (
                    normalized_folder_name.endswith(normalized_dataset_name)
                    or normalized_dataset_name in normalized_folder_name
                )
                exact_match = normalized_dataset_name == normalized_adata_name

                if not exact_match and not folder_contains_dataset:
                    raise ValueError(
                        f"Dataset name mismatch for {file_prefixes} modality. "
                        f"Expected dataset_name: '{dataset_name}', "
                        f"but loaded file has dataset_name in uns: '{adata_dataset_name}'. "
                        f"File path: {file_path}. "
                        f"Folder name: '{dataset_from_path}'. "
                        f"Please ensure the dataset_name parameter matches the dataset "
                        f"or the folder name contains the expected dataset name."
                    )

            return adata
        else:
            raise FileNotFoundError(
                f"No files found for modality {file_prefixes} from step {exact_step} in {folder}"
            )


def save_processed_data(adata_rna, adata_prot, save_dir, caller_filename=None):
    for col in adata_rna.obs.columns:
        if adata_rna.obs[col].dtype == "object":
            adata_rna.obs[col] = adata_rna.obs[col].astype(str)
    for col in adata_prot.obs.columns:
        if adata_prot.obs[col].dtype == "object":
            adata_prot.obs[col] = adata_prot.obs[col].astype(str)

    # Use provided filename or fallback to detection
    if caller_filename:
        caller_file = caller_filename
        print(f"Saving processed data from {caller_filename}")
    else:
        caller_file = get_caller_file()
        print(f"Saving processed data from {caller_file}")
        caller_filename = Path(caller_file).name

    # Extract dataset name: first check if already set in adata, then extract from filename
    # Check if dataset_name is already set in adata (takes priority)
    dataset_name = adata_rna.uns.get("dataset_name", None)
    if not dataset_name:
        dataset_name = adata_prot.uns.get("dataset_name", None)

    # If not set, extract from caller filename
    if not dataset_name:
        if caller_filename.startswith("_0_"):
            # Extract dataset name from _0_preprocess_xxx.py -> xxx
            dataset_name = caller_filename.replace("_0_preprocess_", "").replace(".py", "")
        else:
            # Fallback to generic name if not found
            dataset_name = "generic"

    # Store dataset information in uns
    adata_rna.uns["file_generated_from"] = caller_file
    adata_rna.uns["dataset_name"] = dataset_name
    adata_prot.uns["file_generated_from"] = caller_file
    adata_prot.uns["dataset_name"] = dataset_name

    # Store stage information based on the original descriptive filename pattern
    if caller_filename.startswith("_0_preprocess"):
        stage_description = "preprocessed"
    elif caller_filename.startswith("_1_align"):
        stage_description = "aligned"
    elif caller_filename.startswith("_2_spatial"):
        stage_description = "spatial_integrated"
    elif caller_filename.startswith("_3_archetype"):
        stage_description = "archetype_generated"
    elif caller_filename.startswith("_4_prepare"):
        stage_description = "prepared_for_training"
    elif caller_filename.startswith("_5_train"):
        stage_description = "trained"
    elif caller_filename.startswith("_6_post"):
        stage_description = "post_hoc_analysis"
    else:
        # Fallback: try to extract meaningful description from filename
        stage_description = caller_filename.replace(".py", "").replace("_", "_")

    adata_rna.uns["processing_stage"] = stage_description
    adata_prot.uns["processing_stage"] = stage_description

    """Save processed data"""
    print("Saving processed data...")
    clean_uns_for_h5ad(adata_prot)
    clean_uns_for_h5ad(adata_rna)

    # Extract step number from caller filename
    step_number = "unknown"
    if caller_filename.startswith("_"):
        # Extract step number from _X_filename.py -> X
        parts = caller_filename.split("_")
        if len(parts) > 1 and parts[1].isdigit():
            step_number = parts[1]

    # Create dataset-specific subdirectory
    save_dir = Path(save_dir)
    dataset_save_dir = save_dir / dataset_name
    dataset_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving to dataset-specific directory: {dataset_save_dir}")

    # Add timestamp to filename
    from datetime import datetime

    timestamp_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    rna_file = dataset_save_dir / f"{step_number}_rna_{timestamp_str}.h5ad"
    prot_file = dataset_save_dir / f"{step_number}_protein_{timestamp_str}.h5ad"

    print(f"\nRNA data dimensions: {adata_rna.shape[0]} samples x {adata_rna.shape[1]} features")
    print(
        f"Protein data dimensions: {adata_prot.shape[0]} samples x {adata_prot.shape[1]} features\n"
    )

    adata_rna.write(rna_file)
    adata_prot.write(prot_file)

    print(f"Saved RNA data: {rna_file} ({rna_file.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Saved protein data: {prot_file} ({prot_file.stat().st_size / (1024*1024):.2f} MB)")


def validate_adata_requirements(adata_rna, adata_prot):
    """
    Validate that both datasets have all required layers and metadata.

    Args:
        adata_rna: RNA-seq AnnData object
        adata_prot: Protein (CODEX) AnnData object

    Raises:
        ValueError: If required data is missing
    """
    print("🔍 Validating dataset requirements...")

    # Check RNA dataset requirements
    rna_errors = []
    if "counts" not in adata_rna.layers:
        rna_errors.append("RNA dataset missing 'counts' layer")
    if "cell_types" not in adata_rna.obs:
        rna_errors.append("RNA dataset missing 'cell_types' in obs")
    if "batch" not in adata_rna.obs:
        rna_errors.append("RNA dataset missing 'batch' in obs")

    # Check protein dataset requirements
    prot_errors = []
    if "counts" not in adata_prot.layers:
        prot_errors.append("Protein dataset missing 'counts' layer")
    if "cell_types" not in adata_prot.obs:
        prot_errors.append("Protein dataset missing 'cell_types' in obs")
    if "batch" not in adata_prot.obs:
        prot_errors.append("Protein dataset missing 'batch' in obs")
    if "spatial" not in adata_prot.obsm:
        prot_errors.append("Protein dataset missing 'spatial' coordinates in obsm")

    # Check for normalization
    if adata_rna.X.min() < 0:
        rna_errors.append("RNA X data appears not normalized (contains negative values)")
    # if adata_prot.X.min() < 0:
    #     prot_errors.append("Protein X data appears not normalized (contains negative values)")

    all_errors = rna_errors + prot_errors
    if all_errors:
        error_msg = "Dataset validation failed:\n" + "\n".join(
            f"  - {error}" for error in all_errors
        )
        raise ValueError(error_msg)

    print("✅ All dataset requirements validated successfully!")
    print(f"RNA dataset: {adata_rna.shape[0]} cells, {adata_rna.shape[1]} genes")
    print(f"Protein dataset: {adata_prot.shape[0]} cells, {adata_prot.shape[1]} proteins")
    print(f"RNA layers: {list(adata_rna.layers.keys())}")
    print(f"Protein layers: {list(adata_prot.layers.keys())}")


# ============================================================================
# Dataset-Specific Loaders
# ============================================================================


# ============================================================================
# CITE-seq Dataset Loaders
# ============================================================================


def read_legacy_adata(file_path: str) -> ad.AnnData:
    """

    Read legacy adata that cannot be loaded (mouse_ICB_scRNA_umap_labelled.h5ad)
    Input: file_path
    Output: adata
    """

    def extract_table(group):
        """Extract DataFrame from group (skip _index)"""
        data = {}
        for key in group.keys():
            if key == "_index":
                continue
            obj = group[key]
            if isinstance(obj, h5py.Dataset):
                data[key] = obj[()]
        return pd.DataFrame(data)

    # EXTRACT WITH CORRECT SHAPE
    with h5py.File(file_path, "r") as f:
        # Get ACTUAL shape from obs/var _index lengths
        obs_index = f["obs"]["_index"][()]
        var_index = f["var"]["_index"][()]
        n_obs = len(obs_index)
        n_vars = len(var_index)
        print(f"Actual shape: {n_obs} cells x {n_vars} genes")

        # Build sparse X with CORRECT shape
        data = f["X"]["data"][()]
        indices = f["X"]["indices"][()]
        indptr = f["X"]["indptr"][()]
        X = csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))

        # Extract obs/var
        obs = extract_table(f["obs"])
        var = extract_table(f["var"])

        # Set original index from _index
        obs.index = pd.Index(obs_index)
        var.index = pd.Index(var_index)

        # UMAP
        umap = f["obsm"]["X_umap"][()]

        # Extract layers if they exist
        layers = {}
        if "layers" in f:
            for layer_name in f["layers"].keys():
                layer_group = f["layers"][layer_name]
                if "data" in layer_group and "indices" in layer_group and "indptr" in layer_group:
                    layer_data = layer_group["data"][()]
                    layer_indices = layer_group["indices"][()]
                    layer_indptr = layer_group["indptr"][()]
                    layers[layer_name] = csr_matrix(
                        (layer_data, layer_indices, layer_indptr), shape=(n_obs, n_vars)
                    )
                elif isinstance(layer_group, h5py.Dataset):
                    layers[layer_name] = layer_group[()]

        print(f"✅ X: {X.shape}, obs: {obs.shape}, var: {var.shape}")
        print(f"✅ UMAP: {umap.shape}")
        if layers:
            print(f"✅ Layers: {list(layers.keys())}")

    # CREATE AnnData
    adata_rna = sc.AnnData(X=X, obs=obs, var=var)
    adata_rna.obsm["X_umap"] = umap

    # Add layers
    for layer_name, layer_data in layers.items():
        adata_rna.layers[layer_name] = layer_data

    print(f"✅ AnnData: {adata_rna.n_obs} cells, {adata_rna.n_vars} genes")

    return adata_rna


def load_cite_seq_data(batches):
    """Load and process CITE-seq spleen lymph node data from scVI.

    Args:
        batches: List of batch names to load (e.g., ["SLN111-D1", "SLN208-D1"])

    Returns:
        AnnData: Combined CITE-seq data with RNA and protein expression
    """
    print("Loading CITE-seq spleen lymph node data...")
    adata = scvi.data.spleen_lymph_cite_seq(save_path="raw_datasets/cite_seq")
    adata = adata[adata.obs["batch"].isin(batches)]
    print(f"Loaded CITE-seq data: {adata.shape}")
    print(f"Available batches: {sorted(set(adata.obs['batch']))}")
    print(f"Available cell types: {sorted(set(adata.obs['cell_types']))}")

    return adata


def load_cite_seq_rna(adata_cite_seq):
    """Load and process RNA data from CITE-seq dataset.

    Args:
        adata_cite_seq: AnnData object from load_cite_seq_data()

    Returns:
        AnnData: RNA data with proper layers and metadata
    """
    print("Processing RNA data from CITE-seq...")

    print("Using all available batches for better cell type diversity")
    print("Available batches and their sizes:")
    for batch in adata_cite_seq.obs["batch"].unique():
        count = (adata_cite_seq.obs["batch"] == batch).sum()
        print(f"  {batch}: {count} cells")

    adata_rna = adata_cite_seq.copy()

    if sp.isspmatrix_coo(adata_rna.X):
        adata_rna.X = adata_rna.X.tocsr()
        print("Converted RNA data to CSR")
    if issparse(adata_rna.X):
        assert np.allclose(adata_rna.X.data, np.round(adata_rna.X.data))
    else:
        assert np.allclose(adata_rna.X, np.round(adata_rna.X))

    adata_rna.layers["raw"] = adata_rna.X.copy()
    adata_rna.layers["counts"] = adata_rna.X.copy()
    print("Added 'counts' and 'raw' layers for RNA dataset")

    if issparse(adata_rna.X):
        adata_rna.obs["total_counts"] = np.array(adata_rna.X.sum(axis=1)).flatten()
    else:
        adata_rna.obs["total_counts"] = adata_rna.X.sum(axis=1)

    return adata_rna


def load_cite_seq_protein(adata_cite_seq, major_to_minor_dict):
    """Load and process protein data with synthetic spatial coordinates.

    Args:
        adata_cite_seq: AnnData object from load_cite_seq_data()
        major_to_minor_dict: Dictionary mapping major cell types to minor subtypes

    Returns:
        AnnData: Protein data with synthetic spatial coordinates
    """
    print("Processing protein data from CITE-seq...")

    adata_prot = ad.AnnData(adata_cite_seq.obsm["protein_expression"], dtype=np.float32)
    adata_prot.obs = adata_cite_seq.obs.copy()

    print("Creating synthetic spatial coordinates...")

    adata_prot = adata_prot[adata_prot.obs["cell_types"].notna()]
    unique_major_cell_types = sorted(set(adata_prot.obs["cell_types"]))
    max_subtypes = 0

    print(f"Found {len(unique_major_cell_types)} major cell types")
    for major_type in unique_major_cell_types:
        major_cells = adata_prot[adata_prot.obs["cell_types"] == major_type]
        n_subtypes = len(set(major_cells.obs["minor_cell_types"]))
        print(f"  {major_type}: {n_subtypes} subtypes")
        max_subtypes = max(max_subtypes, n_subtypes)

    print(f"Maximum subtypes per major type: {max_subtypes}")

    grid_cols = math.ceil(math.sqrt(max_subtypes))
    grid_rows = math.ceil(max_subtypes / grid_cols)

    space_size = 1000
    region_width = space_size // grid_cols
    region_height = space_size // grid_rows

    print(
        f"Creating {grid_rows}x{grid_cols} grid ({grid_rows * grid_cols} regions) based on max subtypes"
    )

    adata_prot.obs["X"] = 0
    adata_prot.obs["Y"] = 0
    adata_prot.obs["spatial_grid_index"] = -1

    regions_used = []

    for subtype_idx in range(max_subtypes):
        print(f"\nPlacing subtype #{subtype_idx + 1} for each major type:")

        grid_row = subtype_idx // grid_cols
        grid_col = subtype_idx % grid_cols

        x_min = grid_col * region_width
        x_max = (grid_col + 1) * region_width
        y_min = grid_row * region_height
        y_max = (grid_row + 1) * region_height

        print(
            f"  Grid position ({grid_row}, {grid_col}) -> region ({x_min}-{x_max}, {y_min}-{y_max})"
        )

        for major_type in unique_major_cell_types:
            major_cells = adata_prot[adata_prot.obs["cell_types"] == major_type]
            minor_types_in_major = sorted(set(major_cells.obs["minor_cell_types"]))

            if subtype_idx < len(minor_types_in_major):
                minor_type = minor_types_in_major[subtype_idx]
                cell_mask = adata_prot.obs["minor_cell_types"] == minor_type
                n_cells = cell_mask.sum()

                if n_cells > 0:
                    x_coords = np.random.randint(x_min, x_max, n_cells)
                    y_coords = np.random.randint(y_min, y_max, n_cells)

                    adata_prot.obs.loc[cell_mask, "X"] = x_coords
                    adata_prot.obs.loc[cell_mask, "Y"] = y_coords
                    adata_prot.obs.loc[cell_mask, "spatial_grid_index"] = subtype_idx

                    print(f"    {major_type} -> {minor_type}: {n_cells} cells")
                    regions_used.append((minor_type, x_min, x_max, y_min, y_max))

    adata_prot.uns["spatial_grid"] = {
        "horizontal_splits": [i * region_width for i in range(grid_cols + 1)],
        "vertical_splits": [i * region_height for i in range(grid_rows + 1)],
        "regions_used": regions_used,
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "max_subtypes": max_subtypes,
    }

    adata_prot.obsm["spatial"] = adata_prot.obs[["X", "Y"]].to_numpy()

    adata_prot.layers["raw"] = adata_prot.X.copy()
    adata_prot.layers["counts"] = adata_prot.X.copy()
    print("Added 'counts' and 'raw' layers for protein dataset")

    adata_prot.obs["batch"] = "cite_seq_batch"
    adata_prot.obs["condition"] = "cite_seq"
    adata_prot.obs["Image"] = "cite_seq_image"
    adata_prot.obs["Sample"] = "cite_seq_sample"

    if issparse(adata_prot.X):
        adata_prot.obs["total_counts"] = np.array(adata_prot.X.sum(axis=1)).flatten()
    else:
        adata_prot.obs["total_counts"] = adata_prot.X.sum(axis=1)

    return adata_prot


# ============================================================================
# Tonsil (MaxFuse) Dataset Loaders
# ============================================================================


def check_tonsil_data_exists(data_dir):
    """Check if required Tonsil data files exist.

    Args:
        data_dir: Path to data directory

    Returns:
        bool: True if all required files exist
    """
    required_files = [
        "tonsil/tonsil_codex.csv",
        "tonsil/tonsil_rna_counts.txt",
        "tonsil/tonsil_rna_names.csv",
        "tonsil/tonsil_rna_meta.csv",
    ]
    return all((data_dir / file).exists() for file in required_files)


def download_tonsil_data(data_dir):
    """Download and extract MaxFuse Tonsil data.

    Args:
        data_dir: Path to data directory
    """
    if check_tonsil_data_exists(data_dir):
        print("Data files already exist, skipping download.")
        return
    print("Downloading MaxFuse data...")
    r = requests.get("http://stat.wharton.upenn.edu/~zongming/maxfuse/data.zip")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(str(data_dir))
    
    # Move all files from {data_dir}/data/tonsil to {data_dir}
    tonsil_inner_dir = data_dir / "data" / "tonsil"
    if tonsil_inner_dir.exists():
        for fname in tonsil_inner_dir.iterdir():
            target = data_dir / fname.name
            if not target.exists():
                fname.rename(target)
        # Optionally, remove the empty data/tonsil/ directory
        try:
            tonsil_inner_dir.rmdir()
            # Try to remove "data" dir too if empty
            (data_dir / "data").rmdir()
        except Exception:
            pass
    # Rename {data_dir}/data to {data_dir}/data_temp if it exists
    data_inner_dir = data_dir / "data"
    data_temp_dir = data_dir / "data_temp"
    if data_inner_dir.exists() and not data_temp_dir.exists():
        data_inner_dir.rename(data_temp_dir)


def load_tonsil_protein(data_dir):
    """Load and process Tonsil protein data.

    Args:
        data_dir: Path to data directory

    Returns:
        AnnData: Protein data with spatial coordinates
    """
    print("Loading protein data...")
    protein = pd.read_csv(data_dir / "tonsil_codex.csv")

    protein_features = [
        "CD38",
        "CD19",
        "CD31",
        "Vimentin",
        "CD22",
        "Ki67",
        "CD8",
        "CD90",
        "CD123",
        "CD15",
        "CD3",
        "CD152",
        "CD21",
        "cytokeratin",
        "CD2",
        "CD66",
        "collagen IV",
        "CD81",
        "HLA-DR",
        "CD57",
        "CD4",
        "CD7",
        "CD278",
        "podoplanin",
        "CD45RA",
        "CD34",
        "CD54",
        "CD9",
        "IGM",
        "CD117",
        "CD56",
        "CD279",
        "CD45",
        "CD49f",
        "CD5",
        "CD16",
        "CD63",
        "CD11b",
        "CD1c",
        "CD40",
        "CD274",
        "CD27",
        "CD104",
        "CD273",
        "FAPalpha",
        "Ecadherin",
    ]

    protein_locations = ["centroid_x", "centroid_y"]
    adata_prot = ad.AnnData(protein[protein_features].to_numpy(), dtype=np.float32)
    adata_prot.var_names = protein_features
    adata_prot.obsm["spatial"] = protein[protein_locations].to_numpy()
    adata_prot.obs["cell_types"] = protein["cluster.term"].to_numpy()

    adata_prot.obs["X"] = protein["centroid_x"].to_numpy()
    adata_prot.obs["Y"] = protein["centroid_y"].to_numpy()

    adata_prot.obs["batch"] = "tonsil_batch"
    adata_prot.obs["condition"] = "tonsil"
    adata_prot.obs["Image"] = "tonsil_image"
    adata_prot.obs["Sample"] = "tonsil_sample"

    return adata_prot


def load_tonsil_rna(data_dir):
    """Load and process Tonsil RNA data.

    Args:
        data_dir: Path to data directory

    Returns:
        AnnData: RNA data with metadata
    """
    print("Loading RNA data...")
    rna = mmread(data_dir / "tonsil_rna_counts.txt")
    rna_names = pd.read_csv(data_dir / "tonsil_rna_names.csv")["names"].to_numpy()
    adata_rna = ad.AnnData(rna.tocsr(), dtype=np.float32)
    adata_rna.var_names = rna_names
    metadata_rna = pd.read_csv(data_dir / "tonsil_rna_meta.csv", index_col=0)
    adata_rna.obs["cell_types"] = metadata_rna["cluster.info"].to_numpy()
    adata_rna.obs.index = metadata_rna.index.to_numpy()

    adata_rna.obs["batch"] = "tonsil_batch"
    adata_rna.obs["treated"] = "tonsil"
    adata_rna.obs["Sample"] = "tonsil_sample"

    return adata_rna


def determine_dataset_path(save_dir, dataset_name):
    """Determine dataset directory and load path for data loading.

    Args:
        save_dir: Base directory for processed data
        dataset_name: Optional dataset name to search in specific subdirectory

    Returns:
        tuple: (save_dir Path, load_dataset_name str or None)
    """
    save_dir = Path(save_dir)
    load_dataset_name = None

    if dataset_name:
        dataset_save_dir = save_dir / dataset_name
        if dataset_save_dir.exists():
            save_dir = dataset_save_dir
            print(f"Loading data from dataset-specific directory: {save_dir}")
        else:
            print(
                f"Dataset directory {dataset_save_dir} not found, using default directory: {save_dir}"
            )
            load_dataset_name = dataset_name

    return save_dir, load_dataset_name
