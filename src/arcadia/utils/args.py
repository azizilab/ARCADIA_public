"""Argument parsing utilities for hyperparameter search."""

import argparse
import glob
import os
from pathlib import Path


def parse_arguments():
    """Parse command line arguments for hyperparameter search."""
    parser = argparse.ArgumentParser(description="Hyperparameter search for VAE training")

    # Add arguments for all possible grid parameters
    grid_params = [
        "plot_x_times",
        "check_val_every_n_epoch",
        "max_epochs",
        "save_checkpoint_every_n_epochs",
        "batch_size",
        "print_every_n_epoch",
        "plot_first_step",
        "outlier_detection_enabled",
        "contrastive_weight",
        "similarity_weight",
        "similarity_dynamic",
        "diversity_weight",
        "matching_weight",
        "cell_type_clustering_weight",
        "cross_modal_cell_type_weight",
        "cn_distribution_separation_weight",
        "n_hidden_rna",
        "n_hidden_prot",
        "n_layers",
        "latent_dim",
        "dropout_rate",
        "rna_recon_weight",
        "prot_recon_weight",
        "adv_weight",
        "train_size",
        "validation_size",
        "gradient_clip_val",
        "load_optimizer_state",
        "override_skip_warmup",
        "lr",
        "gradnorm_enabled",
        "gradnorm_alpha",
        "gradnorm_lr",
    ]

    for param in grid_params:
        parser.add_argument(f"--{param}", type=str, help=f"Override {param} in parameter grid")

    # Add resume from latest checkpoint argument
    parser.add_argument(
        "--resume_from_latest_checkpoint",
        action="store_true",
        help="Resume training from the latest checkpoint found in MLflow runs",
    )

    # Add baseline model argument
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Train baseline model with only similarity, RNA, and protein reconstruction weights",
    )

    # Add dataset name argument
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Specify which dataset to load (if not provided, defaults to None)",
    )

    return parser.parse_args()


def parse_pipeline_arguments():
    """Parse command line arguments for pipeline scripts."""
    parser = argparse.ArgumentParser(description="Pipeline script with dataset name support")

    # Add dataset name argument
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Specify which dataset to load (if not provided, defaults to None)",
    )

    # Add model checkpoints folder argument
    parser.add_argument(
        "--model_checkpoints_folder",
        type=str,
        default=None,
        help="Path to folder for saving model checkpoints (if not provided, defaults to None)",
    )

    return parser.parse_args()


def convert_arg_type(value, param_name):
    """Convert string argument to appropriate type based on parameter name."""
    if value is None:
        return None

    # Boolean parameters
    bool_params = [
        "plot_first_step",
        "outlier_detection_enabled",
        "similarity_dynamic",
        "load_optimizer_state",
        "override_skip_warmup",
        "gradnorm_enabled",
    ]
    if param_name in bool_params:
        return value.lower() in ["true", "1", "yes", "on"]

    # Integer parameters
    int_params = [
        "plot_x_times",
        "check_val_every_n_epoch",
        "max_epochs",
        "save_checkpoint_every_n_epochs",
        "batch_size",
        "print_every_n_epoch",
        "n_hidden_rna",
        "n_hidden_prot",
        "n_layers",
        "latent_dim",
    ]
    if param_name in int_params:
        return int(value)

    # Float parameters
    float_params = [
        "contrastive_weight",
        "similarity_weight",
        "diversity_weight",
        "matching_weight",
        "cell_type_clustering_weight",
        "cross_modal_cell_type_weight",
        "cn_distribution_separation_weight",
        "dropout_rate",
        "rna_recon_weight",
        "prot_recon_weight",
        "adv_weight",
        "train_size",
        "validation_size",
        "gradient_clip_val",
        "lr",
        "gradnorm_alpha",
        "gradnorm_lr",
    ]
    if param_name in float_params:
        return float(value)

    # Default to string
    return value


def find_latest_checkpoint_folder(logger, dataset_name=None):
    """Find the latest checkpoint folder from MLflow runs with actual model files, ignoring deleted experiments."""
    import yaml

    mlruns_path = "mlruns"
    if not os.path.exists(mlruns_path):
        logger.warning(f"MLflow runs directory not found: {mlruns_path}")
        return None

    # Find all checkpoint folders in all experiments and runs
    checkpoint_pattern = os.path.join(mlruns_path, "*", "*", "artifacts", "checkpoints", "epoch_*")
    checkpoint_folders = glob.glob(checkpoint_pattern)

    if not checkpoint_folders:
        logger.warning("No checkpoint folders found in MLflow runs")
        return None

    # Filter out checkpoints from deleted experiments/runs
    valid_checkpoints = []
    for checkpoint_folder in checkpoint_folders:
        # Extract experiment and run IDs from path
        # Path format: mlruns/experiment_id/run_id/artifacts/checkpoints/epoch_*
        path_parts = Path(checkpoint_folder).parts
        if len(path_parts) >= 3:
            experiment_id = path_parts[-5]  # mlruns/experiment_id/run_id/artifacts/checkpoints
            run_id = path_parts[-4]

            # Check dataset_name parameter if provided
            if dataset_name is not None:
                params_path = os.path.join(mlruns_path, experiment_id, run_id, "params")
                dataset_name_param_path = os.path.join(params_path, "dataset_name")
                if os.path.exists(dataset_name_param_path):
                    with open(dataset_name_param_path, "r") as f:
                        run_dataset_name = f.read().strip()
                    if run_dataset_name != dataset_name:
                        logger.debug(
                            f"Skipping checkpoint from run {run_id} with dataset_name={run_dataset_name} (looking for {dataset_name})"
                        )
                        continue
                else:
                    logger.debug(
                        f"Skipping checkpoint from run {run_id} with no dataset_name parameter"
                    )
                    continue

            # Check if experiment is deleted
            experiment_meta_path = os.path.join(mlruns_path, experiment_id, "meta.yaml")
            if os.path.exists(experiment_meta_path):
                try:
                    with open(experiment_meta_path, "r") as f:
                        experiment_meta = yaml.safe_load(f)
                    if experiment_meta.get("lifecycle_stage") == "deleted":
                        logger.debug(f"Skipping checkpoint from deleted experiment {experiment_id}")
                        continue
                except Exception as e:
                    logger.warning(f"Could not read experiment meta.yaml for {experiment_id}: {e}")

            # Check if run is deleted
            run_meta_path = os.path.join(mlruns_path, experiment_id, run_id, "meta.yaml")
            if os.path.exists(run_meta_path):
                try:
                    with open(run_meta_path, "r") as f:
                        run_meta = yaml.safe_load(f)
                    if run_meta.get("lifecycle_stage") == "deleted":
                        logger.debug(f"Skipping checkpoint from deleted run {run_id}")
                        continue
                except Exception as e:
                    logger.warning(f"Could not read run meta.yaml for {run_id}: {e}")

            valid_checkpoints.append(checkpoint_folder)

    if not valid_checkpoints:
        logger.warning(
            "No valid checkpoint folders found (all may be from deleted experiments/runs)"
        )
        return None

    # Sort by modification time to get the latest first
    valid_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # Try each checkpoint folder in order until we find one with actual model files
    for i, checkpoint_folder in enumerate(valid_checkpoints):
        if os.path.exists(checkpoint_folder):
            # Check for common checkpoint files
            checkpoint_files = os.listdir(checkpoint_folder)
            model_files = [f for f in checkpoint_files if f.endswith((".ckpt", ".pt", ".pth"))]

            if model_files:
                logger.info(f"Found checkpoint folder with model files: {checkpoint_folder}")
                logger.info(f"Checkpoint contains {len(model_files)} model files: {model_files}")
                if i > 0:
                    logger.info(
                        f"Note: Skipped {i} more recent checkpoint folders that had no model files"
                    )
                return Path(checkpoint_folder)
            else:
                logger.debug(
                    f"Checkpoint folder exists but contains no model files, trying next: {checkpoint_folder}"
                )
        else:
            logger.debug(f"Checkpoint folder path does not exist, trying next: {checkpoint_folder}")

    # If we get here, no checkpoint folders contained actual model files
    logger.warning("No checkpoint folders found with actual model files")
    return None


def apply_command_line_overrides(param_grid, args, logger):
    """Apply command line argument overrides to parameter grid."""
    for param_name in param_grid.keys():
        arg_value = getattr(args, param_name, None)
        if arg_value is not None:
            converted_value = convert_arg_type(arg_value, param_name)
            param_grid[param_name] = [converted_value]
            logger.info(f"Overriding {param_name} with command line value: {converted_value}")
    return param_grid


def apply_baseline_settings(param_grid, args, logger):
    """Apply baseline model settings if requested."""
    if args.baseline:
        logger.info(
            "Applying baseline model settings - setting all weights to 0 except similarity (100000), RNA, and protein reconstruction"
        )
        baseline_overrides = {
            "contrastive_weight": [0],
            "diversity_weight": [0.0],
            "matching_weight": [0],
            "cell_type_clustering_weight": [0],
            "cross_modal_cell_type_weight": [0],
            "cn_distribution_separation_weight": [0],
            "adv_weight": [0.0],
            "similarity_weight": [1],  # Set similarity weight to 100000 for baseline
            # Keep these weights as they are (RNA, protein reconstruction)
            # "rna_recon_weight": param_grid["rna_recon_weight"],
            # "prot_recon_weight": param_grid["prot_recon_weight"],
        }

        # Adjust validation frequency for short training runs
        current_max_epochs = param_grid.get("max_epochs", [700])[0]
        current_check_val = param_grid.get("check_val_every_n_epoch", [50])[0]

        # If training is short, ensure validation runs at least once
        if current_max_epochs < current_check_val:
            baseline_overrides["check_val_every_n_epoch"] = [max(1, current_max_epochs // 2)]
            logger.info(
                f"Adjusted validation frequency for short training: check_val_every_n_epoch = {baseline_overrides['check_val_every_n_epoch'][0]}"
            )

        for param_name, value in baseline_overrides.items():
            param_grid[param_name] = value
            logger.info(f"Baseline override: {param_name} = {value[0]}")

    return param_grid


def parse_compare_arguments():
    """
    Parse command line arguments for comparison script.

    Returns:
        tuple: (args, unknown) - parsed arguments and unknown arguments

    Example:
        >>> args, unknown = parse_compare_arguments()
        >>> print(args.experiment_name)
        'cite_seq'
    """
    parser = argparse.ArgumentParser(description="Compare ARCADIA results with baseline models")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to ARCADIA checkpoint folder (e.g., epoch_0499)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="MLflow experiment name (alternative to checkpoint_path)",
    )
    parser.add_argument(
        "--other_model_name",
        type=str,
        default=None,
        help="Name of baseline model to compare against (default: maxfuse)",
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default=None,
        help="MLflow experiment ID for tracking",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="MLflow run ID for tracking",
    )
    return parser.parse_known_args()


def get_default_compare_args():
    """Get default arguments for comparison script (for notebook/interactive use)."""

    class Args:
        checkpoint_path = None
        experiment_name = None
        other_model_name = "maxfuse"
        experiment_id = None
        run_id = None

    return Args()


def find_checkpoint_from_experiment_name(experiment_name, max_results=5):
    """
    Find the latest checkpoint from an MLflow experiment by name using MLflow API.

    Args:
        experiment_name: Name of the MLflow experiment (e.g., "cite_seq", "tonsil")
        max_results: Maximum number of recent runs to check (default: 5)

    Returns:
        tuple: (checkpoint_path, experiment_id, run_id)
            - checkpoint_path: Full path to the latest checkpoint folder
            - experiment_id: MLflow experiment ID
            - run_id: MLflow run ID

    Raises:
        ValueError: If experiment not found or no valid checkpoints exist

    Example:
        >>> checkpoint, exp_id, run_id = find_checkpoint_from_experiment_name("cite_seq")
        >>> print(checkpoint)
        '/home/user/projects/ARCADIA/mlruns/549776.../artifacts/checkpoints/epoch_0499'
    """
    import mlflow

    # Set tracking URI
    mlflow.set_tracking_uri("file:./mlruns")

    # Get experiment by name
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    # Search for latest runs in the experiment
    df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=max_results,
    )

    if df.empty:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    # Find the first run with a valid checkpoint
    for _, run in df.iterrows():
        run_id = run["run_id"]
        artifact_uri = run["artifact_uri"]

        # Convert artifact_uri to local path
        artifact_path = Path(artifact_uri.replace("file://", ""))
        checkpoints_dir = artifact_path / "checkpoints"

        if checkpoints_dir.exists():
            # Find all epoch folders
            epoch_folders = [
                d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")
            ]

            # Find valid checkpoints (with both adata files)
            valid_checkpoints = []
            for epoch_folder in epoch_folders:
                rna_file = epoch_folder / "adata_rna.h5ad"
                prot_file = epoch_folder / "adata_prot.h5ad"
                if rna_file.exists() and prot_file.exists():
                    valid_checkpoints.append(epoch_folder)

            if valid_checkpoints:
                # Get the latest checkpoint (highest epoch number)
                def extract_epoch_num(folder):
                    try:
                        return int(folder.name.split("_")[1])
                    except (IndexError, ValueError):
                        return 0

                latest_checkpoint = max(valid_checkpoints, key=extract_epoch_num)
                return str(latest_checkpoint.resolve()), exp.experiment_id, run_id

    raise ValueError(f"No valid checkpoints found in experiment '{experiment_name}'")


def parse_batch_performance_arguments():
    """
    Parse command line arguments for batch performance analysis script.

    Returns:
        Namespace: parsed arguments with dataset_names, num_runs, skip_first_n_runs, other_model_name

    Example:
        >>> args = parse_batch_performance_arguments()
        >>> print(args.dataset_names)
        ['tonsil', 'cite_seq']
        >>> print(args.num_runs)
        1
        >>> print(args.skip_first_n_runs)
        0
        >>> print(args.other_model_name)
        'maxfuse'
    """
    parser = argparse.ArgumentParser(
        description="Batch analyze ARCADIA performance from MLflow experiments (each experiment = one dataset)"
    )
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        default=["tonsil", "cite_seq"],
        help="Dataset/experiment names to process (e.g., 'cite_seq' 'tonsil') (default: ['tonsil', 'cite_seq'])",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of latest runs to take from each dataset/experiment (default: 1)",
    )
    parser.add_argument(
        "--skip_first_n_runs",
        type=int,
        default=0,
        help="Skip the first N runs (useful for resuming batch processing) (default: 0)",
    )
    parser.add_argument(
        "--other_model_name",
        type=str,
        default="maxfuse",
        help="Baseline model to compare against (e.g., 'maxfuse', 'scmodal') (default: 'maxfuse')",
    )
    return parser.parse_args()
