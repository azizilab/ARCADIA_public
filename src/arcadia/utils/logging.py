import os
import sys
from datetime import timedelta

import mlflow
import numpy as np
import torch
from loguru import logger
from tabulate import tabulate


def filter_and_transform(record):
    # Filter by level
    if record["level"].no < logger.level("INFO").no:
        return False
    # Transform module name
    record["extra"] = {"module_name": record["name"].split(".")[-1]}
    return True


def setup_logger(log_file=None, level="INFO", rotation="500 MB"):
    """Configure logger with custom format and optional file output.

    Args:
        log_file: Single log file path or list of log file paths
        level: Logging level
        rotation: Log rotation size
    """
    # Remove default handler
    logger.remove()

    # Custom format without date and with module name (stripping CODEX_RNA_seq prefix)
    fmt = "<level>{level: <8}</level> | <cyan>{extra[module_name]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    # Add console handler with INFO level
    logger.add(sys.stderr, format=fmt, level="INFO", filter=filter_and_transform)

    # Add file handler(s) if log_file is provided
    if log_file:
        logger.add(
            log_file,
            format=fmt,
            level=level,  # Force INFO level for file too
            rotation=rotation,
            compression="zip",
            enqueue=True,
            filter=filter_and_transform,
        )

    return logger


# Setup default logger with INFO level
logger = setup_logger(level="INFO")

# Create a global logger instance that can be modified
global_logger = logger

# %%
# Imports
# %%


def print_distance_metrics(prot_distances, rna_distances, num_acceptable, num_cells, matching_loss):
    logger.info("\n--- DISTANCE METRICS ---\n")
    table_data = [
        ["Metric", "Value"],
        ["Mean protein distances", f"{prot_distances.mean().item():.4f}"],
        ["Mean RNA distances", f"{rna_distances.mean().item():.4f}"],
        ["Acceptable ratio", f"{num_acceptable.float().item() / num_cells:.4f}"],
        ["Matching loss", f"{matching_loss.item():.4f}"],
    ]
    logger.info("\n" + tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))


def log_epoch_end(current_epoch, train_losses, val_losses, log_to_mlflow=True):
    # Calculate epoch averages
    epoch_avg_train_loss = sum(train_losses) / len(train_losses)
    epoch_avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else np.nan
    logger.info(f"\n--- EPOCH {current_epoch} SUMMARY ---\n")

    table_data = [
        ["Metric", "Value"],
        ["Average train loss", f"{epoch_avg_train_loss:.4f}"],
        ["Average validation loss", f"{epoch_avg_val_loss:.4f}"],
    ]
    logger.info("\n" + tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))

    # Only log to MLflow if explicitly requested (skip during warmup)
    if log_to_mlflow:
        metrics_to_log = {
            "train/epoch_avg_loss": epoch_avg_train_loss,
            "val/epoch_avg_loss": epoch_avg_val_loss,
        }
        mlflow.log_metrics(
            {k: v for k, v in metrics_to_log.items() if not np.isnan(v)},
            step=current_epoch,
        )


def estimate_training_time(rna_cells, prot_cells, params, total_combinations):
    """
    Estimates training time based on dataset sizes and hyperparameters.

    Args:
        rna_cells: Number of RNA cells
        prot_cells: Number of protein cells
        params: Dictionary of hyperparameters
        total_combinations: Total number of parameter combinations to try

    Returns:
        Tuple of (estimated_time_per_iter, total_estimated_time)
    """
    # Base time in seconds based on actual observed runs
    # ~5 minutes per iteration with dataset sizes of ~13k RNA and ~40k protein cells
    base_seconds = 295  # 4min 55sec observed for full dataset with 80 epochs

    # Scale factors based on dataset size
    # Using much gentler scaling based on actual observations
    rna_scale = (rna_cells / 13000) ** 0.5  # Use square root scaling
    prot_scale = (prot_cells / 40000) ** 0.5

    # Hyperparameter scaling factors
    # Actual timing shows epochs have less impact than we initially thought
    epoch_scale = (params["max_epochs"] / 80) ** 0.7  # Reduced impact
    plot_scale = 0.8 + (0.2 * params["plot_x_times"] / 5)  # Minimal impact from plotting
    check_val_scale = 1.0 if params["check_val_every_n_epoch"] > 0 else 0.95  # Minimal impact

    # Scale based on hardware (GPU vs CPU)
    hardware_scale = 1.0 if torch.cuda.is_available() else 3.0

    # Calculate total scale factor
    # Cell scales have less impact than originally thought
    total_scale = (
        ((rna_scale * prot_scale) ** 0.5)
        * epoch_scale
        * plot_scale
        * check_val_scale
        * hardware_scale
    )

    # Estimated time per iteration in seconds
    estimated_seconds_per_iter = base_seconds * total_scale

    # Convert to timedelta
    estimated_time_per_iter = timedelta(seconds=estimated_seconds_per_iter)
    total_estimated_time = estimated_time_per_iter * total_combinations

    return estimated_time_per_iter, total_estimated_time


def save_tabulate_to_txt(losses, global_step, total_steps, is_validation=False):
    """Save losses as a formatted table and log it to MLflow.

    Args:
        losses: Dictionary containing loss values
        global_step: Current global step
        total_steps: Total number of steps
        is_validation: Whether this is validation data
    """
    # Convert tensor values to Python scalars
    losses_to_save = {
        k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.copy().items()
    }

    # Determine modality prefix for filename
    modality = "val" if is_validation else "train"

    # Determine filename based on step
    if global_step is not None:
        last_step = global_step == total_steps - 1 if total_steps is not None else False
    if last_step:
        losses_file = f"{modality}_final_losses.txt"
    else:
        losses_file = f"{modality}_losses_step_{global_step:05d}.txt"

    # Get total loss for percentage calculations
    total_loss = losses_to_save.get("total_loss", 0)

    # Create tabulate table with only main losses
    table_data = [["Loss Type", "Value"]]

    # Define main losses in order
    main_losses = [
        "total_loss",
        "rna_loss",
        "protein_loss",
        "contrastive_loss",
        "matching_loss",
        "similarity_loss",
        "cell_type_clustering_loss",
        "cross_modal_cell_type_loss",
        "cn_distribution_separation_loss",
    ]
    # Format main losses with percentages
    for loss_name in main_losses:
        value = losses_to_save.get(loss_name, 0)

        if loss_name == "total_loss":
            table_data.append([loss_name, f"{value:.4f}"])
        else:
            # Calculate percentage of total
            percentage = (value / total_loss) * 100 if total_loss != 0 else 0
            formatted_value = f"{value:.3f} ({percentage:.1f}%)"
            table_data.append([loss_name, formatted_value])

    # Add iLISI score if available and valid (without percentage)
    if "ilisi_score" in losses_to_save:
        ilisi_score = losses_to_save["ilisi_score"]
        if isinstance(ilisi_score, (int, float)) and np.isfinite(ilisi_score):
            table_data.append(["ilisi_score", f"{ilisi_score:.4f}"])
        else:
            table_data.append(["ilisi_score", "invalid"])

    # Save formatted table to text file (use UTF-8 encoding for fancy_grid Unicode characters)
    with open(losses_file, "w", encoding="utf-8") as f:
        f.write(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))

    # Log to MLflow and clean up
    mlflow.log_artifact(losses_file, f"{modality}_losses")
    os.remove(losses_file)


def log_step(
    losses,
    metrics=None,
    global_step=None,
    current_epoch=None,
    is_validation=False,
    similarity_weight=None,
    similarity_active=None,
    num_acceptable=None,
    num_cells=None,
    latent_distances=None,
    print_to_console=True,
    total_steps=None,
    log_to_mlflow=True,
):
    """Unified function to log and print metrics for both training and validation steps.

    Args:
        losses: Dictionary containing all loss values
        metrics: Dictionary containing additional metrics (iLISI, cLISI, accuracy, etc.)
        global_step: Current global step (optional)
        current_epoch: Current epoch (optional)
        is_validation: Whether this is validation or training
        similarity_weight: Weight for similarity loss (optional for training)
        similarity_active: Whether similarity loss is active (optional for training)
        num_acceptable: Number of acceptable matches (optional for training)
        num_cells: Number of cells (optional for training)
        latent_distances: Latent distances (optional)
        print_to_console: Whether to print metrics to console
    """
    prefix = "Validation " if is_validation else ""
    metrics = metrics or {}

    # Convert tensor values to Python scalars for logging
    def get_value(x):
        if x is None:
            return 0
        return round(x.item(), 4) if isinstance(x, torch.Tensor) else x

    # Extract loss values
    total_loss = get_value(losses.get("total_loss", float("nan")))
    rna_loss = get_value(losses.get("rna_loss", float("nan")))
    protein_loss = get_value(losses.get("protein_loss", float("nan")))
    contrastive_loss = get_value(losses.get("contrastive_loss", float("nan")))
    matching_loss = get_value(losses.get("matching_loss", float("nan")))
    similarity_loss = get_value(losses.get("similarity_loss", float("nan")))
    raw_similarity_loss = get_value(losses.get("raw_similarity_loss", float("nan")))
    cell_type_clustering_loss = get_value(losses.get("cell_type_clustering_loss", float("nan")))
    cross_modal_cell_type_loss = get_value(losses.get("cross_modal_cell_type_loss", float("nan")))
    cn_distribution_separation_loss = get_value(
        losses.get("cn_distribution_separation_loss", float("nan"))
    )
    extreme_alignment_percentage = get_value(
        losses.get("extreme_alignment_percentage", float("nan"))
    )
    mean_per_cell_type_cn_kbet_separation = get_value(
        losses.get("mean_per_cell_type_cn_kbet_separation", float("nan"))
    )
    get_value(losses.get("adversarial_loss", float("nan")))
    get_value(losses.get("diversity_loss", float("nan")))
    reward = get_value(losses.get("reward", float("nan")))

    # Handle parameters that might be in losses dict or passed directly
    num_acceptable = get_value(metrics.get("num_acceptable", float("nan")))
    num_cells = get_value(metrics.get("num_cells", float("nan")))

    # Extract additional metrics
    ilisi_score = get_value(
        losses.get("ilisi_score", float("nan"))
    )  # Extract iLISI score from losses
    # Validate iLISI score
    if not (isinstance(ilisi_score, (int, float)) and np.isfinite(ilisi_score) and ilisi_score > 0):
        ilisi_score = float("nan")

    # clisi_score = get_value(metrics.get("clisi_score", float("nan")))
    accuracy = get_value(metrics.get("accuracy", float("nan")))

    def format_loss(loss, total):
        if loss is None or np.isnan(loss) or np.isinf(loss):
            return None
        percentage = (loss / total) * 100 if total != 0 else 0
        return f"{loss:.3f} ({percentage:.1f}%)"

    def format_loss_mlflow(loss_dict, total=None):
        loss_dict = loss_dict.copy()
        loss_dict = {
            k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()
        }
        return {
            k: round(v, 4) if isinstance(v, (int, float)) else v
            for k, v in loss_dict.items()
            if v is not None and not (isinstance(v, (int, float)) and (np.isnan(v) or np.isinf(v)))
        }

    # Format metrics for printing to console
    if print_to_console:
        save_tabulate_to_txt(
            format_loss_mlflow(losses),
            global_step,
            total_steps,
            is_validation=is_validation,
        )

        logger.info("\n" + "=" * 80)
        step_info = ""
        if global_step is not None:
            step_info += f"Step {global_step}"
        if current_epoch is not None:
            step_info += f", Epoch {current_epoch}" if step_info else f"Epoch {current_epoch}"
        if is_validation:
            logger.info(f"VALIDATION {step_info}")
        else:
            logger.info(f"{step_info}")
            logger.info("=" * 80)

        # Prepare loss data for tabulate
        losses_table = []
        losses_table.append(["Loss Type", "Value"])

        losses_to_print = {
            f"{prefix}RNA Loss": format_loss(rna_loss, total_loss),
            f"{prefix}Protein Loss": format_loss(protein_loss, total_loss),
            f"{prefix}Contrastive Loss": format_loss(contrastive_loss, total_loss),
            f"{prefix}Matching Loss": format_loss(matching_loss, total_loss),
            f"{prefix}Similarity Loss": format_loss(similarity_loss, total_loss),
            f"{prefix}Cell Type Clustering Loss": format_loss(
                cell_type_clustering_loss, total_loss
            ),
            f"{prefix}Cross-Modal Cell Type Loss": format_loss(
                cross_modal_cell_type_loss, total_loss
            ),
            f"{prefix}CN Distribution Separation Loss": format_loss(
                cn_distribution_separation_loss, total_loss
            ),
            f"{prefix}Total Loss": total_loss,
        }

        for loss_name, value in losses_to_print.items():
            if value is not None:
                losses_table.append([loss_name, value])

        logger.info("\nLosses:")
        logger.info("\n" + tabulate(losses_table, headers="firstrow", tablefmt="fancy_grid"))

        # Print additional metrics for training
        if not is_validation:
            similarity_metrics = []
            similarity_metrics.append(["Metric", "Value"])

            similarity_metrics_to_print = {
                f"{prefix}Similarity Loss Raw": raw_similarity_loss,
                f"{prefix}Similarity Weight": similarity_weight,
                f"{prefix}Similarity Active": similarity_active,
                f"{prefix}Num Acceptable": num_acceptable,
                f"{prefix}Num Cells": num_cells,
                f"{prefix}Latent Distances": get_value(latent_distances),
            }

            for metric_name, value in similarity_metrics_to_print.items():
                if value is not None:
                    similarity_metrics.append([metric_name, value])

            logger.info("\nSimilarity Metrics:")
            logger.info(
                "\n" + tabulate(similarity_metrics, headers="firstrow", tablefmt="fancy_grid")
            )

        # Print validation-specific metrics
        if is_validation and latent_distances is not None:
            distance_metrics = []
            distance_metrics.append(["Statistic", "Value"])

            mean_val = get_value(latent_distances)
            distance_metrics.append(["Mean", mean_val])

            if isinstance(latent_distances, torch.Tensor):
                distance_metrics.append(["Min", f"{latent_distances.min().item():.4f}"])
                distance_metrics.append(["Max", f"{latent_distances.max().item():.4f}"])

            logger.info("\nMatching Distances:")
            logger.info(
                "\n" + tabulate(distance_metrics, headers="firstrow", tablefmt="fancy_grid")
            )

        # Print extra metrics if available
        extra_metrics = []
        extra_metrics.append(["Metric", "Value"])

        extra_metrics_to_print = {
            f"{prefix}Reward": reward,
            f"{prefix}iLISI": ilisi_score,  # Use ilisi_score from losses
            # f"{prefix}cLISI": clisi_score,
            f"{prefix}Accuracy": accuracy,
        }

        for metric_name, value in extra_metrics_to_print.items():
            if value is not None and not np.isnan(value):
                extra_metrics.append([metric_name, value])

        # Always print extra metrics section to show iLISI
        if len(extra_metrics) > 1:  # Only if we have at least one valid metric
            logger.info("\nExtra Metrics:")
            logger.info("\n" + tabulate(extra_metrics, headers="firstrow", tablefmt="fancy_grid"))

        logger.info("=" * 80 + "\n")

    # Log to MLflow - Use standard prefixes: "train/" and "val/"
    step = global_step if global_step is not None else None
    mlflow_prefix = "val/" if is_validation else "train/"
    items_to_log = {
        f"{mlflow_prefix}total_loss": total_loss,
        f"{mlflow_prefix}rna_loss": rna_loss,
        f"{mlflow_prefix}protein_loss": protein_loss,
        f"{mlflow_prefix}contrastive_loss": contrastive_loss,
        f"{mlflow_prefix}matching_loss": matching_loss,
        f"{mlflow_prefix}similarity_loss": similarity_loss,
        f"{mlflow_prefix}cell_type_clustering_loss": cell_type_clustering_loss,
        f"{mlflow_prefix}cross_modal_cell_type_loss": cross_modal_cell_type_loss,
        f"{mlflow_prefix}cn_distribution_separation_loss": cn_distribution_separation_loss,
        f"{mlflow_prefix}extreme_alignment_percentage": extreme_alignment_percentage,
        f"{mlflow_prefix}mean_per_cell_type_cn_kbet_separation": mean_per_cell_type_cn_kbet_separation,
        f"{mlflow_prefix}ilisi_score": ilisi_score,
        # f"{mlflow_prefix}clisi": clisi_score,
    }

    # Add training-specific metrics
    if not is_validation and raw_similarity_loss is not None:
        items_to_log[f"{mlflow_prefix}raw_similarity_loss"] = raw_similarity_loss

    if not is_validation and all(
        x is not None for x in [num_acceptable, num_cells, latent_distances]
    ):
        pass
        # items_to_log[f"{prefix}acceptable_ratio"] = (
        #     num_acceptable / num_cells if num_cells > 0 else 0
        # )
        # items_to_log[f"{prefix}latent_distances"] = get_value(latent_distances)

    # Add extra metrics if available

    if log_to_mlflow:
        mlflow.log_metrics(format_loss_mlflow(items_to_log), step=step)

    return items_to_log


def setup_mlflow_experiment(dataset_name, tracking_uri="file:./mlruns"):
    """Setup MLflow experiment with dataset name.

    Args:
        dataset_name: Name of the dataset (used as experiment name)
        tracking_uri: MLflow tracking URI

    Returns:
        tuple: (experiment_name, run_name, mlflow_artifact_dir, log_file_path)
    """
    from datetime import datetime
    from pathlib import Path

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = dataset_name
    mlflow.set_experiment(experiment_name)
    run_name = f"vae_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.start_run(run_name=run_name)

    current_run = mlflow.active_run()
    mlflow_artifact_dir = Path(current_run.info.artifact_uri.replace("file://", ""))
    mlflow_artifact_dir.mkdir(parents=True, exist_ok=True)

    default_log_filename = f"train_vae_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = mlflow_artifact_dir / default_log_filename

    return experiment_name, run_name, mlflow_artifact_dir, log_file_path


def log_training_summary(
    experiment_name,
    run_name,
    mlflow_artifact_dir,
    log_file_path,
    dataset_name,
    save_dir,
    rna_sample_size,
    prot_sample_size,
    max_epochs,
    model_checkpoints_folder=None,
    project_root=None,
):
    """Log training completion summary.

    Args:
        experiment_name: MLflow experiment name
        run_name: MLflow run name
        mlflow_artifact_dir: MLflow artifact directory path
        log_file_path: Path to log file
        dataset_name: Dataset name
        save_dir: Processed data directory
        rna_sample_size: Number of RNA cells trained on
        prot_sample_size: Number of protein cells trained on
        max_epochs: Number of training epochs
        model_checkpoints_folder: Optional checkpoint folder path
        project_root: Optional project root path
    """
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED - Summary of Relevant Paths and Locations")
    logger.info("=" * 80)

    logger.info(f"\n📊 MLflow Tracking:")
    logger.info(f"   Tracking URI: file:./mlruns")
    logger.info(f"   Experiment: {experiment_name}")
    logger.info(f"   Run Name: {run_name}")
    logger.info(f"   Artifact Directory: {mlflow_artifact_dir}")
    logger.info(f"   View results: mlflow ui (from ARCADIA root directory)")
    logger.info(f"   Access at: http://localhost:5000")

    logger.info(f"\n📝 Log Files:")
    logger.info(f"   Training log: {log_file_path}")

    if model_checkpoints_folder:
        logger.info(f"\n💾 Model Checkpoints:")
        logger.info(f"   Checkpoint directory: {model_checkpoints_folder}")
        logger.info(f"   Training resumed from checkpoint: Yes")
    else:
        logger.info(f"\n💾 Model Checkpoints:")
        logger.info(f"   Checkpoint directory: Saved to MLflow artifacts")
        logger.info(f"   Training resumed from checkpoint: No (trained from scratch)")

    logger.info(f"\n📂 Data Paths:")
    if project_root:
        logger.info(f"   Project root: {project_root}")
    logger.info(f"   Processed data directory: {save_dir}")
    logger.info(f"   Dataset name: {dataset_name}")

    logger.info(f"\n🤖 Model Information:")
    logger.info(f"   RNA VAE: Trained on {rna_sample_size} cells")
    logger.info(f"   Protein VAE: Trained on {prot_sample_size} cells")
    logger.info(f"   Training epochs: {max_epochs}")

    logger.info("\n" + "=" * 80)
    logger.info("To view training results, run: mlflow ui")
    logger.info("=" * 80)
