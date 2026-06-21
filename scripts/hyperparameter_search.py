# %% Hyperparameter search for VAE training
"""Hyperparameter search for VAE training with archetypes vectors."""

# Add src to path for arcadia package
import sys
from pathlib import Path

# Handle __file__ for both script and notebook execution
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Running as notebook - use current working directory
    ROOT = Path.cwd().resolve().parent
    if not (ROOT / "src").exists():
        if (ROOT.parent / "src").exists():
            ROOT = ROOT.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: scvi
#     language: python
#     name: python3
# ---

# %% tags=["parameters"]
# Default parameters - can be overridden by papermill
dataset_name = None

# %% Setup and imports
import json
import os
import time
import traceback
import warnings
from datetime import datetime, timedelta

import mlflow
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy.sparse import issparse
from sklearn.model_selection import ParameterGrid

from arcadia.data_utils import load_adata_latest
from arcadia.training import (
    calculate_post_training_metrics,
    clear_memory,
    generate_post_training_visualizations,
    handle_error,
    log_memory_usage,
    log_parameters,
    match_cells_and_calculate_distances,
    process_latent_spaces,
    train_vae,
    validate_scvi_training_mixin,
)
from arcadia.utils.args import (
    apply_baseline_settings,
    apply_command_line_overrides,
    find_latest_checkpoint_folder,
    parse_arguments,
)
from arcadia.utils.environment import get_umap_filtered_fucntion
from arcadia.utils.logging import estimate_training_time, filter_and_transform, setup_logger

# Validate scVI training mixin before proceeding
validate_scvi_training_mixin()

# Set the filename for this script
FILENAME = "hyperparameter_search.py"

# %% Load configuration
config_path = ROOT / "configs" / "config.json"
if config_path.exists():
    with open(config_path, "r") as f:
        config_ = json.load(f)
    num_rna_cells = config_["subsample"]["num_rna_cells"]
    num_protein_cells = config_["subsample"]["num_protein_cells"]
    plot_flag = config_["plot_flag"]
    max_epochs = config_.get("training", {}).get("max_epochs", 400)
else:
    num_rna_cells = num_protein_cells = 2000
    plot_flag = True
    max_epochs = 400

# %% Setup environment
if not hasattr(sc.tl.umap, "_is_wrapped"):
    sc.tl.umap = get_umap_filtered_fucntion()
    sc.tl.umap._is_wrapped = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 10)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="louvain")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
pd.options.display.max_rows = 10
pd.options.display.max_columns = 10
np.set_printoptions(threshold=100)

# %% Parse command line arguments
args = parse_arguments()

# Override dataset_name from command line if provided
if args.dataset_name:
    dataset_name = args.dataset_name

# %% Setup logging
os.makedirs("logs", exist_ok=True)
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
timestamped_log_file = f"logs/hyperparameter_search_{log_timestamp}.log"

logger = setup_logger(level="INFO")
logger.add(
    timestamped_log_file,
    format="<level>{level: <8}</level> | <cyan>{extra[module_name]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    rotation="500 MB",
    compression="zip",
    enqueue=True,
    filter=filter_and_transform,
)

logger.info(f"Starting hyperparameter search at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Log file: {timestamped_log_file}")

# %% Define base parameter grid
base_param_grid = {
    "plot_x_times": [2],
    "check_val_every_n_epoch": [100],
    "max_epochs": [max_epochs],
    "save_checkpoint_every_n_epochs": [280],
    "batch_size": [1024],
    "print_every_n_epoch": [50],
    "plot_first_step": [False],
    "outlier_detection_enabled": [True],
    "contrastive_weight": [0],
    "similarity_weight": [0],
    "similarity_dynamic": [False],
    "diversity_weight": [0.0],
    "matching_weight": [0.1,1,10],
    "cell_type_clustering_weight": [0.1,1,10],
    "cross_modal_cell_type_weight": [0.1,1,10],
    "cn_distribution_separation_weight": [0],
    "n_hidden_rna": [1024],  # 1024 gave better results than 256 layers (f1 score)
    "n_hidden_prot": [512],  # 512 was better than 256 layers in kbet score
    "n_layers": [3],  # 5 layers learned too slow and resulted in low ilisi score
    "latent_dim": [60],  # 60 did not seem better than 30
    "dropout_rate": [0.1],
    "rna_recon_weight": [1],
    "prot_recon_weight": [10],
    "adv_weight": [0.0],
    "train_size": [0.80],
    "validation_size": [0.20],
    "gradient_clip_val": [1.0],
    "load_optimizer_state": [False],
    "override_skip_warmup": [False],
    "lr": [1e-3],
    "gradnorm_enabled": [False],
    "gradnorm_alpha": [1.5],
    "gradnorm_lr": [0.025],
}

# %% Apply command line overrides and baseline settings
param_grid = base_param_grid.copy()
param_grid = apply_command_line_overrides(param_grid, args, logger)
param_grid = apply_baseline_settings(param_grid, args, logger)
combinations_to_skip = []

# %% Load data
adata_rna_subset, adata_prot_subset = load_adata_latest(
    "processed_data",
    ["rna", "protein"],
    exact_step=4,
    index_from_end=0,
    dataset_name=dataset_name,
)

# %% Convert data to sparse format
logger.info("Converting data matrices to sparse format for memory efficiency...")
log_memory_usage("Before sparse conversion: ")

if not issparse(adata_rna_subset.X):
    logger.info(f"Converting RNA X from {type(adata_rna_subset.X)} to sparse CSR matrix")
    adata_rna_subset.X = sp.csr_matrix(adata_rna_subset.X)
else:
    logger.info(f"RNA X already sparse: {type(adata_rna_subset.X)}")

if not issparse(adata_prot_subset.X):
    logger.info(f"Converting protein X from {type(adata_prot_subset.X)} to sparse CSR matrix")
    adata_prot_subset.X = sp.csr_matrix(adata_prot_subset.X)
else:
    logger.info(f"Protein X already sparse: {type(adata_prot_subset.X)}")

if adata_rna_subset.X.dtype != np.float32:
    adata_rna_subset.X = adata_rna_subset.X.astype(np.float32)
if adata_prot_subset.X.dtype != np.float32:
    adata_prot_subset.X = adata_prot_subset.X.astype(np.float32)

log_memory_usage("After sparse conversion: ")

# %% Setup MLflow experiment
mlflow.set_tracking_uri("file:./mlruns")
dataset_name_mlflow = adata_rna_subset.uns.get("dataset_name", "unknown")
experiment_name = dataset_name_mlflow

experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    logger.info(f"Created new MLflow experiment: {experiment_name} with ID: {experiment_id}")
else:
    experiment_id = experiment.experiment_id

mlflow.set_experiment(experiment_name)

# %% Filter out already tried parameter combinations
existing_runs = mlflow.search_runs(experiment_ids=[experiment_id])
existing_params = []
for _, run in existing_runs.iterrows():
    run_params = {}
    for param in param_grid.keys():
        param_key = f"params.{param}"
        if (
            run.status == "FINISHED"
            and param_key in run.index
            and param not in ["plot_x_times", "check_val_every_n_epoch", "max_epochs"]
        ):
            run_params[param] = run[param_key]
    if run_params:
        existing_params.append(run_params)

all_combinations = list(ParameterGrid(param_grid))
new_combinations = []
for combo in all_combinations:
    combo_to_check = {
        k: v
        for k, v in combo.items()
        if k
        not in [
            "plot_x_times",
            "check_val_every_n_epoch",
            "max_epochs",
            "save_checkpoint_every_n_epochs",
        ]
    }

    should_skip = False
    for skip_combo in combinations_to_skip:
        if all(combo.get(key) == value for key, value in skip_combo.items()):
            should_skip = True
            break

    if not should_skip and combo_to_check not in existing_params:
        new_combinations.append(combo)

skipped_combinations = len(all_combinations) - len(new_combinations) - len(existing_params)
total_combinations = len(new_combinations)
logger.info(f"Total combinations: {len(all_combinations)}")
logger.info(f"Already tried: {len(existing_params)}")
logger.info(f"Manually skipped: {skipped_combinations}")
logger.info(f"New combinations to try: {total_combinations}")

# %% Subsample data
sc.pp.subsample(adata_rna_subset, n_obs=min(num_rna_cells, adata_rna_subset.shape[0]))
sc.pp.subsample(adata_prot_subset, n_obs=min(num_protein_cells, adata_prot_subset.shape[0]))

logger.info(f"RNA data shape: {adata_rna_subset.shape}")
logger.info(f"Protein data shape: {adata_prot_subset.shape}")

adata_rna_subset.obs["cell_types"] = adata_rna_subset.obs["major_cell_types"]
adata_prot_subset.obs["cell_types"] = adata_prot_subset.obs["major_cell_types"]
log_memory_usage("After loading protein data: ")

logger.info(f"RNA dataset shape: {adata_rna_subset.shape}")
logger.info(f"Protein dataset shape: {adata_prot_subset.shape}")

# %% Handle checkpoint folder selection
if args.resume_from_latest_checkpoint:
    logger.info("Searching for latest checkpoint folder...")
    model_checkpoints_folder = find_latest_checkpoint_folder(logger, dataset_name)
    if model_checkpoints_folder is None:
        logger.error("Failed to find latest checkpoint folder. Training will start from scratch.")
        raise ValueError(
            "Failed to find latest checkpoint folder. Training will start from scratch."
        )
else:
    model_checkpoints_folder = None

# Manual checkpoint override (if needed)
if False:  # manual if needed
    model_checkpoints_folder = Path(
        "/home/barroz/projects/ARCADIA/mlruns/886491203711373539/1455a14c3edf44a6bb4ee7adf0a97901/artifacts/checkpoints/epoch_0200"
    )

# %% Estimate training time
if total_combinations > 0 and len(new_combinations) > 0:
    first_params = new_combinations[0]
    rna_cells = adata_rna_subset.shape[0]
    prot_cells = adata_prot_subset.shape[0]

    logger.info(
        f"\n--- Time Estimation for {rna_cells} RNA cells and {prot_cells} protein cells ---"
    )
    time_per_iter, total_time = estimate_training_time(
        rna_cells, prot_cells, first_params, total_combinations
    )

    logger.info(f"Estimated time per iteration: {time_per_iter}")
    days = total_time.days
    hours = total_time.seconds // 3600
    minutes = (total_time.seconds % 3600) // 60
    logger.info(
        f"Estimated total time for {total_combinations} combinations: {days} days, {hours} hours, {minutes} minutes"
    )
    logger.info(
        f"Estimated completion time: {(datetime.now() + total_time).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info("Note: This is a rough estimate based on dataset size and hyperparameters")
    logger.info("Actual times may vary based on system load and other factors")
    logger.info("More accurate estimates will be provided after the first iteration completes")
    logger.info("------------------------------------------------------------\\n")

logger.info(f"Subsampled RNA dataset shape: {adata_rna_subset.shape}")
logger.info(f"Subsampled protein dataset shape: {adata_prot_subset.shape}")

# %% Run hyperparameter search
results = []
logger.info(f"Number of new combinations to try: {total_combinations}")

start_time = datetime.now()
elapsed_times = []

for i, params in enumerate(new_combinations):
    if os.path.exists(ROOT / "scales_cache.json"):
        os.remove(ROOT / "scales_cache.json")

    run_name = f"vae_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow_run_status = "FINISHED"
    iter_start_time = datetime.now()
    log_memory_usage(f"Start of iteration {i+1}: ")

    # %% Start MLflow run
    with mlflow.start_run(run_name=run_name):
        current_run = mlflow.active_run()
        mlflow_artifact_dir = Path(current_run.info.artifact_uri.replace("file://", ""))
        mlflow_artifact_dir.mkdir(parents=True, exist_ok=True)

        run_log_file_path = mlflow_artifact_dir / f"{run_name}.log"
        run_logger = setup_logger(log_file=str(run_log_file_path), level="INFO")

        run_logger.info(
            f"\n--- Run {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.2f}%) ---"
        )
        log_parameters(params, i, total_combinations)

        # %% Calculate time estimates
        if i > 0 and len(elapsed_times) > 0:
            total_time = timedelta(0)
            for t in elapsed_times:
                total_time += t
            avg_time_per_iter = total_time / len(elapsed_times)

            remaining_iters = total_combinations - (i + 1)
            est_remaining_time = avg_time_per_iter * remaining_iters

            elapsed_total = datetime.now() - start_time
            est_total_time = elapsed_total + est_remaining_time

            run_logger.info(
                f"\nTiming Information:\n"
                f"├─ Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"├─ Current: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"├─ Elapsed: {elapsed_total}\n"
                f"├─ Avg time/iter: {avg_time_per_iter}\n"
                f"└─ Est. remaining: {est_remaining_time}"
            )

        params["log_file_path"] = str(run_log_file_path)

        run_logger.info(
            f"Starting run {run_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        run_logger.info(
            f"Log file will be written directly to MLflow artifacts: {run_log_file_path}"
        )

        # %% Train model
        try:
            loss_weights = {
                "rna_recon_weight": params["rna_recon_weight"],
                "prot_recon_weight": params["prot_recon_weight"],
                "contrastive_weight": params["contrastive_weight"],
                "similarity_weight": params["similarity_weight"],
                "similarity_dynamic": params["similarity_dynamic"],
                "matching_weight": params["matching_weight"],
                "cell_type_clustering_weight": params["cell_type_clustering_weight"],
                "cross_modal_cell_type_weight": params["cross_modal_cell_type_weight"],
                "cn_distribution_separation_weight": params["cn_distribution_separation_weight"],
            }

            loss_weights_path = "loss_weights.json"
            with open(loss_weights_path, "w") as f:
                json.dump(loss_weights, f, indent=4)

            mlflow.log_artifact(loss_weights_path)
            if os.path.exists(loss_weights_path):
                os.remove(loss_weights_path)

            mlflow.log_params(params)

            mlflow.log_param(
                "rna_dataset_shape", f"{adata_rna_subset.shape[0]}x{adata_rna_subset.shape[1]}"
            )
            mlflow.log_param(
                "protein_dataset_shape",
                f"{adata_prot_subset.shape[0]}x{adata_prot_subset.shape[1]}",
            )
            mlflow.log_param("dataset_name", adata_rna_subset.uns.get("dataset_name", "unknown"))

            trained_from_scratch = model_checkpoints_folder is None
            mlflow.log_param("trained_from_scratch", trained_from_scratch)

            rna_vae, protein_vae = train_vae(
                adata_rna_subset=adata_rna_subset,
                adata_prot_subset=adata_prot_subset,
                model_checkpoints_folder=model_checkpoints_folder,
                **params,
            )

            # %% Log training metrics
            clear_memory()
            history_ = rna_vae._training_plan.get_history()

            metrics_to_log = {}
            metric_mapping = {
                "final_train_similarity_loss": "train_similarity_loss",
                "final_train_raw_similarity_loss": "train_raw_similarity_loss",
                "final_train_total_loss": "train_total_loss",
                "final_val_total_loss": "val_total_loss",
                "final_train_cell_type_clustering_loss": "train_cell_type_clustering_loss",
                "final_val_rna_loss": "val_rna_loss",
                "final_val_protein_loss": "val_protein_loss",
                "final_val_contrastive_loss": "val_contrastive_loss",
                "final_val_matching_loss": "val_matching_loss",
                "final_val_similarity_loss": "val_similarity_loss",
                "final_val_raw_similarity_loss": "val_raw_similarity_loss",
                "final_val_cell_type_clustering_loss": "val_cell_type_clustering_loss",
                "final_val_cross_modal_cell_type_loss": "val_cross_modal_cell_type_loss",
                "final_train_cn_distribution_separation_loss": "train_cn_distribution_separation_loss",
                "final_val_cn_distribution_separation_loss": "val_cn_distribution_separation_loss",
            }

            for key, hist_key in metric_mapping.items():
                if hist_key in history_ and history_[hist_key] and len(history_[hist_key]) > 0:
                    metrics_to_log[key] = history_[hist_key][-1]
                else:
                    logger.warning(
                        f"Skipping metric {key} - no data available in training history for {hist_key}"
                    )

            if metrics_to_log:
                mlflow.log_metrics(metrics_to_log)
                logger.info(
                    f"Logged {len(metrics_to_log)} final metrics (skipped {len(metric_mapping) - len(metrics_to_log)} NaN metrics)"
                )
            else:
                logger.warning(
                    "No final metrics to log - training may have been too short or validation didn't run"
                )

            # %% Process latent spaces and calculate metrics
            adata_rna = rna_vae.adata
            adata_prot = protein_vae.adata
            adata_rna = sc.pp.subsample(adata_rna, n_obs=min(len(adata_rna), 4000), copy=True)
            adata_prot = sc.pp.subsample(adata_prot, n_obs=min(len(adata_prot), 4000), copy=True)

            rna_latent, prot_latent, combined_latent = process_latent_spaces(adata_rna, adata_prot)
            matching_results = match_cells_and_calculate_distances(rna_latent, prot_latent)
            metrics = calculate_post_training_metrics(
                adata_rna, adata_prot, matching_results["prot_matches_in_rna"]
            )

            valid_metrics = {}
            nan_metrics = []
            for k, v in metrics.items():
                if v is not None and not (
                    isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf"))
                ):
                    valid_metrics[k] = round(v, 3)
                else:
                    nan_metrics.append(k)

            if valid_metrics:
                mlflow.log_metrics(valid_metrics)
                logger.info(f"Logged {len(valid_metrics)} post-training metrics")

            if nan_metrics:
                logger.warning(
                    f"Skipped {len(nan_metrics)} NaN/invalid post-training metrics: {nan_metrics}"
                )

            # %% Generate visualizations
            generate_post_training_visualizations(
                adata_rna,
                adata_prot,
                rna_latent,
                prot_latent,
                combined_latent,
                history_,
                matching_results,
            )

            iter_time = datetime.now() - iter_start_time
            elapsed_times.append(iter_time)
            run_logger.info(f"\nIteration completed in: {iter_time}")
            run_logger.info(f"Log file already in MLflow artifacts: {run_log_file_path}")
            mlflow.log_param("run_failed", False)

        except KeyboardInterrupt:
            mlflow.end_run(status="FAILED")
            run_logger.info("Run was manually terminated by user (KeyboardInterrupt).")
            raise

        except Exception as e:
            mlflow_run_status = "FAILED"
            error_log_path = mlflow_artifact_dir / f"{run_name}_error.json"
            with open(error_log_path, "w") as error_file:
                error_file.write(
                    f"Error occurred at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                )
                str_params = {k: str(v) for k, v in params.items()}
                error_file.write(f"Parameters: {json.dumps(str_params, indent=2)}\n\n")
                error_file.write(f"Error message: {str(e)}\n\n")
                error_file.write("Traceback:\n")
                error_file.write(traceback.format_exc())

            run_logger.error(f"Logged detailed error information to: {error_log_path}")
            handle_error(e, params, run_name)
            run_logger.error(f"Error log already in MLflow artifacts: {error_log_path}")
            run_logger.error(f"Log file already in MLflow artifacts: {run_log_file_path}")
            time.sleep(5)
            continue

    clear_memory()
    log_memory_usage(f"End of iteration {i+1}: ")

# %% Save results
results_df = pd.DataFrame(results)
results_df.to_csv(
    f"hyperparameter_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False
)

logger.success(
    f"\nHyperparameter search completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
