# %% Processing

# Add src to path for arcadia package
import sys
from pathlib import Path

# Handle __file__ for both script and notebook execution
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Running as notebook - use current working directory
    # papermill sets cwd to the script directory
    ROOT = Path.cwd().resolve().parent
    if not (ROOT / "src").exists():
        # Try parent if we're in scripts/
        if (ROOT.parent / "src").exists():
            ROOT = ROOT.parent

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

"""Dual VAE Training Script

This script trains dual variational autoencoders (VAEs) for RNA and protein modalities
using a custom training plan that incorporates archetype-based alignment.

Key Operations:
1. Load training-ready data from Step 4
2. Set up MLflow experiment tracking
3. Configure training hyperparameters (from config.json or defaults)
4. Train dual VAEs with custom loss functions:
   - Reconstruction losses for each modality
   - Matching loss: Aligns cells with similar archetype profiles across modalities
   - Cell type clustering loss: Ensures cell types cluster in latent space
   - Cross-modal cell type alignment loss
   - CN distribution separation loss (optional)
   - Similarity loss (optional, adaptive based on iLISI score)
5. Generate visualizations during training (if plot_x_times > 0):
   - Latent space PCA/UMAP visualizations
   - CN-specific cell type UMAPs
   - Counterfactual predictions (RNA→Protein, Protein→RNA)
6. Calculate post-training metrics:
   - Matching accuracy between modalities
   - Silhouette F1 and ARI scores
   - iLISI for integration quality
7. Save model checkpoints and log all results to MLflow

Note: Requires scVI library patch for DualVAETrainingPlan support.
See script comments for patching instructions.

Outputs:
- Model checkpoints saved to checkpoints/ directory
- Training metrics and visualizations logged to MLflow
- Latent representations stored in AnnData objects
"""

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

FILENAME = "_5_train_vae.py"

""" DO NOT REMOVE THIS COMMENT!!!
TO use this script, you need to add the training plan to use the DualVAETrainingPlan (version 1.2.2.post2) class in scVI library.
in _training_mixin.py, line 131, you need to change the line:
training_plan = self._training_plan_cls(self.module, **plan_kwargs) # existing line
self._training_plan = training_plan # add this line

"""
import json
import os
from pathlib import Path

import matplotlib as mpl
import mlflow
import numpy as np
import pandas as pd
import scanpy as sc

os.chdir(str(ROOT))
os.makedirs("logs", exist_ok=True)

from arcadia.data_utils import load_adata_latest
from arcadia.data_utils.loading import determine_dataset_path
from arcadia.data_utils.preprocessing import convert_to_sparse_csr
from arcadia.plotting.training import (
    plot_batch_specific_umaps,
    plot_pre_training_visualizations,
    plot_umap_with_extremes,
)
from arcadia.training import (
    calculate_post_training_metrics,
    clear_memory,
    generate_post_training_visualizations,
    log_memory_usage,
    log_parameters,
    match_cells_and_calculate_distances,
    process_latent_spaces,
    train_vae,
    validate_scvi_training_mixin,
)
from arcadia.training.metrics import (
    extract_training_metrics_from_history,
    process_post_training_metrics,
)
from arcadia.utils.args import parse_pipeline_arguments
from arcadia.utils.environment import get_umap_filtered_fucntion, setup_environment
from arcadia.utils.logging import (
    log_training_summary,
    logger,
    setup_logger,
    setup_mlflow_experiment,
)

validate_scvi_training_mixin()

mpl.rcParams.update(
    {
        "savefig.format": "pdf",
        "figure.figsize": (6, 4),
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": False,
    }
)
sc.set_figure_params(
    scanpy=True, fontsize=10, dpi=100, dpi_save=300, vector_friendly=False, format="pdf"
)
if not hasattr(sc.tl.umap, "_is_wrapped"):
    sc.tl.umap = get_umap_filtered_fucntion()
    sc.tl.umap._is_wrapped = True
pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 10)
np.set_printoptions(threshold=100)

# %% Load configuration
config_path = str(ROOT / "configs" / "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config_ = json.load(f)
    num_rna_cells = config_["subsample"]["num_rna_cells"]
    num_protein_cells = config_["subsample"]["num_protein_cells"]
    plot_flag = config_["plot_flag"]
    max_epochs = config_.get("training", {}).get("max_epochs", 3)
else:
    num_rna_cells = num_protein_cells = 2000
    plot_flag = True
    max_epochs = 3

setup_environment()

# Initialize variables (will be set from args or papermill parameters)
dataset_name = None
model_checkpoints_folder = None

# %% Parse command line arguments
try:
    args = parse_pipeline_arguments()
    if args.dataset_name is not None:
        dataset_name = args.dataset_name
    if args.model_checkpoints_folder is not None:
        model_checkpoints_folder = args.model_checkpoints_folder
except SystemExit:
    # If running in notebook/papermill, variables are already set from parameters cell
    pass

save_dir = ROOT / "processed_data"

print(f"Selected dataset: {dataset_name if dataset_name else 'default (latest)'}")
save_dir, load_dataset_name = determine_dataset_path(save_dir, dataset_name)
load_path = str(save_dir.resolve())

adata_rna_subset, adata_prot_subset = load_adata_latest(
    load_path,
    ["rna", "protein"],
    exact_step=4,
    index_from_end=0,
    dataset_name=load_dataset_name,
)

log_memory_usage("Before sparse conversion: ")
adata_rna_subset, adata_prot_subset = convert_to_sparse_csr(
    adata_rna_subset, adata_prot_subset, logger=logger
)
log_memory_usage("After sparse conversion: ")

spatial_only = adata_prot_subset[:, adata_prot_subset.var["feature_type"] != "protein"]
sc.pp.pca(spatial_only)
sc.pp.neighbors(spatial_only)
sc.tl.umap(spatial_only)

if plot_flag:
    plot_umap_with_extremes(spatial_only, color_key=["CN", "cell_types"], title="spatial only")
sc.pp.subsample(adata_rna_subset, n_obs=min(num_rna_cells, len(adata_rna_subset)))
sc.pp.subsample(adata_prot_subset, n_obs=min(num_protein_cells, len(adata_prot_subset)))

plot_pre_training_visualizations(adata_rna_subset, adata_prot_subset, plot_flag=plot_flag)
plot_batch_specific_umaps(adata_rna_subset, plot_flag=plot_flag)

# Determine dataset name for experiment (prefer from args, then from adata, then default)
if dataset_name is None:
    dataset_name = adata_rna_subset.uns.get("dataset_name", None)
if dataset_name is None:
    dataset_name = "default"

experiment_name, run_name, mlflow_artifact_dir, log_file_path = setup_mlflow_experiment(
    dataset_name
)
logger = setup_logger(log_file=log_file_path, level="INFO")
logger.info(f"logger initialized. Using log file: {log_file_path}")
logger.info(
    f"Log file is being written directly to MLflow artifact directory: {mlflow_artifact_dir}"
)

logger.info(f"Original RNA dataset shape: {adata_rna_subset.shape}")
logger.info(f"Original protein dataset shape: {adata_prot_subset.shape}")

rna_sample_size = min(len(adata_rna_subset), num_rna_cells)
prot_sample_size = min(len(adata_prot_subset), num_protein_cells)
adata_rna_subset = sc.pp.subsample(adata_rna_subset, n_obs=rna_sample_size, copy=True)
adata_prot_subset = sc.pp.subsample(adata_prot_subset, n_obs=prot_sample_size, copy=True)

# Ensure CN and cell_type columns are categorical and have the same categories in both datasets
for col in ["CN", "cell_type"]:
    if col in adata_rna_subset.obs and col in adata_prot_subset.obs:
        cats = sorted(
            set(adata_rna_subset.obs[col].cat.categories)
            | set(adata_prot_subset.obs[col].cat.categories)
        )
        adata_rna_subset.obs[col] = adata_rna_subset.obs[col].cat.set_categories(cats)
        adata_prot_subset.obs[col] = adata_prot_subset.obs[col].cat.set_categories(cats)

adata_rna_subset.obs["cell_types"] = adata_rna_subset.obs["major_cell_types"]
adata_prot_subset.obs["cell_types"] = adata_prot_subset.obs["major_cell_types"]

logger.info(f"Subsampled RNA dataset shape: {adata_rna_subset.shape}")
logger.info(f"Subsampled protein dataset shape: {adata_prot_subset.shape}")
log_memory_usage("After subsampling: ")

training_params = {
    "max_epochs": max_epochs,
    "print_every_n_epoch": 50,
    "plot_x_times": 1,
    "check_val_every_n_epoch": 100,
    "save_checkpoint_every_n_epochs": 280,
    "plot_first_step": False,
    "batch_size": 1024,
    "outlier_detection_enabled": True,
    "contrastive_weight": 0,
    "similarity_weight": 0,
    "similarity_dynamic": False,
    "diversity_weight": 0.0,
    "matching_weight": 1,
    "cell_type_clustering_weight": 0.1,
    "cross_modal_cell_type_weight": 0.1,
    "cn_distribution_separation_weight": 0,
    "n_hidden_rna": 1024,
    "n_hidden_prot": 512,
    "n_layers": 3,
    "latent_dim": 60,
    "dropout_rate": 0.1,
    "rna_recon_weight": 1,
    "prot_recon_weight": 10,
    "adv_weight": 0.0,
    "train_size": 0.80,
    "validation_size": 0.20,
    "gradient_clip_val": 1.0,
    "lr": 1e-3,
    "load_optimizer_state": False,
    "override_skip_warmup": False,
    "gradnorm_enabled": False,
    "gradnorm_alpha": 1.5,
    "gradnorm_lr": 0.025,
}

# Allow config.json training section to override speed-related params
for override_key in ["plot_x_times", "save_checkpoint_every_n_epochs"]:
    config_val = config_.get("training", {}).get(override_key)
    if config_val is not None:
        training_params[override_key] = config_val

loss_weight_keys = [
    "rna_recon_weight",
    "prot_recon_weight",
    "contrastive_weight",
    "similarity_weight",
    "similarity_dynamic",
    "matching_weight",
    "cell_type_clustering_weight",
    "cross_modal_cell_type_weight",
    "cn_distribution_separation_weight",
]
loss_weights = {k: training_params.get(k, 0) for k in loss_weight_keys}

loss_weights_path = "loss_weights.json"
with open(loss_weights_path, "w") as f:
    json.dump(loss_weights, f, indent=4)

log_parameters(training_params, 0, 1)
mlflow.log_artifact(loss_weights_path)
os.remove(loss_weights_path)

mlflow.log_params(training_params)
mlflow.log_param("rna_dataset_shape", f"{adata_rna_subset.shape[0]}x{adata_rna_subset.shape[1]}")
mlflow.log_param(
    "protein_dataset_shape",
    f"{adata_prot_subset.shape[0]}x{adata_prot_subset.shape[1]}",
)
mlflow.log_param("dataset_name", adata_rna_subset.uns.get("dataset_name", "unknown"))

trained_from_scratch = model_checkpoints_folder is None
mlflow.log_param("trained_from_scratch", trained_from_scratch)

rna_vae, protein_vae = train_vae(
    adata_rna_subset,
    adata_prot_subset,
    model_checkpoints_folder=model_checkpoints_folder,
    logger=logger,
    **training_params,
)

clear_memory()

history_ = rna_vae._training_plan.get_history()

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

metrics_to_log = extract_training_metrics_from_history(history_, metric_mapping)
if metrics_to_log:
    mlflow.log_metrics(metrics_to_log)

adata_rna = rna_vae.adata
adata_prot = protein_vae.adata
adata_rna = sc.pp.subsample(adata_rna, n_obs=min(len(adata_rna), 4000), copy=True)
adata_prot = sc.pp.subsample(adata_prot, n_obs=min(len(adata_prot), 4000), copy=True)

rna_latent, prot_latent, combined_latent = process_latent_spaces(adata_rna, adata_prot)

matching_results = match_cells_and_calculate_distances(rna_latent, prot_latent)

metrics = calculate_post_training_metrics(
    adata_rna, adata_prot, matching_results["prot_matches_in_rna"]
)

valid_metrics, nan_metrics = process_post_training_metrics(metrics)

if valid_metrics:
    mlflow.log_metrics(valid_metrics)
    logger.info(f"Logged {len(valid_metrics)} post-training metrics")

if nan_metrics:
    logger.warning(f"Skipped {len(nan_metrics)} NaN/invalid post-training metrics: {nan_metrics}")

generate_post_training_visualizations(
    adata_rna,
    adata_prot,
    rna_latent,
    prot_latent,
    combined_latent,
    history_,
    matching_results,
    plot_flag=training_params.get("plot_x_times", 0) > 0,
)
print("Visualizations generated")

log_training_summary(
    experiment_name=experiment_name,
    run_name=run_name,
    mlflow_artifact_dir=mlflow_artifact_dir,
    log_file_path=log_file_path,
    dataset_name=dataset_name,
    save_dir=save_dir,
    rna_sample_size=rna_sample_size,
    prot_sample_size=prot_sample_size,
    max_epochs=training_params.get("max_epochs", "N/A"),
    model_checkpoints_folder=model_checkpoints_folder,
    project_root=str(ROOT),
)
