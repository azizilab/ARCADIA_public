"""Training function for dual VAE models."""

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from pprint import pprint

import mlflow
import scvi
import torch
from scipy.sparse import issparse

from arcadia.training.dual_vae_training_plan import DualVAETrainingPlan
from arcadia.training.loss_scaling import load_loss_scales_from_cache, save_loss_scales_to_cache
from arcadia.training.utils import log_memory_usage
from arcadia.utils import metadata as pipeline_metadata_utils
from arcadia.utils.logging import setup_logger


def train_vae(
    adata_rna_subset,
    adata_prot_subset,
    model_checkpoints_folder=None,
    max_epochs=10,
    batch_size=128,
    lr=1e-3,
    contrastive_weight=1.0,
    similarity_weight=0.5,
    similarity_dynamic=True,
    diversity_weight=0.1,
    matching_weight=1.0,
    cell_type_clustering_weight=1.0,
    cross_modal_cell_type_weight=1.0,
    cn_distribution_separation_weight=1.0,
    train_size=0.9,
    check_val_every_n_epoch=1,
    adv_weight=0.1,
    n_hidden_rna=128,
    n_hidden_prot=50,
    n_layers=3,
    latent_dim=10,
    validation_size=0.1,
    gradient_clip_val=1.0,
    accumulate_grad_batches=1,
    rna_recon_weight=1.0,
    prot_recon_weight=1.0,
    save_checkpoint_every_n_epochs=30,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    plot_x_times=5,
    print_every_n_epoch=5,
    plot_first_step=False,
    log_file_path=None,
    logger=None,
    load_optimizer_state=True,
    outlier_detection_enabled=True,
    override_skip_warmup=False,  # Boolean flag to force skip warmup
    dropout_rate=0.1,  # Dropout rate for both VAEs
    gradnorm_enabled=True,  # Enable GradNorm for adaptive loss balancing
    gradnorm_alpha=1.5,  # GradNorm alpha parameter
    gradnorm_lr=0.025,  # GradNorm learning rate
):
    """Train the VAE models."""

    # Initialize logger if not provided
    if logger is None:
        # Get MLflow artifact directory (there will always be an active run)
        current_run = mlflow.active_run()
        mlflow_artifact_dir = Path(current_run.info.artifact_uri.replace("file://", ""))
        mlflow_artifact_dir.mkdir(parents=True, exist_ok=True)

        if log_file_path is None:
            # Use MLflow artifact directory for logs
            default_log_filename = f"train_vae_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            current_log_file_path = mlflow_artifact_dir / default_log_filename
        else:
            current_log_file_path = Path(log_file_path)

        logger = setup_logger(log_file=current_log_file_path, level="INFO")
        logger.info(f"train_vae logger initialized. Using log file: {current_log_file_path}")
        logger.info(f"MLflow artifact directory: {mlflow_artifact_dir}")

    # Memory optimization: Clean datasets before training
    log_memory_usage("before cleanup")
    log_memory_usage("after cleanup")
    logger.info("✅ AnnData objects cleaned for memory optimization")

    # Initialize training metadata
    training_parameters = {
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "contrastive_weight": contrastive_weight,
        "similarity_weight": similarity_weight,
        "similarity_dynamic": similarity_dynamic,
        "diversity_weight": diversity_weight,
        "matching_weight": matching_weight,
        "cell_type_clustering_weight": cell_type_clustering_weight,
        "cross_modal_cell_type_weight": cross_modal_cell_type_weight,
        "cn_distribution_separation_weight": cn_distribution_separation_weight,
        "train_size": train_size,
        "validation_size": validation_size,
        "n_hidden_rna": n_hidden_rna,
        "n_hidden_prot": n_hidden_prot,
        "n_layers": n_layers,
        "latent_dim": latent_dim,
        "rna_recon_weight": rna_recon_weight,
        "prot_recon_weight": prot_recon_weight,
        "dropout_rate": dropout_rate,
        "device": str(device),
        "outlier_detection_enabled": outlier_detection_enabled,
    }

    # Initialize training metadata using utility function
    pipeline_metadata_utils.initialize_train_vae_metadata(
        adata_rna_subset, adata_prot_subset, training_parameters
    )

    # Verify that batch columns exist (should be created in data preparation step)
    if "batch" not in adata_rna_subset.obs.columns:
        raise ValueError("RNA data missing 'batch' column. Run data preparation step first.")
    if "batch" not in adata_prot_subset.obs.columns:
        raise ValueError("Protein data missing 'batch' column. Run data preparation step first.")

    logger.info(
        f"RNA batch column: {adata_rna_subset.obs['batch'].dtype}, unique values: {list(adata_rna_subset.obs['batch'].unique())}"
    )
    logger.info(
        f"Protein batch column: {adata_prot_subset.obs['batch'].dtype}, unique values: {list(adata_prot_subset.obs['batch'].unique())}"
    )

    # setup adata for scvi
    scvi.model.SCVI.setup_anndata(
        adata_rna_subset,
        labels_key="index_col",
        batch_key="batch",
    )
    scvi.model.SCVI.setup_anndata(
        adata_prot_subset,
        labels_key="index_col",
        batch_key="batch",
    )

    # Use the statistically determined gene likelihoods from your select_gene_likelihood function
    gene_likelihood_rna = adata_rna_subset.uns["gene_likelihood"]
    gene_likelihood_prot = adata_prot_subset.uns["gene_likelihood"]

    logger.info(f"Using statistically determined gene likelihoods:")
    logger.info(f"  RNA: {gene_likelihood_rna}")
    logger.info(f"  Protein: {gene_likelihood_prot}")

    mlflow.log_param("gene_likelihood_rna", gene_likelihood_rna)
    mlflow.log_param("gene_likelihood_prot", gene_likelihood_prot)
    mlflow.log_param("dropout_rate", dropout_rate)
    use_batch_norm = "both"

    rna_vae = scvi.model.SCVI(
        adata_rna_subset,
        gene_likelihood=gene_likelihood_rna,
        n_hidden=n_hidden_rna,
        n_layers=n_layers,
        n_latent=latent_dim,
        dropout_rate=dropout_rate,  # Add dropout for regularization
        use_observed_lib_size=True,
        use_batch_norm=use_batch_norm,  # crucial for stability
        # use_layer_norm="both",
    )
    protein_vae = scvi.model.SCVI(
        adata_prot_subset,
        gene_likelihood=gene_likelihood_prot,
        n_hidden=n_hidden_prot,
        n_layers=n_layers,
        n_latent=latent_dim,
        dropout_rate=dropout_rate,  # Add dropout for regularization
        use_observed_lib_size=True,
        use_batch_norm=use_batch_norm,  # crucial for stability
        # use_layer_norm="both",
    )
    # Log model parameters as MLflow parameters for easy comparison
    rna_trainable_params = sum(p.numel() for p in rna_vae.module.parameters() if p.requires_grad)
    protein_trainable_params = sum(
        p.numel() for p in protein_vae.module.parameters() if p.requires_grad
    )
    total_trainable_params = rna_trainable_params + protein_trainable_params

    mlflow.log_param("rna_trainable_params", rna_trainable_params)
    mlflow.log_param("protein_trainable_params", protein_trainable_params)
    mlflow.log_param("total_trainable_params", total_trainable_params)

    logger.info(f"Model parameters:")
    logger.info(f"  RNA VAE trainable parameters: {rna_trainable_params:,}")
    logger.info(f"  Protein VAE trainable parameters: {protein_trainable_params:,}")
    logger.info(f"  Total trainable parameters: {total_trainable_params:,}")

    # Create model configuration for easy checkpoint loading
    model_config = {
        "rna_model": {
            "n_latent": latent_dim,
            "n_hidden": n_hidden_rna,
            "n_layers": n_layers,
            "dropout_rate": dropout_rate,
            "gene_likelihood": gene_likelihood_rna,
            "use_observed_lib_size": True,
            "use_batch_norm": use_batch_norm,
        },
        "protein_model": {
            "n_latent": latent_dim,
            "n_hidden": n_hidden_prot,
            "n_layers": n_layers,
            "dropout_rate": dropout_rate,
            "gene_likelihood": gene_likelihood_prot,
            "use_observed_lib_size": True,
            "use_batch_norm": use_batch_norm,
        },
        "anndata_setup": {
            "rna_setup": {
                "labels_key": "index_col",
                "batch_key": "batch",
                "layer": None,
                "categorical_covariate_keys": [],
                "continuous_covariate_keys": [],
            },
            "protein_setup": {
                "labels_key": "index_col",
                "batch_key": "batch",
                "layer": None,
                "categorical_covariate_keys": [],
                "continuous_covariate_keys": [],
            },
        },
        "rna_num_params": rna_trainable_params,
        "protein_num_params": protein_trainable_params,
        "scvi_version": scvi.__version__,
        "creation_timestamp": datetime.now().isoformat(),
        "notes": "Configuration for recreating SCVI models from checkpoints",
    }

    # Save and log model configuration as JSON
    model_config_path = "model_config.json"
    with open(model_config_path, "w") as f:
        json.dump(model_config, f, indent=4)
    mlflow.log_artifact(model_config_path)
    os.remove(model_config_path)

    logger.info("Model configuration JSON logged to MLflow")

    # Define parameters for hashing, which determines if cached scales can be used
    training_params_for_hash = {
        "rna_shape": list(adata_rna_subset.shape),
        "protein_shape": list(adata_prot_subset.shape),
        "latent_dim": latent_dim,
        "rna_n_hidden": n_hidden_rna,
        "outlier_detection_enabled": outlier_detection_enabled,
        "rna_n_layers": n_layers,
        "prot_n_hidden": n_hidden_prot,
        "prot_n_layers": n_layers,
        "dropout_rate": dropout_rate,
        "lr": lr,
        "rna_data_shape": list(adata_rna_subset.shape),
        "protein_data_shape": list(adata_prot_subset.shape),
        "current_commit": os.popen("git rev-parse HEAD").read().strip(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "hash_of_data_rna": hashlib.sha256(
            adata_rna_subset.X.toarray().tobytes()
            if issparse(adata_rna_subset.X)
            else adata_rna_subset.X.tobytes()
        ).hexdigest(),
        "hash_of_data_prot": hashlib.sha256(
            adata_prot_subset.X.toarray().tobytes()
            if issparse(adata_prot_subset.X)
            else adata_prot_subset.X.tobytes()
        ).hexdigest(),
        "use_batch_norm": use_batch_norm,
    }

    # Load loss scales from cache if available
    scales_cache_path = Path("scales_cache.json")
    cached_loss_scales = load_loss_scales_from_cache(training_params_for_hash, scales_cache_path)

    # Define the callback for saving scales
    def save_scales_callback(params_for_hash, scales):
        save_loss_scales_to_cache(params_for_hash, scales, scales_cache_path)

    rna_vae._training_plan_cls = DualVAETrainingPlan

    # Set up training parameters
    plan_kwargs = {
        "protein_vae": protein_vae,
        "rna_vae": rna_vae,
        "contrastive_weight": contrastive_weight,
        "similarity_weight": similarity_weight,
        "similarity_dynamic": similarity_dynamic,
        "diversity_weight": diversity_weight,
        "cell_type_clustering_weight": cell_type_clustering_weight,
        "cross_modal_cell_type_weight": cross_modal_cell_type_weight,
        "cn_distribution_separation_weight": cn_distribution_separation_weight,
        "matching_weight": matching_weight,
        "adv_weight": adv_weight,
        "plot_x_times": plot_x_times,
        "plot_first_step": plot_first_step,
        "save_checkpoint_every_n_epochs": save_checkpoint_every_n_epochs,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "lr": lr,
        "rna_recon_weight": rna_recon_weight,
        "prot_recon_weight": prot_recon_weight,
        "check_val_every_n_epoch": check_val_every_n_epoch,
        "print_every_n_epoch": print_every_n_epoch,
        "log_file_path": log_file_path,
        "loss_scales": cached_loss_scales,  # Pass cached scales
        "training_params_for_hash": training_params_for_hash,
        "save_scales_callback": save_scales_callback,
        "gradient_clip_val": gradient_clip_val,
        "outlier_detection_enabled": outlier_detection_enabled,
        "max_kl_weight": 0.9,
        "min_kl_weight": 0.1,
        "n_epochs_kl_warmup": 80,
        "gradnorm_enabled": gradnorm_enabled,
        "gradnorm_alpha": gradnorm_alpha,
        "gradnorm_lr": gradnorm_lr,
    }
    train_kwargs = {
        "gradient_clip_val": gradient_clip_val,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "train_size": train_size,
        "validation_size": validation_size,
        "accumulate_grad_batches": accumulate_grad_batches,
        "check_val_every_n_epoch": check_val_every_n_epoch,
    }
    logger.info("Plan parameters:")
    pprint(plan_kwargs)
    # Create training plan instance
    logger.info("Creating training plan for initial latent computation...")
    training_plan = DualVAETrainingPlan(rna_vae.module, **plan_kwargs)
    rna_vae._training_plan = training_plan
    logger.info("Training plan created")

    # Train the model
    logger.info("Starting training...")
    # Save training parameters and configs

    rna_vae.is_trained_ = True
    protein_vae.is_trained_ = True
    rna_vae.module.cpu()
    protein_vae.module.cpu()

    if model_checkpoints_folder is not None:
        rna_vae, protein_vae, training_state = DualVAETrainingPlan.load_from_checkpoints(
            model_checkpoints_folder, rna_vae, protein_vae
        )
        plan_kwargs["training_state"] = training_state

        # Check if checkpoint has loss_scales and prioritize them
        checkpoint_loss_scales = training_state.get("loss_scales")
        if checkpoint_loss_scales is not None:
            logger.info("Using loss scales from checkpoint, skipping warmup")
            plan_kwargs["loss_scales"] = checkpoint_loss_scales
        elif override_skip_warmup:
            # Force skip warmup if explicitly requested
            if cached_loss_scales is None:
                logger.info("Skipping warmup (override requested) - using default scales")
                plan_kwargs["loss_scales"] = {
                    "rna_reconstruction": 1.0,
                    "protein_reconstruction": 1.0,
                    "matching": 1.0,
                    "contrastive": 1.0,
                    "similarity": 1.0,  # Include similarity loss in scale warmup
                    "cell_type_clustering": 1.0,
                    "cross_modal_cell_type": 1.0,
                    "cn_distribution_separation": 1.0,
                }
            else:
                logger.info("Skipping warmup (override requested) - using cached scales")
                plan_kwargs["loss_scales"] = cached_loss_scales
        elif cached_loss_scales is not None:
            logger.info("Using cached scales (no checkpoint scales), skipping warmup")
            plan_kwargs["loss_scales"] = cached_loss_scales
        else:
            logger.info("No scales found (checkpoint or cache), will run scale warmup")
    else:
        # Handle warmup override logic for training from scratch
        if override_skip_warmup:
            # Force skip warmup if explicitly requested
            if cached_loss_scales is None:
                logger.info("Skipping warmup (override requested) - using default scales")
                plan_kwargs["loss_scales"] = {
                    "rna_reconstruction": 1.0,
                    "protein_reconstruction": 1.0,
                    "matching": 1.0,
                    "contrastive": 1.0,
                    "similarity": 1.0,  # Include similarity loss in scale warmup
                    "cell_type_clustering": 1.0,
                    "cross_modal_cell_type": 1.0,
                    "cn_distribution_separation": 1.0,
                }
            else:
                logger.info("Skipping warmup (override requested) - using cached scales")
                plan_kwargs["loss_scales"] = cached_loss_scales
        elif cached_loss_scales is not None:
            # Use cached scales if available (same logic as training from scratch)
            logger.info("Using cached scales, skipping warmup")
            plan_kwargs["loss_scales"] = cached_loss_scales
        else:
            # No cached scales - run warmup (same logic as training from scratch)
            logger.info("No cached scales found, will run scale warmup")
            # Don't set loss_scales in plan_kwargs, let warmup run

        rna_vae.module.to(device)
        protein_vae.module.to(device)
        rna_vae.is_trained_ = False
        protein_vae.is_trained_ = False

    training_start_time = time.time()

    mlflow.log_params(plan_kwargs)
    rna_vae.train(**train_kwargs, plan_kwargs=plan_kwargs)
    training_end_time = time.time()
    training_time = training_end_time - training_start_time

    # Ensure on_train_end_custom is called (PyTorch Lightning may not trigger it reliably)
    if hasattr(rna_vae, "_training_plan") and rna_vae._training_plan is not None:
        training_plan = rna_vae._training_plan
        if not training_plan.on_train_end_custom_called:
            logger.info("Calling on_train_end_custom explicitly (not triggered during training)")
            training_plan.on_train_end_custom(plot_flag=True)

    # Update training results metadata
    training_results = {
        "training_time": training_time,
        "best_epoch": None,  # Could be populated from training plan if available
        "final_losses": None,  # Could be populated from training plan if available
        "validation_metrics": None,  # Could be populated from training plan if available
    }

    pipeline_metadata_utils.update_train_vae_metadata(
        adata_rna_subset, adata_prot_subset, training_results
    )

    # Manually set trained flag
    rna_vae.is_trained_ = True
    protein_vae.is_trained_ = True
    logger.info("Training flags set")

    return rna_vae, protein_vae
