# %%
"""Train VAE with archetypes vectors."""

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
# %%

""" DO NOT REMOVE THIS COMMENT!!!
TO use this script, you need to add the training plan to use the DualVAETrainingPlan (version 1.2.2.post2) class in scVI library.
in _training_mixin.py, line 131, you need to change the line:
training_plan = self._training_plan_cls(self.module, **plan_kwargs) # existing line
self._training_plan = training_plan # add this line

"""
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import anndata
import mlflow
import numpy as np
import pandas as pd
import psutil
import scanpy as sc
import scvi
import torch
from anndata import AnnData
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from scvi.train import TrainingPlan
from tabulate import tabulate

from arcadia.data_utils.cleaning import clean_uns_for_h5ad

# Import plotting functions
from arcadia.plotting import training as pf
from arcadia.plotting.training import (
    plot_first_batch_umaps,
    plot_similarity_loss_history,
    plot_warmup_loss_distributions,
)
from arcadia.training.gradnorm import GradNorm
from arcadia.training.losses import (
    calculate_cross_modal_cell_type_loss,
    calculate_modality_balance_loss,
    cn_distribution_separation_loss,
    extreme_archetypes_loss,
    run_cell_type_clustering_loss,
)
from arcadia.training.metrics import (
    calculate_iLISI,
    compute_ari_f1,
    compute_silhouette_f1,
    kbet_within_cell_types,
    matching_accuracy,
)
from arcadia.training.utils import (
    compute_pairwise_distances,
    compute_pairwise_kl_two_items,
    create_counterfactual_adata,
    get_latent_embedding,
    predict_rna_cn_from_protein_neighbors,
)
from arcadia.utils.environment import get_umap_filtered_fucntion
from arcadia.utils.logging import log_epoch_end, log_step, setup_logger

# Force reimport internal modules
# # import CODEX_RNA_seq.logging_functions  # Migrated to arcadia.utils.logging  # Migrated to arcadia.utils


if not hasattr(sc.tl.umap, "_is_wrapped"):
    sc.tl.umap = get_umap_filtered_fucntion()
    sc.tl.umap._is_wrapped = True
pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 10)
warnings.filterwarnings("ignore")
pd.options.display.max_rows = 10
pd.options.display.max_columns = 10
np.set_printoptions(threshold=100)

config_path = Path(__file__).parent.parent.parent.parent / "configs" / "config.json"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config_ = json.load(f)
    num_rna_cells = config_["subsample"]["num_rna_cells"]
    num_protein_cells = config_["subsample"]["num_protein_cells"]
    plot_flag = config_["plot_flag"]
else:
    num_rna_cells = num_protein_cells = 2000
    plot_flag = True


# %%
# Define the DualVAETrainingPlan class
class DualVAETrainingPlan(TrainingPlan):

    @staticmethod
    def normalise_weights(weight_dict: dict[str, float]):
        """
        Input dict with keys as weight label strings and values as weight value floats.
        Scale all positive weights so their sum is 1.0.
        Raises error if no nonezero weights.
        """
        total = sum(v for v in weight_dict.values() if v > 0)
        if total == 0:
            raise ValueError("At least one loss weight must be > 0 before normalisation.")
        # return {k: (v / total if v > 0 else 0.0) for k, v in weight_dict.items()}
        return {k: (1000 * v / total if v > 0 else 0.0) for k, v in weight_dict.items()}

    def __init__(self, rna_module, **kwargs):
        # Pull the two pretrained scVI objects out of kwargs
        rna_vae = kwargs.pop("rna_vae")
        protein_vae = kwargs.pop("protein_vae")
        rna_vae.use_pretrained_checkpoints = False

        self.log_file_path_param = kwargs.pop("log_file_path", None)
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup log file path
        if self.log_file_path_param:
            self.log_file_path = Path(self.log_file_path_param)
        else:
            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)  # Ensure logs directory exists
            log_file_name = f"vae_training_{self.run_timestamp}.log"
            self.log_file_path = logs_dir / log_file_name
        self.logger_ = setup_logger(log_file=self.log_file_path, level="INFO")

        # Initialize logger directly using the determined log_file_path

        self.logger_.info(
            f"DualVAETrainingPlan logger initialized. Using log file: {self.log_file_path}"
        )
        # Caching-related parameters from the training script
        self.training_params_for_hash = kwargs.pop("training_params_for_hash", None)
        self.save_scales_callback = kwargs.pop("save_scales_callback", None)

        # Normalize weights
        # ----------------------------------------------------
        self.contrastive_weight = kwargs.pop("contrastive_weight", 1.0)
        self.similarity_weight = kwargs.pop("similarity_weight", 1.0)
        self.disable_similarity_weight_updates = kwargs.pop(
            "disable_similarity_weight_updates", False
        )
        self.similarity_dynamic = kwargs.pop("similarity_dynamic", True)

        # Automatically disable updates if similarity weight is set to 0 or similarity_dynamic is False
        if self.similarity_weight == 0 or not self.similarity_dynamic:
            self.disable_similarity_weight_updates = True
        self.cell_type_clustering_weight = kwargs.pop("cell_type_clustering_weight", 1000.0)
        self.cross_modal_cell_type_weight = kwargs.pop("cross_modal_cell_type_weight", 1000.0)
        self.cn_distribution_separation_weight = kwargs.pop(
            "cn_distribution_separation_weight", 1.0
        )
        self.cross_modal_mmd_sigma = kwargs.pop(
            "cross_modal_mmd_sigma", None
        )  # RBF kernel bandwidth (None = adaptive via median heuristic)
        self.matching_weight = kwargs.pop("matching_weight", 1000.0)
        self.rna_recon_weight = kwargs.pop("rna_recon_weight", 1.0)
        self.prot_recon_weight = kwargs.pop("prot_recon_weight", 1.0)

        raw_w = {
            "similarity": self.similarity_weight,  # FIXED: Include similarity in normalization
            "contrastive": self.contrastive_weight,
            "matching": self.matching_weight,
            "cell_type_clustering": self.cell_type_clustering_weight,
            "cross_modal_cell_type": self.cross_modal_cell_type_weight,
            "cn_distribution_sep": self.cn_distribution_separation_weight,
            "rna_recon_weight": self.rna_recon_weight,
            "prot_recon_weight": self.prot_recon_weight,
        }
        norm_w = self.normalise_weights(raw_w)

        self.similarity_weight = norm_w["similarity"]  # FIXED: Apply normalized similarity weight
        self.contrastive_weight = norm_w["contrastive"]
        self.matching_weight = norm_w["matching"]
        self.cell_type_clustering_weight = norm_w["cell_type_clustering"]
        self.cross_modal_cell_type_weight = norm_w["cross_modal_cell_type"]
        self.cn_distribution_separation_weight = norm_w["cn_distribution_sep"]
        self.rna_recon_weight = norm_w["rna_recon_weight"]
        self.prot_recon_weight = norm_w["prot_recon_weight"]

        # Log to logger + mlflow
        self.logger_.info(f"[Init] Loss weights normalised to sum=1:\n{norm_w}")
        try:
            if mlflow.active_run() is not None:
                for k, v in norm_w.items():
                    mlflow.log_param(f"loss_weight_{k}", v)
        except Exception as e:
            self.logger_.warning(f"MLflow logging failed: {e}")
        # ----------------------------------------------------
        # Store initial weights for tracking changes
        self.previous_weights_rna = {
            name: param.clone().detach() for name, param in rna_module.named_parameters()
        }
        self.previous_weights_protein = {
            name: param.clone().detach() for name, param in protein_vae.module.named_parameters()
        }
        # Hyperparams, pop() removes these s.t. TrainingPlan has only Lightning-valid keys
        self.save_checkpoint_every_n_epochs = kwargs.pop("save_checkpoint_every_n_epochs", 30)
        self.plot_x_times = kwargs.pop("plot_x_times", 5)
        self.plot_first_step = kwargs.pop("plot_first_step", False)
        # contrastive_weight = kwargs.pop("contrastive_weight", 1.0)
        requested_batch_size = kwargs.pop("batch_size", 1000)  # Store original requested batch size
        self.max_epochs = kwargs.pop("max_epochs", 10)
        # self.similarity_weight = kwargs.pop("similarity_weight")
        # self.cell_type_clustering_weight = kwargs.pop("cell_type_clustering_weight", 1000.0)
        # self.cross_modal_cell_type_weight = kwargs.pop("cross_modal_cell_type_weight", 1000.0)
        # self.variance_similarity_weight = kwargs.pop("variance_similarity_weight", 0.5)
        # self.cn_distribution_separation_weight = kwargs.pop(
        #     "cn_distribution_separation_weight", 1.0
        # )
        self.lr_warmup_epochs = 0  # DISABLE LR WARMUP - was kwargs.pop("lr_warmup_epochs", 5)
        self.training_state = kwargs.pop("training_state", None)
        self.load_optimizer_state = kwargs.pop(
            "load_optimizer_state", False
        )  # New parameter to control optimizer loading
        # self.rna_recon_weight = kwargs.pop("rna_recon_weight", 1.0)
        # self.prot_recon_weight = kwargs.pop("prot_recon_weight", 1.0)
        # self.matching_weight = kwargs.pop("matching_weight", 1000.0)
        train_size = kwargs.pop("train_size", 0.9)
        self.check_val_every_n_epoch = kwargs["check_val_every_n_epoch"]
        self.print_every_n_epoch = kwargs.pop("print_every_n_epoch", 5)
        # Calculate validation plot intervals similar to training plots
        # Total validation epochs = max_epochs // check_val_every_n_epoch
        self.total_val_epochs = max(1, self.max_epochs // self.check_val_every_n_epoch)
        # Calculate validation plot interval to distribute plot_x_times plots across validation epochs
        if self.plot_x_times > 0:
            self.val_plot_interval = max(1, self.total_val_epochs // self.plot_x_times)
        else:
            self.val_plot_interval = -1  # Disable validation plots
        self.val_epoch_counter = 0  # Counter to track validation epochs
        # Initial counters and flags
        self.to_print = False
        self.validation_step_ = 0
        self.train_step_ = 0  # Initialize train_step_ counter
        self.scales_logged_to_mlflow = False
        validation_size = kwargs.pop("validation_size", 0.1)
        device = kwargs.pop("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        # Verify train and validation sizes sum to 1
        self.metrics_history = []
        self.gradient_clip_val = kwargs.pop(
            "gradient_clip_val", 0.8
        )  # Much more aggressive clipping (was 0.8)

        if abs(train_size + validation_size - 1.0) > 1e-6:
            raise ValueError("train_size + validation_size must sum to 1.0")

        # Attach dual VAE
        super().__init__(rna_module, **kwargs)
        super().__init__(protein_vae.module, **kwargs)
        self.lr = kwargs.pop("lr", 0.001)

        self.rna_vae = rna_vae
        self.protein_vae = protein_vae

        # Create train/validation splits
        n_rna = len(self.rna_vae.adata)
        n_prot = len(self.protein_vae.adata)

        # Create indices for RNA data
        rna_indices = np.arange(n_rna)
        np.random.shuffle(rna_indices)
        n_train_rna = int(n_rna * train_size)
        self.train_indices_rna = rna_indices[:n_train_rna]
        self.val_indices_rna = rna_indices[n_train_rna:]

        # Create indices for protein data
        prot_indices = np.arange(n_prot)
        np.random.shuffle(prot_indices)
        n_train_prot = int(n_prot * train_size)
        self.train_indices_prot = prot_indices[:n_train_prot]
        self.val_indices_prot = prot_indices[n_train_prot:]

        # Adjust batch size to be at most the size of the training set
        min_train_size = min(len(self.train_indices_rna), len(self.train_indices_prot))
        self.batch_size = min(requested_batch_size, min_train_size)
        self.logger_.info(
            f"Batch size adjusted: requested={requested_batch_size}, actual={self.batch_size} (limited by training data size)"
        )

        # Add validation batch size that's capped at validation dataset size
        min_val_size = min(len(self.val_indices_rna), len(self.val_indices_prot))
        self.val_batch_size = min(requested_batch_size, min_val_size)
        self.logger_.info(
            f"Validation batch size: {self.val_batch_size} (limited by validation data size of {min_val_size})"
        )

        num_batches = 2
        latent_dim = self.rna_vae.module.n_latent
        self.batch_classifier = torch.nn.Linear(latent_dim, num_batches)
        self.protein_vae.module.to(device)
        self.rna_vae.module = self.rna_vae.module.to(device)
        self.first_step = True

        n_samples = len(self.train_indices_rna)  # Use training set size
        steps_per_epoch = int(np.ceil(n_samples / self.batch_size))
        self.steps_per_epoch = steps_per_epoch  # Store steps_per_epoch for later use
        self.total_steps = steps_per_epoch * (self.max_epochs)
        self.similarity_loss_history = []
        self.steady_state_window = 5
        self.steady_state_tolerance = 0.5
        # Set similarity_active based on current similarity_weight
        self.similarity_active = self.similarity_weight != 0
        self.reactivation_threshold = 0.1

        # Add parameters for improved similarity loss activation mechanism
        self.similarity_loss_steady_counter = 0  # Counter for steps in steady state
        self.similarity_loss_steady_threshold = (
            10  # Deactivate after this many steps in steady state
        )
        self.similarity_weight = self.similarity_weight  # Store original weight

        self.active_similarity_loss_active_history = []
        self.train_losses = []
        self.val_total_loss = []
        self.mode = "training"
        self.similarity_losses = []  # Store similarity losses
        self.raw_similarity_losses = []  # Store raw similarity losses
        self.similarity_weight_history = []  # Store similarity weights
        self.train_rna_losses = []
        self.train_protein_losses = []
        self.train_matching_losses = []
        self.train_contrastive_losses = []
        self.train_adv_losses = []
        self.train_cell_type_clustering_losses = []  # New list for cell type clustering losses
        self.train_cross_modal_cell_type_losses = []  # New list for cross-modal cell type alignment
        self.train_cn_separation_losses = []  # New list for CN distribution separation losses
        self.val_rna_loss = []
        self.val_protein_loss = []
        self.val_matching_loss = []
        self.val_contrastive_loss = []
        self.val_adv_loss = []
        self.val_cell_type_clustering_loss = []  # New list for cell type clustering losses
        self.val_cross_modal_cell_type_loss = []  # New list for cross-modal cell type alignment
        self.val_cn_distribution_separation_loss = (
            []
        )  # New list for CN distribution separation losses
        # Add new validation lists
        self.val_similarity_loss = []
        self.val_raw_similarity_loss = []
        self.val_latent_distance = []
        self.early_stopping_callback = None  # Will be set by trainer

        # iLISI tracking
        self.last_ilisi_score = float(
            "nan"
        )  # Initialize to invalid, will be set on first calculation
        self.ilisi_check_frequency = max(
            1, int(self.total_steps / 500)
        )  # Check ~500 times during training (increased from 100)
        # Setup logging

        # Track if on_train_end_custom has been called
        self.on_train_end_custom_called = False

        # Create run directory for checkpoints saves (checkpoints are saved to MLflow artifacts)
        # This path is kept for backward compatibility but checkpoints are primarily in MLflow
        self.checkpoints_dir = Path(f"checkpoints/run_{self.run_timestamp}/")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.on_epoch_end_called = False

        # Add outlier detection parameters
        # Outlier detection helps handle extreme loss values that can destabilize training
        # by replacing them with the previous step's value when they exceed the z-score threshold
        self.outlier_detection_enabled = kwargs.pop("outlier_detection_enabled", True)
        self.outlier_z_threshold = kwargs.pop(
            "outlier_z_threshold", 8.0
        )  # Standard deviations for outlier detection
        self.outlier_window_size = kwargs.pop(
            "outlier_window_size", 50
        )  # Running window size for statistics

        # Initialize outlier detection tracking
        self.rna_loss_history = []
        self.protein_loss_history = []
        self.outlier_count_rna = 0
        self.outlier_count_protein = 0

        self.logger_.info(f"Outlier detection enabled: {self.outlier_detection_enabled}")
        if self.outlier_detection_enabled:
            self.logger_.info(
                f"Outlier detection parameters: z_threshold={self.outlier_z_threshold}, window_size={self.outlier_window_size}"
            )

        # Add warmup parameters
        provided_loss_scales = kwargs.pop("loss_scales", None)
        self.loss_scales = {
            "rna_reconstruction": 1.0,
            "protein_reconstruction": 1.0,
            "matching": 1.0,
            "contrastive": 1.0,
            "similarity": 1.0,  # Include similarity loss in scale warmup
            "cell_type_clustering": 1.0,
            "cross_modal_cell_type": 1.0,
            "cn_distribution_separation": 1.0,
        }
        # Check if using pretrained checkpoints - force warmup if so
        force_warmup = getattr(self.rna_vae, "use_pretrained_checkpoints", False)

        if force_warmup:
            self.logger_.info(
                "Using pretrained checkpoints - forcing scale warmup regardless of provided scales."
            )
            if provided_loss_scales is not None:
                self.logger_.info(
                    "Ignoring provided/checkpoint loss scales due to pretrained checkpoint usage."
                )
            self.warmup_epochs = (
                self.max_epochs // 2 if self.max_epochs < 12 else 10
            )  # Do warmup to calculate appropriate scales
            self.scales_from_cache = False
        elif provided_loss_scales is not None:
            self.logger_.info("Pre-defined loss scales provided. Skipping warmup.")
            self.loss_scales.update(provided_loss_scales)
            self.warmup_epochs = 0
            self.scales_from_cache = True  # Flag to track cached scales
            if mlflow.active_run() and not self.scales_logged_to_mlflow:
                self.logger_.info("Logging provided loss scales to MLflow.")
                mlflow.log_params({f"scale_{k}": v for k, v in self.loss_scales.items()})
                self.scales_logged_to_mlflow = True
        else:
            self.logger_.info(
                "No pre-defined loss scales. Will perform warmup to calculate scales."
            )
            self.warmup_epochs = (
                self.max_epochs // 2 if self.max_epochs < 12 else 10
            )  # Do warmup to calculate appropriate scales
            self.scales_from_cache = False

        self.warmup_raw_losses = {key: [] for key in self.loss_scales}

        # Freeze model parameters during warmup to prevent weight updates
        if self.warmup_epochs > 0:
            self._freeze_models()
            self.logger_.info(
                f"Model parameters frozen during warmup phase ({self.warmup_epochs} epochs). "
                "Weights will not be updated until warmup completes."
            )

        # setup library size
        # Handle sparse matrices properly for library size calculation
        if issparse(self.protein_vae.adata.X):
            library_size_protein = np.array(self.protein_vae.adata.X.sum(axis=1)).flatten()
        else:
            library_size_protein = self.protein_vae.adata.X.sum(axis=1)
            library_size_protein = np.asarray(library_size_protein).flatten()

        if issparse(self.rna_vae.adata.X):
            library_size_rna = np.array(self.rna_vae.adata.X.sum(axis=1)).flatten()
        else:
            library_size_rna = self.rna_vae.adata.X.sum(axis=1)
            library_size_rna = np.asarray(library_size_rna).flatten()
        # 2. Calculate the log library size
        # scvi-tools uses the natural logarithm.
        log_library_size_protein = np.log(library_size_protein)
        log_library_size_rna = np.log(library_size_rna)
        # 3. Add the result as a new column to adata.obs
        # adata.obs behaves like a pandas DataFrame.
        self.protein_vae.adata.obs["_scvi_library_size"] = log_library_size_protein
        self.rna_vae.adata.obs["_scvi_library_size"] = log_library_size_rna
        # -------------------------------------------------
        library_log_counts_rna = self.rna_vae.adata.obs["_scvi_library_size"].to_numpy()
        library_log_counts_protein = self.protein_vae.adata.obs["_scvi_library_size"].to_numpy()
        # 2. Reshape to be a column vector (N_cells x 1)
        library_log_counts_reshaped_rna = library_log_counts_rna.reshape(-1, 1)
        library_log_counts_reshaped_protein = library_log_counts_protein.reshape(-1, 1)
        # 3. Convert to a PyTorch tensor with the correct data type
        # Also, ensure it's on the same device as your model
        device = self.rna_vae.device
        library_for_rna_generation = torch.tensor(
            library_log_counts_reshaped_rna, dtype=torch.float32, device=device
        )
        library_for_protein_generation = torch.tensor(
            library_log_counts_reshaped_protein, dtype=torch.float32, device=device
        )
        self.rna_vae.library_log_counts = library_for_rna_generation
        self.protein_vae.library_log_counts = library_for_protein_generation
        # Validate constant protein library size
        self._validate_protein_library_size()

        # Ensure both RNA and protein VAE have the same CN categories
        self._unify_cn_categories()

        # Pre-calculate archetype distances to avoid re-computing in each step
        rna_archetypes = torch.tensor(
            self.rna_vae.adata.obsm["archetype_vec"].values, dtype=torch.float32, device=self.device
        )
        protein_archetypes = torch.tensor(
            self.protein_vae.adata.obsm["archetype_vec"].values,
            dtype=torch.float32,
            device=self.device,
        )
        self.archetype_distances = torch.cdist(rna_archetypes, protein_archetypes)
        self.logger_.info(
            f"Pre-calculated archetype distances of shape {self.archetype_distances.shape}."
        )

        # Initialize GradNorm for adaptive loss balancing
        self.gradnorm_enabled = kwargs.pop("gradnorm_enabled", True)
        self.gradnorm_alpha = kwargs.pop("gradnorm_alpha", 1.5)
        self.gradnorm_lr = kwargs.pop("gradnorm_lr", 0.025)

        if self.gradnorm_enabled:
            # Define all possible tasks with their corresponding weights
            all_tasks = [
                ("rna_reconstruction", self.rna_recon_weight),
                ("protein_reconstruction", self.prot_recon_weight),
                ("matching", self.matching_weight),
                ("cell_type_clustering", self.cell_type_clustering_weight),
                ("cross_modal_cell_type", self.cross_modal_cell_type_weight),
                ("cn_distribution_separation", self.cn_distribution_separation_weight),
            ]

            # Filter to only include tasks with non-zero weights
            self.active_tasks = [(name, weight) for name, weight in all_tasks if weight > 0]
            self.task_names = [name for name, weight in self.active_tasks]
            self.task_weights_values = [weight for name, weight in self.active_tasks]

            # Check if we have enough active tasks for GradNorm
            if len(self.active_tasks) < 2:
                self.logger_.warning(
                    f"[GradNorm] Only {len(self.active_tasks)} active tasks found. "
                    "GradNorm requires at least 2 tasks. Disabling GradNorm."
                )
                self.gradnorm_enabled = False
                self.gradnorm = None
                self.gradnorm_optimizer = None
                self.task_names = []
                self.shared_layers = []
            # Also disable GradNorm if only reconstruction losses are active (they don't share parameters effectively)
            elif all(
                name in ["rna_reconstruction", "protein_reconstruction"]
                for name, _ in self.active_tasks
            ):
                self.logger_.warning(
                    f"[GradNorm] Only reconstruction tasks active: {[name for name, _ in self.active_tasks]}. "
                    "These tasks don't share parameters effectively. Disabling GradNorm."
                )
                self.gradnorm_enabled = False
                self.gradnorm = None
                self.gradnorm_optimizer = None
                self.task_names = []
                self.shared_layers = []
            else:
                # Initialize GradNorm with only active tasks
                n_tasks = len(self.active_tasks)
                self.gradnorm = GradNorm(
                    n_tasks=n_tasks, alpha=self.gradnorm_alpha, device=device, logger=self.logger_
                )

                # Get shared layers from both VAE models for gradient computation
                # We'll collect parameters from the encoder layers
                self.shared_layers = []

                # Helper function to extract layers from FCLayers or similar objects
                def extract_layers_from_module(module):
                    layers = []
                    # Try different ways to access layers based on scVI's FCLayers structure
                    if hasattr(module, "children") and callable(module.children):
                        # Use PyTorch's children() method to get submodules
                        children_list = list(module.children())
                        if len(children_list) > 0:
                            # Has children - iterate through them
                            for child in children_list:
                                if hasattr(child, "weight") and child.weight.requires_grad:
                                    layers.append(child)
                        else:
                            # No children but might be a leaf module with weights
                            if hasattr(module, "weight") and module.weight.requires_grad:
                                layers.append(module)
                    elif hasattr(module, "weight") and module.weight.requires_grad:
                        # Direct layer with weights
                        layers.append(module)
                    return layers

                # Add RNA VAE encoder layers
                if hasattr(self.rna_vae.module, "z_encoder") and hasattr(
                    self.rna_vae.module.z_encoder, "encoder"
                ):
                    rna_encoder_layers = extract_layers_from_module(
                        self.rna_vae.module.z_encoder.encoder
                    )
                    self.shared_layers.extend(rna_encoder_layers)

                # Add protein VAE encoder layers
                if hasattr(self.protein_vae.module, "z_encoder") and hasattr(
                    self.protein_vae.module.z_encoder, "encoder"
                ):
                    prot_encoder_layers = extract_layers_from_module(
                        self.protein_vae.module.z_encoder.encoder
                    )
                    self.shared_layers.extend(prot_encoder_layers)

                # If no layers found from encoders, fall back to using parameters from the modules directly
                if len(self.shared_layers) == 0:
                    self.logger_.warning(
                        "[GradNorm] No encoder layers found, using module parameters as fallback"
                    )
                    # Use a subset of model parameters as shared layers
                    for param in list(self.rna_vae.module.parameters())[:3]:  # First few parameters
                        if param.requires_grad:
                            self.shared_layers.append(param)
                    for param in list(self.protein_vae.module.parameters())[
                        :3
                    ]:  # First few parameters
                        if param.requires_grad:
                            self.shared_layers.append(param)

                self.logger_.info(
                    f"[GradNorm] Initialized with {n_tasks} active tasks: {self.task_names}"
                )
                self.logger_.info(f"[GradNorm] Task weights: {self.task_weights_values}")
                self.logger_.info(
                    f"[GradNorm] Alpha: {self.gradnorm_alpha}, Learning rate: {self.gradnorm_lr}"
                )
                self.logger_.info(
                    f"[GradNorm] Found {len(self.shared_layers)} shared layers for gradient computation"
                )

                # Create separate optimizer for GradNorm weights
                self.gradnorm_optimizer = torch.optim.Adam(
                    [self.gradnorm.task_weights], lr=self.gradnorm_lr
                )

        else:
            self.gradnorm = None
            self.gradnorm_optimizer = None
            self.task_names = []
            self.shared_layers = []
            self.logger_.info("[GradNorm] Disabled")

        # Save training parameters
        self.save_training_parameters(kwargs)

    def _freeze_models(self):
        """Freeze model parameters to prevent weight updates during warmup."""
        for param in self.rna_vae.module.parameters():
            param.requires_grad = False
        for param in self.protein_vae.module.parameters():
            param.requires_grad = False

    def _unfreeze_models(self):
        """Unfreeze model parameters to allow weight updates after warmup."""
        for param in self.rna_vae.module.parameters():
            param.requires_grad = True
        for param in self.protein_vae.module.parameters():
            param.requires_grad = True

    def _validate_protein_library_size(self):
        """Validate that protein library size is constant as required for counterfactual generation."""
        # Check if protein data has normalize_total_value in uns
        if "normalize_total_value" not in self.protein_vae.adata.uns:
            raise ValueError(
                "protein_vae.adata.uns must contain 'normalize_total_value' "
                "indicating the constant library size used for normalization"
            )

        expected_lib_size = self.protein_vae.adata.uns["normalize_total_value"]
        self.logger_.info(f"Expected constant protein library size: {expected_lib_size}")

        # Check that all protein library sizes are approximately the same
        protein_lib_sizes = self.protein_vae.adata.obs["_scvi_library_size"].values
        exp_protein_lib_sizes = np.exp(protein_lib_sizes)  # Convert from log space

        # Calculate coefficient of variation (std/mean)
        cv = np.std(exp_protein_lib_sizes) / np.mean(exp_protein_lib_sizes)
        self.logger_.info(
            f"Protein library sizes - Mean: {np.mean(exp_protein_lib_sizes):.2f}, "
            f"Std: {np.std(exp_protein_lib_sizes):.2f}, CV: {cv:.4f}"
        )

        # Allow small variation due to floating point precision (CV < 0.01 = 1%)
        if cv > 0.01:
            self.logger_.warning(
                f"Protein library sizes are not constant! CV = {cv:.4f} > 0.01. "
                f"This may affect counterfactual generation quality."
            )
        else:
            self.logger_.info(
                "✓ Protein library sizes are sufficiently constant for counterfactual generation"
            )

        # Store constant protein library size for counterfactual generation
        self.constant_protein_lib_size = expected_lib_size
        self.constant_protein_lib_size_log = np.log(expected_lib_size)

    def _unify_cn_categories(self):
        """Ensure both RNA and protein VAE have the same CN categories to prevent NaN during dynamic assignment."""
        # Check if both datasets have CN column
        if "CN" not in self.rna_vae.adata.obs.columns:
            self.logger_.warning(
                "RNA dataset missing 'CN' column - skipping CN category unification"
            )
            return

        if "CN" not in self.protein_vae.adata.obs.columns:
            self.logger_.warning(
                "Protein dataset missing 'CN' column - skipping CN category unification"
            )
            return

        # Get current categories from both datasets
        rna_cn_categories = set(self.rna_vae.adata.obs["CN"].cat.categories)
        protein_cn_categories = set(self.protein_vae.adata.obs["CN"].cat.categories)

        # Create union of all categories
        all_cn_categories = sorted(rna_cn_categories.union(protein_cn_categories))

        self.logger_.info(f"RNA CN categories: {sorted(rna_cn_categories)}")
        self.logger_.info(f"Protein CN categories: {sorted(protein_cn_categories)}")
        self.logger_.info(f"Unified CN categories: {all_cn_categories}")

        # Update both datasets to have the same categories
        self.rna_vae.adata.obs["CN"] = self.rna_vae.adata.obs["CN"].cat.set_categories(
            all_cn_categories
        )
        self.protein_vae.adata.obs["CN"] = self.protein_vae.adata.obs["CN"].cat.set_categories(
            all_cn_categories
        )

        # Verify unification worked
        rna_final_categories = set(self.rna_vae.adata.obs["CN"].cat.categories)
        protein_final_categories = set(self.protein_vae.adata.obs["CN"].cat.categories)

        if rna_final_categories == protein_final_categories == set(all_cn_categories):
            self.logger_.info(
                f"✓ Successfully unified CN categories: {len(all_cn_categories)} categories"
            )
        else:
            self.logger_.error("Failed to unify CN categories!")
            self.logger_.error(f"RNA final: {sorted(rna_final_categories)}")
            self.logger_.error(f"Protein final: {sorted(protein_final_categories)}")
            raise ValueError("CN category unification failed")

    def detect_outlier_loss(self, current_loss, loss_history, loss_type="unknown"):
        """
        Detect if the current loss is an outlier based on running statistics.

        Args:
            current_loss: Current loss value (tensor or float)
            loss_history: List of previous loss values
            loss_type: String identifier for the loss type (for logging)

        Returns:
            tuple: (is_outlier: bool, corrected_loss: float)
        """
        if not self.outlier_detection_enabled:
            return False, current_loss.item() if hasattr(current_loss, "item") else current_loss

        current_loss_val = current_loss.item() if hasattr(current_loss, "item") else current_loss

        # Need at least a few samples to calculate meaningful statistics
        if len(loss_history) < 10:
            loss_history.append(current_loss_val)
            return False, current_loss_val

        # Keep only the most recent values for running statistics
        if len(loss_history) > self.outlier_window_size:
            loss_history.pop(0)

        # Calculate running mean and std
        history_array = np.array(loss_history)
        running_mean = np.mean(history_array)
        running_std = np.std(history_array)

        # Avoid division by zero
        if running_std < 1e-8:
            loss_history.append(current_loss_val)
            return False, current_loss_val

        # Calculate z-score
        z_score = abs(current_loss_val - running_mean) / running_std

        # Check if it's an outlier
        is_outlier = z_score > self.outlier_z_threshold

        if is_outlier:
            # Use the most recent value from history (guaranteed to exist since len >= 10)
            corrected_loss = loss_history[-1]  # Use previous step's value

            self.logger_.warning(
                f"Outlier detected in {loss_type} loss! "
                f"Current: {current_loss_val:.6f}, Mean: {running_mean:.6f}, "
                f"Std: {running_std:.6f}, Z-score: {z_score:.2f}. "
                f"Replacing with: {corrected_loss:.6f}"
            )

            # Add the corrected value to history instead of the outlier
            loss_history.append(corrected_loss)
            return True, corrected_loss
        else:
            # Normal value, add to history
            loss_history.append(current_loss_val)
            return False, current_loss_val

    def log_to_mlflow_centralized(self, step_type="step", extra_metrics=None):
        """Centralized MLflow logging function.

        Args:
            step_type: Type of logging - "step", "epoch", or "final"
            extra_metrics: Additional metrics to log
        """
        if not mlflow.active_run():
            return

        # Skip during warmup unless using cached scales
        if step_type != "final":
            should_log = False
            if self.scales_from_cache:
                # Using cached scales - log immediately
                should_log = True
            elif self.current_epoch >= self.warmup_epochs:
                # Past warmup period - log normally
                should_log = True
            else:
                # During warmup period - don't log
                should_log = False

            if not should_log:
                return

        metrics_to_log = {}

        if step_type == "step":
            # Step-level metrics (learning rates)
            if hasattr(self, "current_rna_lr"):
                metrics_to_log["train/lr_rna"] = self.current_rna_lr
                metrics_to_log["train/lr_protein"] = self.current_protein_lr

        elif step_type == "epoch":
            # Epoch-level metrics (weight updates, epoch metrics)
            if hasattr(self, "weight_update_l2_rna"):
                metrics_to_log["train/weight_update_l2_rna"] = self.weight_update_l2_rna
                metrics_to_log["train/weight_update_l2_protein"] = self.weight_update_l2_protein

            # Note: Validation metrics are logged separately in on_validation_epoch_end
            # to ensure they only appear at discrete validation epochs, not every training epoch

            # Add training epoch metrics with proper prefixes (silhouette, ARI, etc.)
            if hasattr(self, "metrics_history") and len(self.metrics_history) > 0:
                latest_metrics = self.metrics_history[-1]
                for key, value in latest_metrics.items():
                    if key.startswith("train_"):
                        # Handle dict values (e.g., ari_f1_score)
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                metrics_to_log[f"train/{key[6:]}_{sub_key}"] = sub_value
                        else:
                            metrics_to_log[f"train/{key[6:]}"] = value

        elif step_type == "final":
            # Final metrics (best metrics) - use a unique step value to avoid conflicts
            if hasattr(self, "final_best_metrics"):
                for k, v in self.final_best_metrics.items():
                    # Handle dict values (e.g., ari_f1_score)
                    if isinstance(v, dict):
                        for sub_key, sub_value in v.items():
                            if k.startswith("val_"):
                                metrics_to_log[f"final/val/{k[4:]}_{sub_key}"] = sub_value
                            elif k.startswith("train_"):
                                metrics_to_log[f"final/train/{k[6:]}_{sub_key}"] = sub_value
                            else:
                                metrics_to_log[f"final/val/{k}_{sub_key}"] = sub_value
                    else:
                        if k.startswith("val_"):
                            metrics_to_log[f"final/val/{k[4:]}"] = v
                        elif k.startswith("train_"):
                            metrics_to_log[f"final/train/{k[6:]}"] = v
                        else:
                            # For metrics without prefix, assume they're validation metrics
                            metrics_to_log[f"final/val/{k}"] = v

        # Add extra metrics if provided
        if extra_metrics:
            metrics_to_log.update(extra_metrics)

        # Log all metrics (filter out zero values to reduce noise)
        if metrics_to_log:
            # Filter out zero losses to reduce MLflow noise
            filtered_metrics = {
                k: v
                for k, v in metrics_to_log.items()
                if not (isinstance(v, (int, float)) and v == 0.0)
            }

            if filtered_metrics:
                # Determine step for logging
                if step_type == "step":
                    step = self.global_step
                elif step_type == "epoch":
                    step = self.current_epoch
                else:  # final - use a unique step value to avoid conflicts
                    step = self.max_epochs + 1000  # Use a step far beyond training epochs

                mlflow.log_metrics(filtered_metrics, step=step)

    def save_training_parameters(self, kwargs):
        """Save training parameters to a JSON file in the checkpoints directory."""
        # Add the parameters from self that aren't in kwargs
        params = {
            "batch_size": self.batch_size,
            "max_epochs": self.total_steps
            // int(np.ceil(len(self.rna_vae.adata) / self.batch_size)),
            "similarity_weight": self.similarity_weight,
            "similarity_dynamic": self.similarity_dynamic,
            "cell_type_clustering_weight": self.cell_type_clustering_weight,
            "cross_modal_cell_type_weight": self.cross_modal_cell_type_weight,
            "lr": self.lr,
            "lr_warmup_epochs": self.lr_warmup_epochs,
            "rna_recon_weight": self.rna_recon_weight,
            "prot_recon_weight": self.prot_recon_weight,
            "contrastive_weight": self.contrastive_weight,
            "plot_x_times": self.plot_x_times,
            "steady_state_window": self.steady_state_window,
            "steady_state_tolerance": self.steady_state_tolerance,
            "reactivation_threshold": self.reactivation_threshold,
            "adata_rna_shape": list(self.rna_vae.adata.shape),
            "protein_adata_shape": list(self.protein_vae.adata.shape),
            "latent_dim": self.rna_vae.module.n_latent,
            "device": str(self.device),
            "timestamp": self.run_timestamp,
            "log_file_path": str(self.log_file_path),  # Add log file path
            "load_optimizer_state": self.load_optimizer_state,  # Add optimizer loading flag
            # Outlier detection parameters
            "outlier_detection_enabled": self.outlier_detection_enabled,
            "outlier_z_threshold": self.outlier_z_threshold,
            "outlier_window_size": self.outlier_window_size,
            "outlier_count_rna": getattr(self, "outlier_count_rna", 0),
            "outlier_count_protein": getattr(self, "outlier_count_protein", 0),
        }

        # Remove non-serializable objects
        params_to_save = {
            k: v for k, v in params.items() if isinstance(v, (str, int, float, bool, list, dict))
        }

        # Save parameters to JSON file
        with open(f"{self.checkpoints_dir}/training_parameters.json", "w") as f:
            json.dump(params_to_save, f, indent=4)

        self.logger_.info(
            f"Training parameters saved to {self.checkpoints_dir}/training_parameters.json"
        )

        # Also save parameters to a separate txt file in readable format
        with open(f"{self.checkpoints_dir}/training_parameters.txt", "w") as f:
            f.write("Training Parameters:\n")
            for key, value in params_to_save.items():
                f.write(f"{key}: {value}\n")

    def backward(self, loss, *args, **kwargs):
        """Override backward to skip during warmup phase."""
        if self.current_epoch < self.warmup_epochs:
            # Skip backward pass during warmup - parameters are frozen anyway
            return
        # Normal backward pass after warmup
        super().backward(loss, *args, **kwargs)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """Override optimizer step to skip during warmup phase."""
        if self.current_epoch < self.warmup_epochs:
            # During warmup, execute closure to compute loss (needed for scaling)
            # Backward is skipped via backward() override, so no gradients computed
            # This satisfies Lightning's requirement that closure must be executed
            if optimizer_closure is not None:
                optimizer_closure()
            # Don't call optimizer.step() - no gradients to apply anyway
            return
        # Normal optimizer step after warmup
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def configure_optimizers(self):
        """Configure optimizers for the model.

        Returns:
            dict: Dictionary with optimizer and lr scheduler.
        """
        # Use separate learning rates for RNA (higher) and protein (standard)
        rna_params = self.rna_vae.module.parameters()
        protein_params = self.protein_vae.module.parameters()

        # Use a higher learning rate for RNA to help unstick it
        rna_lr = self.lr
        protein_lr = self.lr

        # Combined parameters with different parameter groups
        all_params = [
            {"params": rna_params, "lr": rna_lr},
            {"params": protein_params, "lr": protein_lr},
        ]

        # CRITICAL FIX: Do NOT pass global lr parameter when using parameter groups
        # The global lr would override the parameter group learning rates
        optimizer = torch.optim.AdamW(
            all_params,
            # lr=self.lr,  # REMOVED: This was overriding parameter group learning rates
            weight_decay=5e-2,
        )

        if self.training_state is not None and self.load_optimizer_state:
            optimizer.load_state_dict(self.training_state["optimizer_state"])
            self.logger_.info(f"Training state and optimizer state loaded from previous run")
        elif self.training_state is not None and not self.load_optimizer_state:
            self.logger_.info(
                f"Training state loaded BUT optimizer state skipped (fresh learning rates)"
            )
        else:
            self.logger_.info(f"No training state to load - starting fresh")

        # LR Scheduler with warmup phases
        scale_warmup_steps = self.warmup_epochs * self.steps_per_epoch
        lr_warmup_steps = self.lr_warmup_epochs * self.steps_per_epoch

        if scale_warmup_steps + lr_warmup_steps == 0:
            # No warmup, use direct scheduler
            warmup_reason = (
                "cached scales"
                if hasattr(self, "scales_from_cache") and self.scales_from_cache
                else "disabled warmup"
            )
            self.logger_.info(
                f"Warmup disabled ({warmup_reason}), using direct ReduceLROnPlateau scheduler"
            )
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.9,
                patience=10,
                threshold=0.05,
                min_lr=1e-9,
                verbose=True,
                cooldown=10,
            )
        else:
            # FIXED: Proper warmup logic with working scheduler
            if scale_warmup_steps + lr_warmup_steps > self.total_steps:
                self.logger_.warning(
                    f"Total warmup steps ({scale_warmup_steps + lr_warmup_steps}) is greater than total steps ({self.total_steps}). "
                    "The learning rate schedule will be truncated. "
                    "Consider reducing warmup_epochs or lr_warmup_epochs."
                )
                if scale_warmup_steps > self.total_steps:
                    scale_warmup_steps = self.total_steps
                    lr_warmup_steps = 0
                else:
                    lr_warmup_steps = self.total_steps - scale_warmup_steps

            self.total_steps - scale_warmup_steps - lr_warmup_steps

            # Scale warmup uses very low LR for loss assessment
            scale_warmup_factor = 0.1
            lr_warmup_start_factor = 0.1
            lr_warmup_end_factor = 1.0

            # Phase 1: Scale warmup with low LR for loss assessment
            scale_warmup_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=scale_warmup_factor, total_iters=scale_warmup_steps
            )

            # Phase 2: Linear LR warmup
            lr_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=lr_warmup_start_factor,
                end_factor=lr_warmup_end_factor,
                total_iters=lr_warmup_steps,
            )

            # Phase 3: Main training with ReduceLROnPlateau
            main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.95,
                patience=10,
                threshold=0.05,
                min_lr=1e-6,
                verbose=True,
                cooldown=10,
            )

            # SIMPLIFIED: Just use ReduceLROnPlateau after warmup epochs
            # The warmup is handled by the loss scaling system, not the LR scheduler
            lr_scheduler = main_scheduler
            self.logger_.info(
                "Using ReduceLROnPlateau scheduler (warmup handled by loss scaling system)"
            )

            self.logger_.info(
                f"Warmup system enabled: scale_warmup={scale_warmup_steps} steps, lr_warmup={lr_warmup_steps} steps"
            )

        # Log effective learning rates for debugging
        self.logger_.info(f"Learning rate configuration (FIXED - PROPER WARMUP + SCHEDULER):")
        self.logger_.info(f"  - RNA base lr: {rna_lr:.2e}")
        self.logger_.info(f"  - Protein base lr: {protein_lr:.2e}")
        if scale_warmup_steps + lr_warmup_steps == 0:
            self.logger_.info(f"  - Direct ReduceLROnPlateau scheduler (patience=20, factor=0.95)")
        else:
            self.logger_.info(
                f"  - Scale warmup ({scale_warmup_steps} steps): lr * {scale_warmup_factor:.2e}"
            )
            self.logger_.info(
                f"  - LR warmup ({lr_warmup_steps} steps): {lr_warmup_start_factor:.2e} → {lr_warmup_end_factor:.2e}"
            )
            self.logger_.info(
                f"  - Main training: ReduceLROnPlateau monitoring training loss (patience=20, factor=0.95)"
            )
        self.logger_.info(f"  - Gradient clipping: {self.gradient_clip_val}")

        # Log effective learning rates during each phase
        if scale_warmup_steps + lr_warmup_steps > 0:
            effective_scale_warmup_rna = rna_lr * scale_warmup_factor
            effective_scale_warmup_protein = protein_lr * scale_warmup_factor
            self.logger_.info(
                f"  - Effective scale warmup lr - RNA: {effective_scale_warmup_rna:.2e}, Protein: {effective_scale_warmup_protein:.2e}"
            )

        # Always use epoch-based ReduceLROnPlateau (simplified approach)
        scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",  # ReduceLROnPlateau works on epochs
            "frequency": 1,
            "monitor": "epoch_train_loss",  # Monitor epoch-level training loss
            "strict": False,  # Don't fail if metric is missing
        }

        d = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
            "gradient_clip_val": self.gradient_clip_val,
            "gradient_clip_algorithm": "value",
        }
        return d

    def training_step(self, batch, batch_idx):
        self.mode = "training"
        self.rna_vae.module.train()
        self.protein_vae.module.train()

        indices = range(
            min(
                self.batch_size,
                len(self.train_indices_rna),
                len(self.train_indices_prot),
            )
        )

        self.train_step_ += 1

        # Sample RNA/protein indices with REPLACEMENT if the pool is smaller than the requested size
        indices_rna = np.random.choice(
            self.train_indices_rna,
            size=len(indices),
            replace=True if len(indices) > len(self.train_indices_rna) else False,
        )
        indices_prot = np.random.choice(
            self.train_indices_prot,
            size=len(indices),
            replace=True if len(indices) > len(self.train_indices_prot) else False,
        )

        # Build torch tensors per modality
        rna_batch = self._get_rna_batch(batch, indices_rna)
        protein_batch = self._get_protein_batch(batch, indices_prot)

        # diagnostic plot - only if plot_x_times > 0
        if self.global_step == 0 and self.plot_first_step and self.plot_x_times > 0:

            plot_first_batch_umaps(
                rna_batch,
                protein_batch,
                self.rna_vae,
                self.protein_vae,
                adata_rna=self.rna_vae.adata,
                adata_prot=self.protein_vae.adata,
                mlflow_name="first_batch",
                colors="cell_types",
                use_subsample=True,
            )

        check_ilisi = self.global_step % self.ilisi_check_frequency == 0
        steps_in_epoch = int(np.ceil(len(self.train_indices_rna) / self.batch_size) - 1)
        is_last_step_of_epoch = self.train_step_ >= steps_in_epoch

        plot_interval = (
            (1 + int(self.total_steps / self.plot_x_times)) if self.plot_x_times > 0 else -1
        )

        to_plot = False  # default: no plotting

        if self.plot_x_times > 0:
            to_plot = (
                (
                    plot_interval > 0
                    and self.global_step > 0
                    and self.global_step % plot_interval == 0
                )
                # Optionally plot on the very first step if explicitly requested
                or (self.global_step == 0 and self.plot_first_step)
                # Always plot on the very last step of training
                or self.global_step >= self.total_steps - 1
            )
        else:
            # plotting disabled when plot_x_times is 0
            to_plot = False

        # Determine if we should print to console based on print_every_n_epoch
        self.to_print = (
            is_last_step_of_epoch and (self.current_epoch % self.print_every_n_epoch == 0)
        ) or self.global_step == 0
        rna_inference_outputs_tuple = self.rna_vae.module(rna_batch)
        protein_inference_outputs_tuple = self.protein_vae.module(protein_batch)
        losses = self.calculate_losses(
            rna_batch=rna_batch,
            protein_batch=protein_batch,
            rna_vae=self.rna_vae,
            protein_vae=self.protein_vae,
            rna_inference_outputs_tuple=rna_inference_outputs_tuple,
            protein_inference_outputs_tuple=protein_inference_outputs_tuple,
            device=self.device,
            similarity_weight=self.similarity_weight,
            similarity_active=self.similarity_active,
            contrastive_weight=self.contrastive_weight,
            matching_weight=self.matching_weight,
            cell_type_clustering_weight=self.cell_type_clustering_weight,
            cross_modal_cell_type_weight=self.cross_modal_cell_type_weight,
            rna_recon_weight=self.rna_recon_weight,
            prot_recon_weight=self.prot_recon_weight,
            global_step=self.global_step,
            total_steps=self.total_steps,
            to_plot=to_plot,
            check_ilisi=check_ilisi,
        )
        if self.current_epoch < self.warmup_epochs:
            raw_losses_dict = losses.pop("raw_losses_for_scaling")
            for name, value in raw_losses_dict.items():
                self.warmup_raw_losses[name].append(value.item())
        elif self.scales_from_cache and self.first_step:
            # Log that we're using cached scales from the start
            self.logger_.info(
                f"Training step 0: Using cached loss scales from start: {self.loss_scales}"
            )
            self.first_step = False

        self.update_similarity_weight(losses)

        if len(self.similarity_loss_history) >= self.steady_state_window:
            self.similarity_loss_history.pop(0)
        self.similarity_loss_history.append(losses["raw_similarity_loss"])

        in_steady_state = False
        if len(self.similarity_loss_history) == self.steady_state_window:
            mean_loss = sum(self.similarity_loss_history) / self.steady_state_window
            std_loss = (
                sum((x - mean_loss) ** 2 for x in self.similarity_loss_history)
                / self.steady_state_window
            ) ** 0.5

            # Check if variation is small enough to be considered steady state
            coeff_of_variation = std_loss / mean_loss if mean_loss > 0 else float("inf")
            in_steady_state = coeff_of_variation < self.steady_state_tolerance

        # Update steady state counter and status (only if similarity_dynamic is True)
        if self.similarity_dynamic:
            if in_steady_state and self.similarity_active:
                self.similarity_loss_steady_counter += 1
                if self.similarity_loss_steady_counter >= self.similarity_loss_steady_threshold:
                    self.similarity_active = False
                    self.logger_.info(
                        f"[Step {self.global_step}] DEACTIVATING similarity loss - In steady state for {self.similarity_loss_steady_counter} steps"
                    )
            elif not in_steady_state and self.similarity_active:
                self.similarity_loss_steady_counter = 0  # Reset counter if not in steady state

            # Check for loss increase if similarity is currently inactive
            if not self.similarity_active and len(self.similarity_loss_history) > 0:
                recent_loss = losses["raw_similarity_loss"].item()
                min_steady_loss = min(self.similarity_loss_history)

                if recent_loss > min_steady_loss * (1 + self.reactivation_threshold):
                    # Loss has increased significantly, reactivate similarity loss
                    self.similarity_active = True
                    self.similarity_loss_steady_counter = 0
                    self.logger_.info(
                        f"[Step {self.global_step}] REACTIVATING similarity loss - Loss increased from {min_steady_loss:.4f} to {recent_loss:.4f}"
                    )
        else:
            # When similarity_dynamic=False, keep similarity_active constant
            # Just log the steady state status for monitoring without changing similarity_active
            if self.global_step <= 5:  # Only log for first few steps to avoid spam
                self.logger_.info(
                    f"[Step {self.global_step}] Similarity dynamic disabled - keeping similarity_active={self.similarity_active}"
                )

        # Store losses in history
        # NOTE: Individual task losses (rna_loss, protein_loss, etc.) are stored as original values
        # When GradNorm is active, these represent the unweighted task losses for debugging/monitoring
        # The actual weighted contributions are logged separately as gradnorm_weighted_* metrics
        self.similarity_losses.append(losses["similarity_loss"].item())
        self.raw_similarity_losses.append(losses["raw_similarity_loss"].item())
        self.similarity_weight_history.append(self.similarity_weight)
        self.active_similarity_loss_active_history.append(self.similarity_active)
        self.train_losses.append(
            losses["total_loss"].item()
        )  # This will be the original total when GradNorm is active
        self.train_rna_losses.append(
            losses["rna_loss"].item()
        )  # Original RNA loss (unweighted by GradNorm)
        self.train_protein_losses.append(
            losses["protein_loss"].item()
        )  # Original protein loss (unweighted by GradNorm)
        self.train_matching_losses.append(losses["matching_loss"].item())
        self.train_contrastive_losses.append(losses["contrastive_loss"].item())
        self.train_adv_losses.append(losses["adv_loss"].item() if "adv_loss" in losses else 0.0)
        self.train_cell_type_clustering_losses.append(losses["cell_type_clustering_loss"].item())
        self.train_cross_modal_cell_type_losses.append(losses["cross_modal_cell_type_loss"].item())
        self.train_cn_separation_losses.append(losses["cn_distribution_separation_loss"].item())
        if to_plot:
            plot_similarity_loss_history(
                self.similarity_losses,
                self.active_similarity_loss_active_history,
                self.global_step,
                similarity_weight=self.similarity_weight,
                similarity_dynamic=self.similarity_dynamic,
                plot_flag=to_plot,
            )

        # Determine if we should log to MLflow (skip during warmup unless using cached scales)
        should_log_to_mlflow = False
        if self.scales_from_cache:
            # Using cached scales - log immediately
            should_log_to_mlflow = True
            if self.global_step <= 5:  # Debug log for first few steps
                self.logger_.info(
                    f"[DEBUG] Step {self.global_step}: Logging to MLflow (cached scales)"
                )
        elif self.current_epoch >= self.warmup_epochs:
            # Past warmup period - log normally
            should_log_to_mlflow = True
            if self.global_step <= 5 or self.current_epoch == self.warmup_epochs:  # Debug log
                self.logger_.info(
                    f"[DEBUG] Step {self.global_step}, Epoch {self.current_epoch}: Logging to MLflow (past warmup)"
                )
        else:
            # During warmup period - don't log
            should_log_to_mlflow = False
            if self.global_step <= 5 or self.current_epoch <= 2:  # Debug log for first few epochs
                self.logger_.info(
                    f"[DEBUG] Step {self.global_step}, Epoch {self.current_epoch}: NOT logging to MLflow (warmup period, warmup_epochs={self.warmup_epochs})"
                )

        # Filter out zero-weighted losses before logging
        filtered_losses = losses.copy()

        # Define loss-weight mappings for zero-weight filtering
        loss_weight_mappings = [
            (self.contrastive_weight, ["contrastive_loss"]),
            (self.matching_weight, ["matching_loss"]),
            (self.similarity_weight, ["similarity_loss", "raw_similarity_loss"]),
            (self.cell_type_clustering_weight, ["cell_type_clustering_loss"]),
            (self.cross_modal_cell_type_weight, ["cross_modal_cell_type_loss"]),
            (self.cn_distribution_separation_weight, ["cn_distribution_separation_loss"]),
        ]

        # Remove losses with zero weights
        for weight, loss_names in loss_weight_mappings:
            if weight == 0:
                for loss_name in loss_names:
                    filtered_losses.pop(loss_name, None)

        log_step(
            filtered_losses,
            metrics=None,
            global_step=self.global_step,
            current_epoch=self.current_epoch,
            is_validation=False,
            total_steps=self.total_steps,
            print_to_console=self.to_print,
            log_to_mlflow=should_log_to_mlflow,
        )

        # Check if this is the last step of the epoch
        is_last_step_of_training = (
            self.current_epoch + 1 >= self.max_epochs
        ) and is_last_step_of_epoch

        if is_last_step_of_epoch:
            if self.to_print:
                self.logger_.info(
                    f"Completed last step of epoch {self.current_epoch} in global step {self.global_step}/{self.total_steps}"
                )
            self.train_step_ = 0
            self.on_epoch_end()
        if is_last_step_of_training:
            self.logger_.info(f"Completed last step of training at epoch {self.current_epoch}")
            # Call the custom end of training function if it hasn't been called yet
            if not self.on_train_end_custom_called:
                self.on_train_end_custom(plot_flag=True)
                self.on_train_end_custom_called = True

        # During warmup, return the loss normally but skip optimizer step
        # Parameters are frozen so no gradients will flow to them
        if self.current_epoch < self.warmup_epochs:
            if self.global_step == 0:
                self.logger_.info(f"[Warmup] Model weights frozen. Optimizer step will be skipped.")
            # Return the actual loss - backward() will be called but no gradients
            # will flow to frozen parameters, and optimizer_step will be skipped
            final_loss = losses["total_loss"]
            return final_loss

        # Apply GradNorm for adaptive loss balancing
        final_loss = losses["total_loss"]
        if (
            self.gradnorm_enabled
            and self.gradnorm is not None
            and self.current_epoch >= self.warmup_epochs
        ):
            # Map task names to their corresponding loss values
            loss_mapping = {
                "rna_reconstruction": losses["rna_loss"],
                "protein_reconstruction": losses["protein_loss"],
                "matching": losses["matching_loss"],
                "cell_type_clustering": losses["cell_type_clustering_loss"],
                "cross_modal_cell_type": losses["cross_modal_cell_type_loss"],
                "cn_distribution_separation": losses["cn_distribution_separation_loss"],
            }

            # Extract only the losses for active tasks (tasks with non-zero weights)
            task_losses = [loss_mapping[task_name] for task_name in self.task_names]

            # Store original task losses for Lightning (before gradnorm modifies them)
            # These keep their connection to model parameters but are separate tensors
            original_task_losses = [loss.clone() for loss in task_losses]

            # Apply GradNorm with separate RNA and Protein VAE models
            # Use detached copies of task losses for GradNorm to avoid double backward
            task_losses_detached = [loss.detach().requires_grad_(True) for loss in task_losses]
            balanced_loss, gradnorm_loss, updated_weights = self.gradnorm(
                task_losses_detached,
                self.shared_layers,  # Kept for compatibility but not used
                task_names=self.task_names,
                global_step=self.global_step,
                rna_model=self.rna_vae.module,  # Pass RNA VAE model
                protein_model=self.protein_vae.module,  # Pass Protein VAE model
            )

            # Update GradNorm weights
            self.gradnorm_optimizer.zero_grad()
            gradnorm_loss.backward(retain_graph=False)
            self.gradnorm_optimizer.step()

            # Get updated weights (detached, no gradients)
            with torch.no_grad():
                new_weights = self.gradnorm.task_weights.detach().clone()
            # Create final loss using the ORIGINAL task losses with new weights
            # This ensures Lightning gets a loss that was never used in gradnorm backward
            weighted_losses = new_weights.detach() * torch.stack(original_task_losses)
            final_loss = torch.sum(weighted_losses)

            # Calculate GradNorm-weighted individual task contributions
            gradnorm_weighted_losses = {}
            for i, (task_name, task_loss) in enumerate(zip(self.task_names, task_losses)):
                weighted_contribution = (updated_weights[i] * task_loss).item()
                gradnorm_weighted_losses[task_name] = weighted_contribution

            # Log GradNorm information and metrics
            if self.to_print or (self.global_step % 100 == 0):
                self.logger_.info(
                    f"[GradNorm Step {self.global_step}] GradNorm Loss: {gradnorm_loss.item():.6f}"
                )
                self.logger_.info(
                    f"[GradNorm Step {self.global_step}] Updated task weights: {updated_weights}"
                )

                # Log weighted contributions
                weighted_contrib_str = ", ".join(
                    [f"{name}: {contrib:.4f}" for name, contrib in gradnorm_weighted_losses.items()]
                )
                self.logger_.info(
                    f"[GradNorm Step {self.global_step}] Weighted contributions: {weighted_contrib_str}"
                )

            # Note: MLflow logging is handled centrally by log_step() to avoid duplicates
            # GradNorm-specific metrics will be logged there if needed

            # Store GradNorm information for history tracking
            if not hasattr(self, "gradnorm_history"):
                self.gradnorm_history = {
                    "gradnorm_losses": [],
                    "task_weights": [],
                    "weighted_contributions": [],
                }

            self.gradnorm_history["gradnorm_losses"].append(gradnorm_loss.item())
            self.gradnorm_history["task_weights"].append(
                updated_weights.clone().detach().cpu().numpy()
            )
            self.gradnorm_history["weighted_contributions"].append(gradnorm_weighted_losses.copy())

        return final_loss

    def validation_step(self, batch, batch_idx):
        self.mode = "validation"
        """Validation step using the same loss calculations as training."""
        # Get validation batches
        self.rna_vae.module.eval()  # Keep models in eval mode for validation
        self.protein_vae.module.eval()
        indices = range(self.val_batch_size)  # Use val_batch_size instead of batch_size
        self.validation_step_ += 1
        indices_prot = np.random.choice(
            self.val_indices_prot,
            size=len(indices),
            replace=True if len(indices) > len(self.val_indices_prot) else False,
        )
        indices_rna = np.random.choice(
            self.val_indices_rna,
            size=len(indices),
            replace=True if len(indices) > len(self.val_indices_rna) else False,
        )
        rna_batch = self._get_rna_batch(batch, indices_rna)
        protein_batch = self._get_protein_batch(batch, indices_prot)

        # Check if we should calculate iLISI for this validation step
        # Calculate iLISI every 5 validation steps or at the start and end of validation
        val_steps_per_epoch = int(
            np.ceil(len(self.val_indices_rna) / self.val_batch_size)
        )  # Use val_batch_size
        check_ilisi = (
            self.validation_step_ % 5 == 0
            or self.validation_step_ == 1  # First step
            or self.validation_step_ >= val_steps_per_epoch  # Last step
        )
        # Same plotting guard as in training_step
        plot_interval = (
            (1 + int(self.total_steps / self.plot_x_times)) if self.plot_x_times > 0 else -1
        )

        to_plot = False
        if self.plot_x_times > 0:
            to_plot = (
                plot_interval > 0 and self.global_step > 0 and self.global_step % plot_interval == 0
            ) or (self.global_step == 0 and self.plot_first_step)
        else:
            # plotting disabled when plot_x_times is 0
            to_plot = False

        # Foward pass loss calculation
        # Calculate all losses using the same function as training
        with torch.no_grad():
            rna_inference_outputs_tuple = self.rna_vae.module(rna_batch)
            protein_inference_outputs_tuple = self.protein_vae.module(protein_batch)

        losses = self.calculate_losses(
            rna_batch=rna_batch,
            protein_batch=protein_batch,
            rna_vae=self.rna_vae,
            protein_vae=self.protein_vae,
            rna_inference_outputs_tuple=rna_inference_outputs_tuple,
            protein_inference_outputs_tuple=protein_inference_outputs_tuple,
            device=self.device,
            similarity_weight=self.similarity_weight,
            similarity_active=self.similarity_active,
            contrastive_weight=self.contrastive_weight,
            matching_weight=self.matching_weight,
            cell_type_clustering_weight=self.cell_type_clustering_weight,
            cross_modal_cell_type_weight=self.cross_modal_cell_type_weight,
            rna_recon_weight=self.rna_recon_weight,
            prot_recon_weight=self.prot_recon_weight,
            check_ilisi=check_ilisi,
            to_plot=to_plot,
            global_step=self.global_step,
            total_steps=self.total_steps,
        )
        if "raw_losses_for_scaling" in losses:
            losses.pop("raw_losses_for_scaling")

        # We'll accumulate losses in self.current_val_losses rather than self.val_losses directly
        # This accumulation will be handled in on_validation_epoch_end
        if not hasattr(self, "current_val_losses"):
            # Initialize dictionary
            self.current_val_losses = {}

        # Process each loss value
        for k, v in losses.items():
            # Skip nested dictionaries like 'raw_losses_for_scaling'
            if isinstance(v, dict):
                continue

            value = v.item() if hasattr(v, "item") else float(v)

            # Add to current_val_losses
            if k not in self.current_val_losses:
                self.current_val_losses[k] = [value]
            else:
                self.current_val_losses[k].append(value)

        # Debug log periodically
        if self.validation_step_ % 20 == 0:
            self.logger_.info(
                f"Validation step {self.validation_step_}, accumulated {len(self.current_val_losses.get('total_loss', []))} validation samples"
            )
            if "ilisi_score" in losses:
                self.logger_.info(f"Validation iLISI score: {losses['ilisi_score']:.4f}")

        # Log metrics
        metrics = {}

        # Use validation_step_ to determine the last batch
        # Calculate total validation steps needed
        val_steps_per_epoch = int(np.ceil(len(self.val_indices_rna) / self.val_batch_size))
        is_last_batch = self.validation_step_ >= val_steps_per_epoch

        self.logger_.info(
            f"validation_step_: {self.validation_step_}, val_steps_per_epoch: {val_steps_per_epoch}"
        )
        self.logger_.info(f"is_last_batch: {is_last_batch}")

        if is_last_batch:
            # Reset validation_step_ counter for the next validation phase
            self.validation_step_ = 0
            to_plot = True
            # Just log a summary of validation at the end of validation
            mean_total_loss = sum(self.current_val_losses.get("total_loss", [0])) / max(
                1, len(self.current_val_losses.get("total_loss", []))
            )
            self.logger_.info(f"\nVALIDATION Step {self.global_step}, Epoch {self.current_epoch}")
            self.logger_.info(f"Validation total loss: {mean_total_loss:.4f}")

            # Note: on_validation_epoch_end() will be called automatically by Lightning
        self.to_print = is_last_batch

        # Note: Don't log validation metrics per batch - they will be logged per epoch in on_validation_epoch_end
        # This prevents duplicate logging of validation losses at multiple steps

        log_step(
            losses,
            metrics=metrics,
            global_step=self.global_step,
            current_epoch=self.current_epoch,
            is_validation=True,
            total_steps=self.total_steps,
            print_to_console=self.to_print,
            log_to_mlflow=False,  # Don't log per batch - only log per epoch in on_validation_epoch_end
        )

        return losses["total_loss"]

    def calculate_metrics_for_data(
        self, rna_latent_adata, prot_latent_adata, prefix="", global_step=None
    ):
        """Calculate metrics for given RNA and protein data."""

        self.logger_.info(f"   Calculating {prefix}metrics...")
        accuracy = matching_accuracy(rna_latent_adata, prot_latent_adata, global_step)
        silhouette_f1 = compute_silhouette_f1(rna_latent_adata, prot_latent_adata)
        combined_latent = anndata.concat(
            [rna_latent_adata, prot_latent_adata],
            join="outer",
            label="modality",
            keys=["RNA", "Protein"],
        )

        # Force recomputation of neighbors with cosine metric for better integration
        # This helps with modality alignment in UMAP visualization
        (combined_latent.obsm.pop("X_pca", None) if "X_pca" in combined_latent.obsm else None)
        (
            combined_latent.obsp.pop("connectivities", None)
            if "connectivities" in combined_latent.obsp
            else None
        )
        (
            combined_latent.obsp.pop("distances", None)
            if "distances" in combined_latent.obsp
            else None
        )
        (combined_latent.uns.pop("neighbors", None) if "neighbors" in combined_latent.uns else None)

        # Calculate with parameters optimized for integration
        sc.pp.neighbors(combined_latent, use_rep="X")

        # pf.plot_end_of_val_epoch_pca_umap_latent_space(
        #     prefix, combined_latent, epoch=self.current_epoch
        # )

        ari_f1 = compute_ari_f1(combined_latent)
        self.logger_.info(f"{prefix}ARI F1 calculated")

        # # Calculate CN iLISI score within each cell type cluster
        # cn_ilisi_metrics = CODEX_RNA_seq.metrics.calculate_cn_ilisi_within_cell_types(
        #     combined_latent, prefix=prefix, plot_flag=self.to_print
        # )

        # Combine all metrics
        # Note: ari_f1 returns a dict, silhouette_f1 returns a float
        ari_f1_value = ari_f1 if isinstance(ari_f1, (int, float)) else ari_f1.get("ari_f1", 0.0)
        metrics = {
            f"{prefix}cell_type_matching_accuracy": accuracy,
            f"{prefix}silhouette_f1_score": silhouette_f1,
            f"{prefix}ari_f1_score": ari_f1_value,
            # f"{prefix}cn_ilisi_score": cn_ilisi_metrics,  # Commented out since cn_ilisi_metrics is not defined
        }
        return metrics

    def on_validation_epoch_end(self):
        """Calculate and store metrics at the end of each validation epoch."""
        self.logger_.info(f"\nProcessing validation epoch {self.current_epoch}...")
        self.rna_vae.module.eval()
        self.protein_vae.module.eval()

        subsample_size = min(
            len(self.val_indices_rna), len(self.val_indices_prot), 1000
        )  # change this s.t. this adjusts with different sample sizes, was set to 3000
        val_indices_rna = np.random.choice(self.val_indices_rna, size=subsample_size, replace=False)
        val_indices_prot = np.random.choice(
            self.val_indices_prot, size=subsample_size, replace=False
        )
        # Keep validation batches separate to avoid being overwritten later
        rna_batch_val = self._get_rna_batch(None, val_indices_rna)
        prot_batch_val = self._get_protein_batch(None, val_indices_prot)

        # Get latent representations
        # Set models to train mode for consistent latent generation (eval mode gives wrong results)
        self.rna_vae.module.train()
        self.protein_vae.module.train()
        with torch.no_grad():
            # RNA batch should already be in proper format from data preparation

            # DEBUG: Log input batch statistics
            self.logger_.info(f"[VALIDATION DEBUG] RNA batch X shape: {rna_batch_val['X'].shape}")
            self.logger_.info(
                f"[VALIDATION DEBUG] RNA batch X mean: {rna_batch_val['X'].mean():.6f}, std: {rna_batch_val['X'].std():.6f}"
            )
            self.logger_.info(
                f"[VALIDATION DEBUG] RNA batch X min: {rna_batch_val['X'].min():.6f}, max: {rna_batch_val['X'].max():.6f}"
            )
            self.logger_.info(
                f"[VALIDATION DEBUG] Protein batch X shape: {prot_batch_val['X'].shape}"
            )
            self.logger_.info(
                f"[VALIDATION DEBUG] Protein batch X mean: {prot_batch_val['X'].mean():.6f}, std: {prot_batch_val['X'].std():.6f}"
            )
            self.logger_.info(
                f"[VALIDATION DEBUG] Protein batch X min: {prot_batch_val['X'].min():.6f}, max: {prot_batch_val['X'].max():.6f}"
            )
            rna_inference_outputs, _, _ = self.rna_vae.module(rna_batch_val)
            rna_latent = rna_inference_outputs["qz"].mean.detach().cpu().numpy()

            # DEBUG: Log RNA latent statistics
            self.logger_.info(f"[VALIDATION DEBUG] RNA latent shape: {rna_latent.shape}")
            self.logger_.info(
                f"[VALIDATION DEBUG] RNA latent mean: {rna_latent.mean():.6f}, std: {rna_latent.std():.6f}"
            )
            self.logger_.info(
                f"[VALIDATION DEBUG] RNA latent min: {rna_latent.min():.6f}, max: {rna_latent.max():.6f}"
            )
            self.logger_.info(
                f"[VALIDATION DEBUG] RNA latent per-dim means: {rna_latent.mean(axis=0)[:5]}"
            )
            self.logger_.info(
                f"[VALIDATION DEBUG] RNA latent per-dim stds: {rna_latent.std(axis=0)[:5]}"
            )
            prot_inference_outputs, _, _ = self.protein_vae.module(prot_batch_val)
            prot_latent = prot_inference_outputs["qz"].mean.detach().cpu().numpy()

            # DEBUG: Log protein latent statistics
            self.logger_.info(f"[VALIDATION DEBUG] Protein latent shape: {prot_latent.shape}")
            self.logger_.info(
                f"[VALIDATION DEBUG] Protein latent mean: {prot_latent.mean():.6f}, std: {prot_latent.std():.6f}"
            )
            self.logger_.info(
                f"[VALIDATION DEBUG] Protein latent min: {prot_latent.min():.6f}, max: {prot_latent.max():.6f}"
            )
            self.logger_.info(
                f"[VALIDATION DEBUG] Protein latent per-dim means: {prot_latent.mean(axis=0)[:5]}"
            )
            self.logger_.info(
                f"[VALIDATION DEBUG] Protein latent per-dim stds: {prot_latent.std(axis=0)[:5]}"
            )

        # Set models back to eval mode for validation
        self.rna_vae.module.eval()
        self.protein_vae.module.eval()

        # Create AnnData objects with ONLY the observations for the selected indices
        rna_latent_adata = AnnData(rna_latent)
        prot_latent_adata = AnnData(prot_latent)

        # Use only the observations corresponding to the selected indices
        rna_latent_adata.obs = self.rna_vae.adata[val_indices_rna].obs.copy()
        prot_latent_adata.obs = self.protein_vae.adata[val_indices_prot].obs.copy()

        # Calculate validation metrics
        val_metrics = self.calculate_metrics_for_data(
            rna_latent_adata,
            prot_latent_adata,
            prefix="val_",
            global_step=self.global_step,
        )
        # now for train
        subsample_size = min(len(self.train_indices_rna), len(self.train_indices_prot), 3000)
        train_indices_rna = np.random.choice(
            self.train_indices_rna, size=subsample_size, replace=False
        )
        train_indices_prot = np.random.choice(
            self.train_indices_prot, size=subsample_size, replace=False
        )
        rna_batch_train = self._get_rna_batch(None, train_indices_rna)
        prot_batch_train = self._get_protein_batch(None, train_indices_prot)

        # Generate TRAINING latent representations (not validation!)
        self.rna_vae.module.train()
        self.protein_vae.module.train()
        with torch.no_grad():
            # RNA batch should already be in proper format from data preparation
            train_rna_inference_outputs, _, _ = self.rna_vae.module(rna_batch_train)
            train_rna_latent = train_rna_inference_outputs["qz"].mean.detach().cpu().numpy()
            train_prot_inference_outputs, _, _ = self.protein_vae.module(prot_batch_train)
            train_prot_latent = train_prot_inference_outputs["qz"].mean.detach().cpu().numpy()
        self.rna_vae.module.eval()
        self.protein_vae.module.eval()

        # Create TRAINING latent AnnData objects
        train_rna_latent_adata = AnnData(train_rna_latent)
        train_prot_latent_adata = AnnData(train_prot_latent)
        train_rna_latent_adata.obs = self.rna_vae.adata[train_indices_rna].obs.copy()
        train_prot_latent_adata.obs = self.protein_vae.adata[train_indices_prot].obs.copy()

        # Calculate training metrics with subsampling
        self.logger_.info("Calculating training metrics...")
        train_metrics = self.calculate_metrics_for_data(
            train_rna_latent_adata,
            train_prot_latent_adata,
            prefix="train_",
            global_step=self.global_step,
        )

        # Combine metrics
        epoch_metrics = {**val_metrics, **train_metrics}

        # Store in history
        self.metrics_history.append(epoch_metrics)
        self.logger_.info(f"Metrics: {epoch_metrics}")

        # Now process accumulated validation losses from validation_step
        # These are the epoch-level means of the batch-level losses
        if hasattr(self, "current_val_losses") and self.current_val_losses:
            validation_sample_count = len(self.current_val_losses.get("total_loss", []))
            self.logger_.info(f"Processing {validation_sample_count} validation samples")

            # Calculate mean for each loss type
            epoch_val_losses = {}
            for loss_type, values in self.current_val_losses.items():
                if values:
                    mean_value = sum(values) / len(values)
                    epoch_val_losses[loss_type] = mean_value

                    # Map loss types to their corresponding lists
                    loss_map = {
                        "total_loss": self.val_total_loss,
                        "rna_loss": self.val_rna_loss,
                        "protein_loss": self.val_protein_loss,
                        "matching_loss": self.val_matching_loss,
                        "contrastive_loss": self.val_contrastive_loss,
                        "cell_type_clustering_loss": self.val_cell_type_clustering_loss,
                        "similarity_loss": self.val_similarity_loss,
                        "raw_similarity_loss": self.val_raw_similarity_loss,
                        "latent_distances": self.val_latent_distance,
                        "cross_modal_cell_type_loss": self.val_cross_modal_cell_type_loss,
                        "cn_distribution_separation_loss": self.val_cn_distribution_separation_loss,
                    }

                    if loss_type in loss_map:
                        loss_map[loss_type].append(mean_value)
                    elif loss_type == "ilisi_score":
                        if not hasattr(self, "val_ilisi_score"):
                            self.val_ilisi_score = []
                        self.val_ilisi_score.append(mean_value)
            # Reset current_val_losses for next epoch
            self.current_val_losses = {}

            self.logger_.info(f"Validation losses stored (epoch {self.current_epoch})")
            self.logger_.info(f"Current validation history length: {len(self.val_total_loss)}")
        else:
            self.logger_.info("WARNING: No validation losses to process for this epoch!")

        # Debug print validation losses after processing
        # Reset validation step counter
        self.validation_step_ = 0

        # Log validation metrics immediately to MLflow (using epoch as x-axis)
        if mlflow.active_run():
            validation_metrics_to_log = {}

            # Add latest validation losses (filter out zero values)
            for attr_name in dir(self):
                if attr_name.startswith("val_") and hasattr(self, attr_name):
                    loss_list = getattr(self, attr_name)
                    if isinstance(loss_list, list) and len(loss_list) > 0:
                        latest_value = loss_list[-1]
                        # Only log non-zero losses
                        if not (isinstance(latest_value, (int, float)) and latest_value == 0.0):
                            # Convert 'val_' to 'val/' - use epoch as x-axis for validation
                            metric_name = attr_name.replace("val_", "val/", 1)
                            validation_metrics_to_log[metric_name] = latest_value

            # Add validation epoch metrics (silhouette, ARI, etc.) - filter out zero values
            if hasattr(self, "metrics_history") and len(self.metrics_history) > 0:
                latest_metrics = self.metrics_history[-1]
                for key, value in latest_metrics.items():
                    if key.startswith("val_"):
                        # Handle dict values (e.g., ari_f1_score)
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                if not (isinstance(sub_value, (int, float)) and sub_value == 0.0):
                                    validation_metrics_to_log[f"val/{key[4:]}_{sub_key}"] = (
                                        sub_value
                                    )
                        # Only log non-zero scalar metrics
                        elif not (isinstance(value, (int, float)) and value == 0.0):
                            validation_metrics_to_log[f"val/{key[4:]}"] = value

        # Increment validation epoch counter first
        self.val_epoch_counter += 1

        # Log all non-zero validation metrics using training epoch as step
        # The val/ prefix in metric names already distinguishes them from training metrics
        # Skip logging on final epoch to prevent conflicts with final logging
        if validation_metrics_to_log and self.current_epoch < self.max_epochs - 1:
            mlflow.log_metrics(validation_metrics_to_log, step=self.current_epoch)
            self.logger_.info(
                f"Logged {len(validation_metrics_to_log)} non-zero validation metrics to MLflow at epoch {self.current_epoch} (val epoch {self.val_epoch_counter})"
            )
        elif validation_metrics_to_log:
            self.logger_.info(
                f"Skipping validation MLflow logging at final epoch {self.current_epoch} to prevent duplicates with final logging"
            )

        # Determine if we should generate validation plots (similar to training plot logic)
        should_plot_validation = False
        if self.plot_x_times > 0:
            # Use similar logic to training plots: plot at intervals to achieve approximately plot_x_times plots
            should_plot_validation = (
                (
                    self.val_plot_interval > 0
                    and self.val_epoch_counter % self.val_plot_interval == 0
                )
                or (self.val_epoch_counter == 1)  # Always plot first validation epoch
                or (
                    self.val_epoch_counter >= self.total_val_epochs
                )  # Always plot last validation epoch
            )

        # Generate validation plots only when scheduled
        if should_plot_validation:
            self.logger_.info(
                f"Generating validation plots for epoch {self.current_epoch} (val epoch {self.val_epoch_counter}/{self.total_val_epochs})..."
            )
        else:
            next_plot_epoch = (
                ((self.val_epoch_counter // self.val_plot_interval) + 1) * self.val_plot_interval
                if self.val_plot_interval > 0
                else -1
            )
            self.logger_.info(
                f"Skipping validation plots for epoch {self.current_epoch} (val epoch {self.val_epoch_counter}/{self.total_val_epochs}, next plot at val epoch {next_plot_epoch})"
            )

        original_mode = getattr(self, "mode", "training")
        self.mode = "validation"

        # Only generate plots when scheduled
        if should_plot_validation:
            # IMPORTANT: match training-time stochastic layers behavior (dropout)
            # Use train() during latent generation for plots as in training plots
            self.rna_vae.module.train()
            self.protein_vae.module.train()
            with torch.no_grad():
                rna_inference_outputs_tuple = self.rna_vae.module(rna_batch_val)
                protein_inference_outputs_tuple = self.protein_vae.module(prot_batch_val)
            # Use the same plotting logic as training, but with validation data
            self.calculate_losses(
                rna_batch=rna_batch_val,
                protein_batch=prot_batch_val,
                rna_vae=self.rna_vae,
                protein_vae=self.protein_vae,
                rna_inference_outputs_tuple=rna_inference_outputs_tuple,
                protein_inference_outputs_tuple=protein_inference_outputs_tuple,
                device=self.device,
                similarity_weight=self.similarity_weight,
                similarity_active=self.similarity_active,
                contrastive_weight=self.contrastive_weight,
                matching_weight=self.matching_weight,
                cell_type_clustering_weight=self.cell_type_clustering_weight,
                cross_modal_cell_type_weight=self.cross_modal_cell_type_weight,
                rna_recon_weight=self.rna_recon_weight,
                prot_recon_weight=self.prot_recon_weight,
                check_ilisi=False,  # Skip iLISI calculation for plotting
                to_plot=True,  # Force plotting for validation
                global_step=self.global_step,
                total_steps=self.total_steps,
            )
            self.logger_.info(
                f"Validation plots generated successfully for epoch {self.current_epoch}"
            )

        # Restore original mode
        self.rna_vae.module.eval()
        self.protein_vae.module.eval()
        self.mode = original_mode

        self.logger_.info(f"Validation epoch {self.current_epoch} completed successfully!")

    def on_train_end_custom(self, plot_flag=True):
        """Called when training ends."""
        self.rna_vae.module.eval()
        self.protein_vae.module.eval()

        self.mode = "validation"
        if self.on_train_end_custom_called:
            self.logger_.info("on_train_end_custom already called, skipping")
            return

        self.logger_.success("\nTraining completed!")

        # Get final latent representations
        with torch.no_grad():
            # Process RNA data in batches to avoid memory overflow
            rna_latent_list = []
            # Adjust batch size if needed
            for i in range(0, self.rna_vae.adata.shape[0], self.batch_size):
                batch_indices = np.arange(i, min(i + self.batch_size, self.rna_vae.adata.shape[0]))
                rna_batch = self._get_rna_batch(None, batch_indices)

                # RNA batch should already be in proper format from data preparation
                rna_inference_outputs, _, _ = self.rna_vae.module(rna_batch)
                rna_latent_list.append(rna_inference_outputs["qz"].mean.detach().cpu().numpy())
            rna_latent = np.concatenate(rna_latent_list, axis=0)

            # Process protein data in batches to avoid memory overflow
            prot_latent_list = []
            for i in range(0, self.protein_vae.adata.shape[0], self.batch_size):
                batch_indices = np.arange(
                    i, min(i + self.batch_size, self.protein_vae.adata.shape[0])
                )
                protein_batch = self._get_protein_batch(None, batch_indices)
                prot_inference_outputs, _, _ = self.protein_vae.module(protein_batch)
                prot_latent_list.append(prot_inference_outputs["qz"].mean.detach().cpu().numpy())
            prot_latent = np.concatenate(prot_latent_list, axis=0)

            rna_batch = self._get_rna_batch(
                None,
                np.random.choice(
                    range(self.rna_vae.adata.shape[0]),
                    size=self.batch_size,
                    replace=False if self.rna_vae.adata.shape[0] > self.batch_size else True,
                ),
            )
            protein_batch = self._get_protein_batch(
                None,
                np.random.choice(
                    range(self.protein_vae.adata.shape[0]),
                    size=self.batch_size,
                    replace=False if self.protein_vae.adata.shape[0] > self.batch_size else True,
                ),
            )
        # RNA batch should already be in proper format from data preparation
        with torch.no_grad():
            rna_inference_outputs_tuple = self.rna_vae.module(rna_batch)
            protein_inference_outputs_tuple = self.protein_vae.module(protein_batch)
        losses = self.calculate_losses(  # for the plots
            rna_batch=rna_batch,
            protein_batch=protein_batch,
            rna_vae=self.rna_vae,
            protein_vae=self.protein_vae,
            rna_inference_outputs_tuple=rna_inference_outputs_tuple,
            protein_inference_outputs_tuple=protein_inference_outputs_tuple,
            device=self.device,
            similarity_weight=self.similarity_weight,
            similarity_active=self.similarity_active,
            contrastive_weight=self.contrastive_weight,
            matching_weight=self.matching_weight,
            cell_type_clustering_weight=self.cell_type_clustering_weight,
            cross_modal_cell_type_weight=self.cross_modal_cell_type_weight,
            rna_recon_weight=self.rna_recon_weight,
            prot_recon_weight=self.prot_recon_weight,
            check_ilisi=True,
            to_plot=True,
            global_step=self.global_step,
            total_steps=self.total_steps,
        )

        # Format losses for nice display

        # Convert tensor values to float and round
        formatted_losses = {
            k: round(float(v.item() if torch.is_tensor(v) else v), 4)
            for k, v in losses.items()
            if k
            not in [
                "num_cells",
                "raw_losses_for_scaling",
            ]  # Exclude non-loss metrics and nested dicts
        }

        # Calculate total loss for percentage
        total_loss = formatted_losses["total_loss"]

        # Create table data
        table_data = []
        for loss_name, value in formatted_losses.items():
            if loss_name == "total_loss":
                table_data.append([loss_name.replace("_", " ").title(), f"{value:,.2f}"])
            else:
                percentage = (value / total_loss * 100) if total_loss != 0 else 0
                table_data.append(
                    [loss_name.replace("_", " ").title(), f"{value:,.2f} ({percentage:.1f}%)"]
                )

        # Add iLISI score separately
        if "ilisi_score" in formatted_losses:
            table_data.append(["iLISI Score", f"{formatted_losses['ilisi_score']:.4f}"])

        # Print formatted table
        self.logger_.info("\nFinal Training Losses:")
        self.logger_.info(
            "\n" + tabulate(table_data, headers=["Loss Type", "Value"], tablefmt="fancy_grid")
        )

        # Add outlier detection summary
        if self.outlier_detection_enabled:
            outlier_table = [
                ["RNA Outliers Detected", f"{self.outlier_count_rna}"],
                ["Protein Outliers Detected", f"{self.outlier_count_protein}"],
                ["Total Training Steps", f"{self.global_step}"],
                [
                    "RNA Outlier Rate",
                    f"{(self.outlier_count_rna/max(1, self.global_step)*100):.2f}%",
                ],
                [
                    "Protein Outlier Rate",
                    f"{(self.outlier_count_protein/max(1, self.global_step)*100):.2f}%",
                ],
            ]
            self.logger_.info("\nOutlier Detection Summary:")
            self.logger_.info(
                "\n" + tabulate(outlier_table, headers=["Metric", "Value"], tablefmt="fancy_grid")
            )

        # Store in adata
        self.rna_vae.adata.obsm["X_scVI"] = rna_latent
        self.protein_vae.adata.obsm["X_scVI"] = prot_latent
        latent_distances = cdist(rna_latent, prot_latent, metric="euclidean")
        rna_best_prot_match = np.argmin(latent_distances, axis=1)

        self.rna_vae.adata.obs["CN"] = self.protein_vae.adata.obs["CN"].values[rna_best_prot_match]

        # Save the final checkpoints
        self.save_checkpoints()

        # Plot metrics over time
        if plot_flag and hasattr(self, "metrics_history"):
            pf.plot_training_metrics_history(
                self.metrics_history,
            )
        history_ = self.get_history()

        # Save the training history to a JSON file for reference
        history_path = f"{self.checkpoints_dir}/training_history.json"
        json_history = {}
        for key, value in history_.items():
            if isinstance(value, np.ndarray):
                json_history[key] = value.tolist()
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                json_history[key] = [v.tolist() for v in value]
            else:
                json_history[key] = value

        with open(history_path, "w") as f:
            json.dump(json_history, f, indent=2)
        self.logger_.info(f"Training history saved to {history_path}")

        # Log history JSON as artifact
        mlflow.log_artifact(history_path)

        # Plot the losses
        pf.plot_train_val_normalized_losses(history_, plot_flag=plot_flag)
        self.logger_.info("Train/validation normalized losses plotted")

        # Find best metrics
        if hasattr(self, "metrics_history") and len(self.metrics_history) > 0:
            best_metrics = {}
            for metric in self.metrics_history[0].keys():
                values = [epoch[metric] for epoch in self.metrics_history]
                if "loss" in metric or "error" in metric:
                    best_metrics[metric] = min(values)
                else:
                    best_metrics[metric] = max(values)

            # Store best metrics for centralized logging
            self.final_best_metrics = best_metrics
            self.logger_.info("Best metrics will be logged to MLflow")

        # Explicitly log the main run log file
        self.logger_.info("Attempting to log the main run log file to MLflow...")
        log_file_path = self.log_file_path  # Use the stored log_file_path

        if log_file_path.exists():
            # By this point, most logging operations should have completed.
            # Loguru typically flushes its buffers upon script completion or when buffers are full.
            # Logging the artifact here ensures it's captured after the bulk of log messages.
            mlflow.log_artifact(str(log_file_path), artifact_path="logs")
            self.logger_.info(
                f"Successfully logged '{log_file_path}' to MLflow artifacts under 'logs/'."
            )

        else:
            self.logger_.warning(
                f"Log file '{log_file_path}' not found, cannot log to MLflow. Check CWD and log configuration."
            )

        # Log final metrics to MLflow ONLY if not already logged
        if not hasattr(self, "final_metrics_logged"):
            self.log_to_mlflow_centralized("final")
            self.final_metrics_logged = True
            self.logger_.info("Final metrics logged to MLflow")
        else:
            self.logger_.info("Final metrics already logged, skipping duplicate logging")

        # Mark that this function has been called
        self.on_train_end_custom_called = True

    def save_checkpoints(self, checkpoints_path=None):
        """Save the model checkpoints including AnnData objects with latent representations.
        Feed RNA latents into protein decoder and vice versa for downstream QC plots."""
        self.logger_.info(f"\nSaving checkpoints at epoch {self.current_epoch}...")

        # Create checkpoints directory in MLflow artifacts folder
        if checkpoints_path is None:
            # Get MLflow artifact directory and create checkpoints subfolder
            artifact_uri = mlflow.get_artifact_uri()
            # Remove 'file://' prefix if present
            if artifact_uri.startswith("file://"):
                artifact_path = artifact_uri[7:]  # Remove 'file://'
            else:
                artifact_path = artifact_uri
            checkpoints_path = f"{artifact_path}/checkpoints/epoch_{self.current_epoch:04d}"
        os.makedirs(checkpoints_path, exist_ok=True)

        adata_rna_save = self.rna_vae.adata.copy()
        protein_adata_save = self.protein_vae.adata.copy()

        # --- Add inferred latent space to obsm before saving ---
        rna_latent = get_latent_embedding(
            self.rna_vae, adata_rna_save, batch_size=self.batch_size, device=str(self.device)
        )
        protein_latent = get_latent_embedding(
            self.protein_vae,
            protein_adata_save,
            batch_size=self.batch_size,
            device=str(self.device),
        )
        adata_rna_save.obsm["X_scVI"] = rna_latent
        protein_adata_save.obsm["X_scVI"] = protein_latent
        # ------------------------------------------------------
        # Counterfactual decoding: RNA latent decoded by protein VAE, and vice versa
        with torch.no_grad():
            # MEMORY OPTIMIZATION: Subsample data BEFORE counterfactual generation to prevent OOM
            max_cells_for_counterfactual = 2000  # Further reduced from 5000 to prevent OOM

            # Check available memory and skip counterfactual generation if too low
            memory_percent = psutil.virtual_memory().percent
            self.logger_.info(f"Current memory usage: {memory_percent:.1f}%")

            if memory_percent > 85:  # If memory usage is above 85%, skip counterfactual generation
                self.logger_.warning(
                    f"Memory usage too high ({memory_percent:.1f}%), skipping counterfactual generation to prevent OOM"
                )
                # Save basic checkpoints without counterfactual analysis
                clean_uns_for_h5ad(adata_rna_save)
                clean_uns_for_h5ad(protein_adata_save)
                sc.write(f"{checkpoints_path}/adata_rna.h5ad", adata_rna_save)
                sc.write(f"{checkpoints_path}/adata_prot.h5ad", protein_adata_save)
                torch.save(self.rna_vae.module.state_dict(), f"{checkpoints_path}/rna_vae_model.pt")
                torch.save(
                    self.protein_vae.module.state_dict(), f"{checkpoints_path}/protein_vae_model.pt"
                )
                self.logger_.info(
                    f"✓ Basic checkpoint saved to {checkpoints_path} (counterfactual analysis skipped)"
                )
                return

            # Subsample RNA data if needed
            if adata_rna_save.shape[0] > max_cells_for_counterfactual:
                self.logger_.info(
                    f"Subsampling RNA data from {adata_rna_save.shape[0]} to {max_cells_for_counterfactual} cells for counterfactual generation"
                )
                rna_indices = np.random.choice(
                    adata_rna_save.shape[0], max_cells_for_counterfactual, replace=False
                )
                adata_rna_for_counterfactual = adata_rna_save[rna_indices].copy()
                rna_latent_for_counterfactual = rna_latent[rna_indices]
            else:
                adata_rna_for_counterfactual = adata_rna_save
                rna_latent_for_counterfactual = rna_latent

            # Subsample protein data if needed
            if protein_adata_save.shape[0] > max_cells_for_counterfactual:
                self.logger_.info(
                    f"Subsampling protein data from {protein_adata_save.shape[0]} to {max_cells_for_counterfactual} cells for counterfactual generation"
                )
                protein_indices = np.random.choice(
                    protein_adata_save.shape[0], max_cells_for_counterfactual, replace=False
                )
                protein_adata_for_counterfactual = protein_adata_save[protein_indices].copy()
                protein_latent_for_counterfactual = protein_latent[protein_indices]
            else:
                protein_adata_for_counterfactual = protein_adata_save
                protein_latent_for_counterfactual = protein_latent

            # Create counterfactual data on subsampled datasets
            (
                counterfactual_adata_rna,
                counterfactual_protein_adata,
                same_modal_adata_rna,
                same_modal_adata_protein,
            ) = create_counterfactual_adata(
                self,
                adata_rna_for_counterfactual,
                protein_adata_for_counterfactual,
                rna_latent_for_counterfactual,
                protein_latent_for_counterfactual,
                plot_flag=self.plot_x_times > 0,
            )

            # Clear temporary variables to free memory
            del adata_rna_for_counterfactual, protein_adata_for_counterfactual
            del rna_latent_for_counterfactual, protein_latent_for_counterfactual
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Create separate plots for train and validation sets (only if plot_x_times > 0)
            plot_flag = self.plot_x_times > 0
            if not plot_flag:
                self.logger_.info("Skipping counterfactual plots (plot_x_times=0)")
            else:
                self.logger_.info("Creating counterfactual plots for train and validation sets...")

                # --- TRAIN SET COUNTERFACTUAL PLOTS ---
                self.logger_.info("Generating train set counterfactual plots...")

                # Simply subsample training data separately as needed
                max_train_cells = 1000  # Reduced from 2000 to prevent OOM

                # Subsample counterfactual RNA data for training
                if counterfactual_adata_rna.shape[0] > max_train_cells:
                    train_rna_indices = np.random.choice(
                        counterfactual_adata_rna.shape[0], max_train_cells, replace=False
                    )
                    counterfactual_adata_rna_train = counterfactual_adata_rna[
                        train_rna_indices
                    ].copy()
                else:
                    counterfactual_adata_rna_train = counterfactual_adata_rna.copy()

                # Subsample counterfactual protein data for training
                if counterfactual_protein_adata.shape[0] > max_train_cells:
                    train_protein_indices = np.random.choice(
                        counterfactual_protein_adata.shape[0], max_train_cells, replace=False
                    )
                    counterfactual_protein_adata_train = counterfactual_protein_adata[
                        train_protein_indices
                    ].copy()
                else:
                    counterfactual_protein_adata_train = counterfactual_protein_adata.copy()

                # Subsample same-modal reconstructions for training (ground truth)
                if same_modal_adata_rna.shape[0] > max_train_cells:
                    same_modal_rna_indices = np.random.choice(
                        same_modal_adata_rna.shape[0], max_train_cells, replace=False
                    )
                    same_modal_adata_rna_train = same_modal_adata_rna[same_modal_rna_indices].copy()
                else:
                    same_modal_adata_rna_train = same_modal_adata_rna.copy()

                if same_modal_adata_protein.shape[0] > max_train_cells:
                    same_modal_protein_indices = np.random.choice(
                        same_modal_adata_protein.shape[0], max_train_cells, replace=False
                    )
                    same_modal_adata_protein_train = same_modal_adata_protein[
                        same_modal_protein_indices
                    ].copy()
                else:
                    same_modal_adata_protein_train = same_modal_adata_protein.copy()

                # Calculate comprehensive training metrics for counterfactual data
                # Compare counterfactual against same-modal reconstructions (not original data)
                self.logger_.info("Calculating training reconstruction metrics...")
                train_metrics = self.calculate_counterfactual_validation_metrics(
                    same_modal_adata_rna_train,
                    same_modal_adata_protein_train,
                    counterfactual_adata_rna_train,
                    counterfactual_protein_adata_train,
                )

                # Extract metrics for plot titles (rounded to 4 decimal places)
                train_rna_sim = round(train_metrics.get("rna_recon_similarity_loss", 0.0), 4)
                train_protein_sim = round(
                    train_metrics.get("protein_recon_similarity_loss", 0.0), 4
                )
                train_rna_ilisi = round(train_metrics.get("rna_ilisi_score", 0.0), 4)
                train_protein_ilisi = round(train_metrics.get("protein_ilisi_score", 0.0), 4)

                # Plot train set: same-modal reconstruction (ground truth) vs counterfactual
                pf.plot_counterfactual_comparison_with_suffix(
                    same_modal_adata_rna_train,
                    same_modal_adata_protein_train,
                    counterfactual_adata_rna_train,
                    counterfactual_protein_adata_train,
                    epoch=self.current_epoch,
                    suffix="train",
                    rna_similarity_score=train_rna_sim,
                    protein_similarity_score=train_protein_sim,
                    rna_ilisi_score=train_rna_ilisi,
                    protein_ilisi_score=train_protein_ilisi,
                    plot_flag=plot_flag,
                )

                # --- ENCODED LATENT SPACE PLOTS FOR TRAINING SET ---
                self.logger_.info("Generating encoded latent space plots for training set...")

                # Get encoded latent representations for training set
                self.rna_vae.module.train()
                self.protein_vae.module.train()
                with torch.no_grad():
                    # Create batches for RNA and protein training data
                    rna_train_batch = self._get_rna_batch(None, self.train_indices_rna)
                    protein_train_batch = self._get_protein_batch(None, self.train_indices_prot)

                    # Get latent representations using model forward pass
                    rna_train_inference_outputs, _, _ = self.rna_vae.module(rna_train_batch)
                    rna_train_latent = rna_train_inference_outputs["qz"].mean.detach().cpu().numpy()

                    protein_train_inference_outputs, _, _ = self.protein_vae.module(
                        protein_train_batch
                    )
                    protein_train_latent = (
                        protein_train_inference_outputs["qz"].mean.detach().cpu().numpy()
                    )
                self.rna_vae.module.eval()
                self.protein_vae.module.eval()

                # Create AnnData objects for the training indices used
                adata_rna_train_indices = self.rna_vae.adata[self.train_indices_rna].copy()
                adata_prot_train_indices = self.protein_vae.adata[self.train_indices_prot].copy()

                # Plot encoded latent space comparison for training set
                pf.plot_encoded_latent_space_comparison(
                    adata_rna_train_indices,
                    adata_prot_train_indices,
                    rna_train_latent,
                    protein_train_latent,
                    index_rna=range(len(adata_rna_train_indices)),
                    index_prot=range(len(adata_prot_train_indices)),
                    global_step=self.global_step,
                    use_subsample=True,
                    n_subsample=2000,
                    suffix="train",
                    plot_flag=plot_flag,
                )

                # Log training metrics to MLflow with epoch as step
                mlflow_train_metrics = {}
                for key, value in train_metrics.items():
                    mlflow_train_metrics[f"train/{key}"] = value

                # --- VALIDATION SET COUNTERFACTUAL PLOTS ---
                self.logger_.info("Generating validation set counterfactual plots...")

                # Simply subsample validation data separately as needed
                max_val_cells = 500  # Reduced from 1000 to prevent OOM

                # Subsample counterfactual RNA data for validation
                if counterfactual_adata_rna.shape[0] > max_val_cells:
                    val_rna_indices = np.random.choice(
                        counterfactual_adata_rna.shape[0], max_val_cells, replace=False
                    )
                    counterfactual_adata_rna_val = counterfactual_adata_rna[val_rna_indices].copy()
                else:
                    counterfactual_adata_rna_val = counterfactual_adata_rna.copy()

                # Subsample counterfactual protein data for validation
                if counterfactual_protein_adata.shape[0] > max_val_cells:
                    val_protein_indices = np.random.choice(
                        counterfactual_protein_adata.shape[0], max_val_cells, replace=False
                    )
                    counterfactual_protein_adata_val = counterfactual_protein_adata[
                        val_protein_indices
                    ].copy()
                else:
                    counterfactual_protein_adata_val = counterfactual_protein_adata.copy()

                # Subsample same-modal reconstructions for validation (ground truth)
                if same_modal_adata_rna.shape[0] > max_val_cells:
                    same_modal_rna_val_indices = np.random.choice(
                        same_modal_adata_rna.shape[0], max_val_cells, replace=False
                    )
                    same_modal_adata_rna_val = same_modal_adata_rna[
                        same_modal_rna_val_indices
                    ].copy()
                else:
                    same_modal_adata_rna_val = same_modal_adata_rna.copy()

                if same_modal_adata_protein.shape[0] > max_val_cells:
                    same_modal_protein_val_indices = np.random.choice(
                        same_modal_adata_protein.shape[0], max_val_cells, replace=False
                    )
                    same_modal_adata_protein_val = same_modal_adata_protein[
                        same_modal_protein_val_indices
                    ].copy()
                else:
                    same_modal_adata_protein_val = same_modal_adata_protein.copy()

                # Calculate comprehensive validation metrics for counterfactual data
                # Compare counterfactual against same-modal reconstructions (not original data)
                self.logger_.info("Calculating validation reconstruction metrics...")
                validation_metrics = self.calculate_counterfactual_validation_metrics(
                    same_modal_adata_rna_val,
                    same_modal_adata_protein_val,
                    counterfactual_adata_rna_val,
                    counterfactual_protein_adata_val,
                )

                # Extract metrics for plot titles (rounded to 4 decimal places)
                val_rna_sim = round(validation_metrics.get("rna_recon_similarity_loss", 0.0), 4)
                val_protein_sim = round(
                    validation_metrics.get("protein_recon_similarity_loss", 0.0), 4
                )
                val_rna_ilisi = round(validation_metrics.get("rna_ilisi_score", 0.0), 4)
                val_protein_ilisi = round(validation_metrics.get("protein_ilisi_score", 0.0), 4)

                # Plot validation set: same-modal reconstruction (ground truth) vs counterfactual
                pf.plot_counterfactual_comparison_with_suffix(
                    same_modal_adata_rna_val,
                    same_modal_adata_protein_val,
                    counterfactual_adata_rna_val,
                    counterfactual_protein_adata_val,
                    epoch=self.current_epoch,
                    suffix="val",
                    rna_similarity_score=val_rna_sim,
                    protein_similarity_score=val_protein_sim,
                    rna_ilisi_score=val_rna_ilisi,
                    protein_ilisi_score=val_protein_ilisi,
                    plot_flag=plot_flag,
                )

                # # --- ENCODED LATENT SPACE PLOTS ---
                self.logger_.info("Generating encoded latent space plots for validation set...")

                # Get encoded latent representations for validation set
                self.rna_vae.module.train()
                self.protein_vae.module.train()
                with torch.no_grad():
                    # Create batches for RNA and protein validation data
                    rna_val_batch = self._get_rna_batch(None, self.val_indices_rna)
                    protein_val_batch = self._get_protein_batch(None, self.val_indices_prot)

                    # Get latent representations using model forward pass
                    rna_val_inference_outputs, _, _ = self.rna_vae.module(rna_val_batch)
                    rna_val_latent = rna_val_inference_outputs["qz"].mean.detach().cpu().numpy()

                    protein_val_inference_outputs, _, _ = self.protein_vae.module(protein_val_batch)
                    protein_val_latent = (
                        protein_val_inference_outputs["qz"].mean.detach().cpu().numpy()
                    )
                self.rna_vae.module.eval()
                self.protein_vae.module.eval()

                # Create AnnData objects for the validation indices used
                adata_rna_val_indices = self.rna_vae.adata[self.val_indices_rna].copy()
                adata_prot_val_indices = self.protein_vae.adata[self.val_indices_prot].copy()

                # Plot encoded latent space comparison for validation set
                pf.plot_encoded_latent_space_comparison(
                    adata_rna_val_indices,
                    adata_prot_val_indices,
                    rna_val_latent,
                    protein_val_latent,
                    index_rna=range(len(adata_rna_val_indices)),
                    index_prot=range(len(adata_prot_val_indices)),
                    global_step=self.global_step,
                    use_subsample=True,
                    n_subsample=2000,
                    suffix="val",
                    plot_flag=plot_flag,
                )

                # Log training metrics to MLflow with epoch as step
                # Note: Validation metrics are already logged in on_validation_epoch_end() to avoid duplicates
                mlflow_train_metrics_only = {}
                for key, value in train_metrics.items():
                    mlflow_train_metrics_only[f"train/{key}"] = value

                # Log only training reconstruction metrics to avoid duplication with regular validation logging
                # Skip if this is the final epoch to prevent duplicate logging with on_train_end_custom
                if self.current_epoch < self.max_epochs - 1:
                    self.log_to_mlflow_centralized("epoch", extra_metrics=mlflow_train_metrics_only)
                    self.logger_.info(
                        f"Logged {len(train_metrics)} train reconstruction metrics to MLflow at epoch {self.current_epoch}"
                        f" (validation metrics logged separately in on_validation_epoch_end)"
                    )
                else:
                    self.logger_.info(
                        f"Skipping checkpoint MLflow logging at final epoch {self.current_epoch} to prevent duplicates"
                    )
        self.logger_.info(f"Saving checkpoints to {checkpoints_path}")
        clean_uns_for_h5ad(adata_rna_save)
        clean_uns_for_h5ad(protein_adata_save)
        sc.write(f"{checkpoints_path}/adata_rna.h5ad", adata_rna_save)
        sc.write(f"{checkpoints_path}/adata_prot.h5ad", protein_adata_save)
        torch.save(self.rna_vae.module.state_dict(), f"{checkpoints_path}/rna_vae_model.pt")
        torch.save(
            self.protein_vae.module.state_dict(),
            f"{checkpoints_path}/protein_vae_model.pt",
        )
        training_state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "optimizer_state": self.optimizers().state_dict(),
            "similarity_weight": self.similarity_weight,
            "similarity_dynamic": self.similarity_dynamic,
            "similarity_active": self.similarity_active,
            "similarity_loss_history": self.similarity_loss_history,
            "similarity_loss_steady_counter": self.similarity_loss_steady_counter,
            "last_ilisi_score": self.last_ilisi_score,
            # loss_scales removed - only use cached scales, not checkpoint scales
        }

        torch.save(training_state, f"{checkpoints_path}/training_state.pt")

        self.logger_.info(f"✓ Checkpoint saved directly to MLflow artifacts at {checkpoints_path}")

    def _get_protein_batch(self, batch, indices):
        protein_data = self.protein_vae.adata[indices]
        X = protein_data.X
        if issparse(X):
            X = X.toarray()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        batch_indices = (
            torch.tensor(protein_data.obs["batch"].cat.codes.values, dtype=torch.long)
            .to(self.device)
            .unsqueeze(1)
        )  # Make 2D (N, 1) for scVI compatibility
        archetype_vec = torch.tensor(
            protein_data.obsm["archetype_vec"].values, dtype=torch.float32
        ).to(self.device)

        # Add library size information for proper reconstruction loss calculation
        library_size = (
            torch.tensor(protein_data.obs["_scvi_library_size"].values, dtype=torch.float32)
            .to(self.device)
            .unsqueeze(1)
        )  # Make it (N, 1) shape

        protein_batch = {
            "X": X,
            "batch": batch_indices,
            "labels": indices,
            "archetype_vec": archetype_vec,
            "library": library_size,
        }

        return protein_batch

    def _get_rna_batch(self, batch, indices):
        rna_data = self.rna_vae.adata[indices]
        X = rna_data.X
        if issparse(X):
            X = X.toarray()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        batch_indices = (
            torch.tensor(rna_data.obs["batch"].cat.codes.values, dtype=torch.long)
            .to(self.device)
            .unsqueeze(1)
        )  # Make 2D (N, 1) for scVI compatibility
        archetype_vec = torch.tensor(rna_data.obsm["archetype_vec"].values, dtype=torch.float32).to(
            self.device
        )

        # Add library size information for proper reconstruction loss calculation
        library_size = (
            torch.tensor(rna_data.obs["_scvi_library_size"].values, dtype=torch.float32)
            .to(self.device)
            .unsqueeze(1)
        )  # Make it (N, 1) shape

        rna_batch = {
            "X": X,
            "batch": batch_indices,
            "labels": indices,
            "archetype_vec": archetype_vec,
            "library": library_size,
        }

        return rna_batch

    def get_history(self):
        """Return the training history including similarity losses with proper epoch alignment"""
        steps_per_epoch = int(np.ceil(len(self.train_indices_rna) / self.batch_size))
        if len(self.val_total_loss) > 0:
            self.logger_.info(f"Found {len(self.val_total_loss)} validation loss entries")
        else:
            self.logger_.info("No validation losses in val_total_loss array")
            if (
                hasattr(self, "current_val_losses")
                and self.current_val_losses
                and len(self.current_val_losses) > 0
            ):
                self.logger_.info(
                    f"  Processing accumulated validation data from current_val_losses"
                )

        # Group losses by epoch to get mean values
        def get_epoch_means(loss_list):
            """Calculate epoch means from a list of loss values"""
            if not loss_list:
                return []
            loss_array = np.array(loss_list)
            loss_array = loss_array[~np.isinf(loss_array) & ~np.isnan(loss_array)]

            if len(loss_array) == 0:
                return []

            num_epochs = max(1, len(loss_array) // steps_per_epoch)
            epoch_means = []

            for i in range(num_epochs):
                start_idx = i * steps_per_epoch
                end_idx = min((i + 1) * steps_per_epoch, len(loss_array))
                if start_idx < end_idx:  # Only compute if we have data
                    epoch_mean = np.mean(loss_array[start_idx:end_idx])
                    epoch_means.append(epoch_mean)
            return epoch_means

        # Calculate validation epochs - default to every 2 epochs
        # Try to get check_val_every_n_epoch from trainer, or use default value of 2
        val_epochs = [i * self.check_val_every_n_epoch for i in range(len(self.val_total_loss))]

        # If we don't have val_epochs but we have validation losses, create them
        if not val_epochs and len(self.val_total_loss) > 0:
            val_epochs = list(range(len(self.val_total_loss)))

        # Create history dictionary with epoch means
        history_ = {
            "train_similarity_loss": get_epoch_means(self.similarity_losses),
            "train_raw_similarity_loss": get_epoch_means(self.raw_similarity_losses),
            "train_total_loss": get_epoch_means(self.train_losses),
            "train_rna_loss": get_epoch_means(self.train_rna_losses),
            "train_protein_loss": get_epoch_means(self.train_protein_losses),
            "train_matching_loss": get_epoch_means(self.train_matching_losses),
            "train_contrastive_loss": get_epoch_means(self.train_contrastive_losses),
            "train_cn_separation_loss": get_epoch_means(self.train_cn_separation_losses),
            "train_cell_type_clustering_loss": get_epoch_means(
                self.train_cell_type_clustering_losses
            ),
            "train_cross_modal_cell_type_loss": get_epoch_means(
                self.train_cross_modal_cell_type_losses
            ),
            "train_cn_distribution_separation_loss": get_epoch_means(
                self.train_cn_separation_losses
            ),
            "val_total_loss": self.val_total_loss,  # Validation losses are already per-epoch
            "val_rna_loss": self.val_rna_loss,
            "val_protein_loss": self.val_protein_loss,
            "val_matching_loss": self.val_matching_loss,
            "val_contrastive_loss": self.val_contrastive_loss,
            "val_similarity_loss": self.val_similarity_loss,
            "val_raw_similarity_loss": self.val_raw_similarity_loss,
            "val_cell_type_clustering_loss": self.val_cell_type_clustering_loss,
            "val_cross_modal_cell_type_loss": self.val_cross_modal_cell_type_loss,
            "val_cn_distribution_separation_loss": self.val_cn_distribution_separation_loss,
            # Also include the validation epoch indices
            "val_epochs": val_epochs,
        }

        # Add iLISI scores if available
        if hasattr(self, "val_ilisi_scores"):
            history_["val_ilisi_scores"] = self.val_ilisi_scores
        return history_

    def on_epoch_end(self):
        """Called at the end of each training epoch."""
        # Calculate and log weight changes
        weight_update_l2_rna = 0.0
        for name, param in self.rna_vae.module.named_parameters():
            if name in self.previous_weights_rna:
                weight_update_l2_rna += torch.norm(
                    param.detach() - self.previous_weights_rna[name]
                ).item()
                self.previous_weights_rna[name] = param.clone().detach()

        weight_update_l2_protein = 0.0
        for name, param in self.protein_vae.module.named_parameters():
            if name in self.previous_weights_protein:
                weight_update_l2_protein += torch.norm(
                    param.detach() - self.previous_weights_protein[name]
                ).item()
                self.previous_weights_protein[name] = param.clone().detach()
        # Weight updates will be logged via the centralized logging function
        self.weight_update_l2_rna = weight_update_l2_rna
        self.weight_update_l2_protein = weight_update_l2_protein

        if self.on_epoch_end_called:
            self.on_epoch_end_called = False  # prevent recursion
            if self.current_epoch == 0:
                self.logger_.info(
                    f"\n[Epoch {self.current_epoch+1}] Initializing training..."
                )  # Log before first epoch

        # Store learning rates for centralized logging
        optimizer = self.optimizers()  # Get the ACTUAL optimizer being used
        rna_lr = optimizer.param_groups[0]["lr"]
        protein_lr = optimizer.param_groups[1]["lr"]
        self.current_rna_lr = rna_lr
        self.current_protein_lr = protein_lr

        if rna_lr < 1e-6 or protein_lr < 1e-6:
            self.logger_.warning(
                f"Very low learning rates detected at epoch {self.current_epoch}: "
                f"RNA LR = {rna_lr:.2e}, Protein LR = {protein_lr:.2e}"
            )

        # Log learning rate info every few epochs
        if self.current_epoch % 5 == 0:
            self.logger_.info(
                f"Epoch {self.current_epoch} learning rates: "
                f"RNA = {rna_lr:.2e}, Protein = {protein_lr:.2e}"
            )

        # Log step-level metrics to MLflow
        self.log_to_mlflow_centralized("step")

        # Determine if we should log to MLflow (skip during warmup unless using cached scales)
        should_log_to_mlflow = False
        if self.scales_from_cache:
            # Using cached scales - log to MLflow immediately
            should_log_to_mlflow = True
        elif self.current_epoch >= self.warmup_epochs:
            # Past warmup period - log to MLflow normally
            should_log_to_mlflow = True
        else:
            # During warmup period - don't log to MLflow
            should_log_to_mlflow = False

        # Log epoch-level training loss for ReduceLROnPlateau scheduler
        if len(self.train_losses) > 0:
            # Calculate average training loss for this epoch
            steps_this_epoch = min(self.steps_per_epoch, len(self.train_losses))
            epoch_train_loss = np.mean(self.train_losses[-steps_this_epoch:])

            # Log it for the scheduler to monitor
            self.log("epoch_train_loss", epoch_train_loss, on_epoch=True, prog_bar=True)

            # Only log to MLflow if we're past warmup or using cached scales
            if should_log_to_mlflow:
                mlflow.log_metric("train/epoch_loss", epoch_train_loss, step=self.current_epoch)

            if self.current_epoch % 5 == 0:
                self.logger_.info(
                    f"Epoch {self.current_epoch} average training loss: {epoch_train_loss:.4f}"
                )

        if self.current_epoch == self.warmup_epochs - 1 and self.warmup_epochs > 0:
            self.logger_.info(f"--- End of warmup phase (epoch {self.current_epoch}) ---")
            self.logger_.info("Calculating loss scales based on warmup period.")
            for name, values in self.warmup_raw_losses.items():
                if values:
                    # Use a small epsilon to avoid division by zero or extremely large scales, and use median to avoid outliers
                    median_loss = np.median(values) + 1e-8
                    self.loss_scales[name] = 100.0 / median_loss
                    self.logger_.info(
                        f"  - Calculated scale for '{name}': {self.loss_scales[name]:.4f} "
                        f"(from median loss: {median_loss:.4f})"
                    )
                else:
                    self.logger_.warning(
                        f"  - No values for '{name}' during warmup, scale remains 1.0."
                    )

            self.logger_.info("Loss scales updated.")
            # Save scales to cache via callback
            if self.save_scales_callback:
                self.save_scales_callback(self.training_params_for_hash, self.loss_scales)

            # Plot the distributions of the raw losses from the warmup
            plot_warmup_loss_distributions(
                self.warmup_raw_losses, precentile=(10, 90), plot_flag=plot_flag
            )
            if mlflow.active_run() and not self.scales_logged_to_mlflow:
                mlflow.log_params({f"scale_{k}": v for k, v in self.loss_scales.items()})
                self.scales_logged_to_mlflow = True

            # Unfreeze model parameters now that warmup is complete
            self._unfreeze_models()
            self.logger_.info(
                "Model parameters unfrozen. Training will begin with calculated loss scales."
            )

            # Clear loss histories to prevent logging huge warmup losses
            self.logger_.info(
                "Clearing loss histories from warmup period to prevent MLflow spikes..."
            )
            self.train_losses.clear()
            self.train_rna_losses.clear()
            self.train_protein_losses.clear()
            self.train_matching_losses.clear()
            self.train_contrastive_losses.clear()
            self.train_adv_losses.clear()
            self.train_cell_type_clustering_losses.clear()
            self.train_cross_modal_cell_type_losses.clear()
            self.train_cn_separation_losses.clear()
            self.similarity_losses.clear()
            self.raw_similarity_losses.clear()
            self.similarity_weight_history.clear()
            self.active_similarity_loss_active_history.clear()

            # Clear outlier detection histories to prevent false outlier detection
            self.logger_.info("Clearing outlier detection histories from warmup period...")
            self.rna_loss_history.clear()
            self.protein_loss_history.clear()

            # Note: Scheduler transition removed - was causing NaN values
            # The warmup->main scheduler transition will be handled differently
            self.outlier_count_rna = 0
            self.outlier_count_protein = 0
            self.logger_.info(
                "All histories cleared. MLflow logging and outlier detection will start fresh from next epoch."
            )

        if self.current_epoch % self.print_every_n_epoch == 0:
            log_epoch_end(
                self.current_epoch,
                self.train_losses,
                self.val_total_loss,
                log_to_mlflow=should_log_to_mlflow,
            )
        # Track if epoch metrics were logged during checkpoint save
        epoch_metrics_logged = False
        if (
            self.current_epoch % self.save_checkpoint_every_n_epochs
        ) == 0 and self.current_epoch > 2:
            self.save_checkpoints()
            epoch_metrics_logged = True  # save_checkpoints calls log_to_mlflow_centralized("epoch")

        # Log epoch-level metrics to MLflow (skip if already logged during checkpoint save)
        if not epoch_metrics_logged:
            self.log_to_mlflow_centralized("epoch")

        self.on_epoch_end_called = True

    @staticmethod
    def load_from_checkpoints(
        checkpoints_path,
        rna_vae,
        protein_vae,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        """Load a checkpoints from disk.

        Args:
            checkpoints_path: Path to the checkpoints directory
            device: Device to load the models to

        Returns:
            rna_vae: Loaded RNA VAE model
            protein_vae: Loaded protein VAE model
            training_state: Dictionary with training state
        """
        print(f"Loading checkpoints from {checkpoints_path}...")

        # Load state dictionaries
        rna_vae.module.load_state_dict(
            torch.load(f"{checkpoints_path}/rna_vae_model.pt", map_location=device)
        )
        protein_vae.module.load_state_dict(
            torch.load(f"{checkpoints_path}/protein_vae_model.pt", map_location=device)
        )
        rna_vae.use_pretrained_checkpoints = True
        # Load training state
        training_state = torch.load(f"{checkpoints_path}/training_state.pt", map_location=device)

        # Set models as trained
        rna_vae.is_trained_ = False
        protein_vae.is_trained_ = False
        # since some of the losses weights were set to 0 during pretraining, we need to force warmup to calculate appropriate scales
        rna_vae.use_pretrained_checkpoints = True
        protein_vae.use_pretrained_checkpoints = True
        rna_vae.warmup_epochs = 10  # Do warmup to calculate appropriate scales
        rna_vae.scales_from_cache = False
        # Move models to device
        rna_vae.module.to(device)
        protein_vae.module.to(device)
        training_state["loss_scales"] = None

        print(f"Models loaded successfully from {checkpoints_path}")

        return rna_vae, protein_vae, training_state

    @staticmethod
    def create_models_from_config(config_path, adata_rna, adata_prot):
        """Create SCVI models from saved model configuration JSON and input AnnData.

        Args:
            config_path: Path to the model_config.json file
            adata_rna: RNA AnnData object
            adata_prot: Protein AnnData object

        Returns:
            rna_vae: RNA SCVI model
            protein_vae: Protein SCVI model
        """
        print(f"Loading model configuration from {config_path}...")

        with open(config_path, "r") as f:
            model_config = json.load(f)

        # Setup AnnData for scVI using saved configuration
        print(model_config)
        if "anndata_setup" in model_config.keys():
            rna_setup = model_config["anndata_setup"]["rna_setup"]
            protein_setup = model_config["anndata_setup"]["protein_setup"]
        else:
            rna_setup = {"labels_key": "index_col", "batch_key": "batch", "layer": None, "": []}
            protein_setup = {"labels_key": "index_col", "batch_key": "batch", "layer": None, "": []}
        scvi.model.SCVI.setup_anndata(
            adata_rna,
            labels_key=rna_setup["labels_key"],
            batch_key=rna_setup["batch_key"],
        )

        scvi.model.SCVI.setup_anndata(
            adata_prot,
            labels_key=protein_setup["labels_key"],
            batch_key=protein_setup["batch_key"],
        )

        # Create RNA VAE model with saved configuration
        if "rna_model" in model_config.keys():
            rna_config = model_config["rna_model"]
        else:
            rna_config = {
                "gene_likelihood": "zinb",
                "n_hidden": 1024,
                "n_layers": 3,
                "n_latent": 60,
                "dropout_rate": 0.1,
                "use_observed_lib_size": True,
                "use_batch_norm": "both",
            }
        rna_vae = scvi.model.SCVI(
            adata_rna,
            gene_likelihood=rna_config["gene_likelihood"],
            n_hidden=rna_config["n_hidden"],
            n_layers=rna_config["n_layers"],
            n_latent=rna_config["n_latent"],
            dropout_rate=rna_config.get("dropout_rate", 0.1),
            use_observed_lib_size=rna_config.get("use_observed_lib_size", True),
            use_batch_norm=rna_config.get("use_batch_norm", "both"),
        )

        # Create protein VAE model with saved configuration
        if "protein_model" in model_config.keys():
            protein_config = model_config["protein_model"]
        else:
            protein_config = {
                "gene_likelihood": "zinb",
                "n_hidden": 512,
                "n_layers": 3,
                "n_latent": 60,
                "dropout_rate": 0.1,
                "use_observed_lib_size": True,
                "use_batch_norm": "both",
            }
        protein_vae = scvi.model.SCVI(
            adata_prot,
            gene_likelihood=protein_config["gene_likelihood"],
            n_hidden=protein_config["n_hidden"],
            n_layers=protein_config["n_layers"],
            n_latent=protein_config["n_latent"],
            dropout_rate=protein_config.get("dropout_rate", 0.1),
            use_observed_lib_size=protein_config.get("use_observed_lib_size", True),
            use_batch_norm=protein_config.get("use_batch_norm", "both"),
        )

        print(f"Models created successfully from configuration")
        print(
            f"RNA model: {rna_config['n_latent']}D latent, {rna_config['n_hidden']} hidden, {rna_config['n_layers']} layers"
        )
        print(
            f"Protein model: {protein_config['n_latent']}D latent, {protein_config['n_hidden']} hidden, {protein_config['n_layers']} layers"
        )

        return rna_vae, protein_vae

    def update_similarity_weight(self, losses):
        """Adaptive loss:
        if integration is poor, double weight
        if integration is good, half weight
        """
        # Update last_ilisi_score if we calculated a new one
        if "ilisi_score" in losses:
            new_score = losses["ilisi_score"]
            # Validate the new score before using it
            if np.isfinite(new_score) and new_score > 0:
                self.last_ilisi_score = float(new_score)
                if self.to_print:
                    self.logger_.info(
                        f"Using newly calculated iLISI score: {self.last_ilisi_score}"
                    )
            else:
                if self.to_print:
                    self.logger_.warning(
                        f"Invalid new iLISI score {new_score}, keeping cached: {self.last_ilisi_score}"
                    )
        else:
            if self.to_print:
                self.logger_.info(f"Using cached iLISI score: {self.last_ilisi_score}")

        # Only include valid iLISI scores in the losses dictionary for logging
        if np.isfinite(self.last_ilisi_score) and self.last_ilisi_score > 0:
            losses["ilisi_score"] = float(self.last_ilisi_score)
        else:
            # Don't include invalid scores in losses dictionary
            if self.to_print:
                self.logger_.warning(
                    f"Cached iLISI score is invalid ({self.last_ilisi_score}), not including in losses"
                )

        # Update similarity weight if we have a valid iLISI score
        # Allow reactivation from 0 if iLISI score is poor enough
        if (
            np.isfinite(self.last_ilisi_score)
            and self.last_ilisi_score > 0
            and not self.disable_similarity_weight_updates
        ):
            ilisi_threshold = 1.5
            if self.last_ilisi_score < ilisi_threshold:
                # If iLISI is too low, increase the similarity weight

                self.similarity_weight = min(1e9, self.similarity_weight * 1.2)
                if self.to_print:
                    self.logger_.info(
                        f"[Step {self.global_step}] iLISI score is {self.last_ilisi_score:.4f} (< {ilisi_threshold}), increasing similarity weight to {self.similarity_weight}"
                    )
                self.similarity_active = True
                self.similarity_loss_steady_counter = 0  # Reset steady state counter
            elif self.last_ilisi_score >= ilisi_threshold:  # self.similarity_weight > 10
                # If iLISI is good and weight is high, reduce it gradually
                self.similarity_weight = max(self.similarity_weight * 0.9, 0.01)
                if self.to_print:
                    self.logger_.info(
                        f"[Step {self.global_step}] iLISI score is {self.last_ilisi_score:.4f} (>= {ilisi_threshold}), reducing similarity weight to {self.similarity_weight}"
                    )
        elif self.disable_similarity_weight_updates and self.global_step <= 5:
            # Log that updates are disabled (only for first few steps to avoid spam)
            if self.to_print:
                reason = "similarity_dynamic=False" if not self.similarity_dynamic else "disabled"
                self.logger_.info(
                    f"[Step {self.global_step}] Similarity weight updates {reason} - keeping weight at {self.similarity_weight}"
                )
        else:
            # Skip similarity weight updates when iLISI score is invalid
            if self.to_print:
                self.logger_.debug(
                    f"Skipping similarity weight update due to invalid iLISI score: {self.last_ilisi_score}"
                )

        return losses

    def calculate_counterfactual_validation_metrics(
        self,
        original_rna_adata: AnnData,
        original_protein_adata: AnnData,
        counterfactual_rna_adata: AnnData,
        counterfactual_protein_adata: AnnData,
        subsample_size=500,  # Further reduced from 1000 to 500 to save memory
    ):
        """Calculate similarity and iLISI metrics for counterfactual reconstruction.

        Args:
            original_rna_adata: Original RNA AnnData object
            original_protein_adata: Original protein AnnData object
            counterfactual_rna_adata: Counterfactual RNA AnnData object
            counterfactual_protein_adata: Counterfactual protein AnnData object

        Returns:
            Dictionary with similarity and iLISI metrics
        """

        metrics = {}

        self.logger_.info("Calculating reconstruction similarity scores...")

        # Convert to latent space for similarity calculations
        # We'll use the raw expression data as a proxy for latent space comparison
        # Safely subsample the data
        def safe_subsample(adata, target_size):
            if adata.shape[0] <= target_size:
                return adata.copy()
            else:
                adata_sub = adata.copy()
                sc.pp.subsample(adata_sub, n_obs=target_size)
                return adata_sub

        original_rna_adata_sub = safe_subsample(original_rna_adata, subsample_size)
        counterfactual_rna_adata_sub = safe_subsample(counterfactual_rna_adata, subsample_size)
        original_protein_adata_sub = safe_subsample(original_protein_adata, subsample_size)
        counterfactual_protein_adata_sub = safe_subsample(
            counterfactual_protein_adata, subsample_size
        )

        # Clean data to remove NaN/inf values
        def clean_data(data):
            if issparse(data):
                data = data.toarray()
            return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Memory-efficient distance calculation using sampling instead of full cdist
        def calculate_similarity_efficient(orig_tensor, recon_tensor):
            """Calculate similarity using sampling to avoid large distance matrices."""
            n_samples = min(200, orig_tensor.shape[0])  # Sample max 200 pairs to reduce memory

            # Randomly sample indices for distance calculation
            indices = torch.randperm(orig_tensor.shape[0])[:n_samples]

            orig_sample = orig_tensor[indices]
            recon_sample = recon_tensor[indices]

            # Calculate pairwise distances for sampled data only
            with torch.no_grad():  # Don't need gradients for metrics
                orig_distances = torch.cdist(orig_sample, orig_sample)
                recon_distances = torch.cdist(recon_sample, recon_sample)
                cross_distances = torch.cdist(orig_sample, recon_sample)

                # Calculate means
                avg_intra_dis = (orig_distances.mean() + recon_distances.mean()) / 2
                inter_dis = cross_distances.mean()

                # Clear distance matrices immediately to free memory
                del orig_distances, recon_distances, cross_distances
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                # Normalize by average intra-modality distance
                if avg_intra_dis > 1e-8:
                    similarity_loss = torch.abs(avg_intra_dis - inter_dis) / avg_intra_dis
                else:
                    similarity_loss = torch.tensor(0.0)

            return similarity_loss

        original_rna_tensor = torch.tensor(
            clean_data(original_rna_adata_sub.X), dtype=torch.float32
        )
        counterfactual_rna_tensor = torch.tensor(
            clean_data(counterfactual_rna_adata_sub.X), dtype=torch.float32
        )
        original_protein_tensor = torch.tensor(
            clean_data(original_protein_adata_sub.X), dtype=torch.float32
        )
        counterfactual_protein_tensor = torch.tensor(
            clean_data(counterfactual_protein_adata_sub.X), dtype=torch.float32
        )

        # 1. Calculate RNA reconstruction similarity using memory-efficient method
        rna_recon_similarity_loss = calculate_similarity_efficient(
            original_rna_tensor, counterfactual_rna_tensor
        )

        # 2. Calculate protein reconstruction similarity using memory-efficient method
        protein_recon_similarity_loss = calculate_similarity_efficient(
            original_protein_tensor, counterfactual_protein_tensor
        )

        metrics["rna_recon_similarity_loss"] = float(rna_recon_similarity_loss.item())
        metrics["protein_recon_similarity_loss"] = float(protein_recon_similarity_loss.item())

        # Clear tensors to free memory
        del original_rna_tensor, counterfactual_rna_tensor
        del original_protein_tensor, counterfactual_protein_tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        self.logger_.info("Calculating iLISI scores for original vs reconstructed...")
        try:
            # Clean data for iLISI calculation
            def clean_adata_for_neighbors(adata):
                adata_clean = adata.copy()
                if issparse(adata_clean.X):
                    adata_clean.X = adata_clean.X.toarray()
                adata_clean.X = np.nan_to_num(adata_clean.X, nan=0.0, posinf=0.0, neginf=0.0)
                return adata_clean

            # Create combined AnnData for RNA modality (original vs counterfactual)
            rna_combined = anndata.concat(
                [
                    clean_adata_for_neighbors(original_rna_adata_sub),
                    clean_adata_for_neighbors(counterfactual_rna_adata_sub),
                ],
                join="outer",
                label="type",
                keys=["original", "counterfactual"],
            )

            # Create combined AnnData for protein modality (original vs counterfactual)
            protein_combined = anndata.concat(
                [
                    clean_adata_for_neighbors(original_protein_adata_sub),
                    clean_adata_for_neighbors(counterfactual_protein_adata_sub),
                ],
                join="outer",
                label="type",
                keys=["original", "counterfactual"],
            )

            # Calculate neighbors for iLISI calculation with safe n_neighbors
            safe_n_neighbors_rna = min(15, rna_combined.shape[0] - 1)
            safe_n_neighbors_protein = min(15, protein_combined.shape[0] - 1)

            if safe_n_neighbors_rna > 0:
                sc.pp.neighbors(rna_combined, use_rep="X", n_neighbors=safe_n_neighbors_rna)
                rna_ilisi = calculate_iLISI(
                    rna_combined, "type", plot_flag=False, use_subsample=True
                )
            else:
                rna_ilisi = 0.0

            if safe_n_neighbors_protein > 0:
                sc.pp.neighbors(protein_combined, use_rep="X", n_neighbors=safe_n_neighbors_protein)
                protein_ilisi = calculate_iLISI(
                    protein_combined, "type", plot_flag=False, use_subsample=True
                )
            else:
                protein_ilisi = 0.0

        except Exception as e:
            self.logger_.error(f"Error calculating iLISI scores: {e}")
            rna_ilisi = 0.0
            protein_ilisi = 0.0

        # Validate iLISI scores
        if np.isfinite(rna_ilisi) and rna_ilisi > 0:
            metrics["rna_ilisi_score"] = float(rna_ilisi)
        else:
            metrics["rna_ilisi_score"] = 0.0
            self.logger_.warning(f"Invalid rna iLISI score: {rna_ilisi}")

        if np.isfinite(protein_ilisi) and protein_ilisi > 0:
            metrics["protein_ilisi_score"] = float(protein_ilisi)
        else:
            metrics["protein_ilisi_score"] = 0.0
            self.logger_.warning(f"Invalid protein iLISI score: {protein_ilisi}")

        self.logger_.info(f"Calculated {len(metrics)} reconstruction validation metrics")
        return metrics

    def calculate_losses(
        self,
        rna_batch,
        protein_batch,
        rna_vae,
        protein_vae,
        rna_inference_outputs_tuple,
        protein_inference_outputs_tuple,
        device,
        similarity_weight,
        similarity_active,
        contrastive_weight,
        matching_weight,
        cell_type_clustering_weight,
        cross_modal_cell_type_weight,
        rna_recon_weight,
        prot_recon_weight,
        global_step=None,
        total_steps=None,
        to_plot=False,
        check_ilisi=False,
    ):
        """Calculate all losses for a batch of data.

        Args:
            rna_batch: Dictionary containing RNA batch data
            protein_batch: Dictionary containing protein batch data
            rna_vae: RNA VAE model
            protein_vae: Protein VAE model
            device: Device to use for calculations
            similarity_weight: Weight for similarity loss
            similarity_active: Whether similarity loss is active
            contrastive_weight: Weight for contrastive loss
            matching_weight: Weight for matching loss
            cell_type_clustering_weight: Weight for within-modality cell type clustering
            cross_modal_cell_type_weight: Weight for cross-modal cell type alignment
            rna_recon_weight: Weight for RNA KL loss
            prot_recon_weight: Weight for protein KL loss
            global_step: Current global step
            total_steps: Total number of steps
            to_plot: Whether to plot metrics
            check_ilisi: Whether to check iLISI score

        Returns:
            Dictionary containing all calculated losses and metrics
        """
        # Get model outputs
        rna_inference_outputs, _, raw_rna_loss_output = rna_inference_outputs_tuple
        protein_inference_outputs, _, raw_protein_loss_output = protein_inference_outputs_tuple
        # Log raw losses before scaling for debugging
        raw_rna_unscaled = raw_rna_loss_output.loss.item()
        raw_protein_unscaled = raw_protein_loss_output.loss.item()

        # Extract KL losses for logging (similar to simple_protein_vae.py) - global only
        rna_kl_global = getattr(raw_rna_loss_output, "kl_global", None)
        protein_kl_global = getattr(raw_protein_loss_output, "kl_global", None)

        # Apply scaling first
        raw_rna_loss_scaled = raw_rna_loss_output.loss * self.loss_scales["rna_reconstruction"]
        raw_protein_loss_scaled = (
            raw_protein_loss_output.loss * self.loss_scales["protein_reconstruction"]
        )

        # Apply final weighting
        rna_loss_weighted = raw_rna_loss_scaled * rna_recon_weight
        protein_loss_weighted = raw_protein_loss_scaled * prot_recon_weight

        # Apply outlier detection to final weighted losses
        rna_is_outlier, corrected_rna_loss = self.detect_outlier_loss(
            rna_loss_weighted, self.rna_loss_history, "RNA"
        )
        protein_is_outlier, corrected_protein_loss = self.detect_outlier_loss(
            protein_loss_weighted, self.protein_loss_history, "Protein"
        )

        # Update outlier counts
        if rna_is_outlier:
            self.outlier_count_rna += 1
        if protein_is_outlier:
            self.outlier_count_protein += 1

        # Use corrected losses (already scaled and weighted)
        # CRITICAL: Preserve computational graph by scaling original tensors instead of creating new ones
        if rna_is_outlier:
            # Apply outlier correction as scaling factor to preserve gradients
            # Avoid division by zero by checking if original loss is non-zero
            original_rna_val = rna_loss_weighted.item()
            if abs(original_rna_val) > 1e-10:
                outlier_scale_rna = corrected_rna_loss / original_rna_val
                rna_loss_output = rna_loss_weighted * outlier_scale_rna
            else:
                # If original loss is effectively zero, use corrected value directly
                rna_loss_output = torch.tensor(
                    corrected_rna_loss,
                    device=rna_loss_weighted.device,
                    dtype=rna_loss_weighted.dtype,
                    requires_grad=True,
                )
        else:
            rna_loss_output = rna_loss_weighted

        if protein_is_outlier:
            # Apply outlier correction as scaling factor to preserve gradients
            # Avoid division by zero by checking if original loss is non-zero
            original_protein_val = protein_loss_weighted.item()
            if abs(original_protein_val) > 1e-10:
                outlier_scale_protein = corrected_protein_loss / original_protein_val
                protein_loss_output = protein_loss_weighted * outlier_scale_protein
            else:
                # If original loss is effectively zero, use corrected value directly
                protein_loss_output = torch.tensor(
                    corrected_protein_loss,
                    device=protein_loss_weighted.device,
                    dtype=protein_loss_weighted.dtype,
                    requires_grad=True,
                )
        else:
            protein_loss_output = protein_loss_weighted

        # For logging purposes, also keep the scaled (but not weighted) versions
        raw_rna_loss_output = raw_rna_loss_scaled
        raw_protein_loss_output = raw_protein_loss_scaled

        if self.to_print:
            self.logger_.info(
                f"Loss Debug - Raw unscaled: RNA={raw_rna_unscaled:.4f}, Protein={raw_protein_unscaled:.4f}"
            )
            self.logger_.info(
                f"Loss Debug - Corrected: RNA={corrected_rna_loss:.4f} (outlier: {rna_is_outlier}), "
                f"Protein={corrected_protein_loss:.4f} (outlier: {protein_is_outlier})"
            )
            self.logger_.info(
                f"Loss Debug - Outlier counts: RNA={self.outlier_count_rna}, Protein={self.outlier_count_protein}"
            )
            self.logger_.info(
                f"Loss Debug - Scales: RNA={self.loss_scales['rna_reconstruction']:.4f}, "
                f"Protein={self.loss_scales['protein_reconstruction']:.4f}"
            )
            self.logger_.info(
                f"Loss Debug - After scaling: RNA={raw_rna_loss_output.item():.4f}, "
                f"Protein={raw_protein_loss_output.item():.4f}"
            )
            self.logger_.info(
                f"Loss Debug - Final weighted: RNA={rna_loss_output.item():.4f}, "
                f"Protein={protein_loss_output.item():.4f} (weights: RNA={rna_recon_weight:.4f}, Protein={prot_recon_weight:.4f})"
            )
        raw_matching_loss = torch.tensor(0.0)
        matching_loss = torch.tensor(0.0)
        latent_distances = np.array([0])
        rna_latent_mean = None
        protein_latent_mean = None
        # # Get latent representations
        rna_latent_mean = rna_inference_outputs["qz"].mean
        rna_latent_std = rna_inference_outputs["qz"].scale
        protein_latent_mean = protein_inference_outputs["qz"].mean

        if to_plot:
            self.logger_.info(f"[TRAINING DEBUG] RNA latent shape: {rna_latent_mean.shape}")
            self.logger_.info(
                f"[TRAINING DEBUG] RNA latent mean: {rna_latent_mean.mean():.6f}, std: {rna_latent_mean.std():.6f}"
            )
            self.logger_.info(
                f"[TRAINING DEBUG] RNA latent min: {rna_latent_mean.min():.6f}, max: {rna_latent_mean.max():.6f}"
            )
            self.logger_.info(f"[TRAINING DEBUG] Protein latent shape: {protein_latent_mean.shape}")
            self.logger_.info(
                f"[TRAINING DEBUG] Protein latent mean: {protein_latent_mean.mean():.6f}, std: {protein_latent_mean.std():.6f}"
            )
            self.logger_.info(
                f"[TRAINING DEBUG] Protein latent min: {protein_latent_mean.min():.6f}, max: {protein_latent_mean.max():.6f}"
            )

        protein_latent_std = protein_inference_outputs["qz"].scale
        latent_distances = compute_pairwise_kl_two_items(
            rna_latent_mean, protein_latent_mean, rna_latent_std, protein_latent_std
        )
        # todo should only calculate once and reuse
        rna_indices = torch.tensor(rna_batch["labels"], dtype=torch.long)
        prot_indices = torch.tensor(protein_batch["labels"], dtype=torch.long)
        archetype_dis = self.archetype_distances[rna_indices, :][:, prot_indices]
        archetype_dis = torch.clamp(archetype_dis, max=torch.quantile(archetype_dis, 0.90))

        # Predict the RNA CN from the protein CN by finding the closest protein cell in the latent space
        cn_values = predict_rna_cn_from_protein_neighbors(
            latent_distances, protein_vae.adata.obs["CN"].values, protein_batch["labels"], k=3
        )
        # Convert to categorical with the same categories as the target column
        cn_values_categorical = pd.Categorical(
            cn_values, categories=rna_vae.adata.obs["CN"].cat.categories
        )

        rna_vae.adata.obs.loc[rna_vae.adata.obs.index[rna_batch["labels"]], "CN"] = (
            cn_values_categorical
        )

        # Calculate extreme archetype alignment metric
        extreme_alignment_percentage = 0.0
        if matching_weight != 0:

            # Calculate extreme archetypes loss and alignment metric together
            raw_matching_loss, extreme_alignment_percentage = extreme_archetypes_loss(
                rna_batch,
                protein_batch,
                latent_distances,
                self.logger_,
                to_print=self.to_print,
                rna_vae=rna_vae,
                protein_vae=protein_vae,
            )
            # Apply loss scaling first, then weight (consistent with other losses)
            raw_matching_loss = raw_matching_loss * self.loss_scales["matching"]
            matching_loss = raw_matching_loss * matching_weight

        # Debug: Print matching loss details every 50 steps
        if global_step is not None and global_step % 50 == 0 and matching_weight > 0:
            self.logger_.info(
                f"\n[DEBUG Step {global_step}] Matching Loss Analysis:\n"
                f"├─ Raw matching loss: {raw_matching_loss.item():.6f}\n"
                f"├─ Matching weight: {matching_weight:.6f}\n"
                f"├─ Final matching loss: {matching_loss.item():.6f}\n"
                f"└─ Effective contribution: {(matching_loss / (matching_loss + 1e-8)).item():.4f}"
            )
        # # Calculate contrastive loss
        # rna_distances = compute_pairwise_kl(rna_latent_mean, rna_latent_std)
        # prot_distances = compute_pairwise_kl(protein_latent_mean, protein_latent_std)
        # distances = prot_distances + rna_distances

        # # Get cell type and neighborhood info
        # cell_neighborhood_info_protein = torch.tensor(
        #     protein_vae.adata[protein_batch["labels"]].obs["CN"].cat.codes.values
        # ).to(device)
        # cell_neighborhood_info_rna = torch.tensor(
        #     rna_vae.adata[rna_batch["labels"]].obs["CN"].cat.codes.values
        # ).to(device)

        # rna_major_cell_type = (
        #     torch.tensor(rna_vae.adata[rna_batch["labels"]].obs["major_cell_types"].values.codes)
        #     .to(device)
        #     .squeeze()
        # )
        # protein_major_cell_type = (
        #     torch.tensor(
        #         protein_vae.adata[protein_batch["labels"]].obs["major_cell_types"].values.codes
        #     )
        #     .to(device)
        #     .squeeze()
        # )

        # # Create masks for different cell type and neighborhood combinations
        num_cells = rna_batch["X"].shape[0]
        # same_cn_mask = cell_neighborhood_info_rna.unsqueeze(
        #     0
        # ) == cell_neighborhood_info_protein.unsqueeze(1)
        # same_major_cell_type = rna_major_cell_type.unsqueeze(
        #     0
        # ) == protein_major_cell_type.unsqueeze(1)
        # diagonal_mask = torch.eye(num_cells, dtype=torch.bool, device=device)

        # distances = distances.masked_fill(diagonal_mask, 0)

        # same_major_type_same_cn_mask = (same_major_cell_type * same_cn_mask).type(torch.bool)
        # same_major_type_different_cn_mask = (same_major_cell_type * ~same_cn_mask).type(torch.bool)
        # different_major_type_same_cn_mask = (~same_major_cell_type * same_cn_mask).type(torch.bool)
        # different_major_type_different_cn_mask = (~same_major_cell_type * ~same_cn_mask).type(
        #     torch.bool
        # )

        # same_major_type_same_cn_mask.masked_fill_(diagonal_mask, 0)
        # same_major_type_different_cn_mask.masked_fill_(diagonal_mask, 0)
        # different_major_type_same_cn_mask.masked_fill_(diagonal_mask, 0)
        # different_major_type_different_cn_mask.masked_fill_(diagonal_mask, 0)

        # Calculate contrastive loss
        raw_contrastive_loss = torch.tensor(0.0)
        contrastive_loss = torch.tensor(0.0)
        # NOTE: Contrastive loss is currently disabled (weight set to 0)
        # Keeping this structure for potential future use
        if self.contrastive_weight != 0:
            # This section would need full implementation if contrastive loss is re-enabled
            # Currently commented out variables would need to be properly defined
            self.logger_.info("Contrastive loss is enabled but implementation is incomplete")
            contrastive_loss = torch.tensor(0.0)
        cell_type_clustering_loss = torch.tensor(0.0)
        raw_rna_cell_type_clustering_loss = torch.tensor(0.0)
        raw_prot_cell_type_clustering_loss = torch.tensor(0.0)
        raw_cross_modal_cell_type_loss = torch.tensor(0.0)
        cross_modal_cell_type_loss = torch.tensor(0.0)
        raw_cell_type_clustering_loss = torch.tensor(0.0)

        if cell_type_clustering_weight != 0:
            # Calculate cell type clustering loss with component breakdown
            rna_components = run_cell_type_clustering_loss(
                rna_vae.adata,
                rna_latent_mean,
                rna_batch["labels"],
                plot_flag=to_plot,
                modality_type="rna",
                return_components=True,
            )
            protein_components = run_cell_type_clustering_loss(
                protein_vae.adata,
                protein_latent_mean,
                protein_batch["labels"],
                plot_flag=to_plot,
                modality_type="protein",
                return_components=True,
            )

            # Extract raw losses for backward compatibility
            raw_rna_cell_type_clustering_loss = rna_components["total_loss"]
            raw_prot_cell_type_clustering_loss = protein_components["total_loss"]

            # Add the original balance term with margin
            margin = 0.1  # Acceptable difference margin
            loss_diff = torch.abs(
                raw_rna_cell_type_clustering_loss - raw_prot_cell_type_clustering_loss
            )
            balance_term = torch.nn.functional.relu(loss_diff - margin)  # Zero if within margin

            # NEW: Add modality balance loss for individual components
            modality_balance_loss = calculate_modality_balance_loss(
                rna_components, protein_components
            )

            # Combined cell type clustering loss (with new modality balance component)
            raw_cell_type_clustering_loss = (
                raw_rna_cell_type_clustering_loss
                + raw_prot_cell_type_clustering_loss
                + balance_term
                + modality_balance_loss
            ) * self.loss_scales["cell_type_clustering"]
            cell_type_clustering_loss = raw_cell_type_clustering_loss * cell_type_clustering_weight
            # Consolidate cell type clustering logs
            if self.to_print:
                self.logger_.info(
                    f"\nCell Type Clustering Summary:\n"
                    f"├─ RNA raw loss: {raw_rna_cell_type_clustering_loss:.4f}\n"
                    f"│  ├─ Structure: {rna_components['structure_preservation']:.4f}\n"
                    f"│  ├─ Cohesion: {rna_components['cohesion']:.4f}\n"
                    f"│  └─ Separation: {rna_components['separation']:.4f}\n"
                    f"├─ Protein raw loss: {raw_prot_cell_type_clustering_loss:.4f}\n"
                    f"│  ├─ Structure: {protein_components['structure_preservation']:.4f}\n"
                    f"│  ├─ Cohesion: {protein_components['cohesion']:.4f}\n"
                    f"│  └─ Separation: {protein_components['separation']:.4f}\n"
                    f"├─ Cross-modal raw loss: {raw_cross_modal_cell_type_loss:.4f}\n"
                    f"├─ Cross-modal weighted loss: {cross_modal_cell_type_loss:.4f}\n"
                    f"├─ Loss difference: {loss_diff:.4f} (margin: {margin:.4f})\n"
                    f"├─ Balance term: {balance_term:.4f}\n"
                    f"├─ Modality balance loss: {modality_balance_loss:.4f}\n"
                    f"└─ Total cell type clustering loss: {cell_type_clustering_loss:.4f}"
                )

        if cross_modal_cell_type_weight != 0:
            # Calculate MMD-based cross-modal cell type clustering loss
            raw_cross_modal_cell_type_loss = (
                calculate_cross_modal_cell_type_loss(
                    rna_vae.adata,
                    protein_vae.adata,
                    rna_latent_mean,
                    protein_latent_mean,
                    rna_batch["labels"],
                    protein_batch["labels"],
                    device,
                    sigma=self.cross_modal_mmd_sigma,  # Configurable RBF kernel bandwidth
                )
                * self.loss_scales["cross_modal_cell_type"]
            )
            cross_modal_cell_type_loss = (
                raw_cross_modal_cell_type_loss * cross_modal_cell_type_weight
            )

        # Calculate similarity loss
        raw_similarity_loss_unscaled = torch.tensor(0.0, device=device)  # Initialize for scope
        if similarity_weight == 0 or not similarity_active:
            similarity_loss = torch.tensor(0.0, device=device)
            raw_similarity_loss = torch.tensor(0.0, device=device)
        else:
            # Use gradient-compatible distance computation instead of torch.cdist
            # Compute pairwise Euclidean distances manually
            rna_dis = compute_pairwise_distances(rna_latent_mean, rna_latent_mean)
            prot_dis = compute_pairwise_distances(protein_latent_mean, protein_latent_mean)
            rna_prot_dis = compute_pairwise_distances(rna_latent_mean, protein_latent_mean)

            average_intra_dis = (rna_dis.abs().mean() + prot_dis.abs().mean()) / 2
            inter_dis = rna_prot_dis.abs().mean()

            # Normalize by average intra-modality distance to make it scale-invariant
            if average_intra_dis > 1e-8:
                raw_similarity_loss_unscaled = (
                    torch.abs(average_intra_dis - inter_dis) / average_intra_dis
                )
            else:
                raw_similarity_loss_unscaled = torch.tensor(0.0, device=device)

            # Apply loss scaling first, then weight (consistent with other losses)
            raw_similarity_loss = raw_similarity_loss_unscaled * self.loss_scales["similarity"]
            similarity_loss = raw_similarity_loss * similarity_weight

            # Debug logging for similarity loss scaling
            if self.to_print and global_step is not None and global_step % 50 == 0:
                self.logger_.info(
                    f"\n[DEBUG Step {global_step}] Similarity Loss Analysis:\n"
                    f"├─ Raw unscaled similarity loss: {raw_similarity_loss_unscaled.item():.6f}\n"
                    f"├─ Similarity scale factor: {self.loss_scales['similarity']:.6f}\n"
                    f"├─ Raw scaled similarity loss: {raw_similarity_loss.item():.6f}\n"
                    f"├─ Similarity weight: {similarity_weight:.6f}\n"
                    f"├─ Similarity active: {similarity_active}\n"
                    f"└─ Final similarity loss: {similarity_loss.item():.6f}"
                )

        raw_cn_distribution_separation_loss = torch.tensor(0.0)
        cn_distribution_separation_loss_combined = torch.tensor(0.0)
        rna_cn_separation_loss = torch.tensor(0.0)
        protein_cn_separation_loss = torch.tensor(0.0)
        if self.cn_distribution_separation_weight != 0:
            # Calculate CN distribution separation loss for RNA and protein
            rna_cn_separation_loss = cn_distribution_separation_loss(
                rna_vae.adata[rna_batch["labels"]],
                rna_latent_mean,
                device=device,
                sigma=None,  # Auto-calculate sigma
                global_step=global_step,
            )
            # Apply 10x weight multiplier for protein CN separation
            protein_cn_separation_loss = cn_distribution_separation_loss(
                protein_vae.adata[protein_batch["labels"]],
                protein_latent_mean,
                device=device,
                sigma=None,  # Auto-calculate sigma
                global_step=global_step,
            )

            # Combined CN distribution separation loss
            raw_cn_distribution_separation_loss = (
                rna_cn_separation_loss + protein_cn_separation_loss
            )
            raw_cn_distribution_separation_loss = (
                raw_cn_distribution_separation_loss * self.loss_scales["cn_distribution_separation"]
            )
            cn_distribution_separation_loss_combined = (
                raw_cn_distribution_separation_loss * self.cn_distribution_separation_weight
            )

        # Calculate total loss

        # log a waringin if any of the raw losses are negative or has abs values larger than 100
        raw_losses_and_weights = {
            "raw_rna_cell_type_clustering_loss": {
                "loss": raw_rna_cell_type_clustering_loss,
                "weight": self.cell_type_clustering_weight,
            },
            "raw_prot_cell_type_clustering_loss": {
                "loss": raw_prot_cell_type_clustering_loss,
                "weight": self.cell_type_clustering_weight,
            },
            "raw_cross_modal_cell_type_loss": {
                "loss": raw_cross_modal_cell_type_loss,
                "weight": self.cross_modal_cell_type_weight,
            },
            "raw_cn_distribution_separation_loss": {
                "loss": raw_cn_distribution_separation_loss,
                "weight": self.cn_distribution_separation_weight,
            },
            "raw_contrastive_loss": {
                "loss": raw_contrastive_loss,
                "weight": self.contrastive_weight,
            },
            "raw_matching_loss": {"loss": raw_matching_loss, "weight": self.matching_weight},
            "raw_similarity_loss": {"loss": raw_similarity_loss, "weight": self.similarity_weight},
            "raw_cell_type_clustering_loss": {
                "loss": raw_cell_type_clustering_loss,
                "weight": self.cell_type_clustering_weight,
            },
        }

        for loss_name, loss_and_weight in raw_losses_and_weights.items():
            loss = loss_and_weight["loss"]
            weight = loss_and_weight["weight"]
            if weight > 0 and (loss < 0 or loss.abs() > 100) and self.to_print:
                self.logger_.warning(
                    f"Raw loss is negative or has abs values larger than 100: {loss_name}: {loss}"
                )

        total_loss = (
            rna_loss_output  # RNA reconstruction loss
            + protein_loss_output  # Protein reconstruction loss
            + contrastive_loss  #  will separate CN whithin each cell type cluster
            + matching_loss  #  will make the cell distances to be similar to the archetype distances
            + similarity_loss  # the modalities will be fouced to overlap
            + cell_type_clustering_loss  # cell types will be clustered in the latent space
            + cross_modal_cell_type_loss  # cell types will be close to each other in the latent space
            + cn_distribution_separation_loss_combined  # separate CN distributions within each cell type
        )

        # Debug: Show matching loss contribution relative to total loss
        if global_step is not None and global_step % 50 == 0 and matching_weight > 0:
            matching_contribution_pct = (matching_loss / total_loss * 100).item()
            self.logger_.info(
                f"[DEBUG Step {global_step}] Matching loss contribution: {matching_contribution_pct:.2f}% of total loss"
            )

        # Prepare metrics for plotting if needed
        # CRITICAL FIX: Use unscaled raw similarity loss for warmup scaling calculation
        # This ensures the scale calculation is based on the true magnitude of the loss
        raw_similarity_for_scaling = raw_similarity_loss_unscaled

        raw_losses_for_scaling = {
            "rna_reconstruction": raw_rna_loss_output,
            "protein_reconstruction": raw_protein_loss_output,
            "matching": raw_matching_loss,
            "contrastive": raw_contrastive_loss,
            "similarity": raw_similarity_for_scaling,  # Use unscaled version for proper scale calculation
            "cell_type_clustering": raw_cell_type_clustering_loss,
            "cross_modal_cell_type": raw_cross_modal_cell_type_loss,
            "cn_distribution_separation": raw_cn_distribution_separation_loss,
        }

        # Create losses dictionary
        losses = {
            "total_loss": total_loss,
            "rna_loss": rna_loss_output,
            "protein_loss": protein_loss_output,
            "contrastive_loss": contrastive_loss,
            "matching_loss": matching_loss,
            "similarity_loss": similarity_loss,
            "raw_similarity_loss": raw_similarity_loss,
            "cell_type_clustering_loss": cell_type_clustering_loss,
            "cross_modal_cell_type_loss": cross_modal_cell_type_loss,
            "raw_cross_modal_cell_type_loss": raw_cross_modal_cell_type_loss,
            "cn_distribution_separation_loss": cn_distribution_separation_loss_combined,
            "rna_cn_separation_loss": rna_cn_separation_loss,
            "protein_cn_separation_loss": protein_cn_separation_loss,
            "latent_distances": latent_distances.mean(),
            "extreme_alignment_percentage": extreme_alignment_percentage,
            # "num_acceptable": num_acceptable,
            "num_cells": num_cells,
            "raw_losses_for_scaling": raw_losses_for_scaling,
        }
        if (
            (to_plot or check_ilisi)
            and rna_latent_mean is not None
            and protein_latent_mean is not None
        ):
            rna_latent_mean_numpy = rna_latent_mean.detach().cpu().numpy()
            protein_latent_mean_numpy = protein_latent_mean.detach().cpu().numpy()
            # Create combined latent AnnData
            combined_latent = anndata.concat(
                [
                    AnnData(rna_latent_mean_numpy, obs=rna_vae.adata[rna_batch["labels"]].obs),
                    AnnData(
                        protein_latent_mean_numpy,
                        obs=protein_vae.adata[protein_batch["labels"]].obs,
                    ),
                ],
                join="outer",
                label="modality",
                keys=["RNA", "Protein"],
            )
            # Clear any existing neighbors data to ensure clean calculation
            (
                combined_latent.obsp.pop("connectivities", None)
                if "connectivities" in combined_latent.obsp
                else None
            )
            (
                combined_latent.obsp.pop("distances", None)
                if "distances" in combined_latent.obsp
                else None
            )
            (
                combined_latent.uns.pop("neighbors", None)
                if "neighbors" in combined_latent.uns
                else None
            )

            # Calculate neighbors with cosine metric for iLISI
            sc.pp.neighbors(combined_latent, use_rep="X", n_neighbors=15)
        # Explicitly check plot_x_times to prevent plotting when disabled
        if (
            to_plot
            and self.plot_x_times > 0
            and rna_latent_mean is not None
            and protein_latent_mean is not None
        ):
            rna_latent_mean_numpy = rna_latent_mean.detach().cpu().numpy()
            rna_latent_std_numpy = rna_latent_std.detach().cpu().numpy()
            protein_latent_mean_numpy = protein_latent_mean.detach().cpu().numpy()
            protein_latent_std_numpy = protein_latent_std.detach().cpu().numpy()

            # Use self.current_epoch for plotting as it's managed by the Trainer and more robust
            epoch_for_plot = self.current_epoch
            if global_step is not None:  # Add context if global_step is available
                self.logger_.info(
                    f"Plotting for mode '{self.mode}', epoch {epoch_for_plot}, global_step {global_step}"
                )
            else:
                self.logger_.info(f"Plotting for mode '{self.mode}', epoch {epoch_for_plot}")

            # Guard against plotting an empty or non-AnnData object
            if hasattr(combined_latent, "shape") and combined_latent.shape[0] > 0:
                # plot_latent_pca_both_modalities_cn(
                #     rna_latent_mean_numpy,
                #     protein_latent_mean_numpy,
                #     rna_vae.adata,
                #     protein_vae.adata,
                #     index_rna=rna_batch["labels"],
                #     index_prot=protein_batch["labels"],
                #     global_step=global_step,
                # )
                # plot_latent_pca_both_modalities_by_celltype(
                #     rna_vae.adata,
                #     protein_vae.adata,
                #     rna_latent_mean_numpy,
                #     protein_latent_mean_numpy,
                #     index_rna=rna_batch["labels"],
                #     index_prot=protein_batch["labels"],
                #     global_step=global_step,
                # )
                # plot_rna_protein_matching_means_and_scale(
                #     rna_latent_mean_numpy,
                #     protein_latent_mean_numpy,
                #     rna_latent_std_numpy,
                #     protein_latent_std_numpy,
                #     archetype_dis,
                #     global_step=global_step,
                # )
                # Add the new extreme archetypes alignment plot
                # plot_extreme_archetypes_alignment(
                #     rna_latent_mean,
                #     protein_latent_mean,
                #     rna_batch,
                #     protein_batch,
                #     rna_vae.adata,
                #     protein_vae.adata,
                #     rna_batch["labels"],
                #     protein_batch["labels"],
                #     global_step=global_step,
                #     alpha=0.7,
                #     use_subsample=True,
                #     mode=self.mode,
                # )
                pf.plot_pca_umap_latent_space_during_train(
                    self.mode,
                    combined_latent,
                    epoch=epoch_for_plot,  # Use robust epoch value
                    global_step=global_step,
                    plot_flag=to_plot,
                )
                # does not work well due to the zero inflated distribution
                # instead we use the decoded data for the comparison when we use
                # counterfactual generation
                # pf.reconstruction_comparison_plot(
                #     rna_vae,
                #     protein_vae,
                #     rna_batch,
                #     protein_batch,
                #     rna_inference_outputs,
                #     protein_inference_outputs,
                #     global_step=global_step,
                # )
            else:
                self.logger_.warning(
                    f"[Step {global_step}, Epoch {epoch_for_plot}] combined_latent is not plottable (e.g., empty or not AnnData). Skipping UMAP plot."
                )
                if hasattr(combined_latent, "shape"):
                    self.logger_.warning(f"combined_latent shape: {combined_latent.shape}")
                # Log details that might help diagnose why combined_latent is problematic
                if "rna_latent_mean_numpy" in locals() and "protein_latent_mean_numpy" in locals():
                    self.logger_.warning(
                        f"RNA latent_mean_numpy shape: {rna_latent_mean_numpy.shape}"
                    )
                    self.logger_.warning(
                        f"Protein latent_mean_numpy shape: {protein_latent_mean_numpy.shape}"
                    )
                if "rna_batch" in locals() and "protein_batch" in locals():
                    self.logger_.warning(
                        f"RNA batch['labels'] length: {len(rna_batch.get('labels', []))}"
                    )
                    self.logger_.warning(
                        f"Protein batch['labels'] length: {len(protein_batch.get('labels', []))}"
                    )
        if check_ilisi:
            ilisi_score = calculate_iLISI(combined_latent, "modality", plot_flag=False)
            # Validate the iLISI score
            if np.isfinite(ilisi_score) and ilisi_score > 0:
                losses["ilisi_score"] = float(ilisi_score)
                if self.to_print:
                    self.logger_.info(f"iLISI score calculated: {ilisi_score}")
            else:
                self.logger_.warning(f"Invalid iLISI score calculated: {ilisi_score}. Skipping.")

            # Calculate kBET rejection rate within each cell type cluster
            kbet_score = kbet_within_cell_types(
                combined_latent,
                label_key="CN",
                group_key="cell_types",
                rep_key="X",
                prefix="",
                to_print=self.to_print,
            )
            if np.isfinite(kbet_score):
                losses["mean_per_cell_type_cn_kbet_separation"] = float(kbet_score)
                if self.to_print:
                    self.logger_.info(
                        f"Mean kBET rejection rate across cell types: {kbet_score:.4f}"
                    )
            else:
                self.logger_.warning(f"Invalid kBET score: {kbet_score}")
        return losses
