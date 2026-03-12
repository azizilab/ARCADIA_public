"""Post-hoc analysis utilities for checkpoint loading and counterfactual generation."""

from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.metrics import pairwise_distances, r2_score

from arcadia.training.dual_vae_training_plan import DualVAETrainingPlan
from arcadia.training.utils import create_counterfactual_adata, get_latent_embedding
from arcadia.utils.logging import logger


# %%
def find_latest_checkpoint_folder():
    """Find the latest checkpoint folder in MLflow artifacts."""
    mlruns_path = Path("mlruns")
    if not mlruns_path.exists():
        logger.error("MLruns directory not found")
        return None

    # Find all experiment directories
    experiment_dirs = [d for d in mlruns_path.iterdir() if d.is_dir() and d.name.isdigit()]
    if not experiment_dirs:
        logger.error("No experiment directories found in mlruns")
        return None

    # Look for checkpoint folders in all runs
    checkpoint_folders = []
    for exp_dir in experiment_dirs:
        run_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name != ".trash"]
        for run_dir in run_dirs:
            artifacts_dir = run_dir / "artifacts"
            if artifacts_dir.exists():
                checkpoints_dir = artifacts_dir / "checkpoints"
                if checkpoints_dir.exists():
                    # Find epoch folders
                    epoch_folders = [
                        d
                        for d in checkpoints_dir.iterdir()
                        if d.is_dir() and d.name.startswith("epoch_")
                    ]
                    for epoch_folder in epoch_folders:
                        # Check if both adata files exist
                        rna_file = epoch_folder / "adata_rna.h5ad"
                        prot_file = epoch_folder / "adata_prot.h5ad"
                        if rna_file.exists() and prot_file.exists():
                            checkpoint_folders.append(epoch_folder)

    if not checkpoint_folders:
        logger.error("No valid checkpoint folders found")
        return None

    # Sort by modification time and return the latest
    latest_checkpoint = max(checkpoint_folders, key=lambda x: x.stat().st_mtime)
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


# %%
def load_checkpoint_data(checkpoint_folder):
    """Load RNA and protein AnnData objects from checkpoint folder."""
    rna_file = checkpoint_folder / "adata_rna.h5ad"
    prot_file = checkpoint_folder / "adata_prot.h5ad"

    if not rna_file.exists() or not prot_file.exists():
        raise FileNotFoundError(f"Checkpoint files not found in {checkpoint_folder}")

    logger.info(f"Loading RNA data from: {rna_file}")
    adata_rna = sc.read_h5ad(rna_file)

    logger.info(f"Loading protein data from: {prot_file}")
    adata_prot = sc.read_h5ad(prot_file)

    logger.info(
        f"✓ RNA data loaded: {adata_rna.shape} with latent dim {adata_rna.obsm['X_scVI'].shape[1]}"
    )
    logger.info(
        f"✓ Protein data loaded: {adata_prot.shape} with latent dim {adata_prot.obsm['X_scVI'].shape[1]}"
    )

    return adata_rna, adata_prot


# %%
def create_combined_latent_space(adata_rna: AnnData, adata_prot: AnnData):
    """Create combined latent space AnnData object from RNA and protein data."""
    # Extract latent embeddings
    rna_latent = adata_rna.obsm["X_scVI"]
    prot_latent = adata_prot.obsm["X_scVI"]

    # Create AnnData objects for latent spaces
    rna_latent_adata = AnnData(rna_latent)
    rna_latent_adata.obs = adata_rna.obs.copy()

    prot_latent_adata = AnnData(prot_latent)
    prot_latent_adata.obs = adata_prot.obs.copy()

    # Combine latent spaces (similar to DualVAETrainingPlan)
    combined_latent = anndata.concat(
        [rna_latent_adata, prot_latent_adata],
        join="outer",
        label="modality",
        keys=["RNA", "Protein"],
    )

    # Clear any existing neighbors data to ensure clean calculation
    for key in ["connectivities", "distances"]:
        if key in combined_latent.obsp:
            combined_latent.obsp.pop(key)
    if "neighbors" in combined_latent.uns:
        combined_latent.uns.pop("neighbors")

    # Calculate neighbors for downstream analysis
    sc.pp.neighbors(combined_latent, use_rep="X", n_neighbors=15)

    logger.info(f"✓ Combined latent space created: {combined_latent.shape}")
    logger.info(f"  - Modalities: {combined_latent.obs['modality'].value_counts().to_dict()}")

    # Check for required columns
    required_cols = ["cell_types", "CN"]
    missing_cols = [col for col in required_cols if col not in combined_latent.obs.columns]
    if missing_cols:
        logger.warning(f"Missing columns for CN iLISI calculation: {missing_cols}")
    else:
        logger.info("✓ All required columns present for CN iLISI calculation")

    return combined_latent


# %%
# Function to load models from checkpoint
def load_models_from_checkpoint(checkpoint_path, adata_rna, adata_prot, config_path=None):
    """
    Load RNA and protein VAE models from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory
        adata_rna: AnnData object for RNA data
        adata_prot: AnnData object for protein data

    Returns:
        rna_vae: Loaded RNA VAE model
        protein_vae: Loaded protein VAE model
    """
    logger.info(f"Loading models from checkpoint: {checkpoint_path}")

    # Find model configuration file
    if config_path is None:
        config_path = Path(checkpoint_path).parent / "model_config.json"

    # Create models based on configuration or defaults
    if config_path.exists():
        logger.info(f"Found model configuration at {config_path}")
        # Use DualVAETrainingPlan's create_models_from_config function
        rna_vae, protein_vae = DualVAETrainingPlan.create_models_from_config(
            config_path, adata_rna, adata_prot
        )
        logger.info("Models created successfully using create_models_from_config")

    # Set training plan class and load state dictionaries
    rna_vae._training_plan_cls = DualVAETrainingPlan
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load state dictionaries
    rna_vae.module.load_state_dict(
        torch.load(f"{checkpoint_path}/rna_vae_model.pt", map_location=device)
    )
    protein_vae.module.load_state_dict(
        torch.load(f"{checkpoint_path}/protein_vae_model.pt", map_location=device)
    )

    # Set models as trained and move to device
    rna_vae.is_trained_ = True
    protein_vae.is_trained_ = True
    rna_vae.module.to(device)
    protein_vae.module.to(device)

    logger.info(f"Models loaded successfully from {checkpoint_path}")

    return rna_vae, protein_vae


# Function to generate counterfactual data
def generate_counterfactual_data(rna_vae, protein_vae, adata_rna, adata_prot, plot_flag=False):
    """
    Generate counterfactual data using the loaded models.

    Args:
        rna_vae: RNA VAE model
        protein_vae: Protein VAE model
        adata_rna: AnnData object for RNA data
        adata_prot: AnnData object for protein data

    Returns:
        same_modal_adata_rna: RNA reconstruction (RNA→RNA encoder→RNA decoder)
        counterfactual_adata_rna: RNA counterfactual (Protein→Protein encoder→RNA decoder)
        same_modal_adata_prot: Protein reconstruction (Protein→Protein encoder→Protein decoder)
        counterfactual_adata_prot: Protein counterfactual (RNA→RNA encoder→Protein decoder)
    """
    logger.info("Generating counterfactual data...")

    # Ensure models are on the same device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Make sure models are on the correct device
    rna_vae.module.to(device)
    protein_vae.module.to(device)

    # Get latent embeddings
    rna_latent = get_latent_embedding(rna_vae, adata_rna, device=device)
    protein_latent = get_latent_embedding(protein_vae, adata_prot, device=device)
    rna_latent = AnnData(rna_latent)
    rna_latent.obs = adata_rna.obs.copy()
    protein_latent = AnnData(protein_latent)
    protein_latent.obs = adata_prot.obs.copy()
    merged_latent = sc.concat(
        [rna_latent, protein_latent], join="outer", label="modality", keys=["RNA", "Protein"]
    )
    if plot_flag:
        # use subsample if needed
        if merged_latent.n_obs > 4000:
            merged_latent_subsampled = merged_latent[
                np.random.choice(merged_latent.n_obs, 4000, replace=False)
            ]
        else:
            merged_latent_subsampled = merged_latent.copy()
        sc.pp.pca(merged_latent_subsampled)
        sc.pp.neighbors(merged_latent_subsampled)
        sc.tl.umap(merged_latent_subsampled)
        sc.pl.umap(merged_latent_subsampled, color=["cell_types", "CN", "modality"])
        sc.pl.umap(
            merged_latent_subsampled[merged_latent_subsampled.obs["modality"] == "RNA"],
            color=["cell_types", "CN"],
        )
        sc.pl.umap(
            merged_latent_subsampled[merged_latent_subsampled.obs["modality"] == "Protein"],
            color=["cell_types", "CN"],
        )
    logger.info(
        f"Latent embeddings extracted: RNA={rna_latent.shape}, Protein={protein_latent.shape}"
    )

    # Create a DualVAETrainingPlan instance to use its create_counterfactual_adata method
    # Initialize DualVAETrainingPlan with the correct parameters
    training_plan = DualVAETrainingPlan(
        rna_vae.module,  # First argument is the RNA module
        rna_vae=rna_vae,  # Pass the full RNA VAE object as kwarg
        protein_vae=protein_vae,  # Pass the full protein VAE object as kwarg
        # Add required parameters with default values
        check_val_every_n_epoch=10,
        max_epochs=500,
        batch_size=1024,
        train_size=0.9,
        print_every_n_epoch=5,
        plot_x_times=5,
        lr=0.001,
        weight_decay=0.0,
        similarity_weight=0.0,
        similarity_dynamic=False,
        cell_type_clustering_weight=1.0,
        cross_modal_cell_type_weight=1.0,
        contrastive_weight=0.0,
        matching_weight=1.0,
        rna_recon_weight=1.0,
        prot_recon_weight=1.0,
        log_file_path=None,
    )

    # Calculate constant protein library size for counterfactual generation
    # Make sure _scvi_library_size exists in both adata_rna and adata_prot.obs
    if "_scvi_library_size" not in adata_rna.obs:
        logger.info("_scvi_library_size not found in adata_rna.obs, calculating it now")
        # Calculate library size (sum of counts per cell)
        if issparse(adata_rna.X):
            library_size_rna = np.array(adata_rna.X.sum(axis=1)).flatten()
        else:
            library_size_rna = adata_rna.X.sum(axis=1)
            library_size_rna = np.asarray(library_size_rna).flatten()

        # Calculate log library size
        log_library_size_rna = np.log(library_size_rna)
        adata_rna.obs["_scvi_library_size"] = log_library_size_rna
        logger.info("Added _scvi_library_size to adata_rna.obs")

    if "_scvi_library_size" not in adata_prot.obs:
        logger.info("_scvi_library_size not found in adata_prot.obs, calculating it now")
        # Calculate library size (sum of counts per cell)
        if issparse(adata_prot.X):
            library_size_protein = np.array(adata_prot.X.sum(axis=1)).flatten()
        else:
            library_size_protein = adata_prot.X.sum(axis=1)
            library_size_protein = np.asarray(library_size_protein).flatten()

        # Calculate log library size
        log_library_size_protein = np.log(library_size_protein)
        adata_prot.obs["_scvi_library_size"] = log_library_size_protein
        logger.info("Added _scvi_library_size to adata_prot.obs")

    protein_lib_sizes = adata_prot.obs["_scvi_library_size"].values
    exp_protein_lib_sizes = np.exp(protein_lib_sizes)  # Convert from log space

    # Use median as constant library size
    constant_protein_lib_size = np.median(exp_protein_lib_sizes)
    constant_protein_lib_size_log = np.log(constant_protein_lib_size)

    logger.info(
        f"Using constant protein library size: {constant_protein_lib_size:.2f} (log: {constant_protein_lib_size_log:.4f})"
    )

    # DIAGNOSTIC: Verify checkpoint data normalization state
    logger.info("=== DIAGNOSTIC: Checking data normalization state ===")
    rna_lib_sums = np.array(adata_rna.X.sum(axis=1)).flatten()
    prot_lib_sums = np.array(adata_prot.X.sum(axis=1)).flatten()
    logger.info(
        f"RNA library sums - mean: {rna_lib_sums.mean():.2f}, std: {rna_lib_sums.std():.2f}, min: {rna_lib_sums.min():.2f}, max: {rna_lib_sums.max():.2f}"
    )
    logger.info(
        f"Protein library sums - mean: {prot_lib_sums.mean():.2f}, std: {prot_lib_sums.std():.2f}, min: {prot_lib_sums.min():.2f}, max: {prot_lib_sums.max():.2f}"
    )
    logger.info(f"Expected library size from training: {constant_protein_lib_size:.2f}")

    # Check if data is already normalized (library sums should be close to constant_protein_lib_size)
    rna_is_normalized = np.abs(rna_lib_sums.mean() - constant_protein_lib_size) < 1000
    prot_is_normalized = np.abs(prot_lib_sums.mean() - constant_protein_lib_size) < 1000
    logger.info(f"RNA appears normalized: {rna_is_normalized}")
    logger.info(f"Protein appears normalized: {prot_is_normalized}")

    if not rna_is_normalized or not prot_is_normalized:
        logger.warning("WARNING: Checkpoint data may not be properly normalized!")
        logger.warning("This could indicate the checkpoint was saved with unnormalized data.")
    else:
        logger.info("✓ Checkpoint data is already normalized as expected")

    # NO NORMALIZATION - data is already normalized in checkpoint
    # Commenting out to avoid double normalization:
    # sc.pp.normalize_total(adata_rna_normalized, target_sum=constant_protein_lib_size)
    # sc.pp.normalize_total(adata_prot_normalized, target_sum=constant_protein_lib_size)

    logger.info("Using checkpoint data as-is (already normalized during training)")

    # Ensure models are in eval mode for consistent inference (matches get_latent_embedding)
    rna_vae.module.eval()
    protein_vae.module.eval()
    logger.info("Models set to eval mode for counterfactual generation")

    # Set required attributes for counterfactual generation
    training_plan = training_plan.to(device)
    training_plan.rna_vae = rna_vae
    training_plan.protein_vae = protein_vae
    training_plan.constant_protein_lib_size = constant_protein_lib_size
    training_plan.constant_protein_lib_size_log = constant_protein_lib_size_log

    # Generate counterfactual data using checkpoint data directly (no additional normalization)
    # Note: create_counterfactual_adata returns in this order:
    # (counterfactual_adata_rna, counterfactual_adata_protein, same_modal_adata_rna, same_modal_adata_protein)
    print(plot_flag)
    print(plot_flag)
    print(plot_flag)
    print(plot_flag)
    print(plot_flag)
    (
        counterfactual_adata_rna,
        counterfactual_adata_prot,
        same_modal_adata_rna,
        same_modal_adata_prot,
    ) = create_counterfactual_adata(
        training_plan,
        adata_rna,  # Use checkpoint data as-is (already normalized)
        adata_prot,  # Use checkpoint data as-is (already normalized)
        rna_latent,
        protein_latent,
        plot_flag=plot_flag,
        skip_normalization=True,  # Skip normalization since checkpoint data is already normalized
    )
    logger.info("Counterfactual data generated successfully")
    logger.info(f"Same modal RNA: {same_modal_adata_rna.shape}")
    logger.info(f"Counterfactual RNA: {counterfactual_adata_rna.shape}")
    logger.info(f"Same modal Protein: {same_modal_adata_prot.shape}")
    logger.info(f"Counterfactual Protein: {counterfactual_adata_prot.shape}")

    return (
        same_modal_adata_rna,
        counterfactual_adata_rna,
        same_modal_adata_prot,
        counterfactual_adata_prot,
    )


# %%
def generate_counterfactual_distributions(rna_vae, protein_vae, adata_rna, adata_prot):
    """
    Generate counterfactual distributions by returning the full decoder outputs instead of just means.
    This allows sampling from the distributions later.

    Returns:
        dict with keys:
        - 'rna_same_modal_dist': RNA reconstruction distribution (RNA→RNA encoder→RNA decoder)
        - 'rna_counterfactual_dist': RNA counterfactual distribution (Protein→Protein encoder→RNA decoder)
        - 'protein_same_modal_dist': Protein reconstruction distribution (Protein→Protein encoder→Protein decoder)
        - 'protein_counterfactual_dist': Protein counterfactual distribution (RNA→RNA encoder→Protein decoder)
    """
    logger.info("Generating counterfactual distributions (full decoder outputs)...")

    # Ensure models are on the same device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    rna_vae.module.to(device)
    protein_vae.module.to(device)

    # Calculate constant protein library size
    if "_scvi_library_size" not in adata_prot.obs:
        logger.info("_scvi_library_size not found in adata_prot.obs, calculating it now")
        if issparse(adata_prot.X):
            library_size_protein = np.array(adata_prot.X.sum(axis=1)).flatten()
        else:
            library_size_protein = adata_prot.X.sum(axis=1)
            library_size_protein = np.asarray(library_size_protein).flatten()

        log_library_size_protein = np.log(library_size_protein)
        adata_prot.obs["_scvi_library_size"] = log_library_size_protein
        logger.info("Added _scvi_library_size to adata_prot.obs")

    protein_lib_sizes = adata_prot.obs["_scvi_library_size"].values
    exp_protein_lib_sizes = np.exp(protein_lib_sizes)
    constant_protein_lib_size = np.median(exp_protein_lib_sizes)
    constant_protein_lib_size_log = np.log(constant_protein_lib_size)

    logger.info(
        f"Using constant protein library size: {constant_protein_lib_size:.2f} (log: {constant_protein_lib_size_log:.4f})"
    )

    # Normalize data to constant library size
    adata_rna_normalized = adata_rna.copy()
    adata_prot_normalized = adata_prot.copy()

    sc.pp.normalize_total(adata_rna_normalized, target_sum=constant_protein_lib_size)
    adata_rna_normalized.X = adata_rna_normalized.X.astype(np.int32)

    sc.pp.normalize_total(adata_prot_normalized, target_sum=constant_protein_lib_size)
    adata_prot_normalized.X = adata_prot_normalized.X.astype(np.int32)

    logger.info(f"✓ Both datasets normalized to constant library size: {constant_protein_lib_size}")

    # Prepare constant library size tensor
    constant_protein_lib_size_log_tensor = torch.tensor(
        constant_protein_lib_size_log, dtype=torch.float32, device=device
    )

    # === RNA SAME-MODAL RECONSTRUCTION ===
    logger.info("Generating RNA same-modal reconstruction distributions...")

    rna_X = adata_rna_normalized.X
    if issparse(rna_X):
        rna_X = rna_X.toarray()
    rna_X_tensor = torch.tensor(rna_X, dtype=torch.float32, device=device)

    rna_batch_codes = adata_rna_normalized.obs["batch"].cat.codes.values
    rna_batch_tensor = torch.tensor(rna_batch_codes, dtype=torch.long, device=device).unsqueeze(1)

    n_rna_cells = rna_X_tensor.shape[0]
    constant_lib_tensor_rna = constant_protein_lib_size_log_tensor.repeat(n_rna_cells, 1)

    with torch.no_grad():
        rna_batch_for_encoding = {
            "X": rna_X_tensor,
            "batch": rna_batch_tensor,
            "library": constant_lib_tensor_rna,
            "labels": torch.arange(n_rna_cells, device=device).unsqueeze(1),
        }
        rna_encoded_outputs, rna_decoded_outputs, _ = rna_vae.module(rna_batch_for_encoding)
        rna_encoded_latent = rna_encoded_outputs["qz"].mean
        rna_same_modal_dist = rna_decoded_outputs["px"]  # Full distribution

    logger.info(f"✓ RNA same-modal distribution generated: {type(rna_same_modal_dist)}")

    # === PROTEIN SAME-MODAL RECONSTRUCTION ===
    logger.info("Generating protein same-modal reconstruction distributions...")

    protein_X = adata_prot_normalized.X
    if issparse(protein_X):
        protein_X = protein_X.toarray()
    protein_X_tensor = torch.tensor(protein_X, dtype=torch.float32, device=device)

    protein_batch_codes = adata_prot_normalized.obs["batch"].cat.codes.values
    protein_batch_tensor = torch.tensor(
        protein_batch_codes, dtype=torch.long, device=device
    ).unsqueeze(1)

    n_protein_cells = protein_X_tensor.shape[0]
    constant_lib_tensor_protein = constant_protein_lib_size_log_tensor.repeat(n_protein_cells, 1)

    with torch.no_grad():
        protein_batch_for_encoding = {
            "X": protein_X_tensor,
            "batch": protein_batch_tensor,
            "library": constant_lib_tensor_protein,
            "labels": torch.arange(n_protein_cells, device=device).unsqueeze(1),
        }
        protein_encoded_outputs, protein_decoded_outputs, _ = protein_vae.module(
            protein_batch_for_encoding
        )
        protein_encoded_latent = protein_encoded_outputs["qz"].mean
        protein_same_modal_dist = protein_decoded_outputs["px"]  # Full distribution

    logger.info(f"✓ Protein same-modal distribution generated: {type(protein_same_modal_dist)}")

    # === COUNTERFACTUAL PROTEIN (RNA → Protein decoder) ===
    logger.info("Generating counterfactual protein distributions (RNA → Protein decoder)...")

    # Use most common protein batch for RNA cells
    most_common_protein_batch = torch.mode(protein_batch_tensor.flatten())[0].item()
    protein_batch_tensor_for_rna = torch.full(
        (n_rna_cells, 1), most_common_protein_batch, dtype=torch.long, device=device
    )
    constant_lib_tensor_for_rna_cells = constant_protein_lib_size_log_tensor.repeat(n_rna_cells, 1)

    with torch.no_grad():
        counterfactual_protein_outputs = protein_vae.module.generative(
            z=rna_encoded_latent,
            library=constant_lib_tensor_for_rna_cells,
            batch_index=protein_batch_tensor_for_rna,
        )
        protein_counterfactual_dist = counterfactual_protein_outputs["px"]  # Full distribution

    logger.info(
        f"✓ Protein counterfactual distribution generated: {type(protein_counterfactual_dist)}"
    )

    # === COUNTERFACTUAL RNA (Protein → RNA decoder) ===
    logger.info("Generating counterfactual RNA distributions (Protein → RNA decoder)...")

    # Use most common RNA batch for protein cells
    most_common_rna_batch = torch.mode(rna_batch_tensor.flatten())[0].item()
    rna_batch_tensor_for_protein = torch.full(
        (n_protein_cells, 1), most_common_rna_batch, dtype=torch.long, device=device
    )
    constant_lib_tensor_for_protein_cells = constant_protein_lib_size_log_tensor.repeat(
        n_protein_cells, 1
    )

    with torch.no_grad():
        counterfactual_rna_outputs = rna_vae.module.generative(
            z=protein_encoded_latent,
            library=constant_lib_tensor_for_protein_cells,
            batch_index=rna_batch_tensor_for_protein,
        )
        rna_counterfactual_dist = counterfactual_rna_outputs["px"]  # Full distribution

    logger.info(f"✓ RNA counterfactual distribution generated: {type(rna_counterfactual_dist)}")

    # Return all distributions
    distributions = {
        "rna_same_modal_dist": rna_same_modal_dist,
        "rna_counterfactual_dist": rna_counterfactual_dist,
        "protein_same_modal_dist": protein_same_modal_dist,
        "protein_counterfactual_dist": protein_counterfactual_dist,
        "adata_rna_normalized": adata_rna_normalized,
        "adata_prot_normalized": adata_prot_normalized,
        "constant_protein_lib_size": constant_protein_lib_size,
        "constant_protein_lib_size_log": constant_protein_lib_size_log,
    }

    logger.info("✓ All counterfactual distributions generated successfully")
    logger.info("Available distribution attributes:")
    for key, dist in distributions.items():
        if hasattr(dist, "mu"):
            logger.info(f"  {key}: has .mu (shape: {dist.mu.shape})")
        if hasattr(dist, "zi_logits"):
            logger.info(f"  {key}: has .zi_logits (shape: {dist.zi_logits.shape})")
        if hasattr(dist, "loc"):
            logger.info(f"  {key}: has .loc (shape: {dist.loc.shape})")
        if hasattr(dist, "scale"):
            logger.info(f"  {key}: has .scale (shape: {dist.scale.shape})")

    return distributions


# %%
def sample_from_counterfactual_distributions(
    distributions, n_samples=10, batch_size=10, use_float16=False
):
    """
    Sample from counterfactual distributions to generate multiple realizations with memory optimization.

    Args:
        distributions: Output from generate_counterfactual_distributions()
        n_samples: Number of samples to draw from each distribution
        batch_size: Process samples in batches to save memory (default: 10)
        use_float16: Use float16 to save memory (default: False)

    Returns:
        dict with keys:
        - 'rna_same_modal_samples': (n_samples, n_rna_cells, n_rna_genes)
        - 'rna_counterfactual_samples': (n_samples, n_protein_cells, n_rna_genes)
        - 'protein_same_modal_samples': (n_samples, n_protein_cells, n_proteins)
        - 'protein_counterfactual_samples': (n_samples, n_rna_cells, n_proteins)
        - 'adata_rna_normalized': AnnData object with normalized RNA data
        - 'adata_prot_normalized': AnnData object with normalized protein data
        - 'constant_protein_lib_size': Constant protein library size
        - 'constant_protein_lib_size_log': Constant protein library size in log space
    """
    logger.info(f"Sampling {n_samples} realizations from each counterfactual distribution...")
    logger.info(f"Using batch size: {batch_size}, float16: {use_float16}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    samples = {}

    # Determine data type
    dtype = torch.float16 if use_float16 else torch.float32
    np_dtype = np.float16 if use_float16 else np.float32

    def sample_from_distribution(dist, dist_name, n_samples, batch_size):
        """Helper function to sample from a distribution in batches."""
        logger.info(f"Sampling {dist_name}...")

        # Get distribution shape
        if hasattr(dist, "mu"):
            shape = dist.mu.shape
        elif hasattr(dist, "loc"):
            shape = dist.loc.shape
        else:
            raise ValueError(f"Unknown distribution type for {dist_name}")

        # Pre-allocate result array
        result_shape = (n_samples,) + shape
        total_elements = np.prod(result_shape)

        logger.info(f"  - Distribution shape: {shape}")
        logger.info(f"  - Result shape: {result_shape}")
        logger.info(f"  - Total elements: {total_elements:,}")

        # Check memory requirements (rough estimate)
        bytes_per_element = 2 if use_float16 else 4
        estimated_memory_gb = (total_elements * bytes_per_element) / (1024**3)
        logger.info(f"  - Estimated memory: {estimated_memory_gb:.2f} GB")

        if estimated_memory_gb > 8:  # If > 8GB, warn user
            logger.warning(f"Large memory requirement detected: {estimated_memory_gb:.2f} GB")
            logger.warning("Consider reducing n_samples or using smaller batch_size")

        # Sample in batches
        all_samples = []
        n_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            current_batch_size = end_idx - start_idx

            logger.info(
                f"  - Processing batch {batch_idx + 1}/{n_batches} (samples {start_idx}-{end_idx-1})"
            )

            batch_samples = []

            for i in range(current_batch_size):
                with torch.no_grad():  # Ensure no gradients
                    if hasattr(dist, "mu") and hasattr(dist, "zi_logits"):
                        # Zero-inflated negative binomial sampling
                        mu = dist.mu.to(dtype)
                        zi_logits = dist.zi_logits.to(dtype)
                        zi_probs = torch.sigmoid(zi_logits)

                        # Sample from Bernoulli for zero-inflation
                        zi_samples = torch.bernoulli(zi_probs)

                        # Sample from negative binomial (using Poisson approximation)
                        nb_samples = torch.poisson(mu)

                        # Combine: if zi_sample=1, then 0, else nb_sample
                        sample = (1 - zi_samples) * nb_samples

                    elif hasattr(dist, "loc") and hasattr(dist, "scale"):
                        # Normal distribution sampling
                        loc = dist.loc.to(dtype)
                        scale = dist.scale.to(dtype)
                        sample = torch.normal(loc, scale)
                        sample = torch.clamp(sample, min=0)  # Ensure non-negative

                    else:
                        # Fallback: use mu if available
                        if hasattr(dist, "mu"):
                            mu = dist.mu.to(dtype)
                            sample = torch.poisson(mu)
                        else:
                            raise ValueError(f"Unknown distribution type for {dist_name}")

                    # Convert to CPU and numpy immediately to free GPU memory
                    sample_np = sample.detach().cpu().numpy().astype(np_dtype)
                    batch_samples.append(sample_np)

                    # Clear GPU cache periodically
                    if torch.cuda.is_available() and i % 5 == 0:
                        torch.cuda.empty_cache()

            # Stack batch samples and add to results
            batch_array = np.stack(batch_samples, axis=0)
            all_samples.append(batch_array)

            # Clear batch samples from memory
            del batch_samples
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"  - Batch {batch_idx + 1} completed, shape: {batch_array.shape}")

        # Concatenate all batches
        logger.info(f"  - Concatenating {len(all_samples)} batches...")
        final_samples = np.concatenate(all_samples, axis=0)

        # Clear intermediate arrays
        del all_samples

        logger.info(f"✓ {dist_name} samples shape: {final_samples.shape}")
        return final_samples

    # === SAMPLE ALL DISTRIBUTIONS ===
    samples["rna_same_modal_samples"] = sample_from_distribution(
        distributions["rna_same_modal_dist"], "RNA same-modal", n_samples, batch_size
    )

    samples["rna_counterfactual_samples"] = sample_from_distribution(
        distributions["rna_counterfactual_dist"], "RNA counterfactual", n_samples, batch_size
    )

    samples["protein_same_modal_samples"] = sample_from_distribution(
        distributions["protein_same_modal_dist"], "Protein same-modal", n_samples, batch_size
    )

    samples["protein_counterfactual_samples"] = sample_from_distribution(
        distributions["protein_counterfactual_dist"],
        "Protein counterfactual",
        n_samples,
        batch_size,
    )

    # Add metadata
    samples["n_samples"] = n_samples
    samples["adata_rna_normalized"] = distributions["adata_rna_normalized"]
    samples["adata_prot_normalized"] = distributions["adata_prot_normalized"]

    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("✓ All sampling completed successfully")
    logger.info("Sample shapes summary:")
    for key, value in samples.items():
        if isinstance(value, np.ndarray):
            memory_gb = (value.nbytes) / (1024**3)
            logger.info(f"  {key}: {value.shape} ({memory_gb:.2f} GB)")

    return samples


def compare_same_modal_vs_counterfactual_rna_differential_expression(
    same_modal_adata_rna, counterfactual_adata_rna
):
    """
    Compare same modal vs counterfactual RNA differential expression.
    Args:
        same_modal_adata_rna: AnnData object for same modal RNA data
        counterfactual_adata_rna: AnnData object for counterfactual RNA data
    Returns:
        r2_scores: list of R2 scores
        r2_scores_shuffled: list of shuffled R2 scores
    """
    r2_scores = []
    r2_scores_shuffled = []
    for cell_type in same_modal_adata_rna.obs["cell_types"].unique():
        same_modal_current = same_modal_adata_rna[
            same_modal_adata_rna.obs["cell_types"] == cell_type
        ].copy()
        counterfactual_current = counterfactual_adata_rna[
            counterfactual_adata_rna.obs["cell_types"] == cell_type
        ].copy()

        same_cn_counts = same_modal_current.obs["CN"].value_counts()
        counter_cn_counts = counterfactual_current.obs["CN"].value_counts()

        # Filter out CN groups with < 2 cells
        valid_cns_same = same_cn_counts[same_cn_counts >= 2].index
        valid_cns_counter = counter_cn_counts[counter_cn_counts >= 2].index
        valid_cns = list(set(valid_cns_same).intersection(set(valid_cns_counter)))

        logger.info(f"\n{cell_type}: {len(valid_cns)} valid CN groups: {sorted(valid_cns)}")

        if len(valid_cns) < 2:
            logger.info(f"Skipping {cell_type}: not enough CN groups with sufficient cells")
            continue

        # Filter to only valid CNs
        same_modal_current = same_modal_current[same_modal_current.obs["CN"].isin(valid_cns)].copy()
        counterfactual_current = counterfactual_current[
            counterfactual_current.obs["CN"].isin(valid_cns)
        ].copy()

        # Log normalize
        sc.pp.log1p(same_modal_current)
        sc.pp.log1p(counterfactual_current)

        # Run DE for same modal
        sc.tl.rank_genes_groups(
            same_modal_current,
            groupby="CN",
            method="wilcoxon",
            pts=True,
            tie_correct=True,
        )

        # Run DE for counterfactual
        sc.tl.rank_genes_groups(
            counterfactual_current,
            groupby="CN",
            method="wilcoxon",
            pts=True,
            tie_correct=True,
        )

        # Collect significant genes (p < 0.005) from BOTH
        same_de_results = same_modal_current.uns["rank_genes_groups"]
        counter_de_results = counterfactual_current.uns["rank_genes_groups"]

        # Track which genes are significant in both same and counterfactual for each CN group
        same_sig_genes_per_cn = {}
        counter_sig_genes_per_cn = {}
        same_logfc_dict = {}
        counter_logfc_dict = {}

        for group in same_modal_current.obs["CN"].unique():
            # Same modal
            same_padj = same_de_results["pvals_adj"][group]
            same_genes = same_de_results["names"][group]
            same_logfc = same_de_results["logfoldchanges"][group]
            same_sig_mask = same_padj < 0.005

            same_sig_genes_per_cn[group] = set(same_genes[same_sig_mask])

            for gene, lfc in zip(same_genes[same_sig_mask], same_logfc[same_sig_mask]):
                if gene not in same_logfc_dict:
                    same_logfc_dict[gene] = {}
                same_logfc_dict[gene][group] = lfc

        for group in counterfactual_current.obs["CN"].unique():
            # Counterfactual
            counter_padj = counter_de_results["pvals_adj"][group]
            counter_genes = counter_de_results["names"][group]
            counter_logfc = counter_de_results["logfoldchanges"][group]
            counter_sig_mask = counter_padj < 0.005

            counter_sig_genes_per_cn[group] = set(counter_genes[counter_sig_mask])

            for gene, lfc in zip(counter_genes[counter_sig_mask], counter_logfc[counter_sig_mask]):
                if gene not in counter_logfc_dict:
                    counter_logfc_dict[gene] = {}
                counter_logfc_dict[gene][group] = lfc

        # Only include genes that are significant in BOTH same and counterfactual for at least one CN group
        genes_sig_in_both = set()
        for cn_group in same_modal_current.obs["CN"].unique():
            same_sig = same_sig_genes_per_cn.get(cn_group, set())
            counter_sig = counter_sig_genes_per_cn.get(cn_group, set())
            overlap = same_sig.intersection(counter_sig)
            genes_sig_in_both.update(overlap)
            logger.info(
                f"  CN {cn_group}: {len(same_sig)} sig genes in same, {len(counter_sig)} in counter, {len(overlap)} overlap"
            )

        logger.info(f"{cell_type}: Total genes significant in both: {len(genes_sig_in_both)}")

        # Create plot data: for each gene, get logfc from same and counterfactual
        plot_data = []
        for gene in genes_sig_in_both:
            for cn_group in same_modal_current.obs["CN"].unique():
                same_lfc = same_logfc_dict.get(gene, {}).get(cn_group, 0)
                counter_lfc = counter_logfc_dict.get(gene, {}).get(cn_group, 0)
                plot_data.append(
                    {
                        "gene": gene,
                        "CN": cn_group,
                        "same_logfc": same_lfc,
                        "counter_logfc": counter_lfc,
                    }
                )

        plot_df = pd.DataFrame(plot_data)

        if len(plot_df) == 0:
            logger.info(
                f"{cell_type}: Skipping - no genes significant in both same and counterfactual"
            )
            continue

        # Sort by same_logfc for first CN group to order genes consistently
        if len(plot_df) > 0:
            logger.info(f"{cell_type}: Generated plot with {len(plot_df)} data points")
            first_cn = same_modal_current.obs["CN"].unique()[0]
            gene_order = (
                plot_df[plot_df["CN"] == first_cn]
                .sort_values("same_logfc", ascending=False)["gene"]
                .values
            )

            r2 = r2_score(plot_df["same_logfc"], plot_df["counter_logfc"])
            r2_scores.append(r2)
            logger.info(f"\n{cell_type} - R2: {r2:.4f}")
            # Compute baseline R2 by randomly shuffling the CN-gene associations
            shuffled_plot_df = plot_df.copy()
            shuffled_plot_df["counter_logfc"] = np.random.permutation(
                shuffled_plot_df["counter_logfc"].values
            )

            r2_shuffled = r2_score(
                shuffled_plot_df["same_logfc"], shuffled_plot_df["counter_logfc"]
            )
            r2_scores_shuffled.append(r2_shuffled)
            logger.info(f"{cell_type} - Shuffled R2: {r2_shuffled:.4f}")

            # Plot: x-axis is same modal logfc, y-axis is counterfactual logfc
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

            cn_groups = same_modal_current.obs["CN"].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(cn_groups)))

            # Plot 1: Real data
            for idx, cn_group in enumerate(cn_groups):
                cn_data = plot_df[plot_df["CN"] == cn_group]
                ax1.scatter(
                    cn_data["same_logfc"],
                    cn_data["counter_logfc"],
                    label=f"CN {cn_group}",
                    alpha=0.6,
                    s=100,
                    color=colors[idx],
                )

            max_val = max(plot_df["same_logfc"].abs().max(), plot_df["counter_logfc"].abs().max())
            ax1.plot([-max_val, max_val], [-max_val, max_val], "k--", alpha=0.3, linewidth=1)
            ax1.set_xlabel("Same Modal Log Fold Change", fontsize=12)
            ax1.set_ylabel("Counterfactual Log Fold Change", fontsize=12)
            ax1.set_title(f"Real CN Labels\nR2={r2:.4f}", fontsize=14)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color="k", linewidth=0.5)
            ax1.axvline(x=0, color="k", linewidth=0.5)
            ax1.set_xscale("symlog", linthresh=0.1)
            ax1.set_yscale("symlog", linthresh=0.1)

            # Plot 2: Shuffled data
            for idx, cn_group in enumerate(cn_groups):
                cn_data = shuffled_plot_df[shuffled_plot_df["CN"] == cn_group]
                ax2.scatter(
                    cn_data["same_logfc"],
                    cn_data["counter_logfc"],
                    label=f"CN {cn_group}",
                    alpha=0.6,
                    s=100,
                    color=colors[idx],
                )

            ax2.plot([-max_val, max_val], [-max_val, max_val], "k--", alpha=0.3, linewidth=1)
            ax2.set_xlabel("Same Modal Log Fold Change", fontsize=12)
            ax2.set_ylabel("Counterfactual Log Fold Change (Shuffled)", fontsize=12)
            ax2.set_title(f"Shuffled CN Labels (Baseline)\nR2={r2_shuffled:.4f}", fontsize=14)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color="k", linewidth=0.5)
            ax2.axvline(x=0, color="k", linewidth=0.5)
            ax2.set_xscale("symlog", linthresh=0.1)
            ax2.set_yscale("symlog", linthresh=0.1)

            fig.suptitle(
                f"DE Comparison: Same vs Counterfactual RNA - Cell Type: {cell_type}",
                fontsize=16,
                y=1.02,
            )
            plt.tight_layout()
            plt.show()

    # Print average R2 across all cell types
    if len(r2_scores) > 0:
        avg_r2 = np.mean(r2_scores)
        avg_r2_shuffled = np.mean(r2_scores_shuffled) if len(r2_scores_shuffled) > 0 else 0
        logger.info(f"\n{'='*60}")
        logger.info(f"Average R2 across all cell types (Real): {avg_r2:.4f}")
        logger.info(f"Average R2 across all cell types (Shuffled): {avg_r2_shuffled:.4f}")
        if avg_r2_shuffled != 0:
            improvement = ((avg_r2 - avg_r2_shuffled) / abs(avg_r2_shuffled)) * 100
            logger.info(f"Improvement over random: {improvement:.2f}%")
        logger.info(f"{'='*60}")


def assign_rna_cn_from_protein(adata_rna: sc.AnnData, adata_prot: sc.AnnData, latent_key="X_scVI"):
    """
    Assign CN to each RNA cell based on the closest protein cell in the latent space.
    Args:
        adata_rna: AnnData object with RNA data
        adata_prot: AnnData object with protein data

    Returns:
        adata_rna: AnnData object with assigned CN
    """
    if latent_key == "X":
        rna_latent = adata_rna.X
        prot_latent = adata_prot.X
    else:
        rna_latent = adata_rna.obsm[latent_key]
        prot_latent = adata_prot.obsm[latent_key]
    rna_names = np.array(adata_rna.obs_names)
    np.array(adata_prot.obs_names)
    prot_cn_labels = adata_prot.obs["CN"]

    # Compute pairwise distances (each RNA cell to all protein cells)
    dists = pairwise_distances(rna_latent, prot_latent, metric="euclidean")

    # For each RNA cell, find 5 nearest protein neighbors
    k = 15
    k_nearest_prot_idx = np.argsort(dists, axis=1)[:, :k]  # Shape: (n_rna, k)

    # Calculate CN label proportions in protein data
    prot_cn_counts = prot_cn_labels.value_counts()
    prot_cn_proportions = prot_cn_counts / len(prot_cn_labels)
    logger.info(f"Protein CN proportions:\n{prot_cn_proportions}")
    cn_weights = {}
    for cn_label in prot_cn_proportions.index:
        cn_weights[cn_label] = 1.0 / prot_cn_proportions[cn_label]
    logger.info(f"CN rarity weights:\n{cn_weights}")
    rna_cn_from_protein = pd.Series(adata_rna.obs["CN"], index=adata_rna.obs_names)
    for rna_idx in range(len(rna_names)):
        # Get k nearest protein neighbors
        neighbor_prot_indices = k_nearest_prot_idx[rna_idx]
        neighbor_cn_labels = prot_cn_labels.iloc[neighbor_prot_indices]
        cn_scores = {}
        for cn_label in neighbor_cn_labels:
            if cn_label not in cn_scores:
                cn_scores[cn_label] = 0.0
            cn_scores[cn_label] += cn_weights[cn_label]

        # Assign the CN with the highest score
        best_cn = max(cn_scores, key=cn_scores.get)
        rna_cn_from_protein[rna_names[rna_idx]] = best_cn

    adata_rna.obs["CN"] = rna_cn_from_protein.loc[adata_rna.obs_names].values
    return adata_rna, dists


def align_rna_prot_indices(adata_rna: sc.AnnData, adata_prot: sc.AnnData, num_protein_cells=None):
    """
    Align RNA and protein AnnData objects by their indices.

    If more than 2000 cells share the same indices, align them directly.
    Otherwise, subsample and sort both datasets independently.

    Args:
        adata_rna: AnnData object with RNA data
        adata_prot: AnnData object with protein data
        num_protein_cells: Number of protein cells to subsample (if not matching indices)

    Returns:
        adata_rna: Aligned RNA AnnData object
        adata_prot: Aligned protein AnnData object
        common_index: Set of common indices (empty set if not matching)
    """
    if (
        adata_rna.obs_names.isin(adata_prot.obs_names).sum() > 2000
        and adata_rna.uns["dataset_name"] == "cite_seq"
    ):  # if we are dealing with adata with the same index
        adata_prot = adata_prot[adata_prot.obs_names.isin(adata_rna.obs_names)]
        adata_rna = adata_rna[adata_rna.obs_names.isin(adata_prot.obs_names)]
        common_index = set(adata_prot.obs_names).intersection(set(adata_rna.obs_names))
        if len(common_index) != len(adata_rna.obs_names) or len(common_index) != len(
            adata_prot.obs_names
        ):
            raise ValueError("Mismatched indices between RNA and protein data")
        adata_rna.obs["CN"] = adata_prot.obs["CN"].loc[adata_rna.obs_names].values
        sorted_index = (
            adata_prot.obs.reset_index()
            .sort_values(["cell_types", "CN", "index"])
            .set_index("index")
            .index
        )
        adata_rna = adata_rna[sorted_index, :].copy()
        adata_prot = adata_prot[sorted_index, :].copy()
        cn_accuracy = (
            adata_rna.obs["CN"].astype(str) == adata_prot.obs["CN"].astype(str)
        ).sum() / len(common_index)
        adata_rna.obs["CN_matched"] = adata_rna.obs["CN"].astype(str) == adata_prot.obs[
            "CN"
        ].astype(str)
        adata_prot.obs["CN_matched"] = adata_prot.obs["CN"].astype(str) == adata_rna.obs[
            "CN"
        ].astype(str)
        if adata_rna.obs.index.equals(adata_prot.obs.index):
            logger.info("RNA and protein data have the same indices")
            logger.info(f"CN accuracy: {cn_accuracy:.4f}")
        else:
            raise ValueError("RNA and protein data have different indices")
        print("the indices are the same")
    else:
        if num_protein_cells is not None:
            sc.pp.subsample(adata_prot, n_obs=min(num_protein_cells, adata_prot.n_obs))
        sorted_index_prot = (
            adata_prot.obs.reset_index()
            .sort_values(["cell_types", "CN", "index"])
            .set_index("index")
            .index
        )
        sorted_index_rna = (
            adata_rna.obs.reset_index()
            .sort_values(["cell_types", "CN", "index"])
            .set_index("index")
            .index
        )
        adata_rna = adata_rna[sorted_index_rna, :].copy()
        adata_prot = adata_prot[sorted_index_prot, :].copy()
        common_index = set()
        logger.info("RNA and protein data have different indices - sorted independently")

    return adata_rna, adata_prot, common_index
