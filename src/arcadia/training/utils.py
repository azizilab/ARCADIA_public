"""Training utilities."""

import gc
import os
import time
import traceback
import warnings

import anndata as ad
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import psutil
import scanpy as sc
import scipy
import seaborn as sns
import torch
from anndata import AnnData
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from scipy.stats import nbinom, norm, poisson
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from umap import UMAP

from arcadia.plotting.archetypes import plot_archetype_embedding
from arcadia.plotting.general import safe_mlflow_log_figure
from arcadia.plotting.training import (
    plot_all_fits_for_gene,
    plot_cell_type_distributions,
    plot_combined_latent_space,
    plot_latent_pca_both_modalities_by_celltype,
    plot_latent_pca_both_modalities_cn,
    plot_normalized_losses,
    plot_rna_protein_latent_cn_cell_type_umap,
)
from arcadia.utils.logging import logger, setup_logger


def select_gene_likelihood(
    adata: AnnData,
    n_sample_genes: int = 1000,
    visualize: bool = True,
    modality=None,
    logger_=None,
    use_autozi=False,
    layer: str = None,
) -> str:
    """
    Selects the best gene likelihood by fitting four candidate distributions
    and comparing them using the Akaike Information Criterion (AIC).

    Optionally visualizes the fits for a representative gene.

    Parameters:
    - adata: AnnData object.
    - n_sample_genes: Number of genes to sample for testing.
    - visualize: If True, generates a plot comparing the four fits for one gene.
    - layer: Name of layer to use for likelihood selection. If None, uses adata.X.
             For VAE training, should use 'counts' layer (raw counts).

    Returns:
    - str: The winning gene likelihood ('normal', 'zinb', 'nb', or 'poisson').
    """
    if use_autozi:

        df, _ = zinb_test_autozi(
            adata,
            layer_counts=None,
            n_epochs=200,
            seed=0,
            min_counts=1,
            n_top_genes=adata.shape[1],
            posterior_threshold=0.5,
            logger_=None,
        )
        if df["is_zero_inflated"].mean() > 0.5:
            return "zinb"
        else:
            raise ValueError("AutoZI did not find enough zero-inflated genes")
    adata = adata.copy()
    if logger_ is None:
        logger = setup_logger(level="INFO")
    else:
        logger = logger_

    logger.info(f"Starting statistical tests on {n_sample_genes} sample genes...")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    gene_indices = np.random.choice(
        adata.shape[1], min(n_sample_genes, adata.shape[1]), replace=False
    )
    cell_indices = np.random.choice(adata.shape[0], min(5000, adata.shape[0]), replace=False)

    # Use specified layer or default to X
    if layer is not None and layer != "X":
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
        data_source = adata.layers[layer]
        logger.info(f"Using layer '{layer}' for likelihood selection")
    else:
        data_source = adata.X
        logger.info("Using adata.X for likelihood selection")

    X_subset = (
        data_source[np.ix_(cell_indices, gene_indices)].toarray()
        if scipy.sparse.issparse(data_source)
        else data_source[np.ix_(cell_indices, gene_indices)]
    )

    votes = {"normal": 0, "zinb": 0, "nb": 0, "poisson": 0}
    all_gene_results = []

    for i in tqdm(range(X_subset.shape[1]), desc="Fitting models", total=X_subset.shape[1]):
        gene_data = X_subset[:, i].astype(float)

        # Check if data contains non-integer values
        if not np.all(np.mod(gene_data, 1) == 0):
            # Data has non-integer values, need to convert for discrete distributions
            # Always normalize by minimum positive value to preserve relative distribution
            min_val = gene_data[gene_data > 0].min()
            gene_data_int = np.round(gene_data / min_val).astype(int)
        else:
            # Data is already integer-valued
            gene_data_int = gene_data.astype(int)
        aics, params = {}, {}

        # Fit all models and store parameters and AICs
        # 1. Normal
        mu_n, std_n = gene_data.mean(), gene_data.std()
        if std_n > 1e-8:
            params["normal"] = {"mu": mu_n, "std": std_n}
            aics["normal"] = 2 * 2 - 2 * norm.logpdf(gene_data, loc=mu_n, scale=std_n).sum()

        # 2. Poisson
        mu_p = gene_data_int.mean()
        if mu_p > 1e-8:
            params["poisson"] = {"mu": mu_p}
            aics["poisson"] = 2 * 1 - 2 * poisson.logpmf(gene_data_int, mu=mu_p).sum()

        # 3. Negative Binomial
        mean, var = gene_data_int.mean(), gene_data_int.var()
        if var > mean + 1e-8:
            size = (mean**2) / (var - mean)
            prob = size / (size + mean)
            params["nb"] = {"size": size, "prob": prob}
            aics["nb"] = 2 * 2 - 2 * nbinom.logpmf(gene_data_int, n=size, p=prob).sum()

        # 4. Zero-Inflated Negative Binomial
        if np.mean(gene_data_int == 0) > 0.5:
            aics["zinb"] = np.inf
            # initial_params = [mean, 1.0, np.mean(gene_data_int == 0)]
            # res = minimize(
            #     _zinb_log_likelihood,
            #     initial_params,
            #     args=(gene_data_int,),
            #     method="L-BFGS-B",
            #     bounds=[(1e-6, None), (1e-6, None), (0, 1 - 1e-6)],
            # )
            # if res.success:
            #     mu_z, alpha_z, pi_z = res.x
            #     params["zinb"] = {"mu": mu_z, "alpha": alpha_z, "pi": pi_z}
            #     aics["zinb"] = 2 * 3 - 2 * (-res.fun)

        # Vote for the best model for this gene
        if aics:
            winner = min(aics, key=aics.get)
            votes[winner] += 1
            all_gene_results.append(
                {
                    "gene_index": gene_indices[i],
                    "cell_index": cell_indices[i],
                    "data": gene_data,
                    "aics": aics,
                    "params": params,
                    "winner": winner,
                }
            )

    logger.info(f"Test complete. Vote counts: {votes}")

    best_likelihood = max(votes, key=votes.get) if votes else "poisson"
    logger.info(f"Selected likelihood: {best_likelihood}")

    # Visualization step
    if visualize:
        # Find a gene that voted for the winning likelihood to use as an example
        representative_gene = next(
            (res for res in all_gene_results if res["winner"] == best_likelihood), None
        )

        if representative_gene:
            logger.info(
                f"\nVisualizing fits for representative gene (index {representative_gene['gene_index']}) which chose '{best_likelihood}'."
            )
            plot_all_fits_for_gene(
                representative_gene["data"],
                representative_gene,
                representative_gene["gene_index"],
                modality=modality,
            )
        else:
            logger.info(
                f"Could not find a sample gene that voted for '{best_likelihood}' to visualize."
            )

    warnings.resetwarnings()
    return best_likelihood


def process_latent_spaces(adata_rna, adata_prot):
    """Process and combine latent spaces from both modalities.

    This function assumes that the latent representations have been computed
    using vae.module() and stored in the "X_scVI" field of the AnnData objects.

    Args:
        adata_rna: RNA AnnData object with latent representation in obsm["X_scVI"]
        adata_prot: Protein AnnData object with latent representation in obsm["X_scVI"]

    Returns:
        rna_latent: RNA latent AnnData
        prot_latent: Protein latent AnnData
        combined_latent: Combined latent AnnData
    """

    # Store latent representations
    SCVI_LATENT_KEY = "X_scVI"
    # Prepare AnnData objects
    rna_latent = AnnData(adata_rna.obsm[SCVI_LATENT_KEY].copy())
    prot_latent = AnnData(adata_prot.obsm[SCVI_LATENT_KEY].copy())
    rna_latent.obs = adata_rna.obs.copy()
    prot_latent.obs = adata_prot.obs.copy()
    n_obs = min(len(rna_latent), len(prot_latent), 2000)
    sc.pp.subsample(rna_latent, n_obs=n_obs)
    sc.pp.subsample(prot_latent, n_obs=n_obs)

    # Clear any existing embeddings
    rna_latent.obsm.pop("X_pca", None)
    prot_latent.obsm.pop("X_pca", None)

    # Use standard parameters for individual modalities
    sc.pp.neighbors(rna_latent, use_rep="X", n_neighbors=10)
    sc.tl.umap(rna_latent)
    sc.pp.neighbors(prot_latent, use_rep="X", n_neighbors=10)
    sc.tl.umap(prot_latent)

    # Combine latent spaces
    combined_latent = ad.concat(
        [rna_latent.copy(), prot_latent.copy()],
        join="outer",
        label="modality",
        keys=["RNA", "Protein"],
    )

    # Clear any existing neighbors data to ensure clean calculation
    combined_latent.obsm.pop("X_pca", None) if "X_pca" in combined_latent.obsm else None
    (
        combined_latent.obsp.pop("connectivities", None)
        if "connectivities" in combined_latent.obsp
        else None
    )
    (combined_latent.obsp.pop("distances", None) if "distances" in combined_latent.obsp else None)
    (combined_latent.uns.pop("neighbors", None) if "neighbors" in combined_latent.uns else None)

    # Use cosine metric and larger n_neighbors for better batch integration
    sc.pp.neighbors(combined_latent, use_rep="X")
    sc.tl.umap(combined_latent)

    return rna_latent, prot_latent, combined_latent


def match_cells_and_calculate_distances(rna_latent, prot_latent):
    """Match cells between modalities and calculate distances."""
    # Calculate pairwise distances
    latent_distances = batched_cdist(rna_latent.X, prot_latent.X)

    # Find matches
    prot_matches_in_rna = np.argmin(latent_distances, axis=0)
    matching_distances = np.min(latent_distances, axis=0)

    # Generate random matches for comparison
    rand_indices = np.random.permutation(len(rna_latent))
    rand_latent_distances = latent_distances[rand_indices, :]
    rand_prot_matches_in_rna = np.argmin(rand_latent_distances, axis=0)
    rand_matching_distances = np.min(rand_latent_distances, axis=0)

    return {
        "prot_matches_in_rna": prot_matches_in_rna,
        "matching_distances": matching_distances,
        "rand_prot_matches_in_rna": rand_prot_matches_in_rna,
        "rand_matching_distances": rand_matching_distances,
    }


def simulate_counts_zero_inflated(adata, threshold=None, target_max=10000, plot_flag=False):

    X = adata.X.copy()
    # If no threshold provided, estimate with GMM
    if threshold is None:
        nonzero_vals = X[X > 0].reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=0).fit(nonzero_vals)
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_).flatten()
        noise_idx = np.argmin(means)
        threshold = means[noise_idx] + 2 * stds[noise_idx]
        threshold = threshold
    cells_below_threshold = np.sum(X < threshold)
    total_cells = X.size
    percentage_below = (cells_below_threshold / total_cells) * 100
    logger.info(f"Cells below threshold: {cells_below_threshold:,} ({percentage_below:.2f}%)")

    # 2. Find nonzero values for scaling
    nonzero_vals = X[X > threshold]
    X[X < threshold] = 0
    if len(nonzero_vals) == 0:
        # All zeros after thresholding, return zeros
        return np.zeros_like(X, dtype=int)

    # 3. Scale nonzero values so that the maximum becomes target_max
    scale = target_max / nonzero_vals.max()
    X_scaled = X * scale

    # 4. Round nonzero values to nearest integer; keep zeros as zeros
    X_int = np.where(X_scaled > 0, np.rint(X_scaled), 0).astype(int)
    if plot_flag:
        visualize_gmm_threshold(adata.X, threshold, gmm, noise_idx)

    # Save integer counts to layers["counts"] for VAE training
    adata.layers["counts"] = X_int
    adata.X = X_int

    return adata


def transfer_to_integer_range_nb(adata, target_max=10000, plot_flag=False):
    """
    Convert continuous data to integer range suitable for negative binomial modeling.
    Preserves the overdispersion characteristics of the original data.
    """
    X = adata.X.copy()
    # Remove negative values if present
    X[X < 0] = 0

    # For NB, we want to preserve the mean-variance relationship
    # Scale by the minimum positive value to maintain relative differences
    X_nonzero = X[X > 0]
    if len(X_nonzero) > 0:
        min_val = np.min(X_nonzero)
        X_scaled = X / min_val

        # Scale to target range while preserving overdispersion
        scale_factor = target_max / np.max(X_scaled)
        X_scaled = X_scaled * scale_factor
    else:
        X_scaled = X

    # Round to nearest integer
    X_int = np.rint(X_scaled).astype(int)

    # Save integer counts to layers["counts"] for VAE training
    adata.layers["counts"] = X_int
    adata.X = X_int

    if plot_flag:
        visualize_integer_conversion_subplots(X, X_int)

    return adata


def transfer_to_integer_range_normal(adata, target_max=10000, plot_flag=False):

    X = adata.X.copy()
    # Remove negative values if present
    X[X < 0] = 0
    # Linearly rescale to [0, target_max]
    X_max = np.max(X)
    if X_max > 0:
        X_scaled = X / X_max * target_max
    else:
        X_scaled = X
    # Round to nearest integer
    X_int = np.rint(X_scaled).astype(int)

    # Save integer counts to layers["counts"] for VAE training
    adata.layers["counts"] = X_int
    adata.X = X_int

    if plot_flag:
        visualize_integer_conversion_subplots(X, X_int)
    return adata


def batched_cdist(X, Y, batch_size=5000):
    """Calculate pairwise distances in batches to prevent memory issues."""
    n_x = X.shape[0]
    n_y = Y.shape[0]
    distances = np.zeros((n_x, n_y))

    for i in tqdm(range(0, n_x, batch_size), desc="Processing rows", total=n_x // batch_size):
        end_i = min(i + batch_size, n_x)
        batch_X = X[i:end_i]

        for j in range(0, n_y, batch_size):
            end_j = min(j + batch_size, n_y)
            batch_Y = Y[j:end_j]

            batch_distances = cdist(batch_X, batch_Y)
            distances[i:end_i, j:end_j] = batch_distances

        logger.info(f"Processed {end_i}/{n_x} rows")

    return distances


def is_already_integer(adata):
    """Check if data is already close to integers"""
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    return np.allclose(X, np.round(X))


def ensure_correct_dtype(adata, target_dtype=np.int32):
    """Ensure data has correct integer dtype if already integer"""
    if adata.X.dtype != target_dtype:
        adata.X = adata.X.astype(target_dtype)
    return adata


# Functions needed by DualVAETrainingPlan
def compute_pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x: [N, D], y: [M, D]
    # Returns: [N, M] distance matrix
    # Ensure both tensors are on the same device
    y = y.to(x.device)
    x_norm = (x**2).sum(dim=1, keepdim=True)  # [N, 1]
    y_norm = (y**2).sum(dim=1, keepdim=True)  # [M, 1]
    xy = torch.matmul(x, y.transpose(-2, -1))  # [N, M]
    return torch.sqrt(torch.clamp(x_norm + y_norm.transpose(-2, -1) - 2 * xy, min=1e-8))


def compute_pairwise_kl_two_items(loc1, loc2, scale1, scale2, eps=1e-8, plot_flag=False):
    # Assumes scale* are std, not variance
    # Ensure all tensors are on the same device
    device = loc1.device
    loc2 = loc2.to(device)
    scale1 = scale1.to(device)
    scale2 = scale2.to(device)

    loc1 = loc1.unsqueeze(1)
    loc2 = loc2.unsqueeze(0)
    s1 = torch.clamp(scale1.unsqueeze(1), min=eps)
    s2 = torch.clamp(scale2.unsqueeze(0), min=eps)
    diff = loc1 - loc2
    kl = (torch.log(s2 / s1) + (s1**2 + diff**2) / (2 * s2**2) - 0.5).sum(dim=-1)
    if plot_flag and False:  # verity this works before plotting
        # Plot PCA of concatenated means, all points blue, closest pair red, farthest pair green
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        # Concatenate means
        rna_means = loc1.detach().cpu().numpy()
        prot_means = loc2.detach().cpu().numpy()
        all_means = np.concatenate([rna_means, prot_means], axis=0)

        # PCA to 2D
        pca = PCA(n_components=2)
        all_means_2d = pca.fit_transform(all_means)
        n_rna = rna_means.shape[0]
        prot_means.shape[0]

        # Find closest and farthest pairs
        latent_dist_np = kl.detach().cpu().numpy()
        min_idx = np.unravel_index(np.argmin(latent_dist_np), latent_dist_np.shape)
        max_idx = np.unravel_index(np.argmax(latent_dist_np), latent_dist_np.shape)

        # Plot all points in blue
        plt.figure(figsize=(8, 8))
        plt.scatter(
            all_means_2d[:n_rna, 0], all_means_2d[:n_rna, 1], c="blue", alpha=0.5, label="RNA"
        )
        plt.scatter(
            all_means_2d[n_rna:, 0], all_means_2d[n_rna:, 1], c="blue", alpha=0.5, label="Protein"
        )

        # Plot closest pair in red
        plt.scatter(
            all_means_2d[min_idx[0], 0],
            all_means_2d[min_idx[0], 1],
            c="red",
            s=120,
            label="Closest RNA",
        )
        plt.scatter(
            all_means_2d[n_rna + min_idx[1], 0],
            all_means_2d[n_rna + min_idx[1], 1],
            c="red",
            s=120,
            label="Closest Protein",
        )
        plt.plot(
            [all_means_2d[min_idx[0], 0], all_means_2d[n_rna + min_idx[1], 0]],
            [all_means_2d[min_idx[0], 1], all_means_2d[n_rna + min_idx[1], 1]],
            c="red",
            linewidth=2,
            alpha=0.7,
        )

        # Plot farthest pair in green
        plt.scatter(
            all_means_2d[max_idx[0], 0],
            all_means_2d[max_idx[0], 1],
            c="green",
            s=120,
            label="Farthest RNA",
        )
        plt.scatter(
            all_means_2d[n_rna + max_idx[1], 0],
            all_means_2d[n_rna + max_idx[1], 1],
            c="green",
            s=120,
            label="Farthest Protein",
        )
        plt.plot(
            [all_means_2d[max_idx[0], 0], all_means_2d[n_rna + max_idx[1], 0]],
            [all_means_2d[max_idx[0], 1], all_means_2d[n_rna + max_idx[1], 1]],
            c="green",
            linewidth=2,
            alpha=0.7,
        )

        plt.title("PCA of RNA/Protein Latent Means\nRed: Closest, Green: Farthest")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.tight_layout()
        plt.savefig("/home/barroz/projects/ARCADIA/.vscode/rna_protein_pca.pdf")
        plt.show()

    return kl


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB


def log_memory_usage(prefix=""):
    """Log current memory usage"""
    mem_usage = get_memory_usage()
    logger.info(f"{prefix}Memory usage: {mem_usage:.2f} GB")
    return mem_usage


def clear_memory():
    """Clear memory by running garbage collection and clearing CUDA cache if available"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return get_memory_usage()


def log_parameters(params, run_index, total_runs):
    """Log parameters for a run."""
    logger.info(f"\nParameters for run {run_index + 1}/{total_runs}:")
    # Group parameters by category
    model_params = {
        "n_hidden_rna": params.get("n_hidden_rna"),
        "n_hidden_prot": params.get("n_hidden_prot"),
        "n_layers": params.get("n_layers"),
        "latent_dim": params.get("latent_dim"),
    }

    training_params = {
        "max_epochs": params.get("max_epochs"),
        "batch_size": params.get("batch_size"),
        "lr": params.get("lr"),
        "train_size": params.get("train_size"),
        "gradient_clip_val": params.get("gradient_clip_val"),
    }

    loss_weights = {
        "contrastive_weight": params.get("contrastive_weight"),
        "similarity_weight": params.get("similarity_weight"),
        "matching_weight": params.get("matching_weight"),
        "cell_type_clustering_weight": params.get("cell_type_clustering_weight"),
        "cross_modal_cell_type_weight": params.get("cross_modal_cell_type_weight"),
        "rna_recon_weight": params.get("rna_recon_weight"),
        "prot_recon_weight": params.get("prot_recon_weight"),
    }

    other_params = {
        "plot_x_times": params.get("plot_x_times"),
        "save_checkpoint_every_n_epochs": params.get("save_checkpoint_every_n_epochs"),
        "load_optimizer_state": params.get("load_optimizer_state", True),
    }

    # Format the message
    msg = "\nModel Architecture:\n"
    for k, v in model_params.items():
        if v is not None:
            msg += f"├─ {k}: {v}\n"

    msg += "\nTraining Settings:\n"
    for k, v in training_params.items():
        if v is not None:
            msg += f"├─ {k}: {v}\n"

    msg += "\nLoss Weights:\n"
    for k, v in loss_weights.items():
        if v is not None:
            msg += f"├─ {k}: {v}\n"

    msg += "\nOther Parameters:\n"
    for k, v in other_params.items():
        if v is not None:
            msg += f"└─ {k}: {v}\n"

    logger.info(msg)


def generate_post_training_visualizations(
    adata_rna,
    adata_prot,
    rna_latent,
    prot_latent,
    combined_latent,
    history,
    matching_results,
    plot_flag=True,
):
    """Generate all visualizations for the model.

    Args:
        plot_flag: If False, skip all plotting. Default True.
    """
    if not plot_flag:
        logger.info("Skipping post-training visualizations (plot_flag=False)")
        return

    # Plot training results
    plot_normalized_losses(history)

    # Plot latent representations
    plot_latent_pca_both_modalities_cn(
        adata_rna.obsm["X_scVI"],
        adata_prot.obsm["X_scVI"],
        adata_rna,
        adata_prot,
        index_rna=range(len(adata_rna.obs.index)),
        index_prot=range(len(adata_prot.obs.index)),
        use_subsample=True,
    )

    plot_latent_pca_both_modalities_by_celltype(
        adata_rna,
        adata_prot,
        adata_rna.obsm["X_scVI"],
        adata_prot.obsm["X_scVI"],
        use_subsample=True,
    )

    # Plot distance distributions using plot_distance_comparison
    # Import here to avoid circular import
    from arcadia.archetypes.metrics import plot_distance_comparison

    plot_distance_comparison(
        matching_results["matching_distances"],
        matching_results["rand_matching_distances"],
    )

    # Plot combined visualizations
    plot_combined_latent_space(combined_latent, use_subsample=True)
    plot_cell_type_distributions(combined_latent, 3, use_subsample=True)

    # Plot archetype and embedding visualizations
    plot_archetype_embedding(adata_rna, adata_prot, use_subsample=True)
    plot_rna_protein_latent_cn_cell_type_umap(adata_rna, adata_prot, use_subsample=True)


def handle_error(e, params, run_name):
    """Handle errors during hyperparameter search."""
    logger.error("\n" + "=" * 80)
    logger.error("❌ RUN FAILED ❌".center(80))
    logger.error("=" * 80 + "\n")

    error_msg = f"""
    Error in run {run_name}:
    Error Type: {type(e).__name__}
    Error Message: {str(e)}
    Memory Usage: {get_memory_usage():.2f} GB
    Stack Trace:
    {traceback.format_exc()}
    Parameters used:
    {params}
    """
    logger.error(error_msg)
    mlflow.log_param("error_type", type(e).__name__)
    mlflow.log_param("error_message", str(e))
    mlflow.log_param("error_memory_usage", f"{get_memory_usage():.2f} GB")
    mlflow.log_param("error_stack_trace", traceback.format_exc())
    clear_memory()  # Clear memory on error


# Import VAE training functions from vae_training_utils (not yet fully migrated)
import sys
from pathlib import Path

# Import GradNorm from separate module

# Add CODEX_RNA_seq to path for vae_training_utils import
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root / "CODEX_RNA_seq") not in sys.path:
    sys.path.insert(0, str(_project_root / "CODEX_RNA_seq"))


def generate_target_cluster_structure_from_cell_types(
    adata_original_for_vae_plots,
    orginal_num_features,
    cell_types_column="cell_types",
):
    from simple_protein_vae import calculate_silhouette_score

    """
    Generate target cluster structure using existing cell type annotations.

    This function calculates target cluster structure statistics based on
    provided cell type labels instead of performing unsupervised clustering.

    Parameters
    ----------
    adata_original_for_vae_plots : AnnData
        Original unscaled protein data for structure calculation
    orginal_num_features : int
        Number of original protein features (before COVET)
    cell_types_column : str, optional
        Column name containing cell type annotations

    Returns
    -------
    tuple
        (target_cluster_structure, cluster_labels, target_silhouette_score)
    """
    logger.info(f"Using cell types from {cell_types_column} column for target cluster structure...")

    # Get cell type labels
    cluster_labels = adata_original_for_vae_plots.obs[cell_types_column].astype("category")
    cluster_labels_values = cluster_labels.values

    # Extract protein features for structure calculation
    if "feature_type" in adata_original_for_vae_plots.var.columns:
        protein_mask = adata_original_for_vae_plots.var["feature_type"] == "protein"
        logger.info(
            f"Using feature_type to identify protein features: {protein_mask.sum()} protein features found"
        )

        if issparse(adata_original_for_vae_plots.X):
            protein_only_data = adata_original_for_vae_plots.X[:, protein_mask].toarray()
        else:
            protein_only_data = adata_original_for_vae_plots.X[:, protein_mask].copy()
    else:
        # Fallback to using orginal_num_features
        logger.info(
            f"Using orginal_num_features to extract protein features: first {orginal_num_features} features"
        )

        if issparse(adata_original_for_vae_plots.X):
            protein_only_data = adata_original_for_vae_plots.X[:, :orginal_num_features].toarray()
        else:
            protein_only_data = adata_original_for_vae_plots.X[:, :orginal_num_features].copy()

    logger.info(f"Protein-only data shape for structure calculation: {protein_only_data.shape}")

    # Calculate target cluster structure from cell types
    unique_clusters = cluster_labels.cat.categories
    n_clusters = len(unique_clusters)
    logger.info(f"Found {n_clusters} cell types: {list(unique_clusters)}")

    original_centroids = []
    original_cluster_spreads = []
    original_cluster_sizes = []

    logger.info(f"Analyzing structure of {len(unique_clusters)} cell types...")

    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels_values == cluster_id
        cluster_data = protein_only_data[cluster_mask]

        # Calculate centroid
        centroid = np.mean(cluster_data, axis=0)
        original_centroids.append(centroid)

        # Calculate spread (average distance from centroid)
        distances_to_centroid = np.linalg.norm(cluster_data - centroid, axis=1)
        cluster_spread = np.mean(distances_to_centroid)
        original_cluster_spreads.append(cluster_spread)

        # Store cluster size
        original_cluster_sizes.append(len(cluster_data))

        logger.info(
            f"  Cell type {cluster_id}: {len(cluster_data)} cells, spread = {cluster_spread:.4f}"
        )

    original_centroids = np.array(original_centroids)
    original_cluster_spreads = np.array(original_cluster_spreads)

    # Calculate inter-cluster distances
    original_inter_cluster_distances = euclidean_distances(original_centroids, original_centroids)
    original_mean_inter_cluster_distance = np.mean(
        original_inter_cluster_distances[
            np.triu_indices_from(original_inter_cluster_distances, k=1)
        ]
    )
    original_mean_intra_cluster_spread = np.mean(original_cluster_spreads)

    logger.info(f"Cell type cluster structure statistics:")
    logger.info(f"  Mean inter-cluster distance: {original_mean_inter_cluster_distance:.4f}")
    logger.info(f"  Mean intra-cluster spread: {original_mean_intra_cluster_spread:.4f}")
    logger.info(
        f"  Separation/compactness ratio: {original_mean_inter_cluster_distance/original_mean_intra_cluster_spread:.4f}"
    )

    # Create target_cluster_structure dictionary
    target_cluster_structure = {
        "centroids": original_centroids,
        "cluster_spreads": original_cluster_spreads,
        "cluster_sizes": original_cluster_sizes,
        "inter_cluster_distances": original_inter_cluster_distances,
        "mean_inter_cluster_distance": original_mean_inter_cluster_distance,
        "mean_intra_cluster_spread": original_mean_intra_cluster_spread,
        "separation_compactness_ratio": original_mean_inter_cluster_distance
        / original_mean_intra_cluster_spread,
        "clustering_method": "cell_types",
        "clustering_resolution": None,
        "n_clusters": n_clusters,
        "protein_features_used": protein_only_data.shape[1],
    }

    # Calculate silhouette score for cell types
    target_silhouette_score = calculate_silhouette_score(
        protein_only_data,
        cluster_labels_values,
        logger,
        "Target Silhouette Score (Cell Types)",
        allow_nan_return=False,
    )
    target_cluster_structure["silhouette_score"] = target_silhouette_score

    logger.info(f"Target silhouette score (cell types): {target_silhouette_score:.4f}")
    logger.info(
        f"Will use cluster structure preservation as dynamic target instead of silhouette score"
    )

    return target_cluster_structure, cluster_labels, target_silhouette_score


def generate_target_cluster_structure_unsuprvised(
    adata_original_for_vae_plots,
    orginal_num_features,
    clustering_resolution=None,
):
    """
    Generate target cluster structure for VAE training.

    This function performs unsupervised clustering on protein features only
    and calculates target cluster structure statistics.

    Parameters
    ----------
    adata_original_for_vae_plots : AnnData
        Original unscaled protein data for clustering
    orginal_num_features : int
        Number of original protein features (before COVET)
    clustering_resolution : float, optional
        Resolution for Leiden clustering

    Returns
    -------
    tuple
        (target_cluster_structure, cluster_labels, target_silhouette_score)
    """
    from simple_protein_vae import calculate_silhouette_score

    logger.info("Performing unsupervised clustering on protein features only...")

    # Extract original protein features only (before COVET) for clustering
    if "feature_type" in adata_original_for_vae_plots.var.columns:
        protein_mask = adata_original_for_vae_plots.var["feature_type"] == "protein"
        logger.info(
            f"Using feature_type to identify protein features: {protein_mask.sum()} protein features found"
        )

        if issparse(adata_original_for_vae_plots.X):
            protein_only_data = adata_original_for_vae_plots.X[:, protein_mask].toarray()
        else:
            protein_only_data = adata_original_for_vae_plots.X[:, protein_mask].copy()
    else:
        # Fallback to using orginal_num_features
        logger.info(
            f"Using orginal_num_features to extract protein features: first {orginal_num_features} features"
        )

        if issparse(adata_original_for_vae_plots.X):
            protein_only_data = adata_original_for_vae_plots.X[:, :orginal_num_features].toarray()
        else:
            protein_only_data = adata_original_for_vae_plots.X[:, :orginal_num_features].copy()

    logger.info(f"Protein-only data shape for clustering: {protein_only_data.shape}")

    # Create temporary AnnData for clustering on protein features only
    adata_protein_only = AnnData(X=protein_only_data, obs=adata_original_for_vae_plots.obs.copy())

    # Perform clustering workflow on protein features only
    logger.info("Running PCA on protein features...")
    sc.pp.pca(adata_protein_only, n_comps=min(50, protein_only_data.shape[1] - 1))

    logger.info("Computing neighbors on protein features...")
    sc.pp.neighbors(adata_protein_only, n_neighbors=15, use_rep="X_pca")

    logger.info("Running Leiden clustering on protein features...")
    resolution = 2
    # Run Leiden clustering with silhouette score optimization

    best_silhouette = -1
    best_resolution = None
    best_clustering = None

    resolution_range = np.arange(0.1, 2, 0.05)  # Test resolutions from 0.1 to 2

    for resolution in resolution_range:
        sc.tl.leiden(adata_protein_only, resolution=resolution, key_added="leiden_temp")
        num_clusters = len(adata_protein_only.obs["leiden_temp"].unique())

        # Skip if too few or too many clusters
        if num_clusters < 6:
            continue
        if num_clusters > 25:
            break

        # Calculate silhouette score
        cluster_labels = adata_protein_only.obs["leiden_temp"].astype(int)
        silhouette = silhouette_score(adata_protein_only.X, cluster_labels)

        logger.info(
            f"Resolution: {resolution:.3f}, clusters: {num_clusters}, silhouette: {silhouette:.3f}"
        )

        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_resolution = resolution
            best_clustering = adata_protein_only.obs["leiden_temp"].copy()

    # Apply best clustering
    adata_protein_only.obs["protein_clusters"] = best_clustering

    logger.info(f"Best resolution: {best_resolution:.3f}, silhouette score: {best_silhouette:.3f}")
    logger.info(f"Final clusters: {len(adata_protein_only.obs['protein_clusters'].unique())}")
    logger.info(f"Used Leiden clustering with resolution={clustering_resolution}")

    cluster_labels = adata_protein_only.obs["protein_clusters"].values
    n_clusters = len(np.unique(cluster_labels))
    logger.info(f"Found {n_clusters} clusters from protein features")
    logger.info(f"Cluster distribution: {pd.Series(cluster_labels).value_counts().sort_index()}")

    # Calculate target cluster structure
    logger.info("Calculating target cluster structure from original protein features...")

    unique_clusters = np.unique(cluster_labels)
    original_centroids = []
    original_cluster_spreads = []
    original_cluster_sizes = []

    logger.info(f"Analyzing structure of {len(unique_clusters)} clusters...")

    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_data = protein_only_data[cluster_mask]

        # Calculate centroid
        centroid = np.mean(cluster_data, axis=0)
        original_centroids.append(centroid)

        # Calculate spread (average distance from centroid)
        distances_to_centroid = np.linalg.norm(cluster_data - centroid, axis=1)
        cluster_spread = np.mean(distances_to_centroid)
        original_cluster_spreads.append(cluster_spread)

        # Store cluster size
        original_cluster_sizes.append(len(cluster_data))

        logger.info(
            f"  Cluster {cluster_id}: {len(cluster_data)} cells, spread = {cluster_spread:.4f}"
        )

    original_centroids = np.array(original_centroids)
    original_cluster_spreads = np.array(original_cluster_spreads)

    # Calculate inter-cluster distances
    original_inter_cluster_distances = euclidean_distances(original_centroids, original_centroids)
    original_mean_inter_cluster_distance = np.mean(
        original_inter_cluster_distances[
            np.triu_indices_from(original_inter_cluster_distances, k=1)
        ]
    )
    original_mean_intra_cluster_spread = np.mean(original_cluster_spreads)

    logger.info(f"Original cluster structure statistics:")
    logger.info(f"  Mean inter-cluster distance: {original_mean_inter_cluster_distance:.4f}")
    logger.info(f"  Mean intra-cluster spread: {original_mean_intra_cluster_spread:.4f}")
    logger.info(
        f"  Separation/compactness ratio: {original_mean_inter_cluster_distance/original_mean_intra_cluster_spread:.4f}"
    )

    # Store target structure
    target_cluster_structure = {
        "centroids": original_centroids,
        "cluster_spreads": original_cluster_spreads,
        "cluster_sizes": original_cluster_sizes,
        "inter_cluster_distances": original_inter_cluster_distances,
        "mean_inter_cluster_distance": original_mean_inter_cluster_distance,
        "mean_intra_cluster_spread": original_mean_intra_cluster_spread,
        "separation_compactness_ratio": original_mean_inter_cluster_distance
        / original_mean_intra_cluster_spread,
        "clustering_method": "leiden",
        "clustering_resolution": clustering_resolution,
        "n_clusters": n_clusters,
        "protein_features_used": protein_only_data.shape[1],
    }

    # Calculate silhouette for comparison
    target_silhouette_score = calculate_silhouette_score(
        protein_only_data,  # Original protein features used for clustering
        cluster_labels,  # The clusters we discovered
        logger,
        "Target Silhouette Score (Original Protein Features)",
        allow_nan_return=False,
    )
    target_cluster_structure["silhouette_score"] = target_silhouette_score

    logger.info(f"Target silhouette score (for comparison): {target_silhouette_score:.4f}")
    logger.info(
        f"Will use cluster structure preservation as dynamic target instead of silhouette score"
    )

    return target_cluster_structure, cluster_labels, target_silhouette_score


def train_vae_for_archetype_generation(
    adata_2_prot,
    adata_original_for_vae_plots,
    target_cluster_structure,
    vae_hyperparams,
    experiment_name="Archetype_Generation_Protein_VAE",
    run_name_prefix="VAE_dim_reduction_protein_in_archetype_script",
):
    """
    Train a VAE for protein data dimensionality reduction in archetype generation pipeline.

    This function extracts the VAE training logic from the main archetype generation script
    and organizes it into a reusable function.

    Parameters
    ----------
    adata_2_prot : AnnData
        The protein data with z-scored features (protein + COVET) for VAE input
    adata_original_for_vae_plots : AnnData
        The unscaled protein data for plotting purposes
    target_cluster_structure : dict
        Target cluster structure information for dynamic cluster loss
    vae_hyperparams : dict
        VAE hyperparameters from configuration file
    experiment_name : str, optional
        MLflow experiment name
    run_name_prefix : str, optional
        Prefix for MLflow run name

    Returns
    -------
    tuple
        (trained_model_vae, adata_2_prot_latent, history_vae)
        - trained_model_vae: The trained VAE model
        - adata_2_prot_latent: AnnData with VAE latent representation replacing X
        - history_vae: Training history
    """
    from simple_protein_vae import SimpleProteinVAE, train_simple_protein_vae

    logger.info("Starting VAE training for archetype generation...")

    # Ensure output directory for VAE exists
    output_dir = Path("CODEX_RNA_seq/vae_output")
    os.makedirs(output_dir, exist_ok=True)
    vae_hyperparams["output_dir"] = str(output_dir)

    input_dim_vae = adata_2_prot.X.shape[1]

    # End any existing MLflow run
    try:
        mlflow.end_run()
    except:
        pass

    # MLflow setup for VAE training
    mlflow.set_experiment(experiment_name)
    time_stamp = time.strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"{time_stamp}_{run_name_prefix}"):
        # Log parameters
        mlflow.log_params(vae_hyperparams)
        mlflow.log_param("input_dim_vae", input_dim_vae)
        mlflow.log_param("original_adata_2_prot_shape_before_vae", adata_2_prot.shape)

        # Log target cluster structure parameters
        if target_cluster_structure:
            mlflow.log_param("clustering_method", target_cluster_structure.get("clustering_method"))
            mlflow.log_param(
                "clustering_resolution", target_cluster_structure.get("clustering_resolution")
            )
            mlflow.log_param("n_clusters_found", target_cluster_structure.get("n_clusters"))
            mlflow.log_param(
                "target_silhouette_score", target_cluster_structure.get("silhouette_score")
            )
            mlflow.log_param(
                "protein_features_for_clustering",
                target_cluster_structure.get("protein_features_used"),
            )
            mlflow.log_param(
                "target_mean_inter_cluster_distance",
                target_cluster_structure.get("mean_inter_cluster_distance"),
            )
            mlflow.log_param(
                "target_mean_intra_cluster_spread",
                target_cluster_structure.get("mean_intra_cluster_spread"),
            )
            mlflow.log_param(
                "target_separation_compactness_ratio",
                target_cluster_structure.get("separation_compactness_ratio"),
            )

        # Initialize VAE model
        model_vae = SimpleProteinVAE(
            input_dim=input_dim_vae,
            latent_dim=vae_hyperparams["latent_dim"],
            n_hidden=vae_hyperparams["n_hidden"],
            dropout_rate=vae_hyperparams["dropout_rate"],
            adata_for_hvg=adata_2_prot,
        )

        logger.info(f"Preparing adata_2_prot for VAE training. Input shape: {adata_2_prot.shape}")

        # Ensure data is dense for VAE training
        from scipy.sparse import issparse

        if issparse(adata_2_prot.X):
            logger.info("Converting adata_2_prot.X to dense for VAE training.")
            adata_2_prot_X_dense = adata_2_prot.X.toarray()
        else:
            adata_2_prot_X_dense = adata_2_prot.X.copy()

        # Create temporary AnnData for VAE training function, ensuring .X is dense
        adata_for_vae_training = AnnData(
            X=adata_2_prot_X_dense,  # Z-scored X for VAE input
            obs=adata_2_prot.obs.copy(),
            obsm=adata_2_prot.obsm.copy(),
            uns=adata_2_prot.uns.copy(),
            var=adata_2_prot.var.copy(),
        )

        # Handle train/validation split using train_size from hyperparameters
        train_size = vae_hyperparams.get("train_size", 1.0)
        if train_size < 1.0:
            logger.info(f"Creating train/validation split with train_size={train_size}")

            # Create train/val split indices
            n_obs = adata_for_vae_training.shape[0]
            indices = np.arange(n_obs)
            np.random.shuffle(indices)
            n_train = int(n_obs * train_size)

            train_idx = indices[:n_train]
            val_idx = indices[n_train:]

            # Create train and validation AnnData objects
            adata_prepared_train = adata_for_vae_training[train_idx].copy()
            adata_prepared_val = adata_for_vae_training[val_idx].copy()
            adata_original_train = adata_original_for_vae_plots[train_idx].copy()
            adata_original_val = adata_original_for_vae_plots[val_idx].copy()

            logger.info(f"Train set: {adata_prepared_train.shape[0]} samples")
            logger.info(f"Validation set: {adata_prepared_val.shape[0]} samples")
        else:
            logger.info("No validation split - using all data for training")
            adata_prepared_train = adata_for_vae_training
            adata_prepared_val = None
            adata_original_train = adata_original_for_vae_plots
            adata_original_val = None

        # Log training start information
        logger.info(
            f"Starting VAE training for protein data. Latent dim: {vae_hyperparams['latent_dim']}"
        )
        if target_cluster_structure:
            logger.info(f"Dynamic cluster structure-based loss:")
            logger.info(
                f"  Target inter-cluster distance: {target_cluster_structure['mean_inter_cluster_distance']:.4f}"
            )
            logger.info(
                f"  Target intra-cluster spread: {target_cluster_structure['mean_intra_cluster_spread']:.4f}"
            )
            logger.info(
                f"  Target separation/compactness ratio: {target_cluster_structure['separation_compactness_ratio']:.4f}"
            )

        # Train the VAE with proper train/val split
        trained_model_vae, history_vae, _, _ = train_simple_protein_vae(
            model=model_vae,
            adata_prepared=adata_prepared_train,  # Training data
            adata_original=adata_original_train,  # Training data for plotting
            adata_prepared_val=adata_prepared_val,  # Validation data or None
            adata_original_val=adata_original_val,  # Validation data for plotting or None
            num_epochs=vae_hyperparams["num_epochs"],
            batch_size=vae_hyperparams["batch_size"],
            learning_rate=vae_hyperparams["learning_rate"],
            kl_weight=vae_hyperparams["kl_weight"],
            cluster_preservation_loss_weight=vae_hyperparams["cluster_preservation_loss_weight"],
            output_dir=vae_hyperparams["output_dir"],
            plot_every_n_epochs=vae_hyperparams["plot_every_n_epochs"],
            print_every_n_epochs=vae_hyperparams.get("print_every_n_epochs", 10),
            early_stopping_patience=vae_hyperparams["early_stopping_patience"],
            early_stopping_min_delta=vae_hyperparams["early_stopping_min_delta"],
            early_stopping_metric=vae_hyperparams["early_stopping_metric"],
            early_stopping_mode=vae_hyperparams["early_stopping_mode"],
            # Dynamic cluster structure-based loss parameters
            target_cluster_structure=target_cluster_structure,
            dynamic_cluster_loss=True,
            structure_check_frequency=5,  # Check cluster structure every 5 epochs
            structure_tolerance=0.001,  # 1% tolerance for structure preservation
            cluster_column="cluster_labels",  # Use unsupervised cluster labels
            skip_first_epoch_plot=vae_hyperparams.get(
                "skip_first_epoch_plot", True
            ),  # Skip first epoch plot by default
        )

        logger.info("VAE training completed.")

        # Get latent representation (mu) for the full dataset
        trained_model_vae.eval()
        with torch.no_grad():
            device_vae = next(trained_model_vae.parameters()).device
            data_for_latent = torch.tensor(adata_for_vae_training.X, dtype=torch.float32).to(
                device_vae
            )
            _, latent_mu, _, _ = trained_model_vae(data_for_latent)

        # Create new AnnData with VAE latent space
        adata_2_prot_latent = AnnData(
            X=latent_mu.cpu().numpy(),
            obs=adata_2_prot.obs.copy(),
            obsm=adata_2_prot.obsm.copy(),
            uns=adata_2_prot.uns.copy(),
        )
        adata_2_prot_latent.var_names = [
            f"vae_feat_{i}" for i in range(adata_2_prot_latent.X.shape[1])
        ]
        adata_2_prot_latent.var["feature_type"] = "VAE_feature"

        # Log final shape
        mlflow.log_param("final_adata_2_prot_shape_after_vae", adata_2_prot_latent.shape)

        logger.info(
            f"Protein data replaced with VAE latent space. New shape: {adata_2_prot_latent.shape}"
        )

    return trained_model_vae, adata_2_prot_latent, history_vae


# Import validate_scvi_training_mixin from training_utils (not yet fully migrated)
import sys
from pathlib import Path

# Add CODEX_RNA_seq to path for training_utils import
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root / "CODEX_RNA_seq") not in sys.path:
    sys.path.insert(0, str(_project_root / "CODEX_RNA_seq"))


def validate_scvi_training_mixin():
    """Validate that the required line exists in scVI's _training_mixin.py file."""
    try:
        # Import scvi to get the actual module path
        import scvi

        # make sume scvi is 1.2.2.post
        if scvi.__version__ != "1.2.2.post2":
            raise ValueError(f"scVI version must be 1.2.2.post, got {scvi.__version__}")

        scvi_path = scvi.__file__
        base_dir = os.path.dirname(os.path.dirname(scvi_path))
        training_mixin_path = os.path.join(base_dir, "scvi", "model", "base", "_training_mixin.py")

        if not os.path.exists(training_mixin_path):
            raise FileNotFoundError(f"Could not find _training_mixin.py at {training_mixin_path}")

        # Read the file
        with open(training_mixin_path, "r") as f:
            lines = f.readlines()

        # Check if the required line exists
        required_line = "self._training_plan = training_plan"
        line_found = any(required_line in line for line in lines)

        if not line_found:
            raise RuntimeError(
                f"Required line '{required_line}' not found in {training_mixin_path}. "
                "Please add this line after the training_plan assignment."
            )
        logger.success("scVI training mixin validation passed")
    except Exception as e:
        raise RuntimeError(
            f"Failed to validate scVI training mixin: {str(e)}\n"
            "Please ensure you have modified the scVI library as described in the comment above."
        )


# Import loss functions from losses module

# Utility functions for counterfactual generation and latent embeddings


def predict_rna_cn_from_protein_neighbors(
    latent_distances: torch.Tensor,
    protein_cn_values: np.ndarray,
    protein_batch_labels: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """
    Predict RNA CN from protein CN by finding closest protein cells in latent space.

    Args:
        latent_distances: Distance matrix between RNA and protein cells
        protein_cn_values: CN values from protein adata
        protein_batch_labels: Batch labels for protein cells
        k: Number of nearest neighbors to consider

    Returns:
        predicted_cn_values: Array of predicted CN values for RNA cells
    """
    # Get the k nearest protein neighbors for each RNA cell
    _, rna_topk_prot_matches = torch.topk(latent_distances, k=k, dim=1, largest=False)
    rna_topk_prot_matches = rna_topk_prot_matches.detach().cpu().numpy()

    # For each RNA cell, find the most common CN among its k nearest protein neighbors
    predicted_cn_values = []
    for i in range(rna_topk_prot_matches.shape[0]):
        neighbor_indices = rna_topk_prot_matches[i]
        neighbor_cn_values = protein_cn_values[protein_batch_labels][neighbor_indices]
        # Find most common CN (mode), handling NaN values
        cn_series = pd.Series(neighbor_cn_values)
        most_common_cn = (
            cn_series.value_counts(dropna=True).index[0]
            if not cn_series.dropna().empty
            else neighbor_cn_values[0]
        )
        predicted_cn_values.append(most_common_cn)
    return np.array(predicted_cn_values)


def get_latent_embedding(
    model,
    adata,
    batch_size=1000,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
):
    """Get latent embedding for data using a trained model.

    Args:
        model: Trained SCVI model (RNA or protein)
        adata: AnnData object with data to embed
        batch_size: Batch size for processing
        device: Device to use

    Returns:
        latent: NumPy array with latent embeddings
    """
    logger.info(f"Getting latent embeddings for {adata.shape[0]} cells...")

    # Make sure model is in eval mode
    model.module.eval()

    # Process in batches to avoid memory issues
    latent_parts = []
    total_cells = adata.shape[0]
    for i in range(0, total_cells, batch_size):
        end_idx = min(i + batch_size, total_cells)
        indices = np.arange(i, end_idx)

        # Get batch data
        X = adata[indices].X
        if issparse(X):
            X = X.toarray()
        X = torch.tensor(X, dtype=torch.float32).to(device)
        library_size = torch.tensor(
            adata[indices].obs["_scvi_library_size"].values, dtype=torch.float32
        ).to(device)
        # Get batch indices if they exist, otherwise use zeros
        batch_values = (
            torch.tensor(adata[indices].obs["batch"].cat.codes.values, dtype=torch.long)
            .to(device)
            .unsqueeze(1)
        )  # Make 2D (N, 1) for scVI compatibility

        batch = {
            "X": X,
            "batch": batch_values,
            "labels": indices,
            "library_size": library_size,
        }

        # Get latent representation
        with torch.no_grad():
            # Batch should already be in proper format from data preparation
            inference_outputs, _, _ = model.module(batch)
            latent_mean = inference_outputs["qz"].mean.detach().cpu().numpy()
            latent_parts.append(latent_mean)

        if (i + batch_size) % (10 * batch_size) == 0:
            logger.info(f"Processed {i + batch_size} / {total_cells} cells")

    # Combine all batches
    latent = np.vstack(latent_parts)
    logger.info(f"Generated latent embeddings with shape {latent.shape}")

    return latent


def _plot_encoded_latent_spaces(
    rna_encoded_latent, protein_encoded_latent, rna_obs, protein_obs, step=0, plot_flag=True
):
    """
    Plot UMAP of encoded latent spaces before decoding, showing modality and cell type distribution.

    Args:
        rna_encoded_latent: RNA encoded latent representation (numpy array)
        protein_encoded_latent: Protein encoded latent representation (numpy array)
        rna_obs: RNA observation metadata (DataFrame)
        protein_obs: Protein observation metadata (DataFrame)
        step: Current training step for file naming
    """
    if not plot_flag:
        return

    logger.info("Computing UMAP and PCA for encoded latent spaces...")

    # Combine both latent spaces
    combined_latent = np.vstack([rna_encoded_latent, protein_encoded_latent])

    # Create modality labels
    rna_modality = np.array(["RNA"] * rna_encoded_latent.shape[0])
    protein_modality = np.array(["Protein"] * protein_encoded_latent.shape[0])
    combined_modality = np.concatenate([rna_modality, protein_modality])

    # Get cell type labels
    rna_cell_types = (
        rna_obs["cell_types"].values
        if "cell_types" in rna_obs.columns
        else np.array(["Unknown"] * len(rna_obs))
    )
    protein_cell_types = (
        protein_obs["cell_types"].values
        if "cell_types" in protein_obs.columns
        else np.array(["Unknown"] * len(protein_obs))
    )
    combined_cell_types = np.concatenate([rna_cell_types, protein_cell_types])

    # Get CN labels if available
    rna_cn = (
        rna_obs["CN"].values if "CN" in rna_obs.columns else np.array(["Unknown"] * len(rna_obs))
    )
    protein_cn = (
        protein_obs["CN"].values
        if "CN" in protein_obs.columns
        else np.array(["Unknown"] * len(protein_obs))
    )
    combined_cn = np.concatenate([rna_cn, protein_cn])

    # Compute UMAP
    logger.info("Running UMAP...")
    umap_model = UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
    umap_embedding = umap_model.fit_transform(combined_latent)

    # Compute PCA
    logger.info("Running PCA...")
    pca_model = PCA(n_components=2, random_state=42)
    pca_embedding = pca_model.fit_transform(combined_latent)
    logger.info(f"PCA explained variance: {pca_model.explained_variance_ratio_}")

    # Create UMAP figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # Plot 1: By Modality
    ax = axes[0]
    for modality in ["RNA", "Protein"]:
        mask = combined_modality == modality
        ax.scatter(
            umap_embedding[mask, 0],
            umap_embedding[mask, 1],
            label=modality,
            alpha=0.6,
            s=10,
        )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Encoded Latent Space by Modality")
    ax.legend()

    # Plot 2: By Cell Type
    ax = axes[1]
    unique_cell_types = np.unique(combined_cell_types)
    colors = sns.color_palette("tab10", n_colors=len(unique_cell_types))
    for i, cell_type in enumerate(unique_cell_types):
        mask = combined_cell_types == cell_type
        ax.scatter(
            umap_embedding[mask, 0],
            umap_embedding[mask, 1],
            label=cell_type,
            alpha=0.6,
            s=10,
            color=colors[i],
        )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Encoded Latent Space by Cell Type")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot 3: By CN
    ax = axes[2]
    unique_cn = np.unique(combined_cn)
    colors = sns.color_palette("tab20", n_colors=len(unique_cn))
    for i, cn in enumerate(unique_cn):
        mask = combined_cn == cn
        ax.scatter(
            umap_embedding[mask, 0],
            umap_embedding[mask, 1],
            label=cn,
            alpha=0.6,
            s=10,
            color=colors[i],
        )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Encoded Latent Space by CN")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    # Save UMAP figure
    filename = f"encoded_latent_umap_step_{step:05d}.pdf"
    safe_mlflow_log_figure(fig, f"counterfactual/{filename}")
    plt.close()

    logger.info(f"Encoded latent space UMAP saved: {filename}")

    # Create PCA figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # Plot 1: By Modality
    ax = axes[0]
    for modality in ["RNA", "Protein"]:
        mask = combined_modality == modality
        ax.scatter(
            pca_embedding[mask, 0],
            pca_embedding[mask, 1],
            label=modality,
            alpha=0.6,
            s=10,
        )
    ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.2%})")
    ax.set_title("Encoded Latent Space by Modality")
    ax.legend()

    # Plot 2: By Cell Type
    ax = axes[1]
    unique_cell_types = np.unique(combined_cell_types)
    colors = sns.color_palette("tab10", n_colors=len(unique_cell_types))
    for i, cell_type in enumerate(unique_cell_types):
        mask = combined_cell_types == cell_type
        ax.scatter(
            pca_embedding[mask, 0],
            pca_embedding[mask, 1],
            label=cell_type,
            alpha=0.6,
            s=10,
            color=colors[i],
        )
    ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.2%})")
    ax.set_title("Encoded Latent Space by Cell Type")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot 3: By CN
    ax = axes[2]
    unique_cn = np.unique(combined_cn)
    colors = sns.color_palette("tab20", n_colors=len(unique_cn))
    for i, cn in enumerate(unique_cn):
        mask = combined_cn == cn
        ax.scatter(
            pca_embedding[mask, 0],
            pca_embedding[mask, 1],
            label=cn,
            alpha=0.6,
            s=10,
            color=colors[i],
        )
    ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.2%})")
    ax.set_title("Encoded Latent Space by CN")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    # Save PCA figure
    filename = f"encoded_latent_pca_step_{step:05d}.pdf"
    safe_mlflow_log_figure(fig, f"counterfactual/{filename}")
    plt.close()

    logger.info(f"Encoded latent space PCA saved: {filename}")


def create_counterfactual_adata(
    self: "DualVAETrainingPlan",
    adata_rna_save: AnnData,
    adata_prot_save: AnnData,
    rna_latent=None,
    protein_latent=None,
    plot_flag=False,
    skip_normalization=False,
) -> tuple[AnnData, AnnData, AnnData, AnnData]:
    """
    Create counterfactual data by comparing same-modal VAE reconstruction vs cross-modal generation.

    NEW APPROACH:
    - Ground truth = VAE reconstruction (e.g., RNA → RNA encoder → RNA decoder) we run forward pass and take the output of the decoder
    - Counterfactual = Cross-modal generation (e.g., Protein → Protein encoder → RNA decoder)

    This allows us to compare what each decoder generates from:
    1. Same-modality latent (reconstruction) - this is our "ground truth"
    2. Cross-modality latent (counterfactual) - this is what we want to evaluate

    Returns 4 AnnData objects:
    - same_modal_adata_rna: RNA reconstruction (RNA→RNA encoder→RNA decoder)
    - counterfactual_adata_rna: RNA counterfactual (Protein→Protein encoder→RNA decoder)
    - same_modal_adata_protein: Protein reconstruction (Protein→Protein encoder→Protein decoder)
    - counterfactual_adata_protein: Protein counterfactual (RNA→RNA encoder→Protein decoder)
    """
    # Use constant protein library size for all counterfactual generation
    constant_protein_lib_size_log = torch.tensor(
        self.constant_protein_lib_size_log, dtype=torch.float32, device=self.device
    )

    logger.info(
        f"Using constant protein library size: {self.constant_protein_lib_size} "
        f"(log: {self.constant_protein_lib_size_log:.4f}) for ALL counterfactual generation operations"
    )
    if skip_normalization:
        logger.info("Step 0: SKIPPING normalization - data is already normalized")
        adata_rna_normalized = adata_rna_save
        adata_prot_normalized = adata_prot_save

        # DIAGNOSTIC: Verify data is already normalized
        rna_lib_sums = np.array(adata_rna_normalized.X.sum(axis=1)).flatten()
        prot_lib_sums = np.array(adata_prot_normalized.X.sum(axis=1)).flatten()
        logger.info(
            f"[TRAINING-SKIP-NORM] Data already normalized - RNA lib sums mean: {rna_lib_sums.mean():.2f}, Protein lib sums mean: {prot_lib_sums.mean():.2f}"
        )
        logger.info(f"[TRAINING-SKIP-NORM] Expected: {self.constant_protein_lib_size:.2f}")
    else:
        logger.info(
            "Step 0: Normalizing both RNA and protein data to constant protein library size"
        )

        adata_rna_normalized = adata_rna_save
        adata_prot_normalized = adata_prot_save

        sc.pp.normalize_total(adata_rna_normalized, target_sum=self.constant_protein_lib_size)
        adata_rna_normalized.uns["normalize_total_value"] = self.constant_protein_lib_size
        adata_rna_normalized.X = adata_rna_normalized.X.astype(np.int32)

        sc.pp.normalize_total(adata_prot_normalized, target_sum=self.constant_protein_lib_size)
        adata_prot_normalized.uns["normalize_total_value"] = self.constant_protein_lib_size
        adata_prot_normalized.X = adata_prot_normalized.X.astype(np.int32)

        # DIAGNOSTIC: Verify normalization
        rna_lib_sums_after = np.array(adata_rna_normalized.X.sum(axis=1)).flatten()
        prot_lib_sums_after = np.array(adata_prot_normalized.X.sum(axis=1)).flatten()
        logger.info(
            f"[TRAINING] After normalization - RNA lib sums mean: {rna_lib_sums_after.mean():.2f}, Protein lib sums mean: {prot_lib_sums_after.mean():.2f}"
        )
        logger.info(f"[TRAINING] Expected: {self.constant_protein_lib_size:.2f}")

        logger.info(
            f"✓ Both datasets normalized to constant library size: {self.constant_protein_lib_size}"
        )
    logger.info("Generating same-modal RNA (RNA data → RNA encoder → RNA decoder)...")

    rna_X = adata_rna_normalized.X
    if issparse(rna_X):
        rna_X = rna_X.toarray()
    rna_X_tensor = torch.tensor(rna_X, dtype=torch.float32, device=self.device)

    rna_batch_codes = adata_rna_normalized.obs["batch"].cat.codes.values
    rna_batch_tensor = torch.tensor(
        rna_batch_codes, dtype=torch.long, device=self.device
    ).unsqueeze(1)

    n_rna_cells = rna_X_tensor.shape[0]

    # CRITICAL: Use per-cell library sizes from the adata object, NOT constant library size
    # During training, the model sees per-cell library sizes (from _scvi_library_size)
    # Using constant library for all cells causes distribution shift
    if "_scvi_library_size" in adata_rna_normalized.obs.columns:
        rna_lib_tensor = torch.tensor(
            adata_rna_normalized.obs["_scvi_library_size"].values,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        logger.info(f"  - Using PER-CELL library sizes from _scvi_library_size column")
        logger.info(
            f"  - Library size stats: mean={adata_rna_normalized.obs['_scvi_library_size'].mean():.4f}, std={adata_rna_normalized.obs['_scvi_library_size'].std():.4f}"
        )
    else:
        # Fallback: compute library sizes from the data
        rna_lib_sizes = np.log(np.array(rna_X_tensor.cpu().sum(axis=1)).flatten())
        rna_lib_tensor = torch.tensor(
            rna_lib_sizes, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        logger.info(f"  - Computing PER-CELL library sizes from data (log-transformed)")
        logger.info(
            f"  - Library size stats: mean={rna_lib_sizes.mean():.4f}, std={rna_lib_sizes.std():.4f}"
        )

    logger.info(f"  - RNA data shape: {rna_X_tensor.shape}")
    logger.info(f"  - Library tensor shape: {rna_lib_tensor.shape}")

    with torch.no_grad():
        rna_batch_for_encoding = {
            "X": rna_X_tensor,
            "batch": rna_batch_tensor,
            "library": rna_lib_tensor,
            "labels": torch.arange(n_rna_cells, device=self.device).unsqueeze(1),
        }
        rna_encoded_outputs, rna_decoded_outputs, _ = self.rna_vae.module(rna_batch_for_encoding)
        rna_encoded_latent = rna_encoded_outputs["qz"].mean
        logger.info(f"  - RNA encoded latent shape: {rna_encoded_latent.shape}")
        rna_px_dist = rna_decoded_outputs["px"]
        if hasattr(rna_px_dist, "mu"):
            rna_mu = rna_px_dist.mu.detach().cpu().numpy()
            if hasattr(rna_px_dist, "zi_logits"):
                logger.info("RNA ZI logits found")
                rna_zi_probs = torch.sigmoid(rna_px_dist.zi_logits).detach().cpu().numpy()
                same_modal_rna_linear = (1 - rna_zi_probs) * rna_mu
            else:
                same_modal_rna_linear = rna_mu
        else:
            same_modal_rna_linear = rna_px_dist.loc.detach().cpu().numpy()

    same_modal_rna_linear = np.round(same_modal_rna_linear).astype(np.int32)
    same_modal_rna_df = pd.DataFrame(same_modal_rna_linear)
    if same_modal_rna_df.shape[1] != len(self.rna_vae.adata.var_names):
        logger.error(
            f"Dimension mismatch in same-modal RNA: generated {same_modal_rna_df.shape[1]} features, "
            f"expected {len(self.rna_vae.adata.var_names)} RNA genes"
        )
        raise ValueError(f"Same-modal RNA generation failed: dimension mismatch")

    same_modal_rna_df.columns = self.rna_vae.adata.var_names
    same_modal_rna_df.index = adata_rna_save.obs.index
    adata_rna_save.obsm["same_modal_rna"] = same_modal_rna_df

    logger.info("Encoding protein data for both same-modal and cross-modal generation...")
    protein_X = adata_prot_normalized.X
    if issparse(protein_X):
        protein_X = protein_X.toarray()
    protein_X_tensor = torch.tensor(protein_X, dtype=torch.float32, device=self.device)

    protein_batch_codes = adata_prot_normalized.obs["batch"].cat.codes.values
    protein_batch_tensor_orig = torch.tensor(
        protein_batch_codes, dtype=torch.long, device=self.device
    ).unsqueeze(1)

    n_protein_cells = protein_X_tensor.shape[0]

    # CRITICAL: Use per-cell library sizes from the adata object, NOT constant library size
    if "_scvi_library_size" in adata_prot_normalized.obs.columns:
        protein_lib_tensor = torch.tensor(
            adata_prot_normalized.obs["_scvi_library_size"].values,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        logger.info(f"  - Using PER-CELL library sizes from _scvi_library_size column")
        logger.info(
            f"  - Library size stats: mean={adata_prot_normalized.obs['_scvi_library_size'].mean():.4f}, std={adata_prot_normalized.obs['_scvi_library_size'].std():.4f}"
        )
    else:
        # Fallback: compute library sizes from the data
        protein_lib_sizes = np.log(np.array(protein_X_tensor.cpu().sum(axis=1)).flatten())
        protein_lib_tensor = torch.tensor(
            protein_lib_sizes, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        logger.info(f"  - Computing PER-CELL library sizes from data (log-transformed)")
        logger.info(
            f"  - Library size stats: mean={protein_lib_sizes.mean():.4f}, std={protein_lib_sizes.std():.4f}"
        )

    logger.info(f"  - Protein data shape: {protein_X_tensor.shape}")
    logger.info(f"  - Library tensor shape: {protein_lib_tensor.shape}")

    with torch.no_grad():
        protein_batch_for_encoding = {
            "X": protein_X_tensor,
            "batch": protein_batch_tensor_orig,
            "library": protein_lib_tensor,
            "labels": torch.arange(n_protein_cells, device=self.device).unsqueeze(1),
        }
        protein_encoded_outputs, protein_decoded_outputs, _ = self.protein_vae.module(
            protein_batch_for_encoding
        )
        protein_encoded_latent = protein_encoded_outputs["qz"].mean
        logger.info(f"  - Protein encoded latent shape: {protein_encoded_latent.shape}")
        protein_px_dist = protein_decoded_outputs["px"]
        if hasattr(protein_px_dist, "mu"):
            protein_mu = protein_px_dist.mu.detach().cpu().numpy()
            if hasattr(protein_px_dist, "zi_logits"):
                protein_zi_probs = torch.sigmoid(protein_px_dist.zi_logits).detach().cpu().numpy()
                same_modal_protein_linear = (1 - protein_zi_probs) * protein_mu
            else:
                same_modal_protein_linear = protein_mu
        else:
            same_modal_protein_linear = protein_px_dist.loc.detach().cpu().numpy()

        same_modal_protein_linear = np.round(same_modal_protein_linear).astype(np.int32)
    same_modal_protein_df = pd.DataFrame(same_modal_protein_linear)
    if same_modal_protein_df.shape[1] != len(self.protein_vae.adata.var_names):
        logger.error(
            f"Dimension mismatch in same-modal protein: generated {same_modal_protein_df.shape[1]} features, "
            f"expected {len(self.protein_vae.adata.var_names)} proteins"
        )
        raise ValueError(f"Same-modal protein generation failed: dimension mismatch")

    # === Visualize encoded latent spaces before decoding ===
    logger.info("Creating latent space visualization before decoding...")
    if plot_flag:
        _plot_encoded_latent_spaces(
            rna_encoded_latent=rna_encoded_latent.detach().cpu().numpy(),
            protein_encoded_latent=protein_encoded_latent.detach().cpu().numpy(),
            rna_obs=adata_rna_save.obs,
            protein_obs=adata_prot_save.obs,
            step=self.current_epoch if hasattr(self, "current_epoch") else 0,
            plot_flag=plot_flag,
        )
    logger.info(
        "Generating same-modal protein (Protein data → Protein encoder → Protein decoder)..."
    )
    same_modal_protein_df.columns = self.protein_vae.adata.var_names
    same_modal_protein_df.index = adata_prot_save.obs.index
    adata_prot_save.obsm["same_modal_protein"] = same_modal_protein_df

    logger.info("Generating counterfactual protein (RNA → RNA encoder → Protein decoder)...")

    most_common_protein_batch = torch.mode(protein_batch_tensor_orig.flatten())[0].item()
    protein_batch_tensor_for_rna = torch.full(
        (n_rna_cells, 1), most_common_protein_batch, dtype=torch.long, device=self.device
    )

    # CRITICAL FIX: Use per-cell RNA library sizes for counterfactual protein generation
    # This ensures consistency with same-modal generation
    if "_scvi_library_size" in adata_rna_normalized.obs.columns:
        lib_tensor_for_rna_cells = torch.tensor(
            adata_rna_normalized.obs["_scvi_library_size"].values,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        logger.info(f"Using PER-CELL RNA library sizes for counterfactual protein generation")
        logger.info(
            f"Library size stats: mean={adata_rna_normalized.obs['_scvi_library_size'].mean():.4f}, std={adata_rna_normalized.obs['_scvi_library_size'].std():.4f}"
        )
    else:
        # Fallback: compute from data
        rna_lib_sizes_for_protein = np.log(np.array(rna_X_tensor.cpu().sum(axis=1)).flatten())
        lib_tensor_for_rna_cells = torch.tensor(
            rna_lib_sizes_for_protein, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        logger.info(f"Computing PER-CELL RNA library sizes for counterfactual protein from data")
        logger.info(
            f"Library size stats: mean={rna_lib_sizes_for_protein.mean():.4f}, std={rna_lib_sizes_for_protein.std():.4f}"
        )

    with torch.no_grad():
        counterfactual_protein_outputs = self.protein_vae.module.generative(
            z=rna_encoded_latent,
            library=lib_tensor_for_rna_cells,
            batch_index=protein_batch_tensor_for_rna,
        )
        protein_px_dist = counterfactual_protein_outputs["px"]
        if hasattr(protein_px_dist, "mu"):
            protein_mu = protein_px_dist.mu.detach().cpu().numpy()
            if hasattr(protein_px_dist, "zi_logits"):
                protein_zi_probs = torch.sigmoid(protein_px_dist.zi_logits).detach().cpu().numpy()
                counterfactual_protein_linear = (1 - protein_zi_probs) * protein_mu
            else:
                counterfactual_protein_linear = protein_mu
        else:
            counterfactual_protein_linear = protein_px_dist.loc.detach().cpu().numpy()
    counterfactual_protein_linear = np.round(counterfactual_protein_linear).astype(np.int32)
    counterfactual_protein_df = pd.DataFrame(counterfactual_protein_linear)
    if counterfactual_protein_df.shape[1] != len(self.protein_vae.adata.var_names):
        logger.error(
            f"Dimension mismatch in protein counterfactuals: generated {counterfactual_protein_df.shape[1]} features, "
            f"expected {len(self.protein_vae.adata.var_names)} proteins"
        )
        raise ValueError(f"Protein counterfactual generation failed: dimension mismatch")

    counterfactual_protein_df.columns = self.protein_vae.adata.var_names
    counterfactual_protein_df.index = adata_rna_save.obs.index
    adata_rna_save.obsm["counterfactual_protein"] = counterfactual_protein_df

    logger.info("Generating counterfactual RNA (Protein → Protein encoder → RNA decoder)...")
    most_common_rna_batch = torch.mode(rna_batch_tensor.flatten())[0].item()
    rna_batch_tensor_for_protein = torch.full(
        (n_protein_cells, 1), most_common_rna_batch, dtype=torch.long, device=self.device
    )

    # CRITICAL FIX: Use per-cell protein library sizes for counterfactual RNA generation
    # This ensures the decoder sees the same library size distribution as in same-modal generation
    # Otherwise, even identical latents will produce different outputs due to library size differences
    if "_scvi_library_size" in adata_prot_normalized.obs.columns:
        lib_tensor_for_protein_cells = torch.tensor(
            adata_prot_normalized.obs["_scvi_library_size"].values,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        logger.info(f"Using PER-CELL protein library sizes for counterfactual RNA generation")
        logger.info(
            f"Library size stats: mean={adata_prot_normalized.obs['_scvi_library_size'].mean():.4f}, std={adata_prot_normalized.obs['_scvi_library_size'].std():.4f}"
        )
    else:
        # Fallback: compute from data
        protein_lib_sizes_for_rna = np.log(np.array(protein_X_tensor.cpu().sum(axis=1)).flatten())
        lib_tensor_for_protein_cells = torch.tensor(
            protein_lib_sizes_for_rna, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        logger.info(f"Computing PER-CELL protein library sizes for counterfactual RNA from data")
        logger.info(
            f"Library size stats: mean={protein_lib_sizes_for_rna.mean():.4f}, std={protein_lib_sizes_for_rna.std():.4f}"
        )

    # === CRITICAL DIAGNOSTIC: Compare latent representations going into RNA decoder ===
    logger.info("=== DIAGNOSTIC: Comparing latent inputs to RNA decoder ===")
    logger.info(f"RNA encoded latent (from RNA data) shape: {rna_encoded_latent.shape}")
    logger.info(f"Protein encoded latent (from Protein data) shape: {protein_encoded_latent.shape}")

    # Compute statistics for the latent representations
    rna_latent_mean = rna_encoded_latent.mean().item()
    rna_latent_std = rna_encoded_latent.std().item()
    protein_latent_mean = protein_encoded_latent.mean().item()
    protein_latent_std = protein_encoded_latent.std().item()

    logger.info(f"RNA latent stats: mean={rna_latent_mean:.4f}, std={rna_latent_std:.4f}")
    logger.info(
        f"Protein latent stats: mean={protein_latent_mean:.4f}, std={protein_latent_std:.4f}"
    )

    # Compute cosine similarity between latent representations (cell-wise)
    # Since we have different numbers of cells, we'll sample matching cells if provided
    if rna_encoded_latent.shape[0] == protein_encoded_latent.shape[0]:
        # Same number of cells - compute pairwise cosine similarity
        rna_norm = torch.nn.functional.normalize(rna_encoded_latent, dim=1)
        protein_norm = torch.nn.functional.normalize(protein_encoded_latent, dim=1)
        cosine_sim = (rna_norm * protein_norm).sum(dim=1)
        logger.info(
            f"Cell-wise cosine similarity: mean={cosine_sim.mean().item():.4f}, std={cosine_sim.std().item():.4f}"
        )

    # Create UMAP visualization of latent representations going into RNA decoder
    if plot_flag:
        logger.info("Creating UMAP of latent representations BEFORE RNA decoding...")

        # Create AnnData objects for the latent representations
        rna_latent_adata = AnnData(rna_encoded_latent.detach().cpu().numpy())
        rna_latent_adata.obs = adata_rna_save.obs.copy()
        rna_latent_adata.obs["source"] = "RNA_encoder"

        protein_latent_adata = AnnData(protein_encoded_latent.detach().cpu().numpy())
        protein_latent_adata.obs = adata_prot_save.obs.copy()
        protein_latent_adata.obs["source"] = "Protein_encoder"

        # Combine both latent representations
        combined_latent_adata = sc.concat([rna_latent_adata, protein_latent_adata], join="outer")

        # Subsample if too large
        if combined_latent_adata.shape[0] > 5000:
            sc.pp.subsample(combined_latent_adata, n_obs=5000)

        # Compute UMAP
        sc.pp.neighbors(combined_latent_adata, use_rep="X", n_neighbors=15)
        sc.tl.umap(combined_latent_adata)

        # Plot UMAP colored by source (RNA vs Protein encoder)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        sc.pl.umap(
            combined_latent_adata,
            color="source",
            title="Latent representations going into RNA decoder\n(Should be well-mixed if similar)",
            ax=axes[0],
            show=False,
            palette={"RNA_encoder": "#1f77b4", "Protein_encoder": "#ff7f0e"},
        )

        if "CN" in combined_latent_adata.obs.columns:
            sc.pl.umap(
                combined_latent_adata,
                color="CN",
                title="Cell Neighborhoods",
                ax=axes[1],
                show=False,
            )

        if "cell_types" in combined_latent_adata.obs.columns:
            sc.pl.umap(
                combined_latent_adata,
                color="cell_types",
                title="Cell Types",
                ax=axes[2],
                show=False,
            )

        plt.tight_layout()

        # Save figure
        safe_mlflow_log_figure(
            fig,
            f"counterfactual/latent_before_rna_decoding_step_{self.current_epoch if hasattr(self, 'current_epoch') else 0:04d}.pdf",
        )
        logger.info("✓ Saved UMAP of latent representations before RNA decoding")
        plt.close()

    logger.info("Now feeding these latents into RNA decoder...")

    # === DIAGNOSTIC: Compare decoder inputs ===
    logger.info("=== DIAGNOSTIC: Decoder input comparison ===")
    logger.info("SAME-MODAL RNA (RNA→RNA decoder):")
    logger.info(f"  - Latent: RNA encoded ({rna_encoded_latent.shape})")
    logger.info(f"  - Library size tensor shape: {rna_lib_tensor.shape}")
    logger.info(
        f"  - Library size stats: mean={rna_lib_tensor.mean().item():.4f}, std={rna_lib_tensor.std().item():.4f}"
    )
    logger.info(
        f"  - Batch indices: shape={rna_batch_tensor.shape}, unique values={torch.unique(rna_batch_tensor).tolist()}"
    )

    logger.info("\nCOUNTERFACTUAL RNA (Protein→RNA decoder):")
    logger.info(f"  - Latent: Protein encoded ({protein_encoded_latent.shape})")
    logger.info(f"  - Library size tensor shape: {lib_tensor_for_protein_cells.shape}")
    logger.info(
        f"  - Library size stats: mean={lib_tensor_for_protein_cells.mean().item():.4f}, std={lib_tensor_for_protein_cells.std().item():.4f}"
    )
    logger.info(
        f"  - Batch indices: shape={rna_batch_tensor_for_protein.shape}, unique values={torch.unique(rna_batch_tensor_for_protein).tolist()}"
    )

    logger.info("\n✓  LIBRARY SIZE HANDLING:")
    logger.info(
        f"  Same-modal uses PER-CELL RNA library sizes (std={rna_lib_tensor.std().item():.4f})"
    )
    logger.info(
        f"  Counterfactual uses PER-CELL Protein library sizes (std={lib_tensor_for_protein_cells.std().item():.4f})"
    )
    logger.info(
        f"  Both use per-cell library sizes - this should give comparable outputs for similar latents!"
    )

    with torch.no_grad():
        counterfactual_rna_outputs = self.rna_vae.module.generative(
            z=protein_encoded_latent,
            library=lib_tensor_for_protein_cells,
            batch_index=rna_batch_tensor_for_protein,
        )
        rna_px_dist = counterfactual_rna_outputs["px"]
        if hasattr(rna_px_dist, "mu"):
            rna_mu = rna_px_dist.mu.detach().cpu().numpy()
            if hasattr(rna_px_dist, "zi_logits"):
                rna_zi_probs = torch.sigmoid(rna_px_dist.zi_logits).detach().cpu().numpy()
                counterfactual_rna_linear = (1 - rna_zi_probs) * rna_mu
            else:
                counterfactual_rna_linear = rna_mu
        else:
            counterfactual_rna_linear = rna_px_dist.loc.detach().cpu().numpy()

    counterfactual_rna_linear = np.round(counterfactual_rna_linear).astype(np.int32)
    counterfactual_rna_df = pd.DataFrame(counterfactual_rna_linear)
    if counterfactual_rna_df.shape[1] != len(self.rna_vae.adata.var_names):
        logger.error(
            f"Dimension mismatch in RNA counterfactuals: generated {counterfactual_rna_df.shape[1]} features, "
            f"expected {len(self.rna_vae.adata.var_names)} RNA genes"
        )
        raise ValueError(f"RNA counterfactual generation failed: dimension mismatch")

    counterfactual_rna_df.columns = self.rna_vae.adata.var_names
    counterfactual_rna_df.index = adata_prot_save.obs.index
    adata_prot_save.obsm["counterfactual_rna"] = counterfactual_rna_df

    same_modal_adata_rna = AnnData(same_modal_rna_df, obs=adata_rna_save.obs.copy())
    same_modal_adata_protein = AnnData(same_modal_protein_df, obs=adata_prot_save.obs.copy())

    counterfactual_adata_protein = AnnData(counterfactual_protein_df, obs=adata_rna_save.obs.copy())
    counterfactual_adata_rna = AnnData(counterfactual_rna_df, obs=adata_prot_save.obs.copy())

    logger.info("✓ Counterfactual generation completed:")
    logger.info(f"  - Same-modal RNA (ground truth): {same_modal_adata_rna.shape}")
    logger.info(f"  - Counterfactual RNA (Protein→RNA): {counterfactual_adata_rna.shape}")
    logger.info(f"  - Same-modal Protein (ground truth): {same_modal_adata_protein.shape}")
    logger.info(f"  - Counterfactual Protein (RNA→Protein): {counterfactual_adata_protein.shape}")

    return (
        counterfactual_adata_rna,
        counterfactual_adata_protein,
        same_modal_adata_rna,
        same_modal_adata_protein,
    )
