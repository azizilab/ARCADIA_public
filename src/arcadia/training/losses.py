"""Loss functions for training."""

import numpy as np
import torch
from torch.nn.functional import normalize

from arcadia.archetypes import identify_extreme_archetypes_balanced
from arcadia.utils.logging import logger


def compute_pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distances between two sets of points.

    Args:
        x: Tensor [N, D]
        y: Tensor [M, D]

    Returns:
        Distance matrix [N, M]
    """
    x_norm = (x**2).sum(dim=1, keepdim=True)  # [N, 1]
    y_norm = (y**2).sum(dim=1, keepdim=True)  # [M, 1]
    xy = torch.matmul(x, y.transpose(-2, -1))  # [N, M]
    return torch.sqrt(torch.clamp(x_norm + y_norm.transpose(-2, -1) - 2 * xy, min=1e-8))


def mmd_loss(x, y, sigma=1.0):
    """Compute (unbiased) MMD between two sets of samples x and y.

    Args:
        x, y: Tensor samples [N, D] and [M, D]
        sigma: RBF bandwidth. If None or <=0, use median heuristic with multi-kernel

    Returns:
        MMD loss (scalar tensor)
    """

    def rbf_kernel(a, b, sigma_val):
        dist = compute_pairwise_distances(a, b) ** 2
        return torch.exp(-dist / (2 * sigma_val**2))

    def mmd_with_sigma(s):
        n = x.shape[0]
        m = y.shape[0]
        Kxx = rbf_kernel(x, x, s)
        Kyy = rbf_kernel(y, y, s)
        Kxy = rbf_kernel(x, y, s)

        # Unbiased MMD: exclude diagonal terms
        if n > 1:
            Kxx_off_diag = Kxx - torch.diag(torch.diag(Kxx))
            mean_Kxx = Kxx_off_diag.sum() / (n * (n - 1))
        else:
            mean_Kxx = torch.tensor(0.0, device=x.device)

        if m > 1:
            Kyy_off_diag = Kyy - torch.diag(torch.diag(Kyy))
            mean_Kyy = Kyy_off_diag.sum() / (m * (m - 1))
        else:
            mean_Kyy = torch.tensor(0.0, device=y.device)

        mean_Kxy = Kxy.mean()
        mmd_val = mean_Kxx + mean_Kyy - 2 * mean_Kxy
        return torch.clamp(mmd_val, min=1e-12)

    # Adaptive sigma via median heuristic if requested
    if sigma is None or (isinstance(sigma, (float, int)) and sigma <= 0):
        combined = torch.cat([x, y], dim=0)
        dists = compute_pairwise_distances(combined, combined)
        mask = ~torch.eye(dists.shape[0], dtype=bool, device=dists.device)
        med = torch.median(dists[mask])

        if torch.isnan(med) or torch.isinf(med) or med <= 1e-8:
            med = torch.tensor(1.0, device=combined.device)

        # Multi-kernel around median for robustness
        sigmas = [med * scale for scale in [0.5, 1.0, 2.0]]
        mmd_vals = [mmd_with_sigma(s) for s in sigmas]
        return sum(mmd_vals) / len(mmd_vals)
    else:
        return mmd_with_sigma(float(sigma))


def run_cell_type_clustering_loss(
    adata,
    latent_mean,
    indices,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    plot_flag=False,
    modality_type=None,
    return_components=False,
):
    """Calculate cell type clustering loss to preserve cell type relationships.

    Args:
        adata: AnnData object with cell type information
        latent_mean: Latent mean representation from the VAE (already computed from module() call)
        indices: Indices of cells to use
        device: Device to use for calculations
        plot_flag: Whether to print debug information
        modality_type: String indicating modality type ('protein' or 'rna')

    Returns:
        Cell type clustering loss tensor
    """
    # Get device from latent_mean to ensure consistency
    if isinstance(latent_mean, torch.Tensor):
        device = latent_mean.device
    
    # Check if this is protein data
    is_protein = modality_type == "protein" if modality_type is not None else False

    cell_types = torch.tensor(adata[indices].obs["cell_types"].cat.codes.values).to(device)

    # Combine cell types and latent representations from both modalities
    # Calculate centroid for each cell type
    unique_cell_types = torch.unique(cell_types)
    num_cell_types = len(unique_cell_types)

    # Skip the cell type clustering loss if there's only one cell type
    cell_type_clustering_loss = torch.tensor(0.0).to(device)

    if num_cell_types < 1:
        raise ValueError("No cell types found in the data")

    # Calculate centroids for each cell type in latent space
    centroids = []
    cells_per_type = []
    type_to_idx = {}

    for i, cell_type in enumerate(unique_cell_types):
        mask = cell_types == cell_type
        type_to_idx[cell_type.item()] = i
        if mask.sum() > 0:
            cells = latent_mean[mask]
            centroid = cells.mean(dim=0)
            centroids.append(centroid)
            cells_per_type.append(cells)

    centroids = torch.stack(centroids)

    # Get original structure from archetype vectors
    # Compute the structure matrix once and cache it
    all_cell_types = adata.obs["cell_types"].cat.codes.values
    all_unique_types = np.unique(all_cell_types)

    # Get centroids in archetype space for each cell type
    original_centroids = []
    for ct in all_unique_types:
        mask = all_cell_types == ct
        if mask.sum() > 0:
            ct_archetype_vecs = adata.obsm["archetype_vec"][mask]
            original_centroids.append(np.mean(ct_archetype_vecs, axis=0))

    # Convert to torch tensor
    original_centroids = torch.tensor(np.array(original_centroids), dtype=torch.float32).to(device)

    # Compute affinity/structure matrix (using Gaussian kernel)
    # Use gradient-compatible distance computation instead of torch.cdist
    dists = compute_pairwise_distances(original_centroids, original_centroids)
    sigma = dists.std()
    original_structure_matrix = torch.exp(-(dists**2) / (2 * sigma**2))

    # Set diagonal to 0 to focus on between-cluster relationships
    original_structure_matrix = original_structure_matrix * (
        1 - torch.eye(len(all_unique_types), device=device)
    )

    # Compute current structure matrix in latent space
    # Use same sigma as original for consistency
    latent_dists = compute_pairwise_distances(centroids, centroids)
    sigma = latent_dists.std()
    current_structure_matrix = torch.exp(-(latent_dists**2) / (2 * sigma**2))

    # Set diagonal to 0 to focus on between-cluster relationships
    current_structure_matrix = current_structure_matrix * (
        1 - torch.eye(len(centroids), device=device)
    )

    # Now compute the structure preservation loss
    structure_preservation_loss = 0.0
    count = 0

    # For each cell type in the batch, compare its relationships
    for i, type_i in enumerate(unique_cell_types):
        if type_i.item() < len(original_structure_matrix):
            for j, type_j in enumerate(unique_cell_types):
                if i != j and type_j.item() < len(original_structure_matrix):
                    # Get original and current affinity values
                    orig_affinity = original_structure_matrix[type_i.item(), type_j.item()]
                    current_affinity = current_structure_matrix[i, j]

                    # Square difference
                    diff = (orig_affinity - current_affinity) ** 2
                    structure_preservation_loss += diff
                    count += 1

    if count > 0:
        structure_preservation_loss = structure_preservation_loss / count

    # Calculate within-cluster cohesion and variance regularization
    cohesion_loss = 0.0
    total_cells = 0

    # Calculate variances for each cluster in each dimension
    valid_clusters = [cells for cells in cells_per_type if len(cells) > 1]
    if len(valid_clusters) > 1:
        # Calculate variance for each cluster in each dimension
        cluster_variances = []
        for cells in valid_clusters:
            # Compute variance per dimension
            vars_per_dim = torch.var(cells, dim=0)
            cluster_variances.append(vars_per_dim)

        # Stack variances for all clusters
        cluster_variances = torch.stack(cluster_variances)

        # Calculate mean variance across all clusters for each dimension
        mean_variance_per_dim = torch.mean(cluster_variances, dim=0)

        # Log the mean variance for some dimensions
        if mean_variance_per_dim.shape[0] > 3 and plot_flag:
            logger.info(
                f"Mean variances for first 3 dimensions: {mean_variance_per_dim[:3].detach().cpu().numpy().round(4)}"
            )
            logger.info(f"Dimension variance std: {torch.std(mean_variance_per_dim).item():.4f}")
    else:
        # If we don't have enough clusters with multiple cells, just use standard cohesion
        for i, cells in enumerate(cells_per_type):
            if len(cells) > 1:
                dists = torch.norm(cells - centroids[i], dim=1)
                cohesion_loss += dists.mean()
                total_cells += 1

    if total_cells > 0:
        cohesion_loss = cohesion_loss / total_cells

    # Normalize the cohesion loss by the average inter-centroid distance
    # This makes it scale-invariant
    avg_inter_centroid_dist = compute_pairwise_distances(centroids, centroids).mean()
    if avg_inter_centroid_dist > 0:
        normalized_cohesion_loss = cohesion_loss / avg_inter_centroid_dist
    else:
        normalized_cohesion_loss = cohesion_loss

    # Apply different weights based on modality
    if is_protein:
        # For protein data, prioritize cohesion over structure preservation
        # Calculate separation loss for protein clusters
        protein_separation_loss = 0.0
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                # Encourage greater distance between different protein cluster centroids
                protein_separation_loss += torch.exp(-torch.norm(centroids[i] - centroids[j]))

        # Adjust weights for protein data
        cell_type_clustering_loss = (
            2.0 * structure_preservation_loss
            + 1.0 * normalized_cohesion_loss
            + 1.0 * protein_separation_loss
        )

        if return_components:
            return {
                "total_loss": cell_type_clustering_loss,
                "structure_preservation": structure_preservation_loss,
                "cohesion": normalized_cohesion_loss,
                "separation": protein_separation_loss,
                "modality": "protein",
            }
    else:
        # For RNA data, also calculate separation loss
        rna_separation_loss = 0.0
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                # Encourage greater distance between different RNA cluster centroids
                rna_separation_loss += torch.exp(-torch.norm(centroids[i] - centroids[j]))

        # Adjust weights for RNA data (add separation component)
        cell_type_clustering_loss = (
            2.0 * structure_preservation_loss + normalized_cohesion_loss + 1.0 * rna_separation_loss
        )

        if return_components:
            return {
                "total_loss": cell_type_clustering_loss,
                "structure_preservation": structure_preservation_loss,
                "cohesion": normalized_cohesion_loss,
                "separation": rna_separation_loss,
                "modality": "rna",
            }

    return cell_type_clustering_loss


def calculate_modality_balance_loss(rna_components, protein_components):
    """
    Calculate loss that punishes differences between modality-specific loss components.

    Args:
        rna_components: Dictionary with RNA loss components
        protein_components: Dictionary with protein loss components

    Returns:
        modality_balance_loss: Tensor representing the balance loss
    """
    # Calculate differences between corresponding components
    structure_diff = (
        torch.abs(
            rna_components["structure_preservation"] - protein_components["structure_preservation"]
        )
        ** 4
    )
    cohesion_diff = torch.abs(rna_components["cohesion"] - protein_components["cohesion"]) ** 4

    # For separation, now both modalities have this component
    separation_diff = (
        torch.abs(rna_components["separation"] - protein_components["separation"]) ** 4
    )

    # Combine all component differences with fixed weights
    total_balance_loss = structure_diff + cohesion_diff + separation_diff

    return total_balance_loss


def cn_distribution_separation_loss(
    adata,
    latent,
    cell_type_key="cell_types",
    cn_key="CN",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    sigma=None,
    global_step=None,
):
    """
    For each cell type, penalize CNs that have overlapping distributions in latent space.
    """
    obs = adata.obs
    cell_types = obs[cell_type_key].cat.categories
    total_loss = 0.0
    count = 0

    # Track sigmas for different cell types
    calculated_sigmas = []

    for ct in cell_types:
        mask_ct = obs[cell_type_key] == ct
        if mask_ct.sum() < 2:
            continue
        # Convert numpy mask to PyTorch boolean mask
        mask_indices = torch.tensor(np.where(mask_ct.values)[0], device=device)
        # Index latent directly with the indices to maintain computational graph
        latent_ct = latent[mask_indices]

        # Auto-tune sigma if not provided
        current_sigma = sigma
        if current_sigma is None:
            # Calculate pairwise distances in the current batch
            # Use gradient-compatible distance computation instead of torch.cdist
            pairwise_dist = compute_pairwise_distances(latent_ct, latent_ct)
            # Get median distance (excluding self-distances)
            mask = ~torch.eye(pairwise_dist.shape[0], dtype=bool, device=pairwise_dist.device)
            current_sigma = torch.median(pairwise_dist[mask])

            # Check for NaN/Inf in computed sigma
            if torch.isnan(current_sigma) or torch.isinf(current_sigma) or current_sigma <= 0:
                logger.warning(f"Invalid sigma computed for cell type {ct}: {current_sigma}")
                current_sigma = torch.tensor(1.0, device=device)  # Fallback value

            calculated_sigmas.append(current_sigma.item())

            # Print sigma on first step
            if global_step is not None and global_step == 0:
                modality = "RNA" if "rna" in str(adata).lower() else "Protein"
                cell_type = ct
                logger.debug(
                    f"[Auto-sigma] {modality} - Cell type {cell_type}: sigma = {current_sigma.item():.4f}"
                )

        cn_labels = obs.loc[mask_ct, cn_key]
        cn_categories = cn_labels.cat.categories
        if len(cn_categories) < 2:
            continue
        # For each pair of CNs, compute MMD
        for i, cni in enumerate(cn_categories):
            mask_i = cn_labels == cni
            # Convert to indices and use to index the already filtered latent_ct
            i_indices = torch.tensor(np.where(mask_i.values)[0], device=device)
            xi = latent_ct[i_indices]
            if xi.shape[0] < 2:
                continue
            for j, cnj in enumerate(cn_categories):
                if j <= i:
                    continue
                mask_j = cn_labels == cnj
                # Same for the second CN group
                j_indices = torch.tensor(np.where(mask_j.values)[0], device=device)
                xj = latent_ct[j_indices]
                if xj.shape[0] < 2:
                    continue
                # Calculate MMD
                mmd = mmd_loss(xi, xj, sigma=current_sigma)

                # Check for NaN/Inf in MMD
                if torch.isnan(mmd) or torch.isinf(mmd):
                    logger.warning(f"MMD loss returned NaN/Inf for cell type {ct}, CNs {cni}-{cnj}")
                    continue

                # Use negative exponential to penalize small MMD values
                # This directly encourages separation and is more stable than 1/MMD
                separation_loss = torch.exp(-5.0 * mmd)  # Exponential decay rate can be tuned

                # Check for NaN/Inf in separation loss
                if torch.isnan(separation_loss) or torch.isinf(separation_loss):
                    logger.warning(
                        f"Separation loss returned NaN/Inf for cell type {ct}, CNs {cni}-{cnj}"
                    )
                    continue

                total_loss += separation_loss
                count += 1

    # Log average sigma if calculated
    if calculated_sigmas and global_step is not None and global_step == 0:
        avg_sigma = sum(calculated_sigmas) / len(calculated_sigmas)
        modality = "RNA" if "rna" in str(adata).lower() else "Protein"
        logger.debug(
            f"[Auto-sigma] {modality} - Average sigma across all cell types: {avg_sigma:.4f}"
        )

    if count == 0:
        return torch.tensor(0.0, device=device)

    # Ensure the result is finite
    result = total_loss / count
    if torch.isnan(result) or torch.isinf(result):
        logger.warning(
            f"cn_distribution_separation_loss returned NaN/Inf: total_loss={total_loss}, count={count}"
        )
        return torch.tensor(0.0, device=device)

    return result


def calculate_cross_modal_cell_type_loss(
    adata_rna,
    adata_prot,
    rna_latent_mean,
    protein_latent_mean,
    rna_indices,
    prot_indices,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    sigma=1.0,
):
    """Calculate MMD-based cross-modal cell type clustering loss to align cell type distributions
    between RNA and protein modalities in the latent space.

    This replaces the previous centroid-based approach with a proper Maximum Mean Discrepancy (MMD)
    loss that matches entire cell type distributions rather than just centroids.

    Args:
        adata_rna: RNA AnnData object with cell type information
        adata_prot: Protein AnnData object with cell type information
        rna_latent_mean: RNA latent mean representation from the VAE
        protein_latent_mean: Protein latent mean representation from the VAE
        rna_indices: Indices of RNA cells to use
        prot_indices: Indices of protein cells to use
        device: Device to use for calculations
        sigma: RBF kernel bandwidth parameter for MMD

    Returns:
        MMD-based cross-modal cell type clustering loss tensor
    """
    # Use string labels to align cell types across modalities (avoid independent cat codes)
    rna_labels = adata_rna[rna_indices].obs["major_cell_types"].astype(str).values
    prot_labels = adata_prot[prot_indices].obs["major_cell_types"].astype(str).values

    # Skip if either modality has only one cell type
    if len(set(rna_labels)) <= 1 or len(set(prot_labels)) <= 1:
        return torch.tensor(0.0, device=device)

    # Find common cell types by name
    common_cell_types = list(set(rna_labels).intersection(set(prot_labels)))

    if len(common_cell_types) == 0:
        return torch.tensor(0.0, device=device)

    total_mmd_loss = torch.tensor(0.0, device=device)
    valid_comparisons = 0

    # For each common cell type, minimize MMD between RNA and protein distributions
    for cell_type_name in common_cell_types:
        rna_mask_np = rna_labels == cell_type_name
        prot_mask_np = prot_labels == cell_type_name

        # Need at least 2 samples from each modality for meaningful MMD
        if rna_mask_np.sum() >= 2 and prot_mask_np.sum() >= 2:
            rna_mask = torch.tensor(rna_mask_np, device=device, dtype=torch.bool)
            prot_mask = torch.tensor(prot_mask_np, device=device, dtype=torch.bool)

            rna_cells = rna_latent_mean[rna_mask]
            prot_cells = protein_latent_mean[prot_mask]

            # Calculate MMD between distributions of the same cell type across modalities
            mmd = mmd_loss(rna_cells, prot_cells, sigma=sigma)
            total_mmd_loss += mmd
            valid_comparisons += 1

    # Return average MMD across all valid cell type comparisons
    if valid_comparisons > 0:
        return total_mmd_loss / valid_comparisons
    else:
        return torch.tensor(0.0, device=device)


def extreme_archetypes_loss(
    rna_batch, prot_batch, latent_distances, logger_, to_print=False, rna_vae=None, protein_vae=None
):
    rna_extreme_mask, _, _ = identify_extreme_archetypes_balanced(
        rna_batch["archetype_vec"],
        rna_vae.adata[rna_batch["labels"]],
        logger_=logger_,
        percentile=90,
        to_print=to_print,
    )
    prot_extreme_mask, _, _ = identify_extreme_archetypes_balanced(
        prot_batch["archetype_vec"],
        protein_vae.adata[prot_batch["labels"]],
        logger_=logger_,
        percentile=90,
        to_print=to_print,
    )
    rna_extreme_indices = torch.where(rna_extreme_mask)[0]
    prot_extreme_indices = torch.where(prot_extreme_mask)[0]

    # Skip if not enough extreme cells
    if len(rna_extreme_indices) < 2 or len(prot_extreme_indices) < 2:
        return torch.tensor(0.0, device=latent_distances.device), 0.0
    # cosine similarity
    # Normalize vectors to unit length for cosine similarity calculation
    rna_extreme_archetypes = normalize(rna_batch["archetype_vec"][rna_extreme_indices], dim=1)
    prot_extreme_archetypes = normalize(prot_batch["archetype_vec"][prot_extreme_indices], dim=1)

    # This is Euclidean distance, not cosine distance
    # For cosine distance, we need 1 - cosine similarity
    cosine_sim = torch.mm(rna_extreme_archetypes, prot_extreme_archetypes.T)
    archetype_dis = 1.0 - cosine_sim  # Convert to cosine distance

    # Get subset of latent distances for extreme cells only
    latent_extreme_distances = latent_distances[rna_extreme_indices, :][:, prot_extreme_indices]
    latent_extreme_distances = torch.clamp(
        latent_extreme_distances, max=torch.quantile(latent_extreme_distances, 0.90)
    )

    # For each RNA extreme cell, find closest and farthest protein cell in archetype space
    rna_to_prot_closest = torch.argmin(archetype_dis, dim=1)
    rna_to_prot_farthest = torch.argmax(archetype_dis, dim=1)
    # For each Protein extreme cell, find closest and farthest RNA cell in archetype space
    prot_to_rna_closest = torch.argmin(archetype_dis, dim=0)
    prot_to_rna_farthest = torch.argmax(archetype_dis, dim=0)

    # Get latent distances for these specific pairs
    if (
        archetype_dis[torch.arange(len(rna_extreme_indices)), rna_to_prot_farthest].sum()
        < archetype_dis[torch.arange(len(rna_extreme_indices)), rna_to_prot_closest].sum()
    ):
        raise ValueError("Farthest pairs are closer than closest pairs")
    rna_to_prot_closest_latent = latent_extreme_distances[
        torch.arange(len(rna_extreme_indices)), rna_to_prot_closest
    ]
    rna_to_prot_farthest_latent = latent_extreme_distances[
        torch.arange(len(rna_extreme_indices)), rna_to_prot_farthest
    ]
    prot_to_rna_closest_latent = latent_extreme_distances[
        prot_to_rna_closest, torch.arange(len(prot_extreme_indices))
    ]
    prot_to_rna_farthest_latent = latent_extreme_distances[
        prot_to_rna_farthest, torch.arange(len(prot_extreme_indices))
    ]

    # IMPROVED MARGINS: Use adaptive margins based on actual data distribution
    # Instead of fixed percentiles, use statistics from the current batch
    latent_mean = latent_extreme_distances.mean()
    latent_std = latent_extreme_distances.std()

    # Set margins relative to the data distribution (much more flexible)
    margin_close = latent_std * 0.01  # close cells need to be close within a reasonable margin
    faraway_threshold = latent_std * 1  # far away cell just need to be far away

    # Ensure margins are reasonable
    margin_close = torch.clamp(margin_close, max=latent_extreme_distances.max())
    faraway_threshold = torch.clamp(faraway_threshold, max=latent_extreme_distances.max())

    # SOFTER CONSTRAINTS: Use smooth losses instead of hard thresholds
    # Replace ReLU with smoother functions that don't create discontinuities

    # Close loss: encourage similar archetype cells to be close in latent space
    close_loss_rna = torch.nn.functional.softplus(
        rna_to_prot_closest_latent - margin_close, beta=0.5
    ).mean()
    close_loss_prot = torch.nn.functional.softplus(
        prot_to_rna_closest_latent - margin_close, beta=0.5
    ).mean()
    close_loss = close_loss_rna + close_loss_prot

    # Far loss: encourage dissimilar archetype cells to maintain distance
    # as long at the furthest are more distant than the faraway_threshold, we will get 0 (softplus of negative number is 0)
    far_loss_rna = torch.nn.functional.softplus(
        faraway_threshold - rna_to_prot_farthest_latent, beta=0.5
    ).mean()
    far_loss_prot = torch.nn.functional.softplus(
        faraway_threshold - prot_to_rna_farthest_latent, beta=0.5
    ).mean()
    far_loss = far_loss_rna + far_loss_prot

    # BALANCED WEIGHTING: Prevent far loss from dominating
    # The close loss is more important for alignment than far loss
    total_loss = close_loss + far_loss * 0.1  # Reduce far loss influence

    # Calculate alignment percentage using already computed data
    # Calculate mean pairwise distance between all extreme cells
    mean_extreme_distance = latent_extreme_distances.mean()
    alignment_threshold = mean_extreme_distance / 4.0

    # Count how many closest matching pairs have latent distance <= alignment_threshold
    rna_aligned_count = (rna_to_prot_closest_latent <= alignment_threshold).sum().item()
    prot_aligned_count = (prot_to_rna_closest_latent <= alignment_threshold).sum().item()

    # Calculate total pairs and aligned pairs
    total_pairs = len(rna_extreme_indices) + len(prot_extreme_indices)
    aligned_pairs = rna_aligned_count + prot_aligned_count

    # Calculate alignment percentage
    alignment_percentage = (aligned_pairs / total_pairs * 100.0) if total_pairs > 0 else 0.0

    if to_print:
        logger_.info(
            f"\nImproved Archetype-guided loss summary:\n"
            f"├─ Close loss: {close_loss.item():.4f} (encourage alignment)\n"
            f"├─ Far loss: {far_loss.item():.4f} (maintain structure)\n"
            f"├─ Total loss: {total_loss.item():.4f}\n"
            f"├─ Extreme cells: RNA={len(rna_extreme_indices)}, Protein={len(prot_extreme_indices)}\n"
            f"├─ close margin={margin_close.item():.4f}, far threshold={faraway_threshold.item():.4f}\n"
            f"├─ Latent stats: mean={latent_mean.item():.4f}, std={latent_std.item():.4f}\n"
            f"├─ Close/Far ratio: {(close_loss / (far_loss + 1e-8)).item():.2f}\n"
            f"├─ Alignment threshold (half mean): {alignment_threshold.item():.4f}\n"
            f"├─ RNA aligned pairs: {rna_aligned_count}/{len(rna_extreme_indices)}\n"
            f"├─ Protein aligned pairs: {prot_aligned_count}/{len(prot_extreme_indices)}\n"
            f"└─ Total alignment: {alignment_percentage:.1f}%"
        )

    return total_loss, alignment_percentage
