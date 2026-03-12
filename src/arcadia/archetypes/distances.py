"""Archetype distance calculations."""

import numpy as np
import scipy
from anndata import AnnData


def compute_archetype_distances(adata_rna: AnnData, adata_prot: AnnData, batch_size=1000):
    """Compute archetype distances between RNA and protein data using batched processing to handle large datasets"""
    print("Computing archetype distances with batched processing...")

    rna_vecs = adata_rna.obsm["archetype_vec"].values
    prot_vecs = adata_prot.obsm["archetype_vec"].values

    n_rna = rna_vecs.shape[0]
    n_prot = prot_vecs.shape[0]

    # For very large datasets, compute only diagonal elements (matching pairs)
    if n_rna > 100000 and n_prot > 100000 and n_rna == n_prot:
        print(f"Large dataset detected ({n_rna} samples), computing only diagonal distances...")
        diag_distances = np.zeros(n_rna)
        for i in range(0, n_rna, batch_size):
            end_idx = min(i + batch_size, n_rna)
            diag_distances[i:end_idx] = np.array(
                [
                    scipy.spatial.distance.cosine(rna_vecs[j], prot_vecs[j])
                    for j in range(i, end_idx)
                ]
            )
        # Create a sparse or dummy matrix with just the diagonal populated
        archetype_distances = np.eye(n_rna)  # Placeholder matrix
        np.fill_diagonal(archetype_distances, diag_distances)
    else:
        # Process in batches to avoid memory issues
        archetype_distances = np.zeros((n_rna, n_prot))
        for i in range(0, n_rna, batch_size):
            end_idx = min(i + batch_size, n_rna)
            batch_distances = scipy.spatial.distance.cdist(
                rna_vecs[i:end_idx], prot_vecs, metric="cosine"
            )
            archetype_distances[i:end_idx] = batch_distances
            print(
                f"Processed batch {i//batch_size + 1}/{(n_rna-1)//batch_size + 1}",
                end="\r",
            )

    # Store distance metadata - handled by finalize function

    return archetype_distances
