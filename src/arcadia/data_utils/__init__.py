"""Data loading and preprocessing utilities."""

from arcadia.data_utils.loading import (
    check_tonsil_data_exists,
    determine_dataset_path,
    download_tonsil_data,
    load_adata_latest,
    load_cite_seq_data,
    load_cite_seq_protein,
    load_cite_seq_rna,
    load_tonsil_protein,
    load_tonsil_rna,
    read_legacy_adata,
    save_processed_data,
    validate_adata_requirements,
)
from arcadia.data_utils.preprocessing import (
    analyze_and_visualize,
    apply_batch_correction_pipeline,
    apply_zero_mask,
    balance_datasets,
    balanced_subsample_by_cell_type,
    compute_pca_and_umap,
    convert_to_sparse_csr,
    filter_unwanted_cell_types,
    harmonize_cell_types_names,
    log1p_rna,
    mad_outlier_removal,
    order_cells_by_type,
    preprocess_rna_final_steps,
    preprocess_rna_initial_steps,
    qc_metrics,
    z_normalize_codex,
)
from arcadia.spatial.analysis import spatial_analysis
from arcadia.spatial.neighbors import create_smart_neighbors

__all__ = [
    "load_adata_latest",
    "read_legacy_adata",
    "save_processed_data",
    "validate_adata_requirements",
    "filter_cell_types",
    "filter_unwanted_cell_types",
    "harmonize_cell_types_names",
    "mad_outlier_removal",
    "z_normalize_codex",
    "create_smart_neighbors",
    "balance_datasets",
    "balanced_subsample_by_cell_type",
    "analyze_and_visualize",
    "qc_metrics",
    "spatial_analysis",
    "log1p_rna",
    "preprocess_rna_initial_steps",
    "preprocess_rna_final_steps",
    "order_cells_by_type",
    "compute_pca_and_umap",
    "apply_zero_mask",
    "convert_to_sparse_csr",
    "determine_dataset_path",
    "setup_dataset",
]


__all__.extend(
    [
        "load_cite_seq_data",
        "load_cite_seq_rna",
        "load_cite_seq_protein",
        "load_tonsil_rna",
        "load_tonsil_protein",
        "check_tonsil_data_exists",
        "download_tonsil_data",
    ]
)


__all__.append("apply_batch_correction_pipeline")
