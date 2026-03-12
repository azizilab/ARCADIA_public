"""Training utilities and metrics."""

from arcadia.training.gradnorm import GradNorm
from arcadia.training.loss_scaling import load_loss_scales_from_cache, save_loss_scales_to_cache
from arcadia.training.metrics import (
    calculate_post_training_metrics,
    extract_training_metrics_from_history,
    mixing_score,
    process_post_training_metrics,
)
from arcadia.training.train_vae import train_vae
from arcadia.training.utils import (
    clear_memory,
    ensure_correct_dtype,
    generate_post_training_visualizations,
    generate_target_cluster_structure_from_cell_types,
    generate_target_cluster_structure_unsuprvised,
    get_memory_usage,
    handle_error,
    is_already_integer,
    log_memory_usage,
    log_parameters,
    match_cells_and_calculate_distances,
    process_latent_spaces,
    select_gene_likelihood,
    simulate_counts_zero_inflated,
    train_vae_for_archetype_generation,
    transfer_to_integer_range_nb,
    transfer_to_integer_range_normal,
    validate_scvi_training_mixin,
)

__all__ = [
    "GradNorm",
    "train_vae",
    "select_gene_likelihood",
    "process_latent_spaces",
    "match_cells_and_calculate_distances",
    "calculate_post_training_metrics",
    "extract_training_metrics_from_history",
    "process_post_training_metrics",
    "simulate_counts_zero_inflated",
    "transfer_to_integer_range_nb",
    "transfer_to_integer_range_normal",
    "is_already_integer",
    "ensure_correct_dtype",
    "mixing_score",
    "load_loss_scales_from_cache",
    "save_loss_scales_to_cache",
    "clear_memory",
    "generate_post_training_visualizations",
    "handle_error",
    "log_parameters",
    "log_memory_usage",
    "get_memory_usage",
]

# Import VAE training functions (wrapped imports from vae_training_utils)
# These are imported at the end of utils.py, so import them directly

__all__.extend(
    [
        "generate_target_cluster_structure_from_cell_types",
        "generate_target_cluster_structure_unsuprvised",
        "train_vae_for_archetype_generation",
    ]
)

# Import validate_scvi_training_mixin

__all__.append("validate_scvi_training_mixin")

# Import DualVAETrainingPlan

__all__.append("DualVAETrainingPlan")

# Import loss functions

__all__.extend(
    [
        "calculate_cross_modal_cell_type_loss",
        "calculate_modality_balance_loss",
        "cn_distribution_separation_loss",
        "extreme_archetypes_loss",
        "mmd_loss",
        "run_cell_type_clustering_loss",
    ]
)
