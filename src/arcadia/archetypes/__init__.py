"""Archetype-related utilities."""

from arcadia.archetypes.distances import compute_archetype_distances
from arcadia.archetypes.generation import (
    add_matched_archetype_weight,
    compute_batch_archetype_proportions,
    compute_global_modality_proportions,
    create_unified_archetype_representation,
    filter_extreme_archetypes_by_cross_modal_quality,
    find_optimal_k_across_modalities,
    generate_archetypes_per_batch,
    get_cell_type_colors,
    match_archetypes_across_batches_for_all_k,
    update_archetype_labels,
    validate_batch_archetype_consistency,
)
from arcadia.archetypes.matching import (
    find_best_pair_by_row_matching,
    identify_extreme_archetypes_balanced,
    reorder_rows_to_maximize_diagonal,
    validate_extreme_archetypes_matching,
)
from arcadia.archetypes.visualization import (
    archetype_vs_latent_distances_plot,
    create_tsne_visualization,
    plot_archetype_heatmap,
    plot_archetype_proportions_before_after_matching,
    plot_archetype_umap,
    plot_archetypes_matching,
    plot_cross_modal_archetype_comparison,
    plot_distance_comparison,
    plot_matching_accuracy_by_cell_type,
    visualize_archetype_proportions_analysis,
)

__all__ = [
    "compute_archetype_distances",
    "find_best_pair_by_row_matching",
    "identify_extreme_archetypes_balanced",
    "reorder_rows_to_maximize_diagonal",
    "validate_extreme_archetypes_matching",
    "add_matched_archetype_weight",
    "compute_batch_archetype_proportions",
    "compute_global_modality_proportions",
    "create_unified_archetype_representation",
    "filter_extreme_archetypes_by_cross_modal_quality",
    "find_optimal_k_across_modalities",
    "generate_archetypes_per_batch",
    "get_cell_type_colors",
    "match_archetypes_across_batches_for_all_k",
    "update_archetype_labels",
    "validate_batch_archetype_consistency",
    # Plotting functions from visualization.py
    "archetype_vs_latent_distances_plot",
    "create_tsne_visualization",
    "plot_archetype_heatmap",
    "plot_archetype_proportions_before_after_matching",
    "plot_archetype_umap",
    "plot_archetypes_matching",
    "plot_cross_modal_archetype_comparison",
    "plot_distance_comparison",
    "plot_matching_accuracy_by_cell_type",
    "visualize_archetype_proportions_analysis",
]


__all__.append("evaluate_distance_metrics")
