"""General utilities."""

from arcadia.utils.args import parse_pipeline_arguments
from arcadia.utils.environment import get_umap_filtered_fucntion, setup_environment
from arcadia.utils.logging import (
    log_training_summary,
    logger,
    setup_logger,
    setup_mlflow_experiment,
)
from arcadia.utils.metadata import (
    finalize_prepare_data_metadata,
    finalize_preprocess_metadata,
    initialize_pipeline_metadata,
    initialize_train_vae_metadata,
    update_train_vae_metadata,
)

__all__ = [
    "logger",
    "setup_logger",
    "setup_mlflow_experiment",
    "log_training_summary",
    "initialize_pipeline_metadata",
    "finalize_preprocess_metadata",
    "finalize_prepare_data_metadata",
    "initialize_train_vae_metadata",
    "update_train_vae_metadata",
    "parse_pipeline_arguments",
    "setup_environment",
    "get_umap_filtered_fucntion",
]

# Import setup_scvi_patch for easy access
try:
    from arcadia.utils.setup_scvi_patch import patch_scvi_library

    __all__.append("patch_scvi_library")
except ImportError:
    pass
