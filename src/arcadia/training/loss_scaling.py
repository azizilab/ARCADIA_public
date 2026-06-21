# %%
"""Utility functions for caching loss scales."""
import hashlib
import json
from pathlib import Path

from arcadia.utils.logging import setup_logger

logger = setup_logger(level="INFO")


def get_training_params_hash(params: dict) -> str:
    """Computes a SHA256 hash for a dictionary of training parameters."""
    params_string = json.dumps(params, sort_keys=True, indent=None)
    return hashlib.sha256(params_string.encode("utf-8")).hexdigest()


def load_loss_scales_from_cache(training_params_for_hash: dict, cache_path: Path):
    """Load loss scales from cache if they exist for the given parameters."""
    if not cache_path.exists():
        return None

    params_hash = get_training_params_hash(training_params_for_hash)
    try:
        with open(cache_path, "r") as f:
            scales_cache = json.load(f)
        if params_hash in scales_cache:
            logger.info(f"Loaded loss scales from cache using hash {params_hash[:7]}...")
            return scales_cache[params_hash]
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not read scales cache file at {cache_path}: {e}")

    return None


def save_loss_scales_to_cache(training_params_for_hash: dict, loss_scales: dict, cache_path: Path):
    """Save calculated loss scales to the cache."""
    params_hash = get_training_params_hash(training_params_for_hash)
    scales_cache = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                scales_cache = json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning(
                f"Could not decode existing scales cache at {cache_path}. It will be overwritten."
            )

    scales_cache[params_hash] = loss_scales
    try:
        with open(cache_path, "w") as f:
            json.dump(scales_cache, f, indent=4)
        logger.info(f"Saved loss scales to cache at {cache_path}")
    except IOError as e:
        logger.error(f"Could not write to scales cache file at {cache_path}: {e}")
