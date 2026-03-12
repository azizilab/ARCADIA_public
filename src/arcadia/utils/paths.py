"""Path utilities for scripts and notebooks."""

import inspect
from pathlib import Path


def here():
    """Get the directory containing the calling script or current working directory.

    Returns:
        Path: Path to the directory containing the calling script, or current working directory
              if called from a notebook or interactive session.
    """
    try:
        frame = inspect.stack()[1]
        caller_file = frame.filename
        return Path(caller_file).resolve().parent
    except (IndexError, NameError):
        return Path.cwd()
