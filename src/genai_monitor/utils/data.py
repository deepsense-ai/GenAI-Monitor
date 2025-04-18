import os
from pathlib import Path
from typing import Optional


def get_absolute_path(relative_path: str, relative_to: Optional[str] = None) -> str:
    """Generate the absolute path for a given relative path.

    If no base path is provided, the function calculates the absolute path relative
    to the location of the current file.

    Args:
        relative_path: The relative path to be resolved.
        relative_to: The base path from which the relative path is resolved. If not provided,
            the path of the current file is used as the base.

    Returns:
        The absolute path as a string.
    """
    if relative_to is None:
        relative_to = os.path.abspath(__file__)
    return str(Path(os.path.dirname(relative_to)) / relative_path)
