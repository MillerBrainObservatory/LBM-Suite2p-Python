from . import _version
from .utils import (
    get_metadata,
    get_files,
    ops_from_metadata,
)

__version__ = _version.get_versions()['version']

__all__ = [
    "get_metadata",
    "get_files",
    "ops_from_metadata",
]
