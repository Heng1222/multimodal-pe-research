"""Shared package for multimodal PE data, models, and experiments."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("multimodal-pe-research")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
