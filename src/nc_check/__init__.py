"""Utilities to coerce xarray datasets toward CF-1.12 compliance."""

from .accessor import CFCoercerAccessor
from .core import (
    CF_STANDARD_NAME_TABLE_URL,
    CF_VERSION,
    check_dataset_compliant,
    make_dataset_compliant,
)
from .ocean import check_ocean_cover, check_time_cover

__all__ = [
    "CFCoercerAccessor",
    "CF_STANDARD_NAME_TABLE_URL",
    "CF_VERSION",
    "check_dataset_compliant",
    "check_ocean_cover",
    "check_time_cover",
    "make_dataset_compliant",
]
