from __future__ import annotations

from functools import wraps
from typing import Any

import xarray as xr

from .core import (
    CF_STANDARD_NAME_TABLE_URL,
    check_dataset_compliant,
    make_dataset_compliant,
)

_WRAPS_ASSIGNED = ("__module__", "__name__", "__qualname__", "__annotations__")


@xr.register_dataset_accessor("cf")
class CFCoercerAccessor:
    """Dataset-level CF helpers.

    Methods:
    - ``check()``: inspect CF-1.12 metadata issues.
    - ``make_compliant()``: return dataset with safe, automatic fixes applied.
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._ds = xarray_obj

    @wraps(check_dataset_compliant, assigned=_WRAPS_ASSIGNED)
    def check(
        self,
        *,
        cf_version: str = "1.12",
        standard_name_table_xml: str | None = CF_STANDARD_NAME_TABLE_URL,
        cf_area_types_xml: str | None = None,
        cf_region_names_xml: str | None = None,
        cache_tables: bool = False,
        domain: str | None = None,
        fallback_to_heuristic: bool = True,
        pretty_print: bool = False,
    ) -> dict[str, Any] | None:
        """Check CF compliance for this dataset.

        Returns:
        - `dict` when `pretty_print=False`
        - `None` when `pretty_print=True` (report is printed)

        Notes:
        - Uses `cfchecker` against an in-memory NetCDF payload.
        - Falls back to heuristic checks when `cfchecker` cannot run and
          `fallback_to_heuristic=True`.
        """
        return check_dataset_compliant(
            self._ds,
            cf_version=cf_version,
            standard_name_table_xml=standard_name_table_xml,
            cf_area_types_xml=cf_area_types_xml,
            cf_region_names_xml=cf_region_names_xml,
            cache_tables=cache_tables,
            domain=domain,
            fallback_to_heuristic=fallback_to_heuristic,
            pretty_print=pretty_print,
        )

    @wraps(make_dataset_compliant, assigned=_WRAPS_ASSIGNED)
    def make_compliant(self) -> xr.Dataset:
        """Return a new dataset with safe CF-1.12 metadata fixes applied."""
        return make_dataset_compliant(self._ds)
