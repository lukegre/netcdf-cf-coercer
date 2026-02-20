from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import xarray as xr

from .core import (
    CF_STANDARD_NAME_TABLE_URL,
    check_dataset_compliant,
    make_dataset_compliant,
)
from .formatting import ReportFormat
from .ocean import (
    check_ocean_cover as run_ocean_cover_check,
    check_time_cover as run_time_cover_check,
)

_WRAPS_ASSIGNED = ("__module__", "__name__", "__qualname__", "__annotations__")


@xr.register_dataset_accessor("check")
class CFCoercerAccessor:
    """Dataset-level CF helpers.

    Methods:
    - ``compliance()``: inspect CF-1.12 metadata issues.
    - ``make_cf_compliant()``: return dataset with safe, automatic fixes applied.
    - ``ocean_cover()``: run fast ocean-coverage sanity checks.
    - ``time_cover()``: run time-coverage checks.
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._ds = xarray_obj

    @wraps(check_dataset_compliant, assigned=_WRAPS_ASSIGNED)
    def compliance(
        self,
        *,
        cf_version: str = "1.12",
        standard_name_table_xml: str | None = CF_STANDARD_NAME_TABLE_URL,
        cf_area_types_xml: str | None = None,
        cf_region_names_xml: str | None = None,
        cache_tables: bool = False,
        domain: str | None = None,
        fallback_to_heuristic: bool = True,
        conventions: str | list[str] | tuple[str, ...] | None = None,
        report_format: ReportFormat = "python",
        report_html_file: str | Path | None = None,
    ) -> dict[str, Any] | str | None:
        """Check CF compliance for this dataset.

        Returns:
        - `dict` when `report_format="python"`
        - `None` when `report_format="tables"` (report is printed)
        - `str` when `report_format="html"` (HTML report)

        Notes:
        - Uses `cfchecker` against an in-memory NetCDF payload.
        - Falls back to heuristic checks when `cfchecker` cannot run and
          `fallback_to_heuristic=True`.
        - Supports extra convention checks such as ``ferret`` via
          `conventions="cf,ferret"` (or list/tuple).
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
            conventions=conventions,
            report_format=report_format,
            report_html_file=report_html_file,
        )

    @wraps(check_dataset_compliant, assigned=_WRAPS_ASSIGNED)
    def cf(
        self,
        *,
        cf_version: str = "1.12",
        standard_name_table_xml: str | None = CF_STANDARD_NAME_TABLE_URL,
        cf_area_types_xml: str | None = None,
        cf_region_names_xml: str | None = None,
        cache_tables: bool = False,
        domain: str | None = None,
        fallback_to_heuristic: bool = True,
        conventions: str | list[str] | tuple[str, ...] | None = None,
        report_format: ReportFormat = "python",
        report_html_file: str | Path | None = None,
    ) -> dict[str, Any] | str | None:
        """Backward-compatible alias for `compliance()`."""
        return self.compliance(
            cf_version=cf_version,
            standard_name_table_xml=standard_name_table_xml,
            cf_area_types_xml=cf_area_types_xml,
            cf_region_names_xml=cf_region_names_xml,
            cache_tables=cache_tables,
            domain=domain,
            fallback_to_heuristic=fallback_to_heuristic,
            conventions=conventions,
            report_format=report_format,
            report_html_file=report_html_file,
        )

    @wraps(make_dataset_compliant, assigned=_WRAPS_ASSIGNED)
    def make_cf_compliant(self) -> xr.Dataset:
        """Return a new dataset with safe CF-1.12 metadata fixes applied."""
        return make_dataset_compliant(self._ds)

    @wraps(make_dataset_compliant, assigned=_WRAPS_ASSIGNED)
    def comply(self) -> xr.Dataset:
        """Alias for `make_cf_compliant()`."""
        return self.make_cf_compliant()

    @wraps(make_dataset_compliant, assigned=_WRAPS_ASSIGNED)
    def make_compliant(self) -> xr.Dataset:
        """Backward-compatible alias for `make_cf_compliant()`."""
        return self.make_cf_compliant()

    @wraps(run_ocean_cover_check, assigned=_WRAPS_ASSIGNED)
    def ocean_cover(
        self,
        *,
        var_name: str | None = None,
        lon_name: str | None = None,
        lat_name: str | None = None,
        time_name: str | None = "time",
        check_edge_of_map: bool = True,
        check_land_ocean_offset: bool = True,
        report_format: ReportFormat = "tables",
        report_html_file: str | Path | None = None,
        check_edge_sliver: bool | None = None,
        check_time_missing: bool | None = None,
    ) -> dict[str, Any] | str | None:
        """Run fast ocean-coverage checks for a gridded ocean variable."""
        return run_ocean_cover_check(
            self._ds,
            var_name=var_name,
            lon_name=lon_name,
            lat_name=lat_name,
            time_name=time_name,
            check_edge_of_map=check_edge_of_map,
            check_land_ocean_offset=check_land_ocean_offset,
            report_format=report_format,
            report_html_file=report_html_file,
            check_edge_sliver=check_edge_sliver,
            check_time_missing=check_time_missing,
        )

    @wraps(run_ocean_cover_check, assigned=_WRAPS_ASSIGNED)
    def check_ocean_cover(
        self,
        *,
        var_name: str | None = None,
        lon_name: str | None = None,
        lat_name: str | None = None,
        time_name: str | None = "time",
        check_edge_of_map: bool = True,
        check_land_ocean_offset: bool = True,
        report_format: ReportFormat = "tables",
        report_html_file: str | Path | None = None,
        check_edge_sliver: bool | None = None,
        check_time_missing: bool | None = None,
    ) -> dict[str, Any] | str | None:
        """Backward-compatible alias for `ocean_cover()`."""
        return self.ocean_cover(
            var_name=var_name,
            lon_name=lon_name,
            lat_name=lat_name,
            time_name=time_name,
            check_edge_of_map=check_edge_of_map,
            check_land_ocean_offset=check_land_ocean_offset,
            report_format=report_format,
            report_html_file=report_html_file,
            check_edge_sliver=check_edge_sliver,
            check_time_missing=check_time_missing,
        )

    @wraps(run_time_cover_check, assigned=_WRAPS_ASSIGNED)
    def time_cover(
        self,
        *,
        var_name: str | None = None,
        time_name: str | None = "time",
        report_format: ReportFormat = "tables",
        report_html_file: str | Path | None = None,
    ) -> dict[str, Any] | str | None:
        """Run time-coverage checks for a variable with a time dimension."""
        return run_time_cover_check(
            self._ds,
            var_name=var_name,
            time_name=time_name,
            report_format=report_format,
            report_html_file=report_html_file,
        )

    @wraps(run_time_cover_check, assigned=_WRAPS_ASSIGNED)
    def check_time_cover(
        self,
        *,
        var_name: str | None = None,
        time_name: str | None = "time",
        report_format: ReportFormat = "tables",
        report_html_file: str | Path | None = None,
    ) -> dict[str, Any] | str | None:
        """Backward-compatible alias for `time_cover()`."""
        return self.time_cover(
            var_name=var_name,
            time_name=time_name,
            report_format=report_format,
            report_html_file=report_html_file,
        )
