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
from .formatting import (
    ReportFormat,
    maybe_display_html_report,
    normalize_report_format,
    print_pretty_full_report,
    render_pretty_full_report_html,
    save_html_report,
)
from .ocean import (
    check_ocean_cover as run_ocean_cover_check,
    check_time_cover as run_time_cover_check,
)

_WRAPS_ASSIGNED = ("__module__", "__name__", "__qualname__", "__annotations__")


def _status_kind(status: Any) -> str:
    if isinstance(status, bool):
        return "pass" if status else "fail"
    normalized = str(status).strip().lower()
    if normalized in {"pass", "passed", "ok", "success", "true"}:
        return "pass"
    if normalized in {"fail", "failed", "error", "fatal", "false"}:
        return "fail"
    return "warn"


def _combine_statuses(statuses: list[str]) -> str:
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "warn" for status in statuses):
        return "warn"
    return "pass"


def _count_to_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except Exception:
        return 0


def _status_from_compliance_report(report: dict[str, Any]) -> str:
    checker_error = report.get("checker_error")
    if checker_error is not None:
        return "fail"

    counts = report.get("counts")
    if isinstance(counts, dict):
        fatal = _count_to_int(counts.get("fatal"))
        error = _count_to_int(counts.get("error"))
        warn = _count_to_int(counts.get("warn"))
        if fatal + error > 0:
            return "fail"
        if warn > 0:
            return "warn"
        return "pass"

    finding_statuses: list[str] = []
    for item in report.get("global", []) or []:
        if isinstance(item, dict):
            finding_statuses.append(_status_kind(item.get("severity")))
        else:
            finding_statuses.append("warn")
    for scope in ("coordinates", "variables"):
        entries = report.get(scope)
        if not isinstance(entries, dict):
            continue
        for findings in entries.values():
            if not isinstance(findings, list):
                continue
            for finding in findings:
                if isinstance(finding, dict):
                    finding_statuses.append(_status_kind(finding.get("severity")))
                else:
                    finding_statuses.append("warn")
    if not finding_statuses:
        return "pass"
    return _combine_statuses(finding_statuses)


def _status_from_ocean_report(report: dict[str, Any]) -> str:
    if report.get("mode") == "all_variables":
        grouped = report.get("reports")
        if not isinstance(grouped, dict) or not grouped:
            return _status_kind(report.get("ok"))
        return _combine_statuses(
            [
                _status_from_ocean_report(per_var)
                for per_var in grouped.values()
                if isinstance(per_var, dict)
            ]
        )

    statuses: list[str] = []
    for check_name in ("edge_of_map", "land_ocean_offset", "time_missing"):
        check_report = report.get(check_name)
        if isinstance(check_report, dict):
            statuses.append(_status_kind(check_report.get("status")))
    if not statuses:
        statuses.append(_status_kind(report.get("ok")))
    return _combine_statuses(statuses)


def _status_from_time_cover_report(report: dict[str, Any]) -> str:
    if report.get("mode") == "all_variables":
        grouped = report.get("reports")
        if not isinstance(grouped, dict) or not grouped:
            return _status_kind(report.get("ok"))
        return _combine_statuses(
            [
                _status_from_time_cover_report(per_var)
                for per_var in grouped.values()
                if isinstance(per_var, dict)
            ]
        )
    statuses: list[str] = []
    time_missing = report.get("time_missing")
    if isinstance(time_missing, dict):
        statuses.append(_status_kind(time_missing.get("status")))
    time_format = report.get("time_format")
    if isinstance(time_format, dict):
        statuses.append(_status_kind(time_format.get("status")))
    if not statuses:
        statuses.append(_status_kind(report.get("ok")))
    return _combine_statuses(statuses)


def _compliance_detail(report: dict[str, Any]) -> str:
    counts = report.get("counts")
    if isinstance(counts, dict):
        return (
            f"fatal={_count_to_int(counts.get('fatal'))} "
            f"error={_count_to_int(counts.get('error'))} "
            f"warn={_count_to_int(counts.get('warn'))}"
        )
    checker_error = report.get("checker_error")
    if checker_error is not None:
        return "checker_error=true"
    return "completed"


def _ocean_cover_detail(report: dict[str, Any]) -> str:
    if report.get("mode") == "all_variables":
        checked = _count_to_int(report.get("checked_variable_count"))
        return f"variables={checked}"
    edge = (
        report.get("edge_of_map") if isinstance(report.get("edge_of_map"), dict) else {}
    )
    offset = (
        report.get("land_ocean_offset")
        if isinstance(report.get("land_ocean_offset"), dict)
        else {}
    )
    return (
        f"missing_longitudes={_count_to_int(edge.get('missing_longitude_count'))} "
        f"mismatches={_count_to_int(offset.get('mismatch_count'))}"
    )


def _time_cover_detail(report: dict[str, Any]) -> str:
    if report.get("mode") == "all_variables":
        checked = _count_to_int(report.get("checked_variable_count"))
        return f"variables={checked}"
    time_missing = (
        report.get("time_missing")
        if isinstance(report.get("time_missing"), dict)
        else {}
    )
    time_format = (
        report.get("time_format") if isinstance(report.get("time_format"), dict) else {}
    )
    return (
        f"missing_slices={_count_to_int(time_missing.get('missing_slice_count'))} "
        f"time_format={str(time_format.get('status'))}"
    )


@xr.register_dataset_accessor("check")
class CFCoercerAccessor:
    """Dataset-level CF helpers.

    Methods:
    - ``compliance()``: inspect CF-1.12 metadata issues.
    - ``make_cf_compliant()``: return dataset with safe, automatic fixes applied.
    - ``ocean_cover()``: run fast ocean-coverage sanity checks.
    - ``time_cover()``: run time-coverage checks.
    - ``all()``: run selected checks and create one combined report.
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
        report_format: ReportFormat = "auto",
        report_html_file: str | Path | None = None,
    ) -> dict[str, Any] | str | None:
        """Check CF compliance for this dataset.

        Returns:
        - environment-dependent output when `report_format="auto"`
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
        report_format: ReportFormat = "auto",
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
        report_format: ReportFormat = "auto",
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
        report_format: ReportFormat = "auto",
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
        report_format: ReportFormat = "auto",
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
        report_format: ReportFormat = "auto",
        report_html_file: str | Path | None = None,
    ) -> dict[str, Any] | str | None:
        """Backward-compatible alias for `time_cover()`."""
        return self.time_cover(
            var_name=var_name,
            time_name=time_name,
            report_format=report_format,
            report_html_file=report_html_file,
        )

    def all(
        self,
        *,
        compliance: bool = True,
        ocean_cover: bool = True,
        time_cover: bool = True,
        cf_version: str = "1.12",
        standard_name_table_xml: str | None = CF_STANDARD_NAME_TABLE_URL,
        cf_area_types_xml: str | None = None,
        cf_region_names_xml: str | None = None,
        cache_tables: bool = False,
        domain: str | None = None,
        fallback_to_heuristic: bool = True,
        conventions: str | list[str] | tuple[str, ...] | None = None,
        var_name: str | None = None,
        lon_name: str | None = None,
        lat_name: str | None = None,
        time_name: str | None = "time",
        check_edge_of_map: bool = True,
        check_land_ocean_offset: bool = True,
        report_format: ReportFormat = "auto",
        report_html_file: str | Path | None = None,
    ) -> dict[str, Any] | str | None:
        """Run selected checks and return a combined report."""
        resolved_format = normalize_report_format(report_format)
        if report_html_file is not None and resolved_format != "html":
            raise ValueError(
                "`report_html_file` is only valid when report_format='html'."
            )

        enabled = {
            "compliance": bool(compliance),
            "ocean_cover": bool(ocean_cover),
            "time_cover": bool(time_cover),
        }
        if not any(enabled.values()):
            raise ValueError("At least one check must be enabled.")

        reports: dict[str, dict[str, Any]] = {}
        check_summary: list[dict[str, str]] = []

        if enabled["compliance"]:
            compliance_report = self.compliance(
                cf_version=cf_version,
                standard_name_table_xml=standard_name_table_xml,
                cf_area_types_xml=cf_area_types_xml,
                cf_region_names_xml=cf_region_names_xml,
                cache_tables=cache_tables,
                domain=domain,
                fallback_to_heuristic=fallback_to_heuristic,
                conventions=conventions,
                report_format="python",
            )
            compliance_dict = (
                compliance_report if isinstance(compliance_report, dict) else {}
            )
            reports["compliance"] = compliance_dict
            check_summary.append(
                {
                    "check": "compliance",
                    "status": _status_from_compliance_report(compliance_dict),
                    "detail": _compliance_detail(compliance_dict),
                }
            )

        if enabled["ocean_cover"]:
            ocean_report = self.ocean_cover(
                var_name=var_name,
                lon_name=lon_name,
                lat_name=lat_name,
                time_name=time_name,
                check_edge_of_map=check_edge_of_map,
                check_land_ocean_offset=check_land_ocean_offset,
                report_format="python",
            )
            ocean_dict = ocean_report if isinstance(ocean_report, dict) else {}
            reports["ocean_cover"] = ocean_dict
            check_summary.append(
                {
                    "check": "ocean_cover",
                    "status": _status_from_ocean_report(ocean_dict),
                    "detail": _ocean_cover_detail(ocean_dict),
                }
            )

        if enabled["time_cover"]:
            time_cover_report = self.time_cover(
                var_name=var_name,
                time_name=time_name,
                report_format="python",
            )
            time_cover_dict = (
                time_cover_report if isinstance(time_cover_report, dict) else {}
            )
            reports["time_cover"] = time_cover_dict
            check_summary.append(
                {
                    "check": "time_cover",
                    "status": _status_from_time_cover_report(time_cover_dict),
                    "detail": _time_cover_detail(time_cover_dict),
                }
            )

        statuses = [entry["status"] for entry in check_summary]
        overall_status = _combine_statuses(statuses)
        full_report = {
            "checks_enabled": enabled,
            "check_summary": check_summary,
            "reports": reports,
            "summary": {
                "checks_run": len(check_summary),
                "failing_checks": sum(1 for status in statuses if status == "fail"),
                "warnings_or_skips": sum(1 for status in statuses if status == "warn"),
                "overall_status": overall_status,
                "overall_ok": overall_status == "pass",
            },
        }

        if resolved_format == "python":
            return full_report
        if resolved_format == "tables":
            print_pretty_full_report(full_report)
            return None

        html_report = render_pretty_full_report_html(full_report)
        save_html_report(html_report, report_html_file)
        maybe_display_html_report(html_report)
        return html_report

    def full(
        self,
        *,
        compliance: bool = True,
        ocean_cover: bool = True,
        time_cover: bool = True,
        cf_version: str = "1.12",
        standard_name_table_xml: str | None = CF_STANDARD_NAME_TABLE_URL,
        cf_area_types_xml: str | None = None,
        cf_region_names_xml: str | None = None,
        cache_tables: bool = False,
        domain: str | None = None,
        fallback_to_heuristic: bool = True,
        conventions: str | list[str] | tuple[str, ...] | None = None,
        var_name: str | None = None,
        lon_name: str | None = None,
        lat_name: str | None = None,
        time_name: str | None = "time",
        check_edge_of_map: bool = True,
        check_land_ocean_offset: bool = True,
        report_format: ReportFormat = "auto",
        report_html_file: str | Path | None = None,
    ) -> dict[str, Any] | str | None:
        """Backward-compatible alias for `all()`."""
        return self.all(
            compliance=compliance,
            ocean_cover=ocean_cover,
            time_cover=time_cover,
            cf_version=cf_version,
            standard_name_table_xml=standard_name_table_xml,
            cf_area_types_xml=cf_area_types_xml,
            cf_region_names_xml=cf_region_names_xml,
            cache_tables=cache_tables,
            domain=domain,
            fallback_to_heuristic=fallback_to_heuristic,
            conventions=conventions,
            var_name=var_name,
            lon_name=lon_name,
            lat_name=lat_name,
            time_name=time_name,
            check_edge_of_map=check_edge_of_map,
            check_land_ocean_offset=check_land_ocean_offset,
            report_format=report_format,
            report_html_file=report_html_file,
        )
