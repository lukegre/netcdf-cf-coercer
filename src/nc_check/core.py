from __future__ import annotations

import re
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .formatting import (
    ReportFormat,
    maybe_display_html_report,
    normalize_report_format,
    print_pretty_report,
    render_pretty_report_html,
    save_html_report,
)
from .standard_names import augment_issues_with_standard_name_suggestions

CF_VERSION = "CF-1.12"
CF_STANDARD_NAME_TABLE_URL = "https://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml"

_LAT_NAMES = {"lat", "latitude", "y"}
_LON_NAMES = {"lon", "longitude", "x"}
_TIME_NAMES = {"time", "t"}
_NON_COMPLIANT_CATEGORIES = ("FATAL", "ERROR", "WARN")
_VALID_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SUPPORTED_CONVENTIONS = ("cf", "ferret")
_DEFAULT_CONVENTIONS = ("cf", "ferret")
_CF_ATTR_CASE_KEYS = (
    "units",
    "standard_name",
    "long_name",
    "axis",
    "calendar",
    "coordinates",
    "bounds",
    "grid_mapping",
    "cell_methods",
    "cell_measures",
    "positive",
)


@dataclass(frozen=True)
class AxisGuess:
    dim: str
    axis_type: str


def _empty_report(cf_version: str = "1.12") -> dict[str, Any]:
    return {
        "cf_version": _format_cf_version(cf_version),
        "engine": "cfchecker",
        "engine_status": "skipped",
        "check_method": "conventions_only",
        "global": [],
        "coordinates": {},
        "variables": {},
        "suggestions": {"variables": {}},
        "notes": [],
    }


def _normalize_requested_conventions(
    conventions: str | list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if conventions is None:
        return _DEFAULT_CONVENTIONS

    if isinstance(conventions, str):
        raw = [part.strip() for part in conventions.split(",")]
    else:
        raw = [str(part).strip() for part in conventions]

    selected: list[str] = []
    for name in raw:
        if not name:
            continue
        lowered = name.lower()
        if lowered not in selected:
            selected.append(lowered)

    if not selected:
        raise ValueError("At least one convention must be selected.")

    unknown = sorted(set(selected) - set(_SUPPORTED_CONVENTIONS))
    if unknown:
        allowed = ", ".join(_SUPPORTED_CONVENTIONS)
        invalid = ", ".join(unknown)
        raise ValueError(
            f"Unsupported conventions: {invalid}. Supported conventions: {allowed}."
        )

    return tuple(selected)


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _is_numeric_dtype(dtype: Any) -> bool:
    return np.issubdtype(dtype, np.number)


def _guess_axis_for_dim(ds: xr.Dataset, dim: str) -> AxisGuess | None:
    dim_norm = _normalize_name(dim)
    coord = ds.coords.get(dim)
    coord_name_norm = _normalize_name(coord.name) if coord is not None else dim_norm

    if dim_norm in _TIME_NAMES or coord_name_norm in _TIME_NAMES:
        return AxisGuess(dim=dim, axis_type="time")

    if dim_norm in _LAT_NAMES or coord_name_norm in _LAT_NAMES:
        return AxisGuess(dim=dim, axis_type="lat")

    if dim_norm in _LON_NAMES or coord_name_norm in _LON_NAMES:
        return AxisGuess(dim=dim, axis_type="lon")

    if coord is not None:
        standard_name = _normalize_name(str(coord.attrs.get("standard_name", "")))
        units = _normalize_name(str(coord.attrs.get("units", "")))

        if standard_name == "latitude" or units == "degrees_north":
            return AxisGuess(dim=dim, axis_type="lat")
        if standard_name == "longitude" or units == "degrees_east":
            return AxisGuess(dim=dim, axis_type="lon")
        if standard_name == "time":
            return AxisGuess(dim=dim, axis_type="time")

    return None


def _expected_coord_attrs(axis_type: str) -> dict[str, str]:
    if axis_type == "lat":
        return {
            "standard_name": "latitude",
            "long_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        }
    if axis_type == "lon":
        return {
            "standard_name": "longitude",
            "long_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        }
    if axis_type == "time":
        return {
            "standard_name": "time",
            "axis": "T",
        }
    return {}


def _heuristic_check_dataset(ds: xr.Dataset) -> dict[str, Any]:
    """Fallback checker used only when cfchecker is unavailable."""
    issues: dict[str, Any] = {
        "cf_version": CF_VERSION,
        "engine": "cfchecker",
        "engine_status": "unavailable",
        "check_method": "heuristic",
        "global": [],
        "coordinates": {},
        "variables": {},
        "notes": [],
    }

    conventions = ds.attrs.get("Conventions")
    if conventions != CF_VERSION:
        issues["global"].append(
            {
                "item": "Conventions",
                "current": conventions,
                "expected": CF_VERSION,
                "suggested_fix": "set_global_attr",
            }
        )

    axis_guesses: dict[str, AxisGuess] = {}
    for dim in ds.dims:
        dim = str(dim)
        guess = _guess_axis_for_dim(ds, dim)
        if guess:
            axis_guesses[dim] = guess
        else:
            issues["notes"].append(
                f"Could not infer CF axis type for dimension '{dim}'."
            )

    for dim, guess in axis_guesses.items():
        coord = ds.coords.get(dim)
        if coord is None:
            issues["coordinates"][dim] = [
                {
                    "item": "missing_dimension_coordinate",
                    "current": None,
                    "expected": f"coordinate variable named '{dim}'",
                    "suggested_fix": "create_dimension_coordinate",
                }
            ]
            continue

        coord_issues = []
        expected_attrs = _expected_coord_attrs(guess.axis_type)
        for key, expected_value in expected_attrs.items():
            current_value = coord.attrs.get(key)
            if current_value != expected_value:
                coord_issues.append(
                    {
                        "item": f"coord_attr:{key}",
                        "current": current_value,
                        "expected": expected_value,
                        "suggested_fix": "set_coord_attr",
                    }
                )

        if guess.axis_type in {"lat", "lon"} and not _is_numeric_dtype(coord.dtype):
            coord_issues.append(
                {
                    "item": "coord_dtype",
                    "current": str(coord.dtype),
                    "expected": "numeric",
                    "suggested_fix": "convert_coord_dtype",
                }
            )

        if coord_issues:
            issues["coordinates"][dim] = coord_issues

    for var_name, da in ds.data_vars.items():
        var_issues = []
        if not _VALID_NAME_RE.match(str(var_name)):
            var_issues.append(
                {
                    "item": "invalid_variable_name",
                    "current": var_name,
                    "expected": "start with a letter, then letters/digits/underscore",
                    "suggested_fix": "rename_variable",
                }
            )
        if _is_numeric_dtype(da.dtype) and da.attrs.get("units") is None:
            var_issues.append(
                {
                    "item": "missing_units_attr",
                    "current": None,
                    "expected": "UDUNITS-compatible units string for dimensional quantities",
                    "suggested_fix": "set_variable_attr",
                }
            )
        if da.attrs.get("standard_name") is None and da.attrs.get("long_name") is None:
            var_issues.append(
                {
                    "item": "missing_standard_or_long_name",
                    "current": None,
                    "expected": "at least one of: standard_name, long_name",
                    "suggested_fix": "set_variable_attr",
                }
            )
        if da.dtype == object:
            var_issues.append(
                {
                    "item": "object_dtype_variable",
                    "current": str(da.dtype),
                    "expected": "netCDF-compatible primitive dtype",
                    "suggested_fix": "cast_variable_dtype",
                }
            )
        std_name = da.attrs.get("standard_name")
        if std_name is not None and da.attrs.get("units") is None:
            var_issues.append(
                {
                    "item": "missing_units_for_standard_name",
                    "current": None,
                    "expected": "units consistent with standard_name canonical_units",
                    "suggested_fix": "set_variable_attr",
                }
            )
        if var_issues:
            issues["variables"][var_name] = var_issues

    for dim, guess in axis_guesses.items():
        if guess.axis_type != "time":
            continue
        coord = ds.coords.get(dim)
        if coord is None:
            continue
        time_issues = issues["coordinates"].setdefault(dim, [])
        if coord.attrs.get("units") is None:
            time_issues.append(
                {
                    "item": "coord_attr:units",
                    "current": None,
                    "expected": "time units e.g. 'days since 1970-01-01'",
                    "suggested_fix": "set_coord_attr",
                }
            )

    return issues


def _append_coordinate_finding(
    issues: dict[str, Any], coord_name: str, finding: dict[str, Any]
) -> None:
    coord_findings = issues.setdefault("coordinates", {})
    items = coord_findings.setdefault(coord_name, [])
    if isinstance(items, list):
        items.append(finding)


def _append_variable_finding(
    issues: dict[str, Any], var_name: str, finding: dict[str, Any]
) -> None:
    var_findings = issues.setdefault("variables", {})
    items = var_findings.setdefault(var_name, [])
    if isinstance(items, list):
        items.append(finding)


def _case_mismatched_attr_names(
    attrs: dict[str, Any], expected_keys: tuple[str, ...]
) -> list[tuple[str, str]]:
    mismatches: list[tuple[str, str]] = []
    attr_names = [str(name) for name in attrs]
    lowered_to_names: dict[str, list[str]] = {}
    for name in attr_names:
        lowered_to_names.setdefault(name.lower(), []).append(name)

    for expected in expected_keys:
        if expected in attrs:
            continue
        candidates = lowered_to_names.get(expected.lower(), [])
        for candidate in candidates:
            if candidate != expected:
                mismatches.append((candidate, expected))
    return mismatches


def _normalize_attr_key_case(
    attrs: dict[str, Any], expected_keys: tuple[str, ...]
) -> dict[str, Any]:
    normalized = deepcopy(attrs)
    keys_in_order = list(normalized.keys())

    for expected in expected_keys:
        expected_lower = expected.lower()
        matching_keys = [
            key
            for key in keys_in_order
            if isinstance(key, str) and key.lower() == expected_lower
        ]
        if not matching_keys:
            continue

        if expected in normalized:
            for key in matching_keys:
                if key != expected:
                    normalized.pop(key, None)
            continue

        source_key = next((key for key in matching_keys if key != expected), None)
        if source_key is None:
            continue
        normalized[expected] = normalized[source_key]
        for key in matching_keys:
            if key != expected:
                normalized.pop(key, None)

    return normalized


def _apply_cf_attribute_case_checks(ds: xr.Dataset, issues: dict[str, Any]) -> None:
    for coord_name, coord in ds.coords.items():
        mismatches = _case_mismatched_attr_names(coord.attrs, _CF_ATTR_CASE_KEYS)
        for actual_key, expected_key in mismatches:
            _append_coordinate_finding(
                issues,
                str(coord_name),
                {
                    "severity": "WARN",
                    "convention": "cf",
                    "item": "attr_case_mismatch",
                    "message": (
                        f"Coordinate '{coord_name}' uses attribute '{actual_key}' but "
                        f"CF expects '{expected_key}'."
                    ),
                    "current": actual_key,
                    "expected": expected_key,
                    "suggested_fix": "rename_attr_lowercase",
                },
            )

    for var_name, da in ds.data_vars.items():
        mismatches = _case_mismatched_attr_names(da.attrs, _CF_ATTR_CASE_KEYS)
        for actual_key, expected_key in mismatches:
            _append_variable_finding(
                issues,
                str(var_name),
                {
                    "severity": "WARN",
                    "convention": "cf",
                    "item": "attr_case_mismatch",
                    "message": (
                        f"Variable '{var_name}' uses attribute '{actual_key}' but "
                        f"CF expects '{expected_key}'."
                    ),
                    "current": actual_key,
                    "expected": expected_key,
                    "suggested_fix": "rename_attr_lowercase",
                },
            )


def _apply_ferret_convention_checks(ds: xr.Dataset, issues: dict[str, Any]) -> None:
    for coord_name, coord in ds.coords.items():
        sources: dict[str, Any] = {}
        attrs_fill = coord.attrs.get("_FillValue")
        if attrs_fill is not None:
            sources["attrs"] = attrs_fill
        if "_FillValue" in coord.encoding:
            encoding_fill = coord.encoding.get("_FillValue")
            if encoding_fill is not None:
                sources["encoding"] = encoding_fill

        if not sources:
            continue

        detail_parts = [f"{where}={value!r}" for where, value in sources.items()]
        _append_coordinate_finding(
            issues,
            str(coord_name),
            {
                "severity": "FATAL",
                "convention": "ferret",
                "item": "coord_fillvalue_forbidden",
                "message": (
                    f"Coordinate '{coord_name}' has forbidden _FillValue "
                    f"({', '.join(detail_parts)})."
                ),
                "current": sources,
                "expected": "no _FillValue in coordinate attrs or encoding",
                "suggested_fix": "clear_coordinate_fillvalue",
            },
        )


def _apply_selected_convention_checks(
    ds: xr.Dataset, issues: dict[str, Any], conventions: tuple[str, ...]
) -> None:
    if "cf" in conventions:
        _apply_cf_attribute_case_checks(ds, issues)
    if "ferret" in conventions:
        _apply_ferret_convention_checks(ds, issues)


def _recompute_counts(issues: dict[str, Any]) -> None:
    counts = {"fatal": 0, "error": 0, "warn": 0}
    for entries in list((issues.get("coordinates") or {}).values()) + list(
        (issues.get("variables") or {}).values()
    ):
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity", "")).upper()
            if severity == "FATAL":
                counts["fatal"] += 1
            elif severity == "ERROR":
                counts["error"] += 1
            elif severity == "WARN":
                counts["warn"] += 1

    global_entries = issues.get("global") or []
    if isinstance(global_entries, list):
        for item in global_entries:
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity", "")).upper()
            if severity == "FATAL":
                counts["fatal"] += 1
            elif severity == "ERROR":
                counts["error"] += 1
            elif severity == "WARN":
                counts["warn"] += 1
    issues["counts"] = counts


def _as_netcdf_bytes(ds: xr.Dataset) -> bytes:
    # cfchecker is metadata-oriented; avoid materializing potentially large,
    # lazy-backed arrays by serializing a compact metadata-only shadow dataset.
    payload_ds = _build_cfchecker_payload_dataset(ds)
    data = payload_ds.to_netcdf(path=None)
    if isinstance(data, memoryview):
        return data.tobytes()
    return bytes(data)


def _placeholder_value_for_dtype(dtype: np.dtype[Any]) -> Any:
    if np.issubdtype(dtype, np.bool_):
        return False
    if np.issubdtype(dtype, np.number):
        return dtype.type(0)
    if np.issubdtype(dtype, np.datetime64):
        return np.datetime64("1970-01-01")
    if np.issubdtype(dtype, np.timedelta64):
        return np.timedelta64(0, "s")
    if np.issubdtype(dtype, np.bytes_):
        return b""
    if np.issubdtype(dtype, np.str_):
        return ""
    return ""


def _to_serializable_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    # netCDF encoders cannot represent object dtype directly.
    if dtype == np.dtype("O"):
        return np.dtype("S1")
    return dtype


def _dummy_array_for_variable(
    var: xr.Variable, reduced_dim_sizes: dict[str, int]
) -> np.ndarray[Any, Any]:
    dtype = _to_serializable_dtype(np.dtype(var.dtype))
    shape = tuple(reduced_dim_sizes[str(dim)] for dim in var.dims)
    return np.full(shape, _placeholder_value_for_dtype(dtype), dtype=dtype)


def _build_cfchecker_payload_dataset(ds: xr.Dataset) -> xr.Dataset:
    reduced_dim_sizes = {
        str(dim): (0 if int(size) == 0 else 1) for dim, size in ds.sizes.items()
    }
    data_vars = {
        str(name): xr.Variable(
            dims=da.variable.dims,
            data=_dummy_array_for_variable(da.variable, reduced_dim_sizes),
            attrs=deepcopy(da.attrs),
        )
        for name, da in ds.data_vars.items()
    }
    coords = {
        str(name): xr.Variable(
            dims=coord.variable.dims,
            data=_dummy_array_for_variable(coord.variable, reduced_dim_sizes),
            attrs=deepcopy(coord.attrs),
        )
        for name, coord in ds.coords.items()
    }
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=deepcopy(ds.attrs))


def _format_cf_version(version: str) -> str:
    if version.startswith("CF-"):
        return version
    return f"CF-{version}"


def _translate_cfchecker_results(
    results: dict[str, Any], version: str, ds: xr.Dataset
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "cf_version": _format_cf_version(version),
        "engine": "cfchecker",
        "engine_status": "ok",
        "check_method": "cfchecker",
        "global": [],
        "coordinates": {},
        "variables": {},
        "suggestions": {"variables": {}},
        "notes": [],
    }

    counts = {category: 0 for category in _NON_COMPLIANT_CATEGORIES}
    global_results = results.get("global", {})
    for category in _NON_COMPLIANT_CATEGORIES:
        for message in global_results.get(category, []):
            out["global"].append({"severity": category, "message": message})
            counts[category] += 1

    for var_name, var_results in results.get("variables", {}).items():
        var_findings = []
        for category in _NON_COMPLIANT_CATEGORIES:
            for message in var_results.get(category, []):
                var_findings.append({"severity": category, "message": message})
                counts[category] += 1
        if var_findings:
            if var_name in ds.coords:
                out["coordinates"][var_name] = var_findings
            else:
                out["variables"][var_name] = var_findings

    out["counts"] = {
        "fatal": counts["FATAL"],
        "error": counts["ERROR"],
        "warn": counts["WARN"],
    }
    return out


def _run_cfchecker_on_dataset(
    ds: xr.Dataset,
    *,
    cf_version: str = "1.12",
    standard_name_table_xml: str | None = None,
    cf_area_types_xml: str | None = None,
    cf_region_names_xml: str | None = None,
    cache_tables: bool = False,
) -> dict[str, Any]:
    # Import inside function so package can still be used for make_compliant
    # even if cfchecker/cfunits system dependencies are not installed.
    with warnings.catch_warnings():
        # cfchecker currently emits Python 3.12+ SyntaxWarning from legacy
        # escape sequences in its regex literals. Suppress only that module.
        warnings.filterwarnings(
            "ignore",
            message=r"invalid escape sequence .*",
            category=SyntaxWarning,
            module=r"cfchecker\.cfchecks",
        )
        from cfchecker import cfchecks as cfchecks

    payload = _as_netcdf_bytes(ds)
    fake_filename = "__in_memory__.nc"
    original_dataset_ctor = cfchecks.netCDF4.Dataset

    def _patched_dataset(path: str, *args: Any, **kwargs: Any) -> Any:
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == fake_filename and mode == "r":
            return original_dataset_ctor("inmemory", "r", memory=payload)
        return original_dataset_ctor(path, *args, **kwargs)

    checker_kwargs: dict[str, Any] = {
        "version": cf_version,
        "cacheTables": cache_tables,
        "silent": True,
    }
    if standard_name_table_xml is not None:
        checker_kwargs["cfStandardNamesXML"] = standard_name_table_xml
    if cf_area_types_xml is not None:
        checker_kwargs["cfAreaTypesXML"] = cf_area_types_xml
    if cf_region_names_xml is not None:
        checker_kwargs["cfRegionNamesXML"] = cf_region_names_xml

    cfchecks.netCDF4.Dataset = _patched_dataset
    try:
        checker = cfchecks.CFChecker(**checker_kwargs)
        results = checker.checker(fake_filename)
    finally:
        cfchecks.netCDF4.Dataset = original_dataset_ctor

    return _translate_cfchecker_results(results=results, version=cf_version, ds=ds)


def check_dataset_compliant(
    ds: xr.Dataset,
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
    """Run compliance checks for selected conventions (e.g. CF, Ferret)."""
    resolved_format = normalize_report_format(report_format)
    if report_html_file is not None and resolved_format != "html":
        raise ValueError("`report_html_file` is only valid when report_format='html'.")

    selected_conventions = _normalize_requested_conventions(conventions)

    if "cf" in selected_conventions:
        try:
            issues = _run_cfchecker_on_dataset(
                ds,
                cf_version=cf_version,
                standard_name_table_xml=standard_name_table_xml,
                cf_area_types_xml=cf_area_types_xml,
                cf_region_names_xml=cf_region_names_xml,
                cache_tables=cache_tables,
            )
            augment_issues_with_standard_name_suggestions(
                ds,
                issues,
                standard_name_table_xml,
                domain=domain,
            )
        except Exception as exc:
            if not fallback_to_heuristic:
                raise
            issues = _heuristic_check_dataset(ds)
            issues["suggestions"] = {"variables": {}}
            augment_issues_with_standard_name_suggestions(
                ds,
                issues,
                standard_name_table_xml,
                domain=domain,
            )
            issues["checker_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            issues["notes"].append(
                "cfchecker could not run; returned heuristic checks instead."
            )
    else:
        issues = _empty_report(cf_version=cf_version)
        issues["notes"].append(
            "CF checks skipped because 'cf' convention was not selected."
        )

    issues["conventions_checked"] = list(selected_conventions)
    _apply_selected_convention_checks(ds, issues, selected_conventions)
    _recompute_counts(issues)

    if resolved_format == "tables":
        print_pretty_report(issues)
        return None
    if resolved_format == "html":
        html_report = render_pretty_report_html(issues)
        save_html_report(html_report, report_html_file)
        maybe_display_html_report(html_report)
        return html_report
    return issues


def make_dataset_compliant(ds: xr.Dataset) -> xr.Dataset:
    """Return a copy of the dataset with safe CF-1.12 compliance fixes."""
    out = ds.copy(deep=True)

    out.attrs = deepcopy(out.attrs)
    out.attrs["Conventions"] = CF_VERSION

    for var_name in out.data_vars:
        out[var_name].attrs = _normalize_attr_key_case(
            deepcopy(out[var_name].attrs), _CF_ATTR_CASE_KEYS
        )
    for coord_name in out.coords:
        out[coord_name].attrs = _normalize_attr_key_case(
            deepcopy(out[coord_name].attrs), _CF_ATTR_CASE_KEYS
        )

    axis_guesses: dict[str, AxisGuess] = {}
    for dim in out.dims:
        dim = str(dim)
        guess = _guess_axis_for_dim(out, dim)
        if guess:
            axis_guesses[dim] = guess

    for dim in list(out.dims):
        if dim not in axis_guesses:
            continue

        if dim not in out.coords:
            size = out.sizes[dim]
            out = out.assign_coords({dim: np.arange(size)})

        coord = out.coords[dim]
        new_attrs = deepcopy(coord.attrs)
        dim = str(dim)
        for key, val in _expected_coord_attrs(axis_guesses[dim].axis_type).items():
            new_attrs[key] = val

        if axis_guesses[dim].axis_type in {"lat", "lon"} and not _is_numeric_dtype(
            coord.dtype
        ):
            # Preserve laziness for dask-backed coordinates by casting through xarray.
            coerced = coord.astype(float).copy(deep=False)
            coerced.attrs = new_attrs
            out = out.assign_coords({dim: coerced})
        else:
            out[dim].attrs = new_attrs

    # Ferret (and some other tools) can fail when coordinate variables include
    # _FillValue. Explicitly disable it for all coordinates.
    for coord_name in out.coords:
        coord_attrs = deepcopy(out[coord_name].attrs)
        coord_attrs.pop("_FillValue", None)
        out[coord_name].attrs = coord_attrs
        out[coord_name].encoding = {"_FillValue": None}

    return out
