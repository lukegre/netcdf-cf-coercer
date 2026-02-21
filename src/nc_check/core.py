from __future__ import annotations

import warnings
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
import re
from typing import Any, Literal, TypeAlias, cast

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
from .heuristic import (
    AxisGuess,
    expected_coord_attrs,
    guess_axis_for_dim,
    heuristic_check_dataset,
    is_numeric_dtype,
)
from .standard_names import augment_issues_with_standard_name_suggestions

CF_VERSION = "CF-1.12"
CF_STANDARD_NAME_TABLE_URL = "https://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml"
ComplianceEngine = Literal["auto", "cfchecker", "cfcheck", "heuristic"]
NormalizedComplianceEngine: TypeAlias = Literal["auto", "cfchecker", "heuristic"]
ConventionName: TypeAlias = Literal["cf", "ferret"]
StandardNameDomain: TypeAlias = Literal[
    "ocean",
    "atmosphere",
    "land",
    "cryosphere",
    "biogeochemistry",
]
_CFCHECKER_INSTALL_HINT = (
    "cfchecker is not installed. Install optional CF dependencies "
    "with `uv sync --extra cf`."
)

_NON_COMPLIANT_CATEGORIES = ("FATAL", "ERROR", "WARN")
_SUPPORTED_CONVENTIONS: tuple[ConventionName, ...] = ("cf", "ferret")
_DEFAULT_CONVENTIONS: tuple[ConventionName, ...] = ("cf", "ferret")
_TIME_UNITS_RE = re.compile(
    r"^\s*(seconds?|minutes?|hours?|days?|months?|years?)\s+since\s+.+$",
    re.IGNORECASE,
)
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


def _empty_report(cf_version: str = "1.12") -> dict[str, Any]:
    return {
        "cf_version": _format_cf_version(cf_version),
        "engine": "none",
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
) -> tuple[ConventionName, ...]:
    if conventions is None:
        return _DEFAULT_CONVENTIONS

    if isinstance(conventions, str):
        raw = [part.strip().lower() for part in conventions.split(",")]
    else:
        raw = [str(part).strip().lower() for part in conventions]

    selected_raw = [name for name in raw if name]
    if not selected_raw:
        raise ValueError("At least one convention must be selected.")

    unknown = sorted(set(selected_raw) - set(_SUPPORTED_CONVENTIONS))
    if unknown:
        allowed = ", ".join(_SUPPORTED_CONVENTIONS)
        invalid = ", ".join(unknown)
        raise ValueError(
            f"Unsupported conventions: {invalid}. Supported conventions: {allowed}."
        )

    selected: list[ConventionName] = []
    for name in selected_raw:
        convention = cast(ConventionName, name)
        if convention not in selected:
            selected.append(convention)
    return tuple(selected)


def _normalize_requested_engine(engine: str | None) -> NormalizedComplianceEngine:
    if engine is None:
        return "auto"
    normalized = str(engine).strip().lower()
    if normalized == "cfcheck":
        return "cfchecker"
    if normalized not in {"auto", "cfchecker", "heuristic"}:
        raise ValueError(
            "Unsupported engine. Expected one of: auto, cfchecker, cfcheck, heuristic."
        )
    return cast(NormalizedComplianceEngine, normalized)


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


def _first_non_null_value(values: np.ndarray[Any, Any]) -> Any | None:
    for value in values.reshape(-1):
        if value is None:
            continue
        if isinstance(value, np.datetime64) and np.isnat(value):
            continue
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            continue
        return value
    return None


def _is_cftime_datetime(value: Any) -> bool:
    module_name = getattr(value.__class__, "__module__", "")
    return module_name.startswith("cftime")


def _time_value_type(values: np.ndarray[Any, Any]) -> str:
    dtype = values.dtype
    if np.issubdtype(dtype, np.datetime64):
        return "datetime64"
    if np.issubdtype(dtype, np.floating):
        return "float"
    if np.issubdtype(dtype, np.integer):
        return "int"
    if np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.bytes_):
        return "string"

    sample = _first_non_null_value(values)
    if sample is None:
        return str(dtype)
    if isinstance(sample, np.datetime64) or isinstance(sample, (datetime, date)):
        return "datetime"
    if _is_cftime_datetime(sample):
        return "cftime"
    if isinstance(sample, (float, np.floating)):
        return "float"
    if isinstance(sample, (int, np.integer)) and not isinstance(
        sample, (bool, np.bool_)
    ):
        return "int"
    if isinstance(sample, (str, bytes, np.str_, np.bytes_)):
        return "string"
    return type(sample).__name__


def _is_time_decoded_by_xarray(values: np.ndarray[Any, Any]) -> bool:
    if np.issubdtype(values.dtype, np.datetime64):
        return True
    sample = _first_non_null_value(values)
    if sample is None:
        return False
    if isinstance(sample, np.datetime64) or isinstance(sample, (datetime, date)):
        return True
    return _is_cftime_datetime(sample)


def _time_coord_candidates(ds: xr.Dataset) -> list[str]:
    names: set[str] = set()
    for dim in ds.dims:
        dim_name = str(dim)
        guess = guess_axis_for_dim(ds, dim_name)
        if guess and guess.axis_type == "time" and dim_name in ds.coords:
            names.add(dim_name)

    for coord_name, coord in ds.coords.items():
        name = str(coord_name)
        lowered_name = name.strip().lower()
        lowered_standard_name = (
            str(coord.attrs.get("standard_name", "")).strip().lower()
        )
        axis = str(coord.attrs.get("axis", "")).strip().upper()
        if (
            lowered_name in {"time", "t"}
            or lowered_standard_name == "time"
            or axis == "T"
        ):
            names.add(name)
    return sorted(names)


def _apply_cf_time_units_and_type_checks(
    ds: xr.Dataset, issues: dict[str, Any]
) -> None:
    for coord_name in _time_coord_candidates(ds):
        coord = ds.coords.get(coord_name)
        if coord is None:
            continue
        values = np.asarray(coord.values)
        value_type = _time_value_type(values)
        decoded = _is_time_decoded_by_xarray(values)
        units: Any = coord.attrs.get("units")
        if units is None:
            units = coord.encoding.get("units")
        units_text = None if units is None else str(units)

        if decoded:
            continue
        if value_type not in {"int", "float"}:
            _append_coordinate_finding(
                issues,
                coord_name,
                {
                    "severity": "ERROR",
                    "convention": "cf",
                    "item": "time_coord_type_invalid",
                    "message": (
                        f"Time coordinate '{coord_name}' has non-time value type "
                        f"'{value_type}' (dtype={values.dtype})."
                    ),
                    "current": {
                        "value_type": value_type,
                        "dtype": str(values.dtype),
                        "units": units_text,
                    },
                    "expected": (
                        "decoded datetime/cftime values or numeric offsets with "
                        "CF-style units (e.g. 'days since 1970-01-01')."
                    ),
                    "suggested_fix": "set_coord_attr",
                },
            )
            continue

        if units_text is None:
            _append_coordinate_finding(
                issues,
                coord_name,
                {
                    "severity": "ERROR",
                    "convention": "cf",
                    "item": "time_coord_units_missing",
                    "message": (
                        f"Time coordinate '{coord_name}' uses numeric values but has no units."
                    ),
                    "current": {
                        "value_type": value_type,
                        "dtype": str(values.dtype),
                        "units": None,
                    },
                    "expected": "CF time units like 'days since 1970-01-01'.",
                    "suggested_fix": "set_coord_attr",
                },
            )
            continue

        if not _TIME_UNITS_RE.match(units_text):
            _append_coordinate_finding(
                issues,
                coord_name,
                {
                    "severity": "ERROR",
                    "convention": "cf",
                    "item": "time_coord_units_format_invalid",
                    "message": (
                        f"Time coordinate '{coord_name}' units are not CF-style: "
                        f"{units_text!r}."
                    ),
                    "current": {
                        "value_type": value_type,
                        "dtype": str(values.dtype),
                        "units": units_text,
                    },
                    "expected": "units in '<unit> since <epoch>' format.",
                    "suggested_fix": "set_coord_attr",
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
    ds: xr.Dataset, issues: dict[str, Any], conventions: tuple[ConventionName, ...]
) -> None:
    if "cf" in conventions:
        _apply_cf_attribute_case_checks(ds, issues)
        _apply_cf_time_units_and_type_checks(ds, issues)
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


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coord_bounds(coord: xr.DataArray) -> tuple[Any, Any] | None:
    try:
        minimum = coord.min(skipna=True)
        maximum = coord.max(skipna=True)
    except TypeError:
        minimum = coord.min()
        maximum = coord.max()

    min_values = np.asarray(minimum.values)
    max_values = np.asarray(maximum.values)
    if min_values.size == 0 or max_values.size == 0:
        return None

    min_value = _to_python_scalar(min_values.reshape(-1)[0])
    max_value = _to_python_scalar(max_values.reshape(-1)[0])

    if min_value is None or max_value is None:
        return None
    if isinstance(min_value, (float, np.floating)) and np.isnan(min_value):
        return None
    if isinstance(max_value, (float, np.floating)) and np.isnan(max_value):
        return None

    return min_value, max_value


def _decoded_numeric_time_bounds(
    coord: xr.DataArray, bounds: tuple[Any, Any]
) -> tuple[Any, Any]:
    lower, upper = bounds
    if not isinstance(lower, (int, float, np.integer, np.floating)):
        return bounds
    if not isinstance(upper, (int, float, np.integer, np.floating)):
        return bounds

    units = coord.attrs.get("units")
    if units is None:
        units = coord.encoding.get("units")
    units_text = "" if units is None else str(units).strip()
    if not units_text or not _TIME_UNITS_RE.match(units_text):
        return bounds

    calendar = str(
        coord.attrs.get("calendar", coord.encoding.get("calendar", "standard"))
    )
    try:
        from netCDF4 import num2date

        decoded = num2date([lower, upper], units=units_text, calendar=calendar)
    except Exception:
        return bounds

    if len(decoded) != 2:
        return bounds
    return _to_python_scalar(decoded[0]), _to_python_scalar(decoded[1])


def _format_time_coverage_value(value: Any) -> str:
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return ""
        return str(np.datetime_as_string(value, unit="s"))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if _is_cftime_datetime(value):
        isoformat = getattr(value, "isoformat", None)
        if callable(isoformat):
            return str(isoformat())
    return str(value)


def _first_axis_coord_name(
    ds: xr.Dataset, axis_guesses: dict[str, AxisGuess], axis_type: str
) -> str | None:
    for dim, guess in sorted(axis_guesses.items()):
        if guess.axis_type == axis_type and dim in ds.coords:
            return dim
    return None


def _update_extent_attrs_from_coords(
    ds: xr.Dataset, axis_guesses: dict[str, AxisGuess]
) -> None:
    time_coord_name = _first_axis_coord_name(ds, axis_guesses, "time")
    if time_coord_name is not None:
        time_bounds = _coord_bounds(ds.coords[time_coord_name])
        if time_bounds is not None:
            decoded_bounds = _decoded_numeric_time_bounds(
                ds.coords[time_coord_name], time_bounds
            )
            start = _format_time_coverage_value(decoded_bounds[0])
            end = _format_time_coverage_value(decoded_bounds[1])
            if start and end:
                ds.attrs["time_coverage_start"] = start
                ds.attrs["time_coverage_end"] = end

    lat_coord_name = _first_axis_coord_name(ds, axis_guesses, "lat")
    if lat_coord_name is not None:
        lat_bounds = _coord_bounds(ds.coords[lat_coord_name])
        if lat_bounds is not None:
            try:
                ds.attrs["geospatial_lat_min"] = float(lat_bounds[0])
                ds.attrs["geospatial_lat_max"] = float(lat_bounds[1])
            except (TypeError, ValueError):
                pass

    lon_coord_name = _first_axis_coord_name(ds, axis_guesses, "lon")
    if lon_coord_name is not None:
        lon_bounds = _coord_bounds(ds.coords[lon_coord_name])
        if lon_bounds is not None:
            try:
                ds.attrs["geospatial_lon_min"] = float(lon_bounds[0])
                ds.attrs["geospatial_lon_max"] = float(lon_bounds[1])
            except (TypeError, ValueError):
                pass


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
    try:
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
    except ModuleNotFoundError as exc:
        # Provide an actionable error when cfchecker is optional and unavailable.
        if exc.name == "cfchecker":
            raise RuntimeError(_CFCHECKER_INSTALL_HINT) from exc
        raise

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
    domain: StandardNameDomain | None = None,
    fallback_to_heuristic: bool = True,
    engine: ComplianceEngine = "auto",
    conventions: str | list[str] | tuple[str, ...] | None = None,
    report_format: ReportFormat = "auto",
    report_html_file: str | Path | None = None,
) -> dict[str, Any] | str | None:
    """Run compliance checks for selected conventions (for example CF/Ferret).

    Parameters
    ----------
    ds
        Dataset to validate.
    cf_version
        CF version passed into the checker (for example ``"1.12"``).
    standard_name_table_xml, cf_area_types_xml, cf_region_names_xml
        Optional local/remote XML tables used by ``cfchecker``.
    cache_tables
        Reuse cached checker tables between runs.
    domain
        Optional standard-name suggestion filter:
        ``Literal["ocean", "atmosphere", "land", "cryosphere", "biogeochemistry"]``.
    fallback_to_heuristic
        If ``True``, fall back to built-in heuristic checks when ``cfchecker``
        is unavailable or fails.
    engine
        ``Literal["auto", "cfchecker", "cfcheck", "heuristic"]``.
    conventions
        Conventions to enforce: ``"cf"``, ``"ferret"``, or both.
    report_format
        ``Literal["auto", "python", "tables", "html"]``.
    report_html_file
        Output path when ``report_format="html"``.

    Returns
    -------
    dict | str | None
        A report dictionary for ``"python"``, HTML for ``"html"``, or ``None``
        for ``"tables"`` (printed output).

    Raises
    ------
    ValueError
        If unsupported conventions/engine are requested, or HTML output options
        are used with a non-HTML format.
    Exception
        Re-raises checker errors when ``fallback_to_heuristic=False``.
    """
    resolved_format = normalize_report_format(report_format)
    if report_html_file is not None and resolved_format != "html":
        raise ValueError("`report_html_file` is only valid when report_format='html'.")

    selected_conventions = _normalize_requested_conventions(conventions)
    selected_engine = _normalize_requested_engine(engine)
    formatted_cf_version = _format_cf_version(cf_version)

    if "cf" in selected_conventions:
        if selected_engine == "heuristic":
            issues = heuristic_check_dataset(ds, cf_version=formatted_cf_version)
            issues["suggestions"] = {"variables": {}}
            augment_issues_with_standard_name_suggestions(
                ds,
                issues,
                standard_name_table_xml,
                domain=domain,
            )
        else:
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
                issues = heuristic_check_dataset(ds, cf_version=formatted_cf_version)
                issues["engine_status"] = "unavailable"
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
    issues["engine_requested"] = selected_engine
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
    """Return a copied dataset with safe, non-destructive CF metadata fixes.

    The function normalizes common metadata issues while preserving data values:
    - sets ``Conventions`` to ``"CF-1.12"``
    - normalizes key CF attribute names to lowercase expected forms
    - ensures inferred axis coordinates carry expected CF metadata
    - updates time and geospatial extent attributes when possible
    - removes coordinate ``_FillValue`` to avoid downstream tool conflicts

    Parameters
    ----------
    ds
        Input dataset. The original object is not modified.

    Returns
    -------
    xr.Dataset
        A deep-copied dataset with compliance-oriented metadata updates.
    """
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
        guess = guess_axis_for_dim(out, dim)
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
        for key, val in expected_coord_attrs(axis_guesses[dim].axis_type).items():
            new_attrs[key] = val

        if axis_guesses[dim].axis_type in {"lat", "lon"} and not is_numeric_dtype(
            coord.dtype
        ):
            # Preserve laziness for dask-backed coordinates by casting through xarray.
            coerced = coord.astype(float).copy(deep=False)
            coerced.attrs = new_attrs
            out = out.assign_coords({dim: coerced})
        else:
            out[dim].attrs = new_attrs

    _update_extent_attrs_from_coords(out, axis_guesses)

    # Ferret (and some other tools) can fail when coordinate variables include
    # _FillValue. Explicitly disable it for all coordinates.
    for coord_name in out.coords:
        coord_attrs = deepcopy(out[coord_name].attrs)
        coord_attrs.pop("_FillValue", None)
        out[coord_name].attrs = coord_attrs
        out[coord_name].encoding = {"_FillValue": None}

    return out
