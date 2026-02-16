from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import xarray as xr
from .standard_names import augment_issues_with_standard_name_suggestions

CF_VERSION = "CF-1.12"
CF_STANDARD_NAME_TABLE_URL = (
    "https://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml"
)

_LAT_NAMES = {"lat", "latitude", "y"}
_LON_NAMES = {"lon", "longitude", "x"}
_TIME_NAMES = {"time", "t"}
_NON_COMPLIANT_CATEGORIES = ("FATAL", "ERROR", "WARN")
_VALID_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class AxisGuess:
    dim: str
    axis_type: str


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
            issues["notes"].append(f"Could not infer CF axis type for dimension '{dim}'.")

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
        known_dim_coords = [d for d in da.dims if d in ds.coords and d in axis_guesses]
        if known_dim_coords and da.attrs.get("coordinates") is None:
            var_issues.append(
                {
                    "item": "missing_coordinates_attr",
                    "current": None,
                    "expected": "space-separated coordinate variable names (optional but recommended for clarity)",
                    "suggested_fix": "set_coordinates_attr",
                }
            )
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


def _as_netcdf_bytes(ds: xr.Dataset) -> bytes:
    data = ds.to_netcdf(path=None)
    if isinstance(data, memoryview):
        return data.tobytes()
    return bytes(data)


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
    cf_standard_names_xml: str | None = None,
    cf_area_types_xml: str | None = None,
    cf_region_names_xml: str | None = None,
    cache_tables: bool = False,
) -> dict[str, Any]:
    # Import inside function so package can still be used for make_compliant
    # even if cfchecker/cfunits system dependencies are not installed.
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
    if cf_standard_names_xml is not None:
        checker_kwargs["cfStandardNamesXML"] = cf_standard_names_xml
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
    cf_standard_names_xml: str | None = CF_STANDARD_NAME_TABLE_URL,
    cf_area_types_xml: str | None = None,
    cf_region_names_xml: str | None = None,
    cache_tables: bool = False,
    standard_name_table_xml: str | None = CF_STANDARD_NAME_TABLE_URL,
    domain: str | None = None,
    fallback_to_heuristic: bool = True,
) -> dict[str, Any]:
    """Run CF compliance checks using cfchecker on an in-memory NetCDF payload."""
    try:
        issues = _run_cfchecker_on_dataset(
            ds,
            cf_version=cf_version,
            cf_standard_names_xml=cf_standard_names_xml,
            cf_area_types_xml=cf_area_types_xml,
            cf_region_names_xml=cf_region_names_xml,
            cache_tables=cache_tables,
        )
        table_xml = standard_name_table_xml or cf_standard_names_xml
        augment_issues_with_standard_name_suggestions(
            ds,
            issues,
            table_xml,
            domain=domain,
        )
        return issues
    except Exception as exc:
        if not fallback_to_heuristic:
            raise
        fallback = _heuristic_check_dataset(ds)
        fallback["suggestions"] = {"variables": {}}
        table_xml = standard_name_table_xml or cf_standard_names_xml
        augment_issues_with_standard_name_suggestions(
            ds,
            fallback,
            table_xml,
            domain=domain,
        )
        fallback["checker_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        fallback["notes"].append(
            "cfchecker could not run; returned heuristic checks instead."
        )
        return fallback


def make_dataset_compliant(ds: xr.Dataset) -> xr.Dataset:
    """Return a copy of the dataset with safe CF-1.12 compliance fixes."""
    out = ds.copy(deep=True)

    out.attrs = deepcopy(out.attrs)
    out.attrs["Conventions"] = CF_VERSION

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

        for key, val in _expected_coord_attrs(axis_guesses[dim].axis_type).items():
            new_attrs[key] = val

        if axis_guesses[dim].axis_type in {"lat", "lon"} and not _is_numeric_dtype(
            coord.dtype
        ):
            coerced = xr.DataArray(
                np.asarray(coord.values).astype(float),
                dims=coord.dims,
                coords=coord.coords,
                attrs=new_attrs,
                name=coord.name,
            )
            out = out.assign_coords({dim: coerced})
        else:
            out[dim].attrs = new_attrs

    for var_name, da in out.data_vars.items():
        known_dim_coords = [d for d in da.dims if d in out.coords and d in axis_guesses]
        if not known_dim_coords:
            continue
        if "coordinates" not in da.attrs:
            out[var_name].attrs = deepcopy(da.attrs)
            out[var_name].attrs["coordinates"] = " ".join(known_dim_coords)

    return out
