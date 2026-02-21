from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import xarray as xr

_LAT_NAMES = {"lat", "latitude", "y"}
_LON_NAMES = {"lon", "longitude", "x"}
_TIME_NAMES = {"time", "t"}
_VALID_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_TIME_UNITS_RE = re.compile(
    r"^\s*(seconds?|minutes?|hours?|days?|months?|years?)\s+since\s+.+$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AxisGuess:
    dim: str
    axis_type: str


def normalize_name(name: str) -> str:
    return name.strip().lower()


def is_numeric_dtype(dtype: Any) -> bool:
    return np.issubdtype(dtype, np.number)


def guess_axis_for_dim(ds: xr.Dataset, dim: str) -> AxisGuess | None:
    dim_norm = normalize_name(dim)
    coord = ds.coords.get(dim)
    coord_name_norm = normalize_name(coord.name) if coord is not None else dim_norm

    if dim_norm in _TIME_NAMES or coord_name_norm in _TIME_NAMES:
        return AxisGuess(dim=dim, axis_type="time")

    if dim_norm in _LAT_NAMES or coord_name_norm in _LAT_NAMES:
        return AxisGuess(dim=dim, axis_type="lat")

    if dim_norm in _LON_NAMES or coord_name_norm in _LON_NAMES:
        return AxisGuess(dim=dim, axis_type="lon")

    if coord is not None:
        standard_name = normalize_name(str(coord.attrs.get("standard_name", "")))
        units = normalize_name(str(coord.attrs.get("units", "")))

        if standard_name == "latitude" or units == "degrees_north":
            return AxisGuess(dim=dim, axis_type="lat")
        if standard_name == "longitude" or units == "degrees_east":
            return AxisGuess(dim=dim, axis_type="lon")
        if standard_name == "time":
            return AxisGuess(dim=dim, axis_type="time")

    return None


def expected_coord_attrs(axis_type: str) -> dict[str, str]:
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


def _finding(
    *,
    severity: str,
    item: str,
    message: str,
    current: Any,
    expected: Any,
    suggested_fix: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    finding = {
        "severity": severity,
        "item": item,
        "message": message,
        "current": current,
        "expected": expected,
        "suggested_fix": suggested_fix,
    }
    if extra:
        finding.update(extra)
    return finding


def _cf_tokens_from_conventions(conventions: Any) -> set[str]:
    if conventions is None:
        return set()
    parts = [part.strip() for part in str(conventions).split(",")]
    return {part for part in parts if part.upper().startswith("CF-")}


def _numeric_values(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any] | None:
    if np.issubdtype(values.dtype, np.number):
        return np.asarray(values, dtype=float)
    if np.issubdtype(values.dtype, np.datetime64):
        nat_mask = np.isnat(values)
        if bool(np.any(nat_mask)):
            return None
        return values.astype("datetime64[ns]").astype(np.int64)
    if np.issubdtype(values.dtype, np.timedelta64):
        nat_mask = np.isnat(values)
        if bool(np.any(nat_mask)):
            return None
        return values.astype("timedelta64[ns]").astype(np.int64)
    return None


def _coordinate_value_findings(
    coord: xr.DataArray, *, axis_type: str
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    values = np.asarray(coord.values)
    if values.ndim != 1 or values.size < 2:
        return findings

    numeric = _numeric_values(values)
    if numeric is None:
        return findings
    if np.issubdtype(numeric.dtype, np.floating):
        finite = numeric[np.isfinite(numeric)]
        if finite.size == 0:
            return findings
    else:
        finite = numeric

    if np.unique(finite).size != finite.size:
        findings.append(
            _finding(
                severity="WARN",
                item="coord_not_unique",
                message=f"Coordinate '{coord.name}' contains duplicate values.",
                current="duplicate values present",
                expected="all coordinate values should be unique",
                suggested_fix="deduplicate_coord_values",
            )
        )

    diffs = np.diff(finite)
    monotonic = bool(np.all(diffs > 0) or np.all(diffs < 0))
    if not monotonic:
        findings.append(
            _finding(
                severity="WARN",
                item="coord_not_monotonic",
                message=f"Coordinate '{coord.name}' is not monotonic.",
                current="non-monotonic sequence",
                expected="strictly increasing or decreasing values",
                suggested_fix="sort_coord_values",
            )
        )

    if axis_type == "lat":
        out_of_range = np.logical_or(finite < -90.0, finite > 90.0)
        count = int(np.count_nonzero(out_of_range))
        if count > 0:
            findings.append(
                _finding(
                    severity="ERROR",
                    item="coord_values_out_of_range",
                    message=(
                        f"Coordinate '{coord.name}' has latitude values outside "
                        "[-90, 90]."
                    ),
                    current={"out_of_range_count": count},
                    expected="latitude values in [-90, 90]",
                    suggested_fix="clamp_or_fix_latitudes",
                )
            )
    elif axis_type == "lon":
        out_of_range = np.logical_or(finite < -180.0, finite > 360.0)
        count = int(np.count_nonzero(out_of_range))
        if count > 0:
            findings.append(
                _finding(
                    severity="ERROR",
                    item="coord_values_out_of_range",
                    message=(
                        f"Coordinate '{coord.name}' has longitude values outside "
                        "[-180, 360]."
                    ),
                    current={"out_of_range_count": count},
                    expected="longitude values in [-180, 360]",
                    suggested_fix="normalize_longitudes",
                )
            )

    return findings


def _references_from_whitespace_list(value: Any) -> tuple[list[str], str | None]:
    if value is None:
        return [], None
    if not isinstance(value, str):
        return [], f"expected string but found {type(value).__name__}"
    names = [token for token in value.split() if token]
    return names, None


def _references_from_cell_measures(value: Any) -> tuple[list[str], str | None]:
    if value is None:
        return [], None
    if not isinstance(value, str):
        return [], f"expected string but found {type(value).__name__}"
    tokens = value.split()
    if len(tokens) < 2:
        return [], "expected one or more '<measure>: <var>' pairs"

    references: list[str] = []
    idx = 0
    while idx + 1 < len(tokens):
        measure = tokens[idx]
        variable_name = tokens[idx + 1]
        if not measure.endswith(":"):
            return [], "expected '<measure>: <var>' pairs"
        if variable_name.endswith(":"):
            return [], "missing variable name after measure"
        references.append(variable_name)
        idx += 2
    if idx != len(tokens):
        return [], "trailing token without a variable name"
    return references, None


def _reference_findings(
    *,
    var_name: str,
    attrs: dict[str, Any],
    available_names: set[str],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    ref_specs = (
        ("coordinates", _references_from_whitespace_list),
        ("bounds", _references_from_whitespace_list),
        ("grid_mapping", _references_from_whitespace_list),
        ("ancillary_variables", _references_from_whitespace_list),
        ("cell_measures", _references_from_cell_measures),
    )
    for attr_name, parser in ref_specs:
        raw_value = attrs.get(attr_name)
        references, error = parser(raw_value)
        if error is not None:
            findings.append(
                _finding(
                    severity="WARN",
                    item=f"invalid_reference_attr:{attr_name}",
                    message=(
                        f"Variable '{var_name}' uses invalid '{attr_name}' syntax: "
                        f"{error}."
                    ),
                    current=raw_value,
                    expected="valid CF variable reference syntax",
                    suggested_fix="set_variable_attr",
                )
            )
            continue
        if not references:
            continue

        missing = sorted({name for name in references if name not in available_names})
        if missing:
            findings.append(
                _finding(
                    severity="ERROR",
                    item=f"missing_referenced_variable:{attr_name}",
                    message=(
                        f"Variable '{var_name}' references missing variables in "
                        f"'{attr_name}': {', '.join(missing)}."
                    ),
                    current={"attr": raw_value, "missing": missing},
                    expected="all referenced variables exist in the dataset",
                    suggested_fix="fix_variable_references",
                )
            )
    return findings


def heuristic_check_dataset(ds: xr.Dataset, *, cf_version: str) -> dict[str, Any]:
    """Heuristic metadata checker used when cfchecker cannot run."""
    issues: dict[str, Any] = {
        "cf_version": cf_version,
        "engine": "heuristic",
        "engine_status": "ok",
        "check_method": "heuristic",
        "global": [],
        "coordinates": {},
        "variables": {},
        "notes": [],
    }

    conventions = ds.attrs.get("Conventions")
    cf_tokens = _cf_tokens_from_conventions(conventions)
    if conventions is None:
        issues["global"].append(
            _finding(
                severity="WARN",
                item="Conventions",
                message="Global attribute 'Conventions' is missing.",
                current=None,
                expected=f"include '{cf_version}' in Conventions",
                suggested_fix="set_global_attr",
            )
        )
    elif cf_version not in cf_tokens:
        issues["global"].append(
            _finding(
                severity="WARN",
                item="Conventions",
                message=(
                    "Global attribute 'Conventions' does not include the expected "
                    f"CF token '{cf_version}'."
                ),
                current=conventions,
                expected=f"include '{cf_version}'",
                suggested_fix="set_global_attr",
            )
        )

    axis_guesses: dict[str, AxisGuess] = {}
    for dim in ds.dims:
        dim = str(dim)
        guess = guess_axis_for_dim(ds, dim)
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
                _finding(
                    severity="ERROR",
                    item="missing_dimension_coordinate",
                    message=f"Dimension '{dim}' has no coordinate variable.",
                    current=None,
                    expected=f"coordinate variable named '{dim}'",
                    suggested_fix="create_dimension_coordinate",
                )
            ]
            continue

        coord_issues: list[dict[str, Any]] = []
        expected_attrs = expected_coord_attrs(guess.axis_type)
        axis_label = {
            "lat": "latitude",
            "lon": "longitude",
            "time": "time",
        }.get(guess.axis_type, guess.axis_type)
        for key, expected_value in expected_attrs.items():
            current_value = coord.attrs.get(key)
            if current_value != expected_value:
                current_text = (
                    "missing" if current_value is None else repr(current_value)
                )
                coord_issues.append(
                    _finding(
                        severity="WARN",
                        item=f"coord_attr:{key}",
                        message=(
                            f"Coordinate '{dim}' is inferred as {axis_label} and "
                            f"expects {key}={expected_value!r}, but found {current_text}."
                        ),
                        current=current_value,
                        expected=expected_value,
                        suggested_fix="set_coord_attr",
                    )
                )

        if guess.axis_type in {"lat", "lon"} and not is_numeric_dtype(coord.dtype):
            coord_issues.append(
                _finding(
                    severity="ERROR",
                    item="coord_dtype",
                    message=(
                        f"Coordinate '{dim}' has non-numeric dtype '{coord.dtype}'."
                    ),
                    current=str(coord.dtype),
                    expected="numeric",
                    suggested_fix="convert_coord_dtype",
                )
            )

        if guess.axis_type == "time":
            units = coord.attrs.get("units")
            if units is None:
                units = coord.encoding.get("units")
            if units is None and not np.issubdtype(coord.dtype, np.datetime64):
                coord_issues.append(
                    _finding(
                        severity="ERROR",
                        item="coord_attr:units",
                        message=(f"Time coordinate '{dim}' is missing units metadata."),
                        current=None,
                        expected="time units, e.g. 'days since 1970-01-01'",
                        suggested_fix="set_coord_attr",
                    )
                )
            elif isinstance(units, str) and not _TIME_UNITS_RE.match(units):
                coord_issues.append(
                    _finding(
                        severity="WARN",
                        item="coord_attr:units_format",
                        message=(
                            f"Time coordinate '{dim}' units do not look CF-compliant."
                        ),
                        current=units,
                        expected="units in '<unit> since <epoch>' format",
                        suggested_fix="set_coord_attr",
                    )
                )

        coord_issues.extend(
            _coordinate_value_findings(coord, axis_type=guess.axis_type)
        )

        if coord_issues:
            issues["coordinates"][dim] = coord_issues

    available_names = {str(name) for name in ds.variables}
    for var_name, da in ds.data_vars.items():
        var_name = str(var_name)
        var_issues: list[dict[str, Any]] = []
        if not _VALID_NAME_RE.match(var_name):
            var_issues.append(
                _finding(
                    severity="ERROR",
                    item="invalid_variable_name",
                    message=f"Variable '{var_name}' is not CF-name compliant.",
                    current=var_name,
                    expected="start with a letter, then letters/digits/underscore",
                    suggested_fix="rename_variable",
                )
            )

        units = da.attrs.get("units")
        std_name = da.attrs.get("standard_name")
        if _is_dimensional_numeric_variable(da) and units is None:
            var_issues.append(
                _finding(
                    severity="WARN",
                    item="missing_units_attr",
                    message=f"Numeric variable '{var_name}' is missing 'units'.",
                    current=None,
                    expected="UDUNITS-compatible units string",
                    suggested_fix="set_variable_attr",
                )
            )
        if da.attrs.get("standard_name") is None and da.attrs.get("long_name") is None:
            var_issues.append(
                _finding(
                    severity="WARN",
                    item="missing_standard_or_long_name",
                    message=(
                        f"Variable '{var_name}' is missing both 'standard_name' and "
                        "'long_name'."
                    ),
                    current=None,
                    expected="at least one of: standard_name, long_name",
                    suggested_fix="set_variable_attr",
                )
            )
        if da.dtype == object:
            var_issues.append(
                _finding(
                    severity="ERROR",
                    item="object_dtype_variable",
                    message=(
                        f"Variable '{var_name}' uses object dtype, which is not "
                        "NetCDF-serializable."
                    ),
                    current=str(da.dtype),
                    expected="netCDF-compatible primitive dtype",
                    suggested_fix="cast_variable_dtype",
                )
            )
        if std_name is not None and units is None:
            var_issues.append(
                _finding(
                    severity="ERROR",
                    item="missing_units_for_standard_name",
                    message=(
                        f"Variable '{var_name}' has standard_name='{std_name}' but "
                        "no units."
                    ),
                    current=None,
                    expected="units consistent with standard_name canonical_units",
                    suggested_fix="set_variable_attr",
                )
            )

        var_issues.extend(
            _reference_findings(
                var_name=var_name,
                attrs=da.attrs,
                available_names=available_names,
            )
        )

        if var_issues:
            issues["variables"][var_name] = var_issues

    for coord_name, coord in ds.coords.items():
        coord_findings = _reference_findings(
            var_name=str(coord_name),
            attrs=coord.attrs,
            available_names=available_names,
        )
        if not coord_findings:
            continue
        coordinate_items = issues["coordinates"].setdefault(str(coord_name), [])
        if isinstance(coordinate_items, list):
            coordinate_items.extend(coord_findings)

    return issues


def _is_dimensional_numeric_variable(da: xr.DataArray) -> bool:
    if not is_numeric_dtype(da.dtype):
        return False
    if np.issubdtype(da.dtype, np.bool_):
        return False
    return True
