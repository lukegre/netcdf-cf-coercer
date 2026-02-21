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
_AXIS_NAME_SETS = (
    ("time", _TIME_NAMES),
    ("lat", _LAT_NAMES),
    ("lon", _LON_NAMES),
)
_EXPECTED_COORD_ATTRS: dict[str, dict[str, str]] = {
    "lat": {
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
        "axis": "Y",
    },
    "lon": {
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
        "axis": "X",
    },
    "time": {
        "standard_name": "time",
        "axis": "T",
    },
}
_AXIS_LABELS = {"lat": "latitude", "lon": "longitude", "time": "time"}
_AXIS_RANGE_SPECS: dict[str, dict[str, Any]] = {
    "lat": {
        "minimum": -90.0,
        "maximum": 90.0,
        "noun": "latitude",
        "suggested_fix": "clamp_or_fix_latitudes",
    },
    "lon": {
        "minimum": -180.0,
        "maximum": 360.0,
        "noun": "longitude",
        "suggested_fix": "normalize_longitudes",
    },
}


@dataclass(frozen=True)
class AxisGuess:
    dim: str
    axis_type: str


def normalize_name(name: str) -> str:
    return name.strip().lower()


def is_numeric_dtype(dtype: Any) -> bool:
    return np.issubdtype(dtype, np.number)


def _axis_from_names(*names: str) -> str | None:
    for axis_type, candidates in _AXIS_NAME_SETS:
        if any(name in candidates for name in names):
            return axis_type
    return None


def _axis_from_coord_attrs(coord: xr.DataArray) -> str | None:
    standard_name = normalize_name(str(coord.attrs.get("standard_name", "")))
    units = normalize_name(str(coord.attrs.get("units", "")))

    if standard_name == "latitude" or units == "degrees_north":
        return "lat"
    if standard_name == "longitude" or units == "degrees_east":
        return "lon"
    if standard_name == "time":
        return "time"
    return None


def guess_axis_for_dim(ds: xr.Dataset, dim: str) -> AxisGuess | None:
    dim_norm = normalize_name(dim)
    coord = ds.coords.get(dim)
    coord_name_norm = normalize_name(coord.name) if coord is not None else dim_norm

    axis_type = _axis_from_names(dim_norm, coord_name_norm)
    if axis_type is not None:
        return AxisGuess(dim=dim, axis_type=axis_type)
    if coord is None:
        return None

    axis_type = _axis_from_coord_attrs(coord)
    if axis_type is None:
        return None
    return AxisGuess(dim=dim, axis_type=axis_type)


def expected_coord_attrs(axis_type: str) -> dict[str, str]:
    return _EXPECTED_COORD_ATTRS.get(axis_type, {}).copy()


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


def _datetime_like_to_int(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any] | None:
    if not np.issubdtype(values.dtype, np.datetime64) and not np.issubdtype(
        values.dtype, np.timedelta64
    ):
        return None
    if bool(np.any(np.isnat(values))):
        return None
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]").astype(np.int64)
    return values.astype("timedelta64[ns]").astype(np.int64)


def _numeric_values(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any] | None:
    if np.issubdtype(values.dtype, np.number):
        return np.asarray(values, dtype=float)
    return _datetime_like_to_int(values)


def _finite_values(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any] | None:
    if not np.issubdtype(values.dtype, np.floating):
        return values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return finite


def _coord_sequence_findings(
    coord_name: str, values: np.ndarray[Any, Any]
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if np.unique(values).size != values.size:
        findings.append(
            _finding(
                severity="WARN",
                item="coord_not_unique",
                message=f"Coordinate '{coord_name}' contains duplicate values.",
                current="duplicate values present",
                expected="all coordinate values should be unique",
                suggested_fix="deduplicate_coord_values",
            )
        )

    diffs = np.diff(values)
    monotonic = bool(np.all(diffs > 0) or np.all(diffs < 0))
    if monotonic:
        return findings
    findings.append(
        _finding(
            severity="WARN",
            item="coord_not_monotonic",
            message=f"Coordinate '{coord_name}' is not monotonic.",
            current="non-monotonic sequence",
            expected="strictly increasing or decreasing values",
            suggested_fix="sort_coord_values",
        )
    )
    return findings


def _coord_range_finding(
    coord_name: str, axis_type: str, values: np.ndarray[Any, Any]
) -> dict[str, Any] | None:
    spec = _AXIS_RANGE_SPECS.get(axis_type)
    if spec is None:
        return None
    out_of_range = np.logical_or(values < spec["minimum"], values > spec["maximum"])
    count = int(np.count_nonzero(out_of_range))
    if count == 0:
        return None
    return _finding(
        severity="ERROR",
        item="coord_values_out_of_range",
        message=(
            f"Coordinate '{coord_name}' has {spec['noun']} values outside "
            f"[{spec['minimum']:.0f}, {spec['maximum']:.0f}]."
        ),
        current={"out_of_range_count": count},
        expected=(
            f"{spec['noun']} values in [{spec['minimum']:.0f}, {spec['maximum']:.0f}]"
        ),
        suggested_fix=spec["suggested_fix"],
    )


def _coordinate_value_findings(
    coord: xr.DataArray, *, axis_type: str
) -> list[dict[str, Any]]:
    values = np.asarray(coord.values)
    if values.ndim != 1 or values.size < 2:
        return []

    numeric = _numeric_values(values)
    if numeric is None:
        return []
    finite = _finite_values(numeric)
    if finite is None:
        return []

    coord_name = str(coord.name)
    findings = _coord_sequence_findings(coord_name, finite)
    range_finding = _coord_range_finding(coord_name, axis_type, finite)
    if range_finding is not None:
        findings.append(range_finding)
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


_REFERENCE_ATTR_SPECS = (
    ("coordinates", _references_from_whitespace_list),
    ("bounds", _references_from_whitespace_list),
    ("grid_mapping", _references_from_whitespace_list),
    ("ancillary_variables", _references_from_whitespace_list),
    ("cell_measures", _references_from_cell_measures),
)


def _reference_findings(
    *,
    var_name: str,
    attrs: dict[str, Any],
    available_names: set[str],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for attr_name, parser in _REFERENCE_ATTR_SPECS:
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


def _new_issues(cf_version: str) -> dict[str, Any]:
    return {
        "cf_version": cf_version,
        "engine": "heuristic",
        "engine_status": "ok",
        "check_method": "heuristic",
        "global": [],
        "coordinates": {},
        "variables": {},
        "notes": [],
    }


def _add_conventions_finding(
    issues: dict[str, Any], ds: xr.Dataset, *, cf_version: str
) -> None:
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
        return
    if cf_version in cf_tokens:
        return
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


def _axis_guesses(ds: xr.Dataset, issues: dict[str, Any]) -> dict[str, AxisGuess]:
    guesses: dict[str, AxisGuess] = {}
    for dim_name in ds.dims:
        dim = str(dim_name)
        guess = guess_axis_for_dim(ds, dim)
        if guess is not None:
            guesses[dim] = guess
            continue
        issues["notes"].append(f"Could not infer CF axis type for dimension '{dim}'.")
    return guesses


def _missing_dimension_coordinate_finding(dim: str) -> dict[str, Any]:
    return _finding(
        severity="ERROR",
        item="missing_dimension_coordinate",
        message=f"Dimension '{dim}' has no coordinate variable.",
        current=None,
        expected=f"coordinate variable named '{dim}'",
        suggested_fix="create_dimension_coordinate",
    )


def _coord_attr_findings(
    dim: str, coord: xr.DataArray, axis_type: str
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    axis_label = _AXIS_LABELS.get(axis_type, axis_type)
    for key, expected_value in expected_coord_attrs(axis_type).items():
        current_value = coord.attrs.get(key)
        if current_value == expected_value:
            continue
        current_text = "missing" if current_value is None else repr(current_value)
        findings.append(
            _finding(
                severity="WARN",
                item=f"coord_attr:{key}",
                message=(
                    f"Coordinate '{dim}' is inferred as {axis_label} and expects "
                    f"{key}={expected_value!r}, but found {current_text}."
                ),
                current=current_value,
                expected=expected_value,
                suggested_fix="set_coord_attr",
            )
        )
    return findings


def _coord_dtype_finding(
    dim: str, coord: xr.DataArray, axis_type: str
) -> dict[str, Any] | None:
    if axis_type not in {"lat", "lon"}:
        return None
    if is_numeric_dtype(coord.dtype):
        return None
    return _finding(
        severity="ERROR",
        item="coord_dtype",
        message=f"Coordinate '{dim}' has non-numeric dtype '{coord.dtype}'.",
        current=str(coord.dtype),
        expected="numeric",
        suggested_fix="convert_coord_dtype",
    )


def _time_coord_unit_findings(
    dim: str, coord: xr.DataArray, axis_type: str
) -> list[dict[str, Any]]:
    if axis_type != "time":
        return []

    units = coord.attrs.get("units")
    if units is None:
        units = coord.encoding.get("units")
    if units is None and not np.issubdtype(coord.dtype, np.datetime64):
        return [
            _finding(
                severity="ERROR",
                item="coord_attr:units",
                message=(f"Time coordinate '{dim}' is missing units metadata."),
                current=None,
                expected="time units, e.g. 'days since 1970-01-01'",
                suggested_fix="set_coord_attr",
            )
        ]
    if isinstance(units, str) and not _TIME_UNITS_RE.match(units):
        return [
            _finding(
                severity="WARN",
                item="coord_attr:units_format",
                message=(f"Time coordinate '{dim}' units do not look CF-compliant."),
                current=units,
                expected="units in '<unit> since <epoch>' format",
                suggested_fix="set_coord_attr",
            )
        ]
    return []


def _coord_findings_for_axis(
    dim: str, guess: AxisGuess, ds: xr.Dataset
) -> list[dict[str, Any]]:
    coord = ds.coords.get(dim)
    if coord is None:
        return [_missing_dimension_coordinate_finding(dim)]

    findings = _coord_attr_findings(dim, coord, guess.axis_type)
    dtype_finding = _coord_dtype_finding(dim, coord, guess.axis_type)
    if dtype_finding is not None:
        findings.append(dtype_finding)
    findings.extend(_time_coord_unit_findings(dim, coord, guess.axis_type))
    findings.extend(_coordinate_value_findings(coord, axis_type=guess.axis_type))
    return findings


def _invalid_variable_name_finding(var_name: str) -> dict[str, Any] | None:
    if _VALID_NAME_RE.match(var_name):
        return None
    return _finding(
        severity="ERROR",
        item="invalid_variable_name",
        message=f"Variable '{var_name}' is not CF-name compliant.",
        current=var_name,
        expected="start with a letter, then letters/digits/underscore",
        suggested_fix="rename_variable",
    )


def _variable_metadata_findings(
    var_name: str, da: xr.DataArray
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    units = da.attrs.get("units")
    std_name = da.attrs.get("standard_name")

    invalid_name = _invalid_variable_name_finding(var_name)
    if invalid_name is not None:
        findings.append(invalid_name)
    if _is_dimensional_numeric_variable(da) and units is None:
        findings.append(
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
        findings.append(
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
        findings.append(
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
        findings.append(
            _finding(
                severity="ERROR",
                item="missing_units_for_standard_name",
                message=(
                    f"Variable '{var_name}' has standard_name='{std_name}' but no units."
                ),
                current=None,
                expected="units consistent with standard_name canonical_units",
                suggested_fix="set_variable_attr",
            )
        )
    return findings


def _add_variable_findings(
    issues: dict[str, Any], ds: xr.Dataset, available_names: set[str]
) -> None:
    for var_name_raw, da in ds.data_vars.items():
        var_name = str(var_name_raw)
        findings = _variable_metadata_findings(var_name, da)
        findings.extend(
            _reference_findings(
                var_name=var_name,
                attrs=da.attrs,
                available_names=available_names,
            )
        )
        if findings:
            issues["variables"][var_name] = findings


def _add_coordinate_reference_findings(
    issues: dict[str, Any], ds: xr.Dataset, available_names: set[str]
) -> None:
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


def heuristic_check_dataset(ds: xr.Dataset, *, cf_version: str) -> dict[str, Any]:
    """Heuristic metadata checker used when cfchecker cannot run."""
    issues = _new_issues(cf_version)
    _add_conventions_finding(issues, ds, cf_version=cf_version)

    for dim, guess in _axis_guesses(ds, issues).items():
        coord_issues = _coord_findings_for_axis(dim, guess, ds)
        if coord_issues:
            issues["coordinates"][dim] = coord_issues

    available_names = {str(name) for name in ds.variables}
    _add_variable_findings(issues, ds, available_names)
    _add_coordinate_reference_findings(issues, ds, available_names)

    return issues


def _is_dimensional_numeric_variable(da: xr.DataArray) -> bool:
    if not is_numeric_dtype(da.dtype):
        return False
    if np.issubdtype(da.dtype, np.bool_):
        return False
    return True
