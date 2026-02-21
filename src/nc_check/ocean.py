from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import xarray as xr
from .formatting import (
    ReportFormat,
    maybe_display_html_report,
    normalize_report_format,
    print_pretty_ocean_reports,
    print_pretty_time_cover_reports,
    render_pretty_ocean_report_html,
    render_pretty_ocean_reports_html,
    render_pretty_time_cover_report_html,
    render_pretty_time_cover_reports_html,
    save_html_report,
)

_LON_CANDIDATES = ("lon", "longitude", "x")
_LAT_CANDIDATES = ("lat", "latitude", "y")
LongitudeConvention: TypeAlias = Literal["-180_180", "0_360", "other"]

_LAND_REFERENCE_POINTS = (
    ("sahara", 23.0, 13.0),
    ("australia_interior", -25.0, 134.0),
    ("mongolia", 47.0, 103.0),
    ("greenland_interior", 72.0, -40.0),
    ("south_america_interior", -15.0, -60.0),
)

_OCEAN_REFERENCE_POINTS = (
    ("equatorial_pacific", 0.0, -140.0),
    ("north_atlantic", 30.0, -40.0),
    ("indian_ocean", -30.0, 80.0),
    ("south_pacific", -45.0, -150.0),
    ("west_pacific", 10.0, 160.0),
)


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _guess_coord_name(
    ds: xr.Dataset,
    candidates: tuple[str, ...],
    units_token: str,
) -> str | None:
    candidate_set = set(candidates)
    for coord_name in ds.coords:
        if _normalize_name(str(coord_name)) in candidate_set:
            return str(coord_name)

    for coord_name, coord in ds.coords.items():
        units = _normalize_name(str(coord.attrs.get("units", "")))
        if units_token in units:
            return str(coord_name)
    return None


def _resolve_1d_coord(
    ds: xr.Dataset, coord_name: str
) -> tuple[str, np.ndarray[Any, Any]]:
    if coord_name not in ds.coords:
        raise ValueError(f"Coordinate '{coord_name}' not found.")
    coord = ds.coords[coord_name]
    if coord.ndim != 1:
        raise ValueError(f"Coordinate '{coord_name}' must be 1D.")
    return str(coord.dims[0]), np.asarray(coord.values)


def _resolve_time_dim(da: xr.DataArray, preferred_name: str | None) -> str | None:
    if preferred_name and preferred_name in da.dims:
        return preferred_name
    if "time" in da.dims:
        return "time"
    for dim in da.dims:
        coord = da.coords.get(dim)
        if coord is None:
            continue
        if _normalize_name(str(coord.attrs.get("standard_name", ""))) == "time":
            return str(dim)
    return None


def _choose_data_vars(
    ds: xr.Dataset,
    *,
    var_name: str | None,
    lon_dim: str,
    lat_dim: str,
) -> list[xr.DataArray]:
    if var_name is not None:
        if var_name not in ds.data_vars:
            raise ValueError(f"Data variable '{var_name}' not found.")
        da = ds[var_name]
        if lon_dim not in da.dims or lat_dim not in da.dims:
            raise ValueError(
                f"Data variable '{var_name}' must include lon dim '{lon_dim}' and lat dim '{lat_dim}'."
            )
        return [da]

    selected: list[xr.DataArray] = []
    for name, da in ds.data_vars.items():
        if lon_dim in da.dims and lat_dim in da.dims:
            selected.append(da)

    if selected:
        return selected

    raise ValueError(
        "Could not infer ocean variable. Provide `var_name` for a variable that has lat/lon dimensions."
    )


def _missing_mask(da: xr.DataArray) -> xr.DataArray:
    mask = da.isnull()
    for source in (da.attrs, da.encoding):
        fill_value = source.get("_FillValue")
        if fill_value is None:
            continue
        try:
            mask = mask | (da == fill_value)
        except Exception:
            continue
    return mask


def _indices_to_ranges(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ordered = sorted(indices)
    ranges: list[tuple[int, int]] = []
    start = ordered[0]
    end = ordered[0]
    for idx in ordered[1:]:
        if idx == end + 1:
            end = idx
            continue
        ranges.append((start, end))
        start = idx
        end = idx
    ranges.append((start, end))
    return ranges


def _value_label(value: Any) -> str:
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit="s")
    return str(value)


def _range_records(
    indices: list[int],
    coord: xr.DataArray | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for start, end in _indices_to_ranges(indices):
        if coord is not None:
            values = np.asarray(coord.values)
            start_label = _value_label(values[start])
            end_label = _value_label(values[end])
        else:
            start_label = str(start)
            end_label = str(end)
        out.append(
            {
                "start_index": int(start),
                "end_index": int(end),
                "start": start_label,
                "end": end_label,
            }
        )
    return out


def _value_ranges_from_indices(
    indices: list[int],
    values: np.ndarray[Any, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for start, end in _indices_to_ranges(indices):
        out.append(
            {
                "start": _value_label(values[start]),
                "end": _value_label(values[end]),
            }
        )
    return out


def _longitude_convention(lon_values: np.ndarray[Any, Any]) -> LongitudeConvention:
    lon_min = float(np.nanmin(lon_values))
    lon_max = float(np.nanmax(lon_values))
    eps = 1e-6
    if lon_min >= -180.0 - eps and lon_max <= 180.0 + eps:
        return "-180_180"
    if lon_min >= 0.0 - eps and lon_max <= 360.0 + eps:
        return "0_360"
    return "other"


def _normalize_lon_for_grid(lon: float, convention: LongitudeConvention) -> float:
    if convention == "0_360":
        return lon % 360.0
    if convention == "-180_180":
        return ((lon + 180.0) % 360.0) - 180.0
    return lon


def _is_global_grid(
    lon_values: np.ndarray[Any, Any], lat_values: np.ndarray[Any, Any]
) -> bool:
    lon_span = float(np.nanmax(lon_values) - np.nanmin(lon_values))
    lat_span = float(np.nanmax(lat_values) - np.nanmin(lat_values))
    return lon_span >= 300.0 and lat_span >= 120.0


def _missing_lon_indices_for_time(
    da: xr.DataArray,
    *,
    lon_dim: str,
    time_dim: str | None,
    time_index: int | None,
) -> np.ndarray[Any, Any]:
    section = (
        da
        if time_dim is None or time_index is None
        else da.isel({time_dim: time_index})
    )
    mask = _missing_mask(section)
    reduce_dims = [dim for dim in mask.dims if dim != lon_dim]
    if reduce_dims:
        mask = mask.all(dim=reduce_dims)
    return np.flatnonzero(np.asarray(mask.values, dtype=bool))


def _edge_of_map_check(
    da: xr.DataArray,
    *,
    lon_name: str,
    lon_dim: str,
    time_dim: str | None,
) -> dict[str, Any]:
    sampled_indices: list[int] = []
    persistent_missing_lon_indices = np.array([], dtype=int)

    if time_dim is None:
        sampled_indices = [0]
        persistent_missing_lon_indices = _missing_lon_indices_for_time(
            da,
            lon_dim=lon_dim,
            time_dim=None,
            time_index=None,
        )
    else:
        time_size = int(da.sizes[time_dim])
        last_idx = time_size - 1
        first_idx = 0

        sampled_indices.append(last_idx)
        missing_last = _missing_lon_indices_for_time(
            da,
            lon_dim=lon_dim,
            time_dim=time_dim,
            time_index=last_idx,
        )

        if first_idx != last_idx:
            sampled_indices.append(first_idx)
        missing_first = _missing_lon_indices_for_time(
            da,
            lon_dim=lon_dim,
            time_dim=time_dim,
            time_index=first_idx,
        )

        if missing_last.size and np.array_equal(missing_last, missing_first):
            persistent_missing_lon_indices = missing_last
            if time_size > 2:
                middle_idx = time_size // 2
                if middle_idx not in sampled_indices:
                    sampled_indices.append(middle_idx)
                missing_middle = _missing_lon_indices_for_time(
                    da,
                    lon_dim=lon_dim,
                    time_dim=time_dim,
                    time_index=middle_idx,
                )
                if not np.array_equal(missing_middle, persistent_missing_lon_indices):
                    persistent_missing_lon_indices = np.array([], dtype=int)

    lon_values = np.asarray(da.coords[lon_name].values)
    missing_lon_list = persistent_missing_lon_indices.tolist()
    missing_lon_values = [float(lon_values[idx]) for idx in missing_lon_list]

    return {
        "enabled": True,
        "status": "fail" if missing_lon_list else "pass",
        "sampled_time_indices": sampled_indices,
        "missing_longitude_count": len(missing_lon_list),
        "missing_longitudes": missing_lon_values,
        "missing_longitude_ranges": _value_ranges_from_indices(
            missing_lon_list, lon_values
        ),
    }


def _choose_time_vars(
    ds: xr.Dataset,
    *,
    var_name: str | None,
    time_name: str | None,
) -> list[tuple[xr.DataArray, str | None]]:
    if var_name is not None:
        if var_name not in ds.data_vars:
            raise ValueError(f"Data variable '{var_name}' not found.")
        da = ds[var_name]
        return [(da, _resolve_time_dim(da, time_name))]

    selected = [
        (da, _resolve_time_dim(da, time_name)) for _, da in ds.data_vars.items()
    ]
    if not selected:
        raise ValueError("Dataset has no data variables to check.")
    return selected


def _point_is_missing(point: xr.DataArray) -> bool:
    mask = _missing_mask(point)
    reduce_dims = list(mask.dims)
    if reduce_dims:
        mask = mask.all(dim=reduce_dims)
    return bool(np.asarray(mask.values).item())


def _point_alignment_check(
    da: xr.DataArray,
    *,
    lon_name: str,
    lat_name: str,
    lon_convention: str,
    time_dim: str | None,
    lon_values: np.ndarray[Any, Any],
    lat_values: np.ndarray[Any, Any],
) -> dict[str, Any]:
    global_grid = _is_global_grid(lon_values, lat_values)
    if not global_grid:
        return {
            "enabled": True,
            "status": "skipped_non_global",
            "mismatch_count": 0,
            "land_points_checked": 0,
            "ocean_points_checked": 0,
            "land_mismatches": [],
            "ocean_mismatches": [],
            "note": "Skipped land/ocean sanity check because grid does not appear global.",
        }

    section = da if time_dim is None else da.isel({time_dim: -1})
    land_mismatches: list[dict[str, Any]] = []
    ocean_mismatches: list[dict[str, Any]] = []

    def check_points(
        points: tuple[tuple[str, float, float], ...],
        *,
        expected_missing: bool,
    ) -> list[dict[str, Any]]:
        mismatches: list[dict[str, Any]] = []
        for label, lat, lon in points:
            target_lon = _normalize_lon_for_grid(lon, lon_convention)
            selected = section.sel(
                {lat_name: lat, lon_name: target_lon},
                method="nearest",
            )
            observed_missing = _point_is_missing(selected)
            if observed_missing == expected_missing:
                continue
            actual_lat = float(np.asarray(selected.coords[lat_name].values).item())
            actual_lon = float(np.asarray(selected.coords[lon_name].values).item())
            mismatches.append(
                {
                    "point": label,
                    "requested_lat": float(lat),
                    "requested_lon": float(target_lon),
                    "actual_lat": actual_lat,
                    "actual_lon": actual_lon,
                    "expected_missing": expected_missing,
                    "observed_missing": observed_missing,
                }
            )
        return mismatches

    land_mismatches = check_points(_LAND_REFERENCE_POINTS, expected_missing=True)
    ocean_mismatches = check_points(_OCEAN_REFERENCE_POINTS, expected_missing=False)
    mismatch_count = len(land_mismatches) + len(ocean_mismatches)

    return {
        "enabled": True,
        "status": "fail" if mismatch_count else "pass",
        "mismatch_count": mismatch_count,
        "land_points_checked": len(_LAND_REFERENCE_POINTS),
        "ocean_points_checked": len(_OCEAN_REFERENCE_POINTS),
        "land_mismatches": land_mismatches,
        "ocean_mismatches": ocean_mismatches,
    }


def _time_missing_check(da: xr.DataArray, *, time_dim: str | None) -> dict[str, Any]:
    if time_dim is None:
        return {
            "enabled": True,
            "status": "skipped_no_time",
            "missing_slice_count": 0,
            "missing_slice_ranges": [],
        }

    missing = _missing_mask(da)
    reduce_dims = [dim for dim in missing.dims if dim != time_dim]
    if reduce_dims:
        missing = missing.all(dim=reduce_dims)
    missing_time_indices = np.flatnonzero(
        np.asarray(missing.values, dtype=bool)
    ).tolist()
    time_coord = da.coords.get(time_dim)
    return {
        "enabled": True,
        "status": "fail" if missing_time_indices else "pass",
        "missing_slice_count": len(missing_time_indices),
        "missing_slice_ranges": _range_records(missing_time_indices, time_coord),
    }


def _single_ocean_report(
    da: xr.DataArray,
    *,
    lon_name: str,
    lat_name: str,
    lon_dim: str,
    lat_dim: str,
    lon_values: np.ndarray[Any, Any],
    lat_values: np.ndarray[Any, Any],
    time_name: str | None,
    check_edge_of_map: bool,
    check_land_ocean_offset: bool,
    check_time_missing: bool | None,
) -> dict[str, Any]:
    time_dim = _resolve_time_dim(da, time_name)
    lon_convention = _longitude_convention(lon_values)

    report: dict[str, Any] = {
        "variable": str(da.name),
        "grid": {
            "lon_name": lon_name,
            "lat_name": lat_name,
            "lon_dim": lon_dim,
            "lat_dim": lat_dim,
            "time_dim": time_dim,
            "longitude_convention": lon_convention,
            "longitude_min": float(np.nanmin(lon_values)),
            "longitude_max": float(np.nanmax(lon_values)),
            "latitude_min": float(np.nanmin(lat_values)),
            "latitude_max": float(np.nanmax(lat_values)),
        },
        "checks_enabled": {
            "edge_of_map": bool(check_edge_of_map),
            "land_ocean_offset": bool(check_land_ocean_offset),
        },
    }
    if check_time_missing is not None:
        report["note"] = (
            "Time coverage has moved to `check_time_cover()` / `ds.check.time_cover()`."
        )

    if check_edge_of_map:
        report["edge_of_map"] = _edge_of_map_check(
            da,
            lon_name=lon_name,
            lon_dim=lon_dim,
            time_dim=time_dim,
        )
    else:
        report["edge_of_map"] = {"enabled": False, "status": "skipped"}
    # Backward-compatible alias.
    report["edge_sliver"] = report["edge_of_map"]

    if check_land_ocean_offset:
        report["land_ocean_offset"] = _point_alignment_check(
            da,
            lon_name=lon_name,
            lat_name=lat_name,
            lon_convention=lon_convention,
            time_dim=time_dim,
            lon_values=lon_values,
            lat_values=lat_values,
        )
    else:
        report["land_ocean_offset"] = {"enabled": False, "status": "skipped"}

    statuses = [
        str(report["edge_of_map"].get("status")),
        str(report["land_ocean_offset"].get("status")),
    ]
    report["ok"] = not any(status in {"fail", "error"} for status in statuses)
    return report


def _single_time_cover_report(
    da: xr.DataArray,
    *,
    time_dim: str | None,
) -> dict[str, Any]:
    time_missing = _time_missing_check(da, time_dim=time_dim)
    statuses = [str(time_missing.get("status")).lower()]
    return {
        "variable": str(da.name),
        "time_dim": time_dim,
        "time_missing": time_missing,
        "ok": not any(status in {"fail", "error"} for status in statuses),
    }


def check_ocean_cover(
    ds: xr.Dataset,
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
    """Run ocean-coverage sanity checks on one or more gridded variables.

    Parameters
    ----------
    ds
        Dataset containing gridded data variables.
    var_name
        Optional variable name. If omitted, all variables that use inferred
        lon/lat dimensions are checked.
    lon_name, lat_name
        Longitude and latitude coordinate names. If omitted, they are inferred
        from common names/units.
    time_name
        Preferred time dimension name used by checks that sample in time.
    check_edge_of_map
        Check for persistent missing longitude bands at map edges.
    check_land_ocean_offset
        Check land/ocean fill alignment using reference points on global grids.
    report_format
        ``"python"``, ``"tables"``, ``"html"``, or ``"auto"``.
    report_html_file
        Output path when ``report_format="html"``.
    check_edge_sliver
        Backward-compatible alias for ``check_edge_of_map``.
    check_time_missing
        Deprecated compatibility option; time coverage moved to
        :func:`check_time_cover`.

    Returns
    -------
    dict | str | None
        Per-variable or multi-variable report for ``"python"``, HTML for
        ``"html"``, or ``None`` for ``"tables"`` (printed output).
    """
    resolved_format = normalize_report_format(report_format)
    if report_html_file is not None and resolved_format != "html":
        raise ValueError("`report_html_file` is only valid when report_format='html'.")

    if check_edge_sliver is not None:
        check_edge_of_map = bool(check_edge_sliver)
    lon_name = lon_name or _guess_coord_name(ds, _LON_CANDIDATES, "degrees_east")
    lat_name = lat_name or _guess_coord_name(ds, _LAT_CANDIDATES, "degrees_north")
    if lon_name is None or lat_name is None:
        raise ValueError(
            "Could not infer longitude/latitude coordinates. Pass `lon_name` and `lat_name`."
        )

    lon_dim, lon_values = _resolve_1d_coord(ds, lon_name)
    lat_dim, lat_values = _resolve_1d_coord(ds, lat_name)
    data_vars = _choose_data_vars(
        ds, var_name=var_name, lon_dim=lon_dim, lat_dim=lat_dim
    )
    reports: dict[str, dict[str, Any]] = {}
    for da in data_vars:
        per_var_report = _single_ocean_report(
            da,
            lon_name=lon_name,
            lat_name=lat_name,
            lon_dim=lon_dim,
            lat_dim=lat_dim,
            lon_values=lon_values,
            lat_values=lat_values,
            time_name=time_name,
            check_edge_of_map=check_edge_of_map,
            check_land_ocean_offset=check_land_ocean_offset,
            check_time_missing=check_time_missing,
        )
        reports[str(da.name)] = per_var_report

    if len(reports) == 1:
        report = next(iter(reports.values()))
        if resolved_format == "tables":
            print_pretty_ocean_reports([report])
            return None
        if resolved_format == "html":
            html_report = render_pretty_ocean_report_html(report)
            save_html_report(html_report, report_html_file)
            maybe_display_html_report(html_report)
            return html_report
        return report

    report = {
        "mode": "all_variables",
        "checked_variable_count": len(reports),
        "checked_variables": list(reports.keys()),
        "reports": reports,
        "ok": all(bool(per_var.get("ok")) for per_var in reports.values()),
    }
    if resolved_format == "tables":
        print_pretty_ocean_reports(list(reports.values()))
        return None
    if resolved_format == "html":
        html_report = render_pretty_ocean_reports_html(list(reports.values()))
        save_html_report(html_report, report_html_file)
        maybe_display_html_report(html_report)
        return html_report
    return report


def check_time_cover(
    ds: xr.Dataset,
    *,
    var_name: str | None = None,
    time_name: str | None = "time",
    report_format: ReportFormat = "auto",
    report_html_file: str | Path | None = None,
) -> dict[str, Any] | str | None:
    """Run time-coverage checks and report missing time-slice ranges.

    Parameters
    ----------
    ds
        Dataset containing variables to inspect.
    var_name
        Optional variable name. If omitted, all data variables are checked.
    time_name
        Preferred time dimension name.
    report_format
        ``"python"``, ``"tables"``, ``"html"``, or ``"auto"``.
    report_html_file
        Output path when ``report_format="html"``.

    Returns
    -------
    dict | str | None
        Per-variable or multi-variable report for ``"python"``, HTML for
        ``"html"``, or ``None`` for ``"tables"`` (printed output).
    """
    resolved_format = normalize_report_format(report_format)
    if report_html_file is not None and resolved_format != "html":
        raise ValueError("`report_html_file` is only valid when report_format='html'.")

    selected = _choose_time_vars(ds, var_name=var_name, time_name=time_name)
    reports: dict[str, dict[str, Any]] = {}
    for da, time_dim in selected:
        reports[str(da.name)] = _single_time_cover_report(da, time_dim=time_dim)

    if len(reports) == 1:
        report = next(iter(reports.values()))
        if resolved_format == "tables":
            print_pretty_time_cover_reports([report])
            return None
        if resolved_format == "html":
            html_report = render_pretty_time_cover_report_html(report)
            save_html_report(html_report, report_html_file)
            maybe_display_html_report(html_report)
            return html_report
        return report

    report = {
        "mode": "all_variables",
        "checked_variable_count": len(reports),
        "checked_variables": list(reports.keys()),
        "reports": reports,
        "ok": all(bool(per_var.get("ok")) for per_var in reports.values()),
    }
    if resolved_format == "tables":
        print_pretty_time_cover_reports(list(reports.values()))
        return None
    if resolved_format == "html":
        html_report = render_pretty_time_cover_reports_html(list(reports.values()))
        save_html_report(html_report, report_html_file)
        maybe_display_html_report(html_report)
        return html_report
    return report
