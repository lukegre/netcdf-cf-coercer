# Python API Guide

Importing `nc_check` registers a dataset accessor at `ds.check`.

```python
import xarray as xr
import nc_check
```

## Main Methods

### `ds.check.compliance(...)`

Run convention checks (`cf`, `ferret`, or both).

Common options:

- `conventions`: `"cf,ferret"` (default), `"cf"`, or `"ferret"`
- `engine`: `"auto"` (default), `"cfchecker"`/`"cfcheck"`, `"heuristic"`
- `fallback_to_heuristic`: `True` by default
- `domain`: optional standard-name suggestion domain (`ocean`, `atmosphere`, `land`, `cryosphere`, `biogeochemistry`)
- `report_format`: `auto`, `python`, `tables`, `html`
- `report_html_file`: path, only valid with `report_format="html"`

Returns:

- `dict` for `python`
- `str` for `html`
- `None` for `tables`

### `ds.check.make_cf_compliant()`

Returns a copied dataset with safe metadata fixes. The original dataset is unchanged.

Key updates:

- sets `Conventions = "CF-1.12"`
- normalizes key CF attribute names to expected lowercase forms
- fills inferred axis coordinate metadata
- creates missing inferred axis coordinates when needed
- sets extent attrs when possible:
  - `time_coverage_start`
  - `time_coverage_end`
  - `geospatial_lat_min`
  - `geospatial_lat_max`
  - `geospatial_lon_min`
  - `geospatial_lon_max`
- removes coordinate `_FillValue` from attrs/encoding

### `ds.check.ocean_cover(...)`

Ocean-grid sanity checks for one variable or all eligible variables.

Common options:

- `var_name`: optional variable name
- `lon_name`, `lat_name`: optional coordinate names (auto-inferred if omitted)
- `time_name`: preferred time dim name (default `time`)
- `check_edge_of_map`: detect persistent missing longitude bands
- `check_land_ocean_offset`: reference-point land/ocean alignment test
- `report_format`, `report_html_file`

### `ds.check.time_cover(...)`

Time slice missing-data coverage checks.

Common options:

- `var_name`: optional variable name
- `time_name`: preferred time dim name (default `time`)
- `report_format`, `report_html_file`

### `ds.check.all(...)`

Runs selected checks and returns one combined report.

Check toggles:

- `compliance=True`
- `ocean_cover=True`
- `time_cover=True`

Forwards method-specific options to each check family and produces:

- `check_summary`
- per-check reports
- overall status fields (`overall_status`, `overall_ok`)

## Aliases

Backward-compatible aliases are available:

- `ds.check.cf()` -> `compliance()`
- `ds.check.comply()` and `ds.check.make_compliant()` -> `make_cf_compliant()`
- `ds.check.check_ocean_cover()` -> `ocean_cover()`
- `ds.check.check_time_cover()` -> `time_cover()`
- `ds.check.full()` -> `all()`
