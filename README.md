# nc-check

Prepare `xarray.Dataset` objects to be written as CF-1.12-compliant NetCDF files.

## Install

```bash
uv sync
```

Install optional CF checker dependencies (recommended for full CF checks):

```bash
uv sync --extra cf
```

## Usage

```python
import xarray as xr
import nc_check  # Registers the .check dataset accessor

ds = xr.Dataset(
    data_vars={"temp": (("time", "lat", "lon"), [[[280.0]]])},
    coords={"time": [0], "lat": [10.0], "lon": [20.0]},
)

issues = ds.check.compliance(report_format="python")
fixed = ds.check.make_cf_compliant()
ocean_report = ds.check.ocean_cover(report_format="python")
time_report = ds.check.time_cover(report_format="python")
```

By default (`report_format="auto"`), reports adapt to environment:
- Jupyter notebooks: HTML
- CLI terminals: rich tables
- Other environments (scripts/tests): Python dicts

You can request rich table reports printed to stdout:

```python
ds.check.compliance(report_format="tables")
```

CLI usage:

```bash
# Prints readable rich tables to stdout (default behavior)
nc-check input.nc

# Explicit check modes
nc-check compliance input.nc
nc-check compliance input.nc --engine heuristic
nc-check ocean-cover input.nc
nc-check time-cover input.nc
nc-check all input.nc
nc-check all input.nc --engine cfchecker

# Explicit coordinate names when they differ from lon/lat/time
nc-check ocean-cover input.nc --lon-name x --lat-name y --time-name t
nc-check time-cover input.nc --time-name t
nc-check all input.nc --lon-name x --lat-name y --time-name t

# Saves an HTML report beside input.nc as input_report.html
nc-check input.nc --save-report

# With "all", saves one combined report:
# input_all_report.html
nc-check all input.nc --save-report
```

For notebooks, use HTML output and optionally save it:

```python
html = ds.check.compliance(
    report_format="html",
    report_html_file="cf-report.html",
)
```

You can choose which conventions to check:

```python
ds.check.compliance(conventions="cf,ferret")
ds.check.compliance(conventions="ferret")  # custom-only checks
ds.check.compliance(engine="heuristic")  # force built-in heuristic engine
ds.check.compliance(engine="cfchecker")  # require cfchecker path

# Explicit coordinate names
ds.check.ocean_cover(lon_name="x", lat_name="y", time_name="t")
ds.check.time_cover(time_name="t")
ds.check.all(lon_name="x", lat_name="y", time_name="t")
```

`compliance()` runs [cf-checker](https://github.com/cedadev/cf-checker/) against an
in-memory NetCDF payload created from dataset metadata (no `.nc` file written to disk).
With `report_format="python"` it returns a dictionary of detected issues.

`make_cf_compliant()` returns a new dataset with safe automatic fixes, including:
- `Conventions = "CF-1.12"`
- standard coordinate attributes for inferred `time`, `lat`, and `lon` axes
- creation of missing dimension coordinates for inferred axes
- global extent attributes derived from inferred axes:
  `time_coverage_start`, `time_coverage_end`, `geospatial_lat_min`, `geospatial_lat_max`,
  `geospatial_lon_min`, and `geospatial_lon_max`

`ocean_cover()` runs fast ocean-grid checks and returns a report with:
- east/west edge-of-map detection (persistent missing longitude columns, reported by longitude),
- land/ocean sanity checks at fixed reference points (offset detection).
- `report_format="auto"` by default.
- When `var_name` is omitted, all data variables with inferred lat/lon dimensions are checked.

`time_cover()` runs time-dimension missing-data checks and reports missing
time-slice ranges.
- `report_format="auto"` by default.
- When `var_name` is omitted, all data variables are checked.
- Time coordinate units/type validation is reported by `compliance()`.

You can disable individual checks:

```python
ds.check.ocean_cover(
    check_edge_of_map=True,
    check_land_ocean_offset=True,
    report_format="python",
)
```

```python
ds.check.time_cover(
    report_format="python",
)
```

Notes:
- `cfchecker` requires the system `udunits2` library via `cfunits`.
- For large files, prefer opening with lazy chunks: `xr.open_dataset(path, chunks={})`.
- The built-in `ferret` convention flags coordinate `_FillValue` usage as an error.
- If `cfchecker` cannot run, `compliance()` falls back to heuristic checks and includes
  a `checker_error` field in the response.
- You can choose the compliance engine explicitly with
  `engine="heuristic"` or `engine="cfchecker"` (CLI: `--engine`).
- The report `engine` field reflects what ran: `cfchecker`, `heuristic`, or `none`
  (when CF checks are skipped, e.g. `conventions="ferret"`).
- You can bias standard-name suggestions by domain, e.g.
  `ds.check.compliance(domain="ocean")` (also supports `atmosphere`, `land`, `cryosphere`,
  and `biogeochemistry`).
