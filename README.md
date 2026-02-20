# nc-check

Prepare `xarray.Dataset` objects to be written as CF-1.12-compliant NetCDF files.

## Install

```bash
uv sync
```

## Usage

```python
import xarray as xr
import nc_check  # Registers the .check dataset accessor

ds = xr.Dataset(
    data_vars={"temp": (("time", "lat", "lon"), [[[280.0]]])},
    coords={"time": [0], "lat": [10.0], "lon": [20.0]},
)

issues = ds.check.cf()
fixed = ds.check.make_cf_compliant()
ocean_report = ds.check.check_ocean_cover()
```

You can also request a YAML-like text report printed to stdout:

```python
ds.check.cf(pretty_print=True)
```

You can choose which conventions to check:

```python
ds.check.cf(conventions="cf,ferret")
ds.check.cf(conventions="ferret")  # custom-only checks
```

`check()` runs [cf-checker](https://github.com/cedadev/cf-checker/) against an
in-memory NetCDF payload created from dataset metadata (no `.nc` file written to disk),
and returns a dictionary of detected issues.

`make_cf_compliant()` returns a new dataset with safe automatic fixes, including:
- `Conventions = "CF-1.12"`
- standard coordinate attributes for inferred `time`, `lat`, and `lon` axes
- creation of missing dimension coordinates for inferred axes

`check_ocean_cover()` runs fast ocean-grid checks and returns a report with:
- east/west edge sliver detection (persistent missing longitude columns),
- land/ocean sanity checks at fixed reference points (offset detection),
- time-slice missing-data ranges.

You can disable individual checks:

```python
ds.check.check_ocean_cover(
    check_edge_sliver=True,
    check_land_ocean_offset=True,
    check_time_missing=True,
)
```

Notes:
- `cfchecker` requires the system `udunits2` library via `cfunits`.
- For large files, prefer opening with lazy chunks: `xr.open_dataset(path, chunks={})`.
- The built-in `ferret` convention flags coordinate `_FillValue` usage as an error.
- If `cfchecker` cannot run, `check()` falls back to heuristic checks and includes
  a `checker_error` field in the response.
- You can bias standard-name suggestions by domain, e.g.
  `ds.check.cf(domain="ocean")` (also supports `atmosphere`, `land`, `cryosphere`,
  and `biogeochemistry`).
