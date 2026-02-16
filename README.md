# netcdf-cf-coercer

Prepare `xarray.Dataset` objects to be written as CF-1.12-compliant NetCDF files.

## Install

```bash
uv sync
```

## Usage

```python
import xarray as xr
import netcdf_cf_coercer  # Registers the .cf dataset accessor

ds = xr.Dataset(
    data_vars={"temp": (("time", "lat", "lon"), [[[280.0]]])},
    coords={"time": [0], "lat": [10.0], "lon": [20.0]},
)

issues = ds.cf.check_compliant()
fixed = ds.cf.make_compliant()
```

`check_compliant()` runs [cf-checker](https://github.com/cedadev/cf-checker/) against an
in-memory NetCDF payload created from the dataset (no `.nc` file written to disk), and
returns a dictionary of detected issues.

`make_compliant()` returns a new dataset with safe automatic fixes, including:
- `Conventions = "CF-1.12"`
- standard coordinate attributes for inferred `time`, `lat`, and `lon` axes
- creation of missing dimension coordinates for inferred axes
- optional `coordinates` attribute on data variables for clarity

Notes:
- `cfchecker` requires the system `udunits2` library via `cfunits`.
- If `cfchecker` cannot run, `check_compliant()` falls back to heuristic checks and includes
  a `checker_error` field in the response.
- You can bias standard-name suggestions by domain, e.g.
  `ds.cf.check_compliant(domain="ocean")` (also supports `atmosphere`, `land`, `cryosphere`,
  and `biogeochemistry`).
