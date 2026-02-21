# Getting Started

## Install

```bash
uv add nc-check
# or
pip install nc-check
```

Install optional CF checker dependencies:

```bash
uv add "nc-check[cf]"
# or
pip install "nc-check[cf]"
```

## First Python Run

```python
import xarray as xr
import nc_check  # registers ds.check accessor

ds = xr.Dataset(
    data_vars={"temp": (("time", "lat", "lon"), [[[280.0]]])},
    coords={"time": [0], "lat": [10.0], "lon": [20.0]},
)

report = ds.check.compliance(report_format="python")
print(report["counts"])
```

## First CLI Run

```bash
# shorthand for: nc-check compliance input.nc
nc-check input.nc

# run all checks
nc-check all input.nc

# apply safe fixes and write a new file
nc-comply input.nc output.nc
```

## Output Formats

All check methods support `report_format`:

- `auto`: notebooks -> HTML, CLI -> rich tables, scripts/tests -> Python dict
- `python`: return a dictionary
- `tables`: print rich tables to stdout
- `html`: return HTML string (and optionally save with `report_html_file`)

## Common Workflow

1. Run `ds.check.compliance()` (or `nc-check compliance`) to find metadata issues.
2. Apply safe fixes with `ds.check.make_cf_compliant()` (or `nc-comply` for files).
3. Run `ds.check.all()` (or `nc-check all`) and save an HTML report for review.
