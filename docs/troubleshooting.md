# Troubleshooting

## `cfchecker is not installed`

Install optional CF extras:

```bash
uv sync --extra cf
# or
pip install "nc-check[cf]"
```

If you do not need external CF checker behavior, you can run heuristic mode:

```python
ds.check.compliance(engine="heuristic", report_format="python")
```

## `report_html_file` errors

`report_html_file` is only valid when `report_format="html"`.

Example:

```python
ds.check.compliance(report_format="html", report_html_file="report.html")
```

## Coordinate inference fails for ocean checks

If lon/lat cannot be inferred, pass explicit names:

```python
ds.check.ocean_cover(lon_name="x", lat_name="y", time_name="t")
```

CLI equivalent:

```bash
nc-check ocean-cover input.nc --lon-name x --lat-name y --time-name t
```

## Unexpected time check skips

`time_cover` can skip a variable if no matching time dimension is found.

Set the preferred time name explicitly:

```python
ds.check.time_cover(time_name="t")
```

## Large datasets

Use lazy opening for low-memory checks:

```python
import xarray as xr
ds = xr.open_dataset("input.nc", chunks={})
```
