# Checks And Reports

## Compliance Check

`ds.check.compliance()` combines:

- CF checks via `cfchecker` when available (or when requested explicitly)
- heuristic checks
- extra convention checks (`cf` and/or `ferret`)

If CF checker cannot run and fallback is enabled, output includes `checker_error` and uses heuristic results.

## Ocean Cover Check

`ds.check.ocean_cover()` includes:

- `edge_of_map`: persistent missing longitude columns (sliver/edge detection)
- `land_ocean_offset`: land/ocean reference-point sanity check on global grids

When `var_name` is omitted, it checks all variables containing inferred lon/lat dims and returns a multi-variable report.

## Time Cover Check

`ds.check.time_cover()` detects missing time slices by variable and reports contiguous missing ranges.

If a variable has no resolved time dim, the time-missing subcheck is marked as skipped (`skipped_no_time`).

## Combined Check

`ds.check.all()` aggregates compliance, ocean, and time checks into one report.

The summary includes:

- `checks_run`
- `failing_checks`
- `warnings_or_skips`
- `overall_status` (`pass`, `warn`, `fail`)
- `overall_ok` (boolean)

## Report Formats

All check methods support:

- `python`: return machine-friendly dict
- `tables`: print rich tables to stdout
- `html`: return HTML
- `auto`: environment-aware default

`auto` resolution:

- Jupyter notebook -> `html`
- interactive terminal -> `tables`
- non-interactive script/test -> `python`

## HTML Reports

For Python calls:

```python
html = ds.check.all(report_format="html", report_html_file="all-report.html")
```

For CLI calls:

```bash
nc-check all input.nc --save-report
```
