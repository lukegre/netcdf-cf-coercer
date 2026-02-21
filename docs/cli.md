# CLI Guide

`nc-check` provides check commands. `nc-comply` writes a CF-improved copy of a dataset.

## Commands

```bash
nc-check <command> <input.nc> [options]
```

Supported commands:

- `compliance`
- `ocean-cover`
- `time-cover`
- `all`

Shorthand:

```bash
nc-check input.nc
```

This is equivalent to:

```bash
nc-check compliance input.nc
```

## Examples

```bash
nc-check compliance input.nc
nc-check compliance input.nc --engine heuristic
nc-check compliance input.nc --conventions ferret

nc-check ocean-cover input.nc
nc-check ocean-cover input.nc --lon-name x --lat-name y --time-name t

nc-check time-cover input.nc
nc-check time-cover input.nc --time-name t

nc-check all input.nc
nc-check all input.nc --engine cfchecker
```

## Save HTML Reports

```bash
nc-check compliance input.nc --save-report
nc-check ocean-cover input.nc --save-report
nc-check time-cover input.nc --save-report
nc-check all input.nc --save-report
```

Output file naming:

- compliance: `<input>_report.html`
- ocean-cover: `<input>_ocean_cover_report.html`
- time-cover: `<input>_time_cover_report.html`
- all: `<input>_all_report.html`

## Apply Safe Fixes

```bash
nc-comply input.nc output.nc
```

This reads `input.nc`, applies `make_dataset_compliant`, and writes `output.nc`.

## Exit Codes

- `0`: success
- `1`: error (invalid file, check failure exception, write failure, etc.)
