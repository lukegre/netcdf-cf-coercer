# Development

## Setup

```bash
uv sync --group dev
```

If you want optional CF checker support in development:

```bash
uv sync --group dev --extra cf
```

## Run Tests

```bash
uv run pytest
```

## Project Layout

- `src/nc_check/core.py`: compliance checks and compliance coercion
- `src/nc_check/ocean.py`: ocean/time coverage checks
- `src/nc_check/accessor.py`: `xarray.Dataset.check` accessor API
- `src/nc_check/cli.py`: CLI entrypoints (`nc-check`, `nc-comply`)
- `src/nc_check/formatting.py`: table/html/python report formatting
- `tests/`: test suite

## Local Smoke Check

```bash
uv run python -c "import nc_check, xarray as xr; print('ok')"
```
