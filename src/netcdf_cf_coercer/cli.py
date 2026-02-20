from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

from .core import check_dataset_compliant, make_dataset_compliant


def _existing_file(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_file():
        raise argparse.ArgumentTypeError(f"file not found: {path}")
    return candidate


def run_check(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nc-cf-check",
        description="Run CF checks and print the same report as pretty_print=True.",
    )
    parser.add_argument("fname", type=_existing_file, help="Input NetCDF file")
    parser.add_argument(
        "--conventions",
        default="cf,ferret",
        help="Comma-separated conventions to check (default: cf,ferret).",
    )
    args = parser.parse_args(argv)

    try:
        with xr.open_dataset(args.fname, chunks={}) as ds:
            check_dataset_compliant(
                ds,
                conventions=args.conventions,
                pretty_print=True,
            )
    except Exception as exc:
        print(f"nc-cf-check: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def run_comply(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nc-cf-comply",
        description="Apply safe CF compliance fixes and write a new NetCDF file.",
    )
    parser.add_argument("fname_in", type=_existing_file, help="Input NetCDF file")
    parser.add_argument("fname_out", type=Path, help="Output NetCDF file")
    args = parser.parse_args(argv)

    try:
        with xr.open_dataset(args.fname_in, chunks={}) as ds:
            compliant = make_dataset_compliant(ds)
            compliant.to_netcdf(args.fname_out)
    except Exception as exc:
        print(f"nc-cf-comply: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def main_check() -> None:
    raise SystemExit(run_check())


def main_comply() -> None:
    raise SystemExit(run_comply())
