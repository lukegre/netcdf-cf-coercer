from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

from .core import check_dataset_compliant, make_dataset_compliant
from .ocean import check_ocean_cover, check_time_cover

_CHECK_MODES = {"compliance", "ocean-cover", "time-cover", "all"}


def _existing_file(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_file():
        raise argparse.ArgumentTypeError(f"file not found: {path}")
    return candidate


def _normalize_check_argv(argv: list[str] | None) -> list[str]:
    """Support `nc-check <file>` as shorthand for `nc-check compliance <file>`."""
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        return raw
    first = raw[0]
    if first in _CHECK_MODES or first.startswith("-"):
        return raw
    return ["compliance", *raw]


def _build_check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nc-check",
        description=(
            "Run NetCDF checks with git-style subcommands.\n"
            "Use `nc-check <command> --help` for command-specific options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  nc-check compliance input.nc\n"
            "  nc-check ocean-cover input.nc\n"
            "  nc-check time-cover input.nc\n"
            "  nc-check all input.nc --save-report\n"
            "  nc-check input.nc   # shorthand for `nc-check compliance input.nc`"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True, title="Commands")

    def _add_shared_options(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "fname", type=_existing_file, help="Input NetCDF file"
        )
        command_parser.add_argument(
            "--save-report",
            action="store_true",
            help=(
                "Save HTML report(s) next to the input file using "
                "'<input>_<command>_report.html'. Compliance uses "
                "'<input>_report.html'."
            ),
        )

    compliance = subparsers.add_parser(
        "compliance",
        help="Run CF/Ferret compliance checks.",
    )
    _add_shared_options(compliance)
    compliance.add_argument(
        "--conventions",
        default="cf,ferret",
        help="Comma-separated conventions to check (default: cf,ferret).",
    )

    ocean_cover = subparsers.add_parser(
        "ocean-cover",
        help="Run ocean-coverage checks.",
    )
    _add_shared_options(ocean_cover)

    time_cover = subparsers.add_parser(
        "time-cover",
        help="Run time-coverage checks.",
    )
    _add_shared_options(time_cover)

    check_all = subparsers.add_parser(
        "all",
        help="Run compliance, ocean-cover, and time-cover checks.",
    )
    _add_shared_options(check_all)
    check_all.add_argument(
        "--conventions",
        default="cf,ferret",
        help="Comma-separated conventions to check for compliance (default: cf,ferret).",
    )

    return parser


def run_check(argv: list[str] | None = None) -> int:
    parser = _build_check_parser()
    args = parser.parse_args(_normalize_check_argv(argv))

    mode = str(args.command)
    input_file: Path = args.fname

    report_format = "html" if args.save_report else "tables"
    report_html_file = (
        _default_report_html_path(input_file, mode) if args.save_report else None
    )
    conventions = getattr(args, "conventions", "cf,ferret")

    try:
        with xr.open_dataset(input_file, chunks={}) as ds:
            if mode == "compliance":
                check_dataset_compliant(
                    ds,
                    conventions=conventions,
                    report_format=report_format,
                    report_html_file=report_html_file,
                )
            elif mode == "ocean-cover":
                check_ocean_cover(
                    ds,
                    report_format=report_format,
                    report_html_file=report_html_file,
                )
            elif mode == "time-cover":
                check_time_cover(
                    ds,
                    report_format=report_format,
                    report_html_file=report_html_file,
                )
            elif mode == "all":
                compliance_report = (
                    _default_report_html_path(input_file, "compliance")
                    if args.save_report
                    else None
                )
                ocean_report = (
                    _default_report_html_path(input_file, "ocean-cover")
                    if args.save_report
                    else None
                )
                time_report = (
                    _default_report_html_path(input_file, "time-cover")
                    if args.save_report
                    else None
                )
                check_dataset_compliant(
                    ds,
                    conventions=conventions,
                    report_format=report_format,
                    report_html_file=compliance_report,
                )
                check_ocean_cover(
                    ds,
                    report_format=report_format,
                    report_html_file=ocean_report,
                )
                check_time_cover(
                    ds,
                    report_format=report_format,
                    report_html_file=time_report,
                )
            else:
                parser.error(f"Unsupported mode: {mode}")
    except Exception as exc:
        print(f"nc-check: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def _default_report_html_path(input_file: Path, mode: str = "compliance") -> Path:
    name = input_file.name
    if name.lower().endswith(".nc"):
        stem = name[:-3]
    else:
        stem = input_file.stem

    if mode == "compliance":
        suffix = "_report"
    else:
        suffix = f"_{mode.replace('-', '_')}_report"
    report_name = f"{stem}{suffix}.html"
    return input_file.with_name(report_name)


def run_comply(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nc-comply",
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
        print(f"nc-comply: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def main_check() -> None:
    raise SystemExit(run_check())


def main_comply() -> None:
    raise SystemExit(run_comply())
