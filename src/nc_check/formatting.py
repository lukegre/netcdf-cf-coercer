from __future__ import annotations

from html import escape
from pathlib import Path
import sys
from typing import Any
from typing import Literal

from .report_templates import render_report_document

ResolvedReportFormat = Literal["python", "tables", "html"]
ReportFormat = Literal["auto", "python", "tables", "html"]
_RICH_BORDER_STYLE = "white"
_RICH_TITLE_STYLE = "bold white"
_RICH_HEADER_STYLE = "bold white"
_RICH_LABEL_STYLE = "bold white"
_RICH_TEXT_STYLE = "white"
_RICH_TABLE_BORDER_STYLE = "bright_black"


def _stringify(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def to_yaml_like(value: Any, indent: int = 0) -> str:
    """Keep a plain YAML-like formatter for simple debug/text use."""
    space = " " * indent
    if isinstance(value, dict):
        if not value:
            return f"{space}{{}}"
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{space}{key}:")
                lines.append(to_yaml_like(item, indent + 2))
            else:
                lines.append(f"{space}{key}: {_stringify(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{space}[]"
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{space}-")
                lines.append(to_yaml_like(item, indent + 2))
            else:
                lines.append(f"{space}- {_stringify(item)}")
        return "\n".join(lines)
    return f"{space}{_stringify(value)}"


def _severity_style(severity: str | None) -> str:
    sev = (severity or "").upper()
    if sev == "FATAL":
        return "bold red"
    if sev == "ERROR":
        return "bold bright_red"
    if sev == "WARN":
        return "yellow"
    return _RICH_TEXT_STYLE


def _status_style(status: str | None) -> str:
    normalized = (status or "").lower()
    if normalized in {"fail", "error"}:
        return "bold bright_red"
    if normalized.startswith("skip"):
        return "bright_black"
    if normalized == "pass":
        return "bright_green"
    return _RICH_TEXT_STYLE


def _status_sort_key(status: Any) -> int:
    normalized = _stringify(status).strip().lower()
    if normalized in {"fail", "failed", "error", "fatal", "false"}:
        return 0
    if normalized in {"warn", "warning", "skip", "skipped"} or normalized.startswith(
        "skip"
    ):
        return 1
    return 2


def _severity_sort_key(severity: Any) -> int:
    normalized = _stringify(severity).strip().upper()
    if normalized == "FATAL":
        return 0
    if normalized == "ERROR":
        return 1
    if normalized == "WARN":
        return 2
    if normalized == "INFO":
        return 3
    return 4


def _count_to_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except Exception:
        return 0


def _running_in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except Exception:
        return False

    shell = get_ipython()
    if shell is None:
        return False

    shell_name = shell.__class__.__name__
    if shell_name == "ZMQInteractiveShell":
        return True
    if shell_name == "TerminalInteractiveShell":
        return False
    return bool(getattr(shell, "kernel", None))


def _running_in_cli() -> bool:
    stdout = getattr(sys, "stdout", None)
    if stdout is None:
        return False
    try:
        return bool(stdout.isatty())
    except Exception:
        return False


def _resolve_auto_report_format() -> ResolvedReportFormat:
    if _running_in_notebook():
        return "html"
    if _running_in_cli():
        return "tables"
    return "python"


def normalize_report_format(report_format: str) -> ResolvedReportFormat:
    normalized = report_format.strip().lower()
    if normalized == "auto":
        return _resolve_auto_report_format()
    allowed = {"auto", "python", "tables", "html"}
    if normalized not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(
            f"Unsupported report_format '{report_format}'. Expected one of: {expected}."
        )
    return normalized  # type: ignore[return-value]


def save_html_report(html: str, report_html_file: str | Path | None) -> None:
    if report_html_file is None:
        return
    output_path = Path(report_html_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def maybe_display_html_report(html: str) -> None:
    try:
        from IPython.display import HTML, display

        display(HTML(html))
    except Exception:
        return


def _yaml_html_fallback(report: Any) -> str:
    yaml_like = to_yaml_like(report) if isinstance(report, dict) else _stringify(report)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Report</title></head>"
        f"<body><pre>{escape(yaml_like)}</pre></body></html>"
    )


def _render_cf_report_with_rich(console: Any, report: dict[str, Any]) -> None:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    title = Text("CF Compliance Report", style=_RICH_TITLE_STYLE)
    console.print(Panel(title, border_style=_RICH_BORDER_STYLE))

    counts = report.get("counts") or {}
    fatal = _count_to_int(counts.get("fatal", 0)) if isinstance(counts, dict) else 0
    error = _count_to_int(counts.get("error", 0)) if isinstance(counts, dict) else 0
    warn = _count_to_int(counts.get("warn", 0)) if isinstance(counts, dict) else 0
    non_compliant = fatal + error
    checker_error = report.get("checker_error")
    if checker_error:
        outcome = ("ERROR", "bold bright_red")
    elif non_compliant > 0:
        outcome = ("FAILED", "bold bright_red")
    elif warn > 0:
        outcome = ("WARNING", "yellow")
    else:
        outcome = ("PASSED", "bright_green")

    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column("k", style=_RICH_LABEL_STYLE, justify="left")
    summary.add_column("v", style=_RICH_TEXT_STYLE, justify="left")
    summary.add_row("Outcome", Text(outcome[0], style=outcome[1]))
    summary.add_row("Errors + fatals", str(non_compliant))
    summary.add_row("Warnings", str(warn))
    console.print(Panel(summary, title="Summary", border_style=_RICH_BORDER_STYLE))

    meta = Table(show_header=False, box=None, pad_edge=False)
    meta.add_column("k", style=_RICH_LABEL_STYLE, justify="left")
    meta.add_column("v", style=_RICH_TEXT_STYLE, justify="left")
    meta.add_row("CF version", _stringify(report.get("cf_version")))
    meta.add_row("Engine", _stringify(report.get("engine")))
    meta.add_row("Engine status", _stringify(report.get("engine_status")))
    meta.add_row("Check method", _stringify(report.get("check_method")))
    conventions_checked = report.get("conventions_checked")
    if isinstance(conventions_checked, list) and conventions_checked:
        meta.add_row("Conventions", ", ".join(str(c) for c in conventions_checked))
    if isinstance(counts, dict):
        meta.add_row(
            "Counts",
            f"fatal={counts.get('fatal', 0)} error={counts.get('error', 0)} warn={counts.get('warn', 0)}",
        )
    console.print(meta)

    def print_finding_table(
        title_text: str, rows: list[tuple[str, str, str, str]]
    ) -> None:
        if not rows:
            return
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                _severity_sort_key(row[2]),
                row[0].lower(),
                row[1].lower(),
                row[3].lower(),
            ),
        )
        table = Table(
            title=title_text,
            title_style=_RICH_TITLE_STYLE,
            header_style=_RICH_HEADER_STYLE,
        )
        table.add_column("Scope", style=_RICH_TEXT_STYLE, justify="left")
        table.add_column("Convention", style=_RICH_TEXT_STYLE, justify="left")
        table.add_column("Severity", style=_RICH_TEXT_STYLE, justify="left")
        table.add_column("Detail", style=_RICH_TEXT_STYLE, justify="left")
        for scope, convention, severity, detail in sorted_rows:
            sev_text = Text(severity, style=_severity_style(severity))
            table.add_row(scope, convention, sev_text, detail)
        console.print(table)

    global_rows: list[tuple[str, str, str, str]] = []
    for item in report.get("global", []) or []:
        if isinstance(item, dict):
            severity = _stringify(item.get("severity", "INFO"))
            detail = _stringify(item.get("message") or item.get("item") or item)
            convention = _stringify(item.get("convention", "cf"))
        else:
            severity = "INFO"
            detail = _stringify(item)
            convention = "n/a"
        global_rows.append(("global", convention, severity, detail))
    print_finding_table("Global Findings", global_rows)

    coord_rows: list[tuple[str, str, str, str]] = []
    for coord_name, items in (report.get("coordinates") or {}).items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                severity = _stringify(item.get("severity", "INFO"))
                detail = _stringify(item.get("message") or item.get("item") or item)
                convention = _stringify(item.get("convention", "cf"))
            else:
                severity = "INFO"
                detail = _stringify(item)
                convention = "n/a"
            coord_rows.append((str(coord_name), convention, severity, detail))
    print_finding_table("Coordinate Findings", coord_rows)

    var_rows: list[tuple[str, str, str, str]] = []
    for var_name, items in (report.get("variables") or {}).items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                severity = _stringify(item.get("severity", "INFO"))
                detail = _stringify(item.get("message") or item.get("item") or item)
                convention = _stringify(item.get("convention", "cf"))
            else:
                severity = "INFO"
                detail = _stringify(item)
                convention = "n/a"
            var_rows.append((str(var_name), convention, severity, detail))
    print_finding_table("Variable Findings", var_rows)

    suggestions = (report.get("suggestions") or {}).get("variables") or {}
    if isinstance(suggestions, dict) and suggestions:
        suggestion_table = Table(
            title="Suggested Improvements",
            title_style=_RICH_TITLE_STYLE,
            header_style=_RICH_HEADER_STYLE,
        )
        suggestion_table.add_column("Variable", style=_RICH_TEXT_STYLE, justify="left")
        suggestion_table.add_column(
            "Suggestion", style=_RICH_TEXT_STYLE, justify="left"
        )
        for var_name, suggestion in suggestions.items():
            suggestion_table.add_row(str(var_name), _stringify(suggestion))
        console.print(suggestion_table)

    notes = report.get("notes") or []
    if isinstance(notes, list) and notes:
        note_table = Table(
            title="Notes",
            title_style=_RICH_TITLE_STYLE,
            header_style=_RICH_HEADER_STYLE,
        )
        note_table.add_column("Note", style=_RICH_TEXT_STYLE, justify="left")
        for note in notes:
            note_table.add_row(_stringify(note))
        console.print(note_table)

    if checker_error:
        error_text = _stringify(checker_error)
        console.print(Panel(error_text, title="Checker Error", border_style="red"))


def _render_ocean_report_with_rich(console: Any, report: dict[str, Any]) -> None:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    variable_name = report.get("variable")
    section_title = str(variable_name) if variable_name is not None else "Variable"
    title = Text(section_title, style=_RICH_TITLE_STYLE)
    console.print(Panel(title, border_style=_RICH_BORDER_STYLE))

    edge = (
        report.get("edge_of_map") if isinstance(report.get("edge_of_map"), dict) else {}
    )
    if not edge and isinstance(report.get("edge_sliver"), dict):
        edge = report.get("edge_sliver")
    offset = (
        report.get("land_ocean_offset")
        if isinstance(report.get("land_ocean_offset"), dict)
        else {}
    )
    time_missing = (
        report.get("time_missing")
        if isinstance(report.get("time_missing"), dict)
        else {}
    )

    summary_rows: list[tuple[str, str, str]] = []
    edge_status = _stringify(edge.get("status"))
    summary_rows.append(
        (
            "edge_of_map",
            edge_status,
            f"missing_longitudes={_stringify(edge.get('missing_longitude_count', 0))}",
        )
    )
    offset_status = _stringify(offset.get("status"))
    summary_rows.append(
        (
            "land_ocean_offset",
            offset_status,
            f"mismatches={_stringify(offset.get('mismatch_count', 0))}",
        )
    )
    if time_missing:
        time_status = _stringify(time_missing.get("status"))
        summary_rows.append(
            (
                "time_missing",
                time_status,
                f"missing_slices={_stringify(time_missing.get('missing_slice_count', 0))}",
            )
        )

    sorted_summary_rows = sorted(summary_rows, key=lambda row: _status_sort_key(row[1]))
    failing_checks = sum(
        1 for _, status, _ in sorted_summary_rows if _status_kind(status) == "fail"
    )
    warning_checks = sum(
        1 for _, status, _ in sorted_summary_rows if _status_kind(status) == "warn"
    )
    summary_head = Table(show_header=False, box=None, pad_edge=False)
    summary_head.add_column("k", style=_RICH_LABEL_STYLE, justify="left")
    summary_head.add_column("v", style=_RICH_TEXT_STYLE, justify="left")
    summary_head.add_row(
        "Overall",
        Text(
            "PASSED" if bool(report.get("ok")) else "FAILED",
            style=_status_style("pass" if bool(report.get("ok")) else "fail"),
        ),
    )
    summary_head.add_row("Failing checks", str(failing_checks))
    summary_head.add_row("Warnings/skips", str(warning_checks))
    console.print(
        Panel(
            summary_head,
            title="Summary",
            border_style=_RICH_BORDER_STYLE,
            expand=False,
        )
    )

    grid = report.get("grid") if isinstance(report.get("grid"), dict) else {}
    meta = Table(show_header=False, box=None, pad_edge=False)
    meta.add_column("k", style=_RICH_LABEL_STYLE, justify="left")
    meta.add_column("v", style=_RICH_TEXT_STYLE, justify="left")
    meta.add_row("Variable", _stringify(report.get("variable")))
    meta.add_row(
        "Longitude",
        f"{_stringify(grid.get('lon_name'))} ({_stringify(grid.get('lon_dim'))})",
    )
    meta.add_row(
        "Latitude",
        f"{_stringify(grid.get('lat_name'))} ({_stringify(grid.get('lat_dim'))})",
    )
    meta.add_row("Time dim", _stringify(grid.get("time_dim")))
    meta.add_row("Lon convention", _stringify(grid.get("longitude_convention")))
    meta.add_row(
        "Lon range",
        f"{_stringify(grid.get('longitude_min'))} .. {_stringify(grid.get('longitude_max'))}",
    )
    meta.add_row(
        "Lat range",
        f"{_stringify(grid.get('latitude_min'))} .. {_stringify(grid.get('latitude_max'))}",
    )
    console.print(meta)

    summary = Table(
        title="Check Summary",
        title_style=_RICH_TITLE_STYLE,
        header_style=_RICH_HEADER_STYLE,
        border_style=_RICH_TABLE_BORDER_STYLE,
    )
    summary.add_column("Check", style=_RICH_TEXT_STYLE, justify="left")
    summary.add_column("Status", style=_RICH_TEXT_STYLE, justify="left")
    summary.add_column("Detail", style=_RICH_TEXT_STYLE, justify="left")
    for check_name, status, detail in sorted_summary_rows:
        summary.add_row(check_name, Text(status, style=_status_style(status)), detail)
    console.print(summary)

    def _print_ranges(title_text: str, ranges: Any) -> None:
        if not isinstance(ranges, list) or not ranges:
            return
        table = Table(
            title=title_text,
            title_style=_RICH_TITLE_STYLE,
            header_style=_RICH_HEADER_STYLE,
        )
        table.add_column("Start", style=_RICH_TEXT_STYLE, justify="left")
        table.add_column("End", style=_RICH_TEXT_STYLE, justify="left")
        table.add_column("Start idx", style=_RICH_TEXT_STYLE, justify="left")
        table.add_column("End idx", style=_RICH_TEXT_STYLE, justify="left")
        for entry in ranges:
            if not isinstance(entry, dict):
                continue
            table.add_row(
                _stringify(entry.get("start")),
                _stringify(entry.get("end")),
                _stringify(entry.get("start_index")),
                _stringify(entry.get("end_index")),
            )
        console.print(table)

    def _print_value_ranges(title_text: str, ranges: Any) -> None:
        if not isinstance(ranges, list) or not ranges:
            return
        table = Table(
            title=title_text,
            title_style=_RICH_TITLE_STYLE,
            header_style=_RICH_HEADER_STYLE,
        )
        table.add_column("Start", style=_RICH_TEXT_STYLE, justify="left")
        table.add_column("End", style=_RICH_TEXT_STYLE, justify="left")
        for entry in ranges:
            if not isinstance(entry, dict):
                continue
            table.add_row(
                _stringify(entry.get("start")),
                _stringify(entry.get("end")),
            )
        console.print(table)

    _print_value_ranges(
        "Edge of Map Missing Longitude Ranges", edge.get("missing_longitude_ranges")
    )
    if time_missing:
        _print_ranges(
            "Time Missing Slice Ranges", time_missing.get("missing_slice_ranges")
        )

    def _print_mismatch_table(title_text: str, mismatches: Any) -> None:
        if not isinstance(mismatches, list) or not mismatches:
            return
        table = Table(
            title=title_text,
            title_style=_RICH_TITLE_STYLE,
            header_style=_RICH_HEADER_STYLE,
        )
        table.add_column("Point", style=_RICH_TEXT_STYLE, justify="left")
        table.add_column("Requested", style=_RICH_TEXT_STYLE, justify="left")
        table.add_column("Actual", style=_RICH_TEXT_STYLE, justify="left")
        table.add_column("Observed missing", style=_RICH_TEXT_STYLE, justify="left")
        table.add_column("Expected missing", style=_RICH_TEXT_STYLE, justify="left")
        for entry in mismatches:
            if not isinstance(entry, dict):
                continue
            table.add_row(
                _stringify(entry.get("point")),
                f"({_stringify(entry.get('requested_lat'))}, {_stringify(entry.get('requested_lon'))})",
                f"({_stringify(entry.get('actual_lat'))}, {_stringify(entry.get('actual_lon'))})",
                _stringify(entry.get("observed_missing")),
                _stringify(entry.get("expected_missing")),
            )
        console.print(table)

    _print_mismatch_table("Land Mismatches", offset.get("land_mismatches"))
    _print_mismatch_table("Ocean Mismatches", offset.get("ocean_mismatches"))

    if "note" in offset:
        note_table = Table(
            title="Notes",
            title_style=_RICH_TITLE_STYLE,
            header_style=_RICH_HEADER_STYLE,
        )
        note_table.add_column("Note", style=_RICH_TEXT_STYLE, justify="left")
        note_table.add_row(_stringify(offset.get("note")))
        console.print(note_table)


def _render_time_cover_report_with_rich(console: Any, report: dict[str, Any]) -> None:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    variable_name = report.get("variable")
    section_title = str(variable_name) if variable_name is not None else "Variable"
    title = Text(section_title, style=_RICH_TITLE_STYLE)
    console.print(Panel(title, border_style=_RICH_BORDER_STYLE))

    time_missing = (
        report.get("time_missing")
        if isinstance(report.get("time_missing"), dict)
        else {}
    )
    time_format = (
        report.get("time_format") if isinstance(report.get("time_format"), dict) else {}
    )

    summary_rows: list[tuple[str, Any, str]] = [
        (
            "time_missing",
            time_missing.get("status"),
            f"missing_slices={_stringify(time_missing.get('missing_slice_count', 0))}",
        )
    ]
    if time_format:
        summary_rows.append(
            (
                "time_format",
                time_format.get("status"),
                (
                    f"value_type={_stringify(time_format.get('value_type'))} "
                    f"units={_stringify(time_format.get('units'))}"
                ),
            )
        )
    failing_checks = sum(
        1 for _, status, _ in summary_rows if _status_kind(status) == "fail"
    )
    warning_checks = sum(
        1 for _, status, _ in summary_rows if _status_kind(status) == "warn"
    )

    summary_head = Table(show_header=False, box=None, pad_edge=False)
    summary_head.add_column("k", style=_RICH_LABEL_STYLE, justify="left")
    summary_head.add_column("v", style=_RICH_TEXT_STYLE, justify="left")
    summary_head.add_row(
        "Overall",
        Text(
            "PASSED" if bool(report.get("ok")) else "FAILED",
            style=_status_style("pass" if bool(report.get("ok")) else "fail"),
        ),
    )
    summary_head.add_row("Failing checks", _stringify(failing_checks))
    summary_head.add_row("Warnings/skips", _stringify(warning_checks))
    console.print(
        Panel(
            summary_head,
            title="Summary",
            border_style=_RICH_BORDER_STYLE,
            expand=False,
        )
    )

    meta = Table(show_header=False, box=None, pad_edge=False)
    meta.add_column("k", style=_RICH_LABEL_STYLE, justify="left")
    meta.add_column("v", style=_RICH_TEXT_STYLE, justify="left")
    meta.add_row("Variable", _stringify(report.get("variable")))
    meta.add_row("Time dim", _stringify(report.get("time_dim")))
    if time_format:
        meta.add_row("Time dtype", _stringify(time_format.get("dtype")))
        meta.add_row("Time units", _stringify(time_format.get("units")))
        meta.add_row(
            "Decoded by xarray", _stringify(time_format.get("decoded_by_xarray"))
        )
    console.print(meta)

    summary = Table(
        title="Check Summary",
        title_style=_RICH_TITLE_STYLE,
        header_style=_RICH_HEADER_STYLE,
        border_style=_RICH_TABLE_BORDER_STYLE,
    )
    summary.add_column("Check", style=_RICH_TEXT_STYLE, justify="left")
    summary.add_column("Status", style=_RICH_TEXT_STYLE, justify="left")
    summary.add_column("Detail", style=_RICH_TEXT_STYLE, justify="left")
    for check_name, status, detail in summary_rows:
        status_text = _stringify(status)
        summary.add_row(
            _stringify(check_name),
            Text(status_text, style=_status_style(status_text)),
            _stringify(detail),
        )
    console.print(summary)

    ranges = time_missing.get("missing_slice_ranges")
    if isinstance(ranges, list) and ranges:
        table = Table(
            title="Missing Time Slice Ranges",
            title_style=_RICH_TITLE_STYLE,
            header_style=_RICH_HEADER_STYLE,
        )
        table.add_column("Start", style=_RICH_TEXT_STYLE, justify="left")
        table.add_column("End", style=_RICH_TEXT_STYLE, justify="left")
        table.add_column("Start idx", style=_RICH_TEXT_STYLE, justify="left")
        table.add_column("End idx", style=_RICH_TEXT_STYLE, justify="left")
        for entry in ranges:
            if not isinstance(entry, dict):
                continue
            table.add_row(
                _stringify(entry.get("start")),
                _stringify(entry.get("end")),
                _stringify(entry.get("start_index")),
                _stringify(entry.get("end_index")),
            )
        console.print(table)

    if time_format:
        message = time_format.get("message")
        suggestion = time_format.get("suggestion")
        if message is not None or suggestion is not None:
            table = Table(
                title="Time Format Guidance",
                title_style=_RICH_TITLE_STYLE,
                header_style=_RICH_HEADER_STYLE,
            )
            table.add_column("Field", style=_RICH_TEXT_STYLE, justify="left")
            table.add_column("Value", style=_RICH_TEXT_STYLE, justify="left")
            if message is not None:
                table.add_row("message", _stringify(message))
            if suggestion is not None:
                table.add_row("suggestion", _stringify(suggestion))
            console.print(table)


def print_pretty_report(report: Any) -> None:
    """Print a human-readable compliance report with rich highlighting when available."""
    if not isinstance(report, dict):
        print(_stringify(report))
        return

    try:
        from rich.console import Console

        console = Console()
        _render_cf_report_with_rich(console, report)
    except Exception:
        print(to_yaml_like(report))


def print_pretty_ocean_report(report: Any) -> None:
    """Print an ocean coverage report with the same rich style as CF reports."""
    if not isinstance(report, dict):
        print(_stringify(report))
        return

    print_pretty_ocean_reports([report])


def print_pretty_ocean_reports(reports: list[dict[str, Any]]) -> None:
    """Print one or more ocean coverage reports under a shared top-level banner."""
    if not reports:
        return

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        console.print(
            Panel(
                Text("OCEAN COVER REPORT", style=_RICH_TITLE_STYLE),
                border_style=_RICH_BORDER_STYLE,
            )
        )
        for index, report in enumerate(reports):
            if index > 0:
                console.print()
            _render_ocean_report_with_rich(console, report)
    except Exception:
        for report in reports:
            print(to_yaml_like(report))


def print_pretty_time_cover_report(report: Any) -> None:
    """Print a time-coverage report with the same rich style as CF reports."""
    if not isinstance(report, dict):
        print(_stringify(report))
        return

    print_pretty_time_cover_reports([report])


def print_pretty_time_cover_reports(reports: list[dict[str, Any]]) -> None:
    """Print one or more time-coverage reports under a shared top-level banner."""
    if not reports:
        return

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        console.print(
            Panel(
                Text("TIME COVER REPORT", style=_RICH_TITLE_STYLE),
                border_style=_RICH_BORDER_STYLE,
            )
        )
        for index, report in enumerate(reports):
            if index > 0:
                console.print()
            _render_time_cover_report_with_rich(console, report)
    except Exception:
        for report in reports:
            print(to_yaml_like(report))


def print_pretty_full_report(report: Any) -> None:
    """Print a combined report summary, then each selected report."""
    if not isinstance(report, dict):
        print(_stringify(report))
        return

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = Console()
        title = Text("Full Dataset Check Report", style=_RICH_TITLE_STYLE)
        console.print(Panel(title, border_style=_RICH_BORDER_STYLE))

        summary = (
            report.get("summary") if isinstance(report.get("summary"), dict) else {}
        )
        summary_table = Table(show_header=False, box=None, pad_edge=False)
        summary_table.add_column("k", style=_RICH_LABEL_STYLE, justify="left")
        summary_table.add_column("v", style=_RICH_TEXT_STYLE, justify="left")
        summary_table.add_row(
            "Overall",
            Text(
                "PASSED" if bool(summary.get("overall_ok")) else "FAILED",
                style=_status_style(summary.get("overall_status")),
            ),
        )
        summary_table.add_row("Checks run", _stringify(summary.get("checks_run")))
        summary_table.add_row(
            "Failing checks", _stringify(summary.get("failing_checks"))
        )
        summary_table.add_row(
            "Warnings/skips", _stringify(summary.get("warnings_or_skips"))
        )
        console.print(
            Panel(summary_table, title="Summary", border_style=_RICH_BORDER_STYLE)
        )

        checks = report.get("check_summary")
        if isinstance(checks, list) and checks:
            checks_table = Table(
                title="Check Summary",
                title_style=_RICH_TITLE_STYLE,
                header_style=_RICH_HEADER_STYLE,
                border_style=_RICH_TABLE_BORDER_STYLE,
            )
            checks_table.add_column("Check", style=_RICH_TEXT_STYLE, justify="left")
            checks_table.add_column("Status", style=_RICH_TEXT_STYLE, justify="left")
            checks_table.add_column("Detail", style=_RICH_TEXT_STYLE, justify="left")
            for item in checks:
                if not isinstance(item, dict):
                    continue
                checks_table.add_row(
                    _stringify(item.get("check")),
                    Text(
                        _stringify(item.get("status")).upper(),
                        style=_status_style(item.get("status")),
                    ),
                    _stringify(item.get("detail")),
                )
            console.print(checks_table)

        reports = (
            report.get("reports") if isinstance(report.get("reports"), dict) else {}
        )
        compliance = reports.get("compliance")
        if isinstance(compliance, dict):
            _render_cf_report_with_rich(console, compliance)
        ocean_cover = reports.get("ocean_cover")
        if isinstance(ocean_cover, dict):
            _render_ocean_report_with_rich(console, ocean_cover)
        time_cover = reports.get("time_cover")
        if isinstance(time_cover, dict):
            _render_time_cover_report_with_rich(console, time_cover)
    except Exception:
        print(to_yaml_like(report))


def _html_status_badge(status: Any) -> str:
    if isinstance(status, bool):
        text = "PASSED" if status else "FAILED"
        badge_class = (
            "bg-success-subtle text-success-emphasis border border-success-subtle"
            if status
            else "bg-danger-subtle text-danger-emphasis border border-danger-subtle"
        )
        return f"<span class='badge report-badge rounded-pill {badge_class}'>{escape(text)}</span>"

    normalized = _stringify(status).strip().lower()
    if normalized in {"pass", "passed", "ok", "success", "true"}:
        text = "PASSED"
        badge_class = (
            "bg-success-subtle text-success-emphasis border border-success-subtle"
        )
    elif normalized in {"fail", "failed", "error", "fatal", "false"}:
        text = "FAILED"
        badge_class = (
            "bg-danger-subtle text-danger-emphasis border border-danger-subtle"
        )
    elif normalized in {"skip", "skipped"} or normalized.startswith("skip"):
        text = "SKIPPED"
        badge_class = (
            "bg-secondary-subtle text-secondary-emphasis border border-secondary-subtle"
        )
    else:
        text = "WARNING"
        badge_class = (
            "bg-warning-subtle text-warning-emphasis border border-warning-subtle"
        )
    return f"<span class='badge report-badge rounded-pill {badge_class}'>{escape(text)}</span>"


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    header_html = "".join(f"<th>{escape(col)}</th>" for col in headers)
    row_html = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return (
        "<div class='table-responsive'>"
        "<table class='table table-sm align-middle report-table'>"
        f"<thead><tr>{header_html}</tr></thead><tbody>{row_html}</tbody></table></div>"
    )


def _html_summary_table(rows: list[tuple[str, Any]]) -> str:
    kv_rows: list[str] = []
    for label, value in rows:
        label_text = escape(_stringify(label))
        key_norm = label.strip().lower()
        if isinstance(value, bool) or "status" in key_norm or key_norm.endswith(" ok"):
            value_html = _html_status_badge(value)
        elif isinstance(value, (dict, list)):
            value_html = f"<pre>{escape(to_yaml_like(value))}</pre>"
        else:
            value_html = escape(_stringify(value))
        kv_rows.append(
            "<div class='kv-row'>"
            f"<dt class='kv-label'>{label_text}</dt>"
            f"<dd class='kv-value'>{value_html}</dd>"
            "</div>"
        )
    return (
        "<section class='summary-table-wrap'>"
        "<dl class='summary-table summary-kv'>"
        + "".join(kv_rows)
        + "</dl>"
        + "</section>"
    )


def _html_details_section(
    title: str, body_html: str, *, open_by_default: bool = False
) -> str:
    open_attr = " open" if open_by_default else ""
    return (
        f"<details class='report-section'{open_attr}>"
        "<summary>"
        f"<span class='section-title'>{escape(title)}</span>"
        "</summary>"
        f"<div class='section-body'>{body_html}</div>"
        "</details>"
    )


def _html_static_section(title: str, body_html: str) -> str:
    return (
        "<section class='report-section static-section'>"
        f"<div class='section-header'><span class='section-title'>{escape(title)}</span></div>"
        f"<div class='section-body'>{body_html}</div>"
        "</section>"
    )


def _html_variable_section(
    variable_name: str,
    status: Any,
    body_html: str,
    *,
    open_by_default: bool = False,
) -> str:
    open_attr = " open" if open_by_default else ""
    return (
        f"<details class='report-section variable-report'{open_attr}>"
        "<summary>"
        f"<span class='section-title'>Variable: {escape(_stringify(variable_name))}</span>"
        f"<span class='summary-badge'>{_html_status_badge(status)}</span>"
        "</summary>"
        f"<div class='section-body'>{body_html}</div>"
        "</details>"
    )


def _html_severity_badge(severity: Any) -> str:
    text = _stringify(severity).upper()
    badge_class = (
        "bg-secondary-subtle text-secondary-emphasis border border-secondary-subtle"
    )
    if text == "FATAL":
        badge_class = "bg-danger text-white border border-danger"
    elif text == "ERROR":
        badge_class = (
            "bg-danger-subtle text-danger-emphasis border border-danger-subtle"
        )
    elif text == "WARN":
        badge_class = (
            "bg-warning-subtle text-warning-emphasis border border-warning-subtle"
        )
    return f"<span class='badge report-badge rounded-pill {badge_class}'>{escape(text)}</span>"


def _cf_suggestion_rows(report: dict[str, Any]) -> list[dict[str, str]]:
    suggestions = (report.get("suggestions") or {}).get("variables")
    if not isinstance(suggestions, dict):
        return []

    rows: list[dict[str, str]] = []
    for var_name, suggestion in suggestions.items():
        if isinstance(suggestion, (dict, list)):
            detail = to_yaml_like(suggestion)
        else:
            detail = _stringify(suggestion)
        rows.append(
            {
                "scope": _stringify(var_name),
                "convention": "suggestion",
                "severity": "INFO",
                "detail": detail,
            }
        )
    return rows


def _status_kind(status: Any) -> str:
    normalized = _stringify(status).strip().lower()
    if normalized in {"pass", "passed", "ok", "success", "true"}:
        return "pass"
    if normalized in {"fail", "failed", "error", "fatal", "false"}:
        return "fail"
    if normalized in {"skip", "skipped"} or normalized.startswith("skip"):
        return "skip"
    return "warn"


def _combine_status_kinds(statuses: list[Any], fallback: Any) -> str:
    kinds = [_status_kind(status) for status in statuses if status is not None]
    if any(kind == "fail" for kind in kinds):
        return "fail"
    if any(kind == "warn" for kind in kinds):
        return "warn"
    if any(kind == "pass" for kind in kinds):
        return "pass"
    if any(kind == "skip" for kind in kinds):
        return "skip"
    return _status_kind(fallback)


def _ocean_variable_status(report: dict[str, Any]) -> str:
    statuses: list[Any] = []
    for check_name in ("edge_of_map", "land_ocean_offset", "time_missing"):
        check = report.get(check_name)
        if isinstance(check, dict):
            statuses.append(check.get("status"))
    return _combine_status_kinds(statuses, report.get("ok"))


def _time_cover_variable_status(report: dict[str, Any]) -> str:
    statuses: list[Any] = []
    for check_name in ("time_missing", "time_format"):
        check = report.get(check_name)
        if isinstance(check, dict):
            statuses.append(check.get("status"))
    return _combine_status_kinds(statuses, report.get("ok"))


def _html_stat_strip(items: list[tuple[str, Any, str | None]]) -> str:
    cards: list[str] = []
    for label, value, status in items:
        classes = "stat-card"
        if status in {"fail", "warn"}:
            classes += f" status-{status}"
        cards.append(
            f"<article class='{classes}'>"
            f"<span class='stat-label'>{escape(_stringify(label))}</span>"
            f"<span class='stat-value'>{escape(_stringify(value))}</span>"
            "</article>"
        )
    return "<section class='stat-strip'>" + "".join(cards) + "</section>"


def _html_issue_cards(findings: list[dict[str, str]]) -> str:
    if not findings:
        return "<p class='issue-empty'>No findings reported.</p>"

    rows: list[str] = []
    for finding in findings:
        scope = _stringify(finding.get("scope", "n/a"))
        convention = _stringify(finding.get("convention", "n/a"))
        severity = _stringify(finding.get("severity", "info")).upper()
        detail = _stringify(finding.get("detail", ""))
        rows.append(
            "<div class='issue-row issue-card'>"
            "<div class='issue-cell issue-status'>"
            f"{_html_severity_badge(severity)}"
            "</div>"
            "<div class='issue-cell issue-test'>"
            f"<p class='issue-scope'>{escape(scope)}</p>"
            f"<p class='issue-convention'>Convention: {escape(convention)}</p>"
            "</div>"
            f"<p class='issue-cell issue-detail'>{escape(detail)}</p>"
            "</div>"
        )
    return (
        "<section class='issue-list'>"
        "<div class='issue-head'>"
        "<span class='issue-head-status'>Status</span>"
        "<span class='issue-head-test'>Test Info</span>"
        "<span class='issue-head-desc'>Description</span>"
        "</div>" + "".join(rows) + "</section>"
    )


def _html_check_summary_table(rows: list[tuple[str, Any, str]]) -> str:
    if not rows:
        return "<p class='issue-empty'>No checks were run.</p>"

    row_html = "".join(
        (
            "<div class='issue-row issue-card'>"
            "<div class='issue-cell issue-status'>"
            f"{_html_status_badge(status)}"
            "</div>"
            "<div class='issue-cell issue-test'>"
            f"<p class='issue-scope'>{escape(_stringify(check_name))}</p>"
            "</div>"
            f"<p class='issue-cell issue-detail'>{escape(_stringify(detail))}</p>"
            "</div>"
        )
        for check_name, status, detail in rows
    )

    return (
        "<section class='check-summary-block issue-list'>"
        "<div class='issue-head'>"
        "<span class='issue-head-status'>Status</span>"
        "<span class='issue-head-test'>Test Info</span>"
        "<span class='issue-head-desc'>Description</span>"
        "</div>"
        f"{row_html}"
        "</section>"
    )


def _cf_finding_rows(report: dict[str, Any], scope: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if scope == "global":
        for item in report.get("global", []) or []:
            if isinstance(item, dict):
                convention = _stringify(item.get("convention", "cf"))
                severity = _stringify(item.get("severity", "info"))
                detail = _stringify(item.get("message") or item.get("item") or item)
            else:
                convention = "n/a"
                severity = "info"
                detail = _stringify(item)
            rows.append(
                {
                    "scope": "global",
                    "convention": convention,
                    "severity": severity,
                    "detail": detail,
                }
            )
        return rows

    source = (
        report.get("coordinates") if scope == "coordinates" else report.get("variables")
    )
    if not isinstance(source, dict):
        return rows
    for entry_scope, items in source.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                convention = _stringify(item.get("convention", "cf"))
                severity = _stringify(item.get("severity", "info"))
                detail = _stringify(item.get("message") or item.get("item") or item)
            else:
                convention = "n/a"
                severity = "info"
                detail = _stringify(item)
            rows.append(
                {
                    "scope": _stringify(entry_scope),
                    "convention": convention,
                    "severity": severity,
                    "detail": detail,
                }
            )
    return rows


def _cf_report_sections(report: dict[str, Any]) -> str:
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    conventions = report.get("conventions_checked")
    conventions_text = (
        ", ".join(str(c) for c in conventions)
        if isinstance(conventions, list) and conventions
        else "n/a"
    )
    variable_findings = report.get("variables")
    variable_count = (
        len(variable_findings) if isinstance(variable_findings, dict) else 0
    )

    fatal_raw = counts.get("fatal", 0)
    error_raw = counts.get("error", 0)
    warn_raw = counts.get("warn", 0)
    fatal_count = fatal_raw if isinstance(fatal_raw, int) else 0
    error_count = error_raw if isinstance(error_raw, int) else 0
    warn_count = warn_raw if isinstance(warn_raw, int) else 0
    problem_count = fatal_count + error_count

    meta = _html_summary_table(
        [
            ("CF version", report.get("cf_version")),
            ("Engine", report.get("engine")),
            ("Engine status", report.get("engine_status")),
            ("Check method", report.get("check_method")),
            ("Conventions", conventions_text),
            (
                "Counts",
                " ".join(
                    [
                        f"fatal={_stringify(counts.get('fatal', 0))}",
                        f"error={_stringify(counts.get('error', 0))}",
                        f"warn={_stringify(counts.get('warn', 0))}",
                    ]
                ),
            ),
        ]
    )

    stats = _html_stat_strip(
        [
            ("Variables checked", variable_count, None),
            ("Errors + fatals", problem_count, "fail" if problem_count > 0 else None),
            ("Warnings", warn_count, "warn" if warn_count > 0 else None),
            (
                "Engine status",
                _stringify(report.get("engine_status")).upper(),
                _status_kind(report.get("engine_status")),
            ),
        ]
    )

    sections: list[str] = []
    section_defs = [
        ("Global Findings", _cf_finding_rows(report, "global")),
        ("Coordinate Findings", _cf_finding_rows(report, "coordinates")),
        ("Variable Findings", _cf_finding_rows(report, "variables")),
    ]
    for title, rows in section_defs:
        if not rows:
            continue
        sections.append(
            _html_static_section(
                title,
                _html_issue_cards(rows),
            )
        )

    suggestion_rows = _cf_suggestion_rows(report)
    if suggestion_rows:
        sections.append(
            _html_static_section(
                "Suggested Improvements",
                _html_issue_cards(suggestion_rows),
            )
        )

    notes = report.get("notes")
    if isinstance(notes, list) and notes:
        rows = [[escape(_stringify(note))] for note in notes]
        sections.append(
            _html_static_section(
                "Notes",
                _html_table(["Note"], rows),
            )
        )

    checker_error = report.get("checker_error")
    if checker_error is not None:
        sections.append(
            _html_static_section(
                "Checker Error",
                f"<pre class='mb-0'>{escape(_stringify(checker_error))}</pre>",
            )
        )

    if not sections:
        sections.append(
            _html_static_section(
                "Findings",
                "<p class='mb-0 issue-empty'>No findings were reported.</p>",
            )
        )

    return (
        stats
        + meta
        + "<section class='section-stack'>"
        + "".join(sections)
        + "</section>"
    )


def render_pretty_report_html(report: Any) -> str:
    if not isinstance(report, dict):
        return _yaml_html_fallback(report)
    intro = (
        "<p class='report-subtitle mb-0'>"
        f"Engine: {escape(_stringify(report.get('engine')))}"
        f" · Method: {escape(_stringify(report.get('check_method')))}"
        "</p>"
    )
    return render_report_document(
        "CF Compliance Report", intro, _cf_report_sections(report)
    )


def _ocean_report_sections(report: dict[str, Any]) -> str:
    grid = report.get("grid") if isinstance(report.get("grid"), dict) else {}
    edge = (
        report.get("edge_of_map") if isinstance(report.get("edge_of_map"), dict) else {}
    )
    if not edge and isinstance(report.get("edge_sliver"), dict):
        edge = report.get("edge_sliver")
    offset = (
        report.get("land_ocean_offset")
        if isinstance(report.get("land_ocean_offset"), dict)
        else {}
    )
    time_missing = (
        report.get("time_missing")
        if isinstance(report.get("time_missing"), dict)
        else {}
    )

    meta = _html_summary_table(
        [
            ("Variable", report.get("variable")),
            ("Overall OK", report.get("ok")),
            (
                "Longitude",
                f"{_stringify(grid.get('lon_name'))} ({_stringify(grid.get('lon_dim'))})",
            ),
            (
                "Latitude",
                f"{_stringify(grid.get('lat_name'))} ({_stringify(grid.get('lat_dim'))})",
            ),
            ("Time dim", grid.get("time_dim")),
            ("Lon convention", grid.get("longitude_convention")),
            (
                "Lon range",
                f"{_stringify(grid.get('longitude_min'))} .. {_stringify(grid.get('longitude_max'))}",
            ),
            (
                "Lat range",
                f"{_stringify(grid.get('latitude_min'))} .. {_stringify(grid.get('latitude_max'))}",
            ),
        ]
    )

    summary_rows: list[tuple[str, Any, str]] = [
        (
            "edge_of_map",
            edge.get("status"),
            f"missing_longitudes={_stringify(edge.get('missing_longitude_count', 0))}",
        ),
        (
            "land_ocean_offset",
            offset.get("status"),
            f"mismatches={_stringify(offset.get('mismatch_count', 0))}",
        ),
    ]
    if time_missing:
        summary_rows.append(
            (
                "time_missing",
                time_missing.get("status"),
                f"missing_slices={_stringify(time_missing.get('missing_slice_count', 0))}",
            )
        )

    failing_checks = sum(
        1 for _, status, _ in summary_rows if _status_kind(status) == "fail"
    )
    warning_checks = sum(
        1 for _, status, _ in summary_rows if _status_kind(status) == "warn"
    )
    stats = _html_stat_strip(
        [
            ("Checks run", len(summary_rows), None),
            ("Failing checks", failing_checks, "fail" if failing_checks > 0 else None),
            ("Warnings/skips", warning_checks, "warn" if warning_checks > 0 else None),
            (
                "Overall",
                "PASSED" if bool(report.get("ok")) else "FAILED",
                _status_kind(report.get("ok")),
            ),
        ]
    )

    sections: list[str] = [
        _html_static_section(
            "Check Summary",
            _html_check_summary_table(summary_rows),
        )
    ]

    missing_lon_ranges = edge.get("missing_longitude_ranges")
    if isinstance(missing_lon_ranges, list) and missing_lon_ranges:
        lon_rows = [
            [
                escape(_stringify(entry.get("start"))),
                escape(_stringify(entry.get("end"))),
            ]
            for entry in missing_lon_ranges
            if isinstance(entry, dict)
        ]
        sections.append(
            _html_static_section(
                "Edge Of Map Missing Longitude Ranges",
                _html_table(["Start", "End"], lon_rows),
            )
        )

    missing_time_ranges = time_missing.get("missing_slice_ranges")
    if isinstance(missing_time_ranges, list) and missing_time_ranges:
        time_rows = [
            [
                escape(_stringify(entry.get("start"))),
                escape(_stringify(entry.get("end"))),
                escape(_stringify(entry.get("start_index"))),
                escape(_stringify(entry.get("end_index"))),
            ]
            for entry in missing_time_ranges
            if isinstance(entry, dict)
        ]
        sections.append(
            _html_static_section(
                "Time Missing Slice Ranges",
                _html_table(["Start", "End", "Start idx", "End idx"], time_rows),
            )
        )

    for title, key in (
        ("Land Mismatches", "land_mismatches"),
        ("Ocean Mismatches", "ocean_mismatches"),
    ):
        mismatches = offset.get(key)
        if not isinstance(mismatches, list) or not mismatches:
            continue
        rows = [
            [
                escape(_stringify(entry.get("point"))),
                escape(
                    f"({_stringify(entry.get('requested_lat'))}, {_stringify(entry.get('requested_lon'))})"
                ),
                escape(
                    f"({_stringify(entry.get('actual_lat'))}, {_stringify(entry.get('actual_lon'))})"
                ),
                escape(_stringify(entry.get("observed_missing"))),
                escape(_stringify(entry.get("expected_missing"))),
            ]
            for entry in mismatches
            if isinstance(entry, dict)
        ]
        sections.append(
            _html_static_section(
                title,
                _html_table(
                    [
                        "Point",
                        "Requested",
                        "Actual",
                        "Observed missing",
                        "Expected missing",
                    ],
                    rows,
                ),
            )
        )

    note = offset.get("note")
    if note is not None:
        sections.append(
            _html_static_section(
                "Notes", f"<p class='mb-0'>{escape(_stringify(note))}</p>"
            )
        )

    return (
        stats
        + meta
        + "<section class='section-stack'>"
        + "".join(sections)
        + "</section>"
    )


def _time_cover_report_sections(report: dict[str, Any]) -> str:
    time_missing = (
        report.get("time_missing")
        if isinstance(report.get("time_missing"), dict)
        else {}
    )
    time_format = (
        report.get("time_format") if isinstance(report.get("time_format"), dict) else {}
    )

    meta_rows: list[tuple[str, Any]] = [
        ("Variable", report.get("variable")),
        ("Time dim", report.get("time_dim")),
        ("Overall OK", report.get("ok")),
    ]
    if time_format:
        meta_rows.extend(
            [
                ("Time dtype", time_format.get("dtype")),
                ("Time units", time_format.get("units")),
                ("Decoded by xarray", time_format.get("decoded_by_xarray")),
            ]
        )
    meta = _html_summary_table(meta_rows)

    summary_rows: list[tuple[str, Any, str]] = [
        (
            "time_missing",
            time_missing.get("status"),
            f"missing_slices={_stringify(time_missing.get('missing_slice_count', 0))}",
        )
    ]
    if time_format:
        summary_rows.append(
            (
                "time_format",
                time_format.get("status"),
                (
                    f"value_type={_stringify(time_format.get('value_type'))} "
                    f"units={_stringify(time_format.get('units'))}"
                ),
            )
        )
    failing_checks = sum(
        1 for _, status, _ in summary_rows if _status_kind(status) == "fail"
    )
    warning_checks = sum(
        1 for _, status, _ in summary_rows if _status_kind(status) == "warn"
    )
    stats = _html_stat_strip(
        [
            ("Checks run", len(summary_rows), None),
            (
                "Failing checks",
                failing_checks,
                "fail" if failing_checks > 0 else None,
            ),
            (
                "Warnings/skips",
                warning_checks,
                "warn" if warning_checks > 0 else None,
            ),
            (
                "Overall",
                "PASSED" if bool(report.get("ok")) else "FAILED",
                _status_kind(report.get("ok")),
            ),
        ]
    )
    sections = [
        _html_static_section(
            "Check Summary",
            _html_check_summary_table(summary_rows),
        )
    ]

    missing_ranges = time_missing.get("missing_slice_ranges")
    if isinstance(missing_ranges, list) and missing_ranges:
        rows = [
            [
                escape(_stringify(entry.get("start"))),
                escape(_stringify(entry.get("end"))),
                escape(_stringify(entry.get("start_index"))),
                escape(_stringify(entry.get("end_index"))),
            ]
            for entry in missing_ranges
            if isinstance(entry, dict)
        ]
        sections.append(
            _html_static_section(
                "Missing Time Slice Ranges",
                _html_table(["Start", "End", "Start idx", "End idx"], rows),
            )
        )

    guidance_rows = []
    if time_format:
        if time_format.get("message") is not None:
            guidance_rows.append(("message", time_format.get("message")))
        if time_format.get("suggestion") is not None:
            guidance_rows.append(("suggestion", time_format.get("suggestion")))
    if guidance_rows:
        sections.append(
            _html_static_section(
                "Time Format Guidance",
                _html_summary_table(guidance_rows),
            )
        )

    return (
        stats
        + meta
        + "<section class='section-stack'>"
        + "".join(sections)
        + "</section>"
    )


def render_pretty_ocean_report_html(report: Any) -> str:
    if not isinstance(report, dict):
        return _yaml_html_fallback(report)
    intro = f"<p class='report-subtitle mb-0'>Variable: {escape(_stringify(report.get('variable')))}</p>"
    return render_report_document(
        "Ocean Coverage Report", intro, _ocean_report_sections(report)
    )


def render_pretty_ocean_reports_html(reports: list[dict[str, Any]]) -> str:
    if not reports:
        return _yaml_html_fallback({})
    blocks: list[str] = []
    for report in reports:
        blocks.append(
            _html_variable_section(
                variable_name=_stringify(report.get("variable")),
                status=_ocean_variable_status(report),
                body_html=_ocean_report_sections(report),
                open_by_default=False,
            )
        )
    intro = "<p class='report-subtitle mb-0'>Variables checked across all eligible variables.</p>"
    return render_report_document(
        "Ocean Coverage Report",
        intro,
        _ocean_reports_summary_sections(reports) + "".join(blocks),
    )


def _ocean_reports_summary_sections(reports: list[dict[str, Any]]) -> str:
    summary_rows: list[tuple[str, Any, str]] = []
    for report in reports:
        edge = (
            report.get("edge_of_map")
            if isinstance(report.get("edge_of_map"), dict)
            else {}
        )
        offset = (
            report.get("land_ocean_offset")
            if isinstance(report.get("land_ocean_offset"), dict)
            else {}
        )
        time_missing = (
            report.get("time_missing")
            if isinstance(report.get("time_missing"), dict)
            else {}
        )
        detail_parts = [
            f"missing_longitudes={_stringify(edge.get('missing_longitude_count', 0))}",
            f"mismatches={_stringify(offset.get('mismatch_count', 0))}",
        ]
        if time_missing:
            detail_parts.append(
                f"missing_slices={_stringify(time_missing.get('missing_slice_count', 0))}"
            )
        summary_rows.append(
            (
                _stringify(report.get("variable")),
                report.get("ok"),
                " ".join(detail_parts),
            )
        )

    failing_checks = sum(
        1 for _, status, _ in summary_rows if _status_kind(status) == "fail"
    )
    warning_checks = sum(
        1 for _, status, _ in summary_rows if _status_kind(status) == "warn"
    )
    overall_ok = all(bool(report.get("ok")) for report in reports)

    stats = _html_stat_strip(
        [
            ("Variables checked", len(reports), None),
            (
                "Failing variables",
                failing_checks,
                "fail" if failing_checks > 0 else None,
            ),
            ("Warnings/skips", warning_checks, "warn" if warning_checks > 0 else None),
            (
                "Overall",
                "PASSED" if overall_ok else "FAILED",
                _status_kind(overall_ok),
            ),
        ]
    )
    meta = _html_summary_table(
        [
            ("Variables checked", len(reports)),
            ("Overall OK", overall_ok),
        ]
    )
    summary = _html_static_section(
        "Top Summary",
        _html_check_summary_table(summary_rows),
    )
    return stats + meta + summary


def render_pretty_time_cover_report_html(report: Any) -> str:
    if not isinstance(report, dict):
        return _yaml_html_fallback(report)
    intro = f"<p class='report-subtitle mb-0'>Variable: {escape(_stringify(report.get('variable')))}</p>"
    return render_report_document(
        "Time Coverage Report", intro, _time_cover_report_sections(report)
    )


def _time_cover_reports_summary_sections(reports: list[dict[str, Any]]) -> str:
    summary_rows: list[tuple[str, Any, str]] = []
    for report in reports:
        time_missing = (
            report.get("time_missing")
            if isinstance(report.get("time_missing"), dict)
            else {}
        )
        time_format = (
            report.get("time_format")
            if isinstance(report.get("time_format"), dict)
            else {}
        )
        detail_parts = [
            f"missing_slices={_stringify(time_missing.get('missing_slice_count', 0))}"
        ]
        if time_format:
            detail_parts.append(f"time_format={_stringify(time_format.get('status'))}")
        summary_rows.append(
            (
                _stringify(report.get("variable")),
                report.get("ok"),
                " ".join(detail_parts),
            )
        )

    failing_checks = sum(
        1 for _, status, _ in summary_rows if _status_kind(status) == "fail"
    )
    warning_checks = sum(
        1 for _, status, _ in summary_rows if _status_kind(status) == "warn"
    )
    overall_ok = all(bool(report.get("ok")) for report in reports)

    stats = _html_stat_strip(
        [
            ("Variables checked", len(reports), None),
            (
                "Failing variables",
                failing_checks,
                "fail" if failing_checks > 0 else None,
            ),
            ("Warnings/skips", warning_checks, "warn" if warning_checks > 0 else None),
            (
                "Overall",
                "PASSED" if overall_ok else "FAILED",
                _status_kind(overall_ok),
            ),
        ]
    )
    meta = _html_summary_table(
        [
            ("Variables checked", len(reports)),
            ("Overall OK", overall_ok),
        ]
    )
    summary = _html_static_section(
        "Top Summary",
        _html_check_summary_table(summary_rows),
    )
    return stats + meta + summary


def render_pretty_time_cover_reports_html(reports: list[dict[str, Any]]) -> str:
    if not reports:
        return _yaml_html_fallback({})
    blocks: list[str] = []
    for report in reports:
        blocks.append(
            _html_variable_section(
                variable_name=_stringify(report.get("variable")),
                status=_time_cover_variable_status(report),
                body_html=_time_cover_report_sections(report),
                open_by_default=False,
            )
        )
    intro = "<p class='report-subtitle mb-0'>Variables checked across all data variables.</p>"
    return render_report_document(
        "Time Coverage Report",
        intro,
        _time_cover_reports_summary_sections(reports) + "".join(blocks),
    )


def _multi_variable_ocean_body(report: dict[str, Any]) -> str:
    reports = report.get("reports")
    if not isinstance(reports, dict):
        return _ocean_report_sections(report)
    report_list = [per_var for per_var in reports.values() if isinstance(per_var, dict)]
    if not report_list:
        return _ocean_report_sections(report)
    intro = _ocean_reports_summary_sections(report_list)
    blocks: list[str] = []
    for per_var in reports.values():
        if not isinstance(per_var, dict):
            continue
        blocks.append(
            _html_variable_section(
                variable_name=_stringify(per_var.get("variable")),
                status=_ocean_variable_status(per_var),
                body_html=_ocean_report_sections(per_var),
                open_by_default=False,
            )
        )
    return intro + "".join(blocks)


def _multi_variable_time_cover_body(report: dict[str, Any]) -> str:
    reports = report.get("reports")
    if not isinstance(reports, dict):
        return _time_cover_report_sections(report)
    report_list = [per_var for per_var in reports.values() if isinstance(per_var, dict)]
    if not report_list:
        return _time_cover_report_sections(report)
    intro = _time_cover_reports_summary_sections(report_list)
    blocks: list[str] = []
    for per_var in reports.values():
        if not isinstance(per_var, dict):
            continue
        blocks.append(
            _html_variable_section(
                variable_name=_stringify(per_var.get("variable")),
                status=_time_cover_variable_status(per_var),
                body_html=_time_cover_report_sections(per_var),
                open_by_default=False,
            )
        )
    return intro + "".join(blocks)


def _full_report_sections(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    check_summary = report.get("check_summary")
    summary_rows: list[tuple[str, Any, str]] = []
    if isinstance(check_summary, list):
        for item in check_summary:
            if not isinstance(item, dict):
                continue
            summary_rows.append(
                (
                    _stringify(item.get("check")),
                    item.get("status"),
                    _stringify(item.get("detail")),
                )
            )

    enabled = report.get("checks_enabled")
    enabled_checks = (
        ", ".join(
            check_name for check_name, is_enabled in enabled.items() if bool(is_enabled)
        )
        if isinstance(enabled, dict)
        else ""
    )

    stats = _html_stat_strip(
        [
            ("Checks run", summary.get("checks_run"), None),
            (
                "Failing checks",
                summary.get("failing_checks"),
                "fail" if _count_to_int(summary.get("failing_checks")) > 0 else None,
            ),
            (
                "Warnings/skips",
                summary.get("warnings_or_skips"),
                "warn" if _count_to_int(summary.get("warnings_or_skips")) > 0 else None,
            ),
            (
                "Overall",
                "PASSED" if bool(summary.get("overall_ok")) else "FAILED",
                _status_kind(summary.get("overall_status")),
            ),
        ]
    )
    meta = _html_summary_table(
        [
            ("Overall OK", summary.get("overall_ok")),
            ("Enabled checks", enabled_checks or "none"),
        ]
    )

    sections: list[str] = [
        _html_static_section(
            "Combined Check Summary",
            _html_check_summary_table(summary_rows),
        )
    ]

    reports = report.get("reports")
    if isinstance(reports, dict):
        compliance = reports.get("compliance")
        if isinstance(compliance, dict):
            sections.append(
                _html_details_section(
                    "CF Compliance",
                    _cf_report_sections(compliance),
                    open_by_default=False,
                )
            )
        ocean_cover = reports.get("ocean_cover")
        if isinstance(ocean_cover, dict):
            ocean_body = (
                _multi_variable_ocean_body(ocean_cover)
                if ocean_cover.get("mode") == "all_variables"
                else _ocean_report_sections(ocean_cover)
            )
            sections.append(
                _html_details_section(
                    "Ocean Coverage",
                    ocean_body,
                    open_by_default=False,
                )
            )
        time_cover = reports.get("time_cover")
        if isinstance(time_cover, dict):
            time_body = (
                _multi_variable_time_cover_body(time_cover)
                if time_cover.get("mode") == "all_variables"
                else _time_cover_report_sections(time_cover)
            )
            sections.append(
                _html_details_section(
                    "Time Coverage",
                    time_body,
                    open_by_default=False,
                )
            )

    return (
        stats
        + meta
        + "<section class='section-stack'>"
        + "".join(sections)
        + "</section>"
    )


def render_pretty_full_report_html(report: Any) -> str:
    if not isinstance(report, dict):
        return _yaml_html_fallback(report)

    enabled = report.get("checks_enabled")
    enabled_checks = (
        ", ".join(
            check_name for check_name, is_enabled in enabled.items() if bool(is_enabled)
        )
        if isinstance(enabled, dict)
        else ""
    )
    intro = (
        "<p class='report-subtitle mb-0'>"
        f"Selected checks: {escape(enabled_checks or 'none')}"
        "</p>"
    )
    return render_report_document(
        "Full Dataset Check Report",
        intro,
        _full_report_sections(report),
    )
