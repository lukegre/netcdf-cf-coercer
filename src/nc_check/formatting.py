from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any
from typing import Literal

from .report_templates import render_report_document

ReportFormat = Literal["python", "tables", "html"]


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
        return "bold #e04c41"
    if sev == "WARN":
        return "dark_orange3"
    return "black"


def _status_style(status: str | None) -> str:
    normalized = (status or "").lower()
    if normalized in {"fail", "error"}:
        return "bold #e04c41"
    if normalized.startswith("skip"):
        return "dark_orange3"
    if normalized == "pass":
        return "green4"
    return "black"


def normalize_report_format(report_format: str) -> ReportFormat:
    normalized = report_format.strip().lower()
    allowed = {"python", "tables", "html"}
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

    title = Text("CF Compliance Report", style="bold black")
    console.print(Panel(title, border_style="#cccccc"))

    meta = Table(show_header=False, box=None, pad_edge=False)
    meta.add_column("k", style="bold black")
    meta.add_column("v", style="black")
    meta.add_row("CF version", _stringify(report.get("cf_version")))
    meta.add_row("Engine", _stringify(report.get("engine")))
    meta.add_row("Engine status", _stringify(report.get("engine_status")))
    meta.add_row("Check method", _stringify(report.get("check_method")))
    conventions_checked = report.get("conventions_checked")
    if isinstance(conventions_checked, list) and conventions_checked:
        meta.add_row("Conventions", ", ".join(str(c) for c in conventions_checked))
    counts = report.get("counts") or {}
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
        table = Table(
            title=title_text, title_style="bold black", header_style="bold black"
        )
        table.add_column("Scope", style="black")
        table.add_column("Convention", style="black")
        table.add_column("Severity", style="black")
        table.add_column("Detail", style="black")
        for scope, convention, severity, detail in rows:
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
            title_style="bold black",
            header_style="bold black",
        )
        suggestion_table.add_column("Variable", style="black")
        suggestion_table.add_column("Suggestion", style="black")
        for var_name, suggestion in suggestions.items():
            suggestion_table.add_row(str(var_name), _stringify(suggestion))
        console.print(suggestion_table)

    notes = report.get("notes") or []
    if isinstance(notes, list) and notes:
        note_table = Table(
            title="Notes", title_style="bold black", header_style="bold black"
        )
        note_table.add_column("Note", style="black")
        for note in notes:
            note_table.add_row(_stringify(note))
        console.print(note_table)

    checker_error = report.get("checker_error")
    if checker_error:
        error_text = _stringify(checker_error)
        console.print(Panel(error_text, title="Checker Error", border_style="red"))


def _render_ocean_report_with_rich(console: Any, report: dict[str, Any]) -> None:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    title = Text("Ocean Coverage Report", style="bold black")
    console.print(Panel(title, border_style="#cccccc"))

    grid = report.get("grid") if isinstance(report.get("grid"), dict) else {}
    meta = Table(show_header=False, box=None, pad_edge=False)
    meta.add_column("k", style="bold black")
    meta.add_column("v", style="black")
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
    meta.add_row("Overall OK", _stringify(report.get("ok")))
    console.print(meta)

    summary = Table(
        title="Check Summary",
        title_style="bold black",
        header_style="bold black",
    )
    summary.add_column("Check", style="black")
    summary.add_column("Status", style="black")
    summary.add_column("Detail", style="black")

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

    edge_status = _stringify(edge.get("status"))
    summary.add_row(
        "edge_of_map",
        Text(edge_status, style=_status_style(edge_status)),
        f"missing_longitudes={_stringify(edge.get('missing_longitude_count', 0))}",
    )
    offset_status = _stringify(offset.get("status"))
    summary.add_row(
        "land_ocean_offset",
        Text(offset_status, style=_status_style(offset_status)),
        f"mismatches={_stringify(offset.get('mismatch_count', 0))}",
    )
    if time_missing:
        time_status = _stringify(time_missing.get("status"))
        summary.add_row(
            "time_missing",
            Text(time_status, style=_status_style(time_status)),
            f"missing_slices={_stringify(time_missing.get('missing_slice_count', 0))}",
        )
    console.print(summary)

    def _print_ranges(title_text: str, ranges: Any) -> None:
        if not isinstance(ranges, list) or not ranges:
            return
        table = Table(
            title=title_text, title_style="bold black", header_style="bold black"
        )
        table.add_column("Start", style="black")
        table.add_column("End", style="black")
        table.add_column("Start idx", style="black")
        table.add_column("End idx", style="black")
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
            title=title_text, title_style="bold black", header_style="bold black"
        )
        table.add_column("Start", style="black")
        table.add_column("End", style="black")
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
            title=title_text, title_style="bold black", header_style="bold black"
        )
        table.add_column("Point", style="black")
        table.add_column("Requested", style="black")
        table.add_column("Actual", style="black")
        table.add_column("Observed missing", style="black")
        table.add_column("Expected missing", style="black")
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
            title="Notes", title_style="bold black", header_style="bold black"
        )
        note_table.add_column("Note", style="black")
        note_table.add_row(_stringify(offset.get("note")))
        console.print(note_table)


def _render_time_cover_report_with_rich(console: Any, report: dict[str, Any]) -> None:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    title = Text("Time Coverage Report", style="bold black")
    console.print(Panel(title, border_style="#cccccc"))

    meta = Table(show_header=False, box=None, pad_edge=False)
    meta.add_column("k", style="bold black")
    meta.add_column("v", style="black")
    meta.add_row("Variable", _stringify(report.get("variable")))
    meta.add_row("Time dim", _stringify(report.get("time_dim")))
    meta.add_row("Overall OK", _stringify(report.get("ok")))
    console.print(meta)

    time_missing = (
        report.get("time_missing")
        if isinstance(report.get("time_missing"), dict)
        else {}
    )
    summary = Table(
        title="Check Summary",
        title_style="bold black",
        header_style="bold black",
    )
    summary.add_column("Check", style="black")
    summary.add_column("Status", style="black")
    summary.add_column("Detail", style="black")
    status = _stringify(time_missing.get("status"))
    summary.add_row(
        "time_missing",
        Text(status, style=_status_style(status)),
        f"missing_slices={_stringify(time_missing.get('missing_slice_count', 0))}",
    )
    console.print(summary)

    ranges = time_missing.get("missing_slice_ranges")
    if isinstance(ranges, list) and ranges:
        table = Table(
            title="Missing Time Slice Ranges",
            title_style="bold black",
            header_style="bold black",
        )
        table.add_column("Start", style="black")
        table.add_column("End", style="black")
        table.add_column("Start idx", style="black")
        table.add_column("End idx", style="black")
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

    try:
        from rich.console import Console

        console = Console()
        _render_ocean_report_with_rich(console, report)
    except Exception:
        print(to_yaml_like(report))


def print_pretty_time_cover_report(report: Any) -> None:
    """Print a time-coverage report with the same rich style as CF reports."""
    if not isinstance(report, dict):
        print(_stringify(report))
        return

    try:
        from rich.console import Console

        console = Console()
        _render_time_cover_report_with_rich(console, report)
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
        return f"<span class='badge rounded-pill {badge_class}'>{escape(text)}</span>"

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
    else:
        text = "WARNING"
        badge_class = (
            "bg-warning-subtle text-warning-emphasis border border-warning-subtle"
        )
    return f"<span class='badge rounded-pill {badge_class}'>{escape(text)}</span>"


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
    table_rows: list[list[str]] = []
    for label, value in rows:
        label_text = escape(label)
        key_norm = label.strip().lower()
        if isinstance(value, bool) or "status" in key_norm or key_norm.endswith(" ok"):
            value_html = _html_status_badge(value)
        else:
            value_html = escape(_stringify(value))
        table_rows.append([label_text, value_html])
    return (
        "<section class='summary-table-wrap'>"
        + _html_table(["Field", "Value"], table_rows).replace(
            "report-table", "report-table summary-table"
        )
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
    return f"<span class='badge rounded-pill {badge_class}'>{escape(text)}</span>"


def _html_cell_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return f"<pre>{escape(to_yaml_like(value))}</pre>"
    return escape(_stringify(value))


def _cf_finding_rows(report: dict[str, Any], scope: str) -> list[list[str]]:
    rows: list[list[str]] = []
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
                [
                    escape("global"),
                    escape(convention),
                    _html_severity_badge(severity),
                    escape(detail),
                ]
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
                [
                    escape(_stringify(entry_scope)),
                    escape(convention),
                    _html_severity_badge(severity),
                    escape(detail),
                ]
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

    sections: list[str] = []
    section_defs = [
        ("Global Findings", _cf_finding_rows(report, "global"), True),
        ("Coordinate Findings", _cf_finding_rows(report, "coordinates"), False),
        ("Variable Findings", _cf_finding_rows(report, "variables"), False),
    ]
    for title, rows, open_by_default in section_defs:
        if not rows:
            continue
        sections.append(
            _html_details_section(
                title,
                _html_table(["Scope", "Convention", "Severity", "Detail"], rows),
                open_by_default=open_by_default,
            )
        )

    suggestions = (report.get("suggestions") or {}).get("variables")
    if isinstance(suggestions, dict) and suggestions:
        rows = [
            [escape(_stringify(var_name)), _html_cell_value(suggestion)]
            for var_name, suggestion in suggestions.items()
        ]
        sections.append(
            _html_details_section(
                "Suggested Improvements",
                _html_table(["Variable", "Suggestion"], rows),
            )
        )

    notes = report.get("notes")
    if isinstance(notes, list) and notes:
        rows = [[escape(_stringify(note))] for note in notes]
        sections.append(
            _html_details_section(
                "Notes",
                _html_table(["Note"], rows),
            )
        )

    checker_error = report.get("checker_error")
    if checker_error is not None:
        sections.append(
            _html_details_section(
                "Checker Error",
                f"<pre class='mb-0'>{escape(_stringify(checker_error))}</pre>",
            )
        )

    if not sections:
        sections.append(
            _html_details_section(
                "Findings",
                "<p class='mb-0'>No findings were reported.</p>",
                open_by_default=True,
            )
        )

    return meta + "<section class='section-stack'>" + "".join(sections) + "</section>"


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

    summary_rows = [
        [
            escape("edge_of_map"),
            _html_status_badge(edge.get("status")),
            escape(
                f"missing_longitudes={_stringify(edge.get('missing_longitude_count', 0))}"
            ),
        ],
        [
            escape("land_ocean_offset"),
            _html_status_badge(offset.get("status")),
            escape(f"mismatches={_stringify(offset.get('mismatch_count', 0))}"),
        ],
    ]
    if time_missing:
        summary_rows.append(
            [
                escape("time_missing"),
                _html_status_badge(time_missing.get("status")),
                escape(
                    f"missing_slices={_stringify(time_missing.get('missing_slice_count', 0))}"
                ),
            ]
        )

    sections: list[str] = [
        _html_static_section(
            "Check Summary",
            _html_table(["Check", "Status", "Detail"], summary_rows),
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
            _html_details_section(
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
            _html_details_section(
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
            _html_details_section(
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
            _html_details_section(
                "Notes", f"<p class='mb-0'>{escape(_stringify(note))}</p>"
            )
        )

    return meta + "<section class='section-stack'>" + "".join(sections) + "</section>"


def _time_cover_report_sections(report: dict[str, Any]) -> str:
    time_missing = (
        report.get("time_missing")
        if isinstance(report.get("time_missing"), dict)
        else {}
    )

    meta = _html_summary_table(
        [
            ("Variable", report.get("variable")),
            ("Time dim", report.get("time_dim")),
            ("Overall OK", report.get("ok")),
        ]
    )

    summary_rows = [
        [
            escape("time_missing"),
            _html_status_badge(time_missing.get("status")),
            escape(
                f"missing_slices={_stringify(time_missing.get('missing_slice_count', 0))}"
            ),
        ]
    ]
    sections = [
        _html_static_section(
            "Check Summary",
            _html_table(["Check", "Status", "Detail"], summary_rows),
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
            _html_details_section(
                "Missing Time Slice Ranges",
                _html_table(["Start", "End", "Start idx", "End idx"], rows),
            )
        )

    return meta + "<section class='section-stack'>" + "".join(sections) + "</section>"


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
    for index, report in enumerate(reports, start=1):
        blocks.append(
            _html_variable_section(
                variable_name=_stringify(report.get("variable")),
                status=report.get("ok"),
                body_html=_ocean_report_sections(report),
                open_by_default=(index == 1),
            )
        )
    intro = f"<p class='report-subtitle mb-0'>Variables checked: {len(reports)}</p>"
    return render_report_document("Ocean Coverage Report", intro, "".join(blocks))


def render_pretty_time_cover_report_html(report: Any) -> str:
    if not isinstance(report, dict):
        return _yaml_html_fallback(report)
    intro = f"<p class='report-subtitle mb-0'>Variable: {escape(_stringify(report.get('variable')))}</p>"
    return render_report_document(
        "Time Coverage Report", intro, _time_cover_report_sections(report)
    )


def render_pretty_time_cover_reports_html(reports: list[dict[str, Any]]) -> str:
    if not reports:
        return _yaml_html_fallback({})
    blocks: list[str] = []
    for index, report in enumerate(reports, start=1):
        blocks.append(
            _html_variable_section(
                variable_name=_stringify(report.get("variable")),
                status=report.get("ok"),
                body_html=_time_cover_report_sections(report),
                open_by_default=(index == 1),
            )
        )
    intro = f"<p class='report-subtitle mb-0'>Variables checked: {len(reports)}</p>"
    return render_report_document("Time Coverage Report", intro, "".join(blocks))
