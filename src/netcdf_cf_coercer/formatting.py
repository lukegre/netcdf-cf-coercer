from __future__ import annotations

from typing import Any


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


def print_pretty_report(report: Any) -> None:
    """Print a human-readable compliance report with rich highlighting when available."""
    if not isinstance(report, dict):
        print(_stringify(report))
        return

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = Console()

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
    except Exception:
        print(to_yaml_like(report))
