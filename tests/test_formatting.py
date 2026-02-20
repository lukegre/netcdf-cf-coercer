import builtins

from nc_check import formatting


def test_to_yaml_like_handles_scalars_and_empty_collections() -> None:
    text = formatting.to_yaml_like(
        {
            "none": None,
            "flag": True,
            "items": [],
            "obj": {},
        }
    )

    assert "none: null" in text
    assert "flag: true" in text
    assert "items:\n  []" in text
    assert "obj:\n  {}" in text


def test_normalize_report_format_auto_uses_html_in_notebook(monkeypatch) -> None:
    monkeypatch.setattr(formatting, "_running_in_notebook", lambda: True)
    monkeypatch.setattr(formatting, "_running_in_cli", lambda: False)

    assert formatting.normalize_report_format("auto") == "html"


def test_normalize_report_format_auto_uses_tables_in_cli(monkeypatch) -> None:
    monkeypatch.setattr(formatting, "_running_in_notebook", lambda: False)
    monkeypatch.setattr(formatting, "_running_in_cli", lambda: True)

    assert formatting.normalize_report_format("auto") == "tables"


def test_normalize_report_format_auto_uses_python_otherwise(monkeypatch) -> None:
    monkeypatch.setattr(formatting, "_running_in_notebook", lambda: False)
    monkeypatch.setattr(formatting, "_running_in_cli", lambda: False)

    assert formatting.normalize_report_format("auto") == "python"


def test_print_pretty_report_handles_non_dict(capsys) -> None:
    formatting.print_pretty_report("plain text")
    out = capsys.readouterr().out
    assert "plain text" in out


def test_print_pretty_report_falls_back_without_rich(monkeypatch, capsys) -> None:
    original_import = builtins.__import__

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("rich"):
            raise ImportError("forced import failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _patched_import)

    formatting.print_pretty_report(
        {
            "cf_version": "CF-1.12",
            "engine_status": "ok",
            "global": [{"severity": "WARN", "message": "example"}],
        }
    )
    out = capsys.readouterr().out

    assert "cf_version: CF-1.12" in out
    assert "engine_status: ok" in out


def test_print_pretty_report_shows_checked_conventions(capsys) -> None:
    formatting.print_pretty_report(
        {
            "cf_version": "CF-1.12",
            "engine": "cfchecker",
            "engine_status": "ok",
            "check_method": "cfchecker",
            "conventions_checked": ["cf", "ferret"],
            "coordinates": {
                "time": [
                    {
                        "convention": "ferret",
                        "severity": "ERROR",
                        "message": "Coordinate 'time' has forbidden _FillValue.",
                    }
                ]
            },
        }
    )
    out = capsys.readouterr().out

    assert "Conventions" in out
    assert "ferret" in out


def test_print_pretty_ocean_report_shows_summary(capsys) -> None:
    formatting.print_pretty_ocean_report(
        {
            "variable": "sst",
            "ok": False,
            "grid": {
                "lon_name": "lon",
                "lon_dim": "lon",
                "lat_name": "lat",
                "lat_dim": "lat",
                "time_dim": "time",
                "longitude_convention": "0_360",
                "longitude_min": 0.0,
                "longitude_max": 359.0,
                "latitude_min": -89.0,
                "latitude_max": 89.0,
            },
            "edge_of_map": {
                "status": "fail",
                "missing_longitude_count": 2,
                "missing_slice_count": 3,
                "missing_longitude_ranges": [{"start": "0.0", "end": "2.0"}],
                "missing_slice_ranges": [],
            },
            "land_ocean_offset": {
                "status": "pass",
                "mismatch_count": 0,
                "land_mismatches": [],
                "ocean_mismatches": [],
            },
            "time_missing": {
                "status": "pass",
                "missing_slice_count": 0,
                "missing_slice_ranges": [],
            },
        }
    )
    out = capsys.readouterr().out

    assert "OCEAN COVER REPORT" in out
    assert "sst" in out
    assert "Check Summary" in out
    assert "edge_of_map" in out
    assert "Longitude" in out


def test_print_pretty_time_cover_report_shows_summary(capsys) -> None:
    formatting.print_pretty_time_cover_report(
        {
            "variable": "sst",
            "time_dim": "time",
            "ok": False,
            "time_missing": {
                "status": "fail",
                "missing_slice_count": 2,
                "missing_slice_ranges": [
                    {"start": "1", "end": "2", "start_index": 1, "end_index": 2}
                ],
            },
        }
    )
    out = capsys.readouterr().out

    assert "TIME COVER REPORT" in out
    assert "sst" in out
    assert "time_missing" in out
    assert "Missing Time Slice Ranges" in out


def test_print_pretty_ocean_reports_prints_banner_once(capsys) -> None:
    formatting.print_pretty_ocean_reports(
        [
            {
                "variable": "sst",
                "ok": True,
                "grid": {},
                "edge_of_map": {"status": "pass", "missing_longitude_count": 0},
                "land_ocean_offset": {"status": "pass", "mismatch_count": 0},
            },
            {
                "variable": "sss",
                "ok": True,
                "grid": {},
                "edge_of_map": {"status": "pass", "missing_longitude_count": 0},
                "land_ocean_offset": {"status": "pass", "mismatch_count": 0},
            },
        ]
    )
    out = capsys.readouterr().out

    assert out.count("OCEAN COVER REPORT") == 1
    assert "sst" in out
    assert "sss" in out


def test_print_pretty_report_summary_and_severity_priority(capsys) -> None:
    formatting.print_pretty_report(
        {
            "cf_version": "CF-1.12",
            "engine": "cfchecker",
            "engine_status": "ok",
            "check_method": "cfchecker",
            "counts": {"fatal": 0, "error": 1, "warn": 1},
            "global": [
                {"severity": "WARN", "message": "warn-detail"},
                {"severity": "ERROR", "message": "error-detail"},
            ],
        }
    )
    out = capsys.readouterr().out

    assert "Summary" in out
    assert out.find("Errors + fatals") < out.find("CF version")
    assert out.find("error-detail") < out.find("warn-detail")


def test_print_pretty_ocean_report_sorts_failing_checks_first(capsys) -> None:
    formatting.print_pretty_ocean_report(
        {
            "variable": "sst",
            "ok": False,
            "grid": {
                "lon_name": "lon",
                "lon_dim": "lon",
                "lat_name": "lat",
                "lat_dim": "lat",
                "time_dim": "time",
                "longitude_convention": "0_360",
                "longitude_min": 0.0,
                "longitude_max": 359.0,
                "latitude_min": -89.0,
                "latitude_max": 89.0,
            },
            "edge_of_map": {"status": "pass", "missing_longitude_count": 0},
            "land_ocean_offset": {"status": "fail", "mismatch_count": 3},
            "time_missing": {"status": "skipped", "missing_slice_count": 0},
        }
    )
    out = capsys.readouterr().out

    assert "Summary" in out
    assert out.find("Failing checks") < out.find("Variable")
    assert out.find("land_ocean_offset") < out.find("time_missing")
    assert out.find("time_missing") < out.find("edge_of_map")


def test_render_pretty_report_html_uses_flat_section_layout() -> None:
    html = formatting.render_pretty_report_html(
        {
            "cf_version": "CF-1.12",
            "engine": "cfchecker",
            "engine_status": "ok",
            "check_method": "cfchecker",
            "conventions_checked": ["cf", "ferret"],
            "counts": {"fatal": 0, "error": 1, "warn": 1},
            "global": [{"severity": "WARN", "message": "example global"}],
            "variables": {
                "temp": [
                    {
                        "convention": "cf",
                        "severity": "ERROR",
                        "message": "missing units",
                    }
                ]
            },
        }
    )

    assert "CF Compliance Report" in html
    assert "<details class='report-section'" not in html
    assert "<section class='report-section static-section'>" in html
    assert "Global Findings" in html
    assert "Variable Findings" in html
    assert "issue-card" in html
    assert "bootstrap@5" in html
    assert "summary-table" in html
    assert "kv-grid" not in html
    assert "PASSED" in html
