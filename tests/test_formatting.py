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

    assert "Ocean Coverage Report" in out
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

    assert "Time Coverage Report" in out
    assert "time_missing" in out
    assert "Missing Time Slice Ranges" in out


def test_render_pretty_report_html_uses_collapsible_bootstrap_layout() -> None:
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
    assert "<details class='report-section'" in html
    assert "Global Findings" in html
    assert "Variable Findings" in html
    assert "bootstrap@5" in html
    assert "summary-table" in html
    assert "kv-grid" not in html
    assert "PASSED" in html
