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
