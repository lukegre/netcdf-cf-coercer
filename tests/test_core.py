import numpy as np
import pytest
import xarray as xr

from nc_check import core


def test_make_compliant_creates_missing_axis_coordinates() -> None:
    ds = xr.Dataset(
        data_vars={"field": (("lat", "lon"), np.ones((2, 3)))},
    )

    out = core.make_dataset_compliant(ds)

    assert "lat" in out.coords
    assert "lon" in out.coords
    assert out["lat"].attrs["standard_name"] == "latitude"
    assert out["lon"].attrs["standard_name"] == "longitude"
    assert out["lat"].encoding.get("_FillValue") is None
    assert out["lon"].encoding.get("_FillValue") is None


def test_make_compliant_casts_string_lat_lon_coords_to_float() -> None:
    ds = xr.Dataset(
        data_vars={"field": (("lat", "lon"), np.ones((2, 2)))},
        coords={"lat": ["10", "11"], "lon": ["20", "21"]},
    )

    out = core.make_dataset_compliant(ds)

    assert np.issubdtype(out["lat"].dtype, np.floating)
    assert np.issubdtype(out["lon"].dtype, np.floating)
    assert out["lat"].attrs["units"] == "degrees_north"
    assert out["lon"].attrs["units"] == "degrees_east"


def test_make_compliant_renames_badly_cased_variable_attrs() -> None:
    ds = xr.Dataset(
        data_vars={
            "temp": (
                ("time",),
                [290.0],
                {
                    "Units": "K",
                    "Standard_Name": "air_temperature",
                    "Long_Name": "air temperature",
                },
            )
        },
        coords={"time": [0]},
    )

    out = core.make_dataset_compliant(ds)

    assert out["temp"].attrs["units"] == "K"
    assert out["temp"].attrs["standard_name"] == "air_temperature"
    assert out["temp"].attrs["long_name"] == "air temperature"
    assert "Units" not in out["temp"].attrs
    assert "Standard_Name" not in out["temp"].attrs
    assert "Long_Name" not in out["temp"].attrs


def test_make_compliant_renamed_coord_attrs_enable_axis_inference() -> None:
    ds = xr.Dataset(
        data_vars={"field": (("xcoord",), [1.0])},
        coords={"xcoord": ("xcoord", [5.0], {"Standard_Name": "longitude"})},
    )

    out = core.make_dataset_compliant(ds)

    assert out["xcoord"].attrs["standard_name"] == "longitude"
    assert out["xcoord"].attrs["long_name"] == "longitude"
    assert out["xcoord"].attrs["units"] == "degrees_east"
    assert out["xcoord"].attrs["axis"] == "X"
    assert "Standard_Name" not in out["xcoord"].attrs


def test_check_dataset_compliant_raises_if_no_fallback(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise RuntimeError("checker failed")

    monkeypatch.setattr(core, "_run_cfchecker_on_dataset", _raise)

    ds = xr.Dataset(
        data_vars={"v": (("time",), [1.0])},
        coords={"time": [0]},
    )

    with pytest.raises(RuntimeError, match="checker failed"):
        core.check_dataset_compliant(ds, fallback_to_heuristic=False)


def test_format_cf_version_normalizes_prefix() -> None:
    assert core._format_cf_version("1.12") == "CF-1.12"
    assert core._format_cf_version("CF-1.12") == "CF-1.12"


def test_translate_cfchecker_results_counts_and_scopes() -> None:
    ds = xr.Dataset(
        data_vars={"temp": (("time",), [1.0])},
        coords={"time": [0]},
    )
    results = {
        "global": {"WARN": ["g-warn"], "ERROR": ["g-error"]},
        "variables": {
            "temp": {"ERROR": ["bad var"]},
            "time": {"WARN": ["coord warn"]},
        },
    }

    translated = core._translate_cfchecker_results(results, "1.12", ds)

    assert translated["cf_version"] == "CF-1.12"
    assert translated["counts"] == {"fatal": 0, "error": 2, "warn": 2}
    assert "temp" in translated["variables"]
    assert "time" in translated["coordinates"]


def test_normalize_requested_conventions_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported conventions"):
        core._normalize_requested_conventions("cf,not-real")


def test_check_dataset_compliant_rejects_unknown_report_format() -> None:
    ds = xr.Dataset(data_vars={"v": (("time",), [1.0])}, coords={"time": [0]})

    with pytest.raises(ValueError, match="Unsupported report_format"):
        core.check_dataset_compliant(
            ds,
            conventions="ferret",
            standard_name_table_xml=None,
            report_format="bad",
        )


def test_check_dataset_compliant_rejects_html_file_without_html_format() -> None:
    ds = xr.Dataset(data_vars={"v": (("time",), [1.0])}, coords={"time": [0]})

    with pytest.raises(ValueError, match="only valid when report_format='html'"):
        core.check_dataset_compliant(
            ds,
            conventions="ferret",
            standard_name_table_xml=None,
            report_format="python",
            report_html_file="report.html",
        )


def _empty_cfchecker_report() -> dict[str, object]:
    return {
        "cf_version": "CF-1.12",
        "engine": "cfchecker",
        "engine_status": "ok",
        "check_method": "cfchecker",
        "global": [],
        "coordinates": {},
        "variables": {},
        "suggestions": {"variables": {}},
        "notes": [],
        "counts": {"fatal": 0, "error": 0, "warn": 0},
    }


def test_check_dataset_compliant_flags_wrong_case_variable_units(monkeypatch) -> None:
    monkeypatch.setattr(
        core, "_run_cfchecker_on_dataset", lambda *a, **k: _empty_cfchecker_report()
    )

    ds = xr.Dataset(
        data_vars={
            "temp": (
                ("time",),
                [290.0],
                {"Units": "K", "standard_name": "air_temperature"},
            )
        },
        coords={"time": [0]},
    )

    issues = core.check_dataset_compliant(
        ds,
        conventions="cf",
        standard_name_table_xml=None,
        report_format="python",
    )

    findings = issues["variables"]["temp"]
    assert any(
        isinstance(item, dict)
        and item.get("item") == "attr_case_mismatch"
        and item.get("current") == "Units"
        and item.get("expected") == "units"
        for item in findings
    )
    assert issues["counts"]["warn"] >= 1


def test_check_dataset_compliant_flags_wrong_case_coordinate_units(monkeypatch) -> None:
    monkeypatch.setattr(
        core, "_run_cfchecker_on_dataset", lambda *a, **k: _empty_cfchecker_report()
    )

    ds = xr.Dataset(
        data_vars={"v": (("time",), [1.0])},
        coords={"time": ("time", [0.0], {"Units": "days since 1970-01-01"})},
    )

    issues = core.check_dataset_compliant(
        ds,
        conventions="cf",
        standard_name_table_xml=None,
        report_format="python",
    )

    findings = issues["coordinates"]["time"]
    assert any(
        isinstance(item, dict)
        and item.get("item") == "attr_case_mismatch"
        and item.get("current") == "Units"
        and item.get("expected") == "units"
        for item in findings
    )
    assert issues["counts"]["warn"] >= 1
