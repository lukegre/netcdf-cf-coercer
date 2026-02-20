import numpy as np
import pytest
import xarray as xr
from pathlib import Path

import nc_check  # noqa: F401
from nc_check import core
from nc_check.formatting import to_yaml_like


def test_check_reports_missing_conventions_and_coord_attrs() -> None:
    ds = xr.Dataset(
        data_vars={"temp": (("time", "lat", "lon"), np.ones((2, 2, 2)))},
        coords={"time": [0, 1], "lat": [10.0, 11.0], "lon": [20.0, 21.0]},
    )

    issues = ds.check.compliance(report_format="python")

    assert issues["cf_version"] == "CF-1.12"
    assert issues["engine"] == "cfchecker"
    assert issues["check_method"] in {"cfchecker", "heuristic"}
    if issues["engine_status"] == "ok":
        assert issues["check_method"] == "cfchecker"
        assert issues["global"] or issues["variables"]
    else:
        assert issues["check_method"] == "heuristic"
        assert "checker_error" in issues
        assert any(i["item"] == "Conventions" for i in issues["global"])
        assert "time" in issues["coordinates"]
        assert "lat" in issues["coordinates"]
        assert "lon" in issues["coordinates"]


def test_comply_sets_expected_metadata_for_time_lat_lon() -> None:
    ds = xr.Dataset(
        data_vars={"temp": (("time", "lat", "lon"), np.ones((1, 1, 1)))},
        coords={"time": [0], "lat": [10.0], "lon": [20.0]},
    )

    out = ds.check.comply()

    assert out.attrs["Conventions"] == "CF-1.12"
    assert out["time"].attrs["standard_name"] == "time"
    assert out["time"].attrs["axis"] == "T"
    assert out["lat"].attrs["standard_name"] == "latitude"
    assert out["lat"].attrs["units"] == "degrees_north"
    assert out["lat"].attrs["axis"] == "Y"
    assert out["lon"].attrs["standard_name"] == "longitude"
    assert out["lon"].attrs["units"] == "degrees_east"
    assert out["lon"].attrs["axis"] == "X"
    assert "coordinates" not in out["temp"].attrs


def test_time_only_dataset_supported() -> None:
    ds = xr.Dataset(
        data_vars={"signal": (("time",), [1.0, 2.0, 3.0])},
        coords={"time": [0, 1, 2]},
    )

    out = ds.check.comply()

    assert out.attrs["Conventions"] == "CF-1.12"
    assert out["time"].attrs["standard_name"] == "time"


def test_lat_lon_only_dataset_supported() -> None:
    ds = xr.Dataset(
        data_vars={"field": (("lat", "lon"), np.ones((2, 2)))},
        coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]},
    )

    out = ds.check.comply()

    assert out.attrs["Conventions"] == "CF-1.12"
    assert out["lat"].attrs["standard_name"] == "latitude"
    assert out["lon"].attrs["standard_name"] == "longitude"


def test_comply_clears_coordinate_encodings() -> None:
    ds = xr.Dataset(
        data_vars={"temp": (("time", "lat", "lon"), np.ones((1, 1, 1)))},
        coords={"time": [0], "lat": [10.0], "lon": [20.0], "depth": ("time", [5.0])},
    )
    ds["time"].encoding = {"_FillValue": -9999}
    ds["lat"].encoding = {"_FillValue": -9999.0}
    ds["lon"].encoding = {"_FillValue": -9999.0}
    ds["depth"].encoding = {"_FillValue": -9999.0}
    ds["depth"].attrs["_FillValue"] = -9999.0

    out = ds.check.comply()

    for coord_name in out.coords:
        assert out[coord_name].encoding.get("_FillValue") is None
        assert "_FillValue" not in out[coord_name].attrs


def test_unknown_dims_are_reported_but_not_forced() -> None:
    ds = xr.Dataset(data_vars={"v": (("station",), [1, 2, 3])})

    issues = ds.check.compliance(report_format="python")
    out = ds.check.comply()

    if issues["engine_status"] == "unavailable":
        assert any("station" in note for note in issues["notes"])
    else:
        assert isinstance(issues["variables"], dict)
    assert out.attrs["Conventions"] == "CF-1.12"


def test_fallback_checker_reports_variable_level_issues(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise RuntimeError("force fallback")

    monkeypatch.setattr(core, "_run_cfchecker_on_dataset", _raise)

    ds = xr.Dataset(
        data_vars={"bad-name": (("time",), [1.0, 2.0])},
        coords={"time": [0, 1]},
    )

    issues = ds.check.compliance(report_format="python")

    assert issues["engine_status"] == "unavailable"
    assert issues["check_method"] == "heuristic"
    assert "bad-name" in issues["variables"]
    items = {entry["item"] for entry in issues["variables"]["bad-name"]}
    assert "invalid_variable_name" in items
    assert "missing_units_attr" in items
    assert "missing_standard_or_long_name" in items


def test_standard_name_suggestions_from_xml(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise RuntimeError("force fallback")

    monkeypatch.setattr(core, "_run_cfchecker_on_dataset", _raise)

    ds = xr.Dataset(
        data_vars={"tos": (("time",), [290.0, 291.0])},
        coords={"time": [0, 1]},
    )
    ds["tos"].attrs["long_name"] = "surface ocean temperature"
    ds["tos"].attrs["units"] = "degC"

    table = Path("tests/data/cf-standard-name-table.xml")
    issues = ds.check.compliance(
        standard_name_table_xml=str(table),
        report_format="python",
    )

    suggestions = issues["suggestions"]["variables"]["tos"]
    assert "sea_surface_temperature" in suggestions["recommended_standard_names"]
    assert suggestions["recommended_units"] == "K"


def test_units_check_for_known_standard_name_from_xml(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise RuntimeError("force fallback")

    monkeypatch.setattr(core, "_run_cfchecker_on_dataset", _raise)

    ds = xr.Dataset(
        data_vars={"sst": (("time",), [290.0, 291.0])},
        coords={"time": [0, 1]},
    )
    ds["sst"].attrs["standard_name"] = "sea_surface_temperature"
    ds["sst"].attrs["units"] = "degC"

    table = Path("tests/data/cf-standard-name-table.xml")
    issues = ds.check.compliance(
        standard_name_table_xml=str(table),
        report_format="python",
    )

    unit_check = issues["suggestions"]["variables"]["sst"]["units_check"]
    assert unit_check["status"] == "mismatch"
    assert unit_check["expected_units"] == "K"


def test_standard_name_table_file_url_is_supported(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise RuntimeError("force fallback")

    monkeypatch.setattr(core, "_run_cfchecker_on_dataset", _raise)

    ds = xr.Dataset(
        data_vars={"tos": (("time",), [290.0, 291.0])},
        coords={"time": [0, 1]},
    )
    ds["tos"].attrs["long_name"] = "surface ocean temperature"

    table_url = Path("tests/data/cf-standard-name-table.xml").resolve().as_uri()
    issues = ds.check.compliance(
        standard_name_table_xml=table_url,
        report_format="python",
    )

    suggestions = issues["suggestions"]["variables"]["tos"]
    assert "sea_surface_temperature" in suggestions["recommended_standard_names"]


def test_domain_bias_changes_suggested_standard_name(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise RuntimeError("force fallback")

    monkeypatch.setattr(core, "_run_cfchecker_on_dataset", _raise)

    ds = xr.Dataset(
        data_vars={"temp": (("time",), [290.0, 291.0])},
        coords={"time": [0, 1]},
    )
    ds["temp"].attrs["long_name"] = "temperature"

    table = Path("tests/data/cf-standard-name-table.xml")

    ocean = ds.check.compliance(
        standard_name_table_xml=str(table),
        domain="ocean",
        report_format="python",
    )
    atmosphere = ds.check.compliance(
        standard_name_table_xml=str(table),
        domain="atmosphere",
        report_format="python",
    )

    ocean_top = ocean["suggestions"]["variables"]["temp"]["recommended_standard_names"][
        0
    ]
    atmosphere_top = atmosphere["suggestions"]["variables"]["temp"][
        "recommended_standard_names"
    ][0]

    assert ocean_top == "sea_surface_temperature"
    assert atmosphere_top == "air_temperature"


def test_tables_report_prints_rich_output(monkeypatch, capsys) -> None:
    def _raise(*args, **kwargs):
        raise RuntimeError("force fallback")

    monkeypatch.setattr(core, "_run_cfchecker_on_dataset", _raise)

    ds = xr.Dataset(
        data_vars={"temp": (("time",), [290.0, 291.0])},
        coords={"time": [0, 1]},
    )

    report = ds.check.compliance(report_format="tables")
    out = capsys.readouterr().out

    assert report is None
    assert "CF Compliance Report" in out
    assert "Engine status" in out
    assert "Check method" in out
    assert "Variable Findings" in out


def test_html_report_can_be_saved(tmp_path) -> None:
    ds = xr.Dataset(
        data_vars={"temp": (("time",), [290.0, 291.0])},
        coords={"time": [0, 1]},
    )
    report_file = tmp_path / "cf-report.html"

    html = ds.check.compliance(
        conventions="ferret",
        standard_name_table_xml=None,
        report_format="html",
        report_html_file=report_file,
    )

    assert isinstance(html, str)
    assert "CF Compliance Report" in html
    assert "<details class='report-section'" not in html
    assert "<section class='report-section static-section'>" in html
    assert "Notes" in html
    assert "bootstrap@5" in html
    assert "summary-table" in html
    assert "kv-grid" not in html
    assert "WARNING" in html
    assert report_file.exists()
    assert report_file.read_text(encoding="utf-8") == html


def test_cf_alias_accepts_html_report_format() -> None:
    ds = xr.Dataset(
        data_vars={"temp": (("time",), [290.0, 291.0])},
        coords={"time": [0, 1]},
    )

    html = ds.check.cf(
        conventions="ferret",
        standard_name_table_xml=None,
        report_format="html",
    )

    assert isinstance(html, str)
    assert "CF Compliance Report" in html
    assert "<details class='report-section'" not in html
    assert "<section class='report-section static-section'>" in html
    assert "summary-table" in html
    assert "WARNING" in html


def test_ferret_convention_flags_coordinate_fillvalue() -> None:
    ds = xr.Dataset(
        data_vars={"temp": (("time",), [290.0, 291.0])},
        coords={"time": [0, 1]},
    )
    ds["time"].encoding = {"_FillValue": -9999}

    issues = ds.check.compliance(
        standard_name_table_xml=None,
        report_format="python",
    )

    findings = issues["coordinates"]["time"]
    assert any(
        item.get("item") == "coord_fillvalue_forbidden"
        and item.get("convention") == "ferret"
        for item in findings
    )
    assert "ferret" in issues["conventions_checked"]


def test_can_disable_ferret_convention_checks() -> None:
    ds = xr.Dataset(
        data_vars={"temp": (("time",), [290.0, 291.0])},
        coords={"time": [0, 1]},
    )
    ds["time"].encoding = {"_FillValue": -9999}

    issues = ds.check.compliance(
        standard_name_table_xml=None,
        conventions="cf",
        report_format="python",
    )

    coord_items = issues["coordinates"].get("time", [])
    assert not any(
        item.get("item") == "coord_fillvalue_forbidden" for item in coord_items
    )
    assert issues["conventions_checked"] == ["cf"]


def test_ferret_only_mode_skips_cfchecker(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise AssertionError("cfchecker should not run for ferret-only checks")

    monkeypatch.setattr(core, "_run_cfchecker_on_dataset", _raise)

    ds = xr.Dataset(
        data_vars={"temp": (("time",), [290.0, 291.0])},
        coords={"time": [0, 1]},
    )
    ds["time"].encoding = {"_FillValue": -9999}

    issues = ds.check.compliance(
        conventions="ferret",
        standard_name_table_xml=None,
        report_format="python",
    )

    assert issues["engine_status"] == "skipped"
    assert issues["check_method"] == "conventions_only"
    assert issues["conventions_checked"] == ["ferret"]
    assert "time" in issues["coordinates"]


def test_yaml_like_formatter_still_available() -> None:
    text = to_yaml_like({"items": [{"name": "alpha", "value": 1}]})
    assert "items:" in text
    assert "-\n" in text


def test_all_html_report_combines_selected_checks() -> None:
    ds = xr.Dataset(
        data_vars={"sst": (("time", "lat", "lon"), np.ones((2, 2, 3)))},
        coords={"time": [0, 1], "lat": [-1.0, 1.0], "lon": [0.0, 120.0, 240.0]},
    )

    html = ds.check.all(
        compliance=True,
        ocean_cover=True,
        time_cover=True,
        conventions="ferret",
        standard_name_table_xml=None,
        check_edge_of_map=False,
        check_land_ocean_offset=False,
        report_format="html",
    )

    assert isinstance(html, str)
    assert "Full Dataset Check Report" in html
    assert "Combined Check Summary" in html
    assert "CF Compliance" in html
    assert "Ocean Coverage" in html
    assert "Time Coverage" in html
    assert "Selected checks: compliance, ocean_cover, time_cover" in html
    assert "bootstrap@5" in html
    assert "<details class='report-section' open>" not in html
    assert "<details class='report-section variable-report' open>" not in html


def test_all_python_report_can_run_subset_of_checks() -> None:
    ds = xr.Dataset(
        data_vars={"sst": (("time", "lat", "lon"), np.ones((2, 2, 3)))},
        coords={"time": [0, 1], "lat": [-1.0, 1.0], "lon": [0.0, 120.0, 240.0]},
    )

    report = ds.check.all(
        compliance=False,
        ocean_cover=True,
        time_cover=False,
        check_edge_of_map=False,
        check_land_ocean_offset=False,
        report_format="python",
    )

    assert isinstance(report, dict)
    assert report["summary"]["checks_run"] == 1
    assert report["reports"].keys() == {"ocean_cover"}
    assert report["check_summary"][0]["check"] == "ocean_cover"


def test_all_rejects_all_checks_disabled() -> None:
    ds = xr.Dataset(
        data_vars={"sst": (("time", "lat", "lon"), np.ones((2, 2, 3)))},
        coords={"time": [0, 1], "lat": [-1.0, 1.0], "lon": [0.0, 120.0, 240.0]},
    )

    with pytest.raises(ValueError, match="At least one check must be enabled"):
        ds.check.all(
            compliance=False,
            ocean_cover=False,
            time_cover=False,
            report_format="python",
        )


def test_python_api_ocean_and_time_cover_accept_custom_coordinate_names() -> None:
    ds = xr.Dataset(
        data_vars={"sst": (("t", "y", "x"), np.ones((2, 2, 3)))},
        coords={"t": [0, 1], "y": [-1.0, 1.0], "x": [0.0, 120.0, 240.0]},
    )

    ocean_report = ds.check.ocean_cover(
        var_name="sst",
        lon_name="x",
        lat_name="y",
        time_name="t",
        check_edge_of_map=False,
        check_land_ocean_offset=False,
        report_format="python",
    )
    time_report = ds.check.time_cover(
        var_name="sst",
        time_name="t",
        report_format="python",
    )

    assert ocean_report["grid"]["lon_name"] == "x"
    assert ocean_report["grid"]["lat_name"] == "y"
    assert ocean_report["grid"]["time_dim"] == "t"
    assert time_report["time_dim"] == "t"


def test_python_api_all_accepts_custom_coordinate_names() -> None:
    ds = xr.Dataset(
        data_vars={"sst": (("t", "y", "x"), np.ones((2, 2, 3)))},
        coords={"t": [0, 1], "y": [-1.0, 1.0], "x": [0.0, 120.0, 240.0]},
    )

    report = ds.check.all(
        compliance=False,
        ocean_cover=True,
        time_cover=True,
        lon_name="x",
        lat_name="y",
        time_name="t",
        check_edge_of_map=False,
        check_land_ocean_offset=False,
        report_format="python",
    )

    ocean_report = report["reports"]["ocean_cover"]
    time_report = report["reports"]["time_cover"]
    assert ocean_report["grid"]["lon_name"] == "x"
    assert ocean_report["grid"]["lat_name"] == "y"
    assert ocean_report["grid"]["time_dim"] == "t"
    assert time_report["time_dim"] == "t"


def test_full_alias_still_works() -> None:
    ds = xr.Dataset(
        data_vars={"sst": (("time", "lat", "lon"), np.ones((2, 2, 3)))},
        coords={"time": [0, 1], "lat": [-1.0, 1.0], "lon": [0.0, 120.0, 240.0]},
    )

    report = ds.check.full(
        compliance=False,
        ocean_cover=True,
        time_cover=False,
        check_edge_of_map=False,
        check_land_ocean_offset=False,
        report_format="python",
    )

    assert isinstance(report, dict)
    assert report["summary"]["checks_run"] == 1
    assert report["reports"].keys() == {"ocean_cover"}
