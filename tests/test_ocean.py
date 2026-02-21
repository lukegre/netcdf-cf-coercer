import numpy as np
import pytest
import xarray as xr

import nc_check  # noqa: F401
from nc_check.ocean import check_ocean_cover


def test_edge_of_map_detects_persistent_missing_longitudes() -> None:
    lon = np.arange(0.0, 360.0, 30.0)
    lat = np.array([-30.0, 0.0, 30.0])
    time = np.arange(6)
    data = np.ones((time.size, lat.size, lon.size), dtype=float)
    data[:, :, [0, -1]] = np.nan
    data[2, :, 4] = np.nan  # non-persistent, should not define the sliver result

    ds = xr.Dataset(
        data_vars={"sst": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    report = check_ocean_cover(
        ds,
        var_name="sst",
        check_land_ocean_offset=False,
        report_format="python",
    )

    assert report["edge_of_map"]["status"] == "fail"
    assert report["edge_of_map"]["missing_longitude_count"] == 2
    assert report["edge_of_map"]["missing_longitudes"] == [0.0, 330.0]
    assert report["edge_of_map"]["missing_longitude_ranges"] == [
        {"start": "0.0", "end": "0.0"},
        {"start": "330.0", "end": "330.0"},
    ]


def test_land_ocean_offset_check_detects_shifted_data() -> None:
    lon = np.arange(-180.0, 181.0, 1.0)
    lat = np.arange(-90.0, 91.0, 1.0)
    data = np.ones((lat.size, lon.size), dtype=float)

    # Match known land points used by the checker.
    land_points = [(23, 13), (-25, 134), (47, 103), (72, -40), (-15, -60)]
    for plat, plon in land_points:
        lat_idx = int(np.where(lat == plat)[0][0])
        lon_idx = int(np.where(lon == plon)[0][0])
        data[lat_idx, lon_idx] = np.nan

    base = xr.Dataset(
        data_vars={"sst": (("lat", "lon"), data)},
        coords={"lat": lat, "lon": lon},
    )

    good = check_ocean_cover(
        base,
        var_name="sst",
        check_edge_of_map=False,
        report_format="python",
    )
    assert good["land_ocean_offset"]["status"] == "pass"

    shifted = base.copy(deep=True)
    shifted["sst"] = shifted["sst"].roll(lon=20, roll_coords=False)

    bad = check_ocean_cover(
        shifted,
        var_name="sst",
        check_edge_of_map=False,
        report_format="python",
    )
    assert bad["land_ocean_offset"]["status"] == "fail"
    assert bad["land_ocean_offset"]["mismatch_count"] >= 1


def test_time_cover_reports_ranges() -> None:
    lon = np.arange(0.0, 360.0, 60.0)
    lat = np.array([-45.0, 45.0])
    time = np.arange(5)
    data = np.ones((time.size, lat.size, lon.size), dtype=float)
    data[1:3, :, :] = np.nan

    ds = xr.Dataset(
        data_vars={"sst": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    report = ds.check.time_cover(
        var_name="sst",
        report_format="python",
    )

    assert report["time_missing"]["status"] == "fail"
    assert report["time_missing"]["missing_slice_count"] == 2
    assert report["time_missing"]["missing_slice_ranges"] == [
        {"start_index": 1, "end_index": 2, "start": "1", "end": "2"}
    ]
    assert "time_format" not in report


def test_time_cover_accepts_xarray_decoded_time_coordinates() -> None:
    lon = np.arange(0.0, 360.0, 60.0)
    lat = np.array([-45.0, 45.0])
    time = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    data = np.ones((time.size, lat.size, lon.size), dtype=float)

    ds = xr.Dataset(
        data_vars={"sst": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    report = ds.check.time_cover(var_name="sst", report_format="python")

    assert report["time_missing"]["status"] == "pass"
    assert "time_format" not in report
    assert report["ok"] is True


@pytest.mark.parametrize(
    "time_values",
    [
        np.array([0.0, 1.0], dtype=float),
        np.array(["2024-01-01", "2024-01-02"], dtype=str),
    ],
)
def test_time_cover_reports_non_cf_time_types(
    time_values: np.ndarray,
) -> None:
    lat = np.array([-45.0, 45.0])
    lon = np.arange(0.0, 360.0, 60.0)
    data = np.ones((time_values.size, lat.size, lon.size), dtype=float)

    ds = xr.Dataset(
        data_vars={"sst": (("time", "lat", "lon"), data)},
        coords={"time": time_values, "lat": lat, "lon": lon},
    )

    report = ds.check.time_cover(var_name="sst", report_format="python")

    assert report["time_missing"]["status"] == "pass"
    assert report["ok"] is True
    assert "time_format" not in report


def test_ocean_cover_without_var_name_checks_all_eligible_variables() -> None:
    lon = np.arange(0.0, 360.0, 30.0)
    lat = np.array([-30.0, 0.0, 30.0])
    bad = np.ones((lat.size, lon.size), dtype=float)
    bad[:, [0, -1]] = np.nan
    good = np.ones((lat.size, lon.size), dtype=float)

    ds = xr.Dataset(
        data_vars={
            "sst": (("lat", "lon"), bad),
            "sss": (("lat", "lon"), good),
        },
        coords={"lat": lat, "lon": lon},
    )

    report = check_ocean_cover(
        ds,
        check_land_ocean_offset=False,
        report_format="python",
    )

    assert report["mode"] == "all_variables"
    assert report["checked_variable_count"] == 2
    assert report["checked_variables"] == ["sst", "sss"]
    assert report["reports"]["sst"]["edge_of_map"]["status"] == "fail"
    assert report["reports"]["sss"]["edge_of_map"]["status"] == "pass"
    assert report["ok"] is False


def test_time_cover_without_var_name_checks_all_variables() -> None:
    lon = np.arange(0.0, 360.0, 60.0)
    lat = np.array([-45.0, 45.0])
    time = np.arange(5)
    sst = np.ones((time.size, lat.size, lon.size), dtype=float)
    sst[1:3, :, :] = np.nan
    sss = np.ones((time.size, lat.size, lon.size), dtype=float)
    mask = np.ones((lat.size, lon.size), dtype=float)

    ds = xr.Dataset(
        data_vars={
            "sst": (("time", "lat", "lon"), sst),
            "sss": (("time", "lat", "lon"), sss),
            "mask": (("lat", "lon"), mask),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )

    report = ds.check.time_cover(report_format="python")

    assert report["mode"] == "all_variables"
    assert report["checked_variable_count"] == 3
    assert report["checked_variables"] == ["sst", "sss", "mask"]
    assert report["reports"]["sst"]["time_missing"]["status"] == "fail"
    assert report["reports"]["sss"]["time_missing"]["status"] == "pass"
    assert report["reports"]["mask"]["time_missing"]["status"] == "skipped_no_time"
    assert report["ok"] is False


def test_ocean_cover_all_checks_can_be_disabled() -> None:
    ds = xr.Dataset(
        data_vars={"sst": (("lat", "lon"), np.ones((2, 3)))},
        coords={"lat": [-1.0, 1.0], "lon": [0.0, 120.0, 240.0]},
    )

    report = check_ocean_cover(
        ds,
        var_name="sst",
        check_edge_of_map=False,
        check_land_ocean_offset=False,
        report_format="python",
    )

    assert report["edge_of_map"]["status"] == "skipped"
    assert report["land_ocean_offset"]["status"] == "skipped"
    assert report["ok"] is True


def test_ocean_cover_html_report_has_collapsible_sections_and_modern_style() -> None:
    ds = xr.Dataset(
        data_vars={"sst": (("lat", "lon"), np.ones((2, 3)))},
        coords={"lat": [-1.0, 1.0], "lon": [0.0, 120.0, 240.0]},
    )

    # Use the top-level export to ensure package users can pass report_format.
    html = nc_check.check_ocean_cover(
        ds,
        var_name="sst",
        check_edge_of_map=False,
        check_land_ocean_offset=False,
        report_format="html",
    )

    assert isinstance(html, str)
    assert "Ocean Coverage Report" in html
    assert "Check Summary" in html
    assert "<section class='report-section static-section'>" in html
    assert "<details class='report-section'" not in html
    assert "PASSED" in html
    assert "SKIPPED" in html
    assert "WARNING" not in html
    assert "bootstrap@5" in html


def test_time_cover_html_report_has_collapsible_sections_and_modern_style() -> None:
    ds = xr.Dataset(
        data_vars={"sst": (("time", "lat", "lon"), np.ones((2, 2, 3)))},
        coords={
            "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "lat": [-1.0, 1.0],
            "lon": [0.0, 120.0, 240.0],
        },
    )

    html = ds.check.time_cover(var_name="sst", report_format="html")

    assert isinstance(html, str)
    assert "Time Coverage Report" in html
    assert "Check Summary" in html
    assert "<section class='report-section static-section'>" in html
    assert "<details class='report-section'" not in html
    assert "PASSED" in html
    assert "bootstrap@5" in html


def test_time_cover_html_report_can_be_saved(tmp_path) -> None:
    ds = xr.Dataset(
        data_vars={"sst": (("time", "lat", "lon"), np.ones((2, 2, 3)))},
        coords={
            "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "lat": [-1.0, 1.0],
            "lon": [0.0, 120.0, 240.0],
        },
    )
    report_file = tmp_path / "time-cover.html"

    html = ds.check.time_cover(
        var_name="sst",
        report_format="html",
        report_html_file=report_file,
    )

    assert isinstance(html, str)
    assert "Time Coverage Report" in html
    assert "<section class='report-section static-section'>" in html
    assert "<details class='report-section'" not in html
    assert "PASSED" in html
    assert report_file.exists()
    assert report_file.read_text(encoding="utf-8") == html


def test_time_cover_html_skipped_is_not_reported_as_warning() -> None:
    ds = xr.Dataset(
        data_vars={"mask": (("lat", "lon"), np.ones((2, 3)))},
        coords={"lat": [-1.0, 1.0], "lon": [0.0, 120.0, 240.0]},
    )

    html = ds.check.time_cover(var_name="mask", report_format="html")

    assert isinstance(html, str)
    assert "SKIPPED" in html
    assert "WARNING" not in html


def test_ocean_multi_variable_html_uses_collapsible_variable_sections() -> None:
    lon = np.array([0.0, 120.0, 240.0])
    lat = np.array([-1.0, 1.0])
    ds = xr.Dataset(
        data_vars={
            "sst": (("lat", "lon"), np.ones((2, 3))),
            "sss": (("lat", "lon"), np.ones((2, 3))),
        },
        coords={"lat": lat, "lon": lon},
    )

    html = check_ocean_cover(
        ds,
        check_edge_of_map=False,
        check_land_ocean_offset=False,
        report_format="html",
    )

    assert isinstance(html, str)
    assert "Top Summary" in html
    assert "variable-report" in html
    assert html.count("<details class='report-section variable-report'") == 2
    assert "<details class='report-section variable-report' open>" not in html
    assert "summary-table" in html
    assert "kv-grid" not in html


def test_time_multi_variable_html_uses_collapsible_variable_sections() -> None:
    lon = np.array([0.0, 120.0, 240.0])
    lat = np.array([-1.0, 1.0])
    time = np.array([0, 1])
    ds = xr.Dataset(
        data_vars={
            "sst": (("time", "lat", "lon"), np.ones((2, 2, 3))),
            "sss": (("time", "lat", "lon"), np.ones((2, 2, 3))),
            "mask": (("lat", "lon"), np.ones((2, 3))),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )

    html = ds.check.time_cover(report_format="html")

    assert isinstance(html, str)
    assert "Top Summary" in html
    assert "variable-report" in html
    assert html.count("<details class='report-section variable-report'") == 3
    assert "<details class='report-section variable-report' open>" not in html
    assert "summary-table" in html
    assert "kv-grid" not in html
    assert (
        "Variable: mask</span><span class='summary-badge'><span class='badge report-badge "
        "rounded-pill bg-secondary-subtle"
    ) in html
