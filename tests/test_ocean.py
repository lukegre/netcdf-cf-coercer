import numpy as np
import xarray as xr

import nc_check  # noqa: F401
from nc_check.ocean import check_ocean_cover


def test_edge_sliver_detects_persistent_missing_longitudes() -> None:
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
        check_time_missing=False,
    )

    assert report["edge_sliver"]["status"] == "fail"
    assert report["edge_sliver"]["missing_longitude_count"] == 2
    assert report["edge_sliver"]["missing_longitude_indices"] == [0, lon.size - 1]
    assert report["edge_sliver"]["missing_slice_ranges"] == [
        {"start_index": 0, "end_index": 5, "start": "0", "end": "5"}
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
        check_edge_sliver=False,
        check_time_missing=False,
    )
    assert good["land_ocean_offset"]["status"] == "pass"

    shifted = base.copy(deep=True)
    shifted["sst"] = shifted["sst"].roll(lon=20, roll_coords=False)

    bad = check_ocean_cover(
        shifted,
        var_name="sst",
        check_edge_sliver=False,
        check_time_missing=False,
    )
    assert bad["land_ocean_offset"]["status"] == "fail"
    assert bad["land_ocean_offset"]["mismatch_count"] >= 1


def test_time_missing_reports_ranges() -> None:
    lon = np.arange(0.0, 360.0, 60.0)
    lat = np.array([-45.0, 45.0])
    time = np.arange(5)
    data = np.ones((time.size, lat.size, lon.size), dtype=float)
    data[1:3, :, :] = np.nan

    ds = xr.Dataset(
        data_vars={"sst": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    report = ds.check.check_ocean_cover(
        var_name="sst",
        check_edge_sliver=False,
        check_land_ocean_offset=False,
    )

    assert report["time_missing"]["status"] == "fail"
    assert report["time_missing"]["missing_slice_count"] == 2
    assert report["time_missing"]["missing_slice_ranges"] == [
        {"start_index": 1, "end_index": 2, "start": "1", "end": "2"}
    ]


def test_ocean_cover_all_checks_can_be_disabled() -> None:
    ds = xr.Dataset(
        data_vars={"sst": (("lat", "lon"), np.ones((2, 3)))},
        coords={"lat": [-1.0, 1.0], "lon": [0.0, 120.0, 240.0]},
    )

    report = check_ocean_cover(
        ds,
        var_name="sst",
        check_edge_sliver=False,
        check_land_ocean_offset=False,
        check_time_missing=False,
    )

    assert report["edge_sliver"]["status"] == "skipped"
    assert report["land_ocean_offset"]["status"] == "skipped"
    assert report["time_missing"]["status"] == "skipped"
    assert report["ok"] is True
