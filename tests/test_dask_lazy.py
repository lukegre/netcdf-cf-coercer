import numpy as np
import pytest
import xarray as xr

from nc_check import core

dask = pytest.importorskip("dask.array")
netCDF4 = pytest.importorskip("netCDF4")


def test_make_compliant_preserves_lazy_data_vars() -> None:
    data = dask.ones((4, 3, 2), chunks=(2, 3, 2))
    ds = xr.Dataset(
        data_vars={"temp": (("time", "lat", "lon"), data)},
        coords={"time": [0, 1, 2, 3], "lat": [10.0, 11.0, 12.0], "lon": [20.0, 21.0]},
    )

    out = ds.check.make_cf_compliant()

    assert isinstance(out["temp"].data, dask.Array)
    assert out["temp"].chunks == ds["temp"].chunks
    assert out.attrs["Conventions"] == "CF-1.12"


def test_make_compliant_keeps_lazy_coord_cast() -> None:
    lat = dask.from_array(np.array(["10", "11", "12"]), chunks=(2,))
    lon = dask.from_array(np.array(["20", "21"]), chunks=(2,))
    data = dask.ones((3, 2), chunks=(2, 2))
    ds = xr.Dataset(
        data_vars={"field": (("lat", "lon"), data)},
        coords={
            "lat": xr.DataArray(lat, dims=("lat",)),
            "lon": xr.DataArray(lon, dims=("lon",)),
        },
    )

    out = core.make_dataset_compliant(ds)

    assert isinstance(out["field"].data, dask.Array)
    assert np.issubdtype(out["lat"].dtype, np.floating)
    assert np.issubdtype(out["lon"].dtype, np.floating)


def test_check_accepts_lazy_dataset_with_fallback(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise RuntimeError("force fallback")

    monkeypatch.setattr(core, "_run_cfchecker_on_dataset", _raise)

    data = dask.ones((4,), chunks=(2,))
    ds = xr.Dataset(
        data_vars={"temp": (("time",), data)},
        coords={"time": [0, 1, 2, 3]},
    )

    issues = ds.check.cf()

    assert issues["check_method"] == "heuristic"
    assert issues["engine_status"] == "unavailable"


def test_check_accepts_lazy_dataset_without_fallback(monkeypatch) -> None:
    def _fake_run(ds, **kwargs):
        assert isinstance(ds["temp"].data, dask.Array)
        return {
            "cf_version": "CF-1.12",
            "engine": "cfchecker",
            "engine_status": "ok",
            "check_method": "cfchecker",
            "global": [],
            "coordinates": {},
            "variables": {},
            "notes": [],
            "counts": {"fatal": 0, "error": 0, "warn": 0},
        }

    monkeypatch.setattr(core, "_run_cfchecker_on_dataset", _fake_run)

    data = dask.ones((4,), chunks=(2,))
    ds = xr.Dataset(
        data_vars={"temp": (("time",), data)},
        coords={"time": [0, 1, 2, 3]},
    )

    issues = ds.check.cf(standard_name_table_xml=None)

    assert issues["check_method"] == "cfchecker"
    assert issues["engine_status"] == "ok"


def test_make_compliant_lazy_write_has_no_coord_fillvalue(tmp_path) -> None:
    data = dask.ones((4, 3, 2), chunks=(2, 3, 2))
    ds = xr.Dataset(
        data_vars={"temp": (("time", "lat", "lon"), data)},
        coords={
            "time": [0, 1, 2, 3],
            "lat": [10.0, 11.0, 12.0],
            "lon": [20.0, 21.0],
            "depth": ("time", [5.0, 5.0, 5.0, 5.0]),
        },
    )
    ds["time"].encoding = {"_FillValue": -9999}
    ds["lat"].encoding = {"_FillValue": -9999.0}
    ds["lon"].encoding = {"_FillValue": -9999.0}
    ds["depth"].encoding = {"_FillValue": -9999.0}

    out = ds.check.make_cf_compliant()

    assert isinstance(out["temp"].data, dask.Array)
    for coord_name in out.coords:
        assert out[coord_name].encoding.get("_FillValue") is None

    path = tmp_path / "lazy_no_coord_fill.nc"
    out.to_netcdf(path)

    nc = netCDF4.Dataset(path, mode="r")
    try:
        for coord_name in out.coords:
            assert "_FillValue" not in nc.variables[str(coord_name)].ncattrs()
    finally:
        nc.close()


def test_checker_payload_bytes_do_not_compute_original_lazy_data() -> None:
    def _raise_if_computed(block: np.ndarray) -> np.ndarray:
        raise RuntimeError("original lazy data was computed")

    base = dask.ones((4,), chunks=(2,))
    guarded = base.map_blocks(
        _raise_if_computed,
        dtype=base.dtype,
        meta=np.array((), dtype=base.dtype),
    )
    ds = xr.Dataset(
        data_vars={"temp": (("time",), guarded)},
        coords={"time": [0, 1, 2, 3]},
    )

    payload = core._as_netcdf_bytes(ds)

    assert isinstance(payload, bytes)
    assert payload
