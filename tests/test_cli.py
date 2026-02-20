from __future__ import annotations

import xarray as xr

from nc_check import cli


def test_run_check_uses_tables_report_format(monkeypatch, tmp_path, capsys) -> None:
    source = tmp_path / "in.nc"
    xr.Dataset(data_vars={"v": (("time",), [1.0])}, coords={"time": [0]}).to_netcdf(
        source
    )

    seen: dict[str, object] = {}

    def _fake_check(
        ds: xr.Dataset,
        *,
        report_format: str = "python",
        **kwargs: object,
    ) -> None:
        seen["report_format"] = report_format
        seen["conventions"] = kwargs.get("conventions")
        print("report-output")

    monkeypatch.setattr(cli, "check_dataset_compliant", _fake_check)

    seen_open: dict[str, object] = {}
    original_open_dataset = cli.xr.open_dataset

    def _spy_open_dataset(*args: object, **kwargs: object):
        seen_open["args"] = args
        seen_open["kwargs"] = kwargs
        return original_open_dataset(*args, **kwargs)

    monkeypatch.setattr(cli.xr, "open_dataset", _spy_open_dataset)

    status = cli.run_check([str(source)])
    out = capsys.readouterr().out

    assert status == 0
    assert seen_open["kwargs"] == {"chunks": {}}
    assert seen["report_format"] == "tables"
    assert seen["conventions"] == "cf,ferret"
    assert "report-output" in out


def test_run_comply_writes_output_file(tmp_path) -> None:
    source = tmp_path / "in.nc"
    target = tmp_path / "out.nc"
    xr.Dataset(
        data_vars={"temp": (("lat", "lon"), [[280.0]])},
        coords={"lat": ["10"], "lon": ["20"]},
    ).to_netcdf(source)

    status = cli.run_comply([str(source), str(target)])

    assert status == 0
    assert target.exists()

    with xr.open_dataset(target) as out:
        assert out.attrs["Conventions"] == "CF-1.12"
        assert out["lat"].attrs["standard_name"] == "latitude"
        assert out["lon"].attrs["standard_name"] == "longitude"


def test_run_comply_opens_input_with_chunks(monkeypatch, tmp_path) -> None:
    source = tmp_path / "in.nc"
    target = tmp_path / "out.nc"
    xr.Dataset(
        data_vars={"temp": (("lat", "lon"), [[280.0]])},
        coords={"lat": ["10"], "lon": ["20"]},
    ).to_netcdf(source)

    seen_open: dict[str, object] = {}
    original_open_dataset = cli.xr.open_dataset

    def _spy_open_dataset(*args: object, **kwargs: object):
        seen_open["args"] = args
        seen_open["kwargs"] = kwargs
        return original_open_dataset(*args, **kwargs)

    monkeypatch.setattr(cli.xr, "open_dataset", _spy_open_dataset)

    status = cli.run_comply([str(source), str(target)])

    assert status == 0
    assert seen_open["kwargs"] == {"chunks": {}}


def test_run_check_forwards_custom_conventions(monkeypatch, tmp_path) -> None:
    source = tmp_path / "in.nc"
    xr.Dataset(data_vars={"v": (("time",), [1.0])}, coords={"time": [0]}).to_netcdf(
        source
    )

    seen: dict[str, object] = {}

    def _fake_check(
        ds: xr.Dataset,
        *,
        report_format: str = "python",
        **kwargs: object,
    ) -> None:
        seen["report_format"] = report_format
        seen["conventions"] = kwargs.get("conventions")

    monkeypatch.setattr(cli, "check_dataset_compliant", _fake_check)

    status = cli.run_check([str(source), "--conventions", "ferret"])

    assert status == 0
    assert seen["report_format"] == "tables"
    assert seen["conventions"] == "ferret"


def test_run_check_save_report_uses_html_and_default_output_path(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "sample.nc"
    xr.Dataset(data_vars={"v": (("time",), [1.0])}, coords={"time": [0]}).to_netcdf(
        source
    )

    seen: dict[str, object] = {}

    def _fake_check(
        ds: xr.Dataset,
        *,
        report_format: str = "python",
        report_html_file: str | None = None,
        **kwargs: object,
    ) -> None:
        seen["report_format"] = report_format
        seen["report_html_file"] = report_html_file
        seen["conventions"] = kwargs.get("conventions")

    monkeypatch.setattr(cli, "check_dataset_compliant", _fake_check)

    status = cli.run_check([str(source), "--save-report"])

    assert status == 0
    assert seen["report_format"] == "html"
    assert seen["report_html_file"] == source.with_name("sample_report.html")
    assert seen["conventions"] == "cf,ferret"


def test_run_check_ocean_cover_mode_routes_to_ocean_checker(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "sample.nc"
    xr.Dataset(
        data_vars={"v": (("time",), [1.0])},
        coords={"time": [0]},
    ).to_netcdf(source)

    seen: dict[str, object] = {}

    def _fake_ocean(
        ds: xr.Dataset,
        *,
        report_format: str = "tables",
        report_html_file: str | None = None,
        **kwargs: object,
    ) -> None:
        seen["report_format"] = report_format
        seen["report_html_file"] = report_html_file
        seen["kwargs"] = kwargs

    def _unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected checker call")

    monkeypatch.setattr(cli, "check_ocean_cover", _fake_ocean)
    monkeypatch.setattr(cli, "check_dataset_compliant", _unexpected)
    monkeypatch.setattr(cli, "check_time_cover", _unexpected)

    status = cli.run_check(["ocean-cover", str(source)])

    assert status == 0
    assert seen["report_format"] == "tables"
    assert seen["report_html_file"] is None
    assert seen["kwargs"] == {}


def test_run_check_all_mode_with_save_report_routes_all_with_html_paths(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "sample.nc"
    xr.Dataset(
        data_vars={"v": (("time",), [1.0])},
        coords={"time": [0]},
    ).to_netcdf(source)

    seen: dict[str, object] = {}

    def _fake_compliance(
        ds: xr.Dataset,
        *,
        report_format: str = "python",
        report_html_file: str | None = None,
        **kwargs: object,
    ) -> None:
        seen["compliance_format"] = report_format
        seen["compliance_file"] = report_html_file
        seen["compliance_conventions"] = kwargs.get("conventions")

    def _fake_ocean(
        ds: xr.Dataset,
        *,
        report_format: str = "tables",
        report_html_file: str | None = None,
        **kwargs: object,
    ) -> None:
        seen["ocean_format"] = report_format
        seen["ocean_file"] = report_html_file
        seen["ocean_kwargs"] = kwargs

    def _fake_time(
        ds: xr.Dataset,
        *,
        report_format: str = "tables",
        report_html_file: str | None = None,
        **kwargs: object,
    ) -> None:
        seen["time_format"] = report_format
        seen["time_file"] = report_html_file
        seen["time_kwargs"] = kwargs

    monkeypatch.setattr(cli, "check_dataset_compliant", _fake_compliance)
    monkeypatch.setattr(cli, "check_ocean_cover", _fake_ocean)
    monkeypatch.setattr(cli, "check_time_cover", _fake_time)

    status = cli.run_check(["all", str(source), "--save-report"])

    assert status == 0
    assert seen["compliance_format"] == "html"
    assert seen["ocean_format"] == "html"
    assert seen["time_format"] == "html"
    assert seen["compliance_file"] == source.with_name("sample_report.html")
    assert seen["ocean_file"] == source.with_name("sample_ocean_cover_report.html")
    assert seen["time_file"] == source.with_name("sample_time_cover_report.html")
    assert seen["compliance_conventions"] == "cf,ferret"
    assert seen["ocean_kwargs"] == {}
    assert seen["time_kwargs"] == {}
