from pathlib import Path

import xarray as xr

from nc_check import standard_names


def test_unknown_standard_name_is_reported() -> None:
    ds = xr.Dataset(
        data_vars={"sst": (("time",), [290.0])},
        coords={"time": [0]},
    )
    ds["sst"].attrs["standard_name"] = "not_a_real_standard_name"
    ds["sst"].attrs["units"] = "K"

    issues = {"suggestions": {"variables": {}}, "notes": []}
    table = str(Path("tests/data/cf-standard-name-table.xml"))
    standard_names.augment_issues_with_standard_name_suggestions(ds, issues, table)

    assert (
        issues["suggestions"]["variables"]["sst"]["unknown_standard_name"]
        == "not_a_real_standard_name"
    )


def test_units_synonym_kelvin_is_accepted() -> None:
    ds = xr.Dataset(
        data_vars={"sst": (("time",), [290.0])},
        coords={"time": [0]},
    )
    ds["sst"].attrs["standard_name"] = "sea_surface_temperature"
    ds["sst"].attrs["units"] = "kelvin"

    issues = {"suggestions": {"variables": {}}, "notes": []}
    table = str(Path("tests/data/cf-standard-name-table.xml"))
    standard_names.augment_issues_with_standard_name_suggestions(ds, issues, table)

    assert "sst" not in issues["suggestions"]["variables"]


def test_bad_standard_name_table_adds_note() -> None:
    ds = xr.Dataset(
        data_vars={"v": (("time",), [1.0])},
        coords={"time": [0]},
    )

    issues = {"suggestions": {"variables": {}}, "notes": []}
    standard_names.augment_issues_with_standard_name_suggestions(
        ds,
        issues,
        "tests/data/does-not-exist.xml",
    )

    assert issues["notes"]
    assert "Could not read CF standard-name table" in issues["notes"][0]
