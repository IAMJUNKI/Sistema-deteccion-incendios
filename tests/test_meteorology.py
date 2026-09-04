import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from src.ingestion.meteorology import (
    _anadir_consecutive_dry_days_al_grid,
    add_meteorology_accumulations,
    crear_meteorologia_diaria,
    interpolar_al_grid,
)


def test_crear_meteorologia_diaria_convierte_unidades_y_acumulado() -> None:
    time = pd.date_range("2020-01-01", periods=24, freq="h")
    shape = (24, 1, 1)
    dataset = xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), np.full(shape, 293.15)),
            "d2m": (("time", "latitude", "longitude"), np.full(shape, 283.15)),
            "u10": (("time", "latitude", "longitude"), np.full(shape, 3.0)),
            "v10": (("time", "latitude", "longitude"), np.full(shape, 4.0)),
            "tp": (("time", "latitude", "longitude"), np.arange(24).reshape(shape) / 1000),
        },
        coords={"time": time, "latitude": [42.0], "longitude": [-8.0]},
    )

    daily = crear_meteorologia_diaria(dataset)

    assert len(daily.time) == 1
    assert np.isclose(daily["temperature_mean"].item(), 20.0)
    assert np.isclose(daily["wind_speed_max"].item(), 18.0)
    assert np.isclose(daily["precipitation_sum"].item(), 23.0)
    assert "precipitation_sum_1d" not in daily
    assert 0 <= daily["relative_humidity_min"].item() <= 100
    assert daily["temperature_mean"].attrs["units"] == "degC"
    assert daily["temperature_mean"].attrs["long_name"] == "Daily mean 2 m air temperature"
    assert "consecutive_dry_days" not in daily


def test_acumulados_incluyen_el_dia_observado_sin_shift() -> None:
    time = pd.date_range("2020-01-01", periods=31, freq="D")
    daily = xr.Dataset(
        {
            "precipitation_sum": (
                ("time", "latitude", "longitude"),
                np.array([0.0] * 30 + [2.0]).reshape(31, 1, 1),
            )
        },
        coords={"time": time, "latitude": [42.0], "longitude": [-8.0]},
    )

    result = add_meteorology_accumulations(daily)

    assert result["precipitation_sum_30d"].sel(time="2020-01-30").item() == 0.0
    assert result["precipitation_sum_3d"].sel(time="2020-01-31").item() == 2.0
    assert "precipitation_sum_1d" not in result
    assert result["consecutive_dry_days"].sel(time="2020-01-30").item() == 30.0
    assert result["consecutive_dry_days"].sel(time="2020-01-31").item() == 0.0


def test_medias_moviles_meteorologicas_incluyen_el_dia_observado() -> None:
    time = pd.date_range("2020-01-01", periods=14, freq="D")
    values = np.arange(14, dtype=np.float32).reshape(14, 1, 1)
    daily = xr.Dataset(
        {
            "precipitation_sum": (("time", "latitude", "longitude"), values),
            "temperature_mean": (("time", "latitude", "longitude"), values),
            "relative_humidity_mean": (("time", "latitude", "longitude"), values + 50),
            "wind_speed_mean": (("time", "latitude", "longitude"), values + 10),
        },
        coords={"time": time, "latitude": [42.0], "longitude": [-8.0]},
    )

    result = add_meteorology_accumulations(daily)

    assert result["temperature_mean_7d"].sel(time="2020-01-07").item() == 3.0
    assert result["relative_humidity_mean_7d"].sel(time="2020-01-07").item() == 53.0
    assert result["wind_speed_mean_7d"].sel(time="2020-01-07").item() == 13.0
    assert result["relative_humidity_mean_14d"].sel(time="2020-01-14").item() == 56.5
    assert result["wind_speed_mean_7d"].attrs["units"] == "km h-1"
    assert result["relative_humidity_mean_14d"].attrs["units"] == "%"


def test_racha_seca_se_calcula_en_grid_final_como_entero(tmp_path) -> None:
    output = tmp_path / "meteorologia_grid.nc"
    xr.Dataset(
        {
            "is_galicia": (("y", "x"), np.array([[1, 0], [1, 1]], dtype=np.uint8)),
            "precipitation_sum": (
                ("time", "y", "x"),
                np.array(
                    [
                        [[0.0, np.nan], [0.5, 2.0]],
                        [[0.0, np.nan], [2.0, 0.0]],
                        [[3.0, np.nan], [0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
            ),
        },
        coords={"time": pd.date_range("2020-01-01", periods=3), "y": [0, 1], "x": [0, 1]},
    ).to_netcdf(output)

    _anadir_consecutive_dry_days_al_grid(output)

    with xr.open_dataset(output) as result:
        np.testing.assert_array_equal(
            result["consecutive_dry_days"].values[:, [0, 1, 1], [0, 0, 1]],
            np.array([[1, 1, 0], [2, 0, 1], [0, 1, 2]], dtype=np.float32),
        )
        assert result["consecutive_dry_days"].attrs["spatial_derivation"] == "target_grid_precipitation"


def test_interpolar_al_grid_por_bloques_conserva_valores_y_racha_seca(tmp_path) -> None:
    """La escritura mensual conserva valores al atravesar más de un bloque."""
    time = pd.date_range("2020-01-01", periods=40, freq="D")
    latitude = [42.0, 43.0]
    longitude = [-9.0, -8.0]
    base = np.arange(40, dtype=np.float32).reshape(40, 1, 1)
    offsets = np.array([[0.0, 1.0], [10.0, 11.0]], dtype=np.float32)
    temperature = base + offsets
    precipitation = np.zeros((40, 2, 2), dtype=np.float32)
    precipitation[5, :, :] = 2.0

    daily_path = tmp_path / "daily.nc"
    xr.Dataset(
        {
            "temperature_mean": (("time", "latitude", "longitude"), temperature),
            "precipitation_sum": (("time", "latitude", "longitude"), precipitation),
        },
        coords={"time": time, "latitude": latitude, "longitude": longitude},
    ).to_netcdf(daily_path)

    cube_path = tmp_path / "spatial.nc"
    xr.Dataset(
        {"is_galicia": (("y", "x"), np.ones((2, 2), dtype=np.uint8))},
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
    ).to_netcdf(cube_path)

    grid_path = tmp_path / "grid.gpkg"
    geometry = gpd.GeoSeries(
        [Point(-9, 42), Point(-8, 42), Point(-9, 43), Point(-8, 43)], crs="EPSG:4326"
    ).to_crs("EPSG:3035")
    gpd.GeoDataFrame(
        {"cell_id": [0, 1, 2, 3], "is_galicia": [1, 1, 1, 1]},
        geometry=geometry,
    ).to_file(grid_path, driver="GPKG")

    output_path = tmp_path / "grid_meteorology.nc"
    interpolar_al_grid(daily_path, cube_path, grid_path, output_path)

    with xr.open_dataset(output_path) as result:
        np.testing.assert_array_equal(result["temperature_mean"].values, temperature)
        assert result["temperature_mean"].attrs["spatial_interpolation"].startswith("Linear")
        assert result["consecutive_dry_days"].isel(time=4, y=0, x=0).item() == 5
        assert result["consecutive_dry_days"].isel(time=5, y=0, x=0).item() == 0
        assert result["consecutive_dry_days"].isel(time=39, y=0, x=0).item() == 34
