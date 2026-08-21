import numpy as np
import pandas as pd
import xarray as xr

from src.ingestion.meteorology import crear_meteorologia_diaria


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
    assert 0 <= daily["relative_humidity_min"].item() <= 100

