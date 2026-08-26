"""Variables de calendario para el periodo histórico del TFM.

El cubo temporal es deliberadamente unidimensional: las variables de calendario
solo dependen de la fecha y se combinarán con la malla espacial al construir las
capas dinámicas y el dataset maestro.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.config import DATACUBE_END, DATACUBE_START

DEFAULT_START_DATE = DATACUBE_START
DEFAULT_END_DATE = DATACUBE_END
TIME_VARIABLES = [
    "year",
    "month",
    "iso_week",
    "day_of_year",
    "day_of_week",
    "is_weekend",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
]
# El calendario se deriva de la coordenada ``time`` cuando haga falta (por
# ejemplo, para particionar Parquet); no se almacena por defecto como señal.
DATACUBE_VARIABLE_FLAGS = {name: True for name in TIME_VARIABLES}
TEST_DATACUBE_VARIABLE_FLAGS = {name: False for name in TIME_VARIABLES}

VARIABLE_METADATA = {
    "year": "Calendar year. Keep for audit and temporal splits; do not use as a model predictor.",
    "month": "Calendar month, from 1 to 12.",
    "iso_week": "ISO-8601 week number, from 1 to 53.",
    "day_of_year": "Ordinal day within the calendar year.",
    "day_of_week": "Day of week, Monday=0 and Sunday=6.",
    "is_weekend": "Weekend indicator, 1 for Saturday or Sunday.",
    "day_of_year_sin": "Cyclical sine encoding of the day of year.",
    "day_of_year_cos": "Cyclical cosine encoding of the day of year.",
    "month_sin": "Cyclical sine encoding of the month.",
    "month_cos": "Cyclical cosine encoding of the month.",
}


def crear_datacubo_temporal(
    start_date: str | pd.Timestamp = DEFAULT_START_DATE,
    end_date: str | pd.Timestamp = DEFAULT_END_DATE,
    inclusion_flags: dict[str, bool] | None = None,
) -> xr.Dataset:
    """Crea las variables temporales diarias entre dos fechas inclusivas.

    Args:
        start_date: Primera fecha diaria del cubo.
        end_date: Última fecha diaria del cubo.

    Returns:
        Dataset xarray de dimensión ``time`` con variables de calendario.
    """
    time = pd.date_range(start_date, end_date, freq="D")
    if time.empty:
        raise ValueError("El rango temporal no contiene días.")

    day_of_year = time.dayofyear.to_numpy(dtype=np.int16)
    month = time.month.to_numpy(dtype=np.int8)
    dataset = xr.Dataset(coords={"time": time})
    flags = DATACUBE_VARIABLE_FLAGS if inclusion_flags is None else inclusion_flags
    dataset["year"] = ("time", time.year.to_numpy(dtype=np.int16))
    dataset["month"] = ("time", month)
    dataset["iso_week"] = ("time", time.isocalendar().week.to_numpy(dtype=np.int8))
    dataset["day_of_year"] = ("time", day_of_year)
    dataset["day_of_week"] = ("time", time.dayofweek.to_numpy(dtype=np.int8))
    dataset["is_weekend"] = ("time", np.asarray(time.dayofweek >= 5, dtype=np.uint8))
    dataset["day_of_year_sin"] = (
        "time",
        np.sin(2 * np.pi * day_of_year / 365.25).astype(np.float32),
    )
    dataset["day_of_year_cos"] = (
        "time",
        np.cos(2 * np.pi * day_of_year / 365.25).astype(np.float32),
    )
    dataset["month_sin"] = ("time", np.sin(2 * np.pi * month / 12).astype(np.float32))
    dataset["month_cos"] = ("time", np.cos(2 * np.pi * month / 12).astype(np.float32))

    dataset.attrs = {
        "title": "Daily temporal variables",
        "description": f"Calendar predictors for the {time.min().date()} to {time.max().date()} modelling period.",
        "module": "time",
        "temporal_resolution": "1 day",
        "prediction_period": f"{time.min().date()} to {time.max().date()}",
    }
    dataset.time.attrs = {"long_name": "Date"}
    for variable in TIME_VARIABLES:
        dataset[variable].attrs = {"long_name": VARIABLE_METADATA[variable], "units": "-"}
    excluded = [name for name in TIME_VARIABLES if not flags.get(name, False)]
    if excluded:
        dataset = dataset.drop_vars(excluded)
    return dataset


def guardar_datacubo_temporal(dataset: xr.Dataset, ruta_salida: str | Path) -> None:
    """Guarda el cubo temporal en formato NetCDF."""
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(ruta_salida)


def main() -> None:
    """Genera ``time.nc`` desde la línea de comandos."""
    parser = argparse.ArgumentParser(description="Crea el cubo temporal diario del TFM.")
    parser.add_argument(
        "--start-date", default=DEFAULT_START_DATE, help="Fecha inicial YYYY-MM-DD."
    )
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Fecha final YYYY-MM-DD.")
    parser.add_argument(
        "--output", default="data/processed/time.nc", help="Ruta NetCDF de salida."
    )
    args = parser.parse_args()
    guardar_datacubo_temporal(crear_datacubo_temporal(args.start_date, args.end_date), args.output)


if __name__ == "__main__":
    main()
