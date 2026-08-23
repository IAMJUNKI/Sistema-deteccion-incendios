"""Daily ERA5-Land processing for the Galicia wildfire data cube."""

import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from src.config import DATACUBE_END, ERA5_DAILY_PATH, METEOROLOGY_CONTEXT_START

DEFAULT_START_DATE = METEOROLOGY_CONTEXT_START
DEFAULT_END_DATE = DATACUBE_END
DEFAULT_RAW_DIR = Path("data/raw/meteorology/era5")
DEFAULT_OUTPUT = ERA5_DAILY_PATH
FILE_PATTERN = re.compile(r"era5_?land_galicia_(\d{4})_(\d{2})\.nc$")

DAILY_VARIABLES = (
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "temperature_max_12_18h",
    "relative_humidity_mean",
    "relative_humidity_min",
    "relative_humidity_min_12_18h",
    "wind_speed_mean",
    "wind_speed_max",
    "wind_speed_max_12_18h",
    "precipitation_sum",
    "precipitation_sum_3d",
    "precipitation_sum_7d",
    "precipitation_sum_14d",
    "precipitation_sum_30d",
    "consecutive_dry_days",
    "temperature_mean_7d",
    "relative_humidity_mean_7d",
)
PRECIPITATION_WINDOWS = (3, 7, 14, 30)
DRY_DAY_THRESHOLD_MM = 1.0


def listar_archivos_era5(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> list[Path]:
    """Find the required monthly ERA5-Land files and reject missing months."""
    start_period = pd.Period(start_date, freq="M")
    end_period = pd.Period(end_date, freq="M")
    files_by_month: dict[pd.Period, Path] = {}

    for path in Path(raw_dir).glob("*.nc"):
        match = FILE_PATTERN.fullmatch(path.name)
        if not match:
            continue
        period = pd.Period(f"{match.group(1)}-{match.group(2)}", freq="M")
        if start_period <= period <= end_period:
            if period in files_by_month:
                raise ValueError(f"Duplicate ERA5-Land file for {period}.")
            files_by_month[period] = path

    expected = pd.period_range(start_period, end_period, freq="M")
    missing = expected.difference(pd.PeriodIndex(files_by_month))
    if len(missing):
        raise FileNotFoundError("Missing ERA5-Land months: " + ", ".join(map(str, missing)))
    return [files_by_month[period] for period in expected]


def cargar_era5_horario(files: list[Path]) -> xr.Dataset:
    """Load all monthly files without requiring Dask."""
    datasets = [xr.open_dataset(path) for path in files]
    # El archivo de diciembre de 2018 puede tener un borde geográfico ligeramente
    # distinto de los ficheros históricos. Conservamos solo la malla común para
    # que ninguna coordenada quede vacía en parte de la serie temporal.
    dataset = xr.concat(datasets, dim="valid_time", join="inner").sortby("valid_time")
    dataset = dataset.rename({"valid_time": "time"})
    missing = {"t2m", "d2m", "u10", "v10", "tp"} - set(dataset.data_vars)
    if missing:
        dataset.close()
        raise ValueError(f"Missing ERA5-Land variables: {sorted(missing)}")
    return dataset


def _daily(data: xr.DataArray, operation: str) -> xr.DataArray:
    result = getattr(data.groupby("time.date"), operation)()
    return result.rename({"date": "time"}).assign_coords(time=pd.to_datetime(result.date.values))


def add_meteorology_accumulations(daily: xr.Dataset) -> xr.Dataset:
    """Añade acumulados de lluvia y rachas secas, incluyendo el día observado.

    No aplica desplazamiento temporal: por ejemplo, ``precipitation_sum_30d``
    para la fecha T contiene la precipitación de T-29 a T, ambas inclusive.
    """
    output = daily.copy()
    precipitation = output["precipitation_sum"]
    for window in PRECIPITATION_WINDOWS:
        name = f"precipitation_sum_{window}d"
        output[name] = (
            precipitation.rolling(time=window, min_periods=window).sum().astype(np.float32)
        )
        output[name].attrs = {
            "long_name": f"Precipitation accumulated over {window} days",
            "units": "mm",
            "time_contract": f"Includes the observation date and the preceding {window - 1} days.",
        }

    for variable in ("temperature_mean", "relative_humidity_mean"):
        if variable not in output:
            continue
        name = f"{variable}_7d"
        output[name] = output[variable].rolling(time=7, min_periods=7).mean().astype(np.float32)
        output[name].attrs = {
            "long_name": f"Seven-day mean of {variable}",
            "time_contract": "Includes the observation date and the preceding 6 days.",
        }

    values = precipitation.values
    dry_days = np.zeros(values.shape, dtype=np.float32)
    streak = np.zeros(values.shape[1:], dtype=np.float32)
    for time_index in range(values.shape[0]):
        is_dry = np.isfinite(values[time_index]) & (values[time_index] < DRY_DAY_THRESHOLD_MM)
        streak = np.where(is_dry, streak + 1, 0.0)
        dry_days[time_index] = np.where(np.isfinite(values[time_index]), streak, np.nan)
    output["consecutive_dry_days"] = (precipitation.dims, dry_days)
    output["consecutive_dry_days"].attrs = {
        "long_name": "Consecutive dry days",
        "units": "days",
        "dry_day_threshold": f"precipitation_sum < {DRY_DAY_THRESHOLD_MM} mm",
        "time_contract": "Includes the observation date; no temporal shift applied.",
    }
    return output


def crear_meteorologia_diaria(hourly: xr.Dataset) -> xr.Dataset:
    """Create daily observations. No temporal shift is applied."""
    temperature = hourly["t2m"] - 273.15
    dewpoint = hourly["d2m"] - 273.15
    humidity = (
        100
        * np.exp(17.625 * dewpoint / (243.04 + dewpoint))
        / np.exp(17.625 * temperature / (243.04 + temperature))
    )
    humidity = humidity.clip(min=0, max=100)
    wind = np.hypot(hourly["u10"], hourly["v10"]) * 3.6
    local_time = (
        pd.DatetimeIndex(hourly.time.values).tz_localize("UTC").tz_convert("Europe/Madrid")
    )
    local_hour = xr.DataArray(local_time.hour, dims="time", coords={"time": hourly.time})
    critical = (local_hour >= 12) & (local_hour <= 18)

    output = xr.Dataset(
        {
            "temperature_mean": _daily(temperature, "mean"),
            "temperature_min": _daily(temperature, "min"),
            "temperature_max": _daily(temperature, "max"),
            "temperature_max_12_18h": _daily(temperature.where(critical), "max"),
            "relative_humidity_mean": _daily(humidity, "mean"),
            "relative_humidity_min": _daily(humidity, "min"),
            "relative_humidity_min_12_18h": _daily(humidity.where(critical), "min"),
            "wind_speed_mean": _daily(wind, "mean"),
            "wind_speed_max": _daily(wind, "max"),
            "wind_speed_max_12_18h": _daily(wind.where(critical), "max"),
            "precipitation_sum": _daily(hourly["tp"] * 1000, "max"),
        }
    ).astype(np.float32)
    output = add_meteorology_accumulations(output)
    output.attrs = {
        "title": "Daily ERA5-Land meteorology for Galicia",
        "source": "Copernicus CDS reanalysis-era5-land",
        "time_contract": "Variables describe the same observation date; no temporal shift applied.",
        "critical_window": "12:00-18:00 Europe/Madrid",
        "precipitation_method": "Daily maximum of ERA5-Land hourly accumulation in mm.",
    }
    for name in DAILY_VARIABLES:
        output[name].attrs["long_name"] = name.replace("_", " ")
    for name in (
        "temperature_mean",
        "temperature_min",
        "temperature_max",
        "temperature_max_12_18h",
        "temperature_mean_7d",
    ):
        output[name].attrs["units"] = "degC"
    for name in (
        "relative_humidity_mean",
        "relative_humidity_min",
        "relative_humidity_min_12_18h",
        "relative_humidity_mean_7d",
    ):
        output[name].attrs["units"] = "%"
    for name in ("wind_speed_mean", "wind_speed_max", "wind_speed_max_12_18h"):
        output[name].attrs["units"] = "km h-1"
    output["precipitation_sum"].attrs["units"] = "mm"
    return output


def procesar_era5(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    output_path: str | Path = DEFAULT_OUTPUT,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> None:
    """Process all raw files into a compact daily NetCDF."""
    hourly = cargar_era5_horario(listar_archivos_era5(raw_dir, start_date, end_date))
    try:
        daily = crear_meteorologia_diaria(hourly)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoding = {
            name: {"zlib": True, "complevel": 4, "dtype": "float32"} for name in daily.data_vars
        }
        daily.to_netcdf(destination, encoding=encoding)
    finally:
        hourly.close()


def interpolar_al_grid(
    daily_path: str | Path,
    cube_path: str | Path,
    grid_path: str | Path,
    output_path: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """Interpola ERA5 diario al grid 1 km, escribiendo una variable cada vez."""
    with xr.open_dataset(daily_path) as daily, xr.open_dataset(cube_path) as cube:
        if start_date or end_date:
            daily = daily.sel(time=slice(start_date, end_date))
        grid = gpd.read_file(grid_path)
        active = grid.loc[grid["is_galicia"] == 1].copy()
        points = gpd.GeoSeries(active.geometry.centroid, crs=active.crs).to_crs("EPSG:4326")
        latitude = xr.DataArray(points.y.to_numpy(), dims="cell")
        longitude = xr.DataArray(points.x.to_numpy(), dims="cell")
        rows = (active["cell_id"].to_numpy() // cube.sizes["x"]).astype(int)
        cols = (active["cell_id"].to_numpy() % cube.sizes["x"]).astype(int)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            with xr.open_dataset(destination) as existing:
                completed = set(existing.data_vars)
        else:
            completed = set()
            xr.Dataset(
                {"is_galicia": cube["is_galicia"]},
                coords={"time": daily.time, "y": cube.y, "x": cube.x},
            ).to_netcdf(destination)

        for name in daily.data_vars:
            if name in completed:
                continue
            source = (
                daily[name]
                .dropna(dim="latitude", how="all")
                .dropna(dim="longitude", how="all")
                .sortby("latitude")
                .sortby("longitude")
            )
            linear = source.interp(latitude=latitude, longitude=longitude, method="linear")
            # ERA5-Land contiene píxeles marítimos nulos. Antes de pedir el vecino
            # más cercano, se rellenan esos huecos espaciales con el píxel terrestre
            # válido más próximo para que las celdas costeras de Galicia no queden
            # fuera del dominio tabular por falta de meteorología.
            nearest_source = source.interpolate_na(
                dim="latitude", method="nearest", fill_value="extrapolate"
            ).interpolate_na(dim="longitude", method="nearest", fill_value="extrapolate")
            nearest = nearest_source.sel(latitude=latitude, longitude=longitude, method="nearest")
            values = linear.where(linear.notnull(), nearest).values.astype(np.float32)
            full = np.full(
                (len(daily.time), cube.sizes["y"], cube.sizes["x"]), np.nan, dtype=np.float32
            )
            full[:, rows, cols] = values
            xr.Dataset({name: (("time", "y", "x"), full)}).to_netcdf(
                destination,
                mode="a",
                encoding={name: {"zlib": True, "complevel": 4, "dtype": "float32"}},
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process hourly ERA5-Land into daily meteorology."
    )
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    args = parser.parse_args()
    procesar_era5(args.raw_dir, args.output, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
