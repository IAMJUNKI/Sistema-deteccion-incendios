"""Descarga reproducible de ERA5-Land horario para Galicia mediante CDS."""

import argparse
import calendar
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_START_DATE = date(2018, 12, 1)
DEFAULT_END_DATE = date(2023, 12, 31)
DATASET = "reanalysis-era5-land"
GALICIA_AREA = [43.8, -9.3, 41.8, -6.7]
ERA5_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "surface_solar_radiation_downwards",
]


def iterar_meses(start_date: date, end_date: date) -> list[tuple[int, int]]:
    """Devuelve los meses incluidos en un intervalo de fechas inclusivo."""
    if start_date > end_date:
        raise ValueError("La fecha inicial no puede ser posterior a la final.")

    months: list[tuple[int, int]] = []
    year, month = start_date.year, start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def crear_peticion_mensual(year: int, month: int) -> dict[str, list[str] | list[float] | str]:
    """Crea una petición CDS de ERA5-Land para un mes completo de Galicia."""
    n_days = calendar.monthrange(year, month)[1]
    return {
        "variable": ERA5_VARIABLES,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{day:02d}" for day in range(1, n_days + 1)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": GALICIA_AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def crear_cliente_cds() -> object:
    """Crea el cliente CDS usando el token del archivo ``.env``.

    Raises:
        RuntimeError: Si falta el token personal de Copernicus CDS.
    """
    load_dotenv()
    api_key = os.getenv("COPERNICUS_CDS_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise RuntimeError(
            "Falta COPERNICUS_CDS_API_KEY. Copia tu token desde "
            "https://cds.climate.copernicus.eu/profile al archivo .env."
        )

    try:
        import cdsapi
    except ImportError as error:
        raise RuntimeError("Falta cdsapi. Instala el entorno definido en environment.yml.") from error

    url = os.getenv("COPERNICUS_CDS_API_URL", "https://cds.climate.copernicus.eu/api")
    return cdsapi.Client(url=url, key=api_key)


def descargar_era5_land(
    output_dir: str | Path,
    start_date: date = DEFAULT_START_DATE,
    end_date: date = DEFAULT_END_DATE,
) -> list[Path]:
    """Descarga un NetCDF por mes para el intervalo solicitado.

    Los ficheros ya existentes no se vuelven a solicitar, para que la descarga
    pueda reanudarse sin repetir peticiones ni sobrescribir datos originales.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = crear_cliente_cds()
    downloaded: list[Path] = []

    for year, month in iterar_meses(start_date, end_date):
        output_path = output_dir / f"era5_land_galicia_{year}_{month:02d}.nc"
        if output_path.exists():
            print(f"Ya existe, se omite: {output_path}")
            continue
        print(f"Solicitando ERA5-Land {year}-{month:02d}...")
        client.retrieve(DATASET, crear_peticion_mensual(year, month), str(output_path))
        downloaded.append(output_path)
    return downloaded


def main() -> None:
    """Ejecuta la descarga mensual desde la línea de comandos."""
    parser = argparse.ArgumentParser(description="Descarga ERA5-Land horario para Galicia.")
    parser.add_argument("--output-dir", default="data/raw/meteorology/era5")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    parser.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())
    args = parser.parse_args()
    descargar_era5_land(
        args.output_dir,
        date.fromisoformat(args.start_date),
        date.fromisoformat(args.end_date),
    )


if __name__ == "__main__":
    main()
