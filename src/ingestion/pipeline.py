"""Orquestación reproducible de la ingesta meteorológica histórica."""

import argparse
from pathlib import Path

from src.ingestion.meteorology import (
    DEFAULT_END_DATE,
    DEFAULT_OUTPUT,
    DEFAULT_RAW_DIR,
    DEFAULT_START_DATE,
    interpolar_al_grid,
    procesar_era5,
)

DEFAULT_METEOROLOGY_CUBE = Path("data/processed/meteorology_2018_2023.nc")


def ejecutar_pipeline_meteorologia(
    spatial_cube_path: str | Path,
    grid_path: str | Path,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    daily_output_path: str | Path = DEFAULT_OUTPUT,
    meteorology_cube_path: str | Path = DEFAULT_METEOROLOGY_CUBE,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    skip_daily: bool = False,
) -> None:
    """Procesa ERA5-Land y lo incorpora al cubo espacial de 1 km.

    No descarga datos: los NetCDF horarios deben estar previamente en
    ``data/raw/meteorology/era5``. Esto permite repetir el pipeline sin hacer
    peticiones a Copernicus y conserva la procedencia de los datos originales.

    Args:
        spatial_cube_path: Cubo estático con coordenadas ``x``, ``y`` e
            ``is_galicia``.
        grid_path: Rejilla vectorial con ``cell_id`` e ``is_galicia``.
        raw_dir: Directorio de NetCDF horarios mensuales de ERA5-Land.
        daily_output_path: Destino del NetCDF diario intermedio.
        meteorology_cube_path: Destino del cubo meteorológico interpolado.
        start_date: Fecha inicial inclusiva, en formato ISO.
        end_date: Fecha final inclusiva, en formato ISO.
        skip_daily: Reutiliza el NetCDF diario existente si es válido.
    """
    daily_output_path = Path(daily_output_path)
    if not skip_daily:
        procesar_era5(raw_dir, daily_output_path, start_date, end_date)
    elif not daily_output_path.exists():
        raise FileNotFoundError(
            f"No se puede omitir el procesado diario: no existe {daily_output_path}."
        )

    interpolar_al_grid(
        daily_output_path,
        spatial_cube_path,
        grid_path,
        meteorology_cube_path,
        start_date,
        end_date,
    )


def main() -> None:
    """Ejecuta el pipeline meteorológico histórico desde la línea de comandos."""
    parser = argparse.ArgumentParser(description="Procesa e interpola ERA5-Land sobre la rejilla.")
    parser.add_argument("--spatial-cube", required=True, help="Cubo estático con la máscara Galicia.")
    parser.add_argument("--grid", required=True, help="Rejilla vectorial de 1 km.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--daily-output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output", default=str(DEFAULT_METEOROLOGY_CUBE))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--skip-daily",
        action="store_true",
        help="Reutiliza el NetCDF diario ya procesado sin recalcularlo.",
    )
    args = parser.parse_args()
    ejecutar_pipeline_meteorologia(
        spatial_cube_path=args.spatial_cube,
        grid_path=args.grid,
        raw_dir=args.raw_dir,
        daily_output_path=args.daily_output,
        meteorology_cube_path=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        skip_daily=args.skip_daily,
    )


if __name__ == "__main__":
    main()
