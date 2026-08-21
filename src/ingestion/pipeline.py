"""Orquestación reproducible de la ingesta histórica del proyecto."""

import argparse
from pathlib import Path
from typing import Any

from src.ingestion.ingest_egif import (
    DEFAULT_EVENTS_OUTPUT,
    DEFAULT_METADATA_OUTPUT,
    DEFAULT_TARGET_OUTPUT,
    procesar_egif,
)
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
    """Procesa ERA5-Land y lo incorpora al cubo espacial de 1 km."""
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


def ejecutar_pipeline_egif(
    xml_path: str | Path,
    grid_path: str | Path,
    events_output_path: str | Path = DEFAULT_EVENTS_OUTPUT,
    target_output_path: str | Path = DEFAULT_TARGET_OUTPUT,
    metadata_output_path: str | Path = DEFAULT_METADATA_OUTPUT,
    start_date: str = "2018-01-01",
    end_date: str = DEFAULT_END_DATE,
) -> dict[str, Any]:
    """Genera eventos EGIF y el target por celda y día sobre la rejilla actual."""
    return procesar_egif(
        xml_path,
        grid_path,
        events_output_path,
        target_output_path,
        metadata_output_path,
        start_date,
        end_date,
    )


def main() -> None:
    """Ejecuta las fases de meteorología y/o EGIF desde la línea de comandos."""
    parser = argparse.ArgumentParser(description="Procesa la ingesta histórica sobre la rejilla.")
    parser.add_argument("--spatial-cube", help="Cubo estático con la máscara Galicia.")
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
    parser.add_argument(
        "--skip-meteorology",
        action="store_true",
        help="Ejecuta exclusivamente la fase EGIF.",
    )
    parser.add_argument("--egif-xml", help="XML oficial de EGIF para incorporar la fase de incendios.")
    parser.add_argument("--egif-events-output", default=str(DEFAULT_EVENTS_OUTPUT))
    parser.add_argument("--egif-target-output", default=str(DEFAULT_TARGET_OUTPUT))
    parser.add_argument("--egif-metadata-output", default=str(DEFAULT_METADATA_OUTPUT))
    parser.add_argument("--egif-start-date", default="2018-01-01")
    args = parser.parse_args()

    if not args.skip_meteorology:
        if not args.spatial_cube:
            parser.error("--spatial-cube es obligatorio salvo que se use --skip-meteorology.")
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

    if args.egif_xml:
        metadata = ejecutar_pipeline_egif(
            args.egif_xml,
            args.grid,
            args.egif_events_output,
            args.egif_target_output,
            args.egif_metadata_output,
            args.egif_start_date,
            args.end_date,
        )
        print(f"EGIF: {metadata}")
    elif args.skip_meteorology:
        parser.error("Indica --egif-xml al usar --skip-meteorology.")


if __name__ == "__main__":
    main()
