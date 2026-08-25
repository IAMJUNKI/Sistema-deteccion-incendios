"""Orquestación reproducible de las fases dinámicas del datacubo histórico."""

import argparse
import shutil
from pathlib import Path

from src.config import (
    BOUNDARY_PATH,
    CORINE_PATH,
    DATACUBE_PATH,
    DATACUBE_START,
    EGIF_EVENTS_PATH,
    EGIF_METADATA_PATH,
    EGIF_TARGET_PATH,
    ERA5_DAILY_PATH,
    ERA5_RAW_DIR,
    GRID_CELL_SIZE_M,
    GRID_DIR,
    GRID_PATH,
    HUMAN_ACTIVITY_CUBE_PATH,
    HUMAN_ACTIVITY_RAW_DIR,
    LANDCOVER_CUBE_PATH,
    METEOROLOGY_CONTEXT_START,
    METEOROLOGY_CUBE_PATH,
    SPATIAL_CUBE_PATH,
    TABULAR_DATASET_DIR,
    TIME_CUBE_PATH,
    TOPOGRAPHY_CUBE_PATH,
)
from src.features.tabular import exportar_datacubo_tabular
from src.features.time import crear_datacubo_temporal, guardar_datacubo_temporal
from src.geospatial.human_activity import construir_capa_actividad_humana
from src.geospatial.pipeline import run_pipeline as construir_capas_estaticas
from src.ingestion.ingest_egif import ultima_fecha_egif
from src.ingestion.pipeline import ejecutar_pipeline_egif, ejecutar_pipeline_meteorologia
from src.pipeline import construir_datacubo_completo


def _limpiar_salidas_dinamicas(
    meteorology_cube_path: str | Path, datacube_path: str | Path, tabular_output_dir: str | Path
) -> None:
    """Elimina únicamente salidas derivadas que el workflow vuelve a generar."""
    for path in (Path(meteorology_cube_path), Path(datacube_path)):
        if path.exists():
            path.unlink()
    output_dir = Path(tabular_output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)


def ejecutar_workflow_historico(
    egif_xml: str | Path,
    grid_path: str | Path = GRID_PATH,
    spatial_cube_path: str | Path = SPATIAL_CUBE_PATH,
    time_cube_path: str | Path = TIME_CUBE_PATH,
    meteorology_cube_path: str | Path = METEOROLOGY_CUBE_PATH,
    target_path: str | Path = EGIF_TARGET_PATH,
    metadata_path: str | Path = EGIF_METADATA_PATH,
    datacube_path: str | Path = DATACUBE_PATH,
    tabular_output_dir: str | Path = TABULAR_DATASET_DIR,
    raw_meteorology_dir: str | Path = ERA5_RAW_DIR,
    daily_meteorology_path: str | Path = ERA5_DAILY_PATH,
    skip_daily_meteorology: bool = False,
    rebuild_static: bool = False,
    rebuild_human_activity: bool = False,
) -> None:
    """Construye tiempo, ERA5, EGIF, NetCDF y Parquet bajo un único contrato.

    La infraestructura estática no se recalcula aquí: DEM y CORINE son capas
    lentas y se generan una vez con ``src.geospatial.pipeline``. Este workflow
    consume esas capas validadas y orquesta todas las fases temporales.
    """
    if rebuild_static:
        construir_capas_estaticas(
            ruta_limites=BOUNDARY_PATH,
            ruta_clc=CORINE_PATH,
            ruta_salida=GRID_DIR / "galicia_grid_1km.parquet",
            ruta_cubo=SPATIAL_CUBE_PATH,
            ruta_vectorial=GRID_PATH,
            ruta_topografia=TOPOGRAPHY_CUBE_PATH,
            ruta_cobertura_suelo=LANDCOVER_CUBE_PATH,
            cell_size=GRID_CELL_SIZE_M,
        )

    if rebuild_static or rebuild_human_activity or not HUMAN_ACTIVITY_CUBE_PATH.exists():
        construir_capa_actividad_humana(
            grid_path=grid_path,
            spatial_cube_path=spatial_cube_path,
            output_path=HUMAN_ACTIVITY_CUBE_PATH,
            raw_dir=HUMAN_ACTIVITY_RAW_DIR,
        )

    required_static = [
        grid_path,
        spatial_cube_path,
        TOPOGRAPHY_CUBE_PATH,
        LANDCOVER_CUBE_PATH,
        HUMAN_ACTIVITY_CUBE_PATH,
    ]
    missing_static = [str(path) for path in required_static if not Path(path).exists()]
    if missing_static:
        raise FileNotFoundError(
            "Faltan capas estáticas: " + ", ".join(missing_static) + ". Usa --rebuild-static."
        )

    time_cube_path = Path(time_cube_path)
    end_date = ultima_fecha_egif(egif_xml).date().isoformat()
    _limpiar_salidas_dinamicas(meteorology_cube_path, datacube_path, tabular_output_dir)
    guardar_datacubo_temporal(crear_datacubo_temporal(DATACUBE_START, end_date), time_cube_path)
    ejecutar_pipeline_meteorologia(
        spatial_cube_path=spatial_cube_path,
        grid_path=grid_path,
        raw_dir=raw_meteorology_dir,
        daily_output_path=daily_meteorology_path,
        meteorology_cube_path=meteorology_cube_path,
        start_date=METEOROLOGY_CONTEXT_START,
        end_date=end_date,
        skip_daily=skip_daily_meteorology,
    )
    ejecutar_pipeline_egif(
        xml_path=egif_xml,
        grid_path=grid_path,
        events_output_path=EGIF_EVENTS_PATH,
        target_output_path=target_path,
        metadata_output_path=metadata_path,
        end_date=end_date,
    )
    construir_datacubo_completo(
        spatial_cube_path=spatial_cube_path,
        topography_cube_path=TOPOGRAPHY_CUBE_PATH,
        landcover_cube_path=LANDCOVER_CUBE_PATH,
        human_activity_cube_path=HUMAN_ACTIVITY_CUBE_PATH,
        time_cube_path=time_cube_path,
        meteorology_cube_path=meteorology_cube_path,
        egif_target_path=target_path,
        output_path=datacube_path,
        start_date=DATACUBE_START,
        end_date=end_date,
    )
    exportar_datacubo_tabular(datacube_path, tabular_output_dir)


def main() -> None:
    """Ejecuta la ruta reproducible desde las fuentes locales ya descargadas."""
    parser = argparse.ArgumentParser(description="Orquesta el datacubo histórico EGIF + ERA5.")
    parser.add_argument("--egif-xml", required=True)
    parser.add_argument("--grid", default=str(GRID_PATH))
    parser.add_argument("--spatial-cube", default=str(SPATIAL_CUBE_PATH))
    parser.add_argument("--rebuild-static", action="store_true")
    parser.add_argument(
        "--rebuild-human-activity",
        action="store_true",
        help="Recalcula las variables OSM de carreteras y zonas residenciales.",
    )
    parser.add_argument("--skip-daily-meteorology", action="store_true")
    args = parser.parse_args()
    ejecutar_workflow_historico(
        egif_xml=args.egif_xml,
        grid_path=args.grid,
        spatial_cube_path=args.spatial_cube,
        skip_daily_meteorology=args.skip_daily_meteorology,
        rebuild_static=args.rebuild_static,
        rebuild_human_activity=args.rebuild_human_activity,
    )


if __name__ == "__main__":
    main()
