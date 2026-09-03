"""Orquestación reproducible de las fases dinámicas del datacubo histórico."""

import argparse
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import xarray as xr

from src.config import (
    BOUNDARY_PATH,
    CORINE_PATH,
    DATACUBE_END,
    DATACUBE_PATH,
    DATACUBE_START,
    EGIF_EVENTS_PATH,
    EGIF_METADATA_PATH,
    EGIF_TARGET_PATH,
    ERA5_DAILY_PATH,
    ERA5_RAW_DIR,
    FWI_RAW_DIR,
    GRID_CELL_SIZE_M,
    GRID_DIR,
    GRID_PATH,
    HUMAN_ACTIVITY_CUBE_PATH,
    HUMAN_ACTIVITY_RAW_DIR,
    LANDCOVER_CUBE_PATH,
    METEOROLOGY_CUBE_PATH,
    SPATIAL_CUBE_PATH,
    TABULAR_DATASET_DIR,
    TIME_CUBE_PATH,
    TOPOGRAPHY_CUBE_PATH,
)
from src.datacube_profile import CANONICAL_PROFILE
from src.features.tabular import exportar_datacubo_tabular
from src.features.time import crear_datacubo_temporal, guardar_datacubo_temporal
from src.geospatial.human_activity import construir_capa_actividad_humana
from src.geospatial.pipeline import run_pipeline as construir_capas_estaticas
from src.ingestion.era5 import descargar_era5_land
from src.ingestion.fwi import descargar_fwi_historico, listar_archivos_fwi
from src.ingestion.meteorology import listar_archivos_era5
from src.ingestion.pipeline import ejecutar_pipeline_egif, ejecutar_pipeline_meteorologia
from src.pipeline import construir_datacubo_completo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeriodoCubo:
    """Contrato anual declarado por quien construye el cubo.

    EGIF es un registro de eventos: la primera y la última ignición no indican
    cuándo empieza o termina su cobertura. Por ello un intervalo histórico se
    declara por años completos. Solo el último año puede cerrarse en una fecha
    concreta cuando la persona usuaria sabe que es parcial.
    """

    start_year: int
    end_year: int
    include_partial_final_year: bool = False
    final_date: str | None = None

    def __post_init__(self) -> None:
        if self.start_year > self.end_year:
            raise ValueError("start_year no puede ser mayor que end_year.")
        if self.include_partial_final_year:
            if not self.final_date:
                raise ValueError(
                    "Un año parcial requiere --final-date (formato AAAA-MM-DD)."
                )
            final = date.fromisoformat(self.final_date)
            if final.year != self.end_year:
                raise ValueError("--final-date debe pertenecer al end_year indicado.")
        elif self.final_date:
            raise ValueError("--final-date solo se usa junto con --partial-final-year.")

    @property
    def start_date(self) -> date:
        return date(self.start_year, 1, 1)

    @property
    def end_date(self) -> date:
        return date.fromisoformat(self.final_date) if self.include_partial_final_year else date(
            self.end_year, 12, 31
        )

    @property
    def meteorology_context_start(self) -> date:
        """Primer día del mes anterior para los acumulados de hasta 30 días."""
        return date(self.start_year - 1, 12, 1)

    @property
    def egif_coverage(self) -> str:
        return (
            "partial_final_year" if self.include_partial_final_year else "full_calendar_years"
        )


def _validar_entrada_egif(path: str | Path) -> Path:
    """Comprueba que hay XML EGIF sin inferir cobertura desde sus incendios."""
    source = Path(path)
    files = [source] if source.is_file() else sorted(source.glob("*.xml"))
    if not files:
        raise FileNotFoundError(
            "No se encontraron XML EGIF. Coloca uno o varios XML oficiales en "
            "data/raw/fire_history/."
        )
    return source


def _guardar_manifiesto_periodo(
    periodo: PeriodoCubo, destination: str | Path, status: str
) -> None:
    """Registra la cobertura declarada y el estado de la ejecución."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": status,
                "period": {
                    **asdict(periodo),
                    "start_date": periodo.start_date.isoformat(),
                    "end_date": periodo.end_date.isoformat(),
                    "meteorology_context_start": periodo.meteorology_context_start.isoformat(),
                    "egif_coverage": periodo.egif_coverage,
                },
                "rule": (
                    "EGIF coverage is declared by selected years, not inferred from the first "
                    "or last observed ignition."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


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


def _filtrar_capa_canonica(
    source_path: str | Path, flags: dict[str, bool], destination: str | Path
) -> None:
    """Copia de una capa estática limitada al contrato canónico de variables."""
    with xr.open_dataset(source_path) as source:
        selected = [name for name in source.data_vars if flags.get(name, False)]
        if not selected:
            raise ValueError(f"El perfil canónico no selecciona variables de {source_path}.")
        source[selected].to_netcdf(destination)


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
    skip_fwi: bool = False,
    rebuild_static: bool = False,
    rebuild_human_activity: bool = False,
    start_year: int = date.fromisoformat(DATACUBE_START).year,
    end_year: int = date.fromisoformat(DATACUBE_END).year,
    partial_final_year: bool = False,
    final_date: str | None = None,
    download_missing_era5: bool = True,
) -> None:
    """Construye tiempo, ERA5, EGIF, NetCDF y Parquet bajo un único contrato.

    La infraestructura estática no se recalcula aquí: DEM y CORINE son capas
    lentas y se generan una vez con ``src.geospatial.pipeline``. Este workflow
    consume esas capas validadas y orquesta todas las fases temporales.
    """
    logger.info("[1/5] Validando las capas estáticas locales.")
    if rebuild_static:
        logger.info("      Reconstruyendo rejilla, topografía y CORINE.")
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
        logger.info("      Construyendo capa de actividad humana.")
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

    egif_xml = _validar_entrada_egif(egif_xml)
    periodo = PeriodoCubo(start_year, end_year, partial_final_year, final_date)
    start_date = periodo.start_date.isoformat()
    end_date = periodo.end_date.isoformat()
    context_start = periodo.meteorology_context_start.isoformat()
    time_cube_path = Path(time_cube_path)
    logger.info(
        "      Periodo declarado: %s a %s (%s).",
        start_date,
        end_date,
        "año final parcial" if partial_final_year else "años completos",
    )
    logger.info("[2/5] Preparando calendario, ERA5-Land y el baseline FWI de CEMS.")
    if download_missing_era5:
        logger.info("      Comprobando y descargando únicamente los meses ERA5 que falten.")
        descargar_era5_land(raw_meteorology_dir, periodo.meteorology_context_start, periodo.end_date)
    else:
        listar_archivos_era5(raw_meteorology_dir, context_start, end_date)
    if not skip_fwi:
        logger.info("      Comprobando y descargando únicamente los años FWI que falten.")
        descargar_fwi_historico(FWI_RAW_DIR, periodo.start_date, periodo.end_date)
        listar_archivos_fwi(FWI_RAW_DIR, start_date, end_date)

    # Solo se reemplazan productos derivados después de validar fuentes, credenciales
    # y cobertura. Un fallo de preflight no destruye un cubo ya construido.
    manifest_path = Path(datacube_path).parent / "run_manifest.json"
    _guardar_manifiesto_periodo(periodo, manifest_path, status="running")
    _limpiar_salidas_dinamicas(meteorology_cube_path, datacube_path, tabular_output_dir)
    if skip_daily_meteorology:
        logger.info("      Reutilizando ERA5 diario: %s", daily_meteorology_path)
    guardar_datacubo_temporal(
        crear_datacubo_temporal(start_date, end_date, CANONICAL_PROFILE["time"]),
        time_cube_path,
    )
    ejecutar_pipeline_meteorologia(
        spatial_cube_path=spatial_cube_path,
        grid_path=grid_path,
        raw_dir=raw_meteorology_dir,
        daily_output_path=daily_meteorology_path,
        meteorology_cube_path=meteorology_cube_path,
        start_date=context_start,
        end_date=end_date,
        skip_daily=skip_daily_meteorology,
        inclusion_flags=CANONICAL_PROFILE["meteorology"],
        include_fwi=not skip_fwi,
        fwi_inclusion_flags=CANONICAL_PROFILE["fwi"],
        fwi_start_date=start_date,
    )
    logger.info("[3/5] Procesando igniciones EGIF y el target diario.")
    ejecutar_pipeline_egif(
        xml_path=egif_xml,
        grid_path=grid_path,
        events_output_path=EGIF_EVENTS_PATH,
        target_output_path=target_path,
        metadata_output_path=metadata_path,
        start_date=start_date,
        end_date=end_date,
    )
    logger.info("[4/5] Ensamblando el datacubo NetCDF canónico.")
    datacube_parent = Path(datacube_path).parent
    datacube_parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="canonical_layers_", dir=datacube_parent) as temporary_dir:
        temporary_dir = Path(temporary_dir)
        topography_path = temporary_dir / "topography.nc"
        landcover_path = temporary_dir / "landcover.nc"
        human_activity_path = temporary_dir / "human_activity.nc"
        _filtrar_capa_canonica(
            TOPOGRAPHY_CUBE_PATH, CANONICAL_PROFILE["topography"], topography_path
        )
        _filtrar_capa_canonica(LANDCOVER_CUBE_PATH, CANONICAL_PROFILE["landcover"], landcover_path)
        _filtrar_capa_canonica(
            HUMAN_ACTIVITY_CUBE_PATH,
            CANONICAL_PROFILE["human_activity"],
            human_activity_path,
        )
        construir_datacubo_completo(
            spatial_cube_path=spatial_cube_path,
            topography_cube_path=topography_path,
            landcover_cube_path=landcover_path,
            human_activity_cube_path=human_activity_path,
            time_cube_path=time_cube_path,
            meteorology_cube_path=meteorology_cube_path,
            egif_target_path=target_path,
            output_path=datacube_path,
            start_date=start_date,
            end_date=end_date,
        )
    logger.info("      NetCDF creado: %s", datacube_path)
    logger.info("[5/5] Exportando dataset tabular Parquet por años.")
    exportar_datacubo_tabular(datacube_path, tabular_output_dir)
    _guardar_manifiesto_periodo(periodo, manifest_path, status="completed")
    logger.info("✓ Pipeline finalizado correctamente.")


def main() -> None:
    """Ejecuta la ruta reproducible desde las fuentes locales ya descargadas."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Construye un datacubo histórico EGIF + ERA5 a partir de años completos."
    )
    parser.add_argument(
        "--egif-xml",
        default="data/raw/fire_history",
        help="Archivo XML o carpeta con XML EGIF oficiales (por defecto: data/raw/fire_history).",
    )
    parser.add_argument("--start-year", type=int, default=date.fromisoformat(DATACUBE_START).year)
    parser.add_argument("--end-year", type=int, default=date.fromisoformat(DATACUBE_END).year)
    parser.add_argument(
        "--partial-final-year",
        action="store_true",
        help="Permite cerrar el último año en --final-date, en vez de el 31 de diciembre.",
    )
    parser.add_argument(
        "--final-date",
        help="Fecha de fin AAAA-MM-DD. Obligatoria solo con --partial-final-year.",
    )
    parser.add_argument("--grid", default=str(GRID_PATH))
    parser.add_argument("--spatial-cube", default=str(SPATIAL_CUBE_PATH))
    parser.add_argument("--rebuild-static", action="store_true")
    parser.add_argument(
        "--rebuild-human-activity",
        action="store_true",
        help="Recalcula las variables OSM de carreteras y zonas residenciales.",
    )
    parser.add_argument("--skip-daily-meteorology", action="store_true")
    parser.add_argument(
        "--no-download-missing-era5",
        action="store_true",
        help="Solo comprueba ERA5 local; no solicita los meses que falten a Copernicus.",
    )
    parser.add_argument(
        "--skip-fwi",
        action="store_true",
        help="No descarga ni incorpora Fire Weather Index (solo para diagnósticos).",
    )
    args = parser.parse_args()
    ejecutar_workflow_historico(
        egif_xml=args.egif_xml,
        grid_path=args.grid,
        spatial_cube_path=args.spatial_cube,
        skip_daily_meteorology=args.skip_daily_meteorology,
        skip_fwi=args.skip_fwi,
        rebuild_static=args.rebuild_static,
        rebuild_human_activity=args.rebuild_human_activity,
        start_year=args.start_year,
        end_year=args.end_year,
        partial_final_year=args.partial_final_year,
        final_date=args.final_date,
        download_missing_era5=not args.no_download_missing_era5,
    )


if __name__ == "__main__":
    main()
