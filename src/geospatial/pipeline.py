"""Script principal de orquestación del pipeline de la Fase 1 (Infraestructura Geoespacial).

Ejecuta en secuencia la generación del grid, la descarga y cálculo del DEM (Copernicus)
y el procesamiento del suelo (CORINE), uniendo todos los atributos y guardando la rejilla
final en formato GeoParquet en la ruta configurada.
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from src.geospatial.grid import crear_rejilla_galicia_pipeline
from src.geospatial.topography import (
    calcular_pendiente_y_orientacion,
    descargar_dem_galicia,
    extraer_estadisticas_topograficas_rapidas,
    reproyectar_raster_utm,
)
from src.geospatial.vegetation import extraer_variables_vegetacion, reproyectar_clc_utm

# Configurar logging descriptivo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("src.geospatial.pipeline")


def run_pipeline(
    ruta_limites: Path,
    ruta_clc: Path,
    ruta_salida: Path,
    cell_size: float = 1000.0,
) -> None:
    """Ejecuta el pipeline geoespacial completo de la Fase 1.

    Args:
        ruta_limites: Ruta para guardar/leer los límites de Galicia.
        ruta_clc: Ruta del raster local de CORINE Land Cover.
        ruta_salida: Ruta del GeoParquet final de salida.
        cell_size: Tamaño de celda en metros.
    """
    logger.info("=== INICIANDO PIPELINE DE INFRAESTRUCTURA GEOESPACIAL (FASE 1) ===")

    # 1. Generar la Rejilla de Celdas Base
    logger.info("--- PASO 1: Generando Rejilla Base de Galicia ---")
    gdf_grid = crear_rejilla_galicia_pipeline(
        ruta_limites=ruta_limites, cell_size_meters=cell_size
    )

    # 2. Descargar y Calcular Topografía (DEM)
    logger.info("--- PASO 2: Procesando Elevaciones y Pendientes (Copernicus DEM) ---")
    # Obtener el bbox en WGS84 para la descarga
    gdf_wgs84 = gdf_grid.to_crs("EPSG:4326")
    min_lon, min_lat, max_lon, max_lat = gdf_wgs84.total_bounds
    # Pequeño buffer para evitar bordes cortados en la interpolación
    bounds_wgs84 = (min_lon - 0.05, min_lat - 0.05, max_lon + 0.05, max_lat + 0.05)

    try:
        dem_raw, profile_raw = descargar_dem_galicia(bounds_wgs84)
        dem_utm, profile_utm = reproyectar_raster_utm(
            dem_raw, profile_raw, crs_destino="EPSG:25829", resolucion_destino=30.0
        )
        slope, aspect = calcular_pendiente_y_orientacion(dem_utm, profile_utm["transform"])
        df_topo = extraer_estadisticas_topograficas_rapidas(
            gdf_grid, dem_utm, slope, aspect, profile_utm
        )
        logger.info(f"Topografía procesada: {len(df_topo)} celdas agregadas.")
    except Exception as e:
        logger.error(f"Error crítico en el cálculo de la topografía: {e}")
        logger.warning(
            "El pipeline continuará rellenando la topografía con valores por defecto."
        )
        df_topo = pd.DataFrame(
            {
                "cell_id": gdf_grid["cell_id"],
                "altitud_media": 0.0,
                "pendiente_media": 0.0,
                "orientacion_media": 0.0,
                "orientacion_clase": "plana",
            }
        )

    # 3. Procesar Vegetación (CORINE)
    logger.info("--- PASO 3: Procesando Cobertura del Suelo (CORINE Land Cover) ---")
    if ruta_clc.exists():
        try:
            clc_utm, clc_profile = reproyectar_clc_utm(
                ruta_clc_original=ruta_clc,
                crs_destino="EPSG:25829",
                resolucion_destino=100.0,
            )
            df_veg = extraer_variables_vegetacion(gdf_grid, clc_utm, clc_profile)
            logger.info(f"Vegetación procesada: {len(df_veg)} celdas agregadas.")
        except Exception as e:
            logger.error(f"Error procesando el archivo CORINE: {e}")
            logger.warning("Rellenando vegetación con valores por defecto...")
            df_veg = pd.DataFrame(
                {
                    "cell_id": gdf_grid["cell_id"],
                    "combustible_clase": "otros",
                    "combustible_pct_forestal": 0.0,
                }
            )
    else:
        logger.warning(
            f"El archivo CORINE Land Cover no se encontró en '{ruta_clc}'. "
            "Rellenando con valores por defecto ('otros', 0%)."
        )
        df_veg = pd.DataFrame(
            {
                "cell_id": gdf_grid["cell_id"],
                "combustible_clase": "otros",
                "combustible_pct_forestal": 0.0,
            }
        )

    # 4. Integrar Datos por cell_id
    logger.info("--- PASO 4: Cruzando y uniendo todas las capas de datos ---")
    gdf_final = gdf_grid.merge(df_topo, on="cell_id", how="left")
    gdf_final = gdf_final.merge(df_veg, on="cell_id", how="left")

    # Asegurar tipos correctos
    gdf_final["cell_id"] = gdf_final["cell_id"].astype(int)
    gdf_final["altitud_media"] = gdf_final["altitud_media"].astype(float)
    gdf_final["pendiente_media"] = gdf_final["pendiente_media"].astype(float)
    gdf_final["combustible_pct_forestal"] = gdf_final[
        "combustible_pct_forestal"
    ].astype(float)

    # 5. Guardar el GeoParquet resultante
    logger.info(f"--- PASO 5: Guardando GeoParquet resultante en: {ruta_salida} ---")
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    gdf_final.to_parquet(ruta_salida)
    logger.info(f"Pipeline ejecutado correctamente. Dataset guardado: {gdf_final.shape}")

    # 6. Generar mapa de verificación de control de calidad
    try:
        logger.info("Generando mapa de verificación para control de calidad...")
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # Mapa de altitud
        gdf_final.plot(
            column="altitud_media",
            ax=axes[0],
            legend=True,
            cmap="terrain",
            legend_kwds={"label": "Altitud media (m)"},
        )
        axes[0].set_title("Rejilla 1km — Altitud Media (DEM)")
        axes[0].axis("off")

        # Mapa de vegetación
        gdf_final.plot(
            column="combustible_clase",
            ax=axes[1],
            legend=True,
            cmap="tab10",
        )
        axes[1].set_title("Rejilla 1km — Tipo de Combustible (CORINE)")
        axes[1].axis("off")

        docs_dir = Path("docs")
        docs_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(docs_dir / "verificacion_grid_fase1.png", dpi=150)
        logger.info("Mapa de verificación guardado con éxito en 'docs/verificacion_grid_fase1.png'.")
    except Exception as e:
        logger.warning(f"No se pudo generar el mapa de control visual: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase 1: Generación del Grid Geoespacial y variables estáticas."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/grid/galicia_grid_1km_v1.parquet",
        help="Ruta donde guardar el GeoParquet de salida.",
    )
    parser.add_argument(
        "--boundary",
        type=str,
        default="data/raw/igm/galicia_boundary.geojson",
        help="Ruta para leer/guardar el límite de Galicia.",
    )
    parser.add_argument(
        "--corine",
        type=str,
        default="data/raw/corine/clc_galicia.tif",
        help="Ruta al GeoTIFF original de CORINE Land Cover.",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=1000.0,
        help="Resolución de la celda en metros (default: 1000m).",
    )

    args = parser.parse_args()

    run_pipeline(
        ruta_limites=Path(args.boundary),
        ruta_clc=Path(args.corine),
        ruta_salida=Path(args.output),
        cell_size=args.cell_size,
    )
