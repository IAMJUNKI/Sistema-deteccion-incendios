"""Módulo para la generación de la rejilla geoespacial base de Galicia.

Este módulo descarga los límites geográficos de Galicia (vía GADM), los proyecta
al sistema métrico local (EPSG:25829) y genera una malla regular de celdas de
1 km x 1 km que cubre todo el territorio terrestre de la comunidad autónoma.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
from shapely.geometry import box

logger = logging.getLogger(__name__)


def descargar_limites_galicia(
    ruta_salida: Union[str, Path], url_fuente: Optional[str] = None
) -> gpd.GeoDataFrame:
    """Descarga los límites administrativos de Galicia desde GADM si no existen localmente.

    Args:
        ruta_salida: Ruta local donde guardar el archivo GeoJSON descargado.
        url_fuente: URL alternativa para descargar el archivo. Por defecto,
            se usa el nivel 1 de GADM para España.

    Returns:
        GeoDataFrame con el polígono georreferenciado de Galicia.
    """
    ruta_salida = Path(ruta_salida)
    if ruta_salida.exists():
        logger.info(f"Cargando límites de Galicia desde el archivo local: {ruta_salida}")
        return gpd.read_file(ruta_salida)

    # URL base de GADM 4.1 para España (Nivel 1 contiene Comunidades Autónomas)
    if not url_fuente:
        url_fuente = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ESP_1.json"

    logger.info(f"Descargando límites de España desde {url_fuente}...")
    try:
        gdf_espana = gpd.read_file(url_fuente)
    except Exception as e:
        logger.error(f"Error al descargar los datos de GADM: {e}")
        raise RuntimeError("No se pudo descargar la frontera de Galicia.") from e

    # Filtrar Galicia
    gdf_galicia = gdf_espana[gdf_espana["NAME_1"] == "Galicia"].copy()
    if gdf_galicia.empty:
        raise ValueError("No se encontró la región 'Galicia' en los datos descargados.")

    # Guardar en local
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    gdf_galicia.to_file(ruta_salida, driver="GeoJSON")
    logger.info(f"Límites de Galicia descargados y guardados en: {ruta_salida}")

    return gdf_galicia


def generar_malla_base(
    gdf_boundary: gpd.GeoDataFrame,
    cell_size_meters: float = 1000.0,
    crs_trabajo: str = "EPSG:25829",
) -> gpd.GeoDataFrame:
    """Genera una rejilla regular de polígonos cuadrados que cubre el bounding box de la región.

    Args:
        gdf_boundary: GeoDataFrame con el límite geográfico de la región.
        cell_size_meters: Tamaño de celda en metros. Por defecto, 1000m (1 km).
        crs_trabajo: Sistema de coordenadas de referencia métrico para la malla.
            Por defecto, EPSG:25829 (ETRS89 / UTM zone 29N).

    Returns:
        GeoDataFrame con la cuadrícula completa sin recortar.
    """
    # Asegurar que el contorno está en el CRS de trabajo
    gdf_proj = gdf_boundary.to_crs(crs_trabajo)
    minx, miny, maxx, maxy = gdf_proj.total_bounds

    logger.info(
        f"Generando cuadrícula base en {crs_trabajo} con resolución de {cell_size_meters}m..."
    )
    cols = np.arange(minx, maxx, cell_size_meters)
    rows = np.arange(miny, maxy, cell_size_meters)

    cells = []
    for x in cols:
        for y in rows:
            # Crear polígono cuadrado
            cells.append(box(x, y, x + cell_size_meters, y + cell_size_meters))

    gdf_grid = gpd.GeoDataFrame(geometry=cells, crs=crs_trabajo)
    logger.info(f"Cuadrícula rectangular bruta generada: {len(gdf_grid):,} celdas.")
    return gdf_grid


def recortar_y_etiquetar_rejilla(
    gdf_grid: gpd.GeoDataFrame,
    gdf_boundary: gpd.GeoDataFrame,
    crs_trabajo: str = "EPSG:25829",
) -> gpd.GeoDataFrame:
    """Filtra las celdas que intersectan con el territorio terrestre y calcula sus índices.

    Args:
        gdf_grid: Cuadrícula bruta generada.
        gdf_boundary: Límite geográfico del territorio.
        crs_trabajo: CRS métrico de trabajo.

    Returns:
        GeoDataFrame final de celdas recortadas e indexadas.
    """
    gdf_boundary_proj = gdf_boundary.to_crs(crs_trabajo)

    logger.info("Realizando intersección espacial con la frontera de Galicia...")
    # Intersección espacial para conservar solo las celdas terrestres
    gdf_final = gpd.sjoin(
        gdf_grid, gdf_boundary_proj[["geometry"]], how="inner", predicate="intersects"
    )

    # Limpiar columnas temporales del sjoin e indexar
    gdf_final = gdf_final.drop(columns=["index_right"]).reset_index(drop=True)
    gdf_final["cell_id"] = gdf_final.index.astype(int)

    # Calcular centroides en coordenadas geográficas (WGS84 - EPSG:4326) para visualizaciones y APIs
    logger.info("Calculando coordenadas geográficas de los centroides (WGS84)...")
    centroids_wgs84 = gdf_final.geometry.centroid.to_crs("EPSG:4326")
    gdf_final["lat_centroid"] = centroids_wgs84.y
    gdf_final["lon_centroid"] = centroids_wgs84.x

    logger.info(f"Rejilla final generada: {len(gdf_final):,} celdas activas.")
    return gdf_final


def crear_rejilla_galicia_pipeline(
    ruta_limites: Union[str, Path],
    cell_size_meters: float = 1000.0,
    crs_trabajo: str = "EPSG:25829",
) -> gpd.GeoDataFrame:
    """Función de alto nivel que ejecuta el pipeline completo de creación del grid.

    Args:
        ruta_limites: Ruta al GeoJSON de límites de Galicia (local o descarga).
        cell_size_meters: Resolución del grid en metros.
        crs_trabajo: CRS métrico.

    Returns:
        GeoDataFrame listo para el cruce de DEM y CORINE.
    """
    gdf_galicia = descargar_limites_galicia(ruta_limites)
    gdf_bruto = generar_malla_base(gdf_galicia, cell_size_meters, crs_trabajo)
    gdf_final = recortar_y_etiquetar_rejilla(gdf_bruto, gdf_galicia, crs_trabajo)
    return gdf_final
