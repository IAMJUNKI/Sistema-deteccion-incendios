"""Módulo para el procesamiento y extracción de variables topográficas.

Descarga los datos del Copernicus DEM GLO-30 para el bounding box de la rejilla,
reproyecta el raster al CRS métrico (EPSG:25829), calcula la pendiente y
la orientación del terreno en metros, y agrega estas variables por celda de 1 km²
mediante estadísticas zonales rasterizadas de alta velocidad.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, calculate_default_transform, reproject

logger = logging.getLogger(__name__)


def descargar_dem_galicia(
    bounds_wgs84: Tuple[float, float, float, float]
) -> Tuple[np.ndarray, dict]:
    """Descarga el Copernicus DEM GLO-30 para un bounding box en WGS84.

    Utiliza la librería `dem-stitcher` para descargar los datos directamente de AWS.

    Args:
        bounds_wgs84: Tupla de (min_lon, min_lat, max_lon, max_lat).

    Returns:
        Tuple del array de elevación y el diccionario de perfil del rasterio.
    """
    logger.info(
        f"Descargando Copernicus DEM GLO-30 para los límites WGS84: {bounds_wgs84}..."
    )
    try:
        from dem_stitcher import stitch_dem
    except ImportError as e:
        logger.error(
            "La librería 'dem-stitcher' no está instalada. Ejecuta 'pip install dem-stitcher'."
        )
        raise e

    # Descarga del DEM sin credenciales desde AWS
    dem_array, dem_profile = stitch_dem(
        bounds_wgs84,
        dem_name="glo_30",
        dst_ellipsoidal_height=False,
        dst_area_or_point="Point",
    )
    logger.info(f"DEM descargado con éxito. Shape original: {dem_array.shape}")
    return dem_array, dem_profile


def reproyectar_raster_utm(
    dem_array: np.ndarray,
    perfil_src: dict,
    crs_destino: str = "EPSG:25829",
    resolucion_destino: float = 30.0,
) -> Tuple[np.ndarray, dict]:
    """Reproyecta un raster de elevación en grados (WGS84) a metros (UTM).

    Args:
        dem_array: Array del DEM descargado.
        perfil_src: Perfil original del raster en EPSG:4326.
        crs_destino: CRS métrico de destino.
        resolucion_destino: Tamaño de pixel en metros en el raster destino.

    Returns:
        Tuple del array reproyectado y el nuevo perfil de rasterio.
    """
    logger.info(f"Reproyectando DEM a {crs_destino} con resolución de {resolucion_destino}m...")

    # Calcular la transformación de destino y dimensiones
    transform, width, height = calculate_default_transform(
        perfil_src["crs"],
        crs_destino,
        perfil_src["width"],
        perfil_src["height"],
        *rasterio.transform.array_bounds(
            perfil_src["height"], perfil_src["width"], perfil_src["transform"]
        ),
        resolution=resolucion_destino,
    )

    perfil_dst = perfil_src.copy()
    perfil_dst.update(
        {
            "crs": crs_destino,
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": -9999.0,
        }
    )

    dem_reproyectado = np.empty((height, width), dtype=np.float32)

    # Ejecutar la reproyección
    reproject(
        source=dem_array,
        destination=dem_reproyectado,
        src_transform=perfil_src["transform"],
        src_crs=perfil_src["crs"],
        dst_transform=transform,
        dst_crs=crs_destino,
        resampling=Resampling.bilinear,
        src_nodata=perfil_src.get("nodata", np.nan),
        dst_nodata=-9999.0,
    )

    return dem_reproyectado, perfil_dst


def calcular_pendiente_y_orientacion(
    dem_array: np.ndarray, transform: rasterio.Affine
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula la pendiente (en grados) y la orientación (en grados) a partir del DEM en metros.

    Args:
        dem_array: Array del DEM en un CRS métrico.
        transform: Affine transform del raster.

    Returns:
        Tuple de (pendiente_deg, orientacion_deg).
    """
    logger.info("Calculando pendientes y orientaciones del terreno...")

    # Reemplazar nodata con nans para evitar distorsiones en el cálculo del gradiente
    dem_temp = dem_array.copy()
    dem_temp[dem_temp == -9999.0] = np.nan

    # Resolución del pixel (suponiendo cuadrícula cuadrada)
    px = transform.a
    py = -transform.e

    # Gradientes espaciales (cambio en Z respecto a X e Y)
    dzdx = np.gradient(dem_temp, px, axis=1)
    dzdy = np.gradient(dem_temp, py, axis=0)

    # 1. Pendiente en grados (Slope)
    pendiente_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    pendiente_deg = np.degrees(pendiente_rad)

    # 2. Orientación en grados (Aspect: 0 = Norte, 90 = Este, 180 = Sur, 270 = Oeste)
    aspect_rad = np.arctan2(-dzdx, dzdy)
    orientacion_deg = np.degrees(aspect_rad) % 360

    # Rellenar nans resultantes
    pendiente_deg = np.nan_to_num(pendiente_deg, nan=-9999.0)
    orientacion_deg = np.nan_to_num(orientacion_deg, nan=-9999.0)

    return pendiente_deg, orientacion_deg


def clasificar_orientacion(grados: float) -> str:
    """Clasifica los grados de orientación en las 4 macro-direcciones o plana.

    Args:
        grados: Dirección en grados (0-360).

    Returns:
        Categoría string de orientación.
    """
    if grados == -9999.0 or np.isnan(grados):
        return "plana"
    # Clasificación estándar: Norte (315-45), Este (45-135), Sur (135-225), Oeste (225-315)
    if grados >= 315 or grados < 45:
        return "norte"
    elif 45 <= grados < 135:
        return "este"
    elif 135 <= grados < 225:
        return "sur"
    else:
        return "oeste"


def extraer_estadisticas_topograficas_rapidas(
    gdf_grid: gpd.GeoDataFrame,
    dem_array: np.ndarray,
    pendiente_array: np.ndarray,
    orientacion_array: np.ndarray,
    perfil_raster: dict,
) -> pd.DataFrame:
    """Calcula estadísticas zonales vectorizadas y rápidas mediante rasterización de cell_ids.

    Args:
        gdf_grid: Rejilla base de polígonos.
        dem_array: Array del DEM en UTM.
        pendiente_array: Array de pendientes calculadas.
        orientacion_array: Array de orientaciones calculadas.
        perfil_raster: Perfil del raster UTM.

    Returns:
        DataFrame con [cell_id, altitud_media, pendiente_media, orientacion_media, orientacion_clase].
    """
    logger.info("Iniciando agregación topográfica rápida por celda (vectorized groupby)...")

    # Obtener dimensiones y transform
    height, width = perfil_raster["height"], perfil_raster["width"]
    transform = perfil_raster["transform"]

    # Crear lista de tuplas (geometria, cell_id) para la rasterización
    formas_celdas = ((geom, float(cell_id)) for geom, cell_id in zip(gdf_grid.geometry, gdf_grid.cell_id))

    # Rasterizar los cell_ids en la resolución del DEM
    logger.info("Rasterizando geometrías de las celdas...")
    grid_raster = rasterize(
        shapes=formas_celdas,
        out_shape=(height, width),
        transform=transform,
        fill=-1.0,
        dtype=np.float32,
    )

    # Aplanar arrays para análisis columnar con Pandas (extremadamente rápido)
    flat_grid = grid_raster.flatten()
    flat_alt = dem_array.flatten()
    flat_slope = pendiente_array.flatten()
    flat_aspect = orientacion_array.flatten()

    # Filtrar solo pixeles que pertenecen a alguna celda de nuestro grid
    mask = flat_grid != -1.0
    df_pixels = pd.DataFrame(
        {
            "cell_id": flat_grid[mask].astype(int),
            "alt": flat_alt[mask],
            "slope": flat_slope[mask],
            "aspect": flat_aspect[mask],
        }
    )

    # Limpiar valores de nodata de los arrays originales en el dataframe
    df_pixels = df_pixels[
        (df_pixels["alt"] != -9999.0)
        & (df_pixels["slope"] != -9999.0)
        & (df_pixels["aspect"] != -9999.0)
    ]

    # Agrupar y calcular estadísticas zonales
    logger.info("Agrupando y calculando promedios por cell_id...")
    df_agrupado = (
        df_pixels.groupby("cell_id")
        .agg(
            altitud_media=("alt", "mean"),
            pendiente_media=("slope", "mean"),
            orientacion_media=("aspect", "mean"),
        )
        .reset_index()
    )

    # Clasificar la orientación predominante de cada celda
    df_agrupado["orientacion_clase"] = df_agrupado["orientacion_media"].apply(
        clasificar_orientacion
    )

    return df_agrupado
