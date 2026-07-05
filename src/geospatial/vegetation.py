"""Módulo para el procesamiento de datos de cobertura del suelo (CORINE Land Cover).

Lee el raster de CORINE Land Cover, lo reproyecta al CRS métrico (EPSG:25829),
mapea los 44 códigos originales a macro-clases simplificadas de combustible
forestal y calcula la clase predominante y el porcentaje de cobertura forestal
para cada celda de 1 km² utilizando una agregación rasterizada.
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

# Mapeo de códigos numéricos de CORINE Land Cover (CLC) a macro-clases de combustible
CLC_FUEL_MAPPING: Dict[int, str] = {
    # ─── 1. Zonas Artificiales (Urbano) ──────────────────────────────────────
    111: "urbano",  # Tejido urbano continuo
    112: "urbano",  # Tejido urbano discontinuo
    121: "urbano",  # Zonas industriales o comerciales
    122: "urbano",  # Redes viarias y ferroviarias y terrenos asociados
    123: "urbano",  # Zonas portuarias
    124: "urbano",  # Aeropuertos
    131: "urbano",  # Zonas de extracción de materias primas
    132: "urbano",  # Vertederos
    133: "urbano",  # Zonas en construcción
    141: "urbano",  # Zonas verdes urbanas
    142: "urbano",  # Zonas deportivas y recreativas
    # ─── 2. Zonas Agrícolas ──────────────────────────────────────────────────
    211: "agricola",  # Tierras de labor en secano
    212: "agricola",  # Tierras de labor en regadío
    213: "agricola",  # Arrozeras
    221: "agricola",  # Viñedos
    222: "agricola",  # Frutales
    223: "agricola",  # Olivares
    231: "agricola",  # Praderas
    241: "agricola",  # Cultivos anuales asociados con cultivos permanentes
    242: "agricola",  # Sistemas agrícolas complejos
    243: "agricola",  # Tierras ocupadas principalmente por la agricultura
    244: "agricola",  # Superficies agroforestales
    # ─── 3. Bosques y Zonas de Vegetación Natural (Combustible Forestal) ──────
    311: "bosque_frondosas",  # Bosques de frondosas (ej: roble, eucalipto)
    312: "bosque_coniferas",  # Bosques de coníferas (ej: pino)
    313: "bosque_mixto",  # Bosques mixtos
    321: "pastizal",  # Pastizales naturales
    322: "matorral",  # Brezales y matorrales (matorral bajo/seco)
    323: "matorral",  # Vegetación esclerófila (maquis, garriga)
    324: "matorral",  # Bosque de transición (matorral alto)
    331: "pastizal",  # Playas, dunas y arenales
    332: "pastizal",  # Roquedo desnudo
    333: "pastizal",  # Vegetación dispersa
    334: "pastizal",  # Zonas quemadas (combustible muy bajo temporal)
    335: "pastizal",  # Glaciares y nieves perpetuas
    # ─── 4. Humedales y Masas de Agua ────────────────────────────────────────
    411: "agua_humedal",  # Humedales interiores
    412: "agua_humedal",  # Turberas
    421: "agua_humedal",  # Salinas marítimas
    422: "agua_humedal",  # Humedales intermareales
    423: "agua_humedal",  # Marismas salinas
    511: "agua_humedal",  # Cursos de agua
    512: "agua_humedal",  # Masas de agua (embalses, lagos)
    521: "agua_humedal",  # Lagunas costeras
    522: "agua_humedal",  # Estuarios
    523: "agua_humedal",  # Mares y océanos
}


def reproyectar_clc_utm(
    ruta_clc_original: Union[str, Path],
    crs_destino: str = "EPSG:25829",
    resolucion_destino: float = 100.0,
) -> Tuple[np.ndarray, dict]:
    """Carga y reproyecta el raster original de CORINE Land Cover a UTM 29N.

    Args:
        ruta_clc_original: Ruta al GeoTIFF original de CORINE.
        crs_destino: CRS métrico de trabajo.
        resolucion_destino: Resolución de pixel en metros para el remuestreo.

    Returns:
        Tuple del array reproyectado (códigos CLC) y el perfil de rasterio.
    """
    logger.info(
        f"Cargando y reproyectando CORINE Land Cover desde: {ruta_clc_original}..."
    )
    ruta_clc_original = Path(ruta_clc_original)

    if not ruta_clc_original.exists():
        raise FileNotFoundError(
            f"El archivo CORINE Land Cover no existe en: {ruta_clc_original}. "
            "Por favor, descárgalo del CNIG o del Copernicus Land Service."
        )

    with rasterio.open(ruta_clc_original) as src:
        # Calcular transformación de destino
        transform, width, height = calculate_default_transform(
            src.crs,
            crs_destino,
            src.width,
            src.height,
            *src.bounds,
            resolution=resolucion_destino,
        )

        perfil_dst = src.meta.copy()
        perfil_dst.update(
            {
                "crs": crs_destino,
                "transform": transform,
                "width": width,
                "height": height,
                "nodata": 0,  # Código 0 indica "sin datos" en CLC
            }
        )

        clc_reproyectado = np.empty((height, width), dtype=np.uint16)

        # Reproyectar usando el vecino más cercano (conservar códigos categóricos)
        reproject(
            source=rasterio.band(src, 1),
            destination=clc_reproyectado,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs_destino,
            resampling=Resampling.nearest,
            src_nodata=src.nodata or 0,
            dst_nodata=0,
        )

    logger.info(f"CORINE reproyectado. Dimensiones: {clc_reproyectado.shape}")
    return clc_reproyectado, perfil_dst


def extraer_variables_vegetacion(
    gdf_grid: gpd.GeoDataFrame, clc_array: np.ndarray, perfil_raster: dict
) -> pd.DataFrame:
    """Extrae las clases de combustible predominantes y porcentaje forestal por celda.

    Args:
        gdf_grid: GeoDataFrame de la rejilla.
        clc_array: Array del raster CLC en UTM.
        perfil_raster: Perfil del raster de entrada.

    Returns:
        DataFrame con [cell_id, combustible_clase, combustible_pct_forestal].
    """
    logger.info("Agrupando clases de vegetación por celda (vectorized groupby)...")

    height, width = perfil_raster["height"], perfil_raster["width"]
    transform = perfil_raster["transform"]

    # Crear lista de tuplas para la rasterización de las celdas
    formas_celdas = ((geom, float(cell_id)) for geom, cell_id in zip(gdf_grid.geometry, gdf_grid.cell_id))

    # Rasterizar los cell_ids en la resolución del raster CLC
    grid_raster = rasterize(
        shapes=formas_celdas,
        out_shape=(height, width),
        transform=transform,
        fill=-1.0,
        dtype=np.float32,
    )

    flat_grid = grid_raster.flatten()
    flat_clc = clc_array.flatten()

    # Filtrar pixeles activos en el grid
    mask = flat_grid != -1.0
    df_pixels = pd.DataFrame(
        {
            "cell_id": flat_grid[mask].astype(int),
            "clc_code": flat_clc[mask].astype(int),
        }
    )

    # Filtrar sin datos (código 0)
    df_pixels = df_pixels[df_pixels["clc_code"] > 0]

    # Mapear códigos CLC a macro-clases de combustible
    df_pixels["combustible"] = df_pixels["clc_code"].map(CLC_FUEL_MAPPING).fillna("otros")

    # 1. Calcular clase predominante por cell_id
    logger.info("Calculando clase predominante por celda...")
    predominantes = (
        df_pixels.groupby("cell_id")["combustible"]
        .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else "otros")
        .reset_index(name="combustible_clase")
    )

    # 2. Calcular porcentaje de cobertura forestal (bosques de frondosas, coníferas o mixto)
    logger.info("Calculando porcentaje de cobertura forestal por celda...")
    df_pixels["es_forestal"] = df_pixels["combustible"].isin(
        ["bosque_frondosas", "bosque_coniferas", "bosque_mixto"]
    )
    pct_forestal = (
        df_pixels.groupby("cell_id")["es_forestal"]
        .mean()
        .reset_index(name="combustible_pct_forestal")
    )
    # Convertir a porcentaje (0-100)
    pct_forestal["combustible_pct_forestal"] = (
        pct_forestal["combustible_pct_forestal"] * 100.0
    )

    # Combinar resultados
    df_vegetacion = pd.merge(predominantes, pct_forestal, on="cell_id", how="left")
    return df_vegetacion
