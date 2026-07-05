"""Script de preprocesamiento para recortar y rasterizar CORINE Land Cover para Galicia.

Carga el archivo nacional en formato vectorial (GeoPackage) descargado del CNIG,
lo recorta a los límites de Galicia y lo rasteriza a una resolución de 100 metros
generando el GeoTIFF que requiere el pipeline principal.
"""

import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("src.geospatial.preprocesar_corine")


def preprocesar_corine(
    ruta_gpkg_nacional: Path,
    ruta_limites_galicia: Path,
    ruta_tif_salida: Path,
    resolucion_m: float = 100.0,
) -> None:
    """Recorta y rasteriza el GeoPackage nacional de CORINE para Galicia.

    Args:
        ruta_gpkg_nacional: Ruta al archivo .gpkg de España descargado del CNIG.
        ruta_limites_galicia: Ruta al GeoJSON de límites de Galicia.
        ruta_tif_salida: Ruta donde guardar el GeoTIFF resultante para Galicia.
        resolucion_m: Resolución de celda en metros para el raster (default: 100m).
    """
    logger.info("=== INICIANDO PREPROCESAMIENTO DE CORINE LAND COVER ===")

    # 1. Cargar la frontera de Galicia
    if not ruta_limites_galicia.exists():
        raise FileNotFoundError(
            f"Falta el límite de Galicia en: {ruta_limites_galicia}. "
            "Ejecuta primero el pipeline principal para descargarlo."
        )
    logger.info(f"Cargando límites de Galicia desde: {ruta_limites_galicia}")
    gdf_galicia = gpd.read_file(ruta_limites_galicia)
    # Proyectar al CRS métrico de trabajo (EPSG:25829)
    gdf_galicia_proj = gdf_galicia.to_crs("EPSG:25829")

    # 2. Cargar el GeoPackage nacional
    if not ruta_gpkg_nacional.exists():
        raise FileNotFoundError(
            f"No se encontró el GeoPackage nacional de CORINE en: {ruta_gpkg_nacional}. "
            "Descárgalo del CNIG y guárdalo en esa ruta."
        )

    logger.info(f"Cargando GeoPackage nacional de CORINE desde: {ruta_gpkg_nacional}...")
    
    # Auto-detectar capas usando pyogrio
    import pyogrio
    capas = pyogrio.list_layers(ruta_gpkg_nacional)
    capa_clc = None
    
    # Buscar una capa que empiece por 'CLC' (el mapa completo) en lugar de 'CHA' (cambios)
    for capa in capas[:, 0]:
        if capa.startswith("CLC"):
            capa_clc = capa
            break
            
    if capa_clc:
        logger.info(f"Capa detectada y seleccionada automáticamente: '{capa_clc}'")
        gdf_clc_completo = gpd.read_file(ruta_gpkg_nacional, layer=capa_clc)
    else:
        logger.warning("No se detectó ninguna capa que empiece por 'CLC'. Se leerá la capa por defecto.")
        gdf_clc_completo = gpd.read_file(ruta_gpkg_nacional)

    # Asegurar el mismo CRS proyectado
    gdf_clc_completo = gdf_clc_completo.to_crs("EPSG:25829")

    # 3. Recortar espacialmente por Galicia
    logger.info("Recortando CORINE vectorialmente por la frontera de Galicia...")
    gdf_clc_galicia = gpd.clip(gdf_clc_completo, gdf_galicia_proj)

    # 4. Rasterizar el vector resultante a 100 metros
    logger.info(f"Rasterizando CORINE Galicia a resolución de {resolucion_m} metros...")
    minx, miny, maxx, maxy = gdf_galicia_proj.total_bounds

    # Calcular dimensiones y transformación afín del raster de salida
    width = int(np.ceil((maxx - minx) / resolucion_m))
    height = int(np.ceil((maxy - miny) / resolucion_m))
    transform = from_origin(minx, maxy, resolucion_m, resolucion_m)

    # Convertir las etiquetas de código a enteros en una tupla (geometria, codigo_clc)
    # Buscamos la columna de código numérico automáticamente en base al año
    posibles_columnas = [
        "code_18", "clc18", "CODE_18", "CLC18", "clc_18", "CODE18",
        "code_12", "clc12", "CODE_12", "CLC12", "clc_12", "CODE12",
        "code_06", "clc06", "CODE_06", "CLC06", "clc_06", "CODE06",
    ]
    col_codigo = None
    for col in posibles_columnas:
        if col in gdf_clc_galicia.columns:
            col_codigo = col
            break

    if not col_codigo:
        raise ValueError(
            f"No se encontró la columna de códigos de CORINE. Columnas disponibles: {gdf_clc_galicia.columns.tolist()}"
        )

    logger.info(f"Utilizando la columna '{col_codigo}' para mapear las clases.")

    # Asegurar valores enteros limpios para la rasterización
    gdf_clc_galicia[col_codigo] = gdf_clc_galicia[col_codigo].astype(int)

    formas = (
        (geom, float(val))
        for geom, val in zip(gdf_clc_galicia.geometry, gdf_clc_galicia[col_codigo])
    )

    clc_raster = rasterize(
        shapes=formas,
        out_shape=(height, width),
        transform=transform,
        fill=0,  # Código 0 indica "sin datos"
        dtype=np.uint16,
    )

    # 5. Guardar como GeoTIFF
    logger.info(f"Guardando raster de CORINE Galicia en: {ruta_tif_salida}")
    ruta_tif_salida.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        ruta_tif_salida,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.uint16,
        crs="EPSG:25829",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(clc_raster, 1)

    logger.info("=== PREPROCESAMIENTO COMPLETADO CON ÉXITO ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Recorta y rasteriza CORINE Land Cover España para Galicia."
    )
    parser.add_argument(
        "--gpkg",
        type=str,
        default="data/raw/corine/clc_espana_2018.gpkg",
        help="Ruta al GeoPackage de CORINE España descargado del CNIG.",
    )
    parser.add_argument(
        "--boundary",
        type=str,
        default="data/raw/igm/galicia_boundary.geojson",
        help="Ruta al GeoJSON de límites de Galicia.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/corine/clc_galicia.tif",
        help="Ruta de salida del GeoTIFF rasterizado.",
    )

    args = parser.parse_args()

    preprocesar_corine(
        ruta_gpkg_nacional=Path(args.gpkg),
        ruta_limites_galicia=Path(args.boundary),
        ruta_tif_salida=Path(args.output),
    )
