"""Construcción de la rejilla regular que usa el datacube de incendios.

La referencia espacial es una malla rectangular de 1 km en EPSG:3035. Las
posiciones exteriores a Galicia se mantienen para preservar las dimensiones
``(y, x)`` y se identifican con la máscara auxiliar ``is_galicia``.
"""

import logging
import math
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import box

logger = logging.getLogger(__name__)

DEFAULT_CRS = "EPSG:3035"
DEFAULT_CELL_SIZE_METERS = 1000.0


def descargar_limites_galicia(
    ruta_salida: Union[str, Path], url_fuente: Optional[str] = None
) -> gpd.GeoDataFrame:
    """Carga los límites de Galicia, descargándolos de GADM si no existen.

    Args:
        ruta_salida: Ruta local del GeoJSON de límites.
        url_fuente: URL alternativa de GADM.

    Returns:
        GeoDataFrame con el límite administrativo de Galicia.
    """
    ruta_salida = Path(ruta_salida)
    if ruta_salida.exists():
        logger.info("Cargando límites de Galicia desde %s", ruta_salida)
        return gpd.read_file(ruta_salida)

    url_fuente = url_fuente or "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ESP_1.json"
    logger.info("Descargando límites administrativos desde %s", url_fuente)
    try:
        gdf_espana = gpd.read_file(url_fuente)
    except Exception as error:
        raise RuntimeError("No se pudo descargar la frontera de Galicia.") from error

    gdf_galicia = gdf_espana[gdf_espana["NAME_1"] == "Galicia"].copy()
    if gdf_galicia.empty:
        raise ValueError("No se encontró la región 'Galicia' en los datos descargados.")

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    gdf_galicia.to_file(ruta_salida, driver="GeoJSON")
    return gdf_galicia


def generar_coordenadas_rejilla(
    gdf_boundary: gpd.GeoDataFrame,
    cell_size_meters: float = DEFAULT_CELL_SIZE_METERS,
    crs_trabajo: str = DEFAULT_CRS,
) -> tuple[np.ndarray, np.ndarray, gpd.GeoDataFrame]:
    """Genera los orígenes de celda, alineados a múltiplos de la resolución.

    El límite superior se incluye deliberadamente. Es la misma convención del
    notebook ``01_grid``: cada valor de ``x`` e ``y`` representa la esquina
    inferior izquierda de una celda de 1 km.
    """
    if cell_size_meters <= 0:
        raise ValueError("cell_size_meters debe ser mayor que cero.")

    boundary = gdf_boundary.to_crs(crs_trabajo)
    xmin, ymin, xmax, ymax = boundary.total_bounds
    xmin = math.floor(xmin / cell_size_meters) * cell_size_meters
    ymin = math.floor(ymin / cell_size_meters) * cell_size_meters
    xmax = math.ceil(xmax / cell_size_meters) * cell_size_meters
    ymax = math.ceil(ymax / cell_size_meters) * cell_size_meters

    x = np.arange(xmin, xmax + cell_size_meters, cell_size_meters)
    y = np.arange(ymin, ymax + cell_size_meters, cell_size_meters)
    return x, y, boundary


def calcular_mascara_galicia(
    x: np.ndarray,
    y: np.ndarray,
    gdf_boundary: gpd.GeoDataFrame,
    cell_size_meters: float = DEFAULT_CELL_SIZE_METERS,
) -> np.ndarray:
    """Marca celdas cuyo centro geométrico está dentro de Galicia.

    Args:
        x: Orígenes x de las celdas.
        y: Orígenes y de las celdas.
        gdf_boundary: Límite de Galicia ya reproyectado al CRS de la malla.
        cell_size_meters: Tamaño de lado de cada celda.

    Returns:
        Matriz ``uint8`` de dimensiones ``(len(y), len(x))``.
    """
    x_centres, y_centres = np.meshgrid(x + cell_size_meters / 2, y + cell_size_meters / 2)
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x_centres.ravel(), y_centres.ravel()),
        crs=gdf_boundary.crs,
    )
    joined = gpd.sjoin(points, gdf_boundary[["geometry"]], how="left", predicate="within")
    return joined["index_right"].notna().to_numpy().reshape(len(y), len(x)).astype(np.uint8)


def crear_cubo_rejilla_galicia(
    gdf_boundary: gpd.GeoDataFrame,
    cell_size_meters: float = DEFAULT_CELL_SIZE_METERS,
    crs_trabajo: str = DEFAULT_CRS,
) -> xr.Dataset:
    """Crea el datacube espacial regular equivalente a ``01_grid.ipynb``."""
    x, y, boundary = generar_coordenadas_rejilla(gdf_boundary, cell_size_meters, crs_trabajo)
    mask = calcular_mascara_galicia(x, y, boundary, cell_size_meters)

    cube = xr.Dataset(coords={"x": x, "y": y})
    cube["is_galicia"] = (("y", "x"), mask)
    cube.attrs = {
        "title": "Galicia spatial grid",
        "description": "Base spatial grid used by all datasets in the wildfire datacube.",
        "module": "grid",
        "crs": crs_trabajo,
        "spatial_resolution": f"{cell_size_meters:g} m",
    }
    cube.x.attrs = {"long_name": "Projected x coordinate", "units": "m"}
    cube.y.attrs = {"long_name": "Projected y coordinate", "units": "m"}
    cube["is_galicia"].attrs = {
        "long_name": "Galicia mask",
        "description": "1 if the centre of the grid cell lies inside Galicia, 0 otherwise.",
        "units": "-",
    }
    return cube


def crear_rejilla_vectorial(cube: xr.Dataset, cell_size_meters: float) -> gpd.GeoDataFrame:
    """Convierte el cubo regular completo a una rejilla vectorial, sin recortarla."""
    x = cube.x.values
    y = cube.y.values
    polygons = [box(xi, yi, xi + cell_size_meters, yi + cell_size_meters) for yi in y for xi in x]
    is_galicia = cube["is_galicia"].values.ravel().astype(np.uint8)
    return gpd.GeoDataFrame(
        {
            "cell_id": np.arange(len(polygons), dtype=np.int64),
            "x": np.tile(x, len(y)),
            "y": np.repeat(y, len(x)),
            "is_galicia": is_galicia,
        },
        geometry=polygons,
        crs=cube.attrs["crs"],
    )


def crear_rejilla_galicia_pipeline(
    ruta_limites: Union[str, Path],
    cell_size_meters: float = DEFAULT_CELL_SIZE_METERS,
    crs_trabajo: str = DEFAULT_CRS,
) -> tuple[xr.Dataset, gpd.GeoDataFrame]:
    """Crea el cubo y su representación vectorial completa.

    Returns:
        Tupla ``(cube, grid)``. ``grid`` incluye tanto las celdas de Galicia
        como las exteriores; filtre por ``is_galicia == 1`` para cálculos
        estáticos que solo correspondan al área de estudio.
    """
    gdf_galicia = descargar_limites_galicia(ruta_limites)
    cube = crear_cubo_rejilla_galicia(gdf_galicia, cell_size_meters, crs_trabajo)
    grid = crear_rejilla_vectorial(cube, cell_size_meters)
    logger.info(
        "Rejilla creada: %s celdas (%s dentro de Galicia).",
        len(grid),
        int(grid["is_galicia"].sum()),
    )
    return cube, grid


def guardar_rejilla_datacube(
    cube: xr.Dataset,
    grid: gpd.GeoDataFrame,
    ruta_cubo: Union[str, Path],
    ruta_vectorial: Union[str, Path],
) -> None:
    """Guarda las dos salidas del notebook: NetCDF y GeoPackage."""
    ruta_cubo = Path(ruta_cubo)
    ruta_vectorial = Path(ruta_vectorial)
    ruta_cubo.parent.mkdir(parents=True, exist_ok=True)
    ruta_vectorial.parent.mkdir(parents=True, exist_ok=True)
    cube.to_netcdf(ruta_cubo)
    grid.to_file(ruta_vectorial, driver="GPKG")
    logger.info("Cubo guardado en %s y rejilla vectorial en %s", ruta_cubo, ruta_vectorial)
