"""Variables topográficas estáticas para la malla regular EPSG:3035.

El módulo agrega un DEM de 30 m a la malla de 1 km y genera las quince
variables topográficas usadas como referencia en IberFire: estadísticas de
elevación, pendiente y rugosidad, además de ocho proporciones de orientación
y la proporción sin orientación válida.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.features import rasterize
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.ndimage import uniform_filter

logger = logging.getLogger(__name__)

NODATA_VALUE = -9999.0
ASPECT_RANGES = (
    (0, 45),
    (45, 90),
    (90, 135),
    (135, 180),
    (180, 225),
    (225, 270),
    (270, 315),
    (315, 360),
)
ASPECT_VARIABLES = [f"aspect_{start:03d}_{end:03d}_fraction" for start, end in ASPECT_RANGES]
ASPECT_NODATA_VARIABLE = "aspect_no_data_fraction"
TOPOGRAPHY_VARIABLES = [
    "elevation_mean",
    "elevation_std",
    "slope_mean",
    "slope_std",
    "roughness_mean",
    "roughness_std",
    *ASPECT_VARIABLES,
    ASPECT_NODATA_VARIABLE,
]
# Contrato de almacenamiento, no selección de modelo.  Cambiar un valor permite
# regenerar una capa/cubo de prueba sin tocar el resto de la ingeniería.
DATACUBE_VARIABLE_FLAGS = {name: True for name in TOPOGRAPHY_VARIABLES}
TEST_DATACUBE_VARIABLE_FLAGS = {
    "elevation_mean": True, "elevation_std": True, "slope_mean": True, "slope_std": True,
    "roughness_mean": False, "roughness_std": False,
    **{name: True for name in ASPECT_VARIABLES}, ASPECT_NODATA_VARIABLE: False,
}
TOPOGRAPHY_METADATA = {
    "elevation_mean": ("Mean terrain elevation", "m"),
    "elevation_std": ("Terrain elevation standard deviation", "m"),
    "slope_mean": ("Mean terrain slope", "degrees"),
    "slope_std": ("Terrain slope standard deviation", "degrees"),
    "roughness_mean": ("Mean local terrain roughness", "m"),
    "roughness_std": ("Local terrain roughness standard deviation", "m"),
    ASPECT_NODATA_VARIABLE: ("Fraction of pixels without valid terrain aspect", "fraction"),
}
TOPOGRAPHY_METADATA.update(
    {
        variable: (f"Fraction of pixels with aspect from {start} to {end} degrees", "fraction")
        for variable, (start, end) in zip(ASPECT_VARIABLES, ASPECT_RANGES, strict=True)
    }
)


def descargar_dem_galicia(
    bounds_wgs84: tuple[float, float, float, float],
) -> tuple[np.ndarray, dict]:
    """Descarga Copernicus DEM GLO-30 para un bounding box en WGS84."""
    try:
        from dem_stitcher import stitch_dem
    except ImportError as error:
        raise ImportError("Instala dem-stitcher para descargar Copernicus DEM GLO-30.") from error

    dem_array, dem_profile = stitch_dem(
        bounds_wgs84,
        dem_name="glo_30",
        dst_ellipsoidal_height=False,
        dst_area_or_point="Point",
    )
    logger.info("DEM descargado. Dimensiones originales: %s", dem_array.shape)
    return dem_array, dem_profile


def reproyectar_raster_utm(
    dem_array: np.ndarray,
    perfil_src: dict,
    crs_destino: str = "EPSG:3035",
    resolucion_destino: float = 30.0,
) -> tuple[np.ndarray, dict]:
    """Reproyecta un DEM a un CRS métrico, conservando una resolución dada."""
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
            "nodata": NODATA_VALUE,
        }
    )
    dem_reproyectado = np.full((height, width), NODATA_VALUE, dtype=np.float32)
    reproject(
        source=dem_array,
        destination=dem_reproyectado,
        src_transform=perfil_src["transform"],
        src_crs=perfil_src["crs"],
        dst_transform=transform,
        dst_crs=crs_destino,
        resampling=Resampling.bilinear,
        src_nodata=perfil_src.get("nodata", NODATA_VALUE),
        dst_nodata=NODATA_VALUE,
    )
    return dem_reproyectado, perfil_dst


def calcular_pendiente_y_orientacion(
    dem_array: np.ndarray, transform: rasterio.Affine
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula pendiente y orientación a partir de un DEM en unidades métricas."""
    dem = dem_array.astype(np.float32, copy=True)
    dem[dem == NODATA_VALUE] = np.nan
    dx = transform.a
    dy = abs(transform.e)
    dzdx = np.gradient(dem, dx, axis=1)
    dzdy = np.gradient(dem, dy, axis=0)

    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = np.degrees(np.arctan2(-dzdx, dzdy)) % 360
    return (
        np.nan_to_num(slope, nan=NODATA_VALUE).astype(np.float32),
        np.nan_to_num(aspect, nan=NODATA_VALUE).astype(np.float32),
    )


def calcular_rugosidad_local(dem_array: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Calcula la desviación estándar local del DEM como medida de rugosidad.

    IberFire emplea una capa de rugosidad externa cuyo algoritmo no se publica.
    Esta implementación documentada usa la desviación estándar en una ventana
    de ``window_size`` píxeles como aproximación reproducible.
    """
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size debe ser un entero impar positivo.")

    valid = np.isfinite(dem_array) & (dem_array != NODATA_VALUE)
    values = np.where(valid, dem_array, 0.0).astype(np.float32)
    weights = valid.astype(np.float32)
    count = uniform_filter(weights, size=window_size, mode="nearest")
    mean = uniform_filter(values, size=window_size, mode="nearest") / np.where(
        count == 0, 1, count
    )
    mean_sq = uniform_filter(values**2, size=window_size, mode="nearest") / np.where(
        count == 0, 1, count
    )
    roughness = np.sqrt(np.maximum(mean_sq - mean**2, 0.0))
    return np.where(valid, roughness, NODATA_VALUE).astype(np.float32)


def clasificar_aspecto(orientacion_array: np.ndarray) -> np.ndarray:
    """Asigna las ocho clases de 45 grados especificadas por IberFire.

    Las clases siguen los intervalos ``(0, 45]``, ..., ``(315, 360]``.
    El valor 0 grados es equivalente a 360 y se asigna a la clase 8.
    """
    classes = np.full(orientacion_array.shape, -1, dtype=np.int8)
    valid = np.isfinite(orientacion_array) & (orientacion_array != NODATA_VALUE)
    angles = np.mod(orientacion_array[valid], 360.0)
    values = np.ceil(angles / 45.0).astype(np.int8)
    values[values == 0] = 8
    classes[valid] = values
    return classes


def _rasterizar_celdas(gdf_grid: gpd.GeoDataFrame, perfil_raster: dict) -> np.ndarray:
    """Rasteriza identificadores de celda en la rejilla del raster topográfico."""
    return rasterize(
        shapes=(
            (geometry, int(cell_id))
            for geometry, cell_id in zip(gdf_grid.geometry, gdf_grid.cell_id)
        ),
        out_shape=(perfil_raster["height"], perfil_raster["width"]),
        transform=perfil_raster["transform"],
        fill=-1,
        dtype=np.int32,
    )


def extraer_estadisticas_topograficas_rapidas(
    gdf_grid: gpd.GeoDataFrame,
    dem_array: np.ndarray,
    pendiente_array: np.ndarray,
    orientacion_array: np.ndarray,
    rugosidad_array: np.ndarray,
    perfil_raster: dict,
) -> pd.DataFrame:
    """Agrega las quince variables topográficas a cada celda activa de 1 km."""
    grid_raster = _rasterizar_celdas(gdf_grid, perfil_raster)
    in_grid = grid_raster != -1
    pixels = pd.DataFrame(
        {
            "cell_id": grid_raster[in_grid].astype(np.int64),
            "elevation": dem_array[in_grid],
            "slope": pendiente_array[in_grid],
            "roughness": rugosidad_array[in_grid],
            "aspect_class": clasificar_aspecto(orientacion_array)[in_grid],
        }
    )
    result = pd.DataFrame({"cell_id": gdf_grid["cell_id"].to_numpy(dtype=np.int64)})
    valid_stats = pixels[
        (pixels["elevation"] != NODATA_VALUE)
        & (pixels["slope"] != NODATA_VALUE)
        & (pixels["roughness"] != NODATA_VALUE)
    ]
    statistics = (
        valid_stats.groupby("cell_id")
        .agg(
            elevation_mean=("elevation", "mean"),
            elevation_std=("elevation", "std"),
            slope_mean=("slope", "mean"),
            slope_std=("slope", "std"),
            roughness_mean=("roughness", "mean"),
            roughness_std=("roughness", "std"),
        )
        .reset_index()
    )
    result = result.merge(statistics, on="cell_id", how="left")

    total = pixels.groupby("cell_id").size().rename("total")
    valid_aspect = pixels[pixels["aspect_class"] != -1]
    valid_count = valid_aspect.groupby("cell_id").size().rename("valid")
    result = result.merge(total, on="cell_id", how="left").merge(
        valid_count, on="cell_id", how="left"
    )
    result["valid"] = result["valid"].fillna(0)
    for class_index, variable in enumerate(ASPECT_VARIABLES, start=1):
        count = (
            (valid_aspect["aspect_class"] == class_index).groupby(valid_aspect["cell_id"]).sum()
        )
        result = result.merge(count.rename(variable), on="cell_id", how="left")
        result[variable] = result[variable].fillna(0) / result["valid"].replace(0, np.nan)
    result[ASPECT_NODATA_VARIABLE] = 1 - result["valid"] / result["total"].replace(0, np.nan)

    # Alias de compatibilidad para los consumidores vectoriales preexistentes.
    result["orientacion_media"] = np.nan
    result["orientacion_clase"] = (
        result[ASPECT_VARIABLES].idxmax(axis=1).where(result["valid"] > 0)
    )
    result = result.drop(columns=["total", "valid"])
    result["altitud_media"] = result["elevation_mean"]
    result["pendiente_media"] = result["slope_mean"]
    return result


def crear_datacubo_topografia(
    cube: xr.Dataset, topografia: pd.DataFrame, inclusion_flags: dict[str, bool] | None = None
) -> xr.Dataset:
    """Inserta las variables topográficas por ``cell_id`` en el cubo espacial."""
    if "is_galicia" not in cube:
        raise ValueError("El cubo debe incluir la máscara is_galicia.")
    flags = DATACUBE_VARIABLE_FLAGS if inclusion_flags is None else inclusion_flags
    selected = [name for name in TOPOGRAPHY_VARIABLES if flags.get(name, False)]
    missing = set(selected) - set(topografia.columns)
    if missing:
        raise ValueError(f"Faltan variables topográficas: {sorted(missing)}")

    topography = cube.copy()
    ny, nx = cube["is_galicia"].shape
    cell_ids = topografia["cell_id"].to_numpy(dtype=np.int64)
    if (cell_ids < 0).any() or (cell_ids >= ny * nx).any():
        raise ValueError("Los cell_id no pertenecen a la malla del cubo.")
    rows, columns = np.divmod(cell_ids, nx)
    for variable in selected:
        values = np.full((ny, nx), np.nan, dtype=np.float32)
        values[rows, columns] = topografia[variable].to_numpy(dtype=np.float32)
        topography[variable] = (("y", "x"), values)

    topography.attrs = {
        "title": "Topographic variables",
        "description": "Topographic variables aggregated to the 1 km spatial grid.",
        "module": "topography",
        "source": "Copernicus DEM GLO-30",
        "crs": cube.attrs["crs"],
        "spatial_resolution": cube.attrs["spatial_resolution"],
        "roughness_method": "3x3 local elevation standard deviation",
    }
    for variable in selected:
        long_name, units = TOPOGRAPHY_METADATA[variable]
        topography[variable].attrs = {"long_name": long_name, "units": units}
    return topography


def guardar_datacubo_topografia(topography: xr.Dataset, ruta_salida: Path) -> None:
    """Guarda el datacubo topográfico en formato NetCDF."""
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    topography.to_netcdf(ruta_salida)
