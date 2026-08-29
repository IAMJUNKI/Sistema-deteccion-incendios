"""Agregación de CORINE Land Cover al datacube espacial de 1 km.

Admite el GeoTIFF oficial CLC 2018 (índices 1--44) y rasters preprocesados
con códigos CLC de tres dígitos. Genera proporciones en dimensiones ``(y, x)``.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from affine import Affine
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

logger = logging.getLogger(__name__)

LANDCOVER_VARIABLES = [
    "artificial",
    "agriculture",
    "broadleaf_forest",
    "coniferous_forest",
    "mixed_forest",
    "scrub",
    "open_spaces",
    "wetlands",
    "water",
]
FOREST_COVER_VARIABLE = "forest_cover_fraction"
CANONICAL_DATACUBE_VARIABLE_FLAGS = {
    **{name: True for name in LANDCOVER_VARIABLES},
    FOREST_COVER_VARIABLE: False,
}
DATACUBE_VARIABLE_FLAGS = CANONICAL_DATACUBE_VARIABLE_FLAGS
LANDCOVER_METADATA = {
    "artificial": "Artificial surfaces",
    "agriculture": "Agricultural areas",
    "broadleaf_forest": "Broadleaf forest",
    "coniferous_forest": "Coniferous forest",
    "mixed_forest": "Mixed forest",
    "scrub": "Natural grassland, heathland, scrub and transitional woodland",
    "open_spaces": "Open, bare, burned or snow-covered spaces",
    "wetlands": "Wetlands",
    "water": "Water bodies and sea",
}

# Índices de las 44 clases del GeoTIFF oficial U2018_CLC2018_V2020_20u1.tif.
RAW_CLC_TO_LANDCOVER = {
    **{code: "artificial" for code in range(1, 12)},
    **{code: "agriculture" for code in range(12, 23)},
    23: "broadleaf_forest",
    24: "coniferous_forest",
    25: "mixed_forest",
    **{code: "scrub" for code in range(26, 30)},
    **{code: "open_spaces" for code in range(30, 35)},
    **{code: "wetlands" for code in range(35, 40)},
    **{code: "water" for code in range(40, 45)},
}
CLC_CODE_TO_LANDCOVER = {
    **{code: "artificial" for code in (111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142)},
    **{code: "agriculture" for code in (211, 212, 213, 221, 222, 223, 231, 241, 242, 243, 244)},
    311: "broadleaf_forest",
    312: "coniferous_forest",
    313: "mixed_forest",
    **{code: "scrub" for code in (321, 322, 323, 324)},
    **{code: "open_spaces" for code in (331, 332, 333, 334, 335)},
    **{code: "wetlands" for code in (411, 412, 421, 422, 423)},
    **{code: "water" for code in (511, 512, 521, 522, 523)},
}
COMBUSTIBLE_NAMES = {
    "artificial": "urbano",
    "agriculture": "agricola",
    "broadleaf_forest": "bosque_frondosas",
    "coniferous_forest": "bosque_coniferas",
    "mixed_forest": "bosque_mixto",
    "scrub": "matorral_pastizal",
    "open_spaces": "espacios_abiertos",
    "wetlands": "humedal",
    "water": "agua",
}


def cargar_corine_para_rejilla(
    ruta_clc: str | Path,
    gdf_grid: gpd.GeoDataFrame,
    crs_destino: str = "EPSG:3035",
    resolucion_destino: float = 100.0,
) -> tuple[np.ndarray, dict]:
    """Lee solo el entorno de la malla y reproyecta si hace falta."""
    ruta_clc = Path(ruta_clc)
    if not ruta_clc.exists():
        raise FileNotFoundError(f"No se encontró el archivo CORINE: {ruta_clc}")
    if gdf_grid.empty:
        raise ValueError("La rejilla activa no puede estar vacía.")

    with rasterio.open(ruta_clc) as source:
        if source.crs is None:
            raise ValueError("El raster CORINE no declara CRS.")
        grid_bounds = gdf_grid.to_crs(crs_destino).total_bounds
        source_bounds = transform_bounds(crs_destino, source.crs, *grid_bounds, densify_pts=21)
        inverse_transform = ~source.transform
        col_start_float, row_start_float = inverse_transform * (
            source_bounds[0],
            source_bounds[3],
        )
        col_stop_float, row_stop_float = inverse_transform * (
            source_bounds[2],
            source_bounds[1],
        )
        col_start = max(0, int(np.floor(min(col_start_float, col_stop_float))))
        row_start = max(0, int(np.floor(min(row_start_float, row_stop_float))))
        col_stop = min(source.width, int(np.ceil(max(col_start_float, col_stop_float))))
        row_stop = min(source.height, int(np.ceil(max(row_start_float, row_stop_float))))
        if col_start >= col_stop or row_start >= row_stop:
            raise ValueError("La rejilla no se solapa con el raster CORINE.")
        # El GeoTIFF descargado para España está comprimido. En esta instalación
        # de GDAL, las ventanas internas fallan silenciosamente, mientras que la
        # lectura completa seguida de un corte NumPy es estable y reproducible.
        array = source.read(1)[row_start:row_stop, col_start:col_stop]
        transform = source.transform * Affine.translation(col_start, row_start)
        profile = source.profile.copy()
        profile.update({"height": array.shape[0], "width": array.shape[1], "transform": transform})

    if str(profile["crs"]) == crs_destino and np.isclose(abs(transform.a), resolucion_destino):
        return array, profile

    bounds = array_bounds(array.shape[0], array.shape[1], transform)
    dst_transform, width, height = calculate_default_transform(
        profile["crs"],
        crs_destino,
        array.shape[1],
        array.shape[0],
        *bounds,
        resolution=resolucion_destino,
    )
    nodata = profile.get("nodata")
    destination = np.full((height, width), nodata if nodata is not None else 0, dtype=array.dtype)
    reproject(
        source=array,
        destination=destination,
        src_transform=transform,
        src_crs=profile["crs"],
        src_nodata=nodata,
        dst_transform=dst_transform,
        dst_crs=crs_destino,
        dst_nodata=nodata if nodata is not None else 0,
        resampling=Resampling.nearest,
    )
    profile.update(
        {
            "crs": crs_destino,
            "height": height,
            "width": width,
            "transform": dst_transform,
            "nodata": nodata if nodata is not None else 0,
        }
    )
    return destination, profile


def _mapear_codigos_clc(clc_array: np.ndarray, nodata: float | int | None) -> np.ndarray:
    """Convierte códigos CLC en índices de macroclase; -1 significa dato inválido."""
    mapped = np.full(clc_array.shape, -1, dtype=np.int8)
    valid = np.isfinite(clc_array)
    if nodata is not None:
        valid &= clc_array != nodata
    codes = clc_array[valid].astype(np.int32)
    mapping = RAW_CLC_TO_LANDCOVER if codes.size and codes.max() <= 48 else CLC_CODE_TO_LANDCOVER
    category_index = {category: index for index, category in enumerate(LANDCOVER_VARIABLES)}
    maximum_code = max(mapping)
    lookup = np.full(maximum_code + 1, -1, dtype=np.int8)
    for code, category in mapping.items():
        lookup[code] = category_index[category]
    in_range = valid & (clc_array >= 0) & (clc_array <= maximum_code)
    mapped[in_range] = lookup[clc_array[in_range].astype(np.int32)]
    return mapped


def extraer_variables_cobertura_suelo(
    gdf_grid: gpd.GeoDataFrame, clc_array: np.ndarray, perfil_raster: dict
) -> pd.DataFrame:
    """Calcula las proporciones de las nueve macroclases CORINE por celda."""
    grid_raster = rasterize(
        shapes=(
            (geometry, int(cell_id))
            for geometry, cell_id in zip(gdf_grid.geometry, gdf_grid.cell_id)
        ),
        out_shape=(perfil_raster["height"], perfil_raster["width"]),
        transform=perfil_raster["transform"],
        fill=-1,
        dtype=np.int32,
    )
    mapped = _mapear_codigos_clc(clc_array, perfil_raster.get("nodata"))
    valid = (grid_raster >= 0) & (mapped >= 0)
    cell_ids = gdf_grid["cell_id"].to_numpy(dtype=np.int64)
    totals = np.bincount(grid_raster[valid], minlength=int(cell_ids.max()) + 1)
    result = pd.DataFrame({"cell_id": cell_ids})
    for category_index, variable in enumerate(LANDCOVER_VARIABLES):
        counts = np.bincount(
            grid_raster[valid & (mapped == category_index)], minlength=len(totals)
        )
        result[variable] = np.divide(
            counts[cell_ids],
            totals[cell_ids],
            out=np.full(len(cell_ids), np.nan, dtype=np.float32),
            where=totals[cell_ids] > 0,
        ).astype(np.float32)
    result["combustible_pct_forestal"] = (
        result[["broadleaf_forest", "coniferous_forest", "mixed_forest"]].sum(axis=1) * 100.0
    )
    result[FOREST_COVER_VARIABLE] = result["combustible_pct_forestal"] / 100.0
    _rellenar_celdas_sin_corine_por_vecino(result, gdf_grid)
    result["combustible_pct_forestal"] = (
        result[["broadleaf_forest", "coniferous_forest", "mixed_forest"]].sum(axis=1) * 100.0
    )
    result[FOREST_COVER_VARIABLE] = result["combustible_pct_forestal"] / 100.0
    fractions = result[LANDCOVER_VARIABLES]
    has_landcover = fractions.notna().any(axis=1)
    dominant = fractions.fillna(-np.inf).idxmax(axis=1).where(has_landcover)
    result["combustible_clase"] = dominant.map(COMBUSTIBLE_NAMES)
    return result


def _rellenar_celdas_sin_corine_por_vecino(
    landcover: pd.DataFrame, gdf_grid: gpd.GeoDataFrame
) -> None:
    """Completa celdas CORINE sin píxeles válidos con su vecino válido más cercano.

    Se copia el vector completo de nueve fracciones para mantener una composición
    de cobertura coherente, especialmente en las escasas celdas costeras cuyo
    centro pertenece a Galicia pero no intersecta ningún píxel CORINE válido.
    """
    missing = landcover[LANDCOVER_VARIABLES].isna().all(axis=1)
    if not missing.any():
        return

    centroids = gdf_grid.set_index("cell_id").geometry.centroid
    valid_ids = landcover.loc[~missing, "cell_id"].to_numpy()
    valid_points = centroids.loc[valid_ids]
    for index, cell_id in landcover.loc[missing, "cell_id"].items():
        distances = valid_points.distance(centroids.loc[cell_id])
        neighbor_id = distances.idxmin()
        neighbor = landcover.loc[landcover["cell_id"] == neighbor_id, LANDCOVER_VARIABLES].iloc[0]
        landcover.loc[index, LANDCOVER_VARIABLES] = neighbor.to_numpy()


def crear_datacubo_cobertura_suelo(
    cube: xr.Dataset, landcover: pd.DataFrame, inclusion_flags: dict[str, bool] | None = None
) -> xr.Dataset:
    """Inserta las proporciones CORINE en un cubo con la malla base."""
    if "is_galicia" not in cube:
        raise ValueError("El cubo debe incluir la máscara is_galicia.")
    flags = DATACUBE_VARIABLE_FLAGS if inclusion_flags is None else inclusion_flags
    selected = [
        name for name in [*LANDCOVER_VARIABLES, FOREST_COVER_VARIABLE] if flags.get(name, False)
    ]
    missing = set(selected) - set(landcover.columns)
    if missing:
        raise ValueError(f"Faltan variables de cobertura del suelo: {sorted(missing)}")
    output = cube.copy()
    ny, nx = cube["is_galicia"].shape
    cell_ids = landcover["cell_id"].to_numpy(dtype=np.int64)
    if (cell_ids < 0).any() or (cell_ids >= ny * nx).any():
        raise ValueError("Los cell_id no pertenecen a la malla del cubo.")
    rows, columns = np.divmod(cell_ids, nx)
    for variable in selected:
        values = np.full((ny, nx), np.nan, dtype=np.float32)
        values[rows, columns] = landcover[variable].to_numpy(dtype=np.float32)
        output[variable] = (("y", "x"), values)
        output[variable].attrs = {
            "long_name": LANDCOVER_METADATA.get(variable, "Total forest cover"),
            "units": "fraction",
            "description": "Fraction of the 1 km cell classified in CORINE Land Cover 2018.",
        }
    output.attrs = {
        "title": "Land cover variables",
        "description": "CORINE Land Cover fractions aggregated to the 1 km spatial grid.",
        "module": "landcover",
        "source": "CORINE Land Cover 2018 V2020_20u1",
        "crs": cube.attrs["crs"],
        "spatial_resolution": cube.attrs["spatial_resolution"],
    }
    return output


def guardar_datacubo_cobertura_suelo(landcover: xr.Dataset, ruta_salida: str | Path) -> None:
    """Guarda las variables de cobertura del suelo en formato NetCDF."""
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    landcover.to_netcdf(ruta_salida)


def extraer_variables_vegetacion(
    gdf_grid: gpd.GeoDataFrame, clc_array: np.ndarray, perfil_raster: dict
) -> pd.DataFrame:
    """Alias de compatibilidad que devuelve las columnas derivadas históricas."""
    variables = extraer_variables_cobertura_suelo(gdf_grid, clc_array, perfil_raster)
    return variables[["cell_id", "combustible_clase", "combustible_pct_forestal"]]
