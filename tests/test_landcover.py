"""Tests de agregación de CORINE Land Cover al datacube espacial."""

import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

from src.geospatial.vegetation import (
    LANDCOVER_VARIABLES,
    RAW_CLC_TO_LANDCOVER,
    crear_datacubo_cobertura_suelo,
    extraer_variables_cobertura_suelo,
)


def test_agregacion_corine_y_datacubo() -> None:
    """Las proporciones CORINE se agregan y llegan a las celdas correctas."""
    grid = gpd.GeoDataFrame(
        {"cell_id": [0, 1]},
        geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000)],
        crs="EPSG:3035",
    )
    profile = {
        "height": 10,
        "width": 20,
        "transform": from_origin(0, 1000, 100, 100),
        "nodata": -128,
    }
    corine = np.full((10, 20), 23, dtype=np.int16)
    corine[:, 10:15] = 24
    corine[:, 15:] = 25
    dataframe = extraer_variables_cobertura_suelo(grid, corine, profile)

    assert set(LANDCOVER_VARIABLES).issubset(dataframe.columns)
    assert dataframe.loc[0, "broadleaf_forest"] == 1.0
    assert dataframe.loc[1, "coniferous_forest"] == 0.5
    assert dataframe.loc[1, "mixed_forest"] == 0.5
    assert np.allclose(dataframe[LANDCOVER_VARIABLES].sum(axis=1), 1.0)
    assert dataframe.loc[0, "combustible_clase"] == "bosque_frondosas"
    assert dataframe.loc[1, "combustible_pct_forestal"] == 100.0

    cube = xr.Dataset(coords={"x": [0.0, 1000.0], "y": [0.0]})
    cube["is_galicia"] = (("y", "x"), np.array([[1, 1]], dtype=np.uint8))
    cube.attrs = {"crs": "EPSG:3035", "spatial_resolution": "1000 m"}
    landcover = crear_datacubo_cobertura_suelo(cube, dataframe)
    assert landcover["broadleaf_forest"].values.tolist() == [[1.0, 0.0]]
    assert landcover["mixed_forest"].values.tolist() == [[0.0, 0.5]]
    assert landcover["water"].attrs["units"] == "fraction"


def test_mapeo_corine_cubre_las_44_clases_oficiales() -> None:
    """La codificación 1--44 del GeoTIFF oficial no deja clases sin categoría."""
    assert set(RAW_CLC_TO_LANDCOVER) == set(range(1, 45))
