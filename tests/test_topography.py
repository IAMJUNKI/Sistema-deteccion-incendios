"""Tests de agregación y reconstrucción del datacubo topográfico."""

import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

from src.geospatial.topography import (
    TOPOGRAPHY_VARIABLES,
    crear_datacubo_topografia,
    extraer_estadisticas_topograficas_rapidas,
)


def test_agregacion_topografica_y_datacubo() -> None:
    """Las estadísticas y proporciones llegan a las posiciones correctas del cubo."""
    grid = gpd.GeoDataFrame(
        {"cell_id": [0, 1]},
        geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000)],
        crs="EPSG:3035",
    )
    profile = {"height": 10, "width": 20, "transform": from_origin(0, 1000, 100, 100)}
    elevation = np.arange(200, dtype=np.float32).reshape(10, 20)
    slope = np.full((10, 20), 10.0, dtype=np.float32)
    roughness = np.full((10, 20), 2.0, dtype=np.float32)
    aspect = np.full((10, 20), 30.0, dtype=np.float32)
    aspect[:, 10:] = 90.0

    dataframe = extraer_estadisticas_topograficas_rapidas(
        grid, elevation, slope, aspect, roughness, profile
    )
    assert set(TOPOGRAPHY_VARIABLES).issubset(dataframe.columns)
    assert dataframe.loc[0, "aspect_000_045"] == 1.0
    assert dataframe.loc[1, "aspect_045_090"] == 1.0
    assert dataframe.loc[0, "elevation_std"] > 0

    cube = xr.Dataset(coords={"x": [0.0, 1000.0], "y": [0.0]})
    cube["is_galicia"] = (("y", "x"), np.array([[1, 1]], dtype=np.uint8))
    cube.attrs = {"crs": "EPSG:3035", "spatial_resolution": "1000 m"}
    topography = crear_datacubo_topografia(cube, dataframe)

    assert topography["elevation_mean"].shape == (1, 2)
    assert topography["aspect_000_045"].values.tolist() == [[1.0, 0.0]]
    assert topography["aspect_045_090"].values.tolist() == [[0.0, 1.0]]
