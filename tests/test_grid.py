"""Tests de la malla espacial regular del datacube."""

import geopandas as gpd
from shapely.geometry import box

from src.geospatial.grid import crear_cubo_rejilla_galicia, crear_rejilla_vectorial


def test_crear_cubo_conserva_malla_rectangular_y_mascara() -> None:
    """El grid conserva las celdas exteriores y clasifica por el centro."""
    boundary = gpd.GeoDataFrame(geometry=[box(1000, 1000, 2000, 2000)], crs="EPSG:3035")

    cube = crear_cubo_rejilla_galicia(boundary, cell_size_meters=1000)

    assert cube.attrs["crs"] == "EPSG:3035"
    assert cube.x.values.tolist() == [1000.0, 2000.0]
    assert cube.y.values.tolist() == [1000.0, 2000.0]
    assert cube["is_galicia"].values.tolist() == [[1, 0], [0, 0]]

    grid = crear_rejilla_vectorial(cube, cell_size_meters=1000)
    assert len(grid) == 4
    assert grid["is_galicia"].tolist() == [1, 0, 0, 0]
    assert grid["cell_id"].tolist() == [0, 1, 2, 3]
