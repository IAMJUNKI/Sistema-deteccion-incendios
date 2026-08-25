import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import LineString, Polygon, box

from src.geospatial.human_activity import (
    calcular_variables_actividad_humana,
    crear_datacubo_actividad_humana,
)


def test_calcular_actividad_humana_y_crear_cubo() -> None:
    grid = gpd.GeoDataFrame(
        {
            "cell_id": [0, 1, 2, 3],
            "is_galicia": [1, 1, 1, 1],
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2), box(1, 1, 2, 2)],
        crs="EPSG:3035",
    )
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0.5), (2, 0.5)])], crs=grid.crs)
    residential = gpd.GeoDataFrame(
        geometry=[Polygon([(1.2, 1.2), (1.8, 1.2), (1.8, 1.8), (1.2, 1.8)])],
        crs=grid.crs,
    )

    activity = calcular_variables_actividad_humana(grid, roads, residential)
    cube = crear_datacubo_actividad_humana(
        xr.Dataset(
            {"is_galicia": (("y", "x"), np.ones((2, 2), dtype=np.uint8))},
            coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
            attrs={"crs": "EPSG:3035"},
        ),
        activity,
    )

    assert activity.loc[activity.cell_id == 0, "distance_to_road_m"].item() == 0
    assert activity.loc[activity.cell_id == 0, "road_length_km"].item() == 0.001
    assert activity.loc[activity.cell_id == 3, "distance_to_residential_area_m"].item() == 0
    assert cube["road_length_km"].shape == (2, 2)
    assert cube["distance_to_road_m"].sel(y=0, x=0).item() == 0
