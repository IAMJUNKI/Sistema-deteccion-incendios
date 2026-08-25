"""Capa estática de proximidad humana derivada de OpenStreetMap.

Las variables se calculan una única vez para la malla de Galicia. La fuente se
descarga solo si falta y está fijada a un extracto histórico de Geofabrik para
evitar que la capa cambie cuando OpenStreetMap se actualiza.
"""

import logging
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr

logger = logging.getLogger(__name__)

OSM_GALICIA_2022_URL = "https://download.geofabrik.de/europe/spain/galicia-220101-free.shp.zip"
OSM_GALICIA_2022_ARCHIVE_NAME = "galicia-220101-free.shp.zip"
OSM_GALICIA_2022_EXTRACTED_DIR_NAME = "galicia-220101-free-shp"
HUMAN_ACTIVITY_VARIABLES = (
    "distance_to_road_m",
    "road_length_km",
    "distance_to_residential_area_m",
)
ROAD_CLASSES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
}
SETTLEMENT_CLASSES = {"city", "town", "village", "hamlet"}


def descargar_extracto_osm_galicia(
    raw_dir: str | Path,
    url: str = OSM_GALICIA_2022_URL,
) -> Path:
    """Descarga y extrae el Shapefile versionado de Galicia si falta.

    Args:
        raw_dir: Directorio local para el archivo original y las capas Shapefile.
        url: URL fija del extracto OSM de Geofabrik.

    Returns:
        Directorio con las capas Shapefile extraídas.
    """
    raw_dir = Path(raw_dir)
    archive_path = raw_dir / OSM_GALICIA_2022_ARCHIVE_NAME
    extracted_dir = raw_dir / OSM_GALICIA_2022_EXTRACTED_DIR_NAME
    if extracted_dir.exists() and any(extracted_dir.rglob("*.shp")):
        return extracted_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        logger.info("Descargando extracto OSM versionado desde %s", url)
        temporary = archive_path.with_suffix(".partial")
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            temporary.replace(archive_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    extracted_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (extracted_dir / member.filename).resolve()
            if not destination.is_relative_to(extracted_dir.resolve()):
                raise ValueError("El archivo OSM contiene una ruta no segura.")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
    if not any(extracted_dir.rglob("*.shp")):
        raise ValueError("El archivo OSM extraído no contiene capas Shapefile.")
    return extracted_dir


def _leer_capa_osm(extracted_dir: Path, suffix: str) -> gpd.GeoDataFrame:
    """Lee una capa Shapefile de Geofabrik por su sufijo estable."""
    matches = list(extracted_dir.rglob(f"*{suffix}.shp"))
    if len(matches) != 1:
        available = sorted(path.name for path in extracted_dir.rglob("*.shp"))
        raise ValueError(f"No se encontró una capa OSM única con sufijo {suffix!r}: {available}")
    return gpd.read_file(matches[0])


def _filtrar_clases(frame: gpd.GeoDataFrame, allowed: set[str]) -> gpd.GeoDataFrame:
    """Filtra una capa Geofabrik por ``fclass`` con una validación explícita."""
    if "fclass" not in frame:
        raise ValueError("La capa OSM no contiene la columna 'fclass'.")
    result = frame.loc[frame["fclass"].isin(allowed)].copy()
    if result.empty:
        raise ValueError(f"No hay geometrías OSM para las clases: {sorted(allowed)}")
    return result.loc[result.geometry.notna() & ~result.geometry.is_empty].copy()


def _longitud_vias_por_celda(grid: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> pd.Series:
    """Suma la longitud de las carreteras dentro de cada polígono de celda."""
    grid_positions, road_positions = roads.sindex.query(grid.geometry, predicate="intersects")
    if len(grid_positions) == 0:
        return pd.Series(dtype=np.float64)
    intersections = gpd.GeoSeries(
        grid.geometry.iloc[grid_positions].array.intersection(
            roads.geometry.iloc[road_positions].array
        ),
        crs=grid.crs,
    )
    lengths = pd.DataFrame(
        {
            "cell_id": grid["cell_id"].iloc[grid_positions].to_numpy(),
            "length_m": intersections.length.to_numpy(),
        }
    )
    return lengths.groupby("cell_id")["length_m"].sum() / 1000.0


def calcular_variables_actividad_humana(
    grid: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    residential_areas: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Calcula proximidad y densidad de red viaria por celda activa.

    Las distancias se calculan entre polígonos, por lo que valen cero si una
    carretera o área residencial intersecta la celda.
    """
    if grid.crs is None:
        raise ValueError("La rejilla debe tener CRS proyectado para medir distancias.")
    active_grid = grid.loc[grid["is_galicia"].astype(bool), ["cell_id", "geometry"]].copy()
    roads = roads.to_crs(active_grid.crs)
    residential_areas = residential_areas.to_crs(active_grid.crs)

    road_nearest = gpd.sjoin_nearest(
        active_grid, roads[["geometry"]], how="left", distance_col="distance_to_road_m"
    )
    residential_nearest = gpd.sjoin_nearest(
        active_grid,
        residential_areas[["geometry"]],
        how="left",
        distance_col="distance_to_residential_area_m",
    )
    result = active_grid[["cell_id"]].copy()
    result = result.merge(
        road_nearest.groupby("cell_id", as_index=False)["distance_to_road_m"].min(),
        on="cell_id",
        how="left",
    )
    result = result.merge(
        residential_nearest.groupby("cell_id", as_index=False)[
            "distance_to_residential_area_m"
        ].min(),
        on="cell_id",
        how="left",
    )
    result = result.merge(
        _longitud_vias_por_celda(active_grid, roads).rename("road_length_km"),
        on="cell_id",
        how="left",
    )
    result["road_length_km"] = result["road_length_km"].fillna(0.0)
    if result[list(HUMAN_ACTIVITY_VARIABLES)].isna().any().any():
        raise ValueError("Las capas OSM no cubren todas las celdas activas de Galicia.")
    return result[["cell_id", *HUMAN_ACTIVITY_VARIABLES]]


def crear_datacubo_actividad_humana(
    spatial_cube: xr.Dataset, activity: pd.DataFrame
) -> xr.Dataset:
    """Convierte las variables estáticas por ``cell_id`` a un cubo ``(y, x)``."""
    if "is_galicia" not in spatial_cube:
        raise ValueError("El cubo espacial debe incluir la máscara is_galicia.")
    missing = set(HUMAN_ACTIVITY_VARIABLES) - set(activity.columns)
    if missing:
        raise ValueError(f"Faltan variables de actividad humana: {sorted(missing)}")
    ny, nx = spatial_cube["is_galicia"].shape
    rows, columns = np.divmod(activity["cell_id"].to_numpy(dtype=np.int64), nx)
    if (rows < 0).any() or (rows >= ny).any():
        raise ValueError("Los cell_id de actividad humana no pertenecen a la malla.")

    result = xr.Dataset(coords={"y": spatial_cube.y, "x": spatial_cube.x})
    for variable in HUMAN_ACTIVITY_VARIABLES:
        values = np.full((ny, nx), np.nan, dtype=np.float32)
        values[rows, columns] = activity[variable].to_numpy(dtype=np.float32)
        result[variable] = (("y", "x"), values)
    result.attrs = {
        "title": "Human activity variables",
        "source": "OpenStreetMap via Geofabrik, Galicia snapshot 2022-01-01",
        "module": "human_activity",
        "crs": spatial_cube.attrs.get("crs"),
    }
    result["distance_to_road_m"].attrs = {
        "long_name": "Distance to selected road",
        "units": "m",
        "description": "Minimum distance from the cell polygon to a selected road; 0 if intersecting.",
    }
    result["road_length_km"].attrs = {
        "long_name": "Selected road length within grid cell",
        "units": "km",
        "description": "Total selected-road length intersecting the cell polygon.",
    }
    result["distance_to_residential_area_m"].attrs = {
        "long_name": "Distance to residential area or settlement",
        "units": "m",
        "description": "Minimum distance from the cell polygon to an OSM residential area or settlement.",
    }
    return result


def construir_capa_actividad_humana(
    grid_path: str | Path,
    spatial_cube_path: str | Path,
    output_path: str | Path,
    raw_dir: str | Path,
    osm_url: str = OSM_GALICIA_2022_URL,
) -> Path:
    """Descarga, procesa y guarda la capa estática de actividad humana."""
    extracted_dir = descargar_extracto_osm_galicia(raw_dir, osm_url)
    grid = gpd.read_file(grid_path)
    roads = _filtrar_clases(_leer_capa_osm(extracted_dir, "roads_free_1"), ROAD_CLASSES)
    landuse = _filtrar_clases(_leer_capa_osm(extracted_dir, "landuse_a_free_1"), {"residential"})
    places = _filtrar_clases(_leer_capa_osm(extracted_dir, "places_free_1"), SETTLEMENT_CLASSES)
    residential_areas = pd.concat([landuse, places], ignore_index=True)
    residential_areas = gpd.GeoDataFrame(residential_areas, geometry="geometry", crs=landuse.crs)
    activity = calcular_variables_actividad_humana(grid, roads, residential_areas)
    with xr.open_dataset(spatial_cube_path) as spatial:
        cube = crear_datacubo_actividad_humana(spatial, activity)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cube.to_netcdf(output_path)
    return output_path
