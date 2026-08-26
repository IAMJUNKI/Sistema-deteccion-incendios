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
    "road_length_main_km",
    "road_length_local_km",
    "road_length_track_km",
    "road_length_other_km",
    "residential_area_fraction",
    "building_area_fraction",
)
ROAD_CLASS_GROUPS = {
    "road_length_main_km": {
        "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
        "secondary", "secondary_link",
    },
    "road_length_local_km": {
        "tertiary", "tertiary_link", "unclassified", "residential", "living_street", "service",
    },
    "road_length_track_km": {"track"},
}
ROAD_CLASSES = set().union(*ROAD_CLASS_GROUPS.values())
DATACUBE_VARIABLE_FLAGS = {name: True for name in HUMAN_ACTIVITY_VARIABLES}
TEST_DATACUBE_VARIABLE_FLAGS = {
    "distance_to_road_m": False,
    "road_length_km": True,
    "distance_to_residential_area_m": False,
    "road_length_main_km": True,
    "road_length_local_km": True,
    "road_length_track_km": True,
    "road_length_other_km": True,
    "residential_area_fraction": True,
    "building_area_fraction": True,
}
LEGACY_ROAD_CLASSES = {
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
    if roads.empty:
        return pd.Series(dtype=np.float64, name="length_m", index=pd.Index([], name="cell_id"))
    partial: list[pd.Series] = []
    for start in range(0, len(grid), 500):
        subset = grid.iloc[start : start + 500]
        grid_positions, road_positions = roads.sindex.query(subset.geometry, predicate="intersects")
        if len(grid_positions) == 0:
            continue
        intersections = subset.geometry.iloc[grid_positions].array.intersection(
            roads.geometry.iloc[road_positions].array
        )
        lengths = pd.DataFrame({
            "cell_id": subset["cell_id"].iloc[grid_positions].to_numpy(),
            "length_m": intersections.length,
        })
        partial.append(lengths.groupby("cell_id")["length_m"].sum())
    if not partial:
        return pd.Series(dtype=np.float64, name="length_m", index=pd.Index([], name="cell_id"))
    return pd.concat(partial).groupby(level=0).sum() / 1000.0


def _fraccion_poligonos_por_celda(grid: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame) -> pd.Series:
    """Calcula fracción de superficie de polígonos, acotada a [0, 1].

    Se procesan bloques de celdas para no materializar todas las intersecciones
    de los más de 300.000 edificios OSM en memoria a la vez.
    """
    if polygons.empty:
        return pd.Series(dtype=np.float64, index=pd.Index([], name="cell_id"))
    partial: list[pd.Series] = []
    cell_areas = grid.set_index("cell_id").geometry.area
    for start in range(0, len(grid), 500):
        subset = grid.iloc[start : start + 500]
        positions_grid, positions_polygon = polygons.sindex.query(subset.geometry, predicate="intersects")
        if len(positions_grid) == 0:
            continue
        intersections = subset.geometry.iloc[positions_grid].array.intersection(
            polygons.geometry.iloc[positions_polygon].array
        )
        areas = pd.DataFrame({
            "cell_id": subset["cell_id"].iloc[positions_grid].to_numpy(),
            "area_m2": intersections.area,
        })
        partial.append(areas.groupby("cell_id")["area_m2"].sum())
    if not partial:
        return pd.Series(dtype=np.float64, index=pd.Index([], name="cell_id"))
    covered = pd.concat(partial).groupby(level=0).sum()
    return (covered / cell_areas.reindex(covered.index)).clip(lower=0.0, upper=1.0)


def calcular_variables_actividad_humana(
    grid: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    residential_areas: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None = None,
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
    buildings = (buildings if buildings is not None else gpd.GeoDataFrame(geometry=[], crs=active_grid.crs)).to_crs(active_grid.crs)
    if "fclass" not in roads:
        roads = roads.assign(fclass="service")

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
    for variable, classes in ROAD_CLASS_GROUPS.items():
        result = result.merge(
            _longitud_vias_por_celda(active_grid, roads.loc[roads["fclass"].isin(classes)]).rename(variable),
            on="cell_id", how="left",
        )
        result[variable] = result[variable].fillna(0.0)
    # La longitud total sigue una definición explícita y comprobable: suma de
    # las categorías anteriores. ``other`` queda reservado a futuras clases OSM.
    result["road_length_other_km"] = 0.0
    road_parts = [*ROAD_CLASS_GROUPS, "road_length_other_km"]
    result["road_length_km"] = result[road_parts].sum(axis=1)
    for variable, polygons in {
        "residential_area_fraction": residential_areas,
        "building_area_fraction": buildings,
    }.items():
        result = result.merge(_fraccion_poligonos_por_celda(active_grid, polygons).rename(variable), on="cell_id", how="left")
        result[variable] = result[variable].fillna(0.0)
    if result[list(HUMAN_ACTIVITY_VARIABLES)].isna().any().any():
        raise ValueError("Las capas OSM no cubren todas las celdas activas de Galicia.")
    return result[["cell_id", *HUMAN_ACTIVITY_VARIABLES]]


def crear_datacubo_actividad_humana(
    spatial_cube: xr.Dataset, activity: pd.DataFrame, inclusion_flags: dict[str, bool] | None = None
) -> xr.Dataset:
    """Convierte las variables estáticas por ``cell_id`` a un cubo ``(y, x)``."""
    if "is_galicia" not in spatial_cube:
        raise ValueError("El cubo espacial debe incluir la máscara is_galicia.")
    flags = DATACUBE_VARIABLE_FLAGS if inclusion_flags is None else inclusion_flags
    selected = [name for name in HUMAN_ACTIVITY_VARIABLES if flags.get(name, False)]
    missing = set(selected) - set(activity.columns)
    if missing:
        raise ValueError(f"Faltan variables de actividad humana: {sorted(missing)}")
    ny, nx = spatial_cube["is_galicia"].shape
    rows, columns = np.divmod(activity["cell_id"].to_numpy(dtype=np.int64), nx)
    if (rows < 0).any() or (rows >= ny).any():
        raise ValueError("Los cell_id de actividad humana no pertenecen a la malla.")

    result = xr.Dataset(coords={"y": spatial_cube.y, "x": spatial_cube.x})
    for variable in selected:
        values = np.full((ny, nx), np.nan, dtype=np.float32)
        values[rows, columns] = activity[variable].to_numpy(dtype=np.float32)
        result[variable] = (("y", "x"), values)
    result.attrs = {
        "title": "Human activity variables",
        "source": "OpenStreetMap via Geofabrik, Galicia snapshot 2022-01-01",
        "module": "human_activity",
        "crs": spatial_cube.attrs.get("crs"),
    }
    metadata = {
    "distance_to_road_m": {
        "long_name": "Distance to selected road",
        "units": "m",
        "description": "Minimum distance from the cell polygon to a selected road; 0 if intersecting.",
    },
    "road_length_km": {
        "long_name": "Selected road length within grid cell",
        "units": "km",
        "description": "Total road length, equal to the sum of main, local, track and other road categories.",
    },
    "distance_to_residential_area_m": {
        "long_name": "Distance to residential area or settlement",
        "units": "m",
        "description": "Minimum distance from the cell polygon to an OSM residential area or settlement.",
    },
    **{name: {"long_name": name.replace("_", " "), "units": "km", "description": "Road length inside the 1 km cell, classified from OSM fclass."} for name in [*ROAD_CLASS_GROUPS, "road_length_other_km"]},
    "residential_area_fraction": {"long_name": "Residential area fraction", "units": "fraction", "description": "Fraction of cell covered by OSM residential land-use polygons."},
    "building_area_fraction": {"long_name": "Building footprint fraction", "units": "fraction", "description": "Fraction of cell covered by OSM building footprints."},
    }
    for variable in selected:
        result[variable].attrs = metadata[variable]
    return result


def construir_capa_actividad_humana(
    grid_path: str | Path,
    spatial_cube_path: str | Path,
    output_path: str | Path,
    raw_dir: str | Path,
    osm_url: str = OSM_GALICIA_2022_URL,
    inclusion_flags: dict[str, bool] | None = None,
) -> Path:
    """Descarga, procesa y guarda la capa estática de actividad humana."""
    extracted_dir = descargar_extracto_osm_galicia(raw_dir, osm_url)
    grid = gpd.read_file(grid_path)
    roads = _filtrar_clases(_leer_capa_osm(extracted_dir, "roads_free_1"), ROAD_CLASSES)
    landuse = _filtrar_clases(_leer_capa_osm(extracted_dir, "landuse_a_free_1"), {"residential"})
    places = _filtrar_clases(_leer_capa_osm(extracted_dir, "places_free_1"), SETTLEMENT_CLASSES)
    residential_areas = pd.concat([landuse, places], ignore_index=True)
    residential_areas = gpd.GeoDataFrame(residential_areas, geometry="geometry", crs=landuse.crs)
    buildings = _leer_capa_osm(extracted_dir, "buildings_a_free_1")
    buildings = buildings.loc[buildings.geometry.notna() & ~buildings.geometry.is_empty].copy()
    activity = calcular_variables_actividad_humana(grid, roads, residential_areas, buildings)
    with xr.open_dataset(spatial_cube_path) as spatial:
        cube = crear_datacubo_actividad_humana(spatial, activity, inclusion_flags)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cube.to_netcdf(output_path)
    return output_path
